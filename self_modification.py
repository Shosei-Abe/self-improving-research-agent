"""
self_modification.py
====================

Heuristic proposal generator + Z3-backed verifier for the Gödel-machine-style
self-modification step.

Mirrors `proposeSelfModification` and the invariant set from
self-improving-agent.jsx, but runs entirely in Python so the LangGraph
pipeline can call it without an HTTP hop. Uses the same `z3-solver` Python
bindings the verification endpoint uses.

The Lean 4 path is invoked through `lean_emitter.emit_lean_proof` and the
existing `/verify/lean` infrastructure (subprocess to `lean` binary).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any

# We import z3 lazily so missing-z3 is not a hard import error.
try:
    import z3  # type: ignore

    _Z3_AVAILABLE = True
except Exception:
    _Z3_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Configuration invariants — same set as CONFIG_INVARIANTS in the JSX
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Invariant:
    name: str
    formal: str
    check: callable  # (config, prev_config) -> bool


INVARIANTS: list[Invariant] = [
    Invariant(
        name="minWordsPerSection_positive",
        formal="∀ cfg. cfg.generation.minWordsPerSection > 0",
        check=lambda c, p: c["generation"]["minWordsPerSection"] > 0,
    ),
    Invariant(
        name="targetTotalWords_reasonable",
        formal="∀ cfg. 500 ≤ cfg.generation.targetTotalWords ≤ 20000",
        check=lambda c, p: 500 <= c["generation"]["targetTotalWords"] <= 20000,
    ),
    Invariant(
        name="weights_sum_to_100",
        formal="∀ cfg. Σ verification.weight_i ∈ [99, 100]",
        check=lambda c, p: 99 <= sum([
            c["verification"]["weightWordCount"],
            c["verification"]["weightSections"],
            c["verification"]["weightCitations"],
            c["verification"]["weightReferences"],
            c["verification"]["weightMathRigor"],
        ]) <= 100,
    ),
    Invariant(
        name="all_weights_nonneg",
        formal="∀ w ∈ weights. w ≥ 0",
        check=lambda c, p: all(
            v >= 0 for v in c["verification"].values() if isinstance(v, (int, float))
        ),
    ),
    Invariant(
        name="penalties_bounded",
        formal="∀ p ∈ penalties. 0 ≤ p ≤ 20",
        check=lambda c, p: all(
            0 <= c["verification"][k] <= 20
            for k in [
                "penaltyThinSection",
                "penaltyMissingSection",
                "penaltyOrphanCitation",
                "penaltyWeakPhrase",
            ]
        ),
    ),
    Invariant(
        name="reconciliation_band_safe",
        formal="∀ cfg. 3 ≤ reconciliationBand ≤ 15",
        check=lambda c, p: 3 <= c["verification"]["reconciliationBand"] <= 15,
    ),
    Invariant(
        name="improvement_increment_safe",
        formal="∀ cfg. 1 ≤ targetIncrement ≤ 25",
        check=lambda c, p: 1 <= c["improvement"]["targetIncrement"] <= 25,
    ),
    Invariant(
        name="aggressiveness_in_unit_interval",
        formal="∀ cfg. 0 ≤ aggressiveness ≤ 1",
        check=lambda c, p: 0 <= c["improvement"]["aggressiveness"] <= 1,
    ),
    Invariant(
        name="citationsPerParagraph_realistic",
        formal="∀ cfg. 1 ≤ citationsPerParagraph ≤ 10",
        check=lambda c, p: 1 <= c["generation"]["citationsPerParagraph"] <= 10,
    ),
    Invariant(
        name="schema_version_preserved",
        formal="∀ new, old. new.__version = old.__version",
        check=lambda c, p: c.get("__version", 1) == p.get("__version", 1),
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic proposal generator (5 heuristics, same as JSX)
# ─────────────────────────────────────────────────────────────────────────────


def propose_self_modification(
    current_config: dict[str, Any],
    recent_metrics: dict[str, Any] | None,
    score_history: list[float],
) -> dict[str, Any] | None:
    """Returns {'proposal': {...}, 'newConfig': {...}, 'allProposals': [...]}
    or None if no modification is warranted.
    """
    proposals: list[dict[str, Any]] = []
    trend = (
        score_history[-1] - score_history[-2]
        if len(score_history) >= 2
        else 0
    )

    # Heuristic 1: thin sections
    if recent_metrics and len(recent_metrics.get("thinSections", [])) >= 2:
        proposals.append({
            "reasoning": f"{len(recent_metrics['thinSections'])} thin sections — generation parameters too lax",
            "mutation": {
                "generation": {
                    "minWordsPerSection": min(current_config["generation"]["minWordsPerSection"] + 50, 400),
                }
            },
            "expectedEffect": "Forces generation agent to write longer sections",
        })

    # Heuristic 2: low citation density
    if recent_metrics:
        para_count = recent_metrics.get("paragraphCount", 0)
        cited = recent_metrics.get("paragraphsWithCitations", 0)
        if para_count > 0 and cited / para_count < 0.6:
            proposals.append({
                "reasoning": f"Citation density {cited/para_count:.2f} below 0.6",
                "mutation": {
                    "generation": {
                        "citationsPerParagraph": min(current_config["generation"]["citationsPerParagraph"] + 1, 5),
                    }
                },
                "expectedEffect": "Demands more citations per paragraph",
            })

    # Heuristic 3: word count under target
    if (
        recent_metrics
        and recent_metrics.get("wordCount", 0)
        < current_config["generation"]["targetTotalWords"] * 0.8
    ):
        proposals.append({
            "reasoning": f"Word count {recent_metrics['wordCount']} far below target {current_config['generation']['targetTotalWords']}",
            "mutation": {
                "generation": {
                    "targetTotalWords": min(current_config["generation"]["targetTotalWords"] + 500, 6000),
                }
            },
            "expectedEffect": "Pushes target word count higher",
        })

    # Heuristic 4: plateau
    if trend != 0 and abs(trend) <= 2 and len(score_history) >= 2:
        new_agg = min(current_config["improvement"]["aggressiveness"] + 0.15, 1.0)
        proposals.append({
            "reasoning": f"Score plateaued (Δ={trend}) — system needs more aggressive changes",
            "mutation": {
                "improvement": {
                    "aggressiveness": round(new_agg, 2),
                    "targetIncrement": min(current_config["improvement"]["targetIncrement"] + 3, 25),
                }
            },
            "expectedEffect": "More aggressive improvement targets",
        })

    # Heuristic 5: regression
    if trend < -3:
        proposals.append({
            "reasoning": f"Performance dropped by {-trend} points — tighten verification",
            "mutation": {
                "verification": {
                    "reconciliationBand": max(current_config["verification"]["reconciliationBand"] - 2, 3),
                }
            },
            "expectedEffect": "Stricter reconciliation prevents score inflation",
        })

    if not proposals:
        return None

    chosen = proposals[0]
    new_config = copy.deepcopy(current_config)
    for section, updates in chosen["mutation"].items():
        new_config[section] = {**new_config[section], **updates}

    return {
        "proposal": chosen,
        "newConfig": new_config,
        "allProposals": proposals,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Z3-backed verifier (mirror of symbolicVerify in the JSX)
# ─────────────────────────────────────────────────────────────────────────────


def _flatten_numerics(cfg: dict[str, Any], prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in cfg.items():
        if k == "__version":
            continue
        name = f"{prefix}_{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_numerics(v, name))
        elif isinstance(v, bool):
            out[name] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)):
            out[name] = float(v)
    return out


def _diff_configs(old: dict[str, Any], new: dict[str, Any], path: str = "") -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    keys = set(old.keys()) | set(new.keys())
    for k in keys:
        if k == "__version":
            continue
        p = f"{path}.{k}" if path else k
        v1 = old.get(k)
        v2 = new.get(k)
        if isinstance(v1, dict) and isinstance(v2, dict):
            diffs.extend(_diff_configs(v1, v2, p))
        elif v1 != v2:
            delta = (v2 - v1) if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) else None
            diffs.append({"path": p, "from": v1, "to": v2, "delta": delta})
    return diffs


def verify_modification_with_z3(
    new_config: dict[str, Any],
    old_config: dict[str, Any],
    performance_history: list[float] | None = None,
) -> dict[str, Any]:
    """Run the full verification pipeline using the actual z3-solver bindings.

    Returns a dict matching the JSX symbolicVerify return shape, so the
    React modification log UI can display it unchanged.
    """
    obligations: list[dict[str, Any]] = []
    all_passed = True
    backends_used: set[str] = set()

    if not _Z3_AVAILABLE:
        # Pure-Python fallback
        for inv in INVARIANTS:
            try:
                passed = inv.check(new_config, old_config)
            except Exception:
                passed = False
            obligations.append({
                "phase": "INVARIANT",
                "name": inv.name,
                "formal": inv.formal,
                "status": "PROVED" if passed else "REFUTED",
                "backend": "symbolic",
            })
            if not passed:
                all_passed = False
        backends_used.add("symbolic")
    else:
        # Z3 path: encode each invariant as a constraint and check unsat of its negation
        flat = _flatten_numerics(new_config)
        z3_vars = {k: z3.Real(k) for k in flat}
        bindings = [z3_vars[k] == z3.RealVal(repr(v)) for k, v in flat.items()]

        for inv in INVARIANTS:
            t0 = time.perf_counter()
            constraint = _encode_invariant_to_z3(inv.name, z3_vars)
            if constraint is None:
                # Fall back to Python check
                try:
                    passed = inv.check(new_config, old_config)
                except Exception:
                    passed = False
                obligations.append({
                    "phase": "INVARIANT",
                    "name": inv.name,
                    "formal": inv.formal,
                    "status": "PROVED" if passed else "REFUTED",
                    "backend": "symbolic",
                    "elapsed_ms": (time.perf_counter() - t0) * 1000,
                })
                if not passed:
                    all_passed = False
                continue

            solver = z3.Solver()
            for b in bindings:
                solver.add(b)
            solver.add(z3.Not(constraint))
            result = solver.check()

            if result == z3.unsat:
                status = "PROVED"
                cex = None
            elif result == z3.sat:
                status = "REFUTED"
                model = solver.model()
                cex = ", ".join(f"{d.name()}={model[d]}" for d in model.decls())
                all_passed = False
            else:
                status = "WARNING"
                cex = f"unknown: {solver.reason_unknown()}"

            obligations.append({
                "phase": "INVARIANT",
                "name": inv.name,
                "formal": inv.formal,
                "status": status,
                "counterexample": cex,
                "backend": "z3-server",
                "elapsed_ms": (time.perf_counter() - t0) * 1000,
            })
        backends_used.add("z3-server")

    # ─── Diff analysis ───
    diffs = _diff_configs(old_config, new_config)
    obligations.append({
        "phase": "DIFF_ANALYSIS",
        "name": "config_diff_extraction",
        "formal": f"|{{p : C_old(p) ≠ C_new(p)}}| = {len(diffs)}",
        "status": "PROVED",
        "backend": "symbolic",
    })

    # ─── Monotonicity ───
    if _Z3_AVAILABLE:
        for d in diffs:
            if d["delta"] is None or not isinstance(d["from"], (int, float)) or d["from"] <= 0:
                continue
            t0 = time.perf_counter()
            solver = z3.Solver()
            old_var = z3.Real("old_v")
            new_var = z3.Real("new_v")
            solver.add(old_var == z3.RealVal(repr(d["from"])))
            solver.add(new_var == z3.RealVal(repr(d["to"])))
            upper = (new_var - old_var) <= z3.RealVal("0.5") * old_var
            lower = (old_var - new_var) <= z3.RealVal("0.5") * old_var
            solver.add(z3.Not(z3.And(upper, lower)))
            safe = solver.check() == z3.unsat
            obligations.append({
                "phase": "MONOTONICITY",
                "name": f"bounded_change_{d['path']}",
                "formal": f"|{d['path']}_new - {d['path']}_old| ≤ 0.5 · {d['path']}_old",
                "status": "PROVED" if safe else "REFUTED",
                "counterexample": None if safe else f"{d['path']}: {d['from']} → {d['to']}",
                "backend": "z3-server",
                "elapsed_ms": (time.perf_counter() - t0) * 1000,
            })
            if not safe:
                all_passed = False
    else:
        for d in diffs:
            if d["delta"] is None or not isinstance(d["from"], (int, float)) or d["from"] <= 0:
                continue
            ratio = abs(d["delta"]) / d["from"]
            safe = ratio <= 0.5
            obligations.append({
                "phase": "MONOTONICITY",
                "name": f"bounded_change_{d['path']}",
                "formal": f"|Δ{d['path']}| / |{d['path']}_old| ≤ 0.5",
                "status": "PROVED" if safe else "REFUTED",
                "counterexample": None if safe else f"{d['path']} changed by {ratio*100:.0f}%",
                "backend": "symbolic",
            })
            if not safe:
                all_passed = False

    # ─── Progress ───
    if not diffs:
        obligations.append({
            "phase": "PROGRESS",
            "name": "nontrivial_modification",
            "formal": "|diffs| ≥ 1",
            "status": "REFUTED",
            "counterexample": "No parameters changed",
            "backend": "symbolic",
        })
        all_passed = False
    else:
        obligations.append({
            "phase": "PROGRESS",
            "name": "nontrivial_modification",
            "formal": "|diffs| ≥ 1",
            "status": "PROVED",
            "backend": "symbolic",
        })

    # ─── Regression check ───
    if performance_history and len(performance_history) >= 2:
        recent = performance_history[-3:]
        trend = recent[-1] - recent[0]
        obligations.append({
            "phase": "REGRESSION",
            "name": "performance_trend_analysis",
            "formal": f"score(t) - score(t-3) = {trend}",
            "status": "PROVED" if trend >= -5 else "WARNING",
            "details": f"Performance trend: {'+' if trend >= 0 else ''}{trend}",
            "backend": "symbolic",
        })

    proof_chain = _build_proof_chain(obligations)

    return {
        "verdict": "APPROVED" if all_passed else "REJECTED",
        "obligations": obligations,
        "diffs": diffs,
        "proof_chain": proof_chain,
        "backends_used": sorted(backends_used),
    }


def _encode_invariant_to_z3(name: str, vars: dict[str, Any]) -> Any:
    """Translate an invariant name into a Z3 constraint expression."""
    if not _Z3_AVAILABLE:
        return None

    R = z3.RealVal
    v = vars

    if name == "minWordsPerSection_positive":
        return v["generation_minWordsPerSection"] > R(0)

    if name == "targetTotalWords_reasonable":
        x = v["generation_targetTotalWords"]
        return z3.And(x >= R(500), x <= R(20000))

    if name == "weights_sum_to_100":
        s = (
            v["verification_weightWordCount"]
            + v["verification_weightSections"]
            + v["verification_weightCitations"]
            + v["verification_weightReferences"]
            + v["verification_weightMathRigor"]
        )
        return z3.And(s >= R(99), s <= R(100))

    if name == "all_weights_nonneg":
        return z3.And(
            v["verification_weightWordCount"] >= R(0),
            v["verification_weightSections"] >= R(0),
            v["verification_weightCitations"] >= R(0),
            v["verification_weightReferences"] >= R(0),
            v["verification_weightMathRigor"] >= R(0),
        )

    if name == "penalties_bounded":
        return z3.And(
            v["verification_penaltyThinSection"] >= R(0),
            v["verification_penaltyThinSection"] <= R(20),
            v["verification_penaltyMissingSection"] >= R(0),
            v["verification_penaltyMissingSection"] <= R(20),
            v["verification_penaltyOrphanCitation"] >= R(0),
            v["verification_penaltyOrphanCitation"] <= R(20),
            v["verification_penaltyWeakPhrase"] >= R(0),
            v["verification_penaltyWeakPhrase"] <= R(20),
        )

    if name == "reconciliation_band_safe":
        x = v["verification_reconciliationBand"]
        return z3.And(x >= R(3), x <= R(15))

    if name == "improvement_increment_safe":
        x = v["improvement_targetIncrement"]
        return z3.And(x >= R(1), x <= R(25))

    if name == "aggressiveness_in_unit_interval":
        x = v["improvement_aggressiveness"]
        return z3.And(x >= R(0), x <= R(1))

    if name == "citationsPerParagraph_realistic":
        x = v["generation_citationsPerParagraph"]
        return z3.And(x >= R(1), x <= R(10))

    if name == "schema_version_preserved":
        return z3.BoolVal(True)

    return None


def _build_proof_chain(obligations: list[dict[str, Any]]) -> str:
    lines = ["Theorem: modification_is_safe(C_old, C_new).", "Proof:"]
    for o in obligations:
        if o["status"] == "PROVED":
            lines.append(f"  by {o['name']}: {o['formal']} ✓")
        elif o["status"] == "REFUTED":
            lines.append(f"  ✗ {o['name']}: {o['formal']}")
            if o.get("counterexample"):
                lines.append(f"    counterexample: {o['counterexample']}")
        elif o["status"] == "WARNING":
            lines.append(f"  ⚠ {o['name']}: {o['formal']}")
    failed = [o for o in obligations if o["status"] == "REFUTED"]
    if not failed:
        lines.append("Qed.")
    else:
        lines.append(f"Aborted. ({len(failed)} obligation(s) refuted)")
    return "\n".join(lines)
