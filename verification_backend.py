"""
verification_backend.py
=======================

A FastAPI server exposing real Z3 (z3-solver) and Lean 4 verification to
the self-improving research agent React app.

The thesis (Designing a Self-Improving Research Agent) describes a system
that integrates "Z3 SMT solver" and "Lean 4 theorem prover" for verifying
self-modifications. The React app embeds Z3 via WebAssembly as a fallback,
but Lean 4 has no browser build — and many users want the actual `z3-solver`
Python bindings the thesis names. This server provides both.

Endpoints
---------
GET  /health               -> backend status, Z3 / Lean versions
POST /verify/z3            -> verify a constraint set in our JSON DSL with Z3
POST /verify/lean          -> compile and check a Lean 4 source file
POST /verify/invariant_set -> verify the agent's full invariant DSL in one shot
                              (used by the React app — sends the new config and
                              the named invariant list, server encodes and checks)

Run
---
    pip install -r requirements.txt
    # Install Lean 4 (https://leanprover.github.io/lean4/doc/quickstart.html)
    #   curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
    # Verify lean is on PATH:  lean --version
    uvicorn verification_backend:app --reload --port 8000

The React app's "Verification Backend URL" field defaults to
http://localhost:8000 — point it at wherever you run this.

Security
--------
This server takes Lean source code and runs `lean` on it as a subprocess.
That means anyone who can reach the server can execute arbitrary Lean code
(which has limited side effects but can be heavy). Run on localhost only,
or behind authentication, for any non-toy deployment. CORS is wide open by
default to make local development frictionless — tighten it before exposing
the server to a network.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    import z3  # type: ignore

    Z3_AVAILABLE = True
    Z3_VERSION = z3.get_version_string()
except Exception as _e:  # pragma: no cover
    Z3_AVAILABLE = False
    Z3_VERSION = f"unavailable: {_e}"


# ─────────────────────────────────────────────────────────────────────────────
# App + CORS
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Self-Improving Agent Verification Backend",
    description="Real Z3 and Lean 4 verification for the Gödel-machine self-modification loop",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Lean 4 detection
# ─────────────────────────────────────────────────────────────────────────────

LEAN_BIN = shutil.which("lean")


def _lean_version() -> str:
    if LEAN_BIN is None:
        return "unavailable: `lean` not on PATH"
    try:
        out = subprocess.run(
            [LEAN_BIN, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return (out.stdout or out.stderr).strip()
    except Exception as e:
        return f"unavailable: {e}"


LEAN_VERSION = _lean_version()
LEAN_AVAILABLE = LEAN_BIN is not None and "unavailable" not in LEAN_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "z3": {"available": Z3_AVAILABLE, "version": Z3_VERSION},
        "lean": {
            "available": LEAN_AVAILABLE,
            "version": LEAN_VERSION,
            "binary": LEAN_BIN,
        },
        "python": sys.version.split()[0],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Z3 endpoint — generic JSON constraint DSL
# ─────────────────────────────────────────────────────────────────────────────
#
# The DSL is a small s-expression-style tree:
#
#   {"op": "and", "args": [<expr>, <expr>, ...]}
#   {"op": "or",  "args": [...]}
#   {"op": "not", "args": [<expr>]}
#   {"op": "implies", "args": [<expr>, <expr>]}
#   {"op": "eq" | "ne" | "lt" | "le" | "gt" | "ge", "args": [<num>, <num>]}
#   {"op": "add" | "sub" | "mul" | "div", "args": [<num>, <num>, ...]}
#   {"op": "var", "name": "...", "sort": "Int" | "Real" | "Bool"}
#   {"op": "const", "value": 3.14}                # numeric constant
#   {"op": "boolconst", "value": true}
#
# A request lists `declarations` (variables) plus `assumptions` (asserted
# unconditionally) plus `goals` (each goal is a property we want to prove
# holds for every model satisfying the assumptions).
#
# For each goal we ask Z3:  is (assumptions ∧ ¬goal) satisfiable?
#   unsat  → goal is PROVED
#   sat    → goal is REFUTED, return the counterexample model
#   unknown → return UNKNOWN with reason


class Z3VarDecl(BaseModel):
    name: str
    sort: Literal["Int", "Real", "Bool"] = "Real"


class Z3Goal(BaseModel):
    name: str
    formal: str | None = None
    expr: dict[str, Any]


class Z3Request(BaseModel):
    declarations: list[Z3VarDecl] = Field(default_factory=list)
    assumptions: list[dict[str, Any]] = Field(default_factory=list)
    goals: list[Z3Goal]
    timeout_ms: int = 10_000


class Z3GoalResult(BaseModel):
    name: str
    formal: str | None
    status: Literal["PROVED", "REFUTED", "UNKNOWN", "ERROR"]
    counterexample: dict[str, str] | None = None
    reason: str | None = None
    elapsed_ms: float


class Z3Response(BaseModel):
    backend: str = "z3-server"
    z3_version: str
    overall: Literal["APPROVED", "REJECTED", "UNKNOWN", "ERROR"]
    results: list[Z3GoalResult]
    total_elapsed_ms: float


def _z3_build(expr: dict[str, Any], env: dict[str, Any]) -> Any:
    """Recursively build a Z3 expression from a JSON node."""
    if not isinstance(expr, dict) or "op" not in expr:
        raise ValueError(f"malformed expression node: {expr!r}")

    op = expr["op"]

    if op == "var":
        name = expr["name"]
        if name not in env:
            raise ValueError(f"undeclared variable: {name}")
        return env[name]

    if op == "const":
        v = expr["value"]
        if isinstance(v, bool):
            return z3.BoolVal(v)
        if isinstance(v, int):
            return z3.IntVal(v)
        if isinstance(v, float):
            # Use Real for floats, expressed as a string to preserve precision
            return z3.RealVal(repr(v))
        raise ValueError(f"unsupported constant: {v!r}")

    if op == "boolconst":
        return z3.BoolVal(bool(expr["value"]))

    args = [_z3_build(a, env) for a in expr.get("args", [])]

    if op == "and":
        return z3.And(*args) if args else z3.BoolVal(True)
    if op == "or":
        return z3.Or(*args) if args else z3.BoolVal(False)
    if op == "not":
        if len(args) != 1:
            raise ValueError("`not` takes exactly one argument")
        return z3.Not(args[0])
    if op == "implies":
        if len(args) != 2:
            raise ValueError("`implies` takes exactly two arguments")
        return z3.Implies(args[0], args[1])

    if op in ("eq", "ne", "lt", "le", "gt", "ge"):
        if len(args) != 2:
            raise ValueError(f"`{op}` takes exactly two arguments")
        a, b = args
        return {
            "eq": a == b,
            "ne": a != b,
            "lt": a < b,
            "le": a <= b,
            "gt": a > b,
            "ge": a >= b,
        }[op]

    if op in ("add", "sub", "mul", "div"):
        if not args:
            raise ValueError(f"`{op}` needs at least one argument")
        result = args[0]
        for a in args[1:]:
            if op == "add":
                result = result + a
            elif op == "sub":
                result = result - a
            elif op == "mul":
                result = result * a
            elif op == "div":
                result = result / a
        return result

    raise ValueError(f"unknown op: {op}")


def _model_to_dict(model: "z3.ModelRef") -> dict[str, str]:
    out: dict[str, str] = {}
    for d in model.decls():
        try:
            out[d.name()] = str(model[d])
        except Exception:
            out[d.name()] = "?"
    return out


@app.post("/verify/z3", response_model=Z3Response)
def verify_z3(req: Z3Request) -> Z3Response:
    if not Z3_AVAILABLE:
        raise HTTPException(503, f"Z3 unavailable on this server: {Z3_VERSION}")

    t0 = time.perf_counter()

    # Build environment of declared variables
    env: dict[str, Any] = {}
    for decl in req.declarations:
        if decl.sort == "Int":
            env[decl.name] = z3.Int(decl.name)
        elif decl.sort == "Bool":
            env[decl.name] = z3.Bool(decl.name)
        else:
            env[decl.name] = z3.Real(decl.name)

    # Compile assumptions
    try:
        compiled_assumptions = [_z3_build(a, env) for a in req.assumptions]
    except Exception as e:
        raise HTTPException(400, f"failed to compile assumptions: {e}")

    results: list[Z3GoalResult] = []
    overall: Literal["APPROVED", "REJECTED", "UNKNOWN", "ERROR"] = "APPROVED"

    for goal in req.goals:
        g0 = time.perf_counter()
        try:
            goal_expr = _z3_build(goal.expr, env)
        except Exception as e:
            results.append(
                Z3GoalResult(
                    name=goal.name,
                    formal=goal.formal,
                    status="ERROR",
                    reason=f"compile error: {e}",
                    elapsed_ms=(time.perf_counter() - g0) * 1000,
                )
            )
            overall = "ERROR"
            continue

        solver = z3.Solver()
        solver.set("timeout", req.timeout_ms)
        for a in compiled_assumptions:
            solver.add(a)
        solver.add(z3.Not(goal_expr))

        try:
            check = solver.check()
        except Exception as e:
            results.append(
                Z3GoalResult(
                    name=goal.name,
                    formal=goal.formal,
                    status="ERROR",
                    reason=f"solver error: {e}",
                    elapsed_ms=(time.perf_counter() - g0) * 1000,
                )
            )
            overall = "ERROR"
            continue

        elapsed = (time.perf_counter() - g0) * 1000
        if check == z3.unsat:
            results.append(
                Z3GoalResult(
                    name=goal.name,
                    formal=goal.formal,
                    status="PROVED",
                    elapsed_ms=elapsed,
                )
            )
        elif check == z3.sat:
            try:
                model = _model_to_dict(solver.model())
            except Exception:
                model = None
            results.append(
                Z3GoalResult(
                    name=goal.name,
                    formal=goal.formal,
                    status="REFUTED",
                    counterexample=model,
                    elapsed_ms=elapsed,
                )
            )
            if overall == "APPROVED":
                overall = "REJECTED"
        else:
            results.append(
                Z3GoalResult(
                    name=goal.name,
                    formal=goal.formal,
                    status="UNKNOWN",
                    reason=str(solver.reason_unknown()),
                    elapsed_ms=elapsed,
                )
            )
            if overall == "APPROVED":
                overall = "UNKNOWN"

    return Z3Response(
        z3_version=Z3_VERSION,
        overall=overall,
        results=results,
        total_elapsed_ms=(time.perf_counter() - t0) * 1000,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Lean 4 endpoint
# ─────────────────────────────────────────────────────────────────────────────


class LeanRequest(BaseModel):
    source: str
    timeout_seconds: float = 30.0


class LeanResponse(BaseModel):
    backend: str = "lean-server"
    lean_version: str
    status: Literal["PROVED", "REFUTED", "ERROR", "TIMEOUT"]
    stdout: str
    stderr: str
    elapsed_ms: float


@app.post("/verify/lean", response_model=LeanResponse)
async def verify_lean(req: LeanRequest) -> LeanResponse:
    if not LEAN_AVAILABLE:
        raise HTTPException(
            503,
            f"Lean 4 unavailable on this server: {LEAN_VERSION}. "
            "Install with `curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh`",
        )

    t0 = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="lean_verify_") as td:
        path = os.path.join(td, "Proof.lean")
        with open(path, "w", encoding="utf-8") as f:
            f.write(req.source)

        try:
            proc = await asyncio.create_subprocess_exec(
                LEAN_BIN,  # type: ignore[arg-type]
                path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=td,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=req.timeout_seconds
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return LeanResponse(
                    lean_version=LEAN_VERSION,
                    status="TIMEOUT",
                    stdout="",
                    stderr=f"lean timed out after {req.timeout_seconds}s",
                    elapsed_ms=(time.perf_counter() - t0) * 1000,
                )
        except FileNotFoundError as e:
            raise HTTPException(503, f"failed to launch lean: {e}")

        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        rc = proc.returncode

        # Lean 4 prints errors to stdout (not stderr) and exits 1 on failure.
        # Empty stdout + rc==0 means the file type-checked cleanly → PROVED.
        if rc == 0 and "error:" not in stdout.lower() and "error:" not in stderr.lower():
            status: Literal["PROVED", "REFUTED", "ERROR", "TIMEOUT"] = "PROVED"
        elif "sorry" in stdout.lower() or "sorry" in stderr.lower():
            # Sorry is an admitted goal — not a real proof
            status = "ERROR"
        elif rc != 0 or "error:" in stdout.lower():
            status = "REFUTED"
        else:
            status = "ERROR"

        return LeanResponse(
            lean_version=LEAN_VERSION,
            status=status,
            stdout=stdout,
            stderr=stderr,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience endpoint: verify the full agent invariant set with Z3
# ─────────────────────────────────────────────────────────────────────────────
#
# Mirrors the `CONFIG_INVARIANTS` array in self-improving-agent.jsx so the
# React app can submit the new config and get a verdict in one round-trip.
# Adding a new invariant means updating both this function and the JSX list.


class InvariantSetRequest(BaseModel):
    new_config: dict[str, Any]
    old_config: dict[str, Any] | None = None


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


@app.post("/verify/invariant_set", response_model=Z3Response)
def verify_invariant_set(req: InvariantSetRequest) -> Z3Response:
    if not Z3_AVAILABLE:
        raise HTTPException(503, f"Z3 unavailable: {Z3_VERSION}")

    flat = _flatten_numerics(req.new_config)

    # Build a Z3Request that mirrors CONFIG_INVARIANTS in the JSX
    declarations = [Z3VarDecl(name=k, sort="Real") for k in flat.keys()]
    assumptions: list[dict[str, Any]] = [
        {
            "op": "eq",
            "args": [
                {"op": "var", "name": k},
                {"op": "const", "value": v},
            ],
        }
        for k, v in flat.items()
    ]

    def var(n: str) -> dict[str, Any]:
        return {"op": "var", "name": n}

    def num(v: float) -> dict[str, Any]:
        return {"op": "const", "value": v}

    goals: list[Z3Goal] = []

    if "generation_minWordsPerSection" in flat:
        goals.append(
            Z3Goal(
                name="minWordsPerSection_positive",
                formal="generation.minWordsPerSection > 0",
                expr={"op": "gt", "args": [var("generation_minWordsPerSection"), num(0)]},
            )
        )

    if "generation_targetTotalWords" in flat:
        goals.append(
            Z3Goal(
                name="targetTotalWords_reasonable",
                formal="500 ≤ generation.targetTotalWords ≤ 20000",
                expr={
                    "op": "and",
                    "args": [
                        {"op": "ge", "args": [var("generation_targetTotalWords"), num(500)]},
                        {"op": "le", "args": [var("generation_targetTotalWords"), num(20000)]},
                    ],
                },
            )
        )

    weight_keys = [
        "verification_weightWordCount",
        "verification_weightSections",
        "verification_weightCitations",
        "verification_weightReferences",
        "verification_weightMathRigor",
    ]
    if all(k in flat for k in weight_keys):
        sum_expr = {"op": "add", "args": [var(k) for k in weight_keys]}
        goals.append(
            Z3Goal(
                name="weights_sum_to_100",
                formal="Σ verification.weight_i ∈ [99, 100]",
                expr={
                    "op": "and",
                    "args": [
                        {"op": "ge", "args": [sum_expr, num(99)]},
                        {"op": "le", "args": [sum_expr, num(100)]},
                    ],
                },
            )
        )
        goals.append(
            Z3Goal(
                name="all_weights_nonneg",
                formal="∀ w ∈ weights. w ≥ 0",
                expr={
                    "op": "and",
                    "args": [{"op": "ge", "args": [var(k), num(0)]} for k in weight_keys],
                },
            )
        )

    penalty_keys = [
        "verification_penaltyThinSection",
        "verification_penaltyMissingSection",
        "verification_penaltyOrphanCitation",
        "verification_penaltyWeakPhrase",
    ]
    if all(k in flat for k in penalty_keys):
        goals.append(
            Z3Goal(
                name="penalties_bounded",
                formal="∀ p ∈ penalties. 0 ≤ p ≤ 20",
                expr={
                    "op": "and",
                    "args": [
                        clause
                        for k in penalty_keys
                        for clause in (
                            {"op": "ge", "args": [var(k), num(0)]},
                            {"op": "le", "args": [var(k), num(20)]},
                        )
                    ],
                },
            )
        )

    if "verification_reconciliationBand" in flat:
        goals.append(
            Z3Goal(
                name="reconciliation_band_safe",
                formal="3 ≤ reconciliationBand ≤ 15",
                expr={
                    "op": "and",
                    "args": [
                        {"op": "ge", "args": [var("verification_reconciliationBand"), num(3)]},
                        {"op": "le", "args": [var("verification_reconciliationBand"), num(15)]},
                    ],
                },
            )
        )

    if "improvement_targetIncrement" in flat:
        goals.append(
            Z3Goal(
                name="improvement_increment_safe",
                formal="1 ≤ targetIncrement ≤ 25",
                expr={
                    "op": "and",
                    "args": [
                        {"op": "ge", "args": [var("improvement_targetIncrement"), num(1)]},
                        {"op": "le", "args": [var("improvement_targetIncrement"), num(25)]},
                    ],
                },
            )
        )

    if "improvement_aggressiveness" in flat:
        goals.append(
            Z3Goal(
                name="aggressiveness_in_unit_interval",
                formal="0 ≤ aggressiveness ≤ 1",
                expr={
                    "op": "and",
                    "args": [
                        {"op": "ge", "args": [var("improvement_aggressiveness"), num(0)]},
                        {"op": "le", "args": [var("improvement_aggressiveness"), num(1)]},
                    ],
                },
            )
        )

    if "generation_citationsPerParagraph" in flat:
        goals.append(
            Z3Goal(
                name="citationsPerParagraph_realistic",
                formal="1 ≤ citationsPerParagraph ≤ 10",
                expr={
                    "op": "and",
                    "args": [
                        {"op": "ge", "args": [var("generation_citationsPerParagraph"), num(1)]},
                        {"op": "le", "args": [var("generation_citationsPerParagraph"), num(10)]},
                    ],
                },
            )
        )

    return verify_z3(
        Z3Request(declarations=declarations, assumptions=assumptions, goals=goals)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Root
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "self-improving-agent verification backend",
        "endpoints": ["/health", "/verify/z3", "/verify/lean", "/verify/invariant_set"],
        "z3_available": Z3_AVAILABLE,
        "lean_available": LEAN_AVAILABLE,
    }
