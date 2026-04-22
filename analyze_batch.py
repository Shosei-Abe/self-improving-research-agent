#!/usr/bin/env python3.11
"""
analyze_batch.py
================

Reads the JSONL output of run_batch.py (b1_results.jsonl, b2_results.jsonl,
co_results.jsonl) plus optionally the corresponding rows in Postgres (for
per-run deterministic metrics), and produces:

  1. A per-condition summary table (Markdown + raw dict) with:
     - observed n, AQS mean/std/range, TSR, execution-time mean,
       word-count mean, unique-citation mean, mods_attempted/applied.
  2. A Welch's t-test on AQS for B1 vs B2 and B2 vs CO.
  3. A machine-readable dump of all the [[TODO: ...]] placeholders used
     in the thesis §4 tables, so the user can paste them one-shot.
  4. CO learning-curve data (per-run AQS for the CO condition, in order).
  5. The modification-field distribution across B2 + CO (which configuration
     fields were most frequently touched by self-modification).

Usage:
    python3.11 analyze_batch.py \\
        --b1 b1_results.jsonl --b2 b2_results.jsonl --co co_results.jsonl \\
        --out-md batch_summary.md --out-tex batch_placeholders.tex \\
        --out-json batch_values.json

If --pg-url is also given, the script will enrich each run with its
modification_log from checkpoints so the per-field distribution is real
rather than approximated from run_batch.py's mods_attempted/applied counts.

The script degrades gracefully: missing files, partial JSONL (e.g., if a
batch is still running), and DB-unavailable states all produce a report of
"observed" data rather than "intended" data.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file. Returns empty list if file doesn't exist."""
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] {path.name}: skipping malformed line: {e}", file=sys.stderr)
    return out


def mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def stddev(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return var ** 0.5


def welch_t_test(a: list[float], b: list[float]) -> dict:
    """Two-sample Welch's t-test (unequal variances). Returns t, df, p (two-tailed).

    We compute the two-tailed p-value via a Student's-t CDF approximation
    that is good enough for thesis-level reporting. If you need exact
    p, pipe the t and df into scipy offline.
    """
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2:
        return {"t": None, "df": None, "p": None, "note": "insufficient n"}

    na, nb = len(a), len(b)
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)

    if va == 0 and vb == 0:
        return {"t": float("inf") if ma != mb else 0.0, "df": na + nb - 2,
                "p": 0.0 if ma != mb else 1.0, "note": "zero variance"}

    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return {"t": None, "df": None, "p": None, "note": "zero standard error"}

    t = (ma - mb) / se
    # Welch–Satterthwaite df
    num = (va / na + vb / nb) ** 2
    den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = num / den if den > 0 else na + nb - 2

    # Approximate two-tailed p via the t-distribution CDF.
    # We use the Abramowitz & Stegun series for the t-CDF; good to ~4 decimals.
    p = _two_tailed_t_p(abs(t), df)
    return {"t": t, "df": df, "p": p, "note": ""}


def _two_tailed_t_p(t_abs: float, df: float) -> float:
    """Approximate two-tailed p for Student's t using the regularised
    incomplete beta function. Implemented from scratch (no scipy) with a
    continued-fraction evaluation; accurate enough for reporting."""
    # p_two_tailed = I_x(df/2, 1/2) where x = df / (df + t^2)
    x = df / (df + t_abs * t_abs)
    a, b = df / 2.0, 0.5
    return _betai(a, b, x)


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    # Use the symmetric form for numerical stability
    if x < (a + 1) / (a + b + 2):
        bt = math.exp(-lbeta + a * math.log(x) + b * math.log(1 - x))
        return bt * _betacf(a, b, x) / a
    else:
        bt = math.exp(-lbeta + a * math.log(x) + b * math.log(1 - x))
        return 1.0 - bt * _betacf(b, a, 1 - x) / b


def _betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 3e-7) -> float:
    """Continued-fraction evaluation for the incomplete beta. NR §6.4 style."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    return h  # fell off; return what we have


def summarize_condition(rows: list[dict], label: str) -> dict:
    """Aggregate per-condition stats from run_batch.py JSONL rows."""
    completed = [r for r in rows if r.get("status") == "complete"]
    scores = [r.get("final_score") for r in completed]
    scores = [s for s in scores if s is not None]
    ets = [r.get("elapsed_s") for r in completed]
    ets = [e for e in ets if e is not None]
    wcs = [r.get("word_count") for r in completed]
    wcs = [w for w in wcs if w is not None]
    mods_att = [r.get("mods_attempted") or 0 for r in completed]
    mods_app = [r.get("mods_applied") or 0 for r in completed]

    tsr_pass = sum(1 for s in scores if s >= 70.0)

    return {
        "condition": label,
        "n_intended": len(rows),
        "n_completed": len(completed),
        "n_for_AQS": len(scores),
        "AQS_mean": mean(scores),
        "AQS_std":  stddev(scores),
        "AQS_min":  min(scores) if scores else None,
        "AQS_max":  max(scores) if scores else None,
        "TSR":      (tsr_pass / len(scores)) if scores else None,
        "ET_mean":  mean(ets),
        "wc_mean":  mean(wcs),
        "mods_att_total": sum(mods_att),
        "mods_app_total": sum(mods_app),
        "_scores":  scores,  # kept for downstream t-test
    }


def modification_field_distribution(rows: list[dict], pg_url: str | None) -> Counter:
    """
    Count which configuration fields were targeted by self-modification
    proposals, across all supplied rows. Requires Postgres for the full
    modification_log; without DB, returns empty Counter.
    """
    counter: Counter = Counter()
    if not pg_url:
        return counter

    try:
        import psycopg
    except ImportError:
        print("[warn] psycopg not available; skipping field distribution.", file=sys.stderr)
        return counter

    run_ids = [r.get("run_id") for r in rows if r.get("run_id")]
    if not run_ids:
        return counter

    # Pull latest checkpoint per run and walk its modification_log.
    sql = """
        WITH latest AS (
            SELECT DISTINCT ON (run_id) run_id, state_json
            FROM checkpoints
            WHERE run_id = ANY(%s)
            ORDER BY run_id, iteration DESC, id DESC
        )
        SELECT run_id, state_json FROM latest
    """
    try:
        with psycopg.connect(pg_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (run_ids,))
                for _rid, state in cur.fetchall():
                    if not state:
                        continue
                    mod_log = (state or {}).get("modification_log", []) or []
                    for entry in mod_log:
                        prop = entry.get("proposal") or {}
                        path = prop.get("target_path")
                        if path:
                            counter[path] += 1
    except Exception as e:
        print(f"[warn] DB enrichment failed: {e}", file=sys.stderr)

    return counter


def render_md_table(summaries: list[dict]) -> str:
    """Side-by-side Markdown table of the three conditions."""
    def fmt(x, nd=2):
        if x is None:
            return "—"
        if isinstance(x, float):
            return f"{x:.{nd}f}"
        return str(x)

    rows = [
        ("Observed n",              [s["n_completed"] for s in summaries]),
        ("AQS mean",                [fmt(s["AQS_mean"]) for s in summaries]),
        ("AQS std",                 [fmt(s["AQS_std"]) for s in summaries]),
        ("AQS min–max",             [f'{fmt(s["AQS_min"])}–{fmt(s["AQS_max"])}' for s in summaries]),
        ("TSR (≥70)",               [fmt(s["TSR"], 3) for s in summaries]),
        ("ET mean (s)",             [fmt(s["ET_mean"], 1) for s in summaries]),
        ("Word count mean",         [fmt(s["wc_mean"], 0) for s in summaries]),
        ("Mods attempted (total)",  [s["mods_att_total"] for s in summaries]),
        ("Mods applied (total)",    [s["mods_app_total"] for s in summaries]),
    ]
    header = "| Metric | " + " | ".join(s["condition"] for s in summaries) + " |"
    sep = "|---|" + "|".join("---" for _ in summaries) + "|"
    lines = [header, sep]
    for label, cells in rows:
        lines.append("| " + label + " | " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines)


def render_latex_placeholders(summaries: dict, b1_vs_b2: dict, field_counter: Counter) -> str:
    """
    Emit text that directly resolves the [[TODO: ...]] placeholders from the
    thesis §4 tables. The user can open Overleaf and Find & Replace each token.
    """
    def nn(x, nd=2):
        if x is None:
            return "---"
        if isinstance(x, float):
            return f"{x:.{nd}f}"
        return str(x)

    b1, b2, co = summaries["B1"], summaries["B2"], summaries["CO"]

    top_fields = field_counter.most_common(2)
    total_field_mods = sum(field_counter.values()) or 1
    top1 = top_fields[0] if len(top_fields) >= 1 else (None, 0)
    top2 = top_fields[1] if len(top_fields) >= 2 else (None, 0)

    total_mods_att = (b2["mods_att_total"] or 0) + (co["mods_att_total"] or 0)
    total_mods_app = (b2["mods_app_total"] or 0) + (co["mods_app_total"] or 0)
    total_mods_rej = total_mods_att - total_mods_app
    overall_vsr = (total_mods_app / total_mods_att) if total_mods_att else None

    lines = []
    lines.append("% Paste these into Overleaf via Find & Replace.")
    lines.append("% Each line is: [[TODO: KEY]] -> value")
    lines.append("")
    replacements = [
        ("B1\\_n",      b1["n_completed"]),
        ("B2\\_n",      b2["n_completed"]),
        ("CO\\_n",      co["n_completed"]),
        ("B1\\_AQS",    nn(b1["AQS_mean"])),
        ("B2\\_AQS",    nn(b2["AQS_mean"])),
        ("CO\\_AQS",    nn(co["AQS_mean"])),
        ("B1\\_AQS\\_sd", nn(b1["AQS_std"])),
        ("B2\\_AQS\\_sd", nn(b2["AQS_std"])),
        ("CO\\_AQS\\_sd", nn(co["AQS_std"])),
        ("B1\\_AQS\\_range", f'{nn(b1["AQS_min"])}--{nn(b1["AQS_max"])}'),
        ("B2\\_AQS\\_range", f'{nn(b2["AQS_min"])}--{nn(b2["AQS_max"])}'),
        ("CO\\_AQS\\_range", f'{nn(co["AQS_min"])}--{nn(co["AQS_max"])}'),
        ("B1\\_TSR",    nn(b1["TSR"], 3)),
        ("B2\\_TSR",    nn(b2["TSR"], 3)),
        ("CO\\_TSR",    nn(co["TSR"], 3)),
        ("B1\\_ET",     nn(b1["ET_mean"], 1)),
        ("B2\\_ET",     nn(b2["ET_mean"], 1)),
        ("CO\\_ET",     nn(co["ET_mean"], 1)),
        ("B1\\_wc",     nn(b1["wc_mean"], 0)),
        ("B2\\_wc",     nn(b2["wc_mean"], 0)),
        ("CO\\_wc",     nn(co["wc_mean"], 0)),
        ("B1\\_cites",  "see CSV"),
        ("B2\\_cites",  "see CSV"),
        ("CO\\_cites",  "see CSV"),
        ("B2\\_mods\\_att", b2["mods_att_total"]),
        ("CO\\_mods\\_att", co["mods_att_total"]),
        ("B2\\_mods\\_app", b2["mods_app_total"]),
        ("CO\\_mods\\_app", co["mods_app_total"]),
        ("total\\_mods\\_attempted", total_mods_att),
        ("total\\_mods\\_applied",   total_mods_app),
        ("total\\_mods\\_rejected",  total_mods_rej),
        ("overall\\_VSR",            nn(overall_vsr, 3)),
        ("top\\_field\\_1\\_name",   top1[0] if top1[0] else "---"),
        ("top\\_field\\_1\\_pct",    nn(100.0 * top1[1] / total_field_mods, 1) if top1[0] else "---"),
        ("top\\_field\\_2\\_name",   top2[0] if top2[0] else "---"),
        ("top\\_field\\_2\\_pct",    nn(100.0 * top2[1] / total_field_mods, 1) if top2[0] else "---"),
        ("t\\_B1\\_B2",              nn(b1_vs_b2.get("t"), 3)),
        ("p\\_B1\\_B2",              nn(b1_vs_b2.get("p"), 4)),
    ]
    for key, val in replacements:
        lines.append(f"[[TODO: {key}]]  ->  {val}")
    lines.append("")
    # Interpretation sentence for t_B1_B2 / p_B1_B2 stub.
    p = b1_vs_b2.get("p")
    if p is None:
        interp = "Interpretation could not be computed (insufficient n)."
    elif p < 0.01:
        direction = "higher" if b2["AQS_mean"] > b1["AQS_mean"] else "lower"
        interp = (f"B2 is {direction} than B1 in mean AQS and the difference "
                  f"is statistically significant at the 1\\% level.")
    elif p < 0.05:
        direction = "higher" if b2["AQS_mean"] > b1["AQS_mean"] else "lower"
        interp = (f"B2 is {direction} than B1 in mean AQS and the difference "
                  f"is statistically significant at the 5\\% level but not at 1\\%.")
    else:
        interp = ("The observed mean difference between B1 and B2 is not "
                  "statistically significant at the 5\\% level, so we cannot "
                  "reject the null hypothesis that self-modification has no effect on AQS.")
    lines.append(f"[[TODO: one-sentence interpretation: ...]]  ->  {interp}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b1", type=Path, default=Path("b1_results.jsonl"))
    ap.add_argument("--b2", type=Path, default=Path("b2_results.jsonl"))
    ap.add_argument("--co", type=Path, default=Path("co_results.jsonl"))
    ap.add_argument("--out-md",   type=Path, default=Path("batch_summary.md"))
    ap.add_argument("--out-tex",  type=Path, default=Path("batch_placeholders.tex"))
    ap.add_argument("--out-json", type=Path, default=Path("batch_values.json"))
    ap.add_argument("--pg-url",   type=str, default="postgresql://sira:sira@localhost:5434/sira",
                    help="Postgres URL for modification-log enrichment; pass empty to skip")
    args = ap.parse_args()

    b1_rows = load_jsonl(args.b1)
    b2_rows = load_jsonl(args.b2)
    co_rows = load_jsonl(args.co)

    print(f"B1: {len(b1_rows)} rows   B2: {len(b2_rows)} rows   CO: {len(co_rows)} rows")

    b1_sum = summarize_condition(b1_rows, "B1")
    b2_sum = summarize_condition(b2_rows, "B2")
    co_sum = summarize_condition(co_rows, "CO")

    b1_vs_b2 = welch_t_test(b1_sum["_scores"], b2_sum["_scores"])
    b2_vs_co = welch_t_test(b2_sum["_scores"], co_sum["_scores"])

    field_counter = modification_field_distribution(b2_rows + co_rows, args.pg_url or None)

    # ── Markdown report ──
    md = []
    md.append("# Batch Analysis\n")
    md.append(f"B1 observed / intended: {b1_sum['n_completed']} / {b1_sum['n_intended']}")
    md.append(f"B2 observed / intended: {b2_sum['n_completed']} / {b2_sum['n_intended']}")
    md.append(f"CO observed / intended: {co_sum['n_completed']} / {co_sum['n_intended']}")
    md.append("")
    md.append("## Condition summary\n")
    md.append(render_md_table([b1_sum, b2_sum, co_sum]))
    md.append("")
    md.append("## Welch t-tests (two-tailed)\n")
    md.append(f"- B1 vs B2: t={b1_vs_b2['t']}, df={b1_vs_b2['df']}, p={b1_vs_b2['p']}  {b1_vs_b2['note']}")
    md.append(f"- B2 vs CO: t={b2_vs_co['t']}, df={b2_vs_co['df']}, p={b2_vs_co['p']}  {b2_vs_co['note']}")
    md.append("")
    if field_counter:
        md.append("## Self-modification target fields (B2+CO combined)\n")
        md.append("| Target path | Count |")
        md.append("|---|---|")
        for k, v in field_counter.most_common():
            md.append(f"| `{k}` | {v} |")
        md.append("")
    else:
        md.append("## Self-modification target fields\n")
        md.append("_(DB enrichment not available or no modifications yet.)_")
        md.append("")
    md.append("## CO learning curve (per-run AQS, in order)\n")
    co_aqs_ordered = [r.get("final_score") for r in co_rows if r.get("status") == "complete"]
    md.append("```")
    md.append(" ".join(str(x) for x in co_aqs_ordered))
    md.append("```")
    md.append("")

    args.out_md.write_text("\n".join(md))
    print(f"Wrote {args.out_md}")

    # ── LaTeX placeholder dump ──
    tex = render_latex_placeholders(
        {"B1": b1_sum, "B2": b2_sum, "CO": co_sum},
        b1_vs_b2, field_counter
    )
    args.out_tex.write_text(tex)
    print(f"Wrote {args.out_tex}")

    # ── JSON machine-readable dump ──
    def _strip_private(d: dict) -> dict:
        return {k: v for k, v in d.items() if not k.startswith("_")}
    payload = {
        "B1": _strip_private(b1_sum),
        "B2": _strip_private(b2_sum),
        "CO": _strip_private(co_sum),
        "B1_vs_B2": b1_vs_b2,
        "B2_vs_CO": b2_vs_co,
        "field_distribution": dict(field_counter),
        "co_learning_curve": co_aqs_ordered,
    }
    args.out_json.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
