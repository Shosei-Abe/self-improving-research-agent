"""
analyze_runs.py
===============

Extracts metrics from the runs + checkpoints tables and produces CSV and
Markdown summaries suitable for thesis §4 evidence tables. Designed to run
against the Postgres instance that agent_backend.py uses.

Usage:
    python3.11 analyze_runs.py                # writes to current dir
    python3.11 analyze_runs.py --tsr 70       # TSR threshold (default 70)
    python3.11 analyze_runs.py --out /tmp/    # output directory
    python3.11 analyze_runs.py --limit 20     # how many latest runs

Outputs:
    runs_analysis.csv   — one row per run, all metrics flattened
    runs_summary.md     — human-readable summary with aggregates
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

# We need psycopg; it's already in .venv for the backend
try:
    import psycopg
except ImportError:
    print("ERROR: psycopg not available. Run this inside the backend's .venv.", file=sys.stderr)
    sys.exit(1)

# Match agent_backend.py's default connection string. Override via env.
DEFAULT_PG_URL = os.environ.get(
    "SIRA_PG_URL",
    "postgresql://sira:sira@localhost:5434/sira",
)


# ─────────────────────────────────────────────────────────────────────────────
# Metric extraction — operates on a single checkpoint's state_json
# ─────────────────────────────────────────────────────────────────────────────


def extract_metrics(state: dict[str, Any]) -> dict[str, Any]:
    """Pull the metrics we care about from the last verification_results entry.

    Returns a flat dict of field -> value (or None if missing).
    All field names use snake_case for SQL/CSV friendliness.
    """
    vr = state.get("verification_results", [])
    if not vr:
        return {
            "iteration_count": 0,
            "verified_score": None,
            "word_count": None,
        }

    last = vr[-1]
    m = last.get("metrics", {})
    checks = last.get("checks", [])
    recon = last.get("reconciliation", {})

    # Count check statuses
    status_counts = Counter(c.get("status") for c in checks)
    # Treat WARN and WARNING as equivalent
    warn_count = status_counts.get("WARN", 0) + status_counts.get("WARNING", 0)

    failed_categories = sorted(set(
        c.get("category", "?")
        for c in checks
        if c.get("status") == "FAIL"
    ))

    return {
        "iteration_count": len(vr),
        "verified_score": last.get("score"),
        "overall_status": last.get("overall_status"),
        # deterministic metrics
        "word_count": m.get("wordCount"),
        "paragraph_count": m.get("paragraphCount"),
        "paragraphs_with_citations": m.get("paragraphsWithCitations"),
        "citation_count": m.get("citationCount"),
        "unique_citation_count": m.get("uniqueCitationCount"),
        "orphan_citation_count": len(m.get("orphanCitations") or []),
        "reference_count": m.get("referenceCount"),
        "sections_present": m.get("sectionsPresent"),
        "sections_total": m.get("sectionsTotal"),
        "sections_missing_count": len(m.get("sectionsMissing") or []),
        "thin_sections_count": len(m.get("thinSections") or []),
        "code_blocks": m.get("codeBlocks"),
        "equation_count": m.get("equationCount"),
        "numeric_claims": m.get("numericClaims"),
        "statistical_terms": m.get("statisticalTerms"),
        "weak_phrases": m.get("weakPhrases"),
        "deterministic_score": m.get("deterministicScore"),
        # reconciliation
        "llm_raw_score": recon.get("llm"),
        "deterministic_baseline": recon.get("deterministic"),
        "reconciled_final": recon.get("final"),
        # check statistics
        "checks_total": len(checks),
        "checks_pass": status_counts.get("PASS", 0),
        "checks_fail": status_counts.get("FAIL", 0),
        "checks_warn": warn_count,
        "failed_categories": "|".join(failed_categories) or "",
        # derived
        "citation_density": (
            round(
                (m.get("paragraphsWithCitations", 0) or 0)
                / (m.get("paragraphCount") or 1),
                3,
            )
            if m.get("paragraphCount")
            else None
        ),
        "citation_diversity": (
            round(
                (m.get("uniqueCitationCount", 0) or 0)
                / (m.get("citationCount") or 1),
                3,
            )
            if m.get("citationCount")
            else None
        ),
    }


def extract_modification_stats(state: dict[str, Any]) -> dict[str, Any]:
    """Stats about the self-modification log."""
    mod_log = state.get("modification_log", [])
    applied = sum(1 for m in mod_log if m.get("applied"))
    proposed = len(mod_log)
    # VSR: verified (i.e. all obligations proved) / proposed
    # A proposal is "verified" if verdict.verdict == "APPROVED"
    verified = sum(
        1 for m in mod_log
        if m.get("verdict", {}).get("verdict") == "APPROVED"
    )
    return {
        "mods_proposed": proposed,
        "mods_verified": verified,
        "mods_applied": applied,
        "vsr": round(verified / proposed, 3) if proposed else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DB access
# ─────────────────────────────────────────────────────────────────────────────


def fetch_rows(pg_url: str, limit: int) -> list[dict[str, Any]]:
    """LEFT JOIN runs with their latest checkpoint."""
    sql = """
        WITH latest_cp AS (
            SELECT DISTINCT ON (run_id)
                run_id, iteration, state_json, created_at AS cp_created_at
            FROM checkpoints
            ORDER BY run_id, iteration DESC, id DESC
        )
        SELECT
            r.run_id, r.topic, r.status, r.final_score,
            r.iterations, r.created_at, r.updated_at,
            lc.state_json, lc.iteration AS latest_cp_iteration
        FROM runs r
        LEFT JOIN latest_cp lc ON lc.run_id = r.run_id
        ORDER BY r.created_at DESC
        LIMIT %s
    """
    out: list[dict[str, Any]] = []
    with psycopg.connect(pg_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            for row in cur.fetchall():
                out.append({
                    "run_id": row[0],
                    "topic": row[1],
                    "status": row[2],
                    "final_score": row[3],
                    "iterations": row[4],
                    "created_at": row[5],
                    "updated_at": row[6],
                    "state_json": row[7],
                    "latest_cp_iteration": row[8],
                })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation + report
# ─────────────────────────────────────────────────────────────────────────────


def run_to_row(row: dict[str, Any]) -> dict[str, Any]:
    """Flatten one run + its latest checkpoint into a single CSV row."""
    state = row.get("state_json") or {}
    et = None
    if row["created_at"] and row["updated_at"]:
        et = (row["updated_at"] - row["created_at"]).total_seconds()

    base = {
        "run_id": row["run_id"],
        "topic": row["topic"],
        "status": row["status"],
        "final_score": row["final_score"],
        "iterations_db": row["iterations"],
        "latest_cp_iteration": row["latest_cp_iteration"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else "",
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else "",
        "execution_seconds": round(et, 1) if et is not None else None,
    }
    base.update(extract_metrics(state))
    base.update(extract_modification_stats(state))
    return base


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        print("No rows to write.", file=sys.stderr)
        return
    # Unified column order: identity first, then score, then metrics, then mods
    order = [
        "run_id", "topic", "status", "final_score", "execution_seconds",
        "iterations_db", "latest_cp_iteration",
        "verified_score", "deterministic_score", "llm_raw_score",
        "reconciled_final", "deterministic_baseline",
        "word_count", "paragraph_count", "paragraphs_with_citations",
        "citation_count", "unique_citation_count", "orphan_citation_count",
        "citation_density", "citation_diversity",
        "reference_count", "sections_present", "sections_total",
        "sections_missing_count", "thin_sections_count",
        "code_blocks", "equation_count", "numeric_claims",
        "statistical_terms", "weak_phrases",
        "checks_total", "checks_pass", "checks_fail", "checks_warn",
        "failed_categories",
        "mods_proposed", "mods_verified", "mods_applied", "vsr",
        "overall_status", "iteration_count",
        "created_at", "updated_at",
    ]
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    # Any extras we forgot go at the end
    extras = sorted(all_keys - set(order))
    cols = order + extras

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {out_path} ({len(rows)} rows)")


def mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


def stddev(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return round(var ** 0.5, 2)


def write_markdown(rows: list[dict[str, Any]], out_path: Path, tsr_threshold: float) -> None:
    completed = [r for r in rows if r["status"] == "completed"]

    lines: list[str] = []
    lines.append("# Run Analysis Report")
    lines.append("")
    lines.append(f"Generated: {datetime.utcnow().isoformat()}Z")
    lines.append(f"Total runs examined: {len(rows)} ({len(completed)} completed)")
    lines.append(f"TSR threshold: final_score >= {tsr_threshold}")
    lines.append("")

    # Per-run table
    lines.append("## Per-run table (completed runs, newest first)")
    lines.append("")
    headers = ["Topic", "Score", "ET(s)", "Words", "Unique cites", "Orphan", "Sects", "FAIL", "Mods app/ver/prop", "Run"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for r in completed:
        topic = (r["topic"] or "")[:40]
        score = r["final_score"]
        et = r["execution_seconds"]
        wc = r["word_count"]
        uc = r["unique_citation_count"]
        orph = r["orphan_citation_count"]
        sects = f"{r['sections_present']}/{r['sections_total']}" if r.get("sections_total") else "?"
        fails = r["checks_fail"]
        mods = f"{r['mods_applied']}/{r['mods_verified']}/{r['mods_proposed']}"
        run_short = r["run_id"][-10:]
        lines.append(f"| {topic} | {score} | {et} | {wc} | {uc} | {orph} | {sects} | {fails} | {mods} | `…{run_short}` |")
    lines.append("")

    # Aggregates (per topic when more than one run)
    topic_runs: dict[str, list[dict[str, Any]]] = {}
    for r in completed:
        topic_runs.setdefault(r["topic"], []).append(r)

    multi_topic = {t: rs for t, rs in topic_runs.items() if len(rs) >= 2}
    if multi_topic:
        lines.append("## Per-topic aggregates (where n >= 2)")
        lines.append("")
        lines.append("| Topic | n | AQS mean | AQS stddev | AQS min-max | ET mean(s) |")
        lines.append("|---|---|---|---|---|---|")
        for topic, rs in multi_topic.items():
            scores = [r["final_score"] for r in rs if r["final_score"] is not None]
            ets = [r["execution_seconds"] for r in rs if r["execution_seconds"] is not None]
            if not scores:
                continue
            lines.append(
                f"| {topic[:50]} | {len(scores)} | {mean(scores)} | {stddev(scores)} | "
                f"{min(scores)}-{max(scores)} | {mean(ets)} |"
            )
        lines.append("")

    # Overall aggregates
    if completed:
        scores = [r["final_score"] for r in completed if r["final_score"] is not None]
        ets = [r["execution_seconds"] for r in completed if r["execution_seconds"] is not None]
        words = [r["word_count"] for r in completed if r["word_count"] is not None]
        unique_cites = [r["unique_citation_count"] for r in completed if r["unique_citation_count"] is not None]
        orphans = [r["orphan_citation_count"] for r in completed if r["orphan_citation_count"] is not None]
        vsrs = [r["vsr"] for r in completed if r["vsr"] is not None]

        passed_tsr = sum(1 for s in scores if s >= tsr_threshold)
        tsr = round(passed_tsr / len(scores), 3) if scores else None

        lines.append("## Overall (all completed runs)")
        lines.append("")
        lines.append(f"- **AQS** mean: {mean(scores)}, stddev: {stddev(scores)}, min-max: {min(scores) if scores else '?'}-{max(scores) if scores else '?'}")
        lines.append(f"- **TSR** (score >= {tsr_threshold}): {passed_tsr}/{len(scores)} = {tsr}")
        lines.append(f"- **ET** mean: {mean(ets)}s, stddev: {stddev(ets)}s")
        lines.append(f"- **Word count** mean: {mean(words)}, stddev: {stddev(words)}")
        lines.append(f"- **Unique citations** mean: {mean(unique_cites)}, stddev: {stddev(unique_cites)}")
        lines.append(f"- **Orphan citations** mean: {mean(orphans)}")
        if vsrs:
            lines.append(f"- **VSR** mean: {mean(vsrs)} (from {len(vsrs)} runs with mods)")
        lines.append("")

    # Failed check category distribution
    cat_counter: Counter[str] = Counter()
    for r in completed:
        cats = r.get("failed_categories", "")
        if cats:
            for c in cats.split("|"):
                cat_counter[c] += 1
    if cat_counter:
        lines.append("## Failed check categories across all completed runs")
        lines.append("")
        lines.append("| Category | # runs where it failed |")
        lines.append("|---|---|")
        for cat, n in cat_counter.most_common():
            lines.append(f"| {cat} | {n} |")
        lines.append("")

    # Caveats
    lines.append("## Caveats")
    lines.append("")
    lines.append("- IAR and IE are not computed: they require multi-iteration runs (Passes >= 3)")
    lines.append("  to observe per-iteration score deltas.")
    lines.append("- VSR is meaningful only when `mods_proposed > 0`; Pass=1 runs often have 0 or 1.")
    lines.append("- 'failed_categories' merges both deterministic checks (lowercase like 'word_count')")
    lines.append("  and LLM-produced checks (uppercase like 'STRUCTURAL'); same concept may appear twice.")
    lines.append("- ET (execution_seconds) measures wall clock from POST /state/run to finalize —")
    lines.append("  includes LLM latency, not pipeline overhead alone.")

    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze SIRA runs from Postgres.")
    ap.add_argument("--tsr", type=float, default=70.0, help="TSR threshold (default: 70)")
    ap.add_argument("--limit", type=int, default=50, help="Number of latest runs to analyze")
    ap.add_argument("--out", type=str, default=".", help="Output directory")
    ap.add_argument("--pg-url", type=str, default=DEFAULT_PG_URL, help="Postgres connection URL")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to {args.pg_url.split('@')[-1]}...")
    raw_rows = fetch_rows(args.pg_url, args.limit)
    print(f"Fetched {len(raw_rows)} runs")

    rows = [run_to_row(r) for r in raw_rows]

    write_csv(rows, out_dir / "runs_analysis.csv")
    write_markdown(rows, out_dir / "runs_summary.md", args.tsr)


if __name__ == "__main__":
    main()
