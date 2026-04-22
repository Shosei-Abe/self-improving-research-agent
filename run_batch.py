#!/usr/bin/env python3.11
"""
run_batch.py — Semi-automated batch runner for phase 2 measurement.

Runs 30 pipeline jobs under each of three conditions:
  * B1  — self-modification disabled (disable_self_mod=True, fresh default config)
  * B2  — self-modification enabled, config reset to default each run
  * CO  — "carry-over": self-modification enabled, SAME topic repeated, config NOT reset
          (the backend picks up whatever config the previous run left in its own
          run record; for CO we let each run's final config flow into the next via
          a script-side cache rather than localStorage, since backend runs are
          independent processes.)

Usage
-----
    python3.11 run_batch.py --condition B1 --count 30 --out b1_results.jsonl
    python3.11 run_batch.py --condition B2 --count 30 --out b2_results.jsonl
    python3.11 run_batch.py --condition CO --count 30 --out co_results.jsonl \
        --carry-topic "Mitigating LLM hallucinations"

Each run emits a single JSON line to --out when it finishes (or times out).
Fields: run_id, condition, topic, index, started_at, finished_at, status,
        final_score, word_count, mods_attempted, mods_applied, elapsed_s,
        initial_config, final_config, error.

The script does NOT parse the paper itself — that lives in Postgres via the
backend's existing persistence. Use analyze_runs.py afterward.

Design notes
------------
* We use the /pipeline/start + SSE /pipeline/stream endpoints. We consume the
  stream with httpx and break on the first `complete` or `error` event, or when
  the run wall-clock exceeds --timeout-min.
* B2 forces a default system_config into every POST body, which is how we
  guarantee "reset" semantics from the script side (the backend would otherwise
  preserve whatever the user sent, but it does NOT persist prior runs' configs
  into new runs — it just uses what's in the request, defaulting if empty.
  So sending default-cfg == reset.)
* For CO we query /state/runs after each run completes, pull the most recent
  run's final_config (requires the day1 DB migration), and feed it back in as
  the next run's system_config.
* Credit exhaustion: the script stops on HTTP 402/429 and writes a sentinel
  line {"status": "credit_exhausted"} so analyze_runs.py can tell observed n
  apart from intended n.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

BACKEND = os.environ.get("BACKEND_URL", "http://localhost:8000")

# Default system_config (matches backend default in agent_backend.py L1340)
DEFAULT_CFG: dict = {
    "generation": {
        "minWordsPerSection": 150,
        "targetTotalWords": 3000,
        "citationsPerParagraph": 2,
        "requireQuantitativeData": True,
        "paragraphsPerSection": 3,
    },
    "verification": {
        "weightWordCount": 25,
        "weightSections": 24,
        "weightCitations": 25,
        "weightReferences": 15,
        "weightMathRigor": 10,
        "penaltyThinSection": 3,
        "penaltyMissingSection": 4,
        "penaltyOrphanCitation": 1,
        "penaltyWeakPhrase": 1,
        "reconciliationBand": 8,
    },
    "improvement": {"targetIncrement": 10, "aggressiveness": 0.5, "focusOnFailures": True},
    "__version": 1,
}

# 30 topics — mix of research-oriented, technical, and doc-style to avoid
# monoculture bias in B1/B2. CO uses just one, repeated.
TOPICS: list[str] = [
    "Mitigating LLM hallucinations",
    "Bayesian optimization for hyperparameter tuning",
    "Safety in deep reinforcement learning",
    "Federated learning with differential privacy",
    "Explainability methods for convolutional networks",
    "Curriculum learning in language models",
    "Retrieval-augmented generation architectures",
    "Active learning under label noise",
    "Self-supervised representation learning for time series",
    "Adversarial robustness of vision transformers",
    "Graph neural networks for molecular property prediction",
    "Continual learning without catastrophic forgetting",
    "Program synthesis from natural language specifications",
    "Formal verification of smart contracts",
    "Model-based reinforcement learning for robotics",
    "Neural architecture search under compute constraints",
    "Calibration of probabilistic classifiers",
    "Fairness-aware recommender systems",
    "Zero-shot cross-lingual transfer in multilingual models",
    "Memory-efficient training of large language models",
    "Causal inference from observational data",
    "Out-of-distribution detection for safety-critical systems",
    "Knowledge distillation for edge deployment",
    "Contrastive learning for sentence embeddings",
    "Reinforcement learning from human feedback",
    "Mixture-of-experts routing strategies",
    "Diffusion models for structured data generation",
    "Privacy-preserving synthetic data generation",
    "Automated theorem proving with neural guidance",
    "Robust optimization under distribution shift",
]
assert len(TOPICS) == 30, f"expected 30 topics, got {len(TOPICS)}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def wait_for_complete(
    client: httpx.AsyncClient, run_id: str, timeout_s: float
) -> tuple[str, dict]:
    """
    Consume /pipeline/stream/{run_id} until `complete` or `error`.
    Returns (status, last_payload). status in {"complete", "error", "timeout"}.
    """
    url = f"{BACKEND}/pipeline/stream/{run_id}"
    last: dict = {}
    start = time.time()
    try:
        async with client.stream("GET", url, timeout=None) as resp:
            resp.raise_for_status()
            event_kind = None
            async for line in resp.aiter_lines():
                if time.time() - start > timeout_s:
                    return "timeout", last
                if not line:
                    continue
                if line.startswith("event:"):
                    event_kind = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    try:
                        data = json.loads(line.split(":", 1)[1].strip() or "{}")
                    except json.JSONDecodeError:
                        data = {}
                    last = {"kind": event_kind, "data": data}
                    if event_kind == "complete":
                        return "complete", last
                    if event_kind == "error":
                        return "error", last
                    if event_kind == "end":
                        # stream closed by server
                        return "complete" if last.get("kind") == "complete" else "error", last
    except httpx.HTTPError as e:
        return "error", {"kind": "http_error", "data": {"detail": str(e)}}
    return "timeout", last


async def fetch_run_summary(client: httpx.AsyncClient, run_id: str) -> dict:
    """
    Stitch together a per-run summary from three endpoints, because
    /state/runs alone doesn't carry mod counts, word counts, or configs.

    * /state/runs           → status, final_score, iterations, topic
    * /state/checkpoint/{id} → state_json with paper, system_config, modification_log
    * /logs/{id}/modifications → authoritative count of attempted/applied mods

    Returns {} only if all three fail. Otherwise returns a flat dict with the
    fields run_batch.py cares about. Missing fields are None (not absent).
    """
    out: dict = {
        "run_id": run_id,
        "status": None,
        "final_score": None,
        "iterations": None,
        "topic": None,
        "word_count": None,
        "mods_attempted": None,
        "mods_applied": None,
        "initial_config": None,
        "final_config": None,
    }

    # 1) /state/runs — cheap, gives status + final_score
    try:
        r = await client.get(f"{BACKEND}/state/runs?limit=100", timeout=30.0)
        r.raise_for_status()
        for row in r.json().get("runs", []):
            if row.get("run_id") == run_id:
                out["status"] = row.get("status")
                out["final_score"] = row.get("final_score")
                out["iterations"] = row.get("iterations")
                out["topic"] = row.get("topic")
                break
    except httpx.HTTPError:
        pass

    # 2) /state/checkpoint/{id} — latest checkpoint has full state_json
    try:
        r = await client.get(f"{BACKEND}/state/checkpoint/{run_id}", timeout=30.0)
        if r.status_code == 200:
            state = (r.json() or {}).get("state") or {}
            # state_json shape: AgentState TypedDict (models.py)
            paper = state.get("paper") or ""
            out["word_count"] = len([w for w in paper.split() if w]) if paper else None
            cfg = state.get("system_config")
            if isinstance(cfg, dict):
                out["final_config"] = cfg
            # we don't have initial_config here — that's set at run creation
            # time. pipeline.py writes it to the runs table via /state/run
            # finalize. If needed we can query a dedicated endpoint later.
    except httpx.HTTPError:
        pass

    # 3) /logs/{id}/modifications — authoritative mod counts
    try:
        r = await client.get(f"{BACKEND}/logs/{run_id}/modifications", timeout=30.0)
        if r.status_code == 200:
            data = r.json() or {}
            entries = data.get("entries") or []
            out["mods_attempted"] = len(entries)
            out["mods_applied"] = sum(1 for e in entries if e.get("applied"))
    except httpx.HTTPError:
        pass

    return out


async def latest_final_config(client: httpx.AsyncClient, run_id: str) -> dict | None:
    """For CO mode: pull the just-finished run's final_config from checkpoint."""
    summary = await fetch_run_summary(client, run_id)
    fc = summary.get("final_config")
    return fc if isinstance(fc, dict) and fc else None


async def start_one(
    client: httpx.AsyncClient,
    topic: str,
    condition: str,
    system_config: dict,
    max_iterations: int,
) -> tuple[int, str | None, dict | None]:
    """Returns (http_status, run_id, error_info). run_id None on failure."""
    body = {
        "topic": topic,
        "max_iterations": max_iterations,
        "system_config": system_config,
        "disable_self_mod": (condition == "B1"),
        "reset_config": (condition == "B2"),
    }
    try:
        r = await client.post(f"{BACKEND}/pipeline/start", json=body, timeout=60.0)
    except httpx.HTTPError as e:
        return -1, None, {"error": f"request failed: {e}"}

    if r.status_code == 402:
        return 402, None, {"error": "credit_exhausted (402)"}
    if r.status_code == 429:
        return 429, None, {"error": "rate_limited (429)"}
    if r.status_code != 200:
        return r.status_code, None, {"error": f"http {r.status_code}: {r.text[:300]}"}

    try:
        return 200, r.json()["run_id"], None
    except (KeyError, json.JSONDecodeError) as e:
        return r.status_code, None, {"error": f"bad response: {e}"}


async def run_batch(
    condition: str,
    count: int,
    out_path: Path,
    max_iterations: int,
    timeout_min: float,
    carry_topic: str | None,
    start_index: int,
) -> None:
    assert condition in {"B1", "B2", "CO"}
    timeout_s = timeout_min * 60.0

    topics: list[str]
    if condition == "CO":
        topic = carry_topic or TOPICS[0]
        topics = [topic] * count
    else:
        # cycle through TOPICS[start_index : start_index+count]
        topics = [TOPICS[(start_index + i) % len(TOPICS)] for i in range(count)]

    print(f"[batch] condition={condition} count={count} out={out_path}")
    print(f"[batch] max_iter={max_iterations} timeout_min={timeout_min}")
    print(f"[batch] backend={BACKEND}")

    # carry-over config state (CO only)
    carry_cfg: dict = DEFAULT_CFG

    async with httpx.AsyncClient() as client:
        # sanity check backend reachable
        try:
            h = await client.get(f"{BACKEND}/health", timeout=10.0)
            h.raise_for_status()
        except httpx.HTTPError as e:
            print(f"[batch] FATAL: backend unreachable at {BACKEND}: {e}", file=sys.stderr)
            sys.exit(2)

        with out_path.open("a", buffering=1) as f:
            for i, topic in enumerate(topics):
                abs_idx = start_index + i
                t0 = time.time()
                started_at = now_iso()

                if condition == "CO":
                    cfg = dict(carry_cfg)
                elif condition == "B2":
                    cfg = dict(DEFAULT_CFG)
                else:  # B1
                    cfg = dict(DEFAULT_CFG)

                print(f"[batch] [{i+1}/{count}] start: {topic[:60]}", flush=True)

                http_code, run_id, err = await start_one(
                    client, topic, condition, cfg, max_iterations
                )
                if http_code in (402, 429) or run_id is None:
                    rec = {
                        "status": "credit_exhausted" if http_code == 402 else
                                  "rate_limited" if http_code == 429 else
                                  "start_failed",
                        "condition": condition,
                        "topic": topic,
                        "index": abs_idx,
                        "started_at": started_at,
                        "finished_at": now_iso(),
                        "error": (err or {}).get("error"),
                    }
                    f.write(json.dumps(rec) + "\n")
                    if http_code == 402:
                        print("[batch] ABORT: credit exhausted — add funds and resume with --start-index", file=sys.stderr)
                        return
                    if http_code == 429:
                        # back off and retry once
                        print("[batch] rate limited, sleeping 60s then retrying", file=sys.stderr)
                        await asyncio.sleep(60)
                        http_code, run_id, err = await start_one(
                            client, topic, condition, cfg, max_iterations
                        )
                        if run_id is None:
                            print("[batch] retry failed, skipping this slot", file=sys.stderr)
                            continue
                    else:
                        continue

                status, last = await wait_for_complete(client, run_id, timeout_s)
                # fetch final summary from DB
                summary = await fetch_run_summary(client, run_id)

                rec = {
                    "status": status,  # complete / error / timeout
                    "condition": condition,
                    "topic": topic,
                    "index": abs_idx,
                    "run_id": run_id,
                    "started_at": started_at,
                    "finished_at": now_iso(),
                    "elapsed_s": round(time.time() - t0, 1),
                    "final_score": summary.get("final_score"),
                    "word_count": summary.get("word_count"),
                    "mods_attempted": summary.get("mods_attempted"),
                    "mods_applied": summary.get("mods_applied"),
                    "initial_config": summary.get("initial_config"),
                    "final_config": summary.get("final_config"),
                    "last_event": last.get("kind"),
                    "error": None if status == "complete" else (last.get("data") or {}).get("message"),
                }
                f.write(json.dumps(rec) + "\n")
                print(
                    f"[batch] [{i+1}/{count}] {status} score={rec['final_score']} "
                    f"mods={rec['mods_attempted']}/{rec['mods_applied']} "
                    f"elapsed={rec['elapsed_s']}s run_id={run_id}",
                    flush=True,
                )

                # CO: update carry config for next iteration
                if condition == "CO":
                    nxt = await latest_final_config(client, run_id)
                    if nxt:
                        carry_cfg = nxt
                    # else: keep previous carry_cfg

                # small delay to avoid hammering
                await asyncio.sleep(2.0)

    print(f"[batch] done. results → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["B1", "B2", "CO"])
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-iterations", type=int, default=3)
    ap.add_argument("--timeout-min", type=float, default=25.0,
                    help="per-run wall-clock cap in minutes")
    ap.add_argument("--carry-topic", default="Mitigating LLM hallucinations",
                    help="topic used for CO condition (ignored for B1/B2)")
    ap.add_argument("--start-index", type=int, default=0,
                    help="resume after partial run; offsets into TOPICS and the index field")
    args = ap.parse_args()

    asyncio.run(
        run_batch(
            condition=args.condition,
            count=args.count,
            out_path=args.out,
            max_iterations=args.max_iterations,
            timeout_min=args.timeout_min,
            carry_topic=args.carry_topic,
            start_index=args.start_index,
        )
    )


if __name__ == "__main__":
    main()
