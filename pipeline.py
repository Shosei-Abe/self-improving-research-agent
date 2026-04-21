"""
pipeline.py
===========

The LangGraph state machine for the self-improving research agent.

This is the actual `langgraph.graph.StateGraph` the thesis (§3.1.1, §3.4.1,
Appendix A.3) describes. Five nodes — orchestrator, research, generation,
verification, feedback — plus a self-modification step inside the
verification node, wired together with conditional edges that drive the
iterative refinement loop.

The graph is built once at module load and reused across runs. Each run
gets its own AgentState instance, which LangGraph's checkpointer mirrors
to Postgres via PostgresSaver.

Async streaming
---------------
The pipeline yields StreamEvent objects through an asyncio.Queue. The
FastAPI endpoint `/pipeline/stream/{run_id}` consumes that queue and emits
Server-Sent Events to the React app.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable

from langgraph.graph import StateGraph, END

from models import AgentState, StreamEvent
from prompts import (
    plan_system_prompt,
    research_system_prompt,
    generation_system_prompt,
    improve_system_prompt,
    verify_system_prompt,
    feedback_system_prompt,
)
from metrics import compute_metrics, reconcile_score, metrics_to_checks
from self_modification import propose_self_modification, verify_modification_with_z3

# Anthropic SDK
try:
    from anthropic import AsyncAnthropic

    _client = AsyncAnthropic()
    _ANTHROPIC_AVAILABLE = True
except Exception:
    _client = None
    _ANTHROPIC_AVAILABLE = False

# httpx for DB persistence calls back into our own FastAPI backend.
# We call localhost via HTTP rather than importing the DB pool directly so
# pipeline.py stays decoupled from the backend process boundary and remains
# testable without Postgres.
try:
    import httpx

    _HTTPX_AVAILABLE = True
except Exception:
    _HTTPX_AVAILABLE = False

MODEL = "claude-sonnet-4-20250514"
# Claude Sonnet 4 supports up to 64000 output tokens. We use 16000 which is
# enough for ~10000-word papers with margin, so self-modification can raise
# targetTotalWords up to 6000 (see self_modification.py Heuristic 3) without
# hitting the token ceiling. Previous value of 4096 truncated every paper
# mid-generation, producing missing Conclusion/References sections.
MAX_TOKENS = 16000

# Backend base URL for DB persistence round-trips. When pipeline.py runs in
# the same process as agent_backend.py (the normal deployment), this is just
# localhost. Overridable via env var for tests / alternative topologies.
BACKEND_URL = os.environ.get("SIRA_BACKEND_URL", "http://localhost:8000")


# ─────────────────────────────────────────────────────────────────────────────
# Per-run event queues — keyed by run_id, the SSE endpoint reads from these
# ─────────────────────────────────────────────────────────────────────────────

_event_queues: dict[str, asyncio.Queue] = {}
_cancelled_runs: set[str] = set()
# Tracks the most recent verified_score per run_id so run_pipeline() can
# finalize the run with the correct final_score without re-reading the DB.
# Populated in _write_checkpoint(), read in run_pipeline()'s finalize block.
_latest_score: dict[str, float] = {}


def get_queue(run_id: str) -> asyncio.Queue:
    if run_id not in _event_queues:
        _event_queues[run_id] = asyncio.Queue()
    return _event_queues[run_id]


def cancel_run(run_id: str) -> bool:
    """Mark a run as cancelled. The pipeline checks this flag between steps."""
    _cancelled_runs.add(run_id)
    return True


class PipelineCancelled(Exception):
    """Raised when a run is cancelled mid-execution."""


def check_cancelled(run_id: str) -> None:
    if run_id in _cancelled_runs:
        raise PipelineCancelled(run_id)


async def emit(run_id: str, kind: str, payload: dict[str, Any]) -> None:
    check_cancelled(run_id)
    q = get_queue(run_id)
    await q.put(StreamEvent(
        kind=kind,
        payload=payload,
        ts=datetime.now(timezone.utc).isoformat(),
    ))


def emit_sync(run_id: str, kind: str, payload: dict[str, Any]) -> None:
    """Convenience for nodes that aren't async — schedule on the running loop."""
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(emit(run_id, kind, payload))
    except RuntimeError:
        pass  # no loop, drop the event


# ─────────────────────────────────────────────────────────────────────────────
# DB persistence helpers — fire-and-forget HTTP calls to our own backend.
# Any failure here is logged as a warning but NEVER stops the pipeline. DB
# persistence is orthogonal to paper generation; a Postgres outage must not
# block the user from getting their output.
# ─────────────────────────────────────────────────────────────────────────────


async def _db_write(run_id: str, method: str, path: str, json_body: dict[str, Any]) -> dict[str, Any] | None:
    """POST to the backend. Returns None on any failure (silenced)."""
    if not _HTTPX_AVAILABLE:
        return None
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.request(method, f"{BACKEND_URL}{path}", json=json_body)
            if resp.status_code >= 400:
                await emit(run_id, "log", {
                    "agent": "orchestrator",
                    "message": f"DB {method} {path} → {resp.status_code}",
                    "type": "warn",
                })
                return None
            return resp.json()
    except Exception as e:
        # We swallow these so a DB outage doesn't tank the user's generation.
        # emit() itself can raise PipelineCancelled; if it does, let it propagate.
        try:
            await emit(run_id, "log", {
                "agent": "orchestrator",
                "message": f"DB {method} {path} failed: {type(e).__name__}",
                "type": "warn",
            })
        except PipelineCancelled:
            raise
        except Exception:
            pass
        return None


# ─────────────────────────────────────────────────────────────────────────────
# LLM helpers
# ─────────────────────────────────────────────────────────────────────────────


async def call_llm(system: str, user: str) -> str:
    """Call Claude. Falls back to a stub if the SDK is unavailable
    (so the pipeline still type-checks during development)."""
    if not _ANTHROPIC_AVAILABLE:
        return '{"error": "anthropic SDK not configured"}'

    msg = await _client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "\n".join(b.text for b in msg.content if hasattr(b, "text"))


def parse_json(raw: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction from an LLM response."""
    if not raw:
        return None
    # Strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find the first {...} block
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Node: Orchestrator (Plan)
# ─────────────────────────────────────────────────────────────────────────────


async def orchestrator_node(state: AgentState) -> dict[str, Any]:
    run_id = state["run_id"]
    await emit(run_id, "phase", {"phase": "orchestrator"})
    await emit(run_id, "log", {"agent": "orchestrator", "message": "Planning..."})

    # Skip planning on iterations > 1
    if state.get("plan"):
        return {}

    if state.get("is_upload"):
        raw = await call_llm(
            'Analyze this paper and create an improvement plan. Respond ONLY with valid JSON: '
            '{"title":"...","abstract_outline":"...","sections":["..."],"research_queries":["..."],'
            '"key_concepts":["..."],"methodology_hints":"...","reasoning":"...","is_technical_topic":false}',
            f"Paper:\n{state.get('uploaded_paper', '')[:3000]}\n\nCreate improvement plan.",
        )
    else:
        raw = await call_llm(
            plan_system_prompt(state["topic"], ""),
            f'Research topic: "{state["topic"]}"\nCreate execution plan.',
        )

    plan = parse_json(raw) or {
        "title": state.get("topic", "Untitled"),
        "sections": ["Introduction", "Methodology", "Analysis", "Conclusion"],
        "is_technical_topic": False,
        "reasoning": "fallback plan (LLM unparseable)",
    }
    await emit(run_id, "plan", plan)
    await emit(run_id, "log", {
        "agent": "orchestrator",
        "message": f"Plan: \"{plan.get('title', '?')}\" — technical={plan.get('is_technical_topic', False)}",
        "type": "ok",
    })

    return {"plan": plan, "iteration": 1}


# ─────────────────────────────────────────────────────────────────────────────
# Node: Research
# ─────────────────────────────────────────────────────────────────────────────


async def research_node(state: AgentState) -> dict[str, Any]:
    run_id = state["run_id"]
    await emit(run_id, "phase", {"phase": "research"})
    await emit(run_id, "log", {"agent": "research", "message": "Gathering sources..."})

    plan = state.get("plan", {})
    cur = state.get("paper", "")
    iteration = state.get("iteration", 1)

    if iteration == 1 and not state.get("is_upload"):
        user = f"Plan: {json.dumps(plan)}\nResearch for paper."
    else:
        prev_issues = ""
        v_results = state.get("verification_results", [])
        if v_results:
            failed = [c for c in v_results[-1].get("checks", []) if c.get("status") == "FAIL"]
            prev_issues = "\n".join(f"- [{c.get('category','?')}] {c['property']}: {c['details']}" for c in failed[:5])
        user = f"Plan: {json.dumps(plan)}\nDraft (first 2000 chars):\n{cur[:2000]}\nKnown issues: {prev_issues}\nFind additional research."

    raw = await call_llm(research_system_prompt(), user)
    data = parse_json(raw)
    findings_count = len(data.get("findings", [])) if data else 0
    await emit(run_id, "log", {
        "agent": "research",
        "message": f"{findings_count} findings",
        "type": "ok" if data else "warn",
    })

    # Persist findings into the conversation state for the writer
    research_results = list(state.get("research_results", []))
    research_results.append({"iteration": iteration, "raw": raw, "data": data})

    return {"research_results": research_results}


# ─────────────────────────────────────────────────────────────────────────────
# Node: Generation
# ─────────────────────────────────────────────────────────────────────────────


async def generation_node(state: AgentState) -> dict[str, Any]:
    run_id = state["run_id"]
    iteration = state.get("iteration", 1)
    await emit(run_id, "phase", {"phase": "generation"})

    plan = state.get("plan", {})
    cfg = state["system_config"]
    tuning = state.get("tuning", "")
    is_technical = bool(plan.get("is_technical_topic", False))

    research = state.get("research_results", [])
    research_raw = research[-1]["raw"] if research else ""

    if iteration == 1 and not state.get("is_upload"):
        await emit(run_id, "log", {"agent": "generation", "message": "Writing initial draft..."})
        draft = await call_llm(
            generation_system_prompt(tuning, cfg, is_technical),
            f"Plan: {json.dumps(plan)}\n\nResearch: {research_raw}\n\nWrite the COMPLETE paper. Minimum {cfg['generation']['targetTotalWords']} words.",
        )
    else:
        prev_score = state.get("scores", [{}])[-1].get("quality_score", 40) if state.get("scores") else 40
        v_results = state.get("verification_results", [])
        failed = [c for c in v_results[-1].get("checks", []) if c.get("status") == "FAIL"] if v_results else []
        failed_str = "\n".join(f"[{c.get('category', '?')}] {c['property']}: {c['details']}" for c in failed) or "None"
        target = min(prev_score + 10, 98)
        await emit(run_id, "log", {"agent": "generation", "message": f"Improving (target ≥{target})..."})
        draft = await call_llm(
            improve_system_prompt(iteration, prev_score, failed_str, tuning, cfg, is_technical),
            f"Current draft:\n{state.get('paper', '')}\n\nResearch context: {research_raw}\n\nProduce the improved full paper.",
        )

    # Strip code fences if any wrapper appeared
    draft = re.sub(r"^```(?:markdown)?\s*", "", draft.strip())
    draft = re.sub(r"\s*```$", "", draft.strip())

    paper_versions = list(state.get("paper_versions", []))
    paper_versions.append(draft)

    await emit(run_id, "paper_version", {
        "iteration": iteration,
        "length": len(draft),
        "preview": draft[:200],
    })

    return {"paper": draft, "paper_versions": paper_versions}


# ─────────────────────────────────────────────────────────────────────────────
# Node: Verification (deterministic metrics + LLM judge + reconciliation)
# ─────────────────────────────────────────────────────────────────────────────


async def verification_node(state: AgentState) -> dict[str, Any]:
    run_id = state["run_id"]
    iteration = state.get("iteration", 1)
    await emit(run_id, "phase", {"phase": "verification"})
    await emit(run_id, "log", {"agent": "verification", "message": "Computing metrics + LLM check..."})

    cfg = state["system_config"]
    paper = state["paper"]

    # Deterministic metrics
    metrics = compute_metrics(paper, cfg)

    # LLM judge
    prev_issues = ""
    v_prev = state.get("verification_results", [])
    if v_prev:
        prev_issues = json.dumps([c for c in v_prev[-1].get("checks", []) if c.get("status") == "FAIL"][:5])

    raw = await call_llm(
        verify_system_prompt(iteration, prev_issues),
        f"Paper:\n{paper[:6000]}\n\nDeterministic metrics: {json.dumps(metrics)}\n\nVerify and score.",
    )
    llm_data = parse_json(raw) or {"score": metrics["deterministicScore"], "checks": [], "overall_status": "ISSUES_FOUND"}

    # Reconcile
    rec = reconcile_score(llm_data.get("score", 50), metrics, cfg)

    deterministic_checks = metrics_to_checks(metrics, cfg)
    llm_checks = llm_data.get("checks", [])
    all_checks = deterministic_checks + llm_checks

    v_result = {
        "iteration": iteration,
        "score": rec["final"],
        "overall_status": llm_data.get("overall_status", "ISSUES_FOUND"),
        "metrics": metrics,
        "reconciliation": rec,
        "checks": all_checks,
    }

    v_results = list(state.get("verification_results", []))
    v_results.append(v_result)

    await emit(run_id, "verification", v_result)
    # Send the full paper text so the React "paper" tab can display it.
    # SSE payloads are text, so ~30KB of markdown is fine.
    await emit(run_id, "paper_full", {"iteration": iteration, "paper": paper})
    await emit(run_id, "log", {
        "agent": "verification",
        "message": f"Score {rec['final']}/100 (LLM {rec['llm']}, det {rec['deterministic']})",
        "type": "ok",
    })

    return {"verification_results": v_results}


# ─────────────────────────────────────────────────────────────────────────────
# Node: Self-Modification (between verification and feedback)
# ─────────────────────────────────────────────────────────────────────────────


async def self_modification_node(state: AgentState) -> dict[str, Any]:
    run_id = state["run_id"]
    iteration = state.get("iteration", 1)
    cfg = state["system_config"]
    await emit(run_id, "log", {"agent": "orchestrator", "message": "Self-modification: analyzing...", "type": "iter"})

    metrics = state["verification_results"][-1]["metrics"]
    score_history = [s.get("quality_score", 0) for s in state.get("scores", [])]
    score_history.append(state["verification_results"][-1]["score"])

    proposal = propose_self_modification(cfg, metrics, score_history)
    if not proposal:
        await emit(run_id, "log", {"agent": "orchestrator", "message": "No modification proposed", "type": "info"})
        return {}

    await emit(run_id, "log", {"agent": "orchestrator", "message": f"Proposing: {proposal['proposal']['reasoning']}", "type": "info"})

    verdict = verify_modification_with_z3(proposal["newConfig"], cfg, score_history)
    failed_count = sum(1 for o in verdict["obligations"] if o["status"] == "REFUTED")
    total = len(verdict["obligations"])
    await emit(run_id, "log", {
        "agent": "orchestrator",
        "message": f"Verifier: {verdict['verdict']} ({total - failed_count}/{total} obligations proved)",
        "type": "ok" if verdict["verdict"] == "APPROVED" else "warn",
    })

    mod_entry = {
        "iteration": iteration,
        "proposal": proposal["proposal"],
        "verdict": verdict,
        "before_config": cfg,
        "applied": verdict["verdict"] == "APPROVED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    new_state: dict[str, Any] = {}
    if verdict["verdict"] == "APPROVED":
        mod_entry["after_config"] = proposal["newConfig"]
        new_state["system_config"] = proposal["newConfig"]
        diffs_str = ", ".join(f"{d['path']}: {d['from']}→{d['to']}" for d in verdict["diffs"])
        await emit(run_id, "log", {"agent": "orchestrator", "message": f"✓ Self-mod APPLIED: {diffs_str}", "type": "ok"})
    else:
        await emit(run_id, "log", {"agent": "orchestrator", "message": "✗ Self-mod REJECTED", "type": "warn"})

    mod_log = list(state.get("modification_log", []))
    mod_log.append(mod_entry)
    new_state["modification_log"] = mod_log

    await emit(run_id, "modification", mod_entry)

    return new_state


# ─────────────────────────────────────────────────────────────────────────────
# Node: Feedback
# ─────────────────────────────────────────────────────────────────────────────


async def feedback_node(state: AgentState) -> dict[str, Any]:
    run_id = state["run_id"]
    iteration = state.get("iteration", 1)
    await emit(run_id, "phase", {"phase": "feedback"})
    await emit(run_id, "log", {"agent": "feedback", "message": "Evaluating quality..."})

    score_history = ", ".join(str(s.get("quality_score", 0)) for s in state.get("scores", []))
    verified_score = state["verification_results"][-1]["score"]
    paper = state["paper"]

    raw = await call_llm(
        feedback_system_prompt(iteration, score_history),
        f"Paper (first 3000 chars):\n{paper[:3000]}\n\nVerifier score: {verified_score}/100. Your score must be within ±5 of this.\n\nIteration: {iteration}/{state['max_iterations']}",
    )
    data = parse_json(raw)

    if data:
        # Constrain to verifier score ±5
        if abs(data.get("quality_score", 0) - verified_score) > 5:
            adjusted = verified_score + (5 if data.get("quality_score", 0) > verified_score else -5)
            data["quality_score"] = adjusted
        scores = list(state.get("scores", []))
        scores.append(data)
        await emit(run_id, "score", data)
        await emit(run_id, "log", {
            "agent": "feedback",
            "message": f"Quality: {data['quality_score']}/100 · {len(data.get('improvements', []))} suggestions",
            "type": "ok",
        })

        should_stop = data.get("should_continue", True) is False or iteration >= state["max_iterations"]
        await _write_checkpoint(state, iteration, scores, verified_score)
        return {
            "scores": scores,
            "iteration": iteration + 1,
            "should_stop": should_stop,
        }
    else:
        # Fallback to verifier score
        fallback = {"quality_score": verified_score, "improvements": [], "iteration_summary": "feedback unparseable"}
        scores = list(state.get("scores", []))
        scores.append(fallback)
        await _write_checkpoint(state, iteration, scores, verified_score)
        return {
            "scores": scores,
            "iteration": iteration + 1,
            "should_stop": iteration >= state["max_iterations"],
        }


async def _write_checkpoint(
    state: AgentState,
    iteration: int,
    scores: list[dict[str, Any]],
    verified_score: float,
) -> None:
    """Persist a JSONB snapshot of the current iteration to Postgres.

    Called at the end of feedback_node (once per iteration, after scoring).

    day1: now includes the full paper text so downstream analysis
    (metrics.py debugging, citation-pattern audits, paper_versions reconstruction)
    can read paper content directly from the checkpoint. A single 6000-word
    paper is roughly 40-50 KB of text, so with metrics, verification_results
    and modification_log the JSONB row lands around 100-150 KB — well below
    the 1 GB hard limit on a Postgres row, and smaller than a typical LLM
    response payload. The previous "< 100 KB" goal is dropped in favor of
    making the run reproducible from the DB alone.
    """
    run_id = state["run_id"]
    _latest_score[run_id] = verified_score
    snapshot = {
        "topic": state.get("topic"),
        "iteration": iteration,
        "plan": state.get("plan"),
        "scores": scores,
        "verification_results": state.get("verification_results", []),
        "modification_log": state.get("modification_log", []),
        "system_config": state.get("system_config"),
        "verified_score": verified_score,
        "paper_length_chars": len(state.get("paper", "")),
        "paper": state.get("paper", ""),  # day1: store full text for debug/analysis
    }
    await _db_write(run_id, "POST", "/state/checkpoint", {
        "run_id": run_id,
        "iteration": iteration,
        "state": snapshot,
    })

def _extract_final_config(final_state: Any) -> dict[str, Any] | None:
    """Pull system_config out of LangGraph's last-chunk shape.

    graph.astream yields dicts keyed by the node that just wrote, e.g.
        {"feedback_node": {"scores": [...], "system_config": {...}, ...}}
    We want the innermost system_config. Returns None if the shape is
    unexpected (e.g. cancel before any node ran), so the caller can fall
    back to initial_config via the endpoint's COALESCE.
    """
    if not isinstance(final_state, dict):
        return None
    # Two shapes observed: {node_name: state_patch} or a plain state dict.
    for v in final_state.values():
        if isinstance(v, dict) and "system_config" in v:
            cfg = v.get("system_config")
            if isinstance(cfg, dict):
                return cfg
    if "system_config" in final_state and isinstance(final_state["system_config"], dict):
        return final_state["system_config"]
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────


def should_continue(state: AgentState) -> str:
    if state.get("should_stop") or state.get("iteration", 1) > state.get("max_iterations", 3):
        return "end"
    return "research"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("research", research_node)
    graph.add_node("generation", generation_node)
    graph.add_node("verification", verification_node)
    graph.add_node("self_modification", self_modification_node)
    graph.add_node("feedback", feedback_node)

    graph.set_entry_point("orchestrator")

    graph.add_edge("orchestrator", "research")
    graph.add_edge("research", "generation")
    graph.add_edge("generation", "verification")
    graph.add_edge("verification", "self_modification")
    graph.add_edge("self_modification", "feedback")

    graph.add_conditional_edges(
        "feedback",
        should_continue,
        {"research": "research", "end": END},
    )

    return graph.compile()


# Build once at module load
_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


# ─────────────────────────────────────────────────────────────────────────────
# Run entry point
# ─────────────────────────────────────────────────────────────────────────────


async def run_pipeline(
    run_id: str,
    topic: str | None,
    uploaded_paper: str | None,
    max_iterations: int,
    system_config: dict[str, Any],
    tuning: str,
) -> None:
    """Run the full pipeline. All progress flows through emit() to the SSE queue."""
    initial_state: AgentState = {
        "run_id": run_id,
        "topic": topic or "(uploaded paper)",
        "uploaded_paper": uploaded_paper,
        "is_upload": bool(uploaded_paper),
        "system_config": system_config,
        "tuning": tuning,
        "max_iterations": max_iterations,
        "iteration": 0,
        "paper": uploaded_paper or "",
        "paper_versions": [],
        "research_results": [],
        "scores": [],
        "verification_results": [],
        "modification_log": [],
        "should_stop": False,
        "plan": None,
    }

    try:
        # Register the run in Postgres before the graph starts. Idempotent —
        # if the run already exists (e.g. client pre-created it) this is a no-op
        # UPDATE of updated_at. Failures are silenced inside _db_write.
        # day1: also record the starting system_config so before/after diffs
        # across a session are queryable straight from SQL.
        await _db_write(run_id, "POST", "/state/run", {
            "run_id": run_id,
            "topic": topic or "(uploaded paper)",
            "initial_config": initial_state.get("system_config"),
        })

        graph = get_graph()
        final_state = None
        async for chunk in graph.astream(initial_state, {"recursion_limit": 50}):
            final_state = chunk

        # day1: extract the final system_config from whatever node wrote last.
        # graph.astream yields dicts keyed by node name; the config lives inside.
        final_cfg = _extract_final_config(final_state) or initial_state.get("system_config")

        await emit(run_id, "complete", {"final_state": "ok"})
        await _db_write(run_id, "POST", f"/state/run/{run_id}/finalize", {
            "status": "completed",
            "final_score": _latest_score.get(run_id),
            "final_config": final_cfg,
        })
    except PipelineCancelled:
        # Cancellation raised inside emit() — send a direct queue put to avoid
        # hitting check_cancelled again.
        q = get_queue(run_id)
        await q.put(StreamEvent(
            kind="log",
            payload={"agent": "orchestrator", "message": "Cancelled by user", "type": "warn"},
            ts=datetime.now(timezone.utc).isoformat(),
        ))
        # Best-effort finalize. The cancel endpoint already added run_id to
        # _cancelled_runs so emit() raises; we call _db_write directly which
        # uses httpx, not emit(), so it's safe here.
        # day1: final_config may be None if we were cancelled before any node
        # wrote; the endpoint's COALESCE keeps whatever was there.
        await _db_write(run_id, "POST", f"/state/run/{run_id}/finalize", {
            "status": "cancelled",
            "final_score": _latest_score.get(run_id),
            "final_config": _extract_final_config(locals().get("final_state")),
        })
    except Exception as e:
        try:
            await emit(run_id, "error", {"message": str(e)})
        except PipelineCancelled:
            pass
        await _db_write(run_id, "POST", f"/state/run/{run_id}/finalize", {
            "status": "error",
            "final_score": _latest_score.get(run_id),
            "final_config": _extract_final_config(locals().get("final_state")),
        })
    finally:
        # Sentinel to close the SSE stream
        q = get_queue(run_id)
        await q.put(None)
        _cancelled_runs.discard(run_id)
        _latest_score.pop(run_id, None)
