"""
agent_backend.py
================

Unified FastAPI backend for the self-improving research agent.

This file extends `verification_backend.py` with three databases:

  - PostgreSQL  → run state + LangGraph-style checkpoints (relational, durable)
  - MongoDB     → modification log, verification results, agent traces (flexible JSON)
  - ChromaDB    → vector store for the Research Agent's RAG corpus
                  (embeddings via sentence-transformers/all-MiniLM-L6-v2, local)

Why three databases?
--------------------
The thesis (Designing a Self-Improving Research Agent) describes three
distinct kinds of state:

  1. Structured run state — iteration counters, checkpoints, rollback points.
     Relational with strict schema. → Postgres.

  2. Heterogeneous logs — modification proposals, verifier obligations,
     proof traces, per-agent debug output. Schema evolves every iteration.
     → MongoDB.

  3. Embedded research corpus — papers, snippets, retrieved documents,
     queried by semantic similarity. → ChromaDB.

The React app talks to all three through this single FastAPI service.
Browsers can't open Mongo/Postgres connections directly (no driver, no
authentication story, CORS-hostile), so the backend is the only place
that touches the databases.

Endpoints
---------
GET    /health                       — service + DB + Z3 + Lean status
POST   /verify/z3                    — (unchanged) generic Z3 DSL
POST   /verify/lean                  — (unchanged) Lean 4 type-check
POST   /verify/invariant_set         — (unchanged) one-shot invariant verification

POST   /papers/embed                 — add documents to the RAG store
POST   /papers/search                — semantic search over the RAG store
DELETE /papers/collection/{name}     — drop a collection

POST   /logs/append                  — append an entry to a run's log
GET    /logs/{run_id}                — fetch all log entries for a run
GET    /logs/{run_id}/modifications  — fetch only modification log entries

POST   /state/checkpoint             — write a checkpoint for a run
GET    /state/checkpoint/{run_id}    — fetch the latest checkpoint for a run
GET    /state/runs                   — list all runs
DELETE /state/runs/{run_id}          — delete a run + cascade its logs

Run with Docker Compose
-----------------------
    docker compose up

This starts Postgres, Mongo, ChromaDB, and this backend together.
The first run downloads the sentence-transformers model into a named
volume and caches it for subsequent runs.

Run standalone (without Docker)
-------------------------------
    pip install -r requirements.txt
    # plus running Postgres on :5432, Mongo on :27017, Chroma on :8001
    uvicorn agent_backend:app --port 8000

Environment variables
---------------------
    POSTGRES_URL    default: postgresql://agent:agent@localhost:5432/agent
    MONGO_URL       default: mongodb://localhost:27017
    MONGO_DB        default: agent
    CHROMA_HOST     default: localhost
    CHROMA_PORT     default: 8001
    EMBED_MODEL     default: sentence-transformers/all-MiniLM-L6-v2

The DB clients are created lazily and tolerate missing services so the
verification endpoints stay usable even if no databases are running.
The /health endpoint reports each DB's individual status.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# Z3 (unchanged from verification_backend.py)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import z3  # type: ignore

    Z3_AVAILABLE = True
    Z3_VERSION = z3.get_version_string()
except Exception as _e:  # pragma: no cover
    Z3_AVAILABLE = False
    Z3_VERSION = f"unavailable: {_e}"


# ─────────────────────────────────────────────────────────────────────────────
# Database clients — created lazily, all optional
# ─────────────────────────────────────────────────────────────────────────────

POSTGRES_URL = os.environ.get(
    "POSTGRES_URL",
    "postgresql://agent:agent_dev_password@localhost:5434/agent_state",
)
MONGO_URL = os.environ.get(
    "MONGO_URL",
    "mongodb://agent:agent_dev_password@localhost:27018/?authSource=admin",
)
MONGO_DB_NAME = os.environ.get("MONGO_DB", "agent_logs")
CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8001"))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


# Postgres (psycopg3, sync — wrapped in run_in_executor for async endpoints)
_pg_pool: Any = None
_pg_error: str | None = None


def get_pg():
    """Get or initialize the Postgres connection pool."""
    global _pg_pool, _pg_error
    if _pg_pool is not None:
        return _pg_pool
    if _pg_error is not None:
        return None
    try:
        from psycopg_pool import ConnectionPool

        _pg_pool = ConnectionPool(POSTGRES_URL, min_size=1, max_size=4, open=True)
        _ensure_pg_schema(_pg_pool)
        return _pg_pool
    except Exception as e:
        _pg_error = str(e)
        return None


def _ensure_pg_schema(pool: Any) -> None:
    """Idempotent schema migration. Two tables: runs + checkpoints."""
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id        TEXT PRIMARY KEY,
                    topic         TEXT NOT NULL,
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    iterations    INTEGER NOT NULL DEFAULT 0,
                    final_score   REAL,
                    status        TEXT NOT NULL DEFAULT 'running'
                );

                CREATE TABLE IF NOT EXISTS checkpoints (
                    id            BIGSERIAL PRIMARY KEY,
                    run_id        TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    iteration     INTEGER NOT NULL,
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    state_json    JSONB NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_checkpoints_run_iter
                    ON checkpoints (run_id, iteration DESC);
                """
            )
        conn.commit()


# Mongo
_mongo_client: Any = None
_mongo_error: str | None = None


def get_mongo():
    global _mongo_client, _mongo_error
    if _mongo_client is not None:
        return _mongo_client
    if _mongo_error is not None:
        return None
    try:
        from pymongo import MongoClient

        client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=2000)
        # Trigger a connection so we fail fast if Mongo isn't there
        client.admin.command("ping")
        _mongo_client = client
        return _mongo_client
    except Exception as e:
        _mongo_error = str(e)
        return None


def get_mongo_db():
    client = get_mongo()
    if client is None:
        return None
    return client[MONGO_DB_NAME]


# ChromaDB + sentence-transformers
_chroma_client: Any = None
_chroma_error: str | None = None
_embed_fn: Any = None
_embed_error: str | None = None


def get_chroma():
    global _chroma_client, _chroma_error
    if _chroma_client is not None:
        return _chroma_client
    if _chroma_error is not None:
        return None
    try:
        import chromadb

        _chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        # Quick ping
        _chroma_client.heartbeat()
        return _chroma_client
    except Exception as e:
        _chroma_error = str(e)
        return None


def get_embed_fn():
    """Lazy-load the sentence-transformers model. CPU is fine for MiniLM."""
    global _embed_fn, _embed_error
    if _embed_fn is not None:
        return _embed_fn
    if _embed_error is not None:
        return None
    try:
        from chromadb.utils import embedding_functions

        _embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        return _embed_fn
    except Exception as e:
        _embed_error = str(e)
        return None


def get_or_create_collection(name: str):
    client = get_chroma()
    if client is None:
        return None
    embed_fn = get_embed_fn()
    if embed_fn is None:
        return None
    return client.get_or_create_collection(name=name, embedding_function=embed_fn)


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan: pre-warm clients + report status at startup
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm in background — don't block startup if a DB is offline
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, get_pg)
    await loop.run_in_executor(None, get_mongo)
    await loop.run_in_executor(None, get_chroma)
    # Don't pre-warm the embed model — it's large and slow on first load.
    # Let it lazy-load on the first /papers/embed call.
    yield
    # Shutdown
    if _pg_pool is not None:
        try:
            _pg_pool.close()
        except Exception:
            pass
    if _mongo_client is not None:
        try:
            _mongo_client.close()
        except Exception:
            pass


app = FastAPI(
    title="Self-Improving Agent Backend",
    description="Verification (Z3 + Lean 4) + RAG (Chroma) + Logs (Mongo) + State (Postgres)",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Lean detection
# ─────────────────────────────────────────────────────────────────────────────

LEAN_BIN = shutil.which("lean")


def _lean_version() -> str:
    if LEAN_BIN is None:
        return "unavailable: `lean` not on PATH"
    try:
        out = subprocess.run([LEAN_BIN, "--version"], capture_output=True, text=True, timeout=5)
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
    pg_status: dict[str, Any]
    if get_pg() is not None:
        pg_status = {"available": True, "url": POSTGRES_URL.replace(POSTGRES_URL.split("@")[0].split("//")[1], "***")}
    else:
        pg_status = {"available": False, "error": _pg_error}

    mongo_status: dict[str, Any]
    if get_mongo() is not None:
        mongo_status = {"available": True, "url": MONGO_URL, "db": MONGO_DB_NAME}
    else:
        mongo_status = {"available": False, "error": _mongo_error}

    chroma_status: dict[str, Any]
    chroma = get_chroma()
    if chroma is not None:
        try:
            collections = [c.name for c in chroma.list_collections()]
        except Exception:
            collections = []
        chroma_status = {
            "available": True,
            "host": CHROMA_HOST,
            "port": CHROMA_PORT,
            "collections": collections,
            "embed_model": EMBED_MODEL,
            "embed_loaded": _embed_fn is not None,
        }
    else:
        chroma_status = {"available": False, "error": _chroma_error}

    return {
        "ok": True,
        "z3": {"available": Z3_AVAILABLE, "version": Z3_VERSION},
        "lean": {"available": LEAN_AVAILABLE, "version": LEAN_VERSION, "binary": LEAN_BIN},
        "postgres": pg_status,
        "mongo": mongo_status,
        "chroma": chroma_status,
        "python": sys.version.split()[0],
    }


# ═════════════════════════════════════════════════════════════════════════════
# VERIFICATION ENDPOINTS (unchanged from verification_backend.py)
# ═════════════════════════════════════════════════════════════════════════════


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
        return {"eq": a == b, "ne": a != b, "lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b}[op]

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

    env: dict[str, Any] = {}
    for decl in req.declarations:
        if decl.sort == "Int":
            env[decl.name] = z3.Int(decl.name)
        elif decl.sort == "Bool":
            env[decl.name] = z3.Bool(decl.name)
        else:
            env[decl.name] = z3.Real(decl.name)

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
            results.append(Z3GoalResult(name=goal.name, formal=goal.formal, status="PROVED", elapsed_ms=elapsed))
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
        raise HTTPException(503, f"Lean 4 unavailable on this server: {LEAN_VERSION}")

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
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=req.timeout_seconds)
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

        if rc == 0 and "error:" not in stdout.lower() and "error:" not in stderr.lower():
            status: Literal["PROVED", "REFUTED", "ERROR", "TIMEOUT"] = "PROVED"
        elif "sorry" in stdout.lower() or "sorry" in stderr.lower():
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


# ─── Convenience: full invariant set verification ────────────────────────────


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

    declarations = [Z3VarDecl(name=k, sort="Real") for k in flat.keys()]
    assumptions: list[dict[str, Any]] = [
        {"op": "eq", "args": [{"op": "var", "name": k}, {"op": "const", "value": v}]}
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
                expr={"op": "and", "args": [{"op": "ge", "args": [var(k), num(0)]} for k in weight_keys]},
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

    return verify_z3(Z3Request(declarations=declarations, assumptions=assumptions, goals=goals))


# ═════════════════════════════════════════════════════════════════════════════
# CHROMADB — RAG vector store
# ═════════════════════════════════════════════════════════════════════════════
#
# The Research Agent embeds retrieved documents into Chroma so that future
# queries can retrieve semantically similar passages instead of re-fetching
# them. Embeddings come from sentence-transformers/all-MiniLM-L6-v2 (local,
# 80MB model, CPU-friendly).


class EmbedDocument(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbedRequest(BaseModel):
    collection: str = "papers"
    documents: list[EmbedDocument]


class EmbedResponse(BaseModel):
    collection: str
    added: int
    total_in_collection: int


@app.post("/papers/embed", response_model=EmbedResponse)
def papers_embed(req: EmbedRequest) -> EmbedResponse:
    coll = get_or_create_collection(req.collection)
    if coll is None:
        raise HTTPException(
            503,
            f"Chroma unavailable: {_chroma_error or _embed_error or 'unknown'}",
        )
    if not req.documents:
        raise HTTPException(400, "no documents provided")

    ids = [d.id for d in req.documents]
    texts = [d.text for d in req.documents]
    metadatas = [
        # Chroma requires non-empty primitive metadata; coerce empties.
        d.metadata or {"_": ""}
        for d in req.documents
    ]

    try:
        coll.upsert(ids=ids, documents=texts, metadatas=metadatas)
    except Exception as e:
        raise HTTPException(500, f"chroma upsert failed: {e}")

    return EmbedResponse(
        collection=req.collection,
        added=len(req.documents),
        total_in_collection=coll.count(),
    )


class SearchRequest(BaseModel):
    collection: str = "papers"
    query: str
    n_results: int = 5
    where: dict[str, Any] | None = None  # metadata filter


class SearchHit(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float


class SearchResponse(BaseModel):
    collection: str
    query: str
    hits: list[SearchHit]


@app.post("/papers/search", response_model=SearchResponse)
def papers_search(req: SearchRequest) -> SearchResponse:
    coll = get_or_create_collection(req.collection)
    if coll is None:
        raise HTTPException(
            503,
            f"Chroma unavailable: {_chroma_error or _embed_error or 'unknown'}",
        )
    try:
        result = coll.query(
            query_texts=[req.query],
            n_results=req.n_results,
            where=req.where,
        )
    except Exception as e:
        raise HTTPException(500, f"chroma query failed: {e}")

    hits: list[SearchHit] = []
    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]
    for i, doc, meta, dist in zip(ids, docs, metas, dists):
        hits.append(
            SearchHit(
                id=i,
                text=doc or "",
                metadata=meta or {},
                distance=float(dist) if dist is not None else 0.0,
            )
        )

    return SearchResponse(collection=req.collection, query=req.query, hits=hits)


@app.delete("/papers/collection/{name}")
def papers_drop_collection(name: str) -> dict[str, Any]:
    client = get_chroma()
    if client is None:
        raise HTTPException(503, f"Chroma unavailable: {_chroma_error}")
    try:
        client.delete_collection(name=name)
    except Exception as e:
        raise HTTPException(404, f"could not delete collection: {e}")
    return {"deleted": name}


# ═════════════════════════════════════════════════════════════════════════════
# MONGODB — flexible JSON logs
# ═════════════════════════════════════════════════════════════════════════════
#
# Two collections:
#   - logs           : every agent action, proof obligation, error, etc.
#   - modifications  : the dedicated modification log (subset of logs but
#                      stored separately for fast querying / replay).
#
# Schema is intentionally loose — every entry has run_id, iteration, ts,
# kind. Anything else is up to the writer.


class LogEntry(BaseModel):
    run_id: str
    iteration: int
    kind: str  # "agent_step" | "verification" | "modification" | "error" | etc.
    payload: dict[str, Any]
    ts: str | None = None  # ISO timestamp; server fills if absent


@app.post("/logs/append")
def logs_append(entry: LogEntry) -> dict[str, Any]:
    db = get_mongo_db()
    if db is None:
        raise HTTPException(503, f"Mongo unavailable: {_mongo_error}")

    doc = entry.model_dump()
    if not doc.get("ts"):
        doc["ts"] = datetime.now(timezone.utc).isoformat()

    db.logs.insert_one(doc)
    if entry.kind == "modification":
        db.modifications.insert_one(doc)
    # Strip Mongo's injected _id from the response (not JSON-serializable)
    doc.pop("_id", None)
    return {"ok": True, "stored": doc}


@app.get("/logs/{run_id}")
def logs_fetch(run_id: str, limit: int = 500) -> dict[str, Any]:
    db = get_mongo_db()
    if db is None:
        raise HTTPException(503, f"Mongo unavailable: {_mongo_error}")
    cursor = db.logs.find({"run_id": run_id}).sort("ts", 1).limit(limit)
    entries = []
    for d in cursor:
        d.pop("_id", None)
        entries.append(d)
    return {"run_id": run_id, "count": len(entries), "entries": entries}


@app.get("/logs/{run_id}/modifications")
def logs_modifications(run_id: str) -> dict[str, Any]:
    db = get_mongo_db()
    if db is None:
        raise HTTPException(503, f"Mongo unavailable: {_mongo_error}")
    cursor = db.modifications.find({"run_id": run_id}).sort("iteration", 1)
    entries = []
    for d in cursor:
        d.pop("_id", None)
        entries.append(d)
    return {"run_id": run_id, "count": len(entries), "entries": entries}


# ═════════════════════════════════════════════════════════════════════════════
# POSTGRES — runs + checkpoints
# ═════════════════════════════════════════════════════════════════════════════


class RunCreate(BaseModel):
    run_id: str
    topic: str
    initial_config: dict[str, Any] | None = None  # day1: record starting system_config


class CheckpointWrite(BaseModel):
    run_id: str
    iteration: int
    state: dict[str, Any]


class RunFinalize(BaseModel):
    status: str  # 'completed' | 'cancelled' | 'error'
    final_score: float | None = None
    final_config: dict[str, Any] | None = None  # day1: record ending system_config


@app.post("/state/run")
def state_create_run(req: RunCreate) -> dict[str, Any]:
    import json
    pool = get_pg()
    if pool is None:
        raise HTTPException(503, f"Postgres unavailable: {_pg_error}")

    # day1: serialize initial_config to JSONB; preserve on conflict so
    # a re-register never clobbers the originally recorded config.
    init_cfg_json = json.dumps(req.initial_config) if req.initial_config is not None else None

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO runs (run_id, topic, initial_config)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (run_id) DO UPDATE
                    SET updated_at = NOW(),
                        initial_config = COALESCE(runs.initial_config, EXCLUDED.initial_config)
                RETURNING run_id, topic, created_at::text, updated_at::text, iterations, status
                """,
                (req.run_id, req.topic, init_cfg_json),
            )
            row = cur.fetchone()
        conn.commit()

    return {
        "run_id": row[0],
        "topic": row[1],
        "created_at": row[2],
        "updated_at": row[3],
        "iterations": row[4],
        "status": row[5],
    }


@app.post("/state/checkpoint")
def state_checkpoint(req: CheckpointWrite) -> dict[str, Any]:
    import json

    pool = get_pg()
    if pool is None:
        raise HTTPException(503, f"Postgres unavailable: {_pg_error}")

    with pool.connection() as conn:
        with conn.cursor() as cur:
            # Make sure the run exists (auto-create on first checkpoint)
            cur.execute(
                """
                INSERT INTO runs (run_id, topic) VALUES (%s, %s)
                ON CONFLICT (run_id) DO NOTHING
                """,
                (req.run_id, req.state.get("topic", "(unknown)")),
            )
            cur.execute(
                """
                INSERT INTO checkpoints (run_id, iteration, state_json)
                VALUES (%s, %s, %s::jsonb)
                RETURNING id, created_at::text
                """,
                (req.run_id, req.iteration, json.dumps(req.state)),
            )
            row = cur.fetchone()
            cur.execute(
                """
                UPDATE runs
                SET iterations = GREATEST(iterations, %s), updated_at = NOW()
                WHERE run_id = %s
                """,
                (req.iteration, req.run_id),
            )
        conn.commit()

    return {"id": row[0], "created_at": row[1], "run_id": req.run_id, "iteration": req.iteration}


@app.get("/state/checkpoint/{run_id}")
def state_get_latest_checkpoint(run_id: str) -> dict[str, Any]:
    pool = get_pg()
    if pool is None:
        raise HTTPException(503, f"Postgres unavailable: {_pg_error}")

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, iteration, created_at::text, state_json
                FROM checkpoints
                WHERE run_id = %s
                ORDER BY iteration DESC, id DESC
                LIMIT 1
                """,
                (run_id,),
            )
            row = cur.fetchone()

    if not row:
        raise HTTPException(404, f"no checkpoint for run {run_id}")
    return {
        "run_id": run_id,
        "checkpoint_id": row[0],
        "iteration": row[1],
        "created_at": row[2],
        "state": row[3],
    }


@app.get("/state/runs")
def state_list_runs(limit: int = 50) -> dict[str, Any]:
    pool = get_pg()
    if pool is None:
        raise HTTPException(503, f"Postgres unavailable: {_pg_error}")

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.run_id, r.topic, r.created_at::text, r.updated_at::text,
                       r.iterations, r.status, r.final_score,
                       (SELECT COUNT(*) FROM checkpoints c WHERE c.run_id = r.run_id) AS checkpoint_count
                FROM runs r
                ORDER BY r.updated_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()

    return {
        "runs": [
            {
                "run_id": r[0],
                "topic": r[1],
                "created_at": r[2],
                "updated_at": r[3],
                "iterations": r[4],
                "status": r[5],
                "final_score": r[6],
                "checkpoint_count": r[7],
            }
            for r in rows
        ]
    }


@app.post("/state/run/{run_id}/finalize")
def state_finalize_run(run_id: str, req: RunFinalize) -> dict[str, Any]:
    """Mark a run as completed/cancelled/error and optionally record final_score.

    Called by pipeline.py when the LangGraph execution finishes (success or failure).
    Idempotent: if the run doesn't exist, returns 404. If it's already in the
    requested state, re-runs the UPDATE (no harm, updated_at advances).

    day1: now also records final_config so the run-level before/after diff is
    queryable (e.g. to detect config drift via carry-over). Only overwrites
    when the caller actually provides one — a partial finalize with just a
    status change will keep the previously stored final_config intact.
    """
    import json
    pool = get_pg()
    if pool is None:
        raise HTTPException(503, f"Postgres unavailable: {_pg_error}")

    allowed = {"completed", "cancelled", "error", "running"}
    if req.status not in allowed:
        raise HTTPException(400, f"status must be one of {sorted(allowed)}")

    final_cfg_json = json.dumps(req.final_config) if req.final_config is not None else None

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE runs
                SET status = %s,
                    final_score = COALESCE(%s, final_score),
                    final_config = COALESCE(%s::jsonb, final_config),
                    updated_at = NOW()
                WHERE run_id = %s
                RETURNING run_id, status, final_score, iterations, updated_at::text
                """,
                (req.status, req.final_score, final_cfg_json, run_id),
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(404, f"no run with id {run_id}")
    return {
        "run_id": row[0],
        "status": row[1],
        "final_score": row[2],
        "iterations": row[3],
        "updated_at": row[4],
    }

@app.delete("/state/runs/{run_id}")
def state_delete_run(run_id: str) -> dict[str, Any]:
    pool = get_pg()
    if pool is None:
        raise HTTPException(503, f"Postgres unavailable: {_pg_error}")

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM runs WHERE run_id = %s", (run_id,))
            deleted = cur.rowcount
        conn.commit()

    # Cascade-delete the Mongo logs too if Mongo is available
    db = get_mongo_db()
    if db is not None:
        db.logs.delete_many({"run_id": run_id})
        db.modifications.delete_many({"run_id": run_id})

    return {"deleted_run": run_id, "rows_deleted": deleted}


# ═════════════════════════════════════════════════════════════════════════════
# LANGGRAPH PIPELINE — Python implementation of the multi-agent loop
# ═════════════════════════════════════════════════════════════════════════════
#
# The pipeline lives in pipeline.py. We expose three endpoints:
#
#   POST /pipeline/start          — start a run, returns run_id immediately
#   GET  /pipeline/stream/{run_id} — Server-Sent Events stream of progress
#   GET  /pipeline/state/{run_id} — current state snapshot (for late joiners)
#
# Pipeline launching is fire-and-forget: the run executes in a background task,
# and progress flows through an asyncio.Queue that the SSE endpoint drains.

import uuid as _uuid

try:
    from pipeline import run_pipeline, get_queue, get_graph, cancel_run  # noqa: F401

    _PIPELINE_AVAILABLE = True
    _PIPELINE_ERROR: str | None = None
    # Trigger lazy graph build at import time so any langgraph errors surface early
    try:
        get_graph()
    except Exception as _e:
        _PIPELINE_AVAILABLE = False
        _PIPELINE_ERROR = f"langgraph build failed: {_e}"
except Exception as _e:
    _PIPELINE_AVAILABLE = False
    _PIPELINE_ERROR = str(_e)
    run_pipeline = None  # type: ignore
    get_queue = None  # type: ignore
    cancel_run = None  # type: ignore


class PipelineStartReq(BaseModel):
    topic: str | None = None
    uploaded_paper: str | None = None
    max_iterations: int = 3
    system_config: dict[str, Any] = Field(default_factory=dict)
    tuning: str = ""


class PipelineStartResp(BaseModel):
    run_id: str
    status: str


@app.post("/pipeline/start", response_model=PipelineStartResp)
async def pipeline_start(req: PipelineStartReq) -> PipelineStartResp:
    if not _PIPELINE_AVAILABLE:
        raise HTTPException(503, f"pipeline unavailable: {_PIPELINE_ERROR}")

    if not req.topic and not req.uploaded_paper:
        raise HTTPException(400, "either topic or uploaded_paper is required")

    run_id = f"run_{int(time.time())}_{_uuid.uuid4().hex[:8]}"

    # Default system_config if not provided
    cfg = req.system_config or {
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

    # Register the run in Postgres so the Runs & Data tab sees it
    pool = get_pg()
    if pool is not None:
        try:
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO runs (run_id, topic) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                        (run_id, req.topic or "(uploaded)"),
                    )
                conn.commit()
        except Exception:
            pass

    # Fire-and-forget background task
    asyncio.create_task(
        run_pipeline(  # type: ignore[misc]
            run_id=run_id,
            topic=req.topic,
            uploaded_paper=req.uploaded_paper,
            max_iterations=req.max_iterations,
            system_config=cfg,
            tuning=req.tuning,
        )
    )

    return PipelineStartResp(run_id=run_id, status="started")


@app.get("/pipeline/stream/{run_id}")
async def pipeline_stream(run_id: str):
    """Server-Sent Events stream of pipeline progress."""
    from fastapi.responses import StreamingResponse
    import json as _json

    if not _PIPELINE_AVAILABLE:
        raise HTTPException(503, f"pipeline unavailable: {_PIPELINE_ERROR}")

    async def event_gen():
        q = get_queue(run_id)  # type: ignore[misc]
        while True:
            event = await q.get()
            if event is None:
                yield f"event: end\ndata: {{}}\n\n"
                break
            payload = event.model_dump_json() if hasattr(event, "model_dump_json") else _json.dumps(event)
            yield f"event: {event.kind}\ndata: {payload}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/pipeline/cancel/{run_id}")
def pipeline_cancel(run_id: str) -> dict[str, Any]:
    """Cooperatively cancel a running pipeline. Returns immediately; the
    pipeline checks the cancelled flag at its next emit() call and raises
    PipelineCancelled from there, unwinding cleanly."""
    if not _PIPELINE_AVAILABLE:
        raise HTTPException(503, f"pipeline unavailable: {_PIPELINE_ERROR}")
    cancelled = cancel_run(run_id)  # type: ignore[misc]
    return {"run_id": run_id, "cancelled": cancelled}


# ─────────────────────────────────────────────────────────────────────────────
# Root
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "self-improving-agent backend",
        "version": "2.1.0",
        "endpoints": {
            "verification": ["/verify/z3", "/verify/lean", "/verify/invariant_set"],
            "rag": ["/papers/embed", "/papers/search", "/papers/collection/{name}"],
            "logs": ["/logs/append", "/logs/{run_id}", "/logs/{run_id}/modifications"],
            "state": [
                "/state/run",
                "/state/run/{run_id}/finalize",
                "/state/checkpoint",
                "/state/checkpoint/{run_id}",
                "/state/runs",
                "/state/runs/{run_id}",
            ],
            "pipeline": ["/pipeline/start", "/pipeline/stream/{run_id}", "/pipeline/cancel/{run_id}"],
            "health": ["/health"],
        },
        "pipeline_available": _PIPELINE_AVAILABLE,
        "pipeline_error": _PIPELINE_ERROR,
    }
