# Self-Improving Research Agent

A multi-agent research paper generator with formal verification, built on
**LangGraph** for orchestration, **Z3** and **Lean 4** for self-modification
verification, and a three-database backing store (**PostgreSQL** +
**MongoDB** + **ChromaDB**).

The React frontend is a thin client that sends user intent to the Python
backend and streams progress back over Server-Sent Events. All agent logic,
LLM calls, metrics, and verification live in the Python process.

---

## Architecture

```
┌─────────────────────────┐
│  React frontend         │
│  (self-improving-       │
│   agent.jsx)            │
│                         │
│  • SSE client           │
│  • UI only, no LLM      │
└──────────┬──────────────┘
           │ POST /pipeline/start
           │ GET  /pipeline/stream/{run_id}
           ▼
┌─────────────────────────┐      ┌───────────────┐
│  FastAPI backend        │─────▶│  Anthropic    │
│  (agent_backend.py)     │      │  Claude API   │
│                         │      └───────────────┘
│  ┌───────────────────┐  │
│  │ LangGraph         │  │      ┌───────────────┐
│  │ (pipeline.py)     │──┼─────▶│  Z3           │
│  │                   │  │      │  (z3-solver)  │
│  │  orchestrator     │  │      └───────────────┘
│  │       ↓           │  │
│  │  research         │  │      ┌───────────────┐
│  │       ↓           │──┼─────▶│  Lean 4       │
│  │  generation       │  │      │  subprocess   │
│  │       ↓           │  │      └───────────────┘
│  │  verification     │  │
│  │       ↓           │  │      ┌───────────────┐
│  │  self_mod  (Z3)   │──┼─────▶│  Postgres     │
│  │       ↓           │  │      │  Mongo        │
│  │  feedback ────────┘  │      │  Chroma       │
│  │    (loop or END)     │      └───────────────┘
│  └───────────────────┘  │
└─────────────────────────┘
```

### The six LangGraph nodes

| Node             | Role                                                            |
|------------------|------------------------------------------------------------------|
| orchestrator     | Parses the topic, builds a plan, decides if it's a technical topic |
| research         | Gathers findings, embeds them into ChromaDB                     |
| generation       | Writes/rewrites the paper; may include code if topic is technical|
| verification     | Runs deterministic metrics + LLM judge + reconciliation         |
| self_modification| Proposes config mutations, verifies with Z3/Lean, applies if safe|
| feedback         | Scores quality, decides whether to loop back to research        |

The graph is defined in `pipeline.py` using `langgraph.graph.StateGraph`
with a conditional edge from `feedback` that either loops back to
`research` or terminates at `END`.

---

## Quick start

```bash
# 1. Start the three databases
docker compose up -d
docker compose ps     # wait for all healthy

# 2. Install Python deps
pip install -r requirements.txt

# Optional: install Lean 4
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
lean --version

# 3. Set the Anthropic API key (used by the backend, not the browser)
export ANTHROPIC_API_KEY=sk-ant-...

# 4. Run the backend
uvicorn agent_backend:app --reload --port 8000
```

Then in the React app:

1. Open the **Self-Modification** tab
2. Paste `http://localhost:8000` into **Verification Backend URL** and click Probe
3. Verify all five status badges are green: Z3, Lean (optional), Postgres, Mongo, Chroma
4. Go back to the main view, enter a topic or upload a paper, and click Run

The backend drives the full pipeline. The React side is just a status display.

---

## Service ports

All DB containers bind to `127.0.0.1` only (no network exposure) on
non-standard host ports to avoid clashing with locally-installed instances:

| Service     | Container port | Host port |
|-------------|----------------|-----------|
| Postgres    | 5432           | **5433**  |
| MongoDB     | 27017          | **27018** |
| ChromaDB    | 8000           | **8001**  |
| FastAPI     | (host process) | **8000**  |

Override via environment variables:

```
ANTHROPIC_API_KEY=sk-ant-...
POSTGRES_URL=postgresql://agent:agent_dev_password@localhost:5433/agent_state
MONGO_URL=mongodb://agent:agent_dev_password@localhost:27018/?authSource=admin
MONGO_DB=agent_logs
CHROMA_HOST=localhost
CHROMA_PORT=8001
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

## Endpoint reference

### Pipeline

```
POST   /pipeline/start              — start a run, returns {run_id, status}
GET    /pipeline/stream/{run_id}    — Server-Sent Events progress stream
POST   /pipeline/cancel/{run_id}    — cooperatively cancel a run
```

**SSE event types** emitted by the pipeline:

| Event          | Payload                                                        |
|----------------|----------------------------------------------------------------|
| `log`          | `{agent, message, type}` — status lines for the UI             |
| `phase`        | `{phase}` — which agent is active right now                    |
| `plan`         | plan dict — title, sections, is_technical_topic, reasoning     |
| `paper_version`| `{iteration, length, preview}` — draft committed               |
| `paper_full`   | `{iteration, paper}` — full markdown text                      |
| `verification` | verification result with metrics + checks + reconciled score  |
| `score`        | feedback result with quality_score + improvements              |
| `modification` | self-mod proposal + verdict + obligations + diffs              |
| `complete`     | final sentinel                                                 |
| `error`        | `{message}`                                                    |
| `end`          | SSE framing — stream closing                                   |

### Verification (standalone, also used internally)

```
POST /verify/z3              — generic JSON constraint DSL
POST /verify/lean            — raw Lean 4 source, subprocess type-check
POST /verify/invariant_set   — one-shot config invariant verification
```

### RAG (ChromaDB)

```
POST   /papers/embed               — upsert {id, text, metadata}
POST   /papers/search              — semantic search
DELETE /papers/collection/{name}   — drop a collection
```

Embeddings generated server-side by `sentence-transformers/all-MiniLM-L6-v2`
(≈80MB, downloaded on first call, CPU-only, free).

### Logs (MongoDB)

```
POST /logs/append                       — append {run_id, iteration, kind, payload}
GET  /logs/{run_id}                     — all log entries for a run
GET  /logs/{run_id}/modifications       — just the modification log
```

### State (PostgreSQL)

```
POST   /state/run                       — register a run
POST   /state/checkpoint                — persist iteration state
GET    /state/checkpoint/{run_id}       — fetch latest checkpoint
GET    /state/runs                      — list all runs
DELETE /state/runs/{run_id}             — cascade-delete run + its Mongo logs
```

### Health

```
GET /health
```

Reports availability + version for Z3, Lean, Postgres, Mongo, Chroma,
Python, plus `pipeline_available` for the LangGraph status.

---

## File layout

```
agent_backend.py       FastAPI unified server
pipeline.py            LangGraph StateGraph, 6 nodes, SSE queue
models.py              Pydantic schemas + AgentState TypedDict
prompts.py             Agent system prompts (Claude)
metrics.py             Deterministic paper metrics + reconciliation
self_modification.py   Heuristic proposal generator + Z3-backed verifier
verification_backend.py  Legacy verification-only server (kept for reference)
self-improving-agent.jsx Thin React frontend (SSE client)
docker-compose.yml     Postgres + Mongo + Chroma stack
Dockerfile             Optional: containerize the backend
requirements.txt       Python deps including langgraph, anthropic, z3-solver, etc.
```

---

## Verification fallback chain

Self-modification verification is tried in this order, per obligation:

1. **Lean 4 server** (`/verify/lean`) — emits a self-contained Lean 4 script
   proving invariants + per-diff monotonicity over `Rat` with `decide`.
2. **Z3 server** — `z3-solver` Python bindings, called directly from
   `self_modification.verify_modification_with_z3`.
3. **Z3 WebAssembly** — browser-side fallback (only used if the frontend
   runs verification locally; not used in the LangGraph path).
4. **Python fallback** — pure-Python invariant checks, used when z3-solver
   isn't installed.

Each obligation in the modification log is tagged with its `backend`
(`lean`, `z3-server`, `z3-wasm`, or `symbolic`) so you can see provenance
at a glance.

---

## What is and isn't actually being proved

Every "PROVED" verdict in the modification log is establishing that the
new configuration's numeric parameters satisfy bounded ranges (e.g.
weights sum to ≈100, penalties in [0,20], a numeric change is within
±50% of its previous value). Both Z3 and Lean reduce these to decidable
arithmetic goals over concrete rationals.

What this **does not** verify:

- That the proposed prompt change actually improves output quality
- That the system as a whole behaves correctly under the new config
- Anything about the LLM's outputs (those remain probabilistic)
- The `SafeArrayAccess` / `PromptModification` / `CodeGeneration`
  behavioural properties sketched in the thesis Verification DSL examples

It corresponds most closely to the thesis Case Study 3 (the rejected
temperature modification). This matches the limitations the thesis
itself lists in §5.2.3 ("Verification Completeness"): the verification
gate is real, the proofs are real, the bounds being checked are
narrower than the high-level discussion implies.

---

## Security

- The Lean endpoint runs `lean` as a subprocess on whatever source you
  POST to it. That's RCE-equivalent on a public network. Keep this on
  `localhost` or behind authentication.
- CORS is wide open (`allow_origins=["*"]`) for development convenience.
  Tighten before exposing the server.
- Compose deliberately binds all DB ports to `127.0.0.1` only and uses
  development-grade passwords. Change both before exposing this stack
  to anything other than your own machine.
- The Anthropic API key now lives in the **backend's** environment, not
  in browser code. This is correct — browser-embedded API keys are
  exfiltration-vulnerable.
