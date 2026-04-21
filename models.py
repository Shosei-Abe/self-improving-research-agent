"""
models.py
=========

Pydantic schemas + the LangGraph TypedDict state for the pipeline.

The state mirrors what the React app used to maintain in useState hooks,
moved to the server so LangGraph can checkpoint it to Postgres and so
the verification + self-modification steps can use the real Z3/Lean
backends without an extra HTTP hop.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# Run configuration (request body for /pipeline/start)
# ─────────────────────────────────────────────────────────────────────────────


class GenerationConfig(BaseModel):
    minWordsPerSection: int = 150
    targetTotalWords: int = 3000
    citationsPerParagraph: int = 2
    requireQuantitativeData: bool = True
    paragraphsPerSection: int = 3


class VerificationConfig(BaseModel):
    weightWordCount: int = 25
    weightSections: int = 24
    weightCitations: int = 25
    weightReferences: int = 15
    weightMathRigor: int = 10
    penaltyThinSection: int = 3
    penaltyMissingSection: int = 4
    penaltyOrphanCitation: int = 1
    penaltyWeakPhrase: int = 1
    reconciliationBand: int = 8


class ImprovementConfig(BaseModel):
    targetIncrement: int = 10
    aggressiveness: float = 0.5
    focusOnFailures: bool = True


class SystemConfig(BaseModel):
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    improvement: ImprovementConfig = Field(default_factory=ImprovementConfig)
    schema_version: int = Field(default=1, alias="__version")

    class Config:
        populate_by_name = True


class PipelineStartRequest(BaseModel):
    topic: str | None = None
    uploaded_paper: str | None = None  # raw text from a PDF/MD/TeX upload
    max_iterations: int = 3
    system_config: SystemConfig = Field(default_factory=SystemConfig)
    tuning: str = ""  # extra prompt instructions


class PipelineStartResponse(BaseModel):
    run_id: str
    status: Literal["started"]


# ─────────────────────────────────────────────────────────────────────────────
# Per-iteration data
# ─────────────────────────────────────────────────────────────────────────────


class VerificationCheck(BaseModel):
    category: str
    property: str
    status: Literal["PASS", "FAIL", "WARN"]
    details: str


class VerificationMetrics(BaseModel):
    word_count: int
    section_count: int
    citations: int
    references: int
    has_quantitative_data: bool
    avg_section_length: float
    code_blocks: int = 0


class VerificationResult(BaseModel):
    iteration: int
    score: float
    overall_status: Literal["PASS", "FAIL", "WARN"]
    metrics: VerificationMetrics
    checks: list[VerificationCheck]


class IterationScore(BaseModel):
    iteration: int
    quality_score: float
    improvements: list[str] = Field(default_factory=list)
    iteration_summary: str = ""
    should_continue: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Self-modification
# ─────────────────────────────────────────────────────────────────────────────


class ModificationProposal(BaseModel):
    target_path: str  # e.g. "verification.weightCitations"
    old_value: Any
    new_value: Any
    reasoning: str
    expectedEffect: str


class ProofObligation(BaseModel):
    phase: Literal["INVARIANT", "DIFF_ANALYSIS", "MONOTONICITY", "PROGRESS", "REGRESSION", "LEAN4"]
    name: str
    formal: str | None = None
    status: Literal["PROVED", "REFUTED", "WARNING", "SKIPPED"]
    counterexample: str | None = None
    details: str | None = None
    backend: Literal["lean", "z3-server", "z3-wasm", "symbolic"] = "symbolic"
    elapsed_ms: float | None = None


class VerificationVerdict(BaseModel):
    verdict: Literal["APPROVED", "REJECTED"]
    obligations: list[ProofObligation]
    diffs: list[dict[str, Any]] = Field(default_factory=list)
    proof_chain: str
    backends_used: list[str] = Field(default_factory=list)


class ModificationLogEntry(BaseModel):
    iteration: int
    proposal: ModificationProposal
    verdict: VerificationVerdict
    applied: bool
    timestamp: str
    before_config: dict[str, Any]
    after_config: dict[str, Any] | None = None


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph state — single TypedDict that flows through every node
# ─────────────────────────────────────────────────────────────────────────────


class AgentState(TypedDict, total=False):
    # Identity
    run_id: str
    topic: str
    uploaded_paper: str | None
    is_upload: bool

    # Configuration
    system_config: dict[str, Any]
    tuning: str
    max_iterations: int

    # Plan (set once, used by every iteration)
    plan: dict[str, Any] | None

    # Per-iteration accumulators
    iteration: int
    paper: str
    paper_versions: list[str]
    research_results: list[dict[str, Any]]
    scores: list[dict[str, Any]]
    verification_results: list[dict[str, Any]]
    modification_log: list[dict[str, Any]]

    # Control flow
    should_stop: bool
    final_score: float | None
    error: str | None


# ─────────────────────────────────────────────────────────────────────────────
# SSE event format
# ─────────────────────────────────────────────────────────────────────────────


class StreamEvent(BaseModel):
    """Sent to the React app over Server-Sent Events."""

    kind: Literal[
        "log",
        "phase",
        "iteration",
        "paper_version", "paper_full",
        "score",
        "verification",
        "modification",
        "plan",
        "complete",
        "error",
    ]
    payload: dict[str, Any]
    ts: str
