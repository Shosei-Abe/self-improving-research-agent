"""
metrics.py
==========

Deterministic metrics + reconciliation. Direct port of `computeMetrics` and
`reconcileScore` from self-improving-agent.jsx so the Python pipeline gives
identical numbers as the React version did.

The metrics are intentionally regex-based and CPU-only — no LLM calls — so
the verifier's score is reproducible across runs.
"""

from __future__ import annotations

import re
from typing import Any


EXPECTED_SECTIONS = [
    "abstract",
    "introduction",
    "literature review",
    "methodology",
    "analysis",
    "discussion",
    "conclusion",
    "references",
]

CITATION_PATTERN = re.compile(r"\[([A-Z][a-zA-Z\s&,.\-]+,\s*\d{4}[a-z]?)\]")
EQUATION_PATTERN_DOLLAR = re.compile(r"\$[^$]+\$")
EQUATION_PATTERN_PAREN = re.compile(r"\\\([^)]+\\\)")
NUMERIC_CLAIM_PATTERN = re.compile(r"\b\d+(\.\d+)?%")
STATISTICAL_PATTERN = re.compile(
    r"\b(p\s*[<>=]\s*0?\.\d+|p-value|confidence interval|standard deviation"
    r"|correlation|regression|ANOVA|t-test|chi-square|\w+\s*=\s*\d+(\.\d+)?)\b",
    re.IGNORECASE,
)
WEAK_PHRASE_PATTERN = re.compile(
    r"\b(it is well known|in conclusion|it should be noted|it goes without saying"
    r"|needless to say|in today's world|since the dawn of)\b",
    re.IGNORECASE,
)

# day1: code-block stripping so [Mathlib, 2023] or [Smith et al., 2020] inside
# ```python ... ``` fences doesn't inflate the citation count for technical
# topics. Covers triple-backtick (with or without language tag), tilde fences,
# and inline `code` spans. Indented 4-space blocks are intentionally not
# stripped — academic prose legitimately uses indentation for quotes.
_CODE_BLOCK_TRIPLE_BACKTICK = re.compile(r"```[\s\S]*?```")
_CODE_BLOCK_TILDE           = re.compile(r"~~~[\s\S]*?~~~")
_CODE_INLINE_BACKTICK       = re.compile(r"`[^`\n]+`")


def _strip_code_blocks(text: str) -> tuple[str, int]:
    """Remove code spans from `text` so citation regexes only see prose.

    Returns (clean_text, citations_removed_from_code) — the second value is
    the number of [Author, Year]-shaped strings that sat inside a code span
    and were therefore excluded. Useful to expose in metrics so a spike in
    that number signals the LLM is stuffing citations into code examples.
    """
    original_citations_in_code = 0
    clean = text
    for pat in (_CODE_BLOCK_TRIPLE_BACKTICK, _CODE_BLOCK_TILDE, _CODE_INLINE_BACKTICK):
        for block in pat.findall(clean):
            original_citations_in_code += len(CITATION_PATTERN.findall(block))
        clean = pat.sub(" ", clean)  # space, not empty, to preserve word boundaries
    return clean, original_citations_in_code


def _normalize_author(raw: str) -> str:
    """Reduce author strings to a canonical form for unique/orphan checks.

    Handles the common LLM variants for the same paper:
      "Smith et al."       -> "smith"
      "Smith and Jones"    -> "smith"  (pick primary author only)
      "Smith, J."          -> "smith"
      "Smith-Brown"        -> "smith-brown"  (hyphenated surname preserved)
    Non-letters at the end (periods, commas) are stripped. Result is lowercase.
    """
    s = raw.strip().lower()
    # Cut off "et al.", " and ...", ", jr" etc. — everything after a separator
    # that signals multi-author or suffix.
    for sep in (" et al", " and ", " & ", ", jr", ", sr", ","):
        idx = s.find(sep)
        if idx >= 0:
            s = s[:idx]
            break
    # Strip trailing punctuation
    s = s.rstrip(". ")
    return s


def _section_match(present_lower: list[str], required: str) -> bool:    
    for p in present_lower:
        if required in p:
            return True
        if required == "literature review" and ("related work" in p or "background" in p):
            return True
        if required == "methodology" and "method" in p:
            return True
        if required == "analysis" and ("result" in p or "experiment" in p):
            return True
        if required == "references" and "bibliography" in p:
            return True
    return False


def compute_metrics(paper: str, config: dict[str, Any]) -> dict[str, Any]:
    """Compute deterministic metrics for a paper draft.

    `config` is the system_config dict (not the Pydantic model — keeps this
    function callable from anywhere).
    """
    text = paper or ""
    words = [w for w in text.split() if w]
    word_count = len(words)
    lines = text.split("\n")

    # Section detection
    section_headers = [
        line[3:].strip() for line in lines if line.startswith("## ")
    ]
    present_lower = [s.lower() for s in section_headers]
    sections_present = sum(1 for req in EXPECTED_SECTIONS if _section_match(present_lower, req))
    sections_missing = [req for req in EXPECTED_SECTIONS if not _section_match(present_lower, req)]

    # Per-section word counts
    section_word_counts: dict[str, int] = {}
    for i, line in enumerate(lines):
        m = re.match(r"^##\s+(.+)", line)
        if m:
            name = m.group(1).strip()
            w = 0
            for j in range(i + 1, len(lines)):
                if re.match(r"^##\s+", lines[j]):
                    break
                w += len([x for x in lines[j].split() if x])
            section_word_counts[name] = w

    thin_threshold = config["generation"]["minWordsPerSection"]
    thin_sections = [
        f"{k} ({w} words)"
        for k, w in section_word_counts.items()
        if w < thin_threshold
    ]

    # day1: citation detection hardening — strip code spans before running
    # citation regex, and normalize author names for unique/orphan checks.
    # This prevents false positives from [Mathlib, 2023]-style strings inside
    # ```python ... ``` fences, and collapses "Smith et al., 2020" and
    # "Smith, 2020" to the same unique paper.
    clean_text, citations_from_code = _strip_code_blocks(text)

    # Citations (extracted from clean_text; original `text` still used for
    # orphan lookup in references section because refs live outside code).
    citation_matches = CITATION_PATTERN.findall(clean_text)
    citation_count = len(citation_matches)

    # Unique is keyed on (normalized_author, year) so "Smith, 2020" and
    # "Smith et al., 2020" count as the same paper.
    unique_keys: set[tuple[str, str]] = set()
    unique_citation_repr: dict[tuple[str, str], str] = {}  # for orphan reporting
    for raw in citation_matches:
        # raw is e.g. "Smith et al., 2020" or "Smith, 2020a"
        parts = raw.rsplit(",", 1)
        if len(parts) != 2:
            continue
        author_raw, year_raw = parts[0], parts[1].strip()
        key = (_normalize_author(author_raw), year_raw)
        if key[0] and key not in unique_keys:
            unique_keys.add(key)
            unique_citation_repr[key] = raw.strip()
    unique_citation_count = len(unique_keys)

    # Citations per paragraph — paragraphs split from clean_text so
    # code-block paragraphs don't skew the distribution.
    paragraphs = [
        p for p in re.split(r"\n\s*\n", clean_text) if p.strip() and not p.startswith("#")
    ]
    paragraphs_with_citations = sum(1 for p in paragraphs if CITATION_PATTERN.search(p))

    # References — parsed from the ORIGINAL text. Refs section is typically
    # plain prose so code-stripping would be a no-op, but using `text` keeps
    # the behavior predictable even if a paper ever indents a ref entry.
    ref_match = re.search(r"##\s+References[\s\S]*$", text, re.IGNORECASE)
    if not ref_match:
        ref_match = re.search(r"##\s+Bibliography[\s\S]*$", text, re.IGNORECASE)
    references_text = ref_match.group(0) if ref_match else ""

    reference_lines = [
        line for line in references_text.split("\n")
        if line.strip()
        and not re.match(r"^##", line)
        and (
            re.match(r"^\d+\.", line)
            or re.match(r"^\[\d+\]", line)
            or re.match(r"^[-*]\s", line)
            or re.search(r"\d{4}", line)
        )
    ]
    reference_count = len(reference_lines)

    # Orphan citations — a unique citation is "orphan" if its normalized
    # author name doesn't appear anywhere in the references section.
    # Using normalized form avoids "Smith, J." vs "Smith" false orphans.
    refs_lower = references_text.lower()
    orphan_citations: list[str] = []
    for (author_norm, _year), display in unique_citation_repr.items():
        if len(author_norm) > 2 and author_norm not in refs_lower:
            orphan_citations.append(display)
    # Math
    equation_count = (
        len(EQUATION_PATTERN_DOLLAR.findall(text)) + len(EQUATION_PATTERN_PAREN.findall(text))
    )
    numeric_claims = len(NUMERIC_CLAIM_PATTERN.findall(text))
    statistical_terms = len(STATISTICAL_PATTERN.findall(text))
    weak_phrases = len(WEAK_PHRASE_PATTERN.findall(text))

    # Code blocks (informational, not scored)
    code_blocks = text.count("```") // 2

    # Deterministic score using config weights
    W = config["verification"]
    target = config["generation"]["targetTotalWords"]
    base = 0

    if word_count >= target * 1.5:
        base += W["weightWordCount"]
    elif word_count >= target:
        base += int(W["weightWordCount"] * 0.8)
    elif word_count >= target * 0.7:
        base += int(W["weightWordCount"] * 0.55)
    elif word_count >= target * 0.4:
        base += int(W["weightWordCount"] * 0.3)
    else:
        base += int((word_count / target) * W["weightWordCount"] * 0.3)

    base += int((sections_present / 8) * W["weightSections"])

    if unique_citation_count >= 25:
        base += W["weightCitations"]
    elif unique_citation_count >= 15:
        base += int(W["weightCitations"] * 0.72)
    elif unique_citation_count >= 8:
        base += int(W["weightCitations"] * 0.48)
    elif unique_citation_count >= 3:
        base += int(W["weightCitations"] * 0.24)
    else:
        base += int(unique_citation_count * (W["weightCitations"] / 12))

    if reference_count >= 20:
        base += W["weightReferences"]
    elif reference_count >= 10:
        base += int(W["weightReferences"] * 0.67)
    elif reference_count >= 5:
        base += int(W["weightReferences"] * 0.4)
    else:
        base += int(reference_count * (W["weightReferences"] / 12))

    base += min(
        W["weightMathRigor"],
        equation_count * 2 + numeric_claims // 2 + statistical_terms // 2,
    )

    base -= len(thin_sections) * W["penaltyThinSection"]
    base -= len(sections_missing) * W["penaltyMissingSection"]
    base -= len(orphan_citations) * W["penaltyOrphanCitation"]
    base -= weak_phrases * W["penaltyWeakPhrase"]
    base = max(0, min(100, base))

    return {
        "wordCount": word_count,
        "sectionsPresent": sections_present,
        "sectionsTotal": 8,
        "sectionsMissing": sections_missing,
        "sectionWordCounts": section_word_counts,
        "thinSections": thin_sections,
        "thinThreshold": thin_threshold,
        "citationCount": citation_count,
        "uniqueCitationCount": unique_citation_count,
        "paragraphCount": len(paragraphs),
        "paragraphsWithCitations": paragraphs_with_citations,
        "referenceCount": reference_count,
        "orphanCitations": orphan_citations[:5],
        "equationCount": equation_count,
        "numericClaims": numeric_claims,
        "statisticalTerms": statistical_terms,
        "weakPhrases": weak_phrases,
        "codeBlocks": code_blocks,
        "citationsFromCode": citations_from_code,  # day1: citations excluded by code-block stripping
        "deterministicScore": base,
    }


def reconcile_score(llm_score: float, metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Combine LLM-judged score with deterministic floor/ceiling band."""
    det = metrics["deterministicScore"]
    band = config["verification"]["reconciliationBand"]
    floor = max(0, det - band)
    ceiling = min(100, det + band)
    reconciled = max(floor, min(ceiling, llm_score))
    return {
        "final": reconciled,
        "deterministic": det,
        "llm": llm_score,
        "floor": floor,
        "ceiling": ceiling,
        "band": band,
        "adjusted": reconciled != llm_score,
    }


def metrics_to_checks(metrics: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the per-property check list shown in the React UI."""
    target = config["generation"]["targetTotalWords"]
    checks: list[dict[str, Any]] = []

    checks.append({
        "category": "word_count",
        "property": f"length ≥ {target}",
        "status": "PASS" if metrics["wordCount"] >= target else ("WARN" if metrics["wordCount"] >= target * 0.7 else "FAIL"),
        "details": f'{metrics["wordCount"]} words',
    })

    for sec in metrics["sectionsMissing"]:
        checks.append({
            "category": "structure",
            "property": f"section: {sec}",
            "status": "FAIL",
            "details": "missing",
        })

    for ts in metrics["thinSections"]:
        checks.append({
            "category": "structure",
            "property": "section length",
            "status": "WARN",
            "details": f'thin: {ts} (< {metrics["thinThreshold"]} words)',
        })

    checks.append({
        "category": "citations",
        "property": "unique citations ≥ 8",
        "status": "PASS" if metrics["uniqueCitationCount"] >= 8 else ("WARN" if metrics["uniqueCitationCount"] >= 3 else "FAIL"),
        "details": f'{metrics["uniqueCitationCount"]} unique',
    })

    if metrics["orphanCitations"]:
        checks.append({
            "category": "citations",
            "property": "no orphan citations",
            "status": "FAIL",
            "details": f'{len(metrics["orphanCitations"])} orphan(s)',
        })

    checks.append({
        "category": "references",
        "property": "references ≥ 10",
        "status": "PASS" if metrics["referenceCount"] >= 10 else ("WARN" if metrics["referenceCount"] >= 5 else "FAIL"),
        "details": f'{metrics["referenceCount"]} entries',
    })

    if metrics["weakPhrases"] > 0:
        checks.append({
            "category": "style",
            "property": "no weak phrases",
            "status": "WARN",
            "details": f'{metrics["weakPhrases"]} found',
        })

    return checks
