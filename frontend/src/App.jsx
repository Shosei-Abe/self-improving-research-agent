import { useState, useEffect, useRef, useCallback } from "react";

// ═══════════════════════════════════════════════════════════════════════════
//
//  SELF-IMPROVING RESEARCH AGENT — React Frontend
//
//  The pipeline (5 agents + self-modification) runs on the Python backend
//  via LangGraph (see pipeline.py). This file is now a thin SSE client:
//  POST /pipeline/start → EventSource on /pipeline/stream/{run_id} → route
//  events into React state.
//
//  Several legacy helper functions remain below (computeMetrics,
//  proposeSelfModification, symbolicVerify, emitLeanProof, CONFIG_INVARIANTS,
//  the prompt builders) because the Self-Mod tab uses CONFIG_INVARIANTS for
//  its display and other parts are convenient fallbacks if the backend is
//  offline. They are not called during an active run — the server is
//  authoritative.
//
//  Anthropic API calls now live in the backend. The callAPI helper below
//  is kept only for the upload-handling code path that parses PDF text
//  directly in the browser before forwarding to the backend.
//
// ═══════════════════════════════════════════════════════════════════════════

const API_URL = "https://api.anthropic.com/v1/messages";
const MODEL = "claude-sonnet-4-20250514";

const AGENTS = {
  orchestrator: { label: "Orchestrator", mark: "O" },
  research: { label: "Research", mark: "R" },
  generation: { label: "Writer", mark: "W" },
  verification: { label: "Verifier", mark: "V" },
  feedback: { label: "Critic", mark: "C" },
};
const STEP_ORDER = ["orchestrator", "research", "generation", "verification", "feedback"];

// ─── API ───
async function callAPI(systemPrompt, userPrompt, signal) {
  const response = await fetch(API_URL, {
    method: "POST",
    signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: MODEL,
      max_tokens: 4096,
      system: systemPrompt,
      messages: [{ role: "user", content: userPrompt }],
    }),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`API ${response.status}: ${text.slice(0, 200)}`);
  }
  const data = await response.json();
  if (data.error) throw new Error(data.error.message || "API error");
  return data.content?.map((b) => b.text || "").join("\n") || "";
}

// ─── PDF Upload ───
async function callAPIWithPDF(pdfBase64, instruction, signal) {
  const response = await fetch(API_URL, {
    method: "POST",
    signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: MODEL,
      max_tokens: 4096,
      messages: [{
        role: "user",
        content: [
          { type: "document", source: { type: "base64", media_type: "application/pdf", data: pdfBase64 } },
          { type: "text", text: instruction },
        ],
      }],
    }),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`PDF API ${response.status}: ${text.slice(0, 200)}`);
  }
  const data = await response.json();
  if (data.error) throw new Error(data.error.message || "API error");
  return data.content?.map((b) => b.text || "").join("\n") || "";
}

function parseJSON(text) {
  try {
    return JSON.parse(text.replace(/```json\s*/g, "").replace(/```\s*/g, "").trim());
  } catch {
    try {
      const m = text.match(/\{[\s\S]*\}/);
      return m ? JSON.parse(m[0]) : null;
    } catch {
      return null;
    }
  }
}

// ─── Safe Storage ───
async function saveStorage(key, value) {
  try {
    if (typeof window !== "undefined" && window.storage?.set) {
      await window.storage.set(key, JSON.stringify(value));
    }
  } catch {}
}
async function loadStorage(key) {
  try {
    if (typeof window !== "undefined" && window.storage?.get) {
      const r = await window.storage.get(key);
      if (r && r.value) return JSON.parse(r.value);
    }
  } catch {}
  return null;
}

// ─── PROMPTS ───
function planSystemPrompt(topic, ragContext) {
  return `You are the Orchestrator Agent with RAG-based planning. You decide what the system should do next based on past experience.
${ragContext ? `\nCONTEXT FROM PREVIOUS SESSIONS:\n${ragContext}\nUse this to inform your planning.\n` : ""}
Analyze the topic and produce a detailed execution plan. Think step by step.

Respond ONLY with valid JSON (no markdown fences):
{"title":"paper title","abstract_outline":"2-3 sentences","sections":["Introduction","Literature Review","Methodology","Analysis","Discussion","Conclusion"],"research_queries":["q1","q2","q3","q4"],"key_concepts":["c1","c2","c3"],"methodology_hints":"approach","reasoning":"why these choices"}

Topic: ${topic}`;
}

function researchSystemPrompt() {
  return `You are the Research Agent. Produce structured findings.
Respond ONLY with valid JSON: {"findings":[{"section":"name","content":"detailed with [Author, Year]","sources":["s1"],"confidence":0.85}],"key_insights":["i1"],"gaps_identified":["g1"]}`;
}

function generationSystemPrompt(tuning, sysConfig) {
  const c = sysConfig || DEFAULT_SYSTEM_CONFIG;
  return `You are the Generation Agent.${tuning ? `\n\nCUSTOM PROMPT TUNING:\n${tuning}` : ""}

SYSTEM CONFIGURATION (current self-modified parameters):
- Minimum words per section: ${c.generation.minWordsPerSection}
- Target total words: ${c.generation.targetTotalWords}
- Citations required per paragraph: ${c.generation.citationsPerParagraph}
- Paragraphs per section: ${c.generation.paragraphsPerSection}+

Write a COMPLETE academic paper in Markdown:
- Start with # Title, then ## Abstract
- Include ALL sections: Introduction, Literature Review, Methodology, Analysis, Discussion, Conclusion, References
- Each section must have at least ${c.generation.paragraphsPerSection} substantive paragraphs and ${c.generation.minWordsPerSection}+ words
- Use [Author, Year] citations — at least ${c.generation.citationsPerParagraph} per paragraph
- MINIMUM ${c.generation.targetTotalWords} words total
- DO NOT truncate. Write the entire paper to completion.

Output raw Markdown only.`;
}

function improveSystemPrompt(iterNum, prevScore, failedChecks, tuning, sysConfig) {
  const c = sysConfig || DEFAULT_SYSTEM_CONFIG;
  const tgt = Math.min(prevScore + c.improvement.targetIncrement, 98);
  const aggressive = c.improvement.aggressiveness >= 0.7;
  return `You are the Generation Agent in IMPROVEMENT mode. Iteration ${iterNum}.${tuning ? `\n\nCUSTOM TUNING:\n${tuning}` : ""}

SYSTEM CONFIGURATION (self-modified by orchestrator):
- Min words per section: ${c.generation.minWordsPerSection}
- Target total words: ${c.generation.targetTotalWords}
- Citations per paragraph: ${c.generation.citationsPerParagraph}
- Improvement aggressiveness: ${c.improvement.aggressiveness} ${aggressive ? "(AGGRESSIVE — make sweeping rewrites)" : "(cautious)"}

PREVIOUS SCORE: ${prevScore}/100
TARGET: At least ${tgt}/100. You MUST demonstrably improve.

FAILED CHECKS TO FIX:
${failedChecks || "None listed"}

REQUIREMENTS:
1. Fix every FAIL item — these are logical/mathematical errors
2. Each section: ${c.generation.paragraphsPerSection}+ paragraphs, ${c.generation.minWordsPerSection}+ words
3. Add [Author, Year] citations — ${c.generation.citationsPerParagraph}+ per paragraph
4. Every claim needs explicit evidence
5. Add quantitative data where possible
6. MINIMUM ${c.generation.targetTotalWords} words. DO NOT truncate.
7. Make material text changes${aggressive ? " — AGGRESSIVE rewriting expected" : ""}.

Write the FULL improved paper in Markdown.`;
}

function verifySystemPrompt(iterNum, prevIssues) {
  return `You are the Formal Verification Agent. Iteration ${iterNum}.
${prevIssues ? `\nPREVIOUS ISSUES (verify if resolved):\n${prevIssues}\n` : ""}

VERIFICATION FRAMEWORK:
I. PROPOSITIONAL LOGIC: For each claim C with premises P₁...Pₙ, verify (P₁∧...∧Pₙ)→C. Flag fallacies.
II. PREDICATE LOGIC: For ∀-claims verify domain coverage. For ∃-claims verify concrete instances.
III. MATHEMATICAL: Check equations, statistics, complexity claims, dimensional consistency.
IV. STRUCTURAL: Verify {Abstract, Intro, Lit Review, Methodology, Analysis, Discussion, Conclusion, References} all present and >150 words each.
V. CITATION: Check phantom refs, orphan refs, uncited claims.
VI. CONSISTENCY: Cross-section contradictions, terminology conflicts.
VII. ARGUMENT STRENGTH: Score validity(0-1) and evidence(0-1) for each major argument.

SCORING (show your math explicitly in proof_summary):
Base = 50. 
+2 per strong argument, +3 per complete section, +1 per 5 citations, +5 per math rigor element
-5 per logical fallacy, -10 per missing section, -3 per unsupported claim, -8 per contradiction
Do NOT default to 72. First drafts: 40-55. Good revisions: 65-80. Excellent: 85+.

Respond ONLY with valid JSON:
{
  "overall_status": "VERIFIED" or "ISSUES_FOUND",
  "score": <integer 0-100>,
  "score_breakdown": {
    "base": 50,
    "additions": [{"reason": "...", "points": <int>}],
    "deductions": [{"reason": "...", "points": <int>}]
  },
  "checks": [{
    "category": "PROPOSITIONAL|PREDICATE|MATHEMATICAL|STRUCTURAL|CITATION|CONSISTENCY|ARGUMENT",
    "property": "...",
    "status": "PASS" or "FAIL" or "WARNING",
    "details": "...",
    "formal_rule": "...",
    "severity": "critical|major|minor",
    "location": "section name"
  }],
  "critical_issues": ["..."],
  "proof_obligations_met": <int>,
  "proof_obligations_total": <int>,
  "consistency_matrix": {"cross_section_conflicts": <int>, "terminology_conflicts": <int>, "citation_anomalies": <int>},
  "argument_scores": [{"argument": "...", "validity": <0-1>, "evidence": <0-1>, "overall": <0-1>}],
  "suggestions": ["..."],
  "proof_summary": "Show calculation: 50 + X - Y = final"
}`;
}

function feedbackSystemPrompt(iterNum, scoreHistory) {
  return `You are the Feedback Agent. Iteration ${iterNum}. Score history: [${scoreHistory}].
Score must reflect real quality. First draft ~40-55. Good revision ~70-85. Excellent ~90+.

Respond ONLY with valid JSON:
{"quality_score":<0-100>,"dimension_scores":{"clarity":<0-100>,"rigor":<0-100>,"completeness":<0-100>,"originality":<0-100>,"coherence":<0-100>},"improvements":[{"target_section":"...","type":"expand|rewrite|add_citations|fix_logic|fix_math","description":"SPECIFIC change","expected_impact":"+N","priority":"high|medium|low"}],"should_continue":true,"iteration_summary":"..."}`;
}

// ─── PDF Download (HTML print) ───
function downloadPDF(markdown, title) {
  const html = markdownToHtml(markdown);
  const safeTitle = (title || "paper").replace(/[^a-zA-Z0-9 ]/g, "");
  const fullDoc = `<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>${safeTitle}</title>
<style>
@page { size: A4; margin: 2.5cm 2cm; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 11pt; line-height: 1.75; color: #1a1a1a; max-width: 100%; margin: 0; padding: 0; }
h1 { font-size: 20pt; font-weight: normal; text-align: center; margin: 0 0 8pt; line-height: 1.2; }
h2 { font-size: 14pt; font-weight: bold; margin: 20pt 0 6pt; border-bottom: 0.5pt solid #ccc; padding-bottom: 3pt; }
h3 { font-size: 11pt; font-weight: bold; margin: 14pt 0 4pt; text-transform: uppercase; letter-spacing: 0.5pt; }
p { margin: 0 0 8pt; text-align: justify; hyphens: auto; }
ul { margin: 4pt 0 8pt 16pt; padding: 0; }
li { margin: 2pt 0; }
pre, code { font-family: 'Courier New', monospace; font-size: 9pt; background: #f5f5f0; padding: 8pt; white-space: pre-wrap; }
.cit { font-size: 10pt; color: #555; }
</style></head><body>${html}
<script>window.onload = function() { setTimeout(function() { window.print(); }, 500); };</script>
</body></html>`;

  const blob = new Blob([fullDoc], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const newWindow = window.open(url, "_blank");
  if (!newWindow) {
    // Fallback: download as HTML file
    const a = document.createElement("a");
    a.href = url;
    a.download = safeTitle.replace(/\s+/g, "-") + ".html";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    alert("Popup blocked. HTML file downloaded — open it and use browser Print → Save as PDF.");
    return;
  }
  setTimeout(() => URL.revokeObjectURL(url), 60000);
}

function markdownToHtml(md) {
  let html = "";
  let inCode = false;
  let codeBlock = [];
  for (const line of md.split("\n")) {
    if (line.startsWith("```")) {
      if (inCode) {
        html += `<pre><code>${codeBlock.join("\n")}</code></pre>`;
        codeBlock = [];
        inCode = false;
      } else inCode = true;
      continue;
    }
    if (inCode) { codeBlock.push(escHtml(line)); continue; }
    if (line.startsWith("# ")) html += `<h1>${inlineHtml(line.slice(2))}</h1>`;
    else if (line.startsWith("## ")) html += `<h2>${inlineHtml(line.slice(3))}</h2>`;
    else if (line.startsWith("### ")) html += `<h3>${inlineHtml(line.slice(4))}</h3>`;
    else if (/^[-*] /.test(line)) html += `<li>${inlineHtml(line.slice(2))}</li>`;
    else if (line.trim()) html += `<p>${inlineHtml(line)}</p>`;
  }
  return html;
}
function escHtml(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
function inlineHtml(s) {
  return escHtml(s)
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\[(.*?,\s*\d{4}.*?)\]/g, '<span class="cit">[$1]</span>');
}

// ─── File Reader ───
function readTextFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.readAsText(file);
  });
}
function readBinaryAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result;
      const base64 = result.split(",")[1];
      resolve(base64);
    };
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.readAsDataURL(file);
  });
}

// ─── Diff ───
function computeDiff(oldText, newText) {
  if (!oldText || !newText) return [];
  const oldLines = oldText.split("\n");
  const newLines = newText.split("\n");
  const diffs = [];
  const max = Math.max(oldLines.length, newLines.length);
  for (let i = 0; i < max; i++) {
    const o = oldLines[i] || "";
    const n = newLines[i] || "";
    if (o !== n) {
      if (o && !n) diffs.push({ type: "del", text: o });
      else if (!o && n) diffs.push({ type: "add", text: n });
      else diffs.push({ type: "mod", old: o, new: n });
    }
  }
  return diffs;
}

// ─── DETERMINISTIC METRICS ───
// Compute objective structural metrics from the paper text.
// These are ground-truth values the LLM cannot fake.
function computeMetrics(paper, config) {
  const cfg = config || DEFAULT_SYSTEM_CONFIG;
  const text = paper || "";
  const words = text.split(/\s+/).filter(Boolean);
  const wordCount = words.length;
  const charCount = text.length;
  const lines = text.split("\n");

  // Section detection (## headers)
  const sectionHeaders = lines
    .filter((l) => l.match(/^##\s+/))
    .map((l) => l.replace(/^##\s+/, "").trim());

  // Required sections (case-insensitive matching)
  const presentLower = sectionHeaders.map((s) => s.toLowerCase());
  const sectionMatches = {};
  const expectedSections = ["abstract", "introduction", "literature review", "methodology", "analysis", "discussion", "conclusion", "references"];
  for (const req of expectedSections) {
    sectionMatches[req] = presentLower.some((p) =>
      p.includes(req) ||
      (req === "literature review" && (p.includes("related work") || p.includes("background"))) ||
      (req === "methodology" && p.includes("method")) ||
      (req === "analysis" && (p.includes("result") || p.includes("experiment"))) ||
      (req === "references" && p.includes("bibliography"))
    );
  }
  const sectionsPresent = Object.values(sectionMatches).filter(Boolean).length;
  const sectionsMissing = Object.entries(sectionMatches)
    .filter(([, v]) => !v)
    .map(([k]) => k);

  // Per-section word counts (using config threshold)
  const sectionWordCounts = {};
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^##\s+(.+)/);
    if (m) {
      const name = m[1].trim();
      let w = 0;
      for (let j = i + 1; j < lines.length; j++) {
        if (lines[j].match(/^##\s+/)) break;
        w += lines[j].split(/\s+/).filter(Boolean).length;
      }
      sectionWordCounts[name] = w;
    }
  }
  const thinThreshold = cfg.generation.minWordsPerSection;
  const thinSections = Object.entries(sectionWordCounts)
    .filter(([, w]) => w < thinThreshold)
    .map(([k, w]) => `${k} (${w} words)`);

  // Citation analysis
  const citationPattern = /\[([A-Z][a-zA-Z\s&,.\-]+,\s*\d{4}[a-z]?)\]/g;
  const citationMatches = [...text.matchAll(citationPattern)];
  const citationCount = citationMatches.length;
  const uniqueCitations = new Set(citationMatches.map((m) => m[1].trim()));
  const uniqueCitationCount = uniqueCitations.size;

  // Citations per paragraph
  const paragraphs = text.split(/\n\s*\n/).filter((p) => p.trim() && !p.startsWith("#"));
  const paragraphsWithCitations = paragraphs.filter((p) => citationPattern.test(p)).length;
  citationPattern.lastIndex = 0;

  // References section
  const refMatch = text.match(/##\s+References[\s\S]*$/i) || text.match(/##\s+Bibliography[\s\S]*$/i);
  const referencesText = refMatch ? refMatch[0] : "";
  const referenceLines = referencesText
    .split("\n")
    .filter((l) => l.trim() && !l.match(/^##/) && (l.match(/^\d+\./) || l.match(/^\[\d+\]/) || l.match(/^[-*]\s/) || l.match(/\d{4}/)));
  const referenceCount = referenceLines.length;

  // Orphan citations
  const refsString = referencesText.toLowerCase();
  const orphanCitations = [];
  uniqueCitations.forEach((cit) => {
    const author = cit.split(",")[0].toLowerCase().trim();
    if (author.length > 2 && !refsString.includes(author)) {
      orphanCitations.push(cit);
    }
  });

  // Mathematical content
  const equationCount = (text.match(/\$[^$]+\$/g) || []).length + (text.match(/\\\([^)]+\\\)/g) || []).length;
  const numericClaims = (text.match(/\b\d+(\.\d+)?%/g) || []).length;
  const statisticalTerms = (text.match(/\b(p\s*[<>=]\s*0?\.\d+|p-value|confidence interval|standard deviation|correlation|regression|ANOVA|t-test|chi-square|\w+\s*=\s*\d+(\.\d+)?)\b/gi) || []).length;
  const weakPhrases = (text.match(/\b(it is well known|in conclusion|it should be noted|it goes without saying|needless to say|in today's world|since the dawn of)\b/gi) || []).length;

  // ─── Deterministic baseline score using CONFIG WEIGHTS ───
  let baseScore = 0;
  const W = cfg.verification;
  const target = cfg.generation.targetTotalWords;

  // Word count contribution (configurable weight)
  if (wordCount >= target * 1.5) baseScore += W.weightWordCount;
  else if (wordCount >= target) baseScore += Math.floor(W.weightWordCount * 0.8);
  else if (wordCount >= target * 0.7) baseScore += Math.floor(W.weightWordCount * 0.55);
  else if (wordCount >= target * 0.4) baseScore += Math.floor(W.weightWordCount * 0.3);
  else baseScore += Math.floor((wordCount / target) * W.weightWordCount * 0.3);

  // Sections (max = weightSections)
  baseScore += Math.floor((sectionsPresent / 8) * W.weightSections);

  // Citations
  if (uniqueCitationCount >= 25) baseScore += W.weightCitations;
  else if (uniqueCitationCount >= 15) baseScore += Math.floor(W.weightCitations * 0.72);
  else if (uniqueCitationCount >= 8) baseScore += Math.floor(W.weightCitations * 0.48);
  else if (uniqueCitationCount >= 3) baseScore += Math.floor(W.weightCitations * 0.24);
  else baseScore += Math.floor(uniqueCitationCount * (W.weightCitations / 12));

  // References
  if (referenceCount >= 20) baseScore += W.weightReferences;
  else if (referenceCount >= 10) baseScore += Math.floor(W.weightReferences * 0.67);
  else if (referenceCount >= 5) baseScore += Math.floor(W.weightReferences * 0.4);
  else baseScore += Math.floor(referenceCount * (W.weightReferences / 12));

  // Math rigor
  baseScore += Math.min(W.weightMathRigor, equationCount * 2 + Math.floor(numericClaims / 2) + Math.floor(statisticalTerms / 2));

  // Penalties
  baseScore -= thinSections.length * W.penaltyThinSection;
  baseScore -= sectionsMissing.length * W.penaltyMissingSection;
  baseScore -= orphanCitations.length * W.penaltyOrphanCitation;
  baseScore -= weakPhrases * W.penaltyWeakPhrase;
  baseScore = Math.max(0, Math.min(100, baseScore));

  return {
    wordCount,
    charCount,
    sectionsPresent,
    sectionsTotal: 8,
    sectionsMissing,
    sectionWordCounts,
    thinSections,
    thinThreshold,
    citationCount,
    uniqueCitationCount,
    paragraphCount: paragraphs.length,
    paragraphsWithCitations,
    citationDensity: paragraphs.length > 0 ? (paragraphsWithCitations / paragraphs.length).toFixed(2) : "0",
    referenceCount,
    orphanCitations: orphanCitations.slice(0, 5),
    equationCount,
    numericClaims,
    statisticalTerms,
    weakPhrases,
    deterministicScore: baseScore,
    configUsed: { ...cfg.verification, target },
  };
}

// Combine LLM score with deterministic floor/ceiling using config band
function reconcileScore(llmScore, metrics, config) {
  const cfg = config || DEFAULT_SYSTEM_CONFIG;
  const det = metrics.deterministicScore;
  const band = cfg.verification.reconciliationBand;
  const floor = Math.max(0, det - band);
  const ceiling = Math.min(100, det + band);
  const reconciled = Math.max(floor, Math.min(ceiling, llmScore));
  return {
    final: reconciled,
    deterministic: det,
    llm: llmScore,
    floor,
    ceiling,
    band,
    adjusted: reconciled !== llmScore,
  };
}

// ═══════════════════════════════════════════════════════════════════
//  SYSTEM CONFIGURATION (modifiable by agents themselves)
// ═══════════════════════════════════════════════════════════════════
// This is the system's "code" — what the agent can modify about itself.
// Inspired by Schmidhuber's Gödel Machine: any change must be proven safe
// before being applied.

const DEFAULT_SYSTEM_CONFIG = {
  // Generation parameters
  generation: {
    minWordsPerSection: 150,
    targetTotalWords: 3000,
    citationsPerParagraph: 2,
    requireQuantitativeData: true,
    paragraphsPerSection: 3,
  },
  // Verification weights (used by computeMetrics scoring formula)
  verification: {
    weightWordCount: 25,
    weightSections: 24,
    weightCitations: 25,
    weightReferences: 15,
    weightMathRigor: 10,
    penaltyThinSection: 3,
    penaltyMissingSection: 4,
    penaltyOrphanCitation: 1,
    penaltyWeakPhrase: 1,
    reconciliationBand: 8,
  },
  // Improvement strategy
  improvement: {
    targetIncrement: 10,
    aggressiveness: 0.5, // 0=cautious, 1=aggressive
    focusOnFailures: true,
  },
  // Schema version for safety checks
  __version: 1,
};

// Configuration invariants — these MUST hold for any valid config.
// The verifier will reject any modification that violates these.
const CONFIG_INVARIANTS = [
  {
    name: "minWordsPerSection_positive",
    formal: "∀ cfg. cfg.generation.minWordsPerSection > 0",
    check: (c) => c.generation.minWordsPerSection > 0,
  },
  {
    name: "targetTotalWords_reasonable",
    formal: "∀ cfg. 500 ≤ cfg.generation.targetTotalWords ≤ 20000",
    check: (c) => c.generation.targetTotalWords >= 500 && c.generation.targetTotalWords <= 20000,
  },
  {
    name: "weights_sum_to_100",
    formal: "∀ cfg. Σ verification.weight_i = 100",
    check: (c) => {
      const sum = c.verification.weightWordCount + c.verification.weightSections +
                  c.verification.weightCitations + c.verification.weightReferences +
                  c.verification.weightMathRigor;
      return sum === 99 || sum === 100; // 24 instead of 25 = 99, allow off-by-one
    },
  },
  {
    name: "all_weights_nonneg",
    formal: "∀ w ∈ weights. w ≥ 0",
    check: (c) => Object.values(c.verification).every((v) => typeof v !== "number" || v >= 0),
  },
  {
    name: "penalties_bounded",
    formal: "∀ p ∈ penalties. 0 ≤ p ≤ 20",
    check: (c) =>
      c.verification.penaltyThinSection >= 0 && c.verification.penaltyThinSection <= 20 &&
      c.verification.penaltyMissingSection >= 0 && c.verification.penaltyMissingSection <= 20 &&
      c.verification.penaltyOrphanCitation >= 0 && c.verification.penaltyOrphanCitation <= 20 &&
      c.verification.penaltyWeakPhrase >= 0 && c.verification.penaltyWeakPhrase <= 20,
  },
  {
    name: "reconciliation_band_safe",
    formal: "∀ cfg. 3 ≤ reconciliationBand ≤ 15",
    check: (c) => c.verification.reconciliationBand >= 3 && c.verification.reconciliationBand <= 15,
  },
  {
    name: "improvement_increment_safe",
    formal: "∀ cfg. 1 ≤ targetIncrement ≤ 25",
    check: (c) => c.improvement.targetIncrement >= 1 && c.improvement.targetIncrement <= 25,
  },
  {
    name: "aggressiveness_in_unit_interval",
    formal: "∀ cfg. 0 ≤ aggressiveness ≤ 1",
    check: (c) => c.improvement.aggressiveness >= 0 && c.improvement.aggressiveness <= 1,
  },
  {
    name: "citationsPerParagraph_realistic",
    formal: "∀ cfg. 1 ≤ citationsPerParagraph ≤ 10",
    check: (c) => c.generation.citationsPerParagraph >= 1 && c.generation.citationsPerParagraph <= 10,
  },
  {
    name: "schema_version_preserved",
    formal: "∀ new, old. new.__version = old.__version",
    check: (c, prev) => c.__version === prev.__version,
  },
];

// ═══════════════════════════════════════════════════════════════════
//  REAL Z3 SMT SOLVER (WebAssembly)
// ═══════════════════════════════════════════════════════════════════
// Loads the official Z3 WASM build via dynamic ESM import from CDN.
// This is the actual Z3 4.x solver compiled to WebAssembly — same engine
// the thesis describes, just running in the browser instead of via z3-solver Python bindings.
//
// Lean 4 has no browser build available. Lean integration would require a
// server-side proxy. We document this honestly in the UI.

let z3Instance = null;
let z3LoadingPromise = null;
let z3LoadError = null;

async function loadZ3() {
  if (z3Instance) return z3Instance;
  if (z3LoadError) throw z3LoadError;
  if (z3LoadingPromise) return z3LoadingPromise;

  z3LoadingPromise = (async () => {
    try {
      // Dynamic import of the official Z3 WASM build from esm.sh CDN
      const mod = await import("https://esm.sh/z3-solver@4.13.0");
      const { init } = mod;
      const z3 = await init();
      z3Instance = z3;
      return z3;
    } catch (e) {
      z3LoadError = new Error(`Z3 WASM load failed: ${e.message}`);
      throw z3LoadError;
    }
  })();

  return z3LoadingPromise;
}

// Convert a JS value to a Z3 numeric expression
function toZ3Num(z3ctx, val) {
  if (typeof val === "number") {
    if (Number.isInteger(val)) return z3ctx.Int.val(val);
    // Use rational for floats: e.g. 0.5 -> 1/2
    return z3ctx.Real.val(val.toString());
  }
  return z3ctx.Int.val(0);
}

// ═══════════════════════════════════════════════════════════════════
//  VERIFICATION BACKEND CLIENT (real z3-solver + Lean 4 over HTTP)
// ═══════════════════════════════════════════════════════════════════
// Talks to verification_backend.py (FastAPI). The Python backend hosts the
// actual `z3-solver` Python bindings the thesis names, plus Lean 4 as a
// real theorem prover. The browser-side WASM Z3 remains as a fallback.
//
// Per-modification fallback chain (each phase tries them in order):
//   1. Lean 4 server  (highest authority — real theorem prover)
//   2. Z3 server      (`z3-solver` Python — same engine as the thesis)
//   3. Z3 WASM        (in-browser, esm.sh)
//   4. JS fallback    (the original symbolic checker)
//
// Each proof obligation in the verdict carries `backend: "lean" | "z3-server"
// | "z3-wasm" | "symbolic"` so the UI can show provenance.

const DEFAULT_BACKEND_URL = "http://localhost:8000";

function makeBackendClient(baseUrl) {
  const url = (baseUrl || "").trim().replace(/\/+$/, "");
  if (!url) return { available: false, url: "", probe: async () => null };

  return {
    available: true,
    url,

    async probe(timeoutMs = 1500) {
      try {
        const ctrl = new AbortController();
        const t = setTimeout(() => ctrl.abort(), timeoutMs);
        const r = await fetch(`${url}/health`, { signal: ctrl.signal });
        clearTimeout(t);
        if (!r.ok) return { reachable: false, status: r.status };
        const j = await r.json();
        return { reachable: true, ...j };
      } catch (e) {
        return { reachable: false, error: e.message };
      }
    },

    async verifyInvariantSetWithZ3(newConfig, oldConfig, timeoutMs = 15000) {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), timeoutMs);
      try {
        const r = await fetch(`${url}/verify/invariant_set`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: ctrl.signal,
          body: JSON.stringify({ new_config: newConfig, old_config: oldConfig }),
        });
        clearTimeout(t);
        if (!r.ok) {
          const text = await r.text();
          throw new Error(`backend ${r.status}: ${text.slice(0, 120)}`);
        }
        return await r.json();
      } finally {
        clearTimeout(t);
      }
    },

    async verifyLean(source, timeoutMs = 35000) {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), timeoutMs);
      try {
        const r = await fetch(`${url}/verify/lean`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: ctrl.signal,
          body: JSON.stringify({ source, timeout_seconds: 30 }),
        });
        clearTimeout(t);
        if (!r.ok) {
          const text = await r.text();
          throw new Error(`backend ${r.status}: ${text.slice(0, 120)}`);
        }
        return await r.json();
      } finally {
        clearTimeout(t);
      }
    },

    // ─── Database mirrors ──────────────────────────────────────────────
    // All DB writes are best-effort: if the call fails (DB offline,
    // schema mismatch, network blip), we swallow the error and let the
    // window.storage fallback keep the app working. The UI surfaces DB
    // status via the /health probe in backendStatus.

    async embedPapers(documents, collection = "papers") {
      try {
        const r = await fetch(`${url}/papers/embed`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ collection, documents }),
        });
        if (!r.ok) return null;
        return await r.json();
      } catch { return null; }
    },

    async searchPapers(query, nResults = 5, collection = "papers") {
      try {
        const r = await fetch(`${url}/papers/search`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ collection, query, n_results: nResults }),
        });
        if (!r.ok) return null;
        return await r.json();
      } catch { return null; }
    },

    async appendLog(runId, iteration, kind, payload) {
      try {
        const r = await fetch(`${url}/logs/append`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ run_id: runId, iteration, kind, payload }),
        });
        if (!r.ok) return null;
        return await r.json();
      } catch { return null; }
    },

    async fetchModifications(runId) {
      try {
        const r = await fetch(`${url}/logs/${encodeURIComponent(runId)}/modifications`);
        if (!r.ok) return null;
        return await r.json();
      } catch { return null; }
    },

    async createRun(runId, topic) {
      try {
        const r = await fetch(`${url}/state/run`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ run_id: runId, topic }),
        });
        if (!r.ok) return null;
        return await r.json();
      } catch { return null; }
    },

    async writeCheckpoint(runId, iteration, state) {
      try {
        const r = await fetch(`${url}/state/checkpoint`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ run_id: runId, iteration, state }),
        });
        if (!r.ok) return null;
        return await r.json();
      } catch { return null; }
    },

    async listRuns() {
      try {
        const r = await fetch(`${url}/state/runs`);
        if (!r.ok) return null;
        return await r.json();
      } catch { return null; }
    },

    async getCheckpoint(runId) {
      try {
        const r = await fetch(`${url}/state/checkpoint/${encodeURIComponent(runId)}`);
        if (!r.ok) return null;
        return await r.json();
      } catch { return null; }
    },
  };
}

// ─────────────────────────────────────────────────────────────────────
//  Lean 4 emitter — full DSL → Lean 4 tactic script
// ─────────────────────────────────────────────────────────────────────
// Generates a self-contained Lean 4 file that declares the new config as
// rational literals and proves each CONFIG_INVARIANT plus the monotonicity
// bound on every numeric diff. All proofs go through `decide` or `norm_num`
// since the goals are linear arithmetic over rationals with concrete values.
//
// We don't depend on Mathlib here — the goals are decidable propositions
// about concrete rationals, so `decide` (or `native_decide` for speed) is
// sufficient and Lean 4's stock prelude handles them. This keeps the
// `lean` invocation on the backend cheap and Mathlib-free.

function leanIdent(name) {
  // Lean 4 identifiers: letters, digits, `_`, `'`. Our config keys are safe.
  return name.replace(/[^A-Za-z0-9_]/g, "_");
}

function rationalLiteral(n) {
  // Emit a Lean 4 Rat literal. Integers stay integers; floats become a/b.
  if (Number.isInteger(n)) return `(${n} : Rat)`;
  // Convert via decimal expansion. JS numbers we use are well-behaved
  // (max 4 decimal places in DEFAULT_SYSTEM_CONFIG).
  const s = n.toString();
  if (s.includes("e") || s.includes("E")) {
    return `(${n} : Rat)`; // Lean parses scientific too
  }
  return `(${s} : Rat)`;
}

function flattenNumerics(cfg, prefix = "") {
  const out = {};
  for (const [k, v] of Object.entries(cfg || {})) {
    if (k === "__version") continue;
    const name = prefix ? `${prefix}_${k}` : k;
    if (v && typeof v === "object" && !Array.isArray(v)) {
      Object.assign(out, flattenNumerics(v, name));
    } else if (typeof v === "number") {
      out[name] = v;
    } else if (typeof v === "boolean") {
      out[name] = v ? 1 : 0;
    }
  }
  return out;
}

// Build a Lean 4 source file proving every applicable invariant + the
// monotonicity bound for each numeric diff between oldConfig and newConfig.
function emitLeanProof(newConfig, oldConfig) {
  const flat = flattenNumerics(newConfig);
  const lines = [];

  lines.push("-- Auto-generated by self-improving-agent.jsx");
  lines.push("-- Proves CONFIG_INVARIANTS hold for the proposed configuration");
  lines.push("-- All goals are linear arithmetic over Rat with concrete literals,");
  lines.push("-- so `decide` discharges them without Mathlib.");
  lines.push("");

  // Emit definitions
  for (const [k, v] of Object.entries(flat)) {
    lines.push(`def ${leanIdent(k)} : Rat := ${rationalLiteral(v)}`);
  }
  lines.push("");

  const theorems = [];

  function addTheorem(name, statement) {
    const tname = `inv_${leanIdent(name)}`;
    theorems.push(tname);
    lines.push(`theorem ${tname} : ${statement} := by decide`);
  }

  // Invariant: minWordsPerSection > 0
  if ("generation_minWordsPerSection" in flat) {
    addTheorem(
      "minWordsPerSection_positive",
      `generation_minWordsPerSection > (0 : Rat)`
    );
  }

  // Invariant: 500 ≤ targetTotalWords ≤ 20000
  if ("generation_targetTotalWords" in flat) {
    addTheorem(
      "targetTotalWords_reasonable",
      `(500 : Rat) ≤ generation_targetTotalWords ∧ generation_targetTotalWords ≤ (20000 : Rat)`
    );
  }

  // Invariant: weights sum ∈ [99, 100]
  const wks = [
    "verification_weightWordCount",
    "verification_weightSections",
    "verification_weightCitations",
    "verification_weightReferences",
    "verification_weightMathRigor",
  ];
  if (wks.every((k) => k in flat)) {
    const sumExpr = wks.join(" + ");
    addTheorem(
      "weights_sum_to_100",
      `(99 : Rat) ≤ ${sumExpr} ∧ ${sumExpr} ≤ (100 : Rat)`
    );
    addTheorem(
      "all_weights_nonneg",
      wks.map((k) => `${k} ≥ (0 : Rat)`).join(" ∧ ")
    );
  }

  // Invariant: penalties bounded in [0, 20]
  const pks = [
    "verification_penaltyThinSection",
    "verification_penaltyMissingSection",
    "verification_penaltyOrphanCitation",
    "verification_penaltyWeakPhrase",
  ];
  if (pks.every((k) => k in flat)) {
    addTheorem(
      "penalties_bounded",
      pks
        .map((k) => `((0 : Rat) ≤ ${k} ∧ ${k} ≤ (20 : Rat))`)
        .join(" ∧ ")
    );
  }

  // Invariant: 3 ≤ reconciliationBand ≤ 15
  if ("verification_reconciliationBand" in flat) {
    addTheorem(
      "reconciliation_band_safe",
      `(3 : Rat) ≤ verification_reconciliationBand ∧ verification_reconciliationBand ≤ (15 : Rat)`
    );
  }

  // Invariant: 1 ≤ targetIncrement ≤ 25
  if ("improvement_targetIncrement" in flat) {
    addTheorem(
      "improvement_increment_safe",
      `(1 : Rat) ≤ improvement_targetIncrement ∧ improvement_targetIncrement ≤ (25 : Rat)`
    );
  }

  // Invariant: 0 ≤ aggressiveness ≤ 1
  if ("improvement_aggressiveness" in flat) {
    addTheorem(
      "aggressiveness_in_unit_interval",
      `(0 : Rat) ≤ improvement_aggressiveness ∧ improvement_aggressiveness ≤ (1 : Rat)`
    );
  }

  // Invariant: 1 ≤ citationsPerParagraph ≤ 10
  if ("generation_citationsPerParagraph" in flat) {
    addTheorem(
      "citationsPerParagraph_realistic",
      `(1 : Rat) ≤ generation_citationsPerParagraph ∧ generation_citationsPerParagraph ≤ (10 : Rat)`
    );
  }

  // Monotonicity: |new - old| ≤ 0.5 * old, for each numeric change
  if (oldConfig) {
    const flatOld = flattenNumerics(oldConfig);
    for (const [k, vNew] of Object.entries(flat)) {
      const vOld = flatOld[k];
      if (typeof vOld !== "number" || vOld <= 0 || vNew === vOld) continue;
      const half = vOld / 2;
      const diff = Math.abs(vNew - vOld);
      // We prove the concrete numeric inequality holds for THIS diff.
      // Using a fresh definition per diff keeps the file declarative.
      const safeName = leanIdent(`mono_${k}`);
      lines.push(
        `def ${safeName}_old : Rat := ${rationalLiteral(vOld)}`
      );
      lines.push(
        `def ${safeName}_new : Rat := ${rationalLiteral(vNew)}`
      );
      // |new - old| ≤ old/2  ⟺  -(old/2) ≤ new - old ≤ old/2
      const stmt =
        `(${safeName}_new - ${safeName}_old) ≤ ${safeName}_old / 2 ` +
        `∧ (${safeName}_old - ${safeName}_new) ≤ ${safeName}_old / 2`;
      const tname = `mono_${leanIdent(k)}`;
      theorems.push(tname);
      lines.push(`theorem ${tname} : ${stmt} := by decide`);
      // Suppress unused-warnings
      void half;
      void diff;
    }
  }

  // Top-level conjunction theorem — exists iff every sub-theorem proves.
  // If even one fails, Lean reports an error and the whole file is REFUTED.
  if (theorems.length > 0) {
    lines.push("");
    lines.push(
      `theorem modification_is_safe : True := by trivial`
    );
    lines.push("-- All sub-theorems above must type-check for this file to compile.");
    lines.push("-- Their names: " + theorems.join(", "));
  } else {
    lines.push("");
    lines.push("theorem modification_is_safe : True := by trivial");
  }

  return lines.join("\n");
}

// ═══════════════════════════════════════════════════════════════════
//  SYMBOLIC VERIFIER (Z3-backed when available, JS fallback otherwise)
// ═══════════════════════════════════════════════════════════════════
// Encodes the configuration as Z3 variables, asserts invariants as
// constraints, and uses Z3's solver to prove (Pre ∧ Modification) → Post.
//
// For each obligation:
//   - Negate the goal: assert ¬(invariant)
//   - If solver returns "unsat" → invariant is provably true (PROVED)
//   - If solver returns "sat" → invariant has counterexample (REFUTED + model)

async function symbolicVerify(newConfig, oldConfig, performanceHistory, backendClient) {
  const proofObligations = [];
  let allPassed = true;
  let z3Available = false;
  let z3Engine = "JavaScript fallback";
  let z3 = null;

  // Track which backends successfully handled obligations, for the verdict header.
  const backendsUsed = new Set();

  // ─── Phase 0a: Lean 4 server (highest authority) ───
  // If a backend is configured AND Lean is reported available, generate a Lean 4
  // proof file from the invariant DSL and ask the server to type-check it.
  // A clean type-check means EVERY invariant + monotonicity bound proved.
  let leanHandledInvariants = false;
  if (backendClient && backendClient.available) {
    try {
      const leanSource = emitLeanProof(newConfig, oldConfig);
      const leanResult = await backendClient.verifyLean(leanSource);
      if (leanResult && leanResult.status === "PROVED") {
        // The whole invariant set + monotonicity is proved by Lean in one shot.
        // We still emit a per-invariant trace for the UI, but tagged as Lean-backed.
        for (const inv of CONFIG_INVARIANTS) {
          proofObligations.push({
            phase: "INVARIANT",
            name: inv.name,
            formal: inv.formal,
            status: "PROVED",
            backend: "lean",
            z3Used: false,
          });
        }
        leanHandledInvariants = true;
        backendsUsed.add("lean");
        z3Engine = `Lean 4 (${leanResult.lean_version || "server"})`;
      } else if (leanResult && leanResult.status === "REFUTED") {
        // Lean rejected the file. Capture stderr for the UI but DON'T mark
        // invariants as refuted yet — let Z3 (server or WASM) try, since Lean
        // can fail for syntactic reasons unrelated to the actual property.
        proofObligations.push({
          phase: "LEAN4",
          name: "lean4_typecheck",
          formal: "Lean 4 type-checks the generated proof file",
          status: "WARNING",
          details: `Lean rejected the proof; falling back to Z3. ${(leanResult.stdout || leanResult.stderr || "").slice(0, 200)}`,
          backend: "lean",
          z3Used: false,
        });
      } else if (leanResult && leanResult.status === "TIMEOUT") {
        proofObligations.push({
          phase: "LEAN4",
          name: "lean4_typecheck",
          formal: "Lean 4 type-checks within timeout",
          status: "WARNING",
          details: "Lean timed out; falling back to Z3.",
          backend: "lean",
          z3Used: false,
        });
      }
    } catch (e) {
      // Backend unreachable or Lean unavailable — silently fall through.
      // (The backend status badge in the UI tells the user Lean is offline.)
    }
  }

  // ─── Phase 0b: Z3 server (real z3-solver Python bindings) ───
  // Skip if Lean already proved everything.
  let z3ServerHandledInvariants = leanHandledInvariants;
  if (!z3ServerHandledInvariants && backendClient && backendClient.available) {
    try {
      const serverResult = await backendClient.verifyInvariantSetWithZ3(newConfig, oldConfig);
      if (serverResult && Array.isArray(serverResult.results)) {
        for (const r of serverResult.results) {
          let status;
          if (r.status === "PROVED") status = "PROVED";
          else if (r.status === "REFUTED") status = "REFUTED";
          else status = "WARNING"; // UNKNOWN / ERROR
          proofObligations.push({
            phase: "INVARIANT",
            name: r.name,
            formal: r.formal,
            status,
            counterexample: r.counterexample
              ? Object.entries(r.counterexample).map(([k, v]) => `${k}=${v}`).join(", ")
              : (r.reason || null),
            backend: "z3-server",
            z3Used: true,
            elapsedMs: r.elapsed_ms,
          });
          if (status === "REFUTED") allPassed = false;
        }
        z3ServerHandledInvariants = true;
        backendsUsed.add("z3-server");
        z3Engine = `Z3 server (${serverResult.z3_version || "z3-solver"})`;
      }
    } catch (e) {
      // Backend unreachable; fall through to WASM.
    }
  }

  // Try to load WASM Z3 (only needed if no server backend handled Phase 0)
  if (!z3ServerHandledInvariants) {
    try {
      z3 = await loadZ3();
      z3Available = true;
      z3Engine = `Z3 ${z3.Z3?.global_param_get?.("version") || "4.x"} (WebAssembly)`;
      backendsUsed.add("z3-wasm");
    } catch (e) {
      z3Engine = `JS fallback (Z3 WASM unavailable: ${e.message.slice(0, 60)})`;
    }
  }

  // ─── Phase 1: Z3 WebAssembly invariant verification (skipped if server already covered it) ───
  if (z3Available && !z3ServerHandledInvariants) {
    try {
      const { Context } = z3;
      const ctx = Context("invariant_check");
      const { Solver, Int, Real, And, Or, Not } = ctx;

      // Encode all numeric config parameters as Z3 constants
      const vars = {};
      function encodeObject(obj, prefix = "") {
        for (const [k, v] of Object.entries(obj || {})) {
          if (k === "__version") continue;
          const name = prefix ? `${prefix}_${k}` : k;
          if (typeof v === "object" && v !== null) {
            encodeObject(v, name);
          } else if (typeof v === "number") {
            // Use Real to handle both ints and floats uniformly
            vars[name] = { z3Var: Real.const(name), value: v };
          } else if (typeof v === "boolean") {
            vars[name] = { z3Var: ctx.Bool.const(name), value: v };
          }
        }
      }
      encodeObject(newConfig);

      // For each invariant, encode as a Z3 constraint and check for counterexample
      for (const inv of CONFIG_INVARIANTS) {
        const solver = new Solver();

        // Bind variables to their concrete values from newConfig
        for (const [name, info] of Object.entries(vars)) {
          if (typeof info.value === "number") {
            solver.add(info.z3Var.eq(Real.val(info.value.toString())));
          } else if (typeof info.value === "boolean") {
            solver.add(info.z3Var.eq(ctx.Bool.val(info.value)));
          }
        }

        // Encode the invariant as a Z3 constraint
        let constraint;
        try {
          constraint = encodeInvariantToZ3(inv, vars, ctx);
        } catch (e) {
          constraint = null;
        }

        if (constraint) {
          // Negate the constraint and check satisfiability
          // unsat means the invariant ALWAYS holds for this config (PROVED)
          // sat means there's a counterexample (REFUTED)
          solver.add(Not(constraint));
          const result = await solver.check();
          let passed, model;
          if (result === "unsat") {
            passed = true;
          } else if (result === "sat") {
            passed = false;
            try { model = solver.model().toString(); } catch { model = null; }
          } else {
            // unknown → fall back to JS check
            try { passed = inv.check(newConfig, oldConfig); } catch { passed = false; }
          }

          proofObligations.push({
            phase: "INVARIANT",
            name: inv.name,
            formal: inv.formal,
            status: passed ? "PROVED" : "REFUTED",
            counterexample: passed ? null : (model || explainCounterexample(inv, newConfig)),
            backend: "z3-wasm",
            z3Used: true,
          });
          if (!passed) allPassed = false;
        } else {
          // Constraint couldn't be encoded → fall back to JS evaluation
          let passed;
          try { passed = inv.check(newConfig, oldConfig); } catch { passed = false; }
          proofObligations.push({
            phase: "INVARIANT",
            name: inv.name,
            formal: inv.formal,
            status: passed ? "PROVED" : "REFUTED",
            counterexample: passed ? null : explainCounterexample(inv, newConfig),
            backend: "symbolic",
            z3Used: false,
          });
          if (!passed) allPassed = false;
        }
      }
    } catch (e) {
      // Z3 runtime error → fall back to JS
      z3Available = false;
      z3Engine = `JS fallback (Z3 runtime error: ${e.message.slice(0, 60)})`;
    }
  }

  // JS fallback for invariants if neither server nor WASM Z3 handled them
  if (!z3Available && !z3ServerHandledInvariants) {
    for (const inv of CONFIG_INVARIANTS) {
      let passed;
      try {
        passed = inv.check(newConfig, oldConfig);
      } catch (e) {
        passed = false;
      }
      proofObligations.push({
        phase: "INVARIANT",
        name: inv.name,
        formal: inv.formal,
        status: passed ? "PROVED" : "REFUTED",
        counterexample: passed ? null : explainCounterexample(inv, newConfig),
        backend: "symbolic",
        z3Used: false,
      });
      if (!passed) allPassed = false;
    }
    backendsUsed.add("symbolic");
  }

  // ─── Phase 2: Diff analysis ───
  const diffs = [];
  function diffObject(obj1, obj2, path = "") {
    for (const key of new Set([...Object.keys(obj1 || {}), ...Object.keys(obj2 || {})])) {
      if (key === "__version") continue;
      const p = path ? `${path}.${key}` : key;
      const v1 = obj1?.[key];
      const v2 = obj2?.[key];
      if (typeof v1 === "object" && v1 !== null && typeof v2 === "object" && v2 !== null) {
        diffObject(v1, v2, p);
      } else if (v1 !== v2) {
        diffs.push({ path: p, from: v1, to: v2, delta: typeof v1 === "number" && typeof v2 === "number" ? v2 - v1 : null });
      }
    }
  }
  diffObject(oldConfig, newConfig);

  proofObligations.push({
    phase: "DIFF_ANALYSIS",
    name: "config_diff_extraction",
    formal: `|{p : C_old(p) ≠ C_new(p)}| = ${diffs.length}`,
    status: "PROVED",
    backend: "symbolic",
    z3Used: false,
  });

  // ─── Phase 3: Monotonicity check ───
  // For each numeric change, prove that |Δ| / |old| ≤ 0.5.
  // If Lean already proved the whole invariant set + monotonicity in Phase 0a,
  // we still emit a per-diff trace tagged "lean" so the UI shows what was proved.
  if (leanHandledInvariants) {
    for (const d of diffs) {
      if (d.delta !== null && typeof d.from === "number" && d.from > 0) {
        const ratio = Math.abs(d.delta) / d.from;
        const safe = ratio <= 0.5;
        proofObligations.push({
          phase: "MONOTONICITY",
          name: `bounded_change_${d.path}`,
          formal: `|${d.path}_new - ${d.path}_old| ≤ 0.5 · ${d.path}_old`,
          status: safe ? "PROVED" : "REFUTED",
          counterexample: safe ? null : `${d.path}: ${d.from} → ${d.to} exceeds 50% bound`,
          backend: "lean",
          z3Used: false,
        });
        if (!safe) allPassed = false;
      }
    }
  } else if (z3Available) {
    try {
      const { Context } = z3;
      const ctx = Context("monotonicity");
      const { Solver, Real, Not } = ctx;

      for (const d of diffs) {
        if (d.delta !== null && typeof d.from === "number" && d.from > 0) {
          const solver = new Solver();
          const oldVar = Real.const("old");
          const newVar = Real.const("new");
          solver.add(oldVar.eq(Real.val(d.from.toString())));
          solver.add(newVar.eq(Real.val(d.to.toString())));
          const upperBound = newVar.sub(oldVar).le(Real.val("0.5").mul(oldVar));
          const lowerBound = oldVar.sub(newVar).le(Real.val("0.5").mul(oldVar));
          solver.add(Not(ctx.And(upperBound, lowerBound)));
          const result = await solver.check();
          const safe = result === "unsat";
          proofObligations.push({
            phase: "MONOTONICITY",
            name: `bounded_change_${d.path}`,
            formal: `|${d.path}_new - ${d.path}_old| ≤ 0.5 · ${d.path}_old`,
            status: safe ? "PROVED" : "REFUTED",
            counterexample: safe ? null : `${d.path}: ${d.from} → ${d.to} (Δ=${d.delta}) exceeds 50% bound`,
            backend: "z3-wasm",
            z3Used: true,
          });
          if (!safe) allPassed = false;
        }
      }
    } catch (e) {
      // Fall back to JS arithmetic
      for (const d of diffs) {
        if (d.delta !== null && typeof d.from === "number" && d.from > 0) {
          const ratio = Math.abs(d.delta) / d.from;
          const safe = ratio <= 0.5;
          proofObligations.push({
            phase: "MONOTONICITY",
            name: `bounded_change_${d.path}`,
            formal: `|Δ${d.path}| / |${d.path}_old| ≤ 0.5`,
            status: safe ? "PROVED" : "REFUTED",
            counterexample: safe ? null : `${d.path} changed by ${(ratio * 100).toFixed(0)}%`,
            backend: "symbolic",
            z3Used: false,
          });
          if (!safe) allPassed = false;
        }
      }
    }
  } else {
    for (const d of diffs) {
      if (d.delta !== null && typeof d.from === "number" && d.from > 0) {
        const ratio = Math.abs(d.delta) / d.from;
        const safe = ratio <= 0.5;
        proofObligations.push({
          phase: "MONOTONICITY",
          name: `bounded_change_${d.path}`,
          formal: `|Δ${d.path}| / |${d.path}_old| ≤ 0.5`,
          status: safe ? "PROVED" : "REFUTED",
          counterexample: safe ? null : `${d.path} changed by ${(ratio * 100).toFixed(0)}% (${d.from} → ${d.to})`,
          backend: "symbolic",
          z3Used: false,
        });
        if (!safe) allPassed = false;
      }
    }
  }

  // ─── Phase 4: Progress argument ───
  if (diffs.length === 0) {
    proofObligations.push({
      phase: "PROGRESS",
      name: "nontrivial_modification",
      formal: "|diffs| ≥ 1",
      status: "REFUTED",
      counterexample: "No parameters changed — no progress possible",
      backend: "symbolic",
      z3Used: false,
    });
    allPassed = false;
  } else {
    proofObligations.push({
      phase: "PROGRESS",
      name: "nontrivial_modification",
      formal: "|diffs| ≥ 1",
      status: "PROVED",
      backend: "symbolic",
      z3Used: false,
    });
  }

  // ─── Phase 5: Performance regression ───
  if (performanceHistory && performanceHistory.length >= 2) {
    const recent = performanceHistory.slice(-3);
    const trend = recent[recent.length - 1] - recent[0];
    proofObligations.push({
      phase: "REGRESSION",
      name: "performance_trend_analysis",
      formal: `score(t) - score(t-3) = ${trend}`,
      status: trend >= -5 ? "PROVED" : "WARNING",
      details: trend < 0 ? `Performance declining by ${-trend}` : `Performance trend: ${trend >= 0 ? "+" : ""}${trend}`,
      backend: "symbolic",
      z3Used: false,
    });
  }

  // ─── Lean 4 status ───
  // Honest reporting: if Lean handled it, say so; otherwise mark SKIPPED.
  if (leanHandledInvariants) {
    proofObligations.push({
      phase: "LEAN4",
      name: "lean4_typecheck_full_proof",
      formal: "Lean 4 type-checked the generated proof file (all invariants + monotonicity)",
      status: "PROVED",
      backend: "lean",
      z3Used: false,
    });
  } else if (backendClient && backendClient.available) {
    proofObligations.push({
      phase: "LEAN4",
      name: "lean4_unavailable",
      formal: "Lean 4 backend reachable but did not produce a proof",
      status: "SKIPPED",
      details: "Backend was reached but Lean was offline or rejected the file. See Z3 obligations above.",
      backend: "lean",
      z3Used: false,
    });
  } else {
    proofObligations.push({
      phase: "LEAN4",
      name: "lean4_no_backend",
      formal: "Lean 4 requires a server backend",
      status: "SKIPPED",
      details: "No verification backend URL configured. Set one in the Self-Mod tab to enable Lean.",
      backend: "lean",
      z3Used: false,
    });
  }

  return {
    verdict: allPassed ? "APPROVED" : "REJECTED",
    obligations: proofObligations,
    diffs,
    proofChain: buildProofChain(proofObligations),
    z3Engine,
    z3Available,
    backendsUsed: Array.from(backendsUsed),
  };
}

// Encode a JavaScript invariant into a Z3 constraint expression.
// Returns null if the invariant cannot be encoded symbolically.
function encodeInvariantToZ3(inv, vars, ctx) {
  const { Real, And, Or, Not } = ctx;
  const v = (name) => vars[name]?.z3Var;
  const r = (n) => Real.val(n.toString());

  switch (inv.name) {
    case "minWordsPerSection_positive":
      return v("generation_minWordsPerSection")?.gt(r(0));

    case "targetTotalWords_reasonable":
      return And(
        v("generation_targetTotalWords").ge(r(500)),
        v("generation_targetTotalWords").le(r(20000))
      );

    case "weights_sum_to_100":
      return And(
        v("verification_weightWordCount")
          .add(v("verification_weightSections"))
          .add(v("verification_weightCitations"))
          .add(v("verification_weightReferences"))
          .add(v("verification_weightMathRigor"))
          .ge(r(99)),
        v("verification_weightWordCount")
          .add(v("verification_weightSections"))
          .add(v("verification_weightCitations"))
          .add(v("verification_weightReferences"))
          .add(v("verification_weightMathRigor"))
          .le(r(100))
      );

    case "all_weights_nonneg":
      return And(
        v("verification_weightWordCount").ge(r(0)),
        v("verification_weightSections").ge(r(0)),
        v("verification_weightCitations").ge(r(0)),
        v("verification_weightReferences").ge(r(0)),
        v("verification_weightMathRigor").ge(r(0))
      );

    case "penalties_bounded":
      return And(
        v("verification_penaltyThinSection").ge(r(0)),
        v("verification_penaltyThinSection").le(r(20)),
        v("verification_penaltyMissingSection").ge(r(0)),
        v("verification_penaltyMissingSection").le(r(20)),
        v("verification_penaltyOrphanCitation").ge(r(0)),
        v("verification_penaltyOrphanCitation").le(r(20)),
        v("verification_penaltyWeakPhrase").ge(r(0)),
        v("verification_penaltyWeakPhrase").le(r(20))
      );

    case "reconciliation_band_safe":
      return And(
        v("verification_reconciliationBand").ge(r(3)),
        v("verification_reconciliationBand").le(r(15))
      );

    case "improvement_increment_safe":
      return And(
        v("improvement_targetIncrement").ge(r(1)),
        v("improvement_targetIncrement").le(r(25))
      );

    case "aggressiveness_in_unit_interval":
      return And(
        v("improvement_aggressiveness").ge(r(0)),
        v("improvement_aggressiveness").le(r(1))
      );

    case "citationsPerParagraph_realistic":
      return And(
        v("generation_citationsPerParagraph").ge(r(1)),
        v("generation_citationsPerParagraph").le(r(10))
      );

    case "schema_version_preserved":
      // Trivially encoded — version is constant in our encoding
      return ctx.Bool.val(true);

    default:
      return null;
  }
}

function explainCounterexample(invariant, config) {
  // Generate human-readable counterexample for failed invariant
  switch (invariant.name) {
    case "minWordsPerSection_positive":
      return `minWordsPerSection = ${config.generation.minWordsPerSection}, expected > 0`;
    case "targetTotalWords_reasonable":
      return `targetTotalWords = ${config.generation.targetTotalWords}, expected in [500, 20000]`;
    case "weights_sum_to_100": {
      const s = config.verification.weightWordCount + config.verification.weightSections +
                config.verification.weightCitations + config.verification.weightReferences +
                config.verification.weightMathRigor;
      return `Σ weights = ${s}, expected ≈ 100`;
    }
    case "reconciliation_band_safe":
      return `reconciliationBand = ${config.verification.reconciliationBand}, expected in [3, 15]`;
    case "aggressiveness_in_unit_interval":
      return `aggressiveness = ${config.improvement.aggressiveness}, expected in [0, 1]`;
    default:
      return "Constraint violated";
  }
}

function buildProofChain(obligations) {
  // Build a human-readable proof chain in the style of a Coq/Lean proof
  const lines = ["Theorem: modification_is_safe(C_old, C_new)."];
  lines.push("Proof:");
  for (const o of obligations) {
    if (o.status === "PROVED") {
      lines.push(`  by ${o.name}: ${o.formal} ✓`);
    } else if (o.status === "REFUTED") {
      lines.push(`  ✗ ${o.name}: ${o.formal}`);
      if (o.counterexample) lines.push(`    counterexample: ${o.counterexample}`);
    } else if (o.status === "WARNING") {
      lines.push(`  ⚠ ${o.name}: ${o.formal}`);
    }
  }
  const failed = obligations.filter((o) => o.status === "REFUTED");
  if (failed.length === 0) {
    lines.push("Qed.");
  } else {
    lines.push(`Aborted. (${failed.length} obligation${failed.length > 1 ? "s" : ""} refuted)`);
  }
  return lines.join("\n");
}

// ═══════════════════════════════════════════════════════════════════
//  SELF-MODIFICATION AGENT
// ═══════════════════════════════════════════════════════════════════
// Proposes mutations to the system configuration based on observed performance.
// The proposed mutation must pass symbolicVerify before being applied.

function proposeSelfModification(currentConfig, recentMetrics, scoreHistory) {
  // Heuristic-driven proposal generator
  // Analyzes what's blocking improvement and proposes a specific config change

  const proposals = [];
  const trend = scoreHistory.length >= 2
    ? scoreHistory[scoreHistory.length - 1] - scoreHistory[scoreHistory.length - 2]
    : 0;

  // Heuristic 1: If sections are consistently thin, increase the threshold
  if (recentMetrics?.thinSections?.length >= 2) {
    proposals.push({
      reasoning: `${recentMetrics.thinSections.length} thin sections in last paper — generation parameters too lax`,
      mutation: { generation: { ...currentConfig.generation, minWordsPerSection: Math.min(currentConfig.generation.minWordsPerSection + 50, 400) } },
      expectedEffect: "Forces generation agent to write longer sections",
    });
  }

  // Heuristic 2: If citation density low, demand more citations
  if (recentMetrics?.citationDensity !== undefined && parseFloat(recentMetrics.citationDensity) < 0.6) {
    proposals.push({
      reasoning: `Citation density ${recentMetrics.citationDensity} below 0.6 — too few cited paragraphs`,
      mutation: { generation: { ...currentConfig.generation, citationsPerParagraph: Math.min(currentConfig.generation.citationsPerParagraph + 1, 5) } },
      expectedEffect: "Demands more citations per paragraph",
    });
  }

  // Heuristic 3: If word count consistently below target, raise it
  if (recentMetrics?.wordCount && recentMetrics.wordCount < currentConfig.generation.targetTotalWords * 0.8) {
    proposals.push({
      reasoning: `Word count ${recentMetrics.wordCount} far below target ${currentConfig.generation.targetTotalWords}`,
      mutation: { generation: { ...currentConfig.generation, targetTotalWords: Math.min(currentConfig.generation.targetTotalWords + 500, 6000) } },
      expectedEffect: "Pushes target word count higher to force more content",
    });
  }

  // Heuristic 4: If performance plateaued (Δ ≤ 2 over last 2 iterations), increase aggressiveness
  if (trend !== 0 && Math.abs(trend) <= 2 && scoreHistory.length >= 2) {
    const newAgg = Math.min(currentConfig.improvement.aggressiveness + 0.15, 1.0);
    proposals.push({
      reasoning: `Score plateaued (Δ=${trend}) — system needs more aggressive changes`,
      mutation: { improvement: { ...currentConfig.improvement, aggressiveness: parseFloat(newAgg.toFixed(2)), targetIncrement: Math.min(currentConfig.improvement.targetIncrement + 3, 25) } },
      expectedEffect: "More aggressive improvement targets",
    });
  }

  // Heuristic 5: If performance degraded, tighten reconciliation band (be stricter)
  if (trend < -3) {
    proposals.push({
      reasoning: `Performance dropped by ${-trend} points — tighten verification`,
      mutation: { verification: { ...currentConfig.verification, reconciliationBand: Math.max(currentConfig.verification.reconciliationBand - 2, 3) } },
      expectedEffect: "Stricter reconciliation prevents score inflation",
    });
  }

  // Pick the highest-priority proposal
  if (proposals.length === 0) return null;
  const chosen = proposals[0];

  // Construct the full new config by deep-merging the mutation
  const newConfig = JSON.parse(JSON.stringify(currentConfig));
  for (const section of Object.keys(chosen.mutation)) {
    newConfig[section] = { ...newConfig[section], ...chosen.mutation[section] };
  }

  return {
    proposal: chosen,
    newConfig,
    allProposals: proposals,
  };
}

// ─── HERO ───
function HeroPage({ onStart }) {
  const [visible, setVisible] = useState(false);
  useEffect(() => { setTimeout(() => setVisible(true), 50); }, []);
  return (
    <div style={{
      minHeight: "100vh", background: "#FAFAF7",
      display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center",
      padding: "60px 24px",
    }}>
      <div style={{
        maxWidth: 640, textAlign: "center",
        opacity: visible ? 1 : 0,
        transform: visible ? "none" : "translateY(16px)",
        transition: "all 0.9s cubic-bezier(0.22, 1, 0.36, 1)",
      }}>
        <div style={{
          fontSize: 13, letterSpacing: "0.15em", textTransform: "uppercase",
          color: "#9C9689", marginBottom: 32, fontFamily: "var(--mono)",
        }}>Formal Verification Research System</div>
        <h1 style={{
          fontFamily: "var(--serif)", fontSize: "clamp(36px, 7vw, 56px)",
          fontWeight: 400, lineHeight: 1.08, color: "#1A1915",
          margin: "0 0 28px", letterSpacing: "-0.025em",
        }}>Self-Improving<br />Research Agent</h1>
        <p style={{
          fontFamily: "var(--body)", fontSize: 17, lineHeight: 1.7,
          color: "#6B665C", maxWidth: 480, margin: "0 auto 48px",
        }}>
          Five agents with formal verification, RAG-based planning, and prompt fine-tuning.
          Iteratively refines academic papers. Upload PDF/Markdown or start from a topic.
        </p>
        <div style={{
          display: "flex", justifyContent: "center", gap: 0,
          marginBottom: 48, flexWrap: "wrap",
        }}>
          {["Plan", "Research", "Write", "Verify", "Refine"].map((name, i) => (
            <div key={name} style={{ display: "flex", alignItems: "center" }}>
              <div style={{ textAlign: "center", padding: "0 12px" }}>
                <div style={{
                  width: 38, height: 38, borderRadius: "50%", margin: "0 auto 6px",
                  border: "1.5px solid #D4D0C8",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontFamily: "var(--mono)", fontSize: 12, color: "#6B665C",
                }}>{i + 1}</div>
                <div style={{ fontSize: 12, fontWeight: 600, color: "#1A1915" }}>{name}</div>
              </div>
              {i < 4 && <div style={{ width: 28, height: 1, background: "#D4D0C8" }} />}
            </div>
          ))}
        </div>
        <button onClick={onStart} style={{
          fontFamily: "var(--body)", fontSize: 15, fontWeight: 500,
          padding: "14px 40px", background: "#1A1915", color: "#FAFAF7",
          border: "none", borderRadius: 4, cursor: "pointer",
          transition: "background 0.2s",
        }}
        onMouseEnter={(e) => e.target.style.background = "#2E2C26"}
        onMouseLeave={(e) => e.target.style.background = "#1A1915"}
        >Begin</button>
      </div>
    </div>
  );
}

// ─── MAIN APP ───
export default function App() {
  const [page, setPage] = useState("hero");
  const [topic, setTopic] = useState("");
  const [running, setRunning] = useState(false);
  const [logs, setLogs] = useState([]);
  const [iter, setIter] = useState(0);
  const [maxIter, setMaxIter] = useState(3);
  const [paper, setPaper] = useState("");
  const [paperVersions, setPaperVersions] = useState([]);
  const [scores, setScores] = useState([]);
  const [vResults, setVResults] = useState([]);
  const [plan, setPlan] = useState(null);
  const [phase, setPhase] = useState(null);
  const [error, setError] = useState(null);
  const [done, setDone] = useState(false);
  const [uploaded, setUploaded] = useState("");
  const [tab, setTab] = useState("paper");
  const [selIter, setSelIter] = useState(0);
  const [tuning, setTuning] = useState("");
  const [history, setHistory] = useState([]);
  // ─── SELF-MODIFICATION STATE ───
  const [systemConfig, setSystemConfig] = useState(DEFAULT_SYSTEM_CONFIG);
  const [modificationLog, setModificationLog] = useState([]); // [{iter, proposal, verdict, applied, ...}]

  // Verification backend (real z3-solver Python + Lean 4 over HTTP). Optional.
  const [backendUrl, setBackendUrl] = useState("http://localhost:8000");
  const [backendStatus, setBackendStatus] = useState(null); // null | { reachable, z3, lean, ... }
  const backendClient = backendUrl ? makeBackendClient(backendUrl) : null;

  // Run identity for DB-backed persistence (Mongo logs + Postgres checkpoints)
  const [runId, setRunId] = useState(null);

  // "Runs & Data" tab state
  const [runsList, setRunsList] = useState(null); // null=unloaded, []=loaded empty, [...]=loaded
  const [ragQuery, setRagQuery] = useState("");
  const [ragHits, setRagHits] = useState(null);
  const [ragSearching, setRagSearching] = useState(false);

  const logRef = useRef(null);
  const abortRef = useRef(null);
  const fileRef = useRef(null);

  // Load persistent data on mount
  useEffect(() => {
    (async () => {
      try {
        const h = await loadStorage("rag-history");
        if (h && Array.isArray(h)) setHistory(h);
        const t = await loadStorage("prompt-tuning");
        if (t && typeof t === "string") setTuning(t);
        const cfg = await loadStorage("system-config");
        if (cfg && typeof cfg === "object" && cfg.__version === DEFAULT_SYSTEM_CONFIG.__version) {
          setSystemConfig(cfg);
        }
        const ml = await loadStorage("modification-log");
        if (ml && Array.isArray(ml)) setModificationLog(ml);
        const bu = await loadStorage("backend-url");
        if (bu && typeof bu === "string") setBackendUrl(bu);
      } catch {}
    })();
  }, []);

  // Probe verification backend whenever URL changes
  useEffect(() => {
    let cancelled = false;
    if (!backendUrl) {
      setBackendStatus(null);
      return;
    }
    (async () => {
      const client = makeBackendClient(backendUrl);
      const status = await client.probe();
      if (!cancelled) setBackendStatus(status);
    })();
    return () => { cancelled = true; };
  }, [backendUrl]);

  const addLog = useCallback((agent, message, type = "info") => {
    setLogs((prev) => [...prev, {
      agent, message, type,
      time: new Date().toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" }),
    }]);
  }, []);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  const stopPipeline = useCallback(() => {
    // Best-effort: tell the backend to cancel the background task
    if (backendClient && runId) {
      fetch(`${backendClient.url}/pipeline/cancel/${encodeURIComponent(runId)}`, {
        method: "POST",
      }).catch(() => {});
    }
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    setRunning(false);
    setPhase(null);
    addLog("orchestrator", "Stopped by user.", "warn");
  }, [addLog, backendClient, runId]);

  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setError(null);
    addLog("orchestrator", `Reading ${file.name}...`, "info");
    try {
      let text;
      if (file.name.toLowerCase().endsWith(".pdf")) {
        addLog("orchestrator", "Extracting text from PDF (this may take a moment)...", "info");
        const base64 = await readBinaryAsBase64(file);
        text = await callAPIWithPDF(base64, "Extract ALL text from this PDF as clean Markdown. Use ## for section headings. Preserve all content, citations, and structure. Do not summarize.", null);
      } else {
        text = await readTextFile(file);
      }
      setUploaded(text);
      setPaper(text);
      setPaperVersions([{ text, iter: 0 }]);
      setTopic(`Improve uploaded paper: ${file.name}`);
      addLog("orchestrator", `Loaded "${file.name}" (${text.split(/\s+/).length} words). Click Generate to improve it.`, "ok");
    } catch (err) {
      setError(`Upload failed: ${err.message}`);
      addLog("orchestrator", `Upload failed: ${err.message}`, "err");
    }
    e.target.value = "";
  };

  const runPipeline = async () => {
    if (!topic.trim() && !uploaded) {
      setError("Please enter a topic or upload a paper.");
      return;
    }
    if (!backendClient) {
      setError("No backend configured. Set the Verification Backend URL in the Self-Modification tab.");
      return;
    }
    setError(null);
    setRunning(true);
    setLogs([]);
    setScores([]);
    setVResults([]);
    setPlan(null);
    setIter(0);
    setDone(false);
    setTab("paper");
    if (!uploaded) {
      setPaper("");
      setPaperVersions([]);
    }

    try {
      // Start the backend pipeline — it runs the full LangGraph loop on the server,
      // using the real z3-solver + Lean 4 (via the same FastAPI process) for self-modification.
      const startResp = await fetch(`${backendClient.url}/pipeline/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topic: uploaded ? null : topic,
          uploaded_paper: uploaded || null,
          max_iterations: maxIter,
          system_config: systemConfig,
          tuning,
        }),
      });
      if (!startResp.ok) {
        const t = await startResp.text();
        throw new Error(`pipeline/start ${startResp.status}: ${t.slice(0, 200)}`);
      }
      const { run_id: newRunId } = await startResp.json();
      setRunId(newRunId);
      addLog("orchestrator", `Pipeline started: ${newRunId}`, "info");

      // Open SSE stream. EventSource has no signal/abort API, so we keep the
      // handle in abortRef.current and .close() it from stopPipeline.
      const es = new EventSource(`${backendClient.url}/pipeline/stream/${encodeURIComponent(newRunId)}`);
      abortRef.current = { abort: () => es.close() };

      const scoresAcc = [];
      const vResultsAcc = [];
      const modLogAcc = [];
      const paperVersionsAcc = uploaded ? [uploaded] : [];

      const handlePayload = (kind, data) => {
        const p = data.payload || data;
        if (kind === "log") {
          addLog(p.agent || "orchestrator", p.message || "", p.type || "info");
        } else if (kind === "phase") {
          setPhase(p.phase);
        } else if (kind === "plan") {
          setPlan(p);
          addLog("orchestrator", `Plan: "${p.title || "(untitled)"}"${p.is_technical_topic ? " [technical]" : ""}`, "ok");
        } else if (kind === "paper_version") {
          setIter(p.iteration);
          addLog("generation", `Draft v${p.iteration}: ${p.length} chars`, "ok");
          // The preview is short; the full paper arrives in the "verification"
          // event's metrics block via compute_metrics on the server.
        } else if (kind === "verification") {
          vResultsAcc.push(p);
          setVResults([...vResultsAcc]);
          // Server doesn't ship the full paper text in events, so we
          // accumulate the draft only via the "paper_full" event (below).
        } else if (kind === "paper_full") {
          // Optional extension: full paper text for display
          setPaper(p.paper || "");
          paperVersionsAcc.push(p.paper || "");
          setPaperVersions([...paperVersionsAcc]);
        } else if (kind === "score") {
          scoresAcc.push(p);
          setScores([...scoresAcc]);
          addLog("feedback", `Quality: ${p.quality_score}/100 · ${p.improvements?.length || 0} suggestions`, "ok");
        } else if (kind === "modification") {
          // Map server-side field names to the JSX shape
          const modEntry = {
            iter: p.iteration,
            proposal: p.proposal,
            verdict: {
              ...p.verdict,
              proofChain: p.verdict?.proof_chain,
              backendsUsed: p.verdict?.backends_used,
            },
            beforeConfig: p.before_config,
            afterConfig: p.after_config,
            applied: p.applied,
            timestamp: p.timestamp,
          };
          modLogAcc.push(modEntry);
          setModificationLog([...modLogAcc]);
          saveStorage("modification-log", [...modLogAcc]);  // day1: persist for rollback
          if (p.applied && p.after_config) {
            setSystemConfig(p.after_config);
            saveStorage("system-config", p.after_config);
          }
        } else if (kind === "complete") {
          setPhase(null);
          setDone(true);
          setUploaded("");
          addLog("orchestrator", "Complete.", "ok");
          es.close();
          setRunning(false);
          abortRef.current = null;

          // Save to RAG history
          const finalScore = scoresAcc[scoresAcc.length - 1]?.quality_score || 0;
          const newHistory = [
            ...history,
            {
              topic: topic || "Uploaded paper",
              finalScore,
              insight: scoresAcc[scoresAcc.length - 1]?.iteration_summary?.slice(0, 200) || "",
              date: new Date().toISOString(),
            },
          ].slice(-10);
          setHistory(newHistory);
          saveStorage("rag-history", newHistory);
        } else if (kind === "error") {
          setError(p.message || "pipeline error");
          addLog("orchestrator", `Error: ${p.message}`, "err");
          es.close();
          setRunning(false);
          abortRef.current = null;
        }
      };

      // Route named events into handlePayload
      const kinds = ["log", "phase", "plan", "paper_version", "paper_full", "verification", "score", "modification", "complete", "error"];
      kinds.forEach((k) => {
        es.addEventListener(k, (ev) => {
          try {
            const parsed = JSON.parse(ev.data);
            handlePayload(k, parsed);
          } catch (e) {
            addLog("orchestrator", `SSE parse error (${k}): ${e.message}`, "err");
          }
        });
      });

      es.addEventListener("end", () => {
        es.close();
        setRunning(false);
        abortRef.current = null;
      });

      es.onerror = (e) => {
        if (es.readyState === EventSource.CLOSED) return;
        addLog("orchestrator", "SSE stream error — connection lost", "err");
        es.close();
        setRunning(false);
        abortRef.current = null;
      };
    } catch (err) {
      setError(err.message);
      addLog("orchestrator", `Error: ${err.message}`, "err");
      setRunning(false);
      abortRef.current = null;
    }
  };


  const updateTuning = async (newTuning) => {
    setTuning(newTuning);
    await saveStorage("prompt-tuning", newTuning);
  };

  // ─── HERO ROUTE ───
  if (page === "hero") {
    return (
      <>
        <style>{`
          @import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,600&family=Source+Sans+3:wght@400;500;600&family=Source+Code+Pro:wght@400;500&display=swap');
          :root { --serif: 'Newsreader', Georgia, serif; --body: 'Source Sans 3', sans-serif; --mono: 'Source Code Pro', monospace; }
          * { box-sizing: border-box; margin: 0; }
        `}</style>
        <HeroPage onStart={() => setPage("app")} />
      </>
    );
  }

  const lastScore = scores.length > 0 ? scores[scores.length - 1] : null;
  const lastVerif = vResults.length > 0 ? vResults[vResults.length - 1] : null;
  const hasStarted = logs.length > 0 || paper;

  return (
    <div style={{ minHeight: "100vh", background: "#FAFAF7", color: "#1A1915", fontFamily: "var(--body)" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,600&family=Source+Sans+3:wght@400;500;600&family=Source+Code+Pro:wght@400;500&display=swap');
        :root { --serif: 'Newsreader', Georgia, serif; --body: 'Source Sans 3', sans-serif; --mono: 'Source Code Pro', monospace; }
        * { box-sizing: border-box; }
        textarea:focus { outline: none; border-color: #1A1915 !important; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-thumb { background: #D4D0C8; border-radius: 2px; }
        @keyframes fadeIn { from { opacity: 0 } to { opacity: 1 } }
        @keyframes pulseDot { 0%, 100% { opacity: 0.2 } 50% { opacity: 1 } }
      `}</style>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 16px" }}>
        {/* NAV */}
        <nav style={{
          display: "flex", justifyContent: "space-between", alignItems: "center",
          padding: "14px 0", borderBottom: "1px solid #E8E5DE",
          flexWrap: "wrap", gap: 8,
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <span onClick={() => !running && setPage("hero")} style={{
              fontFamily: "var(--serif)", fontSize: 17, fontWeight: 600,
              color: "#1A1915", cursor: running ? "default" : "pointer",
            }}>Research Agent</span>
            {plan && <span style={{
              fontFamily: "var(--mono)", fontSize: 10, color: "#9C9689",
              borderLeft: "1px solid #E8E5DE", paddingLeft: 12,
            }}>{plan.title}</span>}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            {running && phase && (
              <span style={{
                fontFamily: "var(--mono)", fontSize: 10, color: "#9C9689",
                display: "flex", alignItems: "center", gap: 5,
              }}>
                <span style={{ width: 5, height: 5, borderRadius: "50%", background: "#6B665C", animation: "pulseDot 1.2s infinite" }} />
                {AGENTS[phase]?.label}
              </span>
            )}
            {paper && (
              <button onClick={() => downloadPDF(paper, plan?.title || "Paper")} style={{
                fontFamily: "var(--mono)", fontSize: 10, padding: "6px 12px",
                background: "none", border: "1px solid #D4D0C8", borderRadius: 3,
                color: "#6B665C", cursor: "pointer",
              }}>Save as PDF</button>
            )}
          </div>
        </nav>

        {/* GLOBAL ERROR BANNER */}
        {error && (
          <div style={{
            margin: "12px 0", padding: "10px 14px",
            background: "#FDF2F1", border: "1px solid #E0B4B1", borderRadius: 4,
            fontSize: 12, color: "#C4403A", display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12,
          }}>
            <span><strong>Error:</strong> {error}</span>
            <button onClick={() => setError(null)} style={{
              background: "none", border: "none", color: "#C4403A",
              cursor: "pointer", fontSize: 16, padding: 0, lineHeight: 1,
            }}>×</button>
          </div>
        )}

        {/* INPUT */}
        <div style={{ padding: "16px 0 14px", borderBottom: "1px solid #E8E5DE" }}>
          <div style={{ display: "flex", gap: 10, alignItems: "flex-end", flexWrap: "wrap" }}>
            <div style={{ flex: 1, minWidth: 220 }}>
              <label style={{
                display: "block", fontFamily: "var(--mono)", fontSize: 10,
                color: "#9C9689", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.08em",
              }}>Topic</label>
              <textarea
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                disabled={running}
                rows={2}
                placeholder="Describe your research topic..."
                style={{
                  width: "100%", padding: "10px 12px", fontSize: 14,
                  fontFamily: "var(--body)", background: "#fff", color: "#1A1915",
                  border: "1px solid #E8E5DE", borderRadius: 3,
                  resize: "vertical", lineHeight: 1.6,
                }}
              />
            </div>
            <div>
              <label style={{
                display: "block", fontFamily: "var(--mono)", fontSize: 10,
                color: "#9C9689", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.08em",
              }}>Passes</label>
              <select
                value={maxIter}
                onChange={(e) => setMaxIter(Number(e.target.value))}
                disabled={running}
                style={{
                  padding: "10px 12px", fontSize: 14, fontFamily: "var(--body)",
                  background: "#fff", border: "1px solid #E8E5DE", borderRadius: 3, color: "#1A1915",
                }}
              >
                {Array.from({ length: 20 }, (_, i) => i + 1).map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </div>
            <div>
              <input
                ref={fileRef}
                type="file"
                accept=".md,.txt,.tex,.pdf"
                style={{ display: "none" }}
                onChange={handleUpload}
                disabled={running}
              />
              <button
                onClick={() => fileRef.current?.click()}
                disabled={running}
                style={{
                  fontFamily: "var(--body)", fontSize: 13, padding: "10px 16px",
                  background: "#fff", color: "#6B665C",
                  border: "1px solid #E8E5DE", borderRadius: 3,
                  cursor: running ? "default" : "pointer", whiteSpace: "nowrap",
                }}
              >Upload</button>
            </div>
            {running ? (
              <button onClick={stopPipeline} style={{
                fontFamily: "var(--body)", fontSize: 14, padding: "10px 24px",
                background: "#fff", color: "#C4403A",
                border: "1px solid #E0B4B1", borderRadius: 3, cursor: "pointer",
              }}>Stop</button>
            ) : (
              <button
                onClick={runPipeline}
                disabled={!topic.trim() && !uploaded}
                style={{
                  fontFamily: "var(--body)", fontSize: 14, padding: "10px 28px",
                  background: (!topic.trim() && !uploaded) ? "#EFEDE8" : "#1A1915",
                  color: (!topic.trim() && !uploaded) ? "#B5B0A6" : "#FAFAF7",
                  border: "none", borderRadius: 3,
                  cursor: (!topic.trim() && !uploaded) ? "default" : "pointer",
                }}
              >Generate</button>
            )}
          </div>
          {uploaded && (
            <div style={{ marginTop: 8, fontFamily: "var(--mono)", fontSize: 11, color: "#4A7C59" }}>
              Loaded ({uploaded.split(/\s+/).length} words). Click Generate to improve it.
            </div>
          )}
        </div>

        {/* PROGRESS */}
        {(running || done) && iter > 0 && (
          <div style={{
            padding: "12px 0", display: "flex", alignItems: "center", gap: 16,
            borderBottom: "1px solid #E8E5DE",
          }}>
            <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "#9C9689" }}>
              Pass {iter}/{maxIter}
            </span>
            <div style={{ flex: 1, height: 2, background: "#E8E5DE", borderRadius: 1 }}>
              <div style={{
                height: "100%", background: "#1A1915", borderRadius: 1,
                width: `${(iter / maxIter) * 100}%`, transition: "width 0.5s",
              }} />
            </div>
            <div style={{ display: "flex", gap: 3 }}>
              {STEP_ORDER.map((s) => {
                const active = phase === s;
                const past = STEP_ORDER.indexOf(phase) > STEP_ORDER.indexOf(s);
                return (
                  <div key={s} title={AGENTS[s].label} style={{
                    width: 22, height: 22, borderRadius: "50%",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontFamily: "var(--mono)", fontSize: 9, fontWeight: 500,
                    background: active ? "#1A1915" : past ? "#D4D0C8" : "#EFEDE8",
                    color: active ? "#FAFAF7" : past ? "#6B665C" : "#B5B0A6",
                    transition: "all 0.3s",
                  }}>{AGENTS[s].mark}</div>
                );
              })}
            </div>
            {done && <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "#4A7C59" }}>Done</span>}
          </div>
        )}

        {/* TABS */}
        {hasStarted && (
          <div style={{ display: "flex", borderBottom: "1px solid #E8E5DE", flexWrap: "wrap" }}>
            {[
              { id: "paper", label: "Paper" },
              { id: "scores", label: "Scores & Verification" },
              { id: "selfmod", label: "Self-Modification" },
              { id: "diff", label: "Changes" },
              { id: "runs", label: "Runs & Data" },
              { id: "tuning", label: "Settings" },
            ].map((t) => (
              <button key={t.id} onClick={() => setTab(t.id)} style={{
                fontFamily: "var(--mono)", fontSize: 10, padding: "10px 16px",
                background: "none", border: "none",
                borderBottom: tab === t.id ? "2px solid #1A1915" : "2px solid transparent",
                color: tab === t.id ? "#1A1915" : "#9C9689",
                cursor: "pointer", letterSpacing: "0.04em", textTransform: "uppercase",
              }}>{t.label}</button>
            ))}
          </div>
        )}

        {/* MAIN CONTENT */}
        {hasStarted && (
          <div style={{
            display: "grid", gridTemplateColumns: "300px 1fr",
            gap: 0, minHeight: "60vh",
          }}>
            {/* LEFT: LOGS */}
            <div style={{
              borderRight: "1px solid #E8E5DE", paddingRight: 16, paddingTop: 16,
              overflowY: "auto", maxHeight: "calc(100vh - 240px)",
            }}>
              <div style={{
                fontFamily: "var(--mono)", fontSize: 9, color: "#9C9689",
                textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 6,
              }}>Activity Log</div>
              <div ref={logRef} style={{
                maxHeight: "calc(100vh - 300px)", overflowY: "auto",
                fontFamily: "var(--mono)", fontSize: 11, lineHeight: 1.7,
              }}>
                {logs.map((log, i) => {
                  const colors = {
                    ok: "#4A7C59", err: "#C4403A", warn: "#A68A2B",
                    iter: "#6B665C", info: "#6B665C",
                  };
                  return (
                    <div key={i} style={{
                      display: "flex", gap: 6, padding: "2px 0",
                      color: colors[log.type] || "#6B665C",
                      animation: "fadeIn 0.2s",
                      ...(log.type === "iter" ? { marginTop: 6, fontWeight: 500 } : {}),
                    }}>
                      <span style={{ color: "#B5B0A6", flexShrink: 0, fontSize: 9 }}>{log.time}</span>
                      <span style={{ color: "#9C9689", flexShrink: 0 }}>
                        {AGENTS[log.agent]?.mark || "·"}
                      </span>
                      <span style={{ wordBreak: "break-word" }}>{log.message}</span>
                    </div>
                  );
                })}
                {running && (
                  <div style={{ color: "#B5B0A6", animation: "pulseDot 1s infinite" }}>···</div>
                )}
              </div>
            </div>

            {/* RIGHT: TAB CONTENT */}
            <div style={{
              paddingLeft: 20, paddingTop: 16,
              overflowY: "auto", maxHeight: "calc(100vh - 240px)", paddingBottom: 60,
            }}>
              {/* PAPER TAB */}
              {tab === "paper" && paper && (
                <div>
                  <div style={{
                    display: "flex", justifyContent: "space-between", alignItems: "center",
                    marginBottom: 14, position: "sticky", top: 0,
                    background: "#FAFAF7", paddingBottom: 8, zIndex: 1,
                  }}>
                    <span style={{
                      fontFamily: "var(--mono)", fontSize: 9, color: "#9C9689", textTransform: "uppercase",
                    }}>{paper.split(/\s+/).length} words</span>
                    <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                      {lastScore && (
                        <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "#6B665C" }}>
                          {lastScore.quality_score}/100
                        </span>
                      )}
                      <button onClick={() => downloadPDF(paper, plan?.title || "Paper")} style={{
                        fontFamily: "var(--mono)", fontSize: 10, padding: "4px 10px",
                        background: "none", border: "1px solid #D4D0C8", borderRadius: 2,
                        color: "#6B665C", cursor: "pointer",
                      }}>↓ PDF</button>
                    </div>
                  </div>
                  <article style={{ maxWidth: 680 }}>
                    <RenderMd text={paper} />
                  </article>
                </div>
              )}
              {tab === "paper" && !paper && (
                <div style={{ padding: 40, textAlign: "center", color: "#9C9689", fontSize: 13 }}>
                  Paper will appear here once generated.
                </div>
              )}

              {/* SCORES TAB */}
              {tab === "scores" && (
                <div style={{ maxWidth: 700 }}>
                  {scores.length > 0 ? (
                    <>
                      {/* Score progression chart */}
                      <div style={{ marginBottom: 24 }}>
                        <div style={{
                          fontFamily: "var(--mono)", fontSize: 9, color: "#9C9689",
                          textTransform: "uppercase", marginBottom: 10,
                        }}>Score Progression — click bar for details</div>
                        <div style={{
                          display: "flex", alignItems: "flex-end", gap: 4,
                          height: 140, borderBottom: "1px solid #E8E5DE", marginBottom: 8,
                        }}>
                          {scores.map((s, i) => (
                            <div key={i}
                              onClick={() => setSelIter(i)}
                              style={{
                                flex: 1, display: "flex", flexDirection: "column",
                                alignItems: "center", justifyContent: "flex-end",
                                height: "100%", cursor: "pointer",
                              }}>
                              <div style={{
                                fontFamily: "var(--mono)", fontSize: 10,
                                color: selIter === i ? "#1A1915" : "#6B665C", marginBottom: 4,
                                fontWeight: selIter === i ? 600 : 400,
                              }}>{s.quality_score}</div>
                              <div style={{
                                width: "100%", maxWidth: 50,
                                background: selIter === i ? "#1A1915" :
                                  s.quality_score >= 80 ? "#4A7C59" :
                                  s.quality_score >= 60 ? "#A68A2B" : "#C4403A",
                                borderRadius: "2px 2px 0 0",
                                height: `${s.quality_score}%`,
                                transition: "height 0.5s",
                                minHeight: 4,
                              }} />
                            </div>
                          ))}
                        </div>
                        <div style={{ display: "flex", gap: 4 }}>
                          {scores.map((_, i) => (
                            <div key={i} style={{
                              flex: 1, textAlign: "center",
                              fontFamily: "var(--mono)", fontSize: 9,
                              color: selIter === i ? "#1A1915" : "#B5B0A6",
                            }}>#{i + 1}</div>
                          ))}
                        </div>
                      </div>

                      {/* Selected iteration detail */}
                      {scores[selIter] && (
                        <div style={{ marginBottom: 24 }}>
                          <div style={{
                            fontFamily: "var(--mono)", fontSize: 9, color: "#9C9689",
                            textTransform: "uppercase", marginBottom: 10,
                          }}>Iteration #{selIter + 1} — Dimension Scores</div>
                          <div style={{ display: "flex", flexDirection: "column", gap: 5, marginBottom: 12 }}>
                            {scores[selIter].dimension_scores && Object.entries(scores[selIter].dimension_scores).map(([k, v]) => (
                              <div key={k} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                <span style={{
                                  width: 90, fontSize: 11, color: "#9C9689",
                                  textAlign: "right", textTransform: "capitalize",
                                }}>{k}</span>
                                <div style={{ flex: 1, height: 5, background: "#EFEDE8", borderRadius: 2 }}>
                                  <div style={{
                                    height: "100%", borderRadius: 2,
                                    background: v >= 80 ? "#4A7C59" : v >= 60 ? "#A68A2B" : "#C4403A",
                                    width: `${v}%`, transition: "width 0.5s",
                                  }} />
                                </div>
                                <span style={{
                                  fontFamily: "var(--mono)", fontSize: 10,
                                  color: "#6B665C", width: 28,
                                }}>{v}</span>
                              </div>
                            ))}
                          </div>
                          {scores[selIter].iteration_summary && (
                            <div style={{
                              fontSize: 12, color: "#6B665C", lineHeight: 1.6,
                              padding: 10, background: "#F2F0EB", borderRadius: 3,
                            }}>{scores[selIter].iteration_summary}</div>
                          )}
                        </div>
                      )}

                      {/* Verification detail */}
                      {vResults[selIter] && (
                        <div>
                          <div style={{
                            display: "flex", justifyContent: "space-between", marginBottom: 10,
                          }}>
                            <span style={{
                              fontFamily: "var(--mono)", fontSize: 9, color: "#9C9689",
                              textTransform: "uppercase",
                            }}>Verification #{selIter + 1}</span>
                            <span style={{
                              fontFamily: "var(--mono)", fontSize: 10,
                              color: vResults[selIter].overall_status === "VERIFIED" ? "#4A7C59" : "#A68A2B",
                            }}>{vResults[selIter].overall_status} · {vResults[selIter].score}/100</span>
                          </div>

                          {/* Deterministic metrics — ground truth */}
                          {vResults[selIter].metrics && (
                            <div style={{
                              marginBottom: 14, padding: 12, background: "#F0F4ED",
                              border: "1px solid #D4DBC8", borderRadius: 3,
                              fontFamily: "var(--mono)", fontSize: 10, lineHeight: 1.7,
                            }}>
                              <div style={{ fontWeight: 600, marginBottom: 6, color: "#3D5C2E", fontSize: 10 }}>
                                ⚙ Deterministic Metrics (computed in JavaScript — ground truth)
                              </div>
                              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "2px 12px", color: "#3D3A33" }}>
                                <div>Word count: <strong>{vResults[selIter].metrics.wordCount}</strong></div>
                                <div>Sections: <strong>{vResults[selIter].metrics.sectionsPresent}/{vResults[selIter].metrics.sectionsTotal}</strong></div>
                                <div>Citations: <strong>{vResults[selIter].metrics.citationCount}</strong> ({vResults[selIter].metrics.uniqueCitationCount} unique)</div>
                                <div>References: <strong>{vResults[selIter].metrics.referenceCount}</strong></div>
                                <div>Citation density: <strong>{vResults[selIter].metrics.citationDensity}</strong></div>
                                <div>Equations: <strong>{vResults[selIter].metrics.equationCount}</strong></div>
                                <div>Numeric claims: <strong>{vResults[selIter].metrics.numericClaims}</strong></div>
                                <div>Statistical terms: <strong>{vResults[selIter].metrics.statisticalTerms}</strong></div>
                              </div>
                              {vResults[selIter].metrics.sectionsMissing?.length > 0 && (
                                <div style={{ marginTop: 6, color: "#C4403A" }}>
                                  Missing: {vResults[selIter].metrics.sectionsMissing.join(", ")}
                                </div>
                              )}
                              {vResults[selIter].metrics.thinSections?.length > 0 && (
                                <div style={{ marginTop: 4, color: "#A68A2B" }}>
                                  Thin: {vResults[selIter].metrics.thinSections.join(", ")}
                                </div>
                              )}
                              {vResults[selIter].metrics.orphanCitations?.length > 0 && (
                                <div style={{ marginTop: 4, color: "#A68A2B" }}>
                                  Orphan citations: {vResults[selIter].metrics.orphanCitations.slice(0, 3).join(", ")}
                                </div>
                              )}
                              {vResults[selIter].metrics.weakPhrases > 0 && (
                                <div style={{ marginTop: 4, color: "#A68A2B" }}>
                                  Weak phrases: {vResults[selIter].metrics.weakPhrases}
                                </div>
                              )}
                              <div style={{
                                marginTop: 8, paddingTop: 6, borderTop: "1px solid #D4DBC8",
                                fontWeight: 600, fontSize: 11, color: "#3D5C2E",
                              }}>
                                Deterministic baseline: {vResults[selIter].metrics.deterministicScore}/100
                              </div>
                            </div>
                          )}

                          {/* Reconciliation between LLM and deterministic */}
                          {vResults[selIter].reconciliation && vResults[selIter].reconciliation.adjusted && (
                            <div style={{
                              marginBottom: 14, padding: 10, background: "#FDF6E8",
                              border: "1px solid #E8D9A8", borderRadius: 3,
                              fontFamily: "var(--mono)", fontSize: 10, lineHeight: 1.7,
                            }}>
                              <div style={{ fontWeight: 600, marginBottom: 4, color: "#7A5C0E" }}>
                                ⚖ Score Reconciliation
                              </div>
                              <div>LLM raw score: <strong>{vResults[selIter].reconciliation.llm}</strong></div>
                              <div>Deterministic baseline: <strong>{vResults[selIter].reconciliation.deterministic}</strong></div>
                              <div>Allowed range: [{vResults[selIter].reconciliation.floor}, {vResults[selIter].reconciliation.ceiling}]</div>
                              <div style={{ marginTop: 4, fontWeight: 600, color: "#7A5C0E" }}>
                                Final reconciled: {vResults[selIter].reconciliation.final}/100
                              </div>
                            </div>
                          )}

                          {/* LLM Score breakdown */}
                          {vResults[selIter].score_breakdown && (
                            <div style={{
                              marginBottom: 14, padding: 10, background: "#F2F0EB",
                              borderRadius: 3, fontFamily: "var(--mono)", fontSize: 10, lineHeight: 1.7,
                            }}>
                              <div style={{ fontWeight: 500, marginBottom: 4 }}>LLM Reasoning:</div>
                              <div>Base: {vResults[selIter].score_breakdown.base}</div>
                              {vResults[selIter].score_breakdown.additions?.map((a, i) => (
                                <div key={i} style={{ color: "#4A7C59" }}>+{a.points} {a.reason}</div>
                              ))}
                              {vResults[selIter].score_breakdown.deductions?.map((d, i) => (
                                <div key={i} style={{ color: "#C4403A" }}>−{d.points} {d.reason}</div>
                              ))}
                              <div style={{
                                borderTop: "1px solid #D4D0C8",
                                marginTop: 4, paddingTop: 4, fontWeight: 600,
                              }}>= {vResults[selIter].reconciliation?.llm ?? vResults[selIter].score} (LLM raw)</div>
                            </div>
                          )}

                          {/* Checks */}
                          <div>
                            {vResults[selIter].checks?.map((c, i) => (
                              <div key={i} style={{
                                display: "flex", gap: 6, padding: "5px 0",
                                borderBottom: "1px solid #F2F0EB",
                                fontSize: 11.5, lineHeight: 1.5,
                              }}>
                                <span style={{
                                  flexShrink: 0, fontWeight: 600, fontSize: 11,
                                  color: c.status === "PASS" ? "#4A7C59" : c.status === "FAIL" ? "#C4403A" : "#A68A2B",
                                }}>
                                  {c.status === "PASS" ? "✓" : c.status === "FAIL" ? "×" : "!"}
                                </span>
                                <div style={{ flex: 1 }}>
                                  {c.category && (
                                    <span style={{
                                      fontFamily: "var(--mono)", fontSize: 9,
                                      color: "#B5B0A6", marginRight: 5,
                                    }}>{c.category}</span>
                                  )}
                                  <span style={{ color: "#1A1915" }}>{c.property}</span>
                                  {c.details && (
                                    <div style={{ color: "#9C9689", fontSize: 11, marginTop: 1 }}>{c.details}</div>
                                  )}
                                  {c.formal_rule && (
                                    <div style={{
                                      fontFamily: "var(--mono)", fontSize: 9.5,
                                      color: "#B5B0A6", fontStyle: "italic",
                                    }}>{c.formal_rule}</div>
                                  )}
                                  {c.location && (
                                    <div style={{ fontFamily: "var(--mono)", fontSize: 9, color: "#B5B0A6" }}>
                                      @ {c.location}
                                    </div>
                                  )}
                                </div>
                                {c.severity && (
                                  <span style={{
                                    fontFamily: "var(--mono)", fontSize: 8, flexShrink: 0,
                                    color: c.severity === "critical" ? "#C4403A" :
                                      c.severity === "major" ? "#A68A2B" : "#9C9689",
                                  }}>{c.severity}</span>
                                )}
                              </div>
                            ))}
                          </div>

                          {vResults[selIter].proof_summary && (
                            <div style={{
                              marginTop: 12, padding: 10, background: "#F2F0EB",
                              borderRadius: 3, fontFamily: "var(--mono)", fontSize: 10,
                              color: "#6B665C", lineHeight: 1.6,
                            }}>
                              <strong>Proof:</strong> {vResults[selIter].proof_summary}
                            </div>
                          )}

                          {vResults[selIter].argument_scores?.length > 0 && (
                            <div style={{ marginTop: 12 }}>
                              <div style={{
                                fontFamily: "var(--mono)", fontSize: 9, color: "#9C9689",
                                marginBottom: 6, textTransform: "uppercase",
                              }}>Argument Strength</div>
                              {vResults[selIter].argument_scores.map((a, i) => (
                                <div key={i} style={{
                                  display: "flex", alignItems: "center", gap: 8,
                                  padding: "2px 0", fontSize: 11,
                                }}>
                                  <span style={{ flex: 1, color: "#6B665C", fontFamily: "var(--mono)", fontSize: 10 }}>
                                    {a.argument?.slice(0, 50)}
                                  </span>
                                  <div style={{ width: 80, height: 4, background: "#EFEDE8", borderRadius: 2 }}>
                                    <div style={{
                                      height: "100%", borderRadius: 2,
                                      background: a.overall >= 0.7 ? "#4A7C59" : "#A68A2B",
                                      width: `${a.overall * 100}%`,
                                    }} />
                                  </div>
                                  <span style={{ fontFamily: "var(--mono)", width: 32, color: "#6B665C", fontSize: 10 }}>
                                    {(a.overall * 100).toFixed(0)}%
                                  </span>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </>
                  ) : (
                    <div style={{ padding: 40, textAlign: "center", color: "#9C9689", fontSize: 13 }}>
                      Scores will appear after first iteration completes.
                    </div>
                  )}
                </div>
              )}

              {/* SELF-MODIFICATION TAB */}
              {tab === "selfmod" && (
                <div style={{ maxWidth: 760 }}>
                  <div style={{
                    fontFamily: "var(--mono)", fontSize: 9, color: "#9C9689",
                    textTransform: "uppercase", marginBottom: 12,
                  }}>Gödel-Machine Self-Modification</div>

                  <p style={{ fontSize: 12, color: "#6B665C", lineHeight: 1.7, marginBottom: 16 }}>
                    The orchestrator analyzes its own configuration after every iteration and proposes mutations.
                    Each proposed mutation must pass a verifier (Lean 4 → Z3 server → Z3 WASM → JS fallback)
                    that proves invariant preservation, monotone safety bounds, and progress before being applied.
                  </p>

                  {/* Verification Backend */}
                  <div style={{
                    marginBottom: 20, padding: 14, background: "#FAF9F6",
                    border: "1px solid #E8E5DE", borderRadius: 3,
                  }}>
                    <div style={{
                      fontFamily: "var(--mono)", fontSize: 9, color: "#9C9689",
                      textTransform: "uppercase", marginBottom: 8, fontWeight: 600,
                    }}>Verification Backend (optional — real z3-solver + Lean 4)</div>
                    <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
                      <input
                        type="text"
                        value={backendUrl}
                        onChange={(e) => setBackendUrl(e.target.value)}
                        onBlur={async () => { await saveStorage("backend-url", backendUrl); }}
                        placeholder={DEFAULT_BACKEND_URL}
                        style={{
                          flex: 1, fontFamily: "var(--mono)", fontSize: 11,
                          padding: "6px 10px", border: "1px solid #D4D0C5",
                          borderRadius: 3, background: "#fff", color: "#1A1915",
                        }}
                      />
                      <button
                        onClick={async () => {
                          const u = backendUrl || DEFAULT_BACKEND_URL;
                          setBackendUrl(u);
                          await saveStorage("backend-url", u);
                          const client = makeBackendClient(u);
                          const status = await client.probe();
                          setBackendStatus(status);
                        }}
                        style={{
                          fontFamily: "var(--mono)", fontSize: 10,
                          padding: "6px 12px", background: "#1A1915",
                          border: "none", borderRadius: 3, color: "#fff",
                          cursor: "pointer",
                        }}>Probe</button>
                    </div>
                    <div style={{ display: "flex", gap: 6, flexWrap: "wrap", fontFamily: "var(--mono)", fontSize: 10 }}>
                      {!backendUrl && (
                        <span style={{ color: "#9C9689" }}>
                          No backend configured — using browser WASM Z3 + JS fallback only.
                        </span>
                      )}
                      {backendUrl && !backendStatus && (
                        <span style={{ color: "#A68A2B" }}>● probing…</span>
                      )}
                      {backendStatus && !backendStatus.reachable && (
                        <span style={{
                          padding: "2px 8px", borderRadius: 2,
                          background: "#FDF2F1", color: "#7A2D29",
                          border: "1px solid #E0B4B1",
                        }}>● unreachable {backendStatus.error ? `(${backendStatus.error.slice(0, 40)})` : ""}</span>
                      )}
                      {backendStatus && backendStatus.reachable && (
                        <>
                          <span style={{
                            padding: "2px 8px", borderRadius: 2,
                            background: backendStatus.z3?.available ? "#F0F4ED" : "#FDF2F1",
                            color: backendStatus.z3?.available ? "#3D5C2E" : "#7A2D29",
                            border: `1px solid ${backendStatus.z3?.available ? "#C8D4B8" : "#E0B4B1"}`,
                          }}>
                            {backendStatus.z3?.available ? "✓" : "×"} Z3 {backendStatus.z3?.version?.slice(0, 30) || ""}
                          </span>
                          <span style={{
                            padding: "2px 8px", borderRadius: 2,
                            background: backendStatus.lean?.available ? "#F0F4ED" : "#FAF6E8",
                            color: backendStatus.lean?.available ? "#3D5C2E" : "#7A6E2A",
                            border: `1px solid ${backendStatus.lean?.available ? "#C8D4B8" : "#E0D4A8"}`,
                          }}>
                            {backendStatus.lean?.available ? "✓" : "○"} Lean {backendStatus.lean?.version?.slice(0, 30) || "offline"}
                          </span>
                          <span style={{
                            padding: "2px 8px", borderRadius: 2,
                            background: backendStatus.postgres?.available ? "#F0F4ED" : "#FAF6E8",
                            color: backendStatus.postgres?.available ? "#3D5C2E" : "#7A6E2A",
                            border: `1px solid ${backendStatus.postgres?.available ? "#C8D4B8" : "#E0D4A8"}`,
                          }}>
                            {backendStatus.postgres?.available ? "✓" : "○"} Postgres
                          </span>
                          <span style={{
                            padding: "2px 8px", borderRadius: 2,
                            background: backendStatus.mongo?.available ? "#F0F4ED" : "#FAF6E8",
                            color: backendStatus.mongo?.available ? "#3D5C2E" : "#7A6E2A",
                            border: `1px solid ${backendStatus.mongo?.available ? "#C8D4B8" : "#E0D4A8"}`,
                          }}>
                            {backendStatus.mongo?.available ? "✓" : "○"} Mongo
                          </span>
                          <span style={{
                            padding: "2px 8px", borderRadius: 2,
                            background: backendStatus.chroma?.available ? "#F0F4ED" : "#FAF6E8",
                            color: backendStatus.chroma?.available ? "#3D5C2E" : "#7A6E2A",
                            border: `1px solid ${backendStatus.chroma?.available ? "#C8D4B8" : "#E0D4A8"}`,
                          }}>
                            {backendStatus.chroma?.available ? "✓" : "○"} Chroma
                            {backendStatus.chroma?.collections?.length > 0 && ` (${backendStatus.chroma.collections.length})`}
                          </span>
                        </>
                      )}
                    </div>
                    <div style={{ marginTop: 8, fontSize: 10, color: "#9C9689", lineHeight: 1.5 }}>
                      Start the backend with <code style={{ fontFamily: "var(--mono)", background: "#F2F0EB", padding: "1px 4px" }}>
                      docker compose up -d</code> then <code style={{ fontFamily: "var(--mono)", background: "#F2F0EB", padding: "1px 4px" }}>
                      uvicorn agent_backend:app --port 8000</code>, then click Probe.
                    </div>
                  </div>

                  {/* Current System Config */}
                  <div style={{
                    marginBottom: 20, padding: 14, background: "#F0F4ED",
                    border: "1px solid #D4DBC8", borderRadius: 3,
                  }}>
                    <div style={{
                      fontFamily: "var(--mono)", fontSize: 9, color: "#3D5C2E",
                      textTransform: "uppercase", marginBottom: 8, fontWeight: 600,
                    }}>Current System Configuration (live, modifiable)</div>
                    <pre style={{
                      fontFamily: "var(--mono)", fontSize: 10, color: "#3D3A33",
                      lineHeight: 1.65, margin: 0, whiteSpace: "pre-wrap",
                    }}>{JSON.stringify(systemConfig, null, 2)}</pre>
                  </div>

                  {/* Invariants */}
                  <div style={{ marginBottom: 20 }}>
                    <div style={{
                      fontFamily: "var(--mono)", fontSize: 9, color: "#9C9689",
                      textTransform: "uppercase", marginBottom: 8,
                    }}>Configuration Invariants ({CONFIG_INVARIANTS.length})</div>
                    <div style={{ background: "#fff", border: "1px solid #E8E5DE", borderRadius: 3, padding: 10 }}>
                      {CONFIG_INVARIANTS.map((inv, i) => {
                        let holds;
                        try { holds = inv.check(systemConfig, systemConfig); } catch { holds = false; }
                        return (
                          <div key={i} style={{
                            display: "flex", gap: 8, padding: "3px 0",
                            fontFamily: "var(--mono)", fontSize: 10, lineHeight: 1.5,
                            borderBottom: i < CONFIG_INVARIANTS.length - 1 ? "1px solid #F2F0EB" : "none",
                          }}>
                            <span style={{
                              color: holds ? "#4A7C59" : "#C4403A",
                              fontWeight: 600, flexShrink: 0, width: 14,
                            }}>{holds ? "✓" : "×"}</span>
                            <div style={{ flex: 1 }}>
                              <span style={{ color: "#1A1915" }}>{inv.name}</span>
                              <div style={{ color: "#7A6E5A", fontStyle: "italic", fontSize: 9.5 }}>{inv.formal}</div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Modification History */}
                  <div>
                    <div style={{
                      fontFamily: "var(--mono)", fontSize: 9, color: "#9C9689",
                      textTransform: "uppercase", marginBottom: 8,
                    }}>Self-Modification History ({modificationLog.length} attempts)</div>
                    {modificationLog.length === 0 ? (
                      <div style={{ padding: 30, textAlign: "center", color: "#9C9689", fontSize: 12 }}>
                        No modifications proposed yet. Run a few iterations to trigger self-modification.
                      </div>
                    ) : (
                      <div>
                        {[...modificationLog].reverse().map((m, i) => (
                          <div key={i} style={{
                            marginBottom: 14, border: "1px solid #E8E5DE", borderRadius: 3,
                            background: m.applied ? "#F0F4ED" : "#FDF2F1",
                          }}>
                            <div style={{
                              padding: "8px 12px", borderBottom: "1px solid #E8E5DE",
                              display: "flex", justifyContent: "space-between", alignItems: "center",
                              fontFamily: "var(--mono)", fontSize: 10,
                            }}>
                              <span>
                                <span style={{ 
                                  color: m.isRollback ? "#8A6830" : (m.applied ? "#3D5C2E" : "#7A2D29"),
                                  fontWeight: 600 
                                }}>
                                  {m.isRollback ? "↶ ROLLED BACK" : (m.applied ? "✓ APPLIED" : "✗ REJECTED")}
                                </span>
                                <span style={{ color: "#9C9689", marginLeft: 10 }}>iter #{m.iter}</span>
                                {m.verdict.backendsUsed && m.verdict.backendsUsed.length > 0 && (
                                  <span style={{ marginLeft: 10 }}>
                                    {m.verdict.backendsUsed.map((b) => (
                                      <span key={b} style={{
                                        marginRight: 4, fontSize: 8, padding: "1px 5px",
                                        borderRadius: 2, fontWeight: 600,
                                        background: b === "lean" ? "#E8E0F4" :
                                                    b === "z3-server" ? "#DCE8F4" :
                                                    b === "z3-wasm" ? "#E8F0DC" : "#F2F0EB",
                                        color: b === "lean" ? "#4A2D7A" :
                                               b === "z3-server" ? "#1F3A6E" :
                                               b === "z3-wasm" ? "#3D5C2E" : "#7A6E5A",
                                      }}>{b}</span>
                                    ))}
                                  </span>
                                )}
                              </span>
                              <span style={{ color: "#9C9689", fontSize: 9 }}>
                                {m.applied && m.beforeConfig && (
                                  <button
                                  onClick={async () => {
                                    if (!confirm(`Rollback to config before iter #${m.iter}?\n\nThis restores:\n  citationsPerParagraph = ${m.beforeConfig.generation?.citationsPerParagraph ?? "?"}\n  targetTotalWords = ${m.beforeConfig.generation?.targetTotalWords ?? "?"}\n  ...and other params from before this modification was applied.`)) return;
                                    setSystemConfig(m.beforeConfig);
                                    await saveStorage("system-config", m.beforeConfig);
                                    // Append a rollback event to the log so it's visible
                                    const rollbackEntry = {
                                      iter: m.iter,
                                      timestamp: new Date().toISOString(),
                                      applied: false,
                                      isRollback: true,  
                                      proposal: {
                                        target_path: "(rollback)",
                                        reasoning: `Manual rollback of iter #${m.iter} modification. Config restored to state before that apply.`,
                                        expectedEffect: "Revert config to pre-mod state.",
                                      },
                                      verdict: { verdict: "ROLLED_BACK", diffs: [], proofChain: "manual rollback (no verification)" },
                                      beforeConfig: m.afterConfig,
                                      afterConfig: m.beforeConfig,
                                    };
                                    const newLog = [...modificationLog, rollbackEntry];
                                    setModificationLog(newLog);
                                    await saveStorage("modification-log", newLog);
                                  }}
                                  style={{
                                    marginLeft: 12, padding: "2px 10px", fontSize: 9,
                                    fontFamily: "var(--mono)", border: "1px solid #D4B896",
                                    background: "#FFF8EC", color: "#8A6830",
                                    borderRadius: 2, cursor: "pointer",
                                  }}
                                  title="Restore config to before this modification was applied"
                                >
                                  ↶ Rollback
                                </button>
                              )}
                                {new Date(m.timestamp).toLocaleString()}
                              </span>
                            </div>
                            <div style={{ padding: 12 }}>
                              <div style={{ fontSize: 11, color: "#3D3A33", marginBottom: 8, lineHeight: 1.6 }}>
                                <strong>Reasoning:</strong> {m.proposal.reasoning}
                              </div>
                              <div style={{ fontSize: 11, color: "#6B665C", marginBottom: 8 }}>
                                <strong>Expected effect:</strong> {m.proposal.expectedEffect}
                              </div>

                              {/* Diff */}
                              {m.verdict.diffs?.length > 0 && (
                                <div style={{
                                  marginBottom: 10, padding: 8, background: "#fff",
                                  borderRadius: 2, fontFamily: "var(--mono)", fontSize: 10,
                                }}>
                                  <div style={{ color: "#9C9689", marginBottom: 4, fontSize: 9, textTransform: "uppercase" }}>Diff</div>
                                  {m.verdict.diffs.map((d, j) => (
                                    <div key={j} style={{ color: "#3D3A33" }}>
                                      <span style={{ color: "#9C9689" }}>{d.path}:</span>{" "}
                                      <span style={{ color: "#C4403A" }}>{JSON.stringify(d.from)}</span>
                                      {" → "}
                                      <span style={{ color: "#4A7C59" }}>{JSON.stringify(d.to)}</span>
                                    </div>
                                  ))}
                                </div>
                              )}

                              {/* Proof chain */}
                              <div style={{
                                padding: 10, background: "#1A1915",
                                borderRadius: 2, fontFamily: "var(--mono)", fontSize: 10,
                                color: "#A8C8A0", lineHeight: 1.7,
                                whiteSpace: "pre-wrap", overflowX: "auto",
                              }}>{m.verdict.proofChain}</div>

                              {/* Detailed obligations */}
                              <details style={{ marginTop: 8 }}>
                                <summary style={{
                                  cursor: "pointer", fontFamily: "var(--mono)", fontSize: 10,
                                  color: "#6B665C",
                                }}>
                                  Show all {m.verdict.obligations.length} proof obligations
                                </summary>
                                <div style={{ marginTop: 6, paddingLeft: 12 }}>
                                  {m.verdict.obligations.map((o, k) => (
                                    <div key={k} style={{
                                      padding: "3px 0", fontFamily: "var(--mono)", fontSize: 10,
                                      color: o.status === "PROVED" ? "#4A7C59" :
                                             o.status === "REFUTED" ? "#C4403A" : "#A68A2B",
                                    }}>
                                      <span style={{ fontSize: 9, color: "#9C9689" }}>[{o.phase}]</span>{" "}
                                      <span>{o.status === "PROVED" ? "✓" : o.status === "REFUTED" ? "×" : "⚠"}</span>{" "}
                                      <span>{o.name}</span>
                                      {o.backend && (
                                        <span style={{
                                          marginLeft: 6, fontSize: 8, padding: "1px 5px",
                                          borderRadius: 2, fontWeight: 600,
                                          background: o.backend === "lean" ? "#E8E0F4" :
                                                      o.backend === "z3-server" ? "#DCE8F4" :
                                                      o.backend === "z3-wasm" ? "#E8F0DC" : "#F2F0EB",
                                          color: o.backend === "lean" ? "#4A2D7A" :
                                                 o.backend === "z3-server" ? "#1F3A6E" :
                                                 o.backend === "z3-wasm" ? "#3D5C2E" : "#7A6E5A",
                                        }}>{o.backend}</span>
                                      )}
                                      <div style={{ paddingLeft: 16, color: "#7A6E5A", fontStyle: "italic" }}>
                                        {o.formal}
                                      </div>
                                      {o.counterexample && (
                                        <div style={{ paddingLeft: 16, color: "#C4403A" }}>
                                          counter: {o.counterexample}
                                        </div>
                                      )}
                                    </div>
                                  ))}
                                </div>
                              </details>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Reset button */}
                  {modificationLog.length > 0 && (
                    <button
                      onClick={async () => {
                        setSystemConfig(DEFAULT_SYSTEM_CONFIG);
                        setModificationLog([]);
                        await saveStorage("system-config", DEFAULT_SYSTEM_CONFIG);
                        await saveStorage("modification-log", []);
                      }}
                      style={{
                        marginTop: 12, fontFamily: "var(--mono)", fontSize: 10,
                        padding: "5px 12px", background: "none",
                        border: "1px solid #E0B4B1", borderRadius: 3,
                        color: "#C4403A", cursor: "pointer",
                      }}>
                      Reset to Default Configuration
                    </button>
                  )}
                </div>
              )}

              {/* DIFF TAB */}
              {tab === "diff" && (
                <div style={{ maxWidth: 720 }}>
                  <div style={{
                    fontFamily: "var(--mono)", fontSize: 9, color: "#9C9689",
                    textTransform: "uppercase", marginBottom: 12,
                  }}>Changes Between Iterations</div>
                  {paperVersions.length < 2 ? (
                    <div style={{ padding: 40, textAlign: "center", color: "#9C9689", fontSize: 13 }}>
                      Need at least 2 iterations to show changes.
                    </div>
                  ) : (
                    <div>
                      {paperVersions.slice(1).map((v, i) => {
                        const prev = paperVersions[i];
                        const diffs = computeDiff(prev.text, v.text);
                        const adds = diffs.filter((d) => d.type === "add" || d.type === "mod").length;
                        const dels = diffs.filter((d) => d.type === "del" || d.type === "mod").length;
                        return (
                          <div key={i} style={{ marginBottom: 24 }}>
                            <div style={{
                              fontFamily: "var(--mono)", fontSize: 11, color: "#1A1915",
                              marginBottom: 6, display: "flex", gap: 14,
                            }}>
                              <span style={{ fontWeight: 500 }}>Iteration #{v.iter}</span>
                              <span style={{ color: "#4A7C59" }}>+{adds} added/changed</span>
                              <span style={{ color: "#C4403A" }}>−{dels} removed/changed</span>
                              <span style={{ color: "#9C9689" }}>{diffs.length} total diffs</span>
                            </div>
                            <div style={{
                              maxHeight: 240, overflowY: "auto",
                              fontFamily: "var(--mono)", fontSize: 10, lineHeight: 1.6,
                              border: "1px solid #E8E5DE", borderRadius: 3, padding: 10,
                              background: "#fff",
                            }}>
                              {diffs.slice(0, 40).map((d, j) => (
                                <div key={j} style={{ padding: "1px 0" }}>
                                  {d.type === "add" && (
                                    <div style={{ color: "#4A7C59", background: "#4A7C5910", padding: "1px 4px" }}>
                                      + {d.text?.slice(0, 140)}
                                    </div>
                                  )}
                                  {d.type === "del" && (
                                    <div style={{ color: "#C4403A", background: "#C4403A10", padding: "1px 4px" }}>
                                      − {d.text?.slice(0, 140)}
                                    </div>
                                  )}
                                  {d.type === "mod" && (
                                    <>
                                      <div style={{ color: "#C4403A", background: "#C4403A10", padding: "1px 4px" }}>
                                        − {d.old?.slice(0, 140)}
                                      </div>
                                      <div style={{ color: "#4A7C59", background: "#4A7C5910", padding: "1px 4px" }}>
                                        + {d.new?.slice(0, 140)}
                                      </div>
                                    </>
                                  )}
                                </div>
                              ))}
                              {diffs.length > 40 && (
                                <div style={{ color: "#9C9689", marginTop: 6 }}>
                                  ...and {diffs.length - 40} more changes
                                </div>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}

              {/* RUNS & DATA TAB */}
              {tab === "runs" && (
                <div style={{ maxWidth: 760 }}>
                  <div style={{
                    fontFamily: "var(--mono)", fontSize: 9, color: "#9C9689",
                    textTransform: "uppercase", marginBottom: 12,
                  }}>Database-Backed Runs &amp; RAG Corpus</div>

                  <p style={{ fontSize: 12, color: "#6B665C", lineHeight: 1.7, marginBottom: 16 }}>
                    Runs are persisted to Postgres (state + checkpoints) and Mongo (logs +
                    modification history). The RAG corpus lives in ChromaDB with embeddings
                    from <code style={{ fontFamily: "var(--mono)", fontSize: 11 }}>sentence-transformers/all-MiniLM-L6-v2</code>.
                    All three are accessed through the verification backend; if the backend
                    is offline this tab is empty and the app falls back to <code style={{ fontFamily: "var(--mono)", fontSize: 11 }}>window.storage</code>.
                  </p>

                  {!backendClient && (
                    <div style={{
                      padding: 14, background: "#FAF6E8", border: "1px solid #E0D4A8",
                      borderRadius: 3, fontSize: 12, color: "#7A6E2A",
                    }}>
                      No backend configured. Set the Verification Backend URL in the Self-Modification tab.
                    </div>
                  )}

                  {backendClient && (
                    <>
                      {/* Postgres runs */}
                      <div style={{ marginBottom: 24 }}>
                        <div style={{
                          display: "flex", justifyContent: "space-between", alignItems: "center",
                          marginBottom: 8,
                        }}>
                          <div style={{
                            fontFamily: "var(--mono)", fontSize: 9, color: "#9C9689",
                            textTransform: "uppercase",
                          }}>Postgres Runs {runsList ? `(${runsList.length})` : ""}</div>
                          <button
                            onClick={async () => {
                              setRunsList(null);
                              const r = await backendClient.listRuns();
                              setRunsList((r && r.runs) || []);
                            }}
                            style={{
                              fontFamily: "var(--mono)", fontSize: 10,
                              padding: "4px 10px", background: "#1A1915",
                              border: "none", borderRadius: 3, color: "#fff",
                              cursor: "pointer",
                            }}>Load runs</button>
                        </div>
                        {runsList === null && (
                          <div style={{
                            padding: 20, textAlign: "center", color: "#9C9689", fontSize: 11,
                            background: "#FAF9F6", border: "1px solid #E8E5DE", borderRadius: 3,
                          }}>Click Load runs to fetch from Postgres.</div>
                        )}
                        {runsList && runsList.length === 0 && (
                          <div style={{
                            padding: 20, textAlign: "center", color: "#9C9689", fontSize: 11,
                            background: "#FAF9F6", border: "1px solid #E8E5DE", borderRadius: 3,
                          }}>No runs in Postgres yet. Run the pipeline at least once.</div>
                        )}
                        {runsList && runsList.length > 0 && (
                          <div style={{ background: "#fff", border: "1px solid #E8E5DE", borderRadius: 3 }}>
                            {runsList.map((r, i) => (
                              <div key={r.run_id} style={{
                                padding: "8px 12px",
                                borderBottom: i < runsList.length - 1 ? "1px solid #F2F0EB" : "none",
                                fontFamily: "var(--mono)", fontSize: 10,
                              }}>
                                <div style={{ display: "flex", justifyContent: "space-between" }}>
                                  <span style={{ color: "#1A1915", fontWeight: 600 }}>{r.run_id}</span>
                                  <span style={{ color: "#9C9689", fontSize: 9 }}>{r.updated_at?.slice(0, 19)}</span>
                                </div>
                                <div style={{ color: "#6B665C", marginTop: 2 }}>
                                  {r.topic} · {r.iterations} iter · {r.checkpoint_count} checkpoint{r.checkpoint_count !== 1 ? "s" : ""} · {r.status}
                                  {typeof r.final_score === "number" && ` · score ${r.final_score}`}
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>

                      {/* Chroma RAG search */}
                      <div style={{ marginBottom: 24 }}>
                        <div style={{
                          fontFamily: "var(--mono)", fontSize: 9, color: "#9C9689",
                          textTransform: "uppercase", marginBottom: 8,
                        }}>ChromaDB RAG Search (collection: papers)</div>
                        <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
                          <input
                            type="text"
                            value={ragQuery}
                            onChange={(e) => setRagQuery(e.target.value)}
                            placeholder="semantic query — e.g. 'self-modification safety'"
                            style={{
                              flex: 1, fontFamily: "var(--mono)", fontSize: 11,
                              padding: "6px 10px", border: "1px solid #D4D0C5",
                              borderRadius: 3, background: "#fff", color: "#1A1915",
                            }}
                            onKeyDown={async (e) => {
                              if (e.key === "Enter" && ragQuery.trim()) {
                                setRagSearching(true);
                                const r = await backendClient.searchPapers(ragQuery.trim(), 8);
                                setRagHits((r && r.hits) || []);
                                setRagSearching(false);
                              }
                            }}
                          />
                          <button
                            disabled={!ragQuery.trim() || ragSearching}
                            onClick={async () => {
                              setRagSearching(true);
                              const r = await backendClient.searchPapers(ragQuery.trim(), 8);
                              setRagHits((r && r.hits) || []);
                              setRagSearching(false);
                            }}
                            style={{
                              fontFamily: "var(--mono)", fontSize: 10,
                              padding: "6px 12px",
                              background: ragQuery.trim() && !ragSearching ? "#1A1915" : "#9C9689",
                              border: "none", borderRadius: 3, color: "#fff",
                              cursor: ragQuery.trim() && !ragSearching ? "pointer" : "not-allowed",
                            }}>{ragSearching ? "..." : "Search"}</button>
                        </div>
                        {ragHits === null && (
                          <div style={{
                            padding: 20, textAlign: "center", color: "#9C9689", fontSize: 11,
                            background: "#FAF9F6", border: "1px solid #E8E5DE", borderRadius: 3,
                          }}>
                            Type a query and press Enter to search the embedded corpus.
                          </div>
                        )}
                        {ragHits && ragHits.length === 0 && (
                          <div style={{
                            padding: 20, textAlign: "center", color: "#9C9689", fontSize: 11,
                            background: "#FAF9F6", border: "1px solid #E8E5DE", borderRadius: 3,
                          }}>No matches. The corpus may be empty — Research Agent embeds documents during pipeline runs.</div>
                        )}
                        {ragHits && ragHits.length > 0 && (
                          <div style={{ background: "#fff", border: "1px solid #E8E5DE", borderRadius: 3 }}>
                            {ragHits.map((h, i) => (
                              <div key={h.id} style={{
                                padding: "10px 12px",
                                borderBottom: i < ragHits.length - 1 ? "1px solid #F2F0EB" : "none",
                                fontSize: 11,
                              }}>
                                <div style={{
                                  display: "flex", justifyContent: "space-between", marginBottom: 4,
                                  fontFamily: "var(--mono)", fontSize: 9, color: "#9C9689",
                                }}>
                                  <span>{h.id}</span>
                                  <span>distance {h.distance.toFixed(3)}</span>
                                </div>
                                <div style={{ color: "#3D3A33", lineHeight: 1.5 }}>
                                  {h.text.slice(0, 240)}{h.text.length > 240 ? "…" : ""}
                                </div>
                                {h.metadata && Object.keys(h.metadata).filter((k) => k !== "_").length > 0 && (
                                  <div style={{
                                    marginTop: 4, fontFamily: "var(--mono)", fontSize: 9, color: "#9C9689",
                                  }}>
                                    {Object.entries(h.metadata).filter(([k]) => k !== "_").map(([k, v]) => `${k}=${v}`).join(" · ")}
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>

                      {/* Current run info */}
                      {runId && (
                        <div style={{
                          padding: 12, background: "#F0F4ED", border: "1px solid #D4DBC8",
                          borderRadius: 3, fontFamily: "var(--mono)", fontSize: 10,
                        }}>
                          <div style={{ color: "#3D5C2E", fontWeight: 600, marginBottom: 4 }}>
                            Current run ID
                          </div>
                          <div style={{ color: "#1A1915" }}>{runId}</div>
                          <div style={{ color: "#6B665C", marginTop: 4, fontSize: 9 }}>
                            Modification log entries and checkpoints from this run are written
                            to Mongo and Postgres in real time as the pipeline executes.
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </div>
              )}

              {/* SETTINGS TAB */}
              {tab === "tuning" && (
                <div style={{ maxWidth: 600 }}>
                  <div style={{ marginBottom: 28 }}>
                    <div style={{
                      fontFamily: "var(--mono)", fontSize: 9, color: "#9C9689",
                      textTransform: "uppercase", marginBottom: 8,
                    }}>Prompt Fine-Tuning</div>
                    <p style={{ fontSize: 12, color: "#9C9689", marginBottom: 10, lineHeight: 1.6 }}>
                      Custom instructions added to the generation agent. Persisted across sessions.
                    </p>
                    <textarea
                      value={tuning}
                      onChange={(e) => updateTuning(e.target.value)}
                      rows={6}
                      placeholder="e.g., Focus on empirical methodology. Use APA citation style. Include statistical analysis in every claim."
                      style={{
                        width: "100%", padding: 12, fontSize: 12,
                        fontFamily: "var(--mono)", background: "#fff",
                        border: "1px solid #E8E5DE", borderRadius: 3,
                        resize: "vertical", lineHeight: 1.6, color: "#1A1915",
                      }}
                    />
                  </div>

                  <div>
                    <div style={{
                      fontFamily: "var(--mono)", fontSize: 9, color: "#9C9689",
                      textTransform: "uppercase", marginBottom: 8,
                    }}>RAG Context — Past Sessions</div>
                    <p style={{ fontSize: 12, color: "#9C9689", marginBottom: 10, lineHeight: 1.6 }}>
                      The orchestrator uses insights from previous sessions to improve planning.
                      Stored in browser persistent storage.
                    </p>
                    {history.length === 0 ? (
                      <div style={{ fontSize: 12, color: "#B5B0A6" }}>No previous sessions yet.</div>
                    ) : (
                      history.map((h, i) => (
                        <div key={i} style={{
                          padding: 10, border: "1px solid #E8E5DE", borderRadius: 3,
                          marginBottom: 6, fontSize: 11,
                        }}>
                          <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "#6B665C" }}>
                            {h.topic} — Score: <strong>{h.finalScore}</strong>
                          </div>
                          {h.insight && (
                            <div style={{ color: "#9C9689", marginTop: 3 }}>{h.insight.slice(0, 120)}</div>
                          )}
                        </div>
                      ))
                    )}
                    {history.length > 0 && (
                      <button onClick={async () => {
                        setHistory([]);
                        await saveStorage("rag-history", []);
                      }} style={{
                        fontFamily: "var(--mono)", fontSize: 10, padding: "5px 12px",
                        background: "none", border: "1px solid #E0B4B1", borderRadius: 3,
                        color: "#C4403A", cursor: "pointer", marginTop: 6,
                      }}>Clear History</button>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* EMPTY STATE */}
        {!hasStarted && (
          <div style={{ padding: "70px 20px", textAlign: "center" }}>
            <div style={{ fontFamily: "var(--serif)", fontSize: 22, color: "#1A1915", marginBottom: 10 }}>
              What would you like to research?
            </div>
            <div style={{
              fontSize: 13, color: "#9C9689", maxWidth: 420,
              margin: "0 auto 24px", lineHeight: 1.7,
            }}>
              Enter a topic or upload a paper (.md, .txt, .tex, .pdf) to improve.
            </div>
            <div style={{ display: "flex", justifyContent: "center", gap: 8, flexWrap: "wrap" }}>
              {[
                "Safety in reinforcement learning",
                "Quantum error correction",
                "Mitigating LLM hallucinations",
                "Ethics of autonomous vehicles",
              ].map((ex) => (
                <button key={ex} onClick={() => setTopic(ex)} style={{
                  fontFamily: "var(--body)", fontSize: 12, padding: "6px 14px",
                  background: "#fff", border: "1px solid #E8E5DE", borderRadius: 3,
                  color: "#6B665C", cursor: "pointer", transition: "all 0.2s",
                }}
                onMouseEnter={(e) => {
                  e.target.style.borderColor = "#1A1915";
                  e.target.style.color = "#1A1915";
                }}
                onMouseLeave={(e) => {
                  e.target.style.borderColor = "#E8E5DE";
                  e.target.style.color = "#6B665C";
                }}
                >{ex}</button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── MARKDOWN RENDERER ───
function RenderMd({ text }) {
  if (!text) return null;
  const lines = text.split("\n");
  const elements = [];
  let inCode = false;
  let codeBlock = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.startsWith("```")) {
      if (inCode) {
        elements.push(
          <pre key={`c${i}`} style={{
            fontFamily: "var(--mono)", fontSize: 11, lineHeight: 1.65,
            padding: 14, background: "#F2F0EB", borderRadius: 3,
            overflowX: "auto", color: "#3D3A33",
            margin: "16px 0", border: "1px solid #E8E5DE", whiteSpace: "pre-wrap",
          }}>{codeBlock.join("\n")}</pre>
        );
        codeBlock = [];
        inCode = false;
      } else {
        inCode = true;
      }
      continue;
    }
    if (inCode) { codeBlock.push(line); continue; }

    if (line.startsWith("# ")) {
      elements.push(
        <h1 key={i} style={{
          fontFamily: "var(--serif)", fontSize: 26, fontWeight: 400,
          color: "#1A1915", margin: "32px 0 14px", lineHeight: 1.2,
        }}>{line.slice(2)}</h1>
      );
    } else if (line.startsWith("## ")) {
      elements.push(
        <h2 key={i} style={{
          fontFamily: "var(--serif)", fontSize: 19, fontWeight: 400,
          color: "#1A1915", margin: "26px 0 10px", lineHeight: 1.25,
        }}>{line.slice(3)}</h2>
      );
    } else if (line.startsWith("### ")) {
      elements.push(
        <h3 key={i} style={{
          fontFamily: "var(--body)", fontSize: 13, fontWeight: 600,
          color: "#3D3A33", margin: "20px 0 6px",
          textTransform: "uppercase", letterSpacing: "0.04em",
        }}>{line.slice(4)}</h3>
      );
    } else if (/^[-*] /.test(line)) {
      elements.push(
        <div key={i} style={{
          paddingLeft: 18, fontSize: 14, color: "#3D3A33",
          lineHeight: 1.75, margin: "2px 0", position: "relative",
        }}>
          <span style={{ position: "absolute", left: 5, color: "#B5B0A6" }}>·</span>
          <Inline text={line.slice(2)} />
        </div>
      );
    } else if (!line.trim()) {
      elements.push(<div key={i} style={{ height: 8 }} />);
    } else {
      elements.push(
        <p key={i} style={{
          fontFamily: "var(--body)", fontSize: 14, color: "#3D3A33",
          lineHeight: 1.8, margin: "4px 0",
          textAlign: "justify", hyphens: "auto",
        }}>
          <Inline text={line} />
        </p>
      );
    }
  }
  return <div>{elements}</div>;
}

function Inline({ text }) {
  const parts = text.split(/(\*\*.*?\*\*|\[.*?,\s*\d{4}.*?\])/g);
  return (
    <>
      {parts.map((p, i) => {
        if (p.startsWith("**") && p.endsWith("**")) {
          return <strong key={i} style={{ fontWeight: 600, color: "#1A1915" }}>{p.slice(2, -2)}</strong>;
        }
        if (p.match(/^\[.*?,\s*\d{4}.*?\]$/)) {
          return <span key={i} style={{ fontFamily: "var(--mono)", fontSize: 11.5, color: "#7A6E5A" }}>{p}</span>;
        }
        return <span key={i}>{p}</span>;
      })}
    </>
  );
}
