# Run Analysis Report

Generated: 2026-04-20T11:49:56.076291Z
Total runs examined: 20 (8 completed)
TSR threshold: final_score >= 70.0

## Per-run table (completed runs, newest first)

| Topic | Score | ET(s) | Words | Unique cites | Orphan | Sects | FAIL | Mods app/ver/prop | Run |
|---|---|---|---|---|---|---|---|---|---|
| Mitigating LLM hallucinations | 61.0 | 997.5 | 8522 | 0 | 0 | 8/8 | 5 | 0/0/3 | `…1_3331f740` |
| History of AI winters | 46.0 | 238.2 | 5195 | 0 | 0 | 6/8 | 8 | 0/0/1 | `…4_9ce120e0` |
| Climate change mitigation strategies | 58.0 | 247.2 | 5547 | 47 | 5 | 8/8 | 3 | 1/1/1 | `…7_dca0ba5f` |
| Formal verification techniques for ensur | 60.0 | 350.7 | 7621 | 101 | 0 | 5/8 | 5 | 0/0/0 | `…4_e08093b6` |
| Benefits and risks of large language mod | 48.0 | 191.4 | 4248 | 25 | 5 | 5/8 | 7 | 1/1/1 | `…8_82a25e5d` |
| Mitigating LLM hallucinations | 85.0 | 215.3 | 4299 | 30 | 0 | 8/8 | 1 | 0/0/0 | `…7_2cb18510` |
| Mitigating LLM hallucinations | 66.0 | 220.7 | 4488 | 28 | 5 | 8/8 | 3 | 1/1/1 | `…5_dc8d6b88` |
| Mitigating LLM hallucinations | 58.0 | 236.7 | 4628 | 46 | 5 | 8/8 | 3 | 1/1/1 | `…6_55dd6edd` |

## Per-topic aggregates (where n >= 2)

| Topic | n | AQS mean | AQS stddev | AQS min-max | ET mean(s) |
|---|---|---|---|---|---|
| Mitigating LLM hallucinations | 4 | 67.5 | 12.12 | 58.0-85.0 | 417.55 |

## Overall (all completed runs)

- **AQS** mean: 60.25, stddev: 12.01, min-max: 46.0-85.0
- **TSR** (score >= 70.0): 1/8 = 0.125
- **ET** mean: 337.21s, stddev: 270.94s
- **Word count** mean: 5568.5, stddev: 1624.95
- **Unique citations** mean: 34.62, stddev: 32.17
- **Orphan citations** mean: 2.5
- **VSR** mean: 0.67 (from 6 runs with mods)

## Failed check categories across all completed runs

| Category | # runs where it failed |
|---|---|
| STRUCTURAL | 7 |
| CITATION | 6 |
| citations | 6 |
| structure | 3 |
| MATHEMATICAL | 2 |
| PROPOSITIONAL | 1 |
| ARGUMENT | 1 |
| CONSISTENCY | 1 |

## Caveats

- IAR and IE are not computed: they require multi-iteration runs (Passes >= 3)
  to observe per-iteration score deltas.
- VSR is meaningful only when `mods_proposed > 0`; Pass=1 runs often have 0 or 1.
- 'failed_categories' merges both deterministic checks (lowercase like 'word_count')
  and LLM-produced checks (uppercase like 'STRUCTURAL'); same concept may appear twice.
- ET (execution_seconds) measures wall clock from POST /state/run to finalize —
  includes LLM latency, not pipeline overhead alone.