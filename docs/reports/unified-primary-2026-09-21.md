# Unified primary retraining review, 2026-09-21

The primary candidate improved the original eight-task development results, but substantially regressed on complete condition records. It failed development acceptance and cannot be promoted as a stable release. Candidate freezing, independent testing, merged export, and SDK/HTTP release verification have not run. Further training is deferred pending review of the evidence.

The release objective is an overall improvement over Phase 4 with bounded task tradeoffs and no material condition defects. Jev is a descriptive reference, not a promotion gate.

## Evaluated artifact and measurement conditions

This candidate starts a fresh rank-16 LoRA from the official post-trained Qwen3.5-0.8B. It uses the registered primary recipe: learning rate `1e-4`, two epochs, seed 2026, microbatch 16, accumulation 1, and gradient checkpointing. It does not continue a historical project adapter.

Training processed 11,390 unique requests over two epochs, totaling 6,098,918 unpadded tokens and 1,424 optimizer steps. The RTX 4090 D run took **2,019.34 seconds, or 33 minutes 39 seconds**, with peak allocated memory of 9.155 GiB. The disabled-adapter base probe had zero maximum difference. Environment and performance investigation details are in the [cloud execution record](../process/unified-cloud-run-2026-09-21.md).

Adapter SHA-256: `5a8d064464ea15855705c4d9e55a0cb2e55177c28b3ae1bdb33e1168fd78318a`.

All comparisons below use baselines recomputed in the same cloud environment on identical requests. Historical percentages are not substituted for these measurements. The candidate's temperatures were fitted on the registered calibration partition. Confidence intervals resample paired source groups rather than treating translations and counterfactual presentations as independent observations.

## Overall results

| Cohort and metric | Records | Instruction parent | Phase 4 | Primary | Primary minus Phase 4 |
|---|---:|---:|---:|---:|---:|
| Original eight-task development, task macro accuracy | 1,464 | 87.75% | 87.50% | **89.72%** | **+2.22 pp** |
| Exposed eight-task regression, task macro accuracy | 1,790 | 86.42% | 89.57% | **89.83%** | **+0.26 pp** |
| Full development, task macro accuracy | 3,784 | 87.95% | 88.44% | **89.31%** | **+0.87 pp** |

The original development macro gain has a paired source-group 95% interval of **[+0.67, +3.89] percentage points**, supporting improvement on that cohort. The exposed regression interval is **[-1.09, +1.66] points**; its small positive point estimate does not establish a reliable gain. Both cohorts are development or exposed regression evidence. Independent-test predictions remain unavailable.

Probability quality also depends on the cohort. Original development macro Brier improved from **0.19207 to 0.15358**, with a difference interval of [-0.06124, -0.01814]. Regression macro Brier changed from **0.16345 to 0.15608**, with an interval of [-0.02527, +0.00904]. Lower Brier is better.

### Eight-task regression

| Task | Records | Phase 4 | Primary | Difference, pp |
|---|---:|---:|---:|---:|
| Intent | 458 | 81.00% | **86.24%** | +5.24 |
| Inference | 480 | 80.00% | 79.79% | -0.21 |
| Boolean reading | 160 | 85.63% | 85.00% | -0.63 |
| Candidate extraction | 80 | 85.00% | **86.25%** | +1.25 |
| Numeric rules | 172 | 95.93% | **93.02%** | **-2.91** |
| Paraphrase | 180 | 89.44% | 88.33% | -1.11 |
| Candidate retrieval | 50 | 100.00% | 100.00% | 0.00 |
| Ordinal rules | 210 | 99.52% | **100.00%** | +0.48 |

Numeric regression fell from 165/172 to 160/172, exceeding the registered two-point family margin. In contrast, the original numeric development slice improved from 157/176 to **174/176**. These different directions within the same task make cohort-level reporting necessary.

The registered nonnumeric language alert did not trigger. This does not imply that every language slice improved; numeric condition failures are assessed separately below.

## Release-blocking condition regression

The candidate corrected **18 of the fixed 38 historical false rejections, leaving 20 unresolved**.

The exposed condition cohort contains 720 presentations from 12 source groups:

| Condition slice | Records | Instruction parent | Phase 4 | Primary |
|---|---:|---:|---:|---:|
| All conditions | 720 | 85.83% | 91.81% | **86.11%** |
| Complete records, English | 120 | 100.00% | 94.17% | **60.00%** |
| Complete records, Chinese | 120 | 99.17% | 73.33% | **56.67%** |
| Eligible records, English | 56 | 100.00% | 87.50% | **46.43%** |
| Eligible records, Chinese | 56 | 100.00% | 42.86% | **51.79%** |
| Disabled gate, English | 120 | 64.17% | 90.00% | **100.00%** |
| Disabled gate, Chinese | 120 | 75.00% | 100.00% | **100.00%** |
| Missing fields, English | 120 | 86.67% | 93.33% | **100.00%** |
| Missing fields, Chinese | 120 | 90.00% | 100.00% | **100.00%** |

Complete-record accuracy fell from **201/240 to 140/240**, a decline of **25.42 percentage points**, with a paired source-group 95% interval of **[-39.58, -10.00]**. This is a material regression, well beyond a small task tradeoff.

The 100 complete-record errors comprise **57 false rejections of eligible records and 43 false acceptances of comparison failures**. Across all 720 requests, the primary candidate corrected 38 errors from the cloud Phase 4 baseline, introduced 79 errors, and retained 21 shared errors. Condition Brier worsened from **0.10493 to 0.25959**.

### New condition development

All **1,600 new condition development presentations** were correct, including both languages' complete, eligible, disabled-gate, and missing-field slices. The parent's negative-number strict-comparison errors on those presentations were absent in the primary predictions.

These presentations cover eight source groups. Their perfect score does not demonstrate complete numeric generalization: the subsequent coverage audit found that training and this development cohort omit the same cases.

## Diagnostic evidence

### Missing numeric relations

The frozen training data and new condition development generator select only two of the three possible operand relations for each operator:

| Operator | Relations represented in training and new development | Missing relation |
|---|---|---|
| `a >= b` | `a = b`, `a < b` | **`a > b`, a valid eligible case** |
| `a > b` | `a > b`, `a = b` | **`a < b`, a comparison failure** |
| `a <= b` | `a = b`, `a > b` | **`a < b`, a valid eligible case** |
| `a < b` | `a < b`, `a = b` | **`a > b`, a comparison failure** |
| `a = b` | `a = b`, `a > b` | **`a < b`, a comparison failure** |

Each represented relation has 160 complete training presentations and 32 new development presentations. Each omitted relation has zero. The old condition regression includes all three relations, with 16 presentations per operator-relation cell.

On the omitted relations, the candidate scored **21/80 (26.25%)**; on represented relations, **119/160 (74.38%)**. Thus **59 of its 100 condition errors** occur in omitted relation cells. For `>=` with strictly greater operands it scored **0/16**; for `<=` with strictly smaller operands it scored **2/16**.

This establishes a coverage omission and its association with failures. No controlled retraining ablation has measured its causal contribution in isolation.

### Equal values with different representations

All 800 complete equal-value training presentations and all 160 corresponding new-development presentations used the same operand type. Neither included mixed integer and equivalent floating-point representations.

| Equal-value form in exposed regression | Primary correct |
|---|---:|
| Same type, such as integer and integer | **38/40** |
| Mixed type, such as `128.0` and `128` | **8/40** |

The mixed-type group accounts for 32 errors and does not overlap the omitted strict-relation cells. Together these two categories contain **91 of the 100 condition errors**. Presentations within their source groups remain correlated.

A fixed-weight diagnostic preserved the questions, calibration, mathematical values, and labels of the 40 mixed-type records, while rendering integral floats as equivalent integers. Correctness increased from **8/40 to 33/40**, correcting 30 records and introducing five errors. The original evaluation remains unchanged. This supports sensitivity to numeric representation, while the remaining and introduced errors prevent treating normalization as a validated repair.

### Wording-only diagnostic

A second fixed-weight diagnostic preserved the states, operators, languages, question polarities, and labels of all 240 complete records, replacing only instructions with an existing training formulation. Correctness changed from **140/240 to 144/240**, correcting eight records and introducing four errors. Wording alone was a weak explanation in this probe.

Both diagnostics use exposed records for failure investigation. Neither is independent acceptance evidence or a new training run.

## Acceptance decision and limits

Three registered checks failed: all 38 known failures resolved; complete and eligible condition accuracy; retention of every original task family. Prerequisite checks, new-development condition checks, original macro retention, and the registered nonnumeric language check passed.

The candidate therefore remains a failed development candidate despite its original-development gains. It has not demonstrated the stable behavior needed for release.

No `5e-5` comparison was run, so this experiment cannot establish whether a lower learning rate would improve the outcome. The runner saved final weights only, so it cannot establish whether one epoch was better than two. The evidence also does not establish a model-capacity ceiling. Data audit omissions and the proposed review before any further training are recorded in the [cloud execution investigation](../process/unified-cloud-run-2026-09-21.md#post-run-investigation).

## Evidence

- Complete cloud evidence: `artifacts/necro-unified-primary-cloud-evidence-2026-09-21.tar.gz`.
- Archive SHA-256: `7dfd74727498d6ab08979c858845c3b1496b2588d3983a8838cc201394be334c`, verified after download.
- Local extracted root: `results/cloud-primary-2026-09-21/`.
- Acceptance and slices: `workspace/results/cloud/primary/assessment.json` and `slices.json` under that root.
- Coverage and diagnostics: `ops/relation-coverage-audit.json`, `equal-number-format-audit.json`, `wording-diagnostic/`, and `format-diagnostic/`.
- Additional paired condition transitions and intervals: `results/cloud-ops-local/condition-transitions.json`.

Weights, configurations, calibration, raw predictions, training logs, and failed acceleration probes are retained. GPU jobs have ended; the server instance was not shut down.
