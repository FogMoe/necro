# Design review and Phase 2 plan, 2026-09-20

This review examines Phase 1 data, training implementation, and model selection, and records the Phase 2 plan. See the [improvement record](improvement-2026-09-20.md) for Phase 1 history and the [evaluation report](../reports/improvement-2026-09-20.md) for exported-weight results.

## Calibrated development comparison

Each model's temperature was fitted on the same calibration set before repeating the comparison on the original development set:

| Model | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| round3 | 84% | 0.6601 | 0.2369 | 0.0666 |
| round4 | 85% | 0.6450 | 0.2289 | 0.0471 |

round4 performed better in this comparison. Phase 1 rejected round4 on raw NLL and then calibrated only round3, making the comparison inconsistent. Phase 2 needs to compare candidates under the same calibration procedure.

## Data and evaluation issues

| Issue | Evidence | Phase 2 plan |
|---|---|---|
| Tasks concentrated on classification | Public questions came from XNLI and MASSIVE, mainly as Choice | Add natural question answering, paraphrase, candidate-answer and passage selection, and Noul and Score boundary tasks |
| Repeated constructed examples | Removing record_id left 36 distinct pilot rule requests and 33 expanded requests | Record normalized unique requests, templates, labels, and boundary distributions |
| Multiple factors changed together | round3 changed data volume, mixture, templates, and learning rate. round4 also changed the seed and microbatch | Fix data, starting weights, seed, learning rate, and batching when comparing objectives |
| Coarse selection threshold | The original development set contained English and Chinese questions from shared sources, so a few questions could shift percentage-point results | Report paired changes and bootstrap intervals by source group |
| Test data informed later design | Phase 1 test and boundary questions had been inspected | Use them for historical regression and create a new Phase 2 test set |
| Sequential sampling concentrated coverage | Original sampling used records near the start of each split | Use reproducible dispersed sampling and save row IDs and hashes |
| Insufficient structural deduplication | Irrelevant record IDs changed string hashes | Add task-level normalized fingerprints and source groups |
| Probability precision differed across services | All 11,040 Jev probabilities were on a two-decimal grid, and 20 correct-label probabilities were zero | Compare accuracy, Brier, and NLL sensitivity under a common clipping rule |
| Score metrics had unclear roles | Class accuracy used the most probable level, while the API returned a weighted score | Report MAE, MAE normalized by the level span, and level accuracy separately |
| Candidate-selection scope needed clarification | The API selected from supplied answers or passages | Report candidate-extraction hit rate and passage-selection top-1 accuracy |
| Incomplete robustness coverage | Systematic reordering, negation, missing-information, and long-input tests were absent | Add separate consistency and boundary probes |
| Ambiguous rule wording | “only if” and its Chinese equivalent express a necessary condition | State necessary and sufficient conditions explicitly, and check equality, negation, and missing fields |
| Single runs were sensitive to ordering | Each setting was run once | Repeat beneficial settings with another seed |

## Training implementation review

The review covered answer encoding, supervised positions with left padding, frozen base weights, continued LoRA training, gradient accumulation, save/reload behavior, and full numeric-path scoring. Supervision covered only answer tokens. Disabling the adapter produced zero difference on the base-model probe. Merged-weight reload results are in the [evaluation report](../reports/improvement-2026-09-20.md#weight-and-api-validation).

The plan adds optional candidate-restricted cross-entropy. Single-token answers use the candidate objective, while multi-token numeric labels retain full answer-token loss. This is recorded as a mixed objective. Controlled comparisons use the same data and starting weights.

## Phase 2 plan

1. Build a task development set and data manifest. Consider PAWS-X, BoolQ, and question-answering datasets that can be converted to candidate selection. Record human annotations and code-derived rules separately.
2. Prepare approximately 2,000–3,000 training records, with about one quarter replaying the original tasks and the rest covering new natural tasks and paired boundary examples.
3. Split training, development, calibration, and new test data by source groups covering translations, passages, paraphrases, and counterfactual pairs. Evaluate the new test set after model selection.
4. Measure round3 and round4 on the new tasks and choose a common starting point. Fix rank 16, batch 4, accumulation 2, input limit 2048, learning rate, and seed, then compare one epoch of each objective.
5. Select by task-macro accuracy, checking original-task regression and Score MAE. Stop additional training if the gain is below 2 percentage points with no key capability improvement. Investigate any task decline above 3 percentage points.
6. Repeat beneficial settings with one additional seed. Retain the baseline if gains are not stable.
7. Save accuracy, Brier, Score MAE, robustness, latency, and failure examples under a common calibration procedure, and compare against a fixed Jev version on identical questions.

## Review materials

Raw audit figures are stored in `results/design-audit/audit.json`. Upstream license references for candidate datasets are listed below. Record datasets actually used in the training manifest:

- PAWS: [upstream license](https://github.com/google-research-datasets/paws/blob/master/LICENSE).
- BoolQ: [dataset card](https://huggingface.co/datasets/google/boolq).
- SQuAD: [dataset card](https://huggingface.co/datasets/rajpurkar/squad).

Sources used by the project are recorded in [third-party notices](../../THIRD_PARTY_NOTICES.md).
