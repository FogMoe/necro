# Condition repair validation, 2026-09-21

The smaller-update candidate improves the exposed multi-task regression results and repairs the original disabled-gate and missing-field failures. Independent condition transfer still fails the registered per-language check. The candidate remains available for review; stable-release packaging has not passed its capability gate.

## Candidate and evidence

The frozen candidate is `results/phase4/selected/adapter`, with weight SHA-256 `f0ae9537edb9478b26752db9ea3287f61ae8fdf0a8a84d453e04c93679d7d092`. Its training source is `condition-v3-smallstep-seed2026`. The [repair record](condition-repair-2026-09-21.md#smaller-update-comparison-and-selection) describes the development selection and the accepted numeric-development tradeoff.

Independent condition inference began after selection and calibration were frozen. The 720-question cohort contains 12 source groups with held-out numerical states, field names and two wording styles. It is now exposed and must remain a regression cohort in any later experiment. The multi-task cohort contains 1,790 previously exposed Phase 3 questions. Final predictions and paired comparisons are in `results/phase4/final/`; `capability-review.json` records the checks and evidence hashes.

## Independent condition transfer

| Slice | Questions | Predecessor | Candidate | Jev 1.13.0 |
|---|---:|---:|---:|---:|
| All conditions | 720 | 93.19% | 91.94% | 98.33% |
| Complete fields and enabled gate | 240 | 89.17% | 84.17% | 95.42% |
| Disabled gate | 240 | 95.42% | 95.00% | 100.00% |
| Missing fields | 240 | 95.00% | 96.67% | 99.58% |
| English disabled gate | 120 | 90.83% | 90.00% | 100.00% |
| English missing fields | 120 | 90.00% | 93.33% | 100.00% |

Chinese disabled-gate and missing-field accuracy was 120/120 for each category. With complete fields and an enabled gate, Chinese accuracy was 89/120 (74.17%), compared with 98/120 (81.67%) for the predecessor.

The English precondition failures concentrate in one held-out formulation. In style 6, all 60 disabled-gate and 60 missing-field cases were correct. In style 7, those counts fell to 48/60 and 52/60. The style-7 rule says that a missing required field or a false gate makes the record ineligible regardless of its numbers, followed by the numerical eligibility condition. This shows residual sensitivity to rule expression after the original wording was repaired.

Candidate minus predecessor accuracy was -1.25 percentage points, with paired source-group bootstrap 95% interval [-2.92, 0.28]. Candidate minus Jev was -6.39 points, with interval [-9.17, -3.47]. Candidate Brier was 0.104123, compared with 0.111389 for the predecessor and 0.033697 for Jev. The interval containing zero does not establish equivalence. Twelve source groups provide the sampling units.

## Exposed multi-task regression

| Task | Questions | Predecessor | Instruction parent | Candidate | Jev 1.13.0 |
|---|---:|---:|---:|---:|---:|
| Intent | 458 | 76.86% | 81.22% | 80.79% | 87.55% |
| Inference | 480 | 79.38% | 79.58% | 79.79% | 82.92% |
| Boolean reading | 160 | 81.25% | 85.00% | 85.00% | 94.38% |
| Candidate extraction | 80 | 88.75% | 85.00% | 85.00% | 91.25% |
| Numeric rules | 172 | 84.88% | 70.93% | 95.93% | 96.51% |
| Paraphrase | 180 | 88.33% | 89.44% | 89.44% | 86.11% |
| Candidate retrieval | 50 | 100.00% | 100.00% | 100.00% | 100.00% |
| Ordinal rules | 210 | 98.57% | 99.52% | 99.52% | 100.00% |
| Task macro | | 87.25% | 86.34% | 89.43% | 92.34% |

The candidate gained 3.10 macro points over the instruction parent, with paired 95% interval [2.35, 3.84]. The improvement over the predecessor was 2.18 points, with interval [0.54, 3.73]. Macro Brier improved from 0.209705 for the parent to 0.163865. Jev's macro Brier was 0.117880.

Relative to the instruction parent, six of the other seven task accuracies were unchanged or improved; intent lost two answers out of 458. The original numeric precondition counts recovered to 40/40 disabled gates and 12/12 missing fields.

The strict two-point retention check against both baselines failed on candidate extraction. It remained at 68/80, identical to the instruction parent, compared with 71/80 for the earlier predecessor. The candidate-minus-predecessor interval was [-12.50, 2.50] points. This loss predates the condition repair. No task-language slice met the registered material-regression alert rule.

## Disposition

Two capability checks failed: per-language independent precondition accuracy and retention against both local baselines. Their original results remain recorded. The independent condition failure is separate from the accepted four-question numeric-development tradeoff.

The candidate has a useful multi-task improvement, while more training on the same narrow formulations has not established broad condition stability. Additional epochs are not scheduled. Any later repair should address the demonstrated expression sensitivity with a new independent validation design, rather than optimize the exposed holdout or the Jev percentage gap.

This candidate has not yet been exported and verified through the complete release API workflow. The earlier Phase 3 export issue and its validated loader fix remain in the [numeric regression investigation](numeric-regression-2026-09-21.md). The stable report generator rejects a failed capability review, so no stable model card or Hugging Face upload was produced from this candidate.
