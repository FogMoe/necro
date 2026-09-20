# Phase 2 development record

This page records Phase 2 development comparisons, replication, and independent test results.

| Task | round4 starting point | answer-ce | candidate-ce | refinement |
|---|---:|---:|---:|---:|
| paraphrase | 50.00% | 83.00% | 83.00% | 83.00% |
| reading_boolean | 66.25% | 72.50% | 72.50% | 75.00% |
| candidate_extraction | 81.25% | 86.25% | 85.00% | 92.50% |
| candidate_retrieval | 90.24% | 97.56% | 100.00% | 97.56% |
| numeric_rule | 75.58% | 81.40% | 80.23% | 81.40% |
| ordinal_rule | 61.67% | 97.50% | 97.50% | 95.83% |
| inference | 85.00% | 81.00% | 80.00% | 85.00% |
| intent | 85.00% | 82.00% | 81.00% | 84.00% |
| Task macro average | 74.37% | 85.15% | 84.90% | 86.79% |

Score accuracy measures whether the most probable level is correct. The API returns the weighted level, which is also evaluated using normalized MAE.

Under the same calibration method, task-macro Brier was 0.21734 for answer-ce, 0.21850 for candidate-ce, and 0.18863 for refinement. Refinement's normalized Score MAE was 0.01558.

answer-ce improved task-macro accuracy by 10.78 percentage points over the starting weights, with a source-group paired bootstrap 95% interval of +7.57 to +14.02. candidate-ce was -0.25 percentage points relative to answer-ce, with an interval of -1.04 to +0.64. Refinement gained +1.64 percentage points over answer-ce, with an interval of approximately -0.02 to +3.38. Inference recovered to the starting level, while numeric-rule errors did not decrease.

Training took 726.80 seconds for answer-ce, 758.32 seconds for candidate-ce, and 820.27 seconds for refinement. All three runs verified zero base-model probe difference with the adapter disabled. Peak memory was approximately 3.67–3.71 GiB.

`operator-contrast` switches among five comparison operators for the same state, adding atomic comparisons and compound-condition judgments while retaining other-task replay. It contains 3,045 records, including 1,960 newly constructed rule examples from 60 related source groups. The run starts from refinement weights and keeps other major hyperparameters unchanged.

Raw reports and plans are stored in `results/phase2/v3/`. See the [source audit](phase2-source-audit.md) for isolation corrections and the [protocol](phase2-acceptance.md) for acceptance and stopping criteria.

## Operator-contrast results

operator-contrast trained for 725.40 seconds, with peak memory of 3.61 GiB and zero base-model probe difference. Strict-development macro accuracy was 89.13%, up 2.34 percentage points over refinement, with a source-group paired 95% interval of +0.74 to +3.98. Accuracy was 93.02% on numeric rules, 89% on inference, 85% on intent, and 99.17% on the most probable ordinal level. Reading Boolean accuracy remained 75%.

After calibration data alone showed Noul temperature reaching the original upper bound of 4, the grid was extended to 8 for every candidate, retaining the original grid and results. The explicit call was `fit_primitive_temperatures(..., upper=8.0)`, registered in `results/phase2/v3/calibration-amendment.json`. Temperatures for the other three candidates stayed unchanged. operator-contrast's Noul temperature was 5.09824 and task-macro Brier was 0.17016. Performance acceptance thresholds stayed unchanged.

The seed 2027 replication used the same refinement parent weights, data, and major configuration.

## Seed replication and parameter averaging

The seed 2027 replication, with the same parent weights and recipe, reached 89.10% macro accuracy, close to seed 2026's 89.13%. Numeric-rule accuracy was 91.86% and 93.02%, inference 87% and 89%, and paraphrase 86% and 83%, respectively. Replication took 724.79 seconds, with peak memory of 3.61 GiB and zero base-model probe difference.

Equal-weight parameter averaging of LoRA factors from the common starting point retained rank 16 and a single inference pass. It reached 88.81% development macro accuracy and 0.16564 calibrated task-macro Brier. The mean was applied separately to LoRA A and B factors, producing another candidate for comparison. Brier scores for the two individual seeds were 0.17016 and 0.16887. All results used the same `[0.5, 8]` calibration grid. The averaging source snapshot and parent-weight hashes were recorded.

The reading experiment started from operator-contrast seed 2026, using 2,591 records. These included 1,000 reading questions from 950 previously unused passages, with the remainder used for replay. New material excluded existing training sources, protected slices, and previously seen duplicate questions.

## Results at the end of Phase 2

The reading experiment completed in 717.04 seconds. Development macro accuracy was 89.09%, a change of -0.04 percentage points from operator-contrast, with a paired source-group interval of -1.58 to +1.57. Reading Boolean accuracy rose from 75.00% to 76.25% and candidate extraction from 91.25% to 93.75%, while numeric and ordinal rules regressed. The selection therefore retained operator-contrast seed 2026, freezing weights, calibration, source code, and data fingerprints before opening the test set.

The Phase 2 independent test contained 1,171 records across 503 connected source groups:

| Task | Necro | Jev 1.13.0 |
|---|---:|---:|
| paraphrase | 76.52% | 75.65% |
| reading_boolean | 77.00% | 94.00% |
| candidate_extraction | 81.00% | 97.00% |
| numeric_rule | 86.76% | 97.79% |
| ordinal_rule | 98.33% | 100.00% |
| inference | 78.00% | 87.00% |
| intent | 51.50% | 73.50% |
| candidate_retrieval | 98.75% | 100.00% |
| Task macro average | 80.98% | 90.62% |

The accuracy gap was -9.63 percentage points, with a source-group paired 95% interval of -12.08 to -7.29. Task-macro Brier was 0.28498 versus 0.13985. Excess normalized Score MAE was 0.00658. Only the Score MAE threshold passed. Macro accuracy, per-task accuracy, and Brier failed their thresholds.

On 140 perturbation questions linked to the originals, task-macro accuracy was 80.63% versus 90.63%. Reading-negation consistency on these 10 pairs was 30% versus 100%. Option reordering also exposed instability in Chinese candidate extraction and intent. Detailed groups are stored in `results/phase2/v3/final/robustness-summary.json`.

The subsequent audit found that initial intent training covered positive examples for only 31 classes and development for only 20, along with contiguous-prefix and block-sampling bias. The earlier diminishing-return conclusion was withdrawn. The exposed Phase 2 test set is retained for historical reporting and regression. Phase 3 creates a development set with full class coverage and a new independent test, keeping the same performance thresholds. See the [coverage audit and correction](phase3-coverage-audit.md).
