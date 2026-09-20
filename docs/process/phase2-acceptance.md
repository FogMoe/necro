# Phase 2 acceptance and stopping criteria

This protocol was recorded before Phase 2 training and sealed-test predictions. The base model is fixed at Qwen3.5-0.8B and the reference service at Jev 1.13.0. Lightweight experiments end under the performance thresholds or diminishing-return criteria below.

## Data and comparisons

- `data/phase2/experiment-v3` is the registered data snapshot. It contains 726 development records across eight task families, 2,582 training records, and 477 calibration records. At registration, the sealed set contained 775 records covering six new tasks. Inference and intent require additional unused official test records to be registered before any final predictions.
- Earlier test data is used for historical regression only. Isolate by source, passage, bilingual translation, and counterfactual group, and report actual group counts. Test data must not select starting weights, loss, temperature, or training data.
- Report natural human annotations, derived candidate answers, exact rule oracles, and candidate retrieval with unjudged negatives separately. Candidate extraction and passage selection are evaluated within their supplied candidate sets.
- Describe English and Chinese coverage by task. BoolQ and SciFact use English sources.

## Development

1. Use round4 as the common starting point. Its eight-task macro accuracy on the new development set was 74.55%, compared with 74.29% for round3. The earlier calibrated comparison also favored round4. This choice uses development data only.
2. Start both initial experiments from round4 with rank 16, learning rate 3e-5, seed 2026, batch 4, accumulation 2, the same training set, and one epoch. Compare only answer-ce and candidate-ce. The latter retains full-vocabulary answer loss for multi-token labels.
3. Compare task-macro accuracy, individual tasks, language groups, and normalized Score MAE. Candidates generally need a macro gain of at least 2 percentage points, with no decline above 3 percentage points on either original inference or intent. A clear key-capability improvement may be assessed separately, with regressions still reported.
4. Repeat beneficial settings with another seed. Use the same independent calibration slice and method. Evaluate final data and Jev only after freezing selection.
5. Base subsequent experiments on specific development evidence, changing one major factor at a time. If two consecutive targeted approaches each gain less than 1 percentage point and fail to repair a key task weakness, record diminishing returns and end lightweight exploration.

## Performance acceptance

The final same-question test must meet all thresholds: eight-task macro accuracy within 3 percentage points of Jev, every core task within 5 percentage points, Score MAE normalized by the level span no higher than Jev plus 0.05, and task-macro Brier no higher than Jev plus 0.03. Report language results, source-group paired bootstrap intervals, and differences in probability precision.

Also record robustness to option order, negation, and missing input, along with protocol tests, save/reload validation, latency, and memory measurements. Remote latency includes network time.

## Completion records

At completion, assemble the measured report, failure cases, model cards, LoRA and merged weights, licenses, dependencies, and hashes. Record performance acceptance and stopping decisions separately, retaining the selected weights and reproduction materials.
