# Model selection protocol, 2026-09-20

This protocol records the Phase 1 selection rules established before opening the sealed test results. Execution is documented in the [training record](improvement-2026-09-20.md), with subsequent findings in the [design review](design-review-2026-09-20.md).

## Data roles

Starting from pilot-v1, compare training runs on `data/lora-pilot/validation.jsonl`. Use the earlier diagnostic set for final regression checks and evaluate new test-split slices after model selection. Remove source groups that overlap training and test data before evaluation.

Fit the Choice temperature only on a separate calibration slice. Treat English and Chinese records from the same source as one source group.

## Experiment sequence

First, continue the pilot for one epoch on the same data at a lower learning rate. Then try additional public training examples, balanced inference classes, and contrastive examples with insufficient evidence. A version with better results may receive another epoch at a lower learning rate.

## Promotion and stopping

A development accuracy gain of at least 2 percentage points is the practical threshold for further experiments. A version with a gain below 2 percentage points and worse NLL is not promoted. Stop expanding training after two consecutive improvement directions fail to meet that threshold.

Selection considers overall accuracy, task and language groups, NLL, and Noul and Score regressions on development examples.

## Records

Save the weights, configuration, and results for each run. Record the final selection and its rationale in `results/improvement/selection.json`. Assemble the selected weights and evaluation materials using the [export and publishing guide](../publishing.md).
