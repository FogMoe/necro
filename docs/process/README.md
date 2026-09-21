# Process records

Experiment plans, investigations, selection decisions, and execution records are organized below by stage. Use the [documentation index](../README.md) for usage guides and evaluation reports.

## Unified retraining

| Record | Contents |
|---|---|
| [Retraining preparation](unified-retraining-2026-09-21.md) | Fresh LoRA initialization, unified data, portable runner, and registered comparisons |
| [Cloud execution](unified-cloud-run-2026-09-21.md) | Environment, profiling, execution changes, validation stages, and post-run audit |

## Condition regression and repair

| Record | Contents |
|---|---|
| [Release criteria](release-criteria-2026-09-21.md) | Condition repair, task retention, runtime verification, and reporting |
| [Numeric-rule diagnosis](numeric-regression-2026-09-21.md) | Checkpoint comparisons, preconditions, wording, and field-name effects |
| [Precondition repair](condition-repair-2026-09-21.md) | Condition coverage, task retention, and candidate selection |
| [Repair validation](condition-validation-2026-09-21.md) | Independent condition transfer and comparisons with intermediate checkpoints |
| [False-rejection handoff](condition-repair-handoff-2026-09-21.md) | Reproduction fixture, sampling mechanism, and repair direction |
| [Semantic balance trials](condition-balance-repair-2026-09-21.md) | Controlled weighting and resampling, failed checks, and reproduction |

## Earlier experiments

| Record | Contents |
|---|---|
| [Base model baseline](baseline-2026-09-20.md) | Results before training and label-scoring experiments |
| [First LoRA run](lora-pilot-2026-09-20.md) | Pilot data, recipe, cost, and validation |
| [Improvement runs](improvement-2026-09-20.md) | Continued training, development comparisons, and regression results |
| [Model selection protocol](improvement-protocol.md) | Phase 1 selection rules established before opening test results |
| [Design review](design-review-2026-09-20.md) | Selection corrections, data issues, and the Phase 2 plan |
| [Phase 2 acceptance protocol](phase2-acceptance.md) | Comparison methods, acceptance criteria, and stopping rules |
| [Phase 2 source isolation audit](phase2-source-audit.md) | Shared sentences, candidate documents, and revised registration |
| [Phase 2 development record](phase2-development.md) | Development results, calibration changes, and independent test results |
| [Phase 2 workflow](phase2-workflow.md) | Data registration, calibration, paired comparisons, and robustness |
| [Phase 3 data workflow](phase3-data.md) | Full snapshots, historical source exclusion, and class coverage |
| [Phase 3 coverage audit](phase3-coverage-audit.md) | Coverage issues and evidence for sampling changes |
| [Phase 3 development record](phase3-development.md) | Coverage correction, calibration objectives, and paired instruction training |

## Supporting references

| Record | Contents |
|---|---|
| [Phase 1 report generation](publishing-2026-09-20.md) | Inputs and output paths for the recorded release |
| [Open-source implementations](references.md) | External code and methods |
