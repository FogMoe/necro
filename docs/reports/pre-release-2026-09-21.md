# ScarletKc-Necro-0.8b: pre-release review, 2026-09-21

The Phase 4 candidate improved accuracy over the original Qwen3.5-0.8B checkpoint under the local candidate-scoring interface. It failed release review because the condition repair improved disabled-gate and missing-field decisions while introducing false rejections of eligible records.

## Before training, after training and Jev

Before training means the original Qwen3.5-0.8B checkpoint, revision `2fc06364715b967f1860aea9cf38778875588b17`, with no adapter. After training means the frozen candidate with adapter SHA-256 `f0ae9537edb9478b26752db9ea3287f61ae8fdf0a8a84d453e04c93679d7d092`. Both local models use the same direct candidate-scoring prompt. The base uses unit temperatures; the candidate uses its frozen per-primitive calibration. These tables report label accuracy.

All three models answered identical requests within each cohort. The 1,790-question multi-task cohort was already exposed before this review. The 720-question condition cohort was held out when the candidate was frozen. Base-model and direct-parent comparisons were added after that evaluation for diagnosis; they did not select new weights.

| Cohort | Questions | Before training | After training | Jev 1.13.0 |
|---|---:|---:|---:|---:|
| Eight-task regression, task macro | 1,790 | 49.04% | 89.43% | 92.34% |
| Condition transfer | 720 | 34.44% | 91.94% | 98.33% |

| Task | Questions | Before training | After training | Jev 1.13.0 |
|---|---:|---:|---:|---:|
| Intent | 458 | 20.52% | 80.79% | 87.55% |
| Inference | 480 | 39.38% | 79.79% | 82.92% |
| Boolean reading | 160 | 65.62% | 85.00% | 94.38% |
| Candidate extraction | 80 | 63.75% | 85.00% | 91.25% |
| Numeric rules | 172 | 48.84% | 95.93% | 96.51% |
| Paraphrase | 180 | 56.11% | 89.44% | 86.11% |
| Candidate retrieval | 50 | 80.00% | 100.00% | 100.00% |
| Ordinal rules | 210 | 18.10% | 99.52% | 100.00% |

The paired source-group 95% interval for the trained-minus-base task-macro gain is [37.45, 43.35] percentage points. The corresponding condition-transfer interval is [52.92, 61.81] points. These measurements use the candidate-scoring interface, with no generated explanation.

English includes BoolQ and SciFact; Chinese covers the other six task families. Extraction selects among supplied candidate answers, and retrieval uses sampled documents with unjudged negatives. The [validation record](../process/condition-validation-2026-09-21.md) describes cohort coverage, intermediate checkpoints and the original release checks.

## Remaining release blocker

The direct parent is the instruction checkpoint immediately before the final repair. It is distinct from the untrained base above.

| Condition slice | Questions | Direct parent | Phase 4 candidate |
|---|---:|---:|---:|
| All condition questions | 720 | 85.83% | 91.94% |
| Disabled gate | 240 | 70.00% | 95.00% |
| Missing fields | 240 | 87.92% | 96.67% |
| Complete fields and enabled gate | 240 | 99.58% | 84.17% |
| Complete fields and enabled gate, Chinese | 120 | 99.17% | 74.17% |

The complete-field decline is 15.42 percentage points, with paired 95% interval [-22.08, -9.17]. The Chinese decline is 25 points, with interval [-36.67, -13.33]. Every one of the candidate's 38 complete-field errors is a false rejection of an eligible record. This is a material behavioral regression despite the improved overall condition score.

The repair matrix contains 320 eligible and 2,880 ineligible presentations. Its answer labels are balanced at 1,600 true and 1,600 false because question polarity is varied. The underlying eligibility states remain imbalanced at 10% versus 90%. This sampling pattern is consistent with the observed false rejections; its causal contribution has not been isolated experimentally.

## Release decision

The Phase 4 candidate failed capability review on complete-field condition judgments, stopping release before merged-export and API verification. The [repair handoff](../process/condition-repair-handoff-2026-09-21.md) contains the failure fixture, reproduction steps, and proposed sampling intervention. Subsequent experiments are indexed in the [process records](../process/README.md#condition-regression-and-repair).

The earlier loader inconsistency was fixed and verified on the Phase 3 package, as recorded in the [export investigation](../process/numeric-regression-2026-09-21.md).

## Evidence and licenses

Base and direct-parent predictions, paired comparisons, sampling counts and provenance are stored in `results/phase4/quick-release-audit/`. Trained-model and Jev predictions remain in `results/phase4/final/`. `complete-condition-regression.json` records the paired subset analysis, and `semantic-balance.json` records the matrix counts. Frozen weights and original evidence remain unchanged.

[Licensing and third-party notices](../../THIRD_PARTY_NOTICES.md) define the project, training-code, model, and dataset licenses and attribution.
