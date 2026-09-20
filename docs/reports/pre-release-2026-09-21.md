# ScarletKc-Necro-0.8b: pre-release review, 2026-09-21

Training substantially improves the original base model under the local candidate-scoring interface. Stable publication remains blocked by a condition-judgment regression: the latest repair improves disabled-gate and missing-field decisions but falsely rejects many records that satisfy all requirements.

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

| Condition slice | Questions | Direct parent | Current candidate |
|---|---:|---:|---:|
| All condition questions | 720 | 85.83% | 91.94% |
| Disabled gate | 240 | 70.00% | 95.00% |
| Missing fields | 240 | 87.92% | 96.67% |
| Complete fields and enabled gate | 240 | 99.58% | 84.17% |
| Complete fields and enabled gate, Chinese | 120 | 99.17% | 74.17% |

The complete-field decline is 15.42 percentage points, with paired 95% interval [-22.08, -9.17]. The Chinese decline is 25 points, with interval [-36.67, -13.33]. Every one of the candidate's 38 complete-field errors is a false rejection of an eligible record. This is a material behavioral regression despite the improved overall condition score.

The repair matrix contains 320 eligible and 2,880 ineligible presentations. Its answer labels are balanced at 1,600 true and 1,600 false because question polarity is varied. The underlying eligibility states remain imbalanced at 10% versus 90%. This sampling pattern is consistent with the observed false rejections; its causal contribution has not been isolated experimentally.

## Release decision and further work

The model has clear value over the untrained base under this interface. Additional epochs or more examples with the same semantic imbalance are not justified by these results. The completed review adds inference and a sampling audit, with no training or new Jev requests.

Stable publication remains blocked. If development resumes, the concrete change to investigate is balancing the underlying eligibility outcomes while preserving complete-field accuracy. The exposed condition cohort remains diagnostic evidence; acceptance would require fresh held-out expressions. No additional experiment is scheduled by this review.

The [repair handoff](../process/condition-repair-handoff-2026-09-21.md) contains the committed failure fixture, code locations, local artifact paths and reproduction steps.

The current candidate has not completed merged-export and API verification. The earlier loader inconsistency was fixed and verified on the Phase 3 package, as recorded in the [export investigation](../process/numeric-regression-2026-09-21.md). The current candidate's failed capability review still prevents stable-package generation.

## Evidence and licenses

Base and direct-parent predictions, paired comparisons, sampling counts and provenance are stored in `results/phase4/quick-release-audit/`. Trained-model and Jev predictions remain in `results/phase4/final/`. `complete-condition-regression.json` records the paired subset analysis, and `semantic-balance.json` records the matrix counts. Frozen weights and original evidence remain unchanged.

Project and fine-tuning contributions use Apache-2.0. The designated training files use MIT. Training datasets retain their individual licenses, including XNLI's CC BY-NC 4.0. [Licensing and third-party notices](../../THIRD_PARTY_NOTICES.md) define the scope and attribution.
