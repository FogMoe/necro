# Bounded precondition repair, 2026-09-21

This experiment addresses the [wording-sensitive precondition regression](numeric-regression-2026-09-21.md). Scope is limited to that defect, retention of the other tasks, and export consistency, following the [release criteria](release-criteria-2026-09-21.md).

## Recipe

The initial candidate continues from instruction-seed2026 with rank-16 LoRA, answer-token cross entropy, a learning rate of 1e-5, batch size 4, gradient accumulation 2 and one epoch. Seed is 2026. The training set contains 2,464 records: 1,440 balanced condition contrasts and replay from the other task families.

Condition records cover all five comparison operators. Each source state produces an enabled gate, a disabled gate and a missing-field case, with both question polarities in English and Chinese. Passing and failing labels are balanced. Training, development and test use separate field sets, numeric source values and wording variants. `construct` and `instructions` in [condition_refinement.py](../../src/necro/training/data/condition_refinement.py) define the generated cases.

The combined development set contains the existing 1,464 task records and 720 new condition records. Its components are analyzed separately. Another 720 condition records, from 12 source groups, form the independent holdout. Natural-task cohorts already exposed in Phase 3 serve as regression evidence.

## Selection rules

The plan was recorded before training in `results/phase4/condition-repair-plan.json`:

- On the existing development cohort, each task must remain within two percentage points of both the predecessor and instruction parent. Language slices and paired source-group intervals are reported.
- New-condition development accuracy must retain both baselines. Disabled-gate and missing-field accuracy must each reach 95%.
- The original-wording numeric regression must recover predecessor accuracy. Disabled-gate and missing-field accuracy must each reach 90%.
- The independent condition holdout stays sealed until candidate selection and calibration are frozen.

The initial plan permitted two trials. The combination audit below justified a corrective matrix trial, followed by one smaller-update comparison for task retention. Each amendment was recorded before the corresponding run. Ordinary unrelated errors remain reportable outcomes.

## Baselines

On the 720 new development conditions, the predecessor scored 97.36% and the instruction parent scored 96.11%. These figures are distinct from the original-wording diagnostic in the linked investigation.

Dataset hashes and source-group counts are in `data/phase4/condition-v1/experiment.json`. The original builder is saved alongside that manifest. Training configuration, raw predictions and calibration belong to the corresponding run directory in `results/phase4/`.

## Initial repair result

Training took 340.44 seconds for 308 optimizer steps, with peak allocation of 3.572 GiB. Disabling the adapter reproduced the base-model probe exactly.

| Check | Instruction parent | Initial repair |
|---|---:|---:|
| New condition development | 96.11% | 98.19% |
| Original-wording regression | 70.93% | 90.70% |
| Original disabled-gate cases | 4/40 | 36/40 |
| Original missing-field cases | 4/12 | 5/12 |
| Existing numeric development | 91.48% | 89.20% |
| Existing eight-task development macro | 87.98% | 87.69% |

The initial repair failed the missing-field requirement and the two-point task-retention margin. Its complete assessment is `results/phase4/condition-seed2026/repair-assessment.json`.

## Registered follow-up

The second planned trial uses `condition-v2`, with 2,996 training records. Two formulations express required fields implicitly, paired gate and missing-field cases use identical rule text, and 532 previously trained numeric records provide retention replay. The parent, learning rate, seed and other training settings remain unchanged.

Development, calibration and the sealed test are byte-identical to v1. The first dataset and run remain preserved. The selection rules above also remain unchanged. The prospective record is `results/phase4/condition-followup-plan.json`.

## Combination audit after the second trial

The second trial reached 93.60% on the original-wording regression, including 40/40 disabled-gate cases. Missing fields remained at 8/12, and existing numeric development accuracy was 87.50%, so the release checks still failed.

The cross-product audit found no training example combining the implicit guard-first formulation, equality, a missing gate and an otherwise true comparison. More general missing-field counts had concealed that omission. The audit is `results/phase4/missing-combination-audit.json`.

The corrective dataset, `condition-v3`, covers all 800 combinations of four styles, five operators, two languages, two question polarities, five presence/gate conditions and two counterfactual comparison outcomes. Each combination has four presentations. The generated matrix contains 3,200 records, supplemented by the retained replay. Training totals 4,756 presentations and 4,436 unique requests. Some counterfactual pairs become identical inputs after an operand is removed, and remain repeated presentations within the same source group.

`audit_condition_matrix` rejects incomplete or unbalanced combinations before registration. The development, calibration and sealed test files remain byte-identical. The prospective correction is recorded in `results/phase4/condition-matrix-plan.json`. It retains the instruction parent and reduces the learning rate to 5e-6. This correction addresses the confirmed condition defect and task retention under the same release criteria.

## Complete-matrix result

The matrix trial took 613.29 seconds for 595 optimizer steps and peaked at 3.611 GiB of allocated GPU memory. New condition development accuracy reached 98.06%, with 240/240 disabled gates and 240/240 missing fields correct. The original-wording regression reached 166/172 (96.51%), including 40/40 disabled gates and 12/12 missing fields.

Existing numeric development remained at 157/176 (89.20%), compared with 161/176 (91.48%) for both retained baselines. The 2.27-point decline exceeded the registered two-point margin. Every other development task remained inside that margin. The trial therefore failed selection despite repairing the original condition cases.

The smaller-update comparison keeps the complete matrix, parent, seed and all evaluation cohorts unchanged, and halves the learning rate to 2.5e-6. Its prospective record is `results/phase4/condition-retention-plan.json`. The final validation plan records independent condition checks and exposed multi-task regression separately before opening the holdout. Both plans are retained in the candidate freeze.

## Smaller-update comparison and selection

The smaller update took 641.09 seconds, with the same 595 optimizer steps and 3.611 GiB peak allocation. New condition development remained at 706/720 (98.06%). Original-wording regression reached 165/172 (95.93%), including 40/40 disabled gates and 12/12 missing fields. Existing eight-task development macro accuracy was 87.63%, compared with 87.51% for the larger matrix update and 87.98% for the instruction parent.

The existing numeric development result remained 157/176. The original two-point numeric check therefore failed again. The larger update's net four-question loss consisted of five newly wrong answers and one correction; the new errors involved the less-than operator, which was already weak in the parent. The paired source-group interval did not establish equivalence or a reliable improvement from halving the learning rate.

Selection accepted this numeric-development tradeoff before opening the independent condition holdout. The smaller update was chosen for its better overall retention, the recovered gate and missing-field cases, and the limited observed benefit from another training adjustment. The other seven tasks retain their two-point development margin. Final independent condition checks, complete multi-task regression, export, reload and API validation remain required.

`results/phase4/condition-v3-smallstep-seed2026/selection-review.json` records the amendment and links the original failed assessment by hash. The freeze preserves both records as `decision.json` and `preliminary-review.json`. The original numeric result remains a failed preliminary check. Additional training is outside this closing comparison; remaining ordinary errors belong in the measured report.
