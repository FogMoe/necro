# Semantic condition repair, 2026-09-21

This experiment continues the [false-rejection handoff](condition-repair-handoff-2026-09-21.md). It separates semantic eligibility from the requested answer polarity, preserves the established task families, and keeps new condition transfer separate from exposed regression evidence.

## Design and controlled comparison

The local interface scores answer labels directly. Training and inference share the same prompt and answer token boundaries. The reported defect changes the winning answer, so temperature calibration alone cannot repair it.

The previous matrix covers every registered combination, but only 320 of its 3,200 presentations are eligible. Question negation balances Yes/No without balancing the record's eligibility. The 38 known false rejections span both polarities, both held-out styles and all five operators. Chinese accounts for 31 failures. Within genuinely eligible records, the previous candidate scores 66.07% overall and 44.64% in Chinese.

The controlled weighting experiment preserves the matrix requests, labels, order, replay records, initial adapter, seed, objective, optimizer steps and effective batch size. Each eligible matrix presentation receives weight 5; each ineligible presentation receives weight 5/9. Both semantic outcomes therefore contribute total weight 1,600. The 1,556 replay presentations retain weight one, and total weight remains 4,756.

`audit_balance` in [condition_balance.py](../../src/necro/training/data/condition_balance.py) checks semantic balance within each joint wording/operator/language/polarity cell, full presence/gate coverage, and unchanged replay weight. This changes the loss distribution, not the physical presentation counts. Complete records are now more heavily weighted than missing-field and disabled-gate records; selection must check both false rejections and false acceptances rather than assume weighting is sufficient.

The weighted loss uses a dataset-wide unit mean. Microbatches retain their weights during gradient accumulation. Normalizing each microbatch separately would erase weighting for single-record batches. Tests compare full-batch and unequal-microbatch gradients for both answer objectives.

## Recipe and prospective rules

The source is `results/phase3/instruction-seed2026/adapter`, with rank 16, seed 2026, one epoch, answer-token cross entropy, microbatch four and accumulation two. The first weighting trial uses learning rate 2.5e-6. A registered 5e-6 comparison measures whether a larger update repairs remaining slices or merely creates a retention tradeoff.

The data registry is `data/phase5/condition-balance-v1/experiment.json`. The recipe, selection rules and runtime requirements were recorded before completed repair training in `results/phase5/repair-plan.json` and `final-validation-plan.json`. Each failed assessment remains preserved. New test predictions cannot select weights or temperatures.

Selection includes all 38 known failures, complete-field and eligible-record accuracy by language, missing-field and disabled-gate accuracy, existing development tasks, exposed multi-task regression and language alerts. [condition_checks](../../src/necro/training/analysis/condition_balance.py) implements the condition checks. A missing required slice fails assessment.

New development contains 1,600 condition questions. A separate sealed test contains 4,800 questions from 24 source groups, with new wording, field names and numerical values, including negative values and fractional boundaries. Languages, polarities and counterfactual cases within a source group are correlated presentations. Uncertainty uses source groups, not 4,800 independent observations. The old 720-question cohort and 38 failures are exposed regression evidence.

## Training performance

Short profiles measured approximately 0.51 seconds per microbatch with checkpointing and 0.22 seconds without it. The latter used 5.53 GiB on the sampled batches. Batch eight without checkpointing exhausted memory. A batch-four control attempt was stopped after its first optimizer step when a longer batch saturated memory; it produced no adapter or development results.

The performance amendment preserves batch four, accumulation two and gradient checkpointing for complete repair runs. The previous small-step run remains the unweighted reference with those settings. The trainer now checks the largest padded batch before updating weights, as short-batch timing does not establish peak memory safety. Profile failures and the amendment remain under `results/phase5/`.

## Weighting result and sampling follow-up

Weighting corrected 11 of the 38 known failures, leaving 27 false rejections. On the exposed 720-question cohort, accuracy rose from 91.94% to 93.33%; complete-field accuracy rose from 84.17% to 88.75%, and its Chinese slice from 74.17% to 80.00%. English disabled-gate accuracy was 107/120 and missing-field accuracy 112/120. These results fail the registered repair checks.

On new development expressions, overall condition accuracy was 1,550/1,600. Eligible-record accuracy was only 68/80 in English and 69/80 in Chinese. Existing numeric development fell to 156/176, also failing retention against the instruction parent. No independent test was opened. The complete failed assessment remains in `results/phase5/balanced/repair-assessment.json`.

The next registered intervention changes physical sampling. `resample` selects 1,600 eligible and 1,600 ineligible matrix presentations, with unit loss weights, preserving the 1,556 replay records and the total training budget. Each joint wording/operator/language/polarity cell contains twenty presentations of each semantic outcome; all 800 original combinations retain at least two presentations. The matrix contains 1,820 unique requests, so these are repeated training exposures, not new independent examples. Development, calibration and the sealed test remain byte-identical.

The sampling amendment is `results/phase5/sampling-amendment.json`; its dataset is `data/phase5/condition-resample-v1`. The learning rate stays at 2.5e-6 for this comparison. Failure on the short exposed condition screen stops a candidate before expensive full evaluation; passing that screen still requires all development, retention and runtime checks. The weighting trial provides evidence that semantic balance contributes to the defect.

The direct parent scored 68/80 English and 70/80 Chinese eligible records on the newly constructed development set. Every one of these parent errors involved strict comparisons of negative numbers. Weighting retained the same twelve English errors and ten Chinese errors, adding one Chinese error. This new numerical stress differs from the original repair matrix's positive values. A prospective retention amendment separates inherited arithmetic errors from the confirmed condition regression: the original 38 failures and positive-valued condition regression keep their repair requirements; new complete/eligible slices must retain the parent within two points, while new gate and missing-field slices must still reach 95% per language. The original absolute 95% failure remains recorded.

## Sampling result and change of approach

Physical resampling corrected 10/38 known failures. The exposed condition screen scored 93.33% overall, 88.33% on complete records and 78.33% on Chinese complete records. English gate and missing-field slices both scored 91.67%. The candidate failed the short screen, so no full development, calibration or eight-task assessment was run for it. Neither Phase 5 candidate was selected; the sealed test remains unopened.

The weighting candidate's eight-task exposed macro accuracy was 89.5821%, versus 89.43% for the Phase 4 reference. The weighting candidate still failed the condition and original numeric-development checks.

Further repair-only continuations have stopped. The next experiment starts a fresh LoRA on the official post-trained Qwen3.5-0.8B with unified, source-protected multi-task training. The user selected this initialization and deferred Base to a later comparison. See the [cloud preparation record](unified-retraining-2026-09-21.md).

## Reproduction

Use a new output directory for every run. The weighting dataset is prepared from the preserved Phase 4 training data:

```powershell
uv run --extra training python -m necro.training.data.condition_balance
uv run --extra training python -m necro.training --data data/phase5/condition-balance-v1 --output results/phase5/balanced-reproduction --initial-adapter results/phase3/instruction-seed2026/adapter --learning-rate 2.5e-6 --batch-size 4 --accumulation 2 --seed 2026
uv run --extra training python -m necro.training.analysis.assess_candidate data/phase5/condition-balance-v1 results/phase5/balanced-reproduction/adapter results/phase5/balanced-reproduction --objective brier
```

Data preparation rejects an existing directory. Reuse the registered data when reproducing training. Exact run configurations, source snapshots, data hashes, losses, profiles, calibration fits and raw predictions are retained locally under the corresponding run directories.
