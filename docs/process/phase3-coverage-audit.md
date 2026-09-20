# Phase 3 coverage correction after failed independent acceptance

Phase 2 failed the frozen acceptance thresholds. Independent-test task results, paired intervals, and probability metrics are recorded in the [Phase 2 development record](phase2-development.md#results-at-the-end-of-phase-2). Complete values are stored in `results/phase2/v3/final/comparison.json`.

The exposed Phase 2 test set and robustness variants are used for historical regression, retaining their original selection, predictions, hashes, and source snapshots. Subsequent candidates use new splits for selection and evaluation.

## Design issues

1. Intent class coverage was not enforced. The expanded 800 intent training records contained positive examples for only 31 of 60 classes. The strict development set's 100 intent records covered 20 classes, and the earlier independent test covered 26, with different class proportions. Missing classes still appeared as negative candidates, so repeated training could reinforce this bias.
2. Sampling retained effects of source ordering. The initial contiguous prefix was replaced with dispersed blocks of 50 rows, which still preserved clusters within ordered data. Sampling needs to check both individual sources and class coverage.
3. Development gains did not transfer to the independent test. Source isolation addressed overlap, while representative coverage required a separate check. The earlier diminishing-return conclusion was withdrawn pending reassessment after coverage correction.
4. A manually written class description was wrong. `cooking_query` had been described as cooking timers or duration, while official training records included meal suggestions, ingredient substitutions, and cooking methods. Phase 3 corrects the description for all compared models using the original training text.
5. Task definitions need precise records. Extraction selects supplied candidate answers, retrieval selects sampled documents, and rule tasks use programmatic labels. BoolQ and SciFact are English-only in this experiment. Results are reported for the actual task and language coverage.

## Correction plan

See the [Phase 3 data workflow](phase3-data.md) for full snapshots, sampling, and source-key implementation. Sample independent source groups, keeping translations together and excluding repeated utterances, premises, passages, questions, and all candidate documents. Protect every historical registration, including unselected training attempts and unused historical registrations.

Intent training, development, and calibration each require positive examples for all 60 classes in both English and Chinese. Reserve one independent source group per role for rare classes before filling quotas from available material. Training mixes eight-option and full 60-option requests with shuffled order. Evaluation retains the full candidate set.

New intent development and calibration data is sampled from unused official train and validation pools because official validation itself lacks one class. Official test also lacks one class, and historical source exclusion may exhaust other rare classes. Record actual test coverage and small class counts in the registration, preserving missing classes as missing.

Sample XNLI randomly by source from complete splits, balancing relation labels. Other tasks retain their definitions and use new source groups. For SciFact, exclude historical sources from the combined official train/test qrels pool, then select new queries and all candidate documents. Report this as the project's candidate-selection task. Constructed rules retain their difficulty and template definitions, with new seeds and source groups.

The first lightweight candidate continues from the frozen Phase 2 weights using rank 16 LoRA, one epoch of answer-token cross-entropy, and task replay. First measure the old model on the same new development set. Fit temperatures on an independent calibration set using the existing per-primitive method. Further attempts depend on specific development evidence and useful gains.

Before opening the new test, freeze the selected candidate, Phase 2 reference weights, and each model's own calibration. Both local models and Jev receive identical requests. Acceptance thresholds remain unchanged. If the corrected experiments meet the diminishing-return stopping criteria, record the measured gap and end exploration.

## Review entry points

- Frozen Phase 2 weights: `results/phase2/v3/selected/adapter`.
- Independent comparison: `results/phase2/v3/final/comparison.json`.
- [Acceptance protocol](phase2-acceptance.md).
- Full snapshot manifests: `data/phase3/snapshots/*/manifest.json`.
- New splits, coverage, and exclusion records: `data/phase3/coverage-v1/experiment.json`.
- Original sources: [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive), [MTEB Parquet copy](https://huggingface.co/datasets/mteb/amazon_massive_intent).
