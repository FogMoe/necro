# Phase 2 experiment workflow

This page records entry points for the Phase 2 scripts. The [acceptance protocol](phase2-acceptance.md) defines comparison methods and stopping criteria. Result fields are documented in [Evaluation](../evaluation.md).

Measurements, calibration changes, and replication are recorded in the [development record](phase2-development.md).

## Data and registration

`prepare` in [phase2_data.py](../../src/necro/training/data/phase2_data.py) constructs development-stage data, with the output path supplied through `--output`. `snapshot` in [data_catalog.py](../../src/necro/training/data/data_catalog.py) saves natural-source snapshots.

`prepare` in [phase2_final.py](../../src/necro/training/data/phase2_final.py) combines new-task tests with additional original-task records. `prepare_calibration` fills missing calibration tasks. The module's `ROOT`, `OUTPUT`, and function path definitions identify inputs and outputs. After registration, `verify` in [experiment_guard.py](../../src/necro/experiment_guard.py) checks file hashes and model selection.

`prepare` in [source_isolation.py](../../src/necro/training/data/source_isolation.py) handles structural relationships among original sentences, translations, and candidate documents. The [source isolation audit](phase2-source-audit.md) records the reason and scope of revised registration.

Final-set registration includes the acceptance protocol's SHA-256. Protocol revisions must first account for the corresponding registration, retaining the original protocol and its references.

<a id="primitive-calibration"></a>

## Per-primitive calibration

Phase 2 fits NLL with equal source-group and task weights. The [development record](phase2-development.md#operator-contrast-results) documents the evidence and settings for extending the search grid. Functions, input validation, and offline recomputation are documented in the [per-primitive calibration guide](../evaluation.md#per-primitive-calibration).

## Paired comparisons

See the [comparison guide](../evaluation.md#comparing-complete-results) for complete-result commands, fields, and paired intervals.

To reuse existing same-question predictions after source cleanup, use `project_report` in the same module. It verifies the original data hash, requests, and expected labels, then filters and regroups predictions into a new result directory. Metadata retains `projected_from` and the original prediction hash. Timing fields still describe the complete evaluation before cleanup.

## Freezing model selection

`freeze` in [freeze_phase2.py](../../src/necro/training/release/freeze_phase2.py) defines this phase's freeze procedure. It follows adapter parent records to verify the complete training-data lineage, checks isolation between training sources and development, calibration, and test slices, and verifies the correspondence between robustness questions and the final set.

Frozen outputs include the selected adapter, calibration file, `lineage-audit.json`, and source and protocol copies in `frozen-materials/`. Each evaluation directory's `selection.json` binds the weights, temperatures, prompt fingerprint, experiment manifest, and material hashes. The function defines input paths and candidate selection. Existing frozen output causes an error.

After freezing, run tests under the [registered-experiment validation rules](../evaluation.md#registered-experiments). Frozen materials retain their original contents, and subsequent documentation links to the corresponding records.

## Robustness and supplementary training data

`prepare` in [robustness.py](../../src/necro/training/analysis/robustness.py) constructs variants. `summarize` compares original and variant results, recording accuracy and consistency by variant type, task, and language.

`contrast_rules` and `prepare` in [phase2_refinement.py](../../src/necro/training/data/phase2_refinement.py) prepare targeted contrastive rule data. Templates and input/output paths remain in the script. Training records identify the data actually used.

`construct` in [operator_contrast.py](../../src/necro/training/data/operator_contrast.py) switches comparison operators for the same state. `prepare` combines these related examples with other-task replay and registers the dataset. `prepare` in [reading_refinement.py](../../src/necro/training/data/reading_refinement.py) adds BoolQ reading material, filtering by source, duplicate questions, and input length before combining it with existing-task replay. Actual training and comparisons are in the [development record](phase2-development.md).

## LoRA parameter averaging

`average` in [adapter_average.py](../../src/necro/training/adapter_average.py) verifies common parent weights, training data, major recipe settings, prompt contract, and PEFT configuration, then computes arithmetic means separately for corresponding LoRA A and B factors. Computation uses FP32 and restores the original dtype when saving. The output directory must be new.

The output `necro_adapter.json` records the composition sources. Parent-weight hashes and the construction-script fingerprint are saved in `composition.json` one directory above the output, alongside a `source/` directory containing the construction script. See the [development record](phase2-development.md#seed-replication-and-parameter-averaging) for comparisons across seeds and parameter averaging.
