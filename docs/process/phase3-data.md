# Phase 3 data workflow

This page records code entry points for full dataset snapshots and Phase 3 sampling. See the [training guide](../training.md) for training and the [Phase 2 workflow](phase2-workflow.md#data-and-registration) for the preceding source-isolation method.

The [coverage audit](phase3-coverage-audit.md) explains coverage issues and sampling changes.

## Dependencies and full snapshots

Full Parquet snapshots require the project's `data` extra. Subsequent data preparation also uses training modules and the local tokenizer. Install from the project root:

```powershell
uv sync --extra training --extra data
```

`SPECS` in [full_snapshots.py](../../src/necro/training/data/full_snapshots.py) defines datasets and configurations. `ROOT` defines the local snapshot directory. This command downloads the configured Parquet files from Hugging Face:

```powershell
uv run --extra data python -m necro.training.data.full_snapshots
```

Append names from `SPECS` to prepare only those sources. `acquire` writes the conversion branch's exact revision to `revision.json` and records each file's split, row count, size, and SHA-256 in `manifest.json`. If a manifest already exists, it first checks local file hashes. `rows` verifies files, then reads batches with row indices within the complete split.

## Sampling and source isolation

`prepare` in [phase3_data.py](../../src/necro/training/data/phase3_data.py) reads full snapshots and existing experiment materials. XQuAD and SciFact still use snapshots prepared by `download` in [data_catalog.py](../../src/necro/training/data/data_catalog.py). The model tokenizer must already be cached locally.

`historical` defines the scope of historical material scanning, and `ROOT` defines Phase 3 output paths. `SourceIndex.keys` links duplicate utterances, questions, premises, passages, and candidate documents through source relationships. Sampling uses `take` to exclude protected sources. `paired_public` checks bilingual records and labels. `intent_partition` and `coverage` handle class quotas and missing classes.

Once all inputs are ready, run:

```powershell
uv run --extra training --extra data python -m necro.training.data.phase3_data
```

An existing output directory causes an error. Construction checks cross-split source overlap and uses the local tokenizer to check training input lengths.

## Registered outputs

Outputs are saved as JSONL by role, with development written to `validation.jsonl`. `experiment.json` stores split hashes and source-group audits, along with historical input files, full snapshot manifests, task composition, intent class coverage, and training input lengths.

The registration defines actual task counts and coverage. It references the acceptance protocol by path and SHA-256. See the [evaluation guide](../evaluation.md#registered-experiments) for validation rules.

## Candidate assessment and comparison

`assess` in [assess_candidate.py](../../src/necro/training/analysis/assess_candidate.py) runs CUDA inference on registered development and calibration splits, fits temperatures, and produces calibrated development results. Before reusing raw results, it checks data hashes, weight hashes, complete-evaluation markers, and temperatures. Its three positional arguments are the data directory, adapter directory, and output directory:

```powershell
uv run --extra training python -m necro.training.analysis.assess_candidate DATA_DIR ADAPTER_DIR OUTPUT_DIR
```

`compare_development` in [development_comparison.py](../../src/necro/training/analysis/development_comparison.py) compares a local candidate and baseline on the same complete development set, returning task metrics and source-group paired intervals. Once results exist, run:

```powershell
uv run python -m necro.training.analysis.development_comparison CANDIDATE_RESULTS BASELINE_RESULTS --output COMPARISON_JSON
```

The comparison output file must be new and its parent directory must already exist. Performance acceptance continues to use the independent-test comparison workflow.

## Phase 3 freeze

`freeze` in [freeze_phase3.py](../../src/necro/training/release/freeze_phase3.py) registers the selected model and local reference weights together, binding each to its own calibration. It checks the relationship between test and robustness materials, existing test predictions, selection rationale, ancestral training data, and source isolation.

The module's argparse definitions list required path arguments. The selection-rationale file must explicitly record `new_test_exposed` and `rationale`. Frozen outputs include the adapter, both calibrations, selection rationale, lineage audit, and frozen materials. Each evaluation directory receives `selection.json`. Existing output or selection files cause an error.

See the [publishing guide](../publishing.md#validation-and-packaging) for exported-weight reload and HTTP validation.
