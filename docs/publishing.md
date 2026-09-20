# Export and publishing

Export produces a LoRA adapter, merged weights, and runtime code. Choose a split-directory layout or a Hub layout with merged weights at the package root, then assemble the model card and evaluation files for distribution.

## Local export

Run from the project root. Replace `SELECTED_ADAPTER` and `CHOICE_TEMPERATURE` with the chosen adapter directory and calibration value:

```powershell
uv run --extra training python -m necro.export SELECTED_ADAPTER artifacts/ScarletKc-Necro-0.8b --temperature CHOICE_TEMPERATURE
```

The model ID and base checkpoint come from the adapter's `necro_adapter.json`. Export loads that checkpoint at the recorded revision and rejects an existing output directory.

To include temperatures for all question types, add `--calibration-file CALIBRATION_JSON`. Its `temperatures` mapping overrides values by question type. Entries absent from the file use the fallback values defined by `export` in [export.py](../src/necro/export.py). Unknown question types are rejected. The supplied file is copied into both model directories as `calibration.json`. See [per-primitive calibration](evaluation.md#per-primitive-calibration) to fit this mapping.

| Output | Purpose |
|---|---|
| `adapter/` | LoRA safetensors, PEFT configuration, tokenizer, and prompt contract. Loads with the original base model |
| `merged/`, or the package root with `--hub-layout` | Complete merged model, tokenizer, and `necro_model.json` |
| `runtime/` | Candidate-scoring API, training and evaluation code, dependency lockfile, tests, and documentation |
| `export.json` | Model ID, layout, merged-weight path, per-question-type temperatures, base revision, total model parameter count, and weight hashes |

File selection is defined by `export` in [export.py](../src/necro/export.py). It copies an explicit list of runtime files. Raw training data, caches, and credentials stay in the working directory.

For a package ready to assemble at a Hub repository root, add `--hub-layout`. The `adapter/` and `runtime/` directories remain alongside the merged model files. `export.json` identifies the layout and merged-weight location through `layout` and `merged_path`.

The exported `runtime/.env.example` points to the packaged merged model and contains the export's calibrated temperatures. From `runtime/`, copy it to `.env` and follow the [startup instructions](running.md#starting-the-server). The paths assume commands run from that directory.

## Validation and packaging

Load the adapter and merged weights, then compare selected answers and probabilities using the same inputs and temperature. Commands are in [model loading](running.md#loading-a-model) and [evaluation](evaluation.md). Place validation results, selection records, and calibration parameters in the package's `evaluation/` directory.

For an export with a frozen test selection and complete calibrated reference predictions, `verify_package` in [package_verification.py](../src/necro/training/release/package_verification.py) checks weight hashes, reloads the merged model on CUDA, and compares the full test set. It also starts a temporary loopback server to validate the official SDK, response structures, candidate counts, and local HTTP latency. Run it with a new output file:

```powershell
uv run --extra training python -m necro.training.release.package_verification PACKAGE_DIR REGISTERED_DATA_DIR PREDICTIONS_DIR VERIFICATION_JSON
```

The data directory must contain `selection.json`, and the prediction directory must contain complete results for the selected weights and temperatures. Output records reload differences, SDK checks, GPU memory, and latency samples. Adding `--benchmark-jev` also sends latency requests to the configured TypeSafe service.

To include an exposed regression cohort in the same merged-model check, supply both `--regression-data REGRESSION_JSONL` and `--regression-predictions REGRESSION_RESULTS_DIR`. The dataset must be registered with the `regression` role and match the hash in the frozen `validation-plan.json`. Its results must use the same selected weights and calibration. `verify_reload` in the same module checks every answer and probability; `regression_reload` records this cohort separately from the independent test.

Each standalone model card should cover purpose, run commands, training sources, evaluation conditions, results, observed errors, and licensing. Run commands should start in the model's `runtime/` directory and specify whether they load an adapter or merged weights.

The report-generation entry point for the recorded experiment is described in the [packaging record](process/publishing-2026-09-20.md). Reports contain results, while training runs and selection history live in `docs/process/`. Model cards link to the packaged report.

For condition-repair candidates, `freeze` in [freeze_candidate.py](../src/necro/training/release/freeze_candidate.py) validates development selection, calibration, training ancestry and held-out sources before writing a selection. `assess` in [stability_assessment.py](../src/necro/training/release/stability_assessment.py) checks the resulting independent condition and multi-task regression measurements against the frozen validation plan. Jev comparisons are reported separately from release checks.

After the capability review and package verification pass, generate the Hub model card, report, reproduction records and checksums:

```powershell
uv run --extra training python -m necro.training.release.stability_report REGISTERED_DATA_DIR SELECTED_DIR FINAL_RESULTS_DIR PACKAGE_DIR
```

`generate` in [stability_report.py](../src/necro/training/release/stability_report.py) verifies evidence and weight hashes, then writes `evaluation/report.md`, model cards, `release.json` and `SHA256SUMS`. It requires a new package without an existing report or model card. The [repair process record](process/condition-repair-2026-09-21.md) defines the associated cohorts and selection history.

For the split-directory layout, synchronize documentation and runtime code into the package's top-level `runtime/` after generating reports, then copy runtime and evaluation files into each standalone model directory:

```powershell
$package = 'artifacts/ScarletKc-Necro-0.8b'
Copy-Item README.md, LICENSE, LICENSE-MIT, THIRD_PARTY_NOTICES.md -Destination "$package/runtime" -Force
Get-ChildItem docs | Copy-Item -Destination "$package/runtime/docs" -Recurse -Force
Copy-Item src/necro/training/release/release_report.py -Destination "$package/runtime/src/necro/training/release" -Force
foreach ($kind in @('adapter', 'merged')) {
    Copy-Item "$package/runtime" -Destination "$package/$kind" -Recurse
    Copy-Item "$package/evaluation" -Destination "$package/$kind" -Recurse
}
```

These copy commands assume the standalone directories do not yet contain `runtime/` or `evaluation/`. For existing copies, synchronize their contents and remove obsolete documentation paths. Generate `SHA256SUMS` for the complete package after assembly, and regenerate it whenever a file changes.

For the Hub layout, keep `runtime/` and `evaluation/` at the package root and place the model card there. Upload that root as one repository. Its bundled runtime configuration already points to the merged weights in the parent directory.

## Uploading to Hugging Face

Sign in, choose an account, repository name, and visibility, then upload the prepared standalone directory. These commands create or update a remote repository:

```powershell
uv run --extra training hf auth login
uv run --extra training hf upload YOUR_NAMESPACE/ScarletKc-Necro-0.8b artifacts/ScarletKc-Necro-0.8b/merged . --private
```

Replace `YOUR_NAMESPACE` with the target account or organization. `--private` sets visibility only when creating a repository. Existing repositories keep their visibility. Use `--no-private` to create a public repository. Run `hf upload --help` for all options.

The upload example above uses the split layout's `merged/` directory. With `--hub-layout`, use the package root as the upload source instead.

## Licensing

Distribute the project's [Apache-2.0 license](../LICENSE), the training code's [MIT license](../LICENSE-MIT), and upstream licenses with the model and runtime. [Licensing and third-party notices](../THIRD_PARTY_NOTICES.md) define their scope. Model cards link to the bundled copy.
