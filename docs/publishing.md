# Export and publishing

Export produces a LoRA adapter, merged weights, and runtime code. Once the model cards and evaluation files are ready, either the adapter or merged directory can be uploaded as a standalone model repository.

## Local export

Run from the project root. Replace `SELECTED_ADAPTER` and `CHOICE_TEMPERATURE` with the chosen adapter directory and calibration value:

```powershell
uv run --extra training python -m necro.export SELECTED_ADAPTER artifacts/ScarletKc-Necro-0.8b --temperature CHOICE_TEMPERATURE
```

The model ID comes from the adapter's `necro_adapter.json`. `NECRO_MODEL` must match `checkpoint` in that contract. Export rejects an existing output directory.

| Output | Purpose |
|---|---|
| `adapter/` | LoRA safetensors, PEFT configuration, tokenizer, and prompt contract. Loads with the original base model |
| `merged/` | Complete merged model, tokenizer, and `necro_model.json` |
| `runtime/` | Candidate-scoring API, training and evaluation code, dependency lockfile, tests, and documentation |
| `export.json` | Model ID, Choice temperature, base revision, and weight hashes |

File selection is defined by `export` in [export.py](../src/necro/export.py). It copies an explicit list of runtime files. Raw training data, caches, and credentials stay in the working directory.

## Validation and packaging

Load the adapter and merged weights, then compare selected answers and probabilities using the same inputs and temperature. Commands are in [model loading](running.md#loading-a-model) and [evaluation](evaluation.md). Place validation results, selection records, and calibration parameters in the package's `evaluation/` directory.

Each standalone model card should cover purpose, run commands, training sources, evaluation conditions, results, observed errors, and licensing. Run commands should start in the model's `runtime/` directory and specify whether they load an adapter or merged weights.

The report-generation entry point for the recorded experiment is described in the [packaging record](process/publishing-2026-09-20.md). Reports contain results, while training runs and selection history live in `docs/process/`. Model cards link to the packaged report.

After generating reports, synchronize documentation and runtime code into the package's top-level `runtime/`, then copy runtime and evaluation files into each standalone model directory:

```powershell
$package = 'artifacts/ScarletKc-Necro-0.8b'
Copy-Item README.md, LICENSE, LICENSE-MIT, THIRD_PARTY_NOTICES.md -Destination "$package/runtime" -Force
Get-ChildItem docs | Copy-Item -Destination "$package/runtime/docs" -Recurse -Force
Copy-Item src/necro/release_report.py -Destination "$package/runtime/src/necro" -Force
foreach ($kind in @('adapter', 'merged')) {
    Copy-Item "$package/runtime" -Destination "$package/$kind" -Recurse
    Copy-Item "$package/evaluation" -Destination "$package/$kind" -Recurse
}
```

These copy commands assume the standalone directories do not yet contain `runtime/` or `evaluation/`. For existing copies, synchronize their contents and remove obsolete documentation paths. Generate `SHA256SUMS` for the complete package after assembly, and regenerate it whenever a file changes.

## Uploading to Hugging Face

Sign in, choose an account, repository name, and visibility, then upload the prepared standalone directory. These commands create or update a remote repository:

```powershell
uv run --extra training hf auth login
uv run --extra training hf upload YOUR_NAMESPACE/ScarletKc-Necro-0.8b artifacts/ScarletKc-Necro-0.8b/merged . --private
```

Replace `YOUR_NAMESPACE` with the target account or organization. `--private` sets visibility only when creating a repository. Existing repositories keep their visibility. Use `--no-private` to create a public repository. Run `hf upload --help` for all options.

## Licensing

Distribute the project's [Apache-2.0 license](../LICENSE), the training code's [MIT license](../LICENSE-MIT), and upstream licenses with the model and runtime. [Licensing and third-party notices](../THIRD_PARTY_NOTICES.md) define their scope. Model cards link to the bundled copy.
