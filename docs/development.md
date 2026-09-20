# Development

Install dependencies and run checks from the project root:

```powershell
uv sync --extra training
uv run --extra training pytest -q
uv run --extra training ruff check src tests
uv run --extra training ruff format --check src tests
```

Automated tests use small models, mock scorers, or local test clients. Measure model performance with the [evaluation workflow](evaluation.md).

## Code entry points

| Area | Entry point |
|---|---|
| CLI and configuration | `main` in [cli.py](../src/necro/cli.py), `Settings` in [config.py](../src/necro/config.py) |
| HTTP API and schemas | `create_app` in [api.py](../src/necro/api.py), [schema.py](../src/necro/schema.py) |
| Prompts and response conversion | `Task.messages`, `prepare_task`, and `answer_for` in [engine.py](../src/necro/engine.py) |
| Model loading and candidate scoring | `TransformersScorer` in [backend.py](../src/necro/backend.py) |
| Evaluation and temperature fitting | `run_evaluation` in [evaluation.py](../src/necro/evaluation.py), `fit_temperature` in [calibration.py](../src/necro/calibration.py) |
| Training and export | `train` in [trainer.py](../src/necro/training/trainer.py), `export` in [export.py](../src/necro/export.py) |

## Training code layout

Training tools live in `src/necro/training/`. Run their module commands from the project root so relative data and result paths resolve consistently.

```text
src/necro/
├── training/
│   ├── __main__.py         # python -m necro.training
│   ├── trainer.py          # Encoding, loss, batching, and the LoRA loop
│   ├── adapter_average.py  # Combine compatible adapter factors
│   ├── data/               # Source snapshots and dataset builders
│   ├── analysis/           # Candidate assessment and paired comparisons
│   └── release/            # Selection freezes, reports, and package checks
├── experiment_guard.py     # Shared partition and selection verification
├── evaluation.py           # Shared evaluation runner and metrics
└── export.py               # Adapter and merged-model packaging
tests/
└── training/               # Training, data-builder, and selection tests
```

Use `data/training_data.py` for pilot preparation and `data/phase2_data.py` or `data/phase3_data.py` for the corresponding experiment datasets. Source acquisition lives in `data/data_catalog.py` and `data/full_snapshots.py`; targeted refinement builders stay alongside the datasets they produce. `analysis/assess_candidate.py` evaluates candidates before the freeze procedures in `release/` permit final testing.

The training command remains `python -m necro.training`. Other tools use their package path, for example `python -m necro.training.data.training_data` or `python -m necro.training.analysis.assess_candidate`. See the [training guide](training.md) for the pilot workflow and [process records](process/README.md) for phase-specific commands.

Generated datasets and source caches belong in `data/`, run outputs and predictions in `results/`, and distributable model packages in `artifacts/`. These root directories are ignored by Git. Keep registered experiment materials at the paths recorded in their manifests.

## Updating documentation

Each guide has a specific purpose in the [documentation index](README.md). Guides, dated evaluation reports, configuration comments, and new process records use English. Keep experiment history and design reviews in `docs/process/`. Reports describe the measured model and link to process records for candidate selection and training attempts. Link defaults and field limits to their definitions in code.

`release_report` in [release_report.py](../src/necro/training/release/release_report.py) generates the Phase 1 report and standalone cards. `generate` in [phase3_report.py](../src/necro/training/release/phase3_report.py) generates the Phase 3 report and Hub package cards. `export` in [export.py](../src/necro/export.py) writes the bundled configuration example. Update the applicable generator when editing its output, then check relative links and checksums in that export. Frozen evidence snapshots retain their original content.
