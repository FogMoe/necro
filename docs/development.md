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
| Training and export | `train` in [training.py](../src/necro/training.py), `export` in [export.py](../src/necro/export.py) |

## Updating documentation

Each guide has a specific purpose in the [documentation index](README.md). Keep experiment settings, measurements, and conclusions in dated reports. Link defaults and field limits to their definitions in code.

`release_report.py` contains templates for evaluation reports and standalone model cards. Update the templates when editing generated content, then check documentation copies, relative links, and checksums in the export.
