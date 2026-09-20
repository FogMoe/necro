# Report and model card generation, 2026-09-20

This release used `release_report` in [release_report.py](../../src/necro/training/release/release_report.py) to assemble existing evaluation files into the final report, training record, and model cards.

## Inputs

The script reads comparisons, calibration, latency, and export validation from `results/improvement/final/`, and the manifest, test questions, and boundary questions from `data/improvement/`. Run configurations and development results come from their respective training directories. The function's read and copy paths define the full file list.

`SELECTED_RUN` is the training directory containing `run_config.json`. The second argument is the exported package directory:

```powershell
uv run --extra training python -m necro.training.release.release_report SELECTED_RUN artifacts/ScarletKc-Necro-0.8b
```

## Outputs

- `docs/reports/improvement-2026-09-20.md`: evaluation report for the exported weights.
- `docs/process/improvement-2026-09-20.md`: training runs, data, and selection history.
- `adapter/README.md` and `merged/README.md` in the package: standalone model cards.
- `evaluation/` in the package: summaries, configurations, per-question judgments, and failure examples.

After generation, synchronize documentation and runtime code using the [export and publishing guide](../publishing.md#validation-and-packaging), then update checksums.
