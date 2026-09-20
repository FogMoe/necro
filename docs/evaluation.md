# Evaluation

Compare the base model, LoRA adapters, and Jev on the same JSONL data, with model settings and data fingerprints recorded in the results. See the [evaluation report](README.md#evaluation-report) for results and [model loading](running.md#loading-a-model) for configuration.

## Running an evaluation

Run from the project root:

```powershell
uv run --extra inference necro evaluate examples/smoke.jsonl --output results/smoke
uv run --extra inference necro prepare-eval
uv run --extra inference necro evaluate data/baseline.jsonl --output results/local
```

Use `--extra training` when loading LoRA. Add `--limit` to evaluate a smaller number of records first. Full options are listed by `necro evaluate --help`. Evaluation overwrites result files with the same names in the output directory, so use separate directories for comparisons.

Each run produces three files:

| File | Contents |
|---|---|
| `summary.json` | Overall and grouped metrics, plus run metadata |
| `predictions.jsonl` | Complete responses and expected answers for each input |
| `judgments.jsonl` | Per-question predictions, correctness, and probability metrics |

Local metadata records the checkpoint, model ID, adapter path, revision, prompt version, temperatures, library versions, device, and dataset SHA-256. CUDA runs also record the GPU name and peak allocated memory. `--limit` changes the number of evaluated records, while the data hash still covers the entire input file.

## Data format and sources

Each JSONL line is a record containing `id`, `request`, and `expected`. Keys in `expected` match question IDs. Expected values are option names for Choice, booleans for Noul, and zero-based level indices for Score. `source` and `language` control grouping. See [smoke.jsonl](../examples/smoke.jsonl) for complete examples.

The bundled smoke examples cover English and Chinese, all three question types, negation, quotations, structured descriptions, and instructions embedded in input data. Expected answers follow explicit rules.

`necro prepare-eval` selects English and Chinese validation slices from these sources:

| Source | Task | Candidates |
|---|---|---|
| [facebook/xnli](https://huggingface.co/datasets/facebook/xnli) | Determine whether a premise supports, contradicts, or leaves a hypothesis unresolved | Entailment, contradiction, neutral |
| [mteb/amazon_massive_intent](https://huggingface.co/datasets/mteb/amazon_massive_intent) | Intent classification | The complete original label set |

MASSIVE uses MTEB's columnar copy, with labels defined by `INTENTS` in [datasets.py](../src/necro/datasets.py). Sampling options are defined under `prepare-eval` in [cli.py](../src/necro/cli.py). Downloading and conversion are handled by `prepare_public_eval`.

Development, calibration, and test sets use separate slices, with overlapping contexts removed by source group. Some English and Chinese records are translations of the same source, so statistical analysis groups them accordingly. See [data preparation](training.md#data-preparation) for the training workflow. Each experiment report records its slices and sample counts.

## Metrics

Metrics are implemented by `summarize` and `metrics` in [evaluation.py](../src/necro/evaluation.py).

| Metric | Calculation |
|---|---|
| `accuracy` | Whether the highest-probability class is correct. Noul uses a threshold of 0.5, with ties assigned to true. Score compares the highest-probability level |
| `nll` | Negative log probability of the correct label, using `max(p, 1e-12)` |
| `brier` | Sum of squared differences between class probabilities and the one-hot target, averaged over questions |
| `ece_10_bins` | Difference between mean top probability and accuracy in ten equal-width probability bins, weighted by bin frequency |
| `score_mae` | Mean absolute difference between the probability-weighted Score and the expected level |
| `candidate_mass_mean`, `candidate_mass_min` | Local valid-answer probability mass, as defined in the [API](api.md#probabilities-and-scores) |

Higher accuracy is better. Lower NLL, Brier, ECE, and Score MAE are better. Read the dataset, language, and question-type breakdowns alongside the aggregate results. ECE uses the largest candidate probability. The API's `confidence` uses a separate formula.

## Probability calibration

Fix the model first, then fit a Choice temperature on a separate calibration set. This example assumes `data/calibration.jsonl` is ready and model paths have been configured:

```powershell
$env:NECRO_TEMPERATURE = '1.0'
$env:NECRO_CHOICE_TEMPERATURE = ''
uv run --extra training necro evaluate data/calibration.jsonl --output results/calibration-raw
uv run --extra training python -m necro.calibration results/calibration-raw/predictions.jsonl --output results/calibration.json
$calibration = Get-Content results/calibration.json -Raw | ConvertFrom-Json
$env:NECRO_CHOICE_TEMPERATURE = $calibration.temperature.ToString([Globalization.CultureInfo]::InvariantCulture)
```

The fitting command reads Choice predictions and selects the temperature with the lowest mean NLL. Apply it, evaluate the held-out test set, and retain both settings and results. A positive temperature preserves the Choice ranking. Noul and Score continue to use the general temperature. For base-model evaluation, use the `inference` extra instead.

## Latency and throughput

```powershell
uv run --extra inference necro benchmark --repeats 20
uv run --extra inference necro benchmark --repeats 20 --questions 4
```

`benchmark` runs a fixed short request in the local process. Timing starts after model loading and warmup. Output includes all latency samples, p50, p95, and loading time reported separately. Multi-question requests repeat the same question to compare batch sizes.

For `evaluate`, `elapsed_seconds` covers input preparation, inference, and response conversion for the full batch. `questions_per_second` measures batch throughput. Remote timings include network and client scheduling. When measuring HTTP latency, also record the request, concurrency, warmup count, and whether the endpoint is local or remote.

## Jev comparison

Set `TYPESAFE_API_KEY` in `.env`, then run:

```powershell
uv run necro evaluate data/baseline.jsonl --backend jev --output results/jev
```

This sends evaluation inputs to `TYPESAFE_BASE_URL`. `evaluate_remote` in [evaluation.py](../src/necro/evaluation.py) pins the requested model name, and the actual model ID returned by the service is recorded in the results. That function also defines retry behavior for remote errors.

Both backends use the same inputs and expected answers. Inspect the saved per-question results to identify differences by task, language, or condition before choosing the next dataset or evaluation.
