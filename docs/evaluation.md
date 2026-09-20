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

Each run writes three result files:

| File | Contents |
|---|---|
| `summary.json` | Overall and grouped metrics, plus run metadata |
| `predictions.jsonl` | Complete responses and expected answers for each input |
| `judgments.jsonl` | Per-question predictions, correctness, and probability metrics |

Local metadata records the checkpoint, model ID, adapter path, revision, prompt version, temperatures, library versions, device, and dataset SHA-256. CUDA runs also record the GPU name and peak allocated memory. `--limit` changes the number of evaluated records, while the data hash still covers the entire input file.

`evaluated_examples` and `full_dataset_examples` distinguish the selected records from the complete file. `full_dataset_evaluated` records whether all records were evaluated, and `evaluated_questions` counts individual judgments. These fields let comparison tools check completeness even when two runs share the same file hash.

## Data format and sources

Each JSONL line is a record containing `id`, `request`, and `expected`. Keys in `expected` match question IDs. Expected values are option names for Choice, booleans for Noul, and zero-based level indices for Score. `source` and `language` control grouping. See [smoke.jsonl](../examples/smoke.jsonl) for complete examples.

Optional `family`, `group_id`, and `label_quality` fields record the task family, shared source group, and label provenance. Results include a `families` breakdown and `macro_family_accuracy`, the unweighted mean of family accuracies. Missing family and group IDs fall back to the source and record ID respectively.

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
| `score_normalized_mae` | Mean absolute Score error normalized by each question's level range, reported in metric groups containing Score questions |
| `nll_floor_sensitivity` | NLL recomputed with alternative probability floors defined in `metrics` |
| `candidate_mass_mean`, `candidate_mass_min` | Local valid-answer probability mass, as defined in the [API](api.md#probabilities-and-scores) |

Higher accuracy is better. Lower NLL, Brier, ECE, and Score MAE are better. Read the dataset, language, and question-type breakdowns alongside the aggregate results. ECE uses the largest candidate probability. The API's `confidence` uses a separate formula.

## Registered experiments

When the dataset directory contains `experiment.json`, evaluation verifies every registered partition's hash and requires the requested file to be registered. Test evaluation also requires a frozen `selection.json`. Local test runs must use an adapter whose weight hash is allowed by that selection. If the selection includes manifest, artifact, or prompt hashes, those are verified too. These checks are implemented by `verify` in [experiment_guard.py](../src/necro/experiment_guard.py).

If the selection freezes temperatures, local test evaluation accepts raw temperature 1 for every question type or the mapping registered for that adapter's weight hash. Selected weights use `temperatures`, while reference weights use their entry in `reference_temperatures`. The check is implemented by `verify_temperatures` in [experiment_guard.py](../src/necro/experiment_guard.py).

For the recorded Phase 3 experiment, `freeze` in [freeze_phase3.py](../src/necro/training/release/freeze_phase3.py) verifies the complete training ancestry and freezes the selected and reference adapters with their own calibration. Experiment-specific preparation and selection decisions are in the [process records](process/README.md).

## Probability calibration

Fix the model first, then fit a Choice temperature on a separate calibration set. This example assumes `data/calibration.jsonl` is ready and model paths have been configured:

```powershell
$env:NECRO_TEMPERATURE = '1.0'
$env:NECRO_CHOICE_TEMPERATURE = ''
$env:NECRO_NOUL_TEMPERATURE = ''
$env:NECRO_SCORE_TEMPERATURE = ''
uv run --extra training necro evaluate data/calibration.jsonl --output results/calibration-raw
uv run --extra training python -m necro.calibration results/calibration-raw/predictions.jsonl --output results/calibration.json
$calibration = Get-Content results/calibration.json -Raw | ConvertFrom-Json
$env:NECRO_CHOICE_TEMPERATURE = $calibration.temperature.ToString([Globalization.CultureInfo]::InvariantCulture)
```

The fitting command reads Choice predictions and selects the temperature with the lowest mean NLL. Apply it, evaluate the held-out test set, and retain both settings and results. A positive temperature preserves the Choice ranking. In this example, Noul and Score use the general temperature because their overrides were cleared. For base-model evaluation, use the `inference` extra instead.

### Per-primitive calibration

`fit_primitive_temperatures` in [calibration.py](../src/necro/calibration.py) fits separate Choice, Noul, and Score temperatures from a registered calibration split. It verifies prediction order, expected labels, the dataset hash, and raw temperature metadata in the adjacent `summary.json`. The loss averages source groups within each task, then weights tasks equally. Its `objective` selects NLL or Brier. Search bounds and minimum source counts are defined by that function, and the result records `bounds` and `at_search_boundary`.

`assess` in [assess_candidate.py](../src/necro/training/analysis/assess_candidate.py) runs development and calibration evaluation, fits the selected objective, and writes calibrated development results:

```powershell
uv run --extra training python -m necro.training.analysis.assess_candidate REGISTERED_DATA_DIR ADAPTER_DIR RESULTS_DIR --objective brier
```

Replace the uppercase paths with your registered dataset, adapter, and candidate output directory. Keep the objective fixed across candidates. This command evaluates the development and calibration partitions. Final test evaluation follows model selection and freezing.

To recalculate saved predictions, use `transform_report(dataset, predictions_path, output, temperatures)` in [calibration.py](../src/necro/calibration.py). It requires raw predictions and a new output directory, verifies data and labels, and records the source prediction hash. Test predictions must also match the frozen weights and temperatures. Serve the fitted values using [exported calibration settings](running.md#applying-exported-calibration).

## Comparing complete results

Use separate result directories for the local model and reference:

```powershell
uv run python -m necro.training.analysis.phase2_analysis LOCAL_RESULTS REFERENCE_RESULTS --output COMPARISON_JSON
```

`compare` in [phase2_analysis.py](../src/necro/training/analysis/phase2_analysis.py) checks dataset identity, question alignment, and task coverage. It reports task and language breakdowns, label provenance, and source-group bootstrap intervals from `cluster_intervals`. Its acceptance fields implement the [recorded eight-task protocol](process/phase2-acceptance.md). `compare_development` in [development_comparison.py](../src/necro/training/analysis/development_comparison.py) provides paired local-candidate comparisons for development selection.

`prepare` and `summarize` in [robustness.py](../src/necro/training/analysis/robustness.py) create and analyze paired changes in candidate order, question polarity, and irrelevant context. Report both accuracy and consistency with each original answer. A consistent answer can still be incorrect.

## Latency and throughput

```powershell
uv run --extra inference necro benchmark --repeats 20
uv run --extra inference necro benchmark --repeats 20 --questions 4
```

`benchmark` runs a fixed short request in the local process. Timing starts after model loading and warmup. Output includes all latency samples, p50, p95, and loading time reported separately. Multi-question requests repeat the same question to compare batch sizes.

For `evaluate`, `elapsed_seconds` covers input preparation, inference, and response conversion for the full batch. `questions_per_second` measures batch throughput. Remote timings include network and client scheduling. Runs that reuse cached responses report `questions_per_second` as null, as described under [Jev comparison](#jev-comparison). When measuring HTTP latency, also record the request, concurrency, warmup count, and whether the endpoint is local or remote.

## Jev comparison

Set `TYPESAFE_API_KEY` in `.env`, then run:

```powershell
uv run necro evaluate data/baseline.jsonl --backend jev --output results/jev
```

This sends uncached evaluation inputs to `TYPESAFE_BASE_URL`. `evaluate_remote` in [evaluation.py](../src/necro/evaluation.py) pins the requested model name and validates the returned model and question IDs. It also defines retry behavior, including numeric `Retry-After` delays.

Successful responses are saved in the output directory's `remote-cache/`. Reusing the same output directory reuses matching responses after validation. Request fingerprints preserve option order, so reordered candidates have separate cache entries. A TypeSafe key is still required when using the cache.

Result metadata records cache hits and the total request count in `remote_cache`. If any responses were cached, `timing_scope` identifies the resumed run and throughput is left unset. Use a fresh output directory to measure a run consisting entirely of remote calls.

Both backends use the same inputs and expected answers. Inspect the saved per-question results to identify differences by task, language, or condition before choosing the next dataset or evaluation.
