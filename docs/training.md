# LoRA training

The training script uses PyTorch and PEFT to update LoRA parameters in the language backbone. Training and serving share the prompt template and tokenizer, with supervision on the correct answer tokens. Recipes and results for individual runs are kept in the [process records](process/README.md), including the [unified retraining plan and portable Linux GPU package](process/unified-retraining-2026-09-21.md). The [development guide](development.md#training-code-layout) maps training modules to their directories.

## Data preparation

Install the training dependencies, then prepare the diagnostic set and pilot data:

```powershell
uv sync --extra training
uv run necro prepare-eval
uv run --extra training python -m necro.training.data.training_data
```

Create `data/baseline.jsonl` first so data preparation includes it in overlap removal. Pilot output contains `train.jsonl`, `validation.jsonl`, cached source data, and `manifest.json`.

Public records come from the English and Chinese training splits of XNLI and MASSIVE. Some XNLI records become entailment questions in Noul format. Some MASSIVE records use smaller candidate sets containing the correct class, while others keep the complete label set. Choice options are shuffled, and constructed Noul and Score pairs are added.

If a context overlaps protected evaluation data, its entire source group is removed. Sampling and conversion are defined by `prepare`, `rule_pairs`, and `audit_disjoint` in [training_data.py](../src/necro/training/data/training_data.py). Actual counts, sources, and file hashes are written to `manifest.json`.

Preparation reuses cached sources and rewrites the training file and manifest. Use `--output` to select a separate directory for a new dataset. Sources and licenses are listed in [third-party notices](../THIRD_PARTY_NOTICES.md).

Full-split Parquet processing uses the `data` extra defined in [pyproject.toml](../pyproject.toml). The snapshot and sampling procedure is recorded in the [third-phase data workflow](process/phase3-data.md).

## Training and before/after evaluation

This example starts from the base model and clears calibration overrides before the comparison:

```powershell
$env:NECRO_MODEL = 'Qwen/Qwen3.5-0.8B'
$env:NECRO_ADAPTER = ''
$env:NECRO_TEMPERATURE = '1.0'
$env:NECRO_CHOICE_TEMPERATURE = ''
$env:NECRO_NOUL_TEMPERATURE = ''
$env:NECRO_SCORE_TEMPERATURE = ''
uv run --extra training necro evaluate data/lora-pilot/validation.jsonl --output results/lora-pilot/before
uv run --extra training python -m necro.training --output results/lora-pilot/run1
$env:NECRO_ADAPTER = 'results/lora-pilot/run1/adapter'
uv run --extra training necro evaluate data/lora-pilot/validation.jsonl --output results/lora-pilot/after
uv run --extra training necro evaluate data/baseline.jsonl --output results/lora-pilot/after-baseline
uv run --extra training necro evaluate examples/smoke.jsonl --output results/lora-pilot/after-smoke
```

Training rejects an existing output directory. Use a new directory for each run, and keep before/after evaluation results separate.

Records use the [evaluation JSONL format](evaluation.md#data-format-and-sources). The training directory must contain both `train.jsonl` and `validation.jsonl`. Their separation is checked before training. Oversized records raise an error. The length limit is defined by `encode_example` in [trainer.py](../src/necro/training/trainer.py).

For a [registered experiment](evaluation.md#registered-experiments), training verifies the `train` and `development` partitions before reading records.

## Training method

The base model is frozen and loaded in BF16. LoRA is applied to Linear layers in the language backbone. Inputs are bucketed by length, and gradient checkpointing is enabled by default. Set the training duration with `--epochs`. Each epoch is reshuffled deterministically, with warmup followed by linear learning-rate decay across the run. `--expected-revision` rejects an unexpected base checkpoint before parameter updates.

Select the objective with `--objective`. Both modes mask prompt tokens from the loss.

| Objective | Calculation |
|---|---|
| `answer-ce` | Full-vocabulary cross entropy on correct answer tokens |
| `candidate-ce` | Cross entropy over candidates for single-token answers, with full-vocabulary answer loss for multi-token numeric labels |

Multi-token answers use teacher forcing, summing token-level negative log likelihoods. Optional positive `training_weight` values multiply each record's answer loss. Their dataset-wide mean must equal one. Gradient accumulation averages by record count, preserving the weights across microbatches; it does not normalize each microbatch's weights separately.

A few forward and backward passes are timed before parameter updates begin, followed by a memory check on the largest padded batch. Use `--profile-only` to stop before weight updates and `--no-gradient-checkpointing` to measure the alternative memory/speed tradeoff. After training, the adapter is disabled and base-model logits are checked with the same probe.

Default settings are defined by `train` in [trainer.py](../src/necro/training/trainer.py). Run `python -m necro.training --help` for command-line options. The actual settings are saved in `run_config.json`.

| Output | Contents |
|---|---|
| `run_config.json` | Base revision, prompt fingerprint, training settings, data hashes, and initial adapter |
| `profile.json` | Performance measurements before parameter updates |
| `training_summary.json` | Training time, loss, memory use, and base-model probe comparison |
| `adapter/` | LoRA weights, PEFT configuration, tokenizer, and `necro_adapter.json` |

## Continuing training and expanding data

`--initial-adapter` continues from an existing LoRA adapter with a new optimizer. Loading checks the base model, revision, rank, and prompt contract.

The expanded-data workflow uses the pilot data and diagnostic set:

```powershell
uv run --extra training python -m necro.training.data.experiments
uv run --extra training python -m necro.training.data.probes
```

`experiments` prepares expanded training data, calibration data, and held-out test files. It balances XNLI classes, mixes MASSIVE candidate-set sizes, and adds insufficient-evidence examples. Source ranges and group exclusion rules are defined by `download_sources` and `prepare_improvement` in [experiments.py](../src/necro/training/data/experiments.py). An existing expanded directory is rejected.

`probes` generates boundary questions for amounts, status, Score thresholds, and candidate counts. It rejects an existing output file. Historical continuation commands and selection rules are in the [training record](process/improvement-2026-09-20.md).

## Reviewing results

Compare before/after results on the same questions, including breakdowns by task, language, and question type. Keep the training configuration, raw predictions, calibration files, and test-set fingerprint.

See [model loading](running.md#loading-a-model) to serve an adapter, [probability calibration](evaluation.md#probability-calibration) to fit a temperature, and [export and publishing](publishing.md) to package weights.
