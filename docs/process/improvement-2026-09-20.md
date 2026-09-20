# Improvement training record, 2026-09-20

These experiments continued from the [first LoRA run](lora-pilot-2026-09-20.md). Model selection followed the [predefined protocol](improvement-protocol.md). Results for the exported weights are in the [evaluation report](../reports/improvement-2026-09-20.md).

## Training and development comparison

All runs used the same base model revision, rank 16, and BF16, updating only LoRA parameters. Continued training created a fresh optimizer. The development set remained the pilot validation set.

| Experiment | Records in this run | Learning rate | Training time | Development accuracy | Development NLL |
|---|---:|---:|---:|---:|---:|
| round2 | 994 | 3e-05 | 3.77 minutes | 73.00% | 1.3562 |
| round3 | 2336 | 5e-05 | 8.38 minutes | 84.00% | 0.8219 |
| round4 | 2336 | 2e-05 | 5.51 minutes | 85.00% | 1.0229 |

round2 repeated the pilot data. round3 continued from pilot-v1, adding public training examples, balancing NLI classes, and adding insufficient-evidence triplets. round4 trained for another epoch from round3 at a lower learning rate.

This round selected `round3`. The decision is recorded in `results/improvement/selection.json`. The selection method was later examined in the [design review](design-review-2026-09-20.md#calibrated-development-comparison).

## Data

The pilot and expanded datasets contained 3,330 training records in total, or 2,432 unique contexts across 1,302 source groups. English and Chinese included corresponding translations, and continued training reused existing examples. Labels came from public annotations and rules implemented in code.

Expanded XNLI Choice data was balanced by language and class, with 176 examples per group. See the [training guide](../training.md#continuing-training-and-expanding-data) for sampling, transformations, and exclusions.

## Regression

| Data | Exported model |
|---|---:|
| Earlier diagnostic set, 400 questions | 85.75% |
| Custom development examples, 24 questions | 24/24 |

Earlier results are recorded in the [base model baseline](baseline-2026-09-20.md) and [first LoRA run](lora-pilot-2026-09-20.md). The original pilot reached 69.25% accuracy on this round's test slice.

## Reproducing continued training

Prepare the pilot data and weights using the [training guide](../training.md), then run:

```powershell
uv run --extra training python -m necro.training.data.experiments
uv run --extra training python -m necro.training.data.probes
uv run --extra training python -m necro.training --data data/improvement/expanded --output results/improvement/round3 --initial-adapter results/lora-pilot/run1/adapter --learning-rate 0.00005 --seed 29 --model-id necro-qwen3.5-0.8b-r3
```

Training configurations, logs, and per-question results are stored in `results/improvement/`. Data fingerprints are in `data/improvement/manifest.json`.
