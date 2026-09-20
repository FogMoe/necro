# First LoRA run, 2026-09-20

Training Qwen3.5-0.8B for one epoch on 994 English and Chinese examples increased accuracy on a 200-question validation set from 38.5% to 72.0%. Parameter updates took 239.33 seconds.

## Data

The train splits of XNLI and MASSIVE each supplied 200 examples per language. One quarter of XNLI examples were converted to Noul. Half of MASSIVE used eight-option subsets containing the correct answer, while the rest retained all 60 classes. Choice options were shuffled.

Another 100 pairs of rule examples each flipped one relevant Boolean field. Removing 3 source groups that overlapped the old diagnostic set or new validation set left 994 training examples: 694 Choice, 200 Noul, and 100 Score, with 497 in each language.

Validation used rows 100–149 of each source's validation split in each language, totaling 200 questions. MASSIVE retained all 60 classes. See [data preparation](../training.md#data-preparation) for transformation and isolation methods.

## Recipe and cost

| Item | Setting or measurement |
|---|---|
| GPU | RTX 5070 Ti Laptop, 12 GB |
| Base model and computation | Qwen3.5-0.8B, BF16 |
| LoRA | Rank 16, alpha 32, dropout 0.05 |
| Trainable parameters | 10,822,656 |
| Batching | Microbatch 2, accumulation 4, effective batch size 8 |
| Learning rate | `1e-4`, linear decay after 10% warmup |
| Gradient clipping | 1.0 |
| Epochs and optimizer steps | 1 epoch, 125 steps |
| Input tokens | 305,640, excluding padding |
| Maximum input length | Limit 2048, observed maximum 874 |
| Parameter-update time | 239.33 seconds |
| Peak PyTorch allocated training memory | 2.38 GiB |
| Adapter weights | 43,346,432 bytes, approximately 41.34 MiB |

The objective was full-vocabulary cross-entropy on correct answer tokens, with teacher forcing for multi-token labels. Training used length buckets and gradient checkpointing. Before training, 6 microbatches were measured through forward and backward passes. After training, disabling the adapter produced zero difference from the original probe logits. See the [training guide](../training.md) for the full procedure.

## Validation set

| Group | Base model | LoRA |
|---|---:|---:|
| XNLI English | 38% | 64% |
| XNLI Chinese | 38% | 64% |
| MASSIVE English | 40% | 78% |
| MASSIVE Chinese | 38% | 82% |

The model corrected 96 previously wrong answers and introduced 29 errors on previously correct answers, a net gain of 67. NLL fell from 2.2871 to 1.0389 and Brier from 0.7484 to 0.3944. ECE changed from 0.1338 to 0.1360.

## Earlier diagnostic set and development examples

| Data | LoRA |
|---|---:|
| XNLI English, 100 questions | 65% |
| XNLI Chinese, 100 questions | 62% |
| MASSIVE English, 100 questions | 72% |
| MASSIVE Chinese, 100 questions | 79% |
| Public slices combined, 400 questions | 69.50% |
| Custom development examples, 24 questions | 23/24 |

Base model and same-question Jev results are in the [baseline record](baseline-2026-09-20.md#judgment-results). Regression-set NLL was 1.0969, Brier was 0.4379, and ECE was 0.1677. XNLI NLL alone regressed slightly.

## Inference validation

The saved adapter was loaded in a fresh process and merged in memory. Inference took 14.86 seconds for 200 questions and 32.56 seconds for 400 questions. A short single question was run 20 times after warmup, with p50 of 38.18 ms and p95 of 42.69 ms.

HTTP calls through the official Python SDK 0.7.0 verified the model ID `necro-qwen3.5-0.8b-lora-pilot-v1`, all three response types, and Chinese legend content. An exact-match example with 255 options returned a complete distribution and selected the expected `item254`. All 48 automated checks passed in this run.

## Raw records

Training records, weights, before-and-after comparisons, latency, and HTTP checks are stored in `results/lora-pilot/`. `run1/run_config.json` records the base model revision and prompt contract. `run1/training_summary.json` records training results.

The training SHA-256 is `baba10584b1b21d616ee3a68d860fb0e4ceb75c771ea3e9a28554eb76ff79cf8`. The validation SHA-256 is `3cafef6f57d547cc2f9ca5d6a5aecd8f1135d879618534cb0b7fbe15faf6fb8b`. Subsequent experiments are documented in the [improvement record](improvement-2026-09-20.md).
