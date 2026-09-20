# ScarletKc-Necro-0.8b evaluation report, 2026-09-20

This report evaluates the exported `ScarletKc-Necro-0.8b` weights. On a test slice of 374 XNLI and MASSIVE questions, accuracy was 78.61%, compared with 81.28% for Jev 1.13.0 on the same questions, a gap of 2.67 percentage points.

The weights come from `round3`. See the [training record](../process/improvement-2026-09-20.md) for their lineage and development-set comparisons, and [setup and configuration](../running.md) to run the model.

## Test data and results

The test data uses the first 100 rows per language from each dataset's test split. Removing 13 source groups whose contexts overlapped candidate training, development, or calibration data left 374 questions. English and Chinese include records from the same sources. Test results were opened after model selection.

| Group | Questions | Necro | Jev 1.13.0 |
|---|---:|---:|---:|
| facebook/xnli/en/choice | 100 | 78.00% | 89.00% |
| mteb/amazon_massive_intent/en/choice | 87 | 83.91% | 85.06% |
| facebook/xnli/zh/choice | 100 | 71.00% | 66.00% |
| mteb/amazon_massive_intent/zh/choice | 87 | 82.76% | 86.21% |

A paired bootstrap over source groups with 5,000 resamples gives a 95% interval of [-7.75, 2.41] percentage points for the accuracy difference, Necro minus Jev. See [evaluation metrics](../evaluation.md#metrics) for definitions.

## Probability calibration

A Choice temperature of 1.721103 was selected by NLL on 198 questions from a separate validation slice after overlap removal. On the held-out test set, NLL decreased from 0.9582 to 0.7230, and ECE changed from 0.1340 to 0.0444.

| Metric | Necro | Jev API output |
|---|---:|---:|
| NLL | 0.7230 | 1.8249 |
| Brier | 0.2991 | 0.2886 |
| ECE | 0.0444 | 0.1018 |

Jev returned some zero probabilities. NLL uses `max(p, 1e-12)`, so output precision and clipping affect the contribution of low-probability answers.

The calibration value is included in the export. See [applying exported calibration](../running.md#applying-exported-calibration) to use it.

## Boundary questions and errors

The 64 boundary questions cover Noul amount and status checks, four-level Score questions, and exact matching across different candidate counts. Necro answered 56/64 correctly, and Jev answered 64/64 correctly.

Errors comprised seven Noul amount comparisons and one Chinese Score threshold question. For example, the model judged a payment settled when the amount paid was 77, the amount due was 78, and the status field was settled. In another case, it assigned two failed checks to the level for three to four failures.

Per-question requests and errors are included in `evaluation/probe-failures.jsonl`.

## Latency

The same short Choice request was run serially 20 times after warmup on an RTX 5070 Ti Laptop GPU. Local measurements use loopback HTTP. Jev measurements use remote HTTPS and include network and service overhead.

| Service | p50 | p95 |
|---|---:|---:|
| Necro | 38.72 ms | 70.97 ms |
| Jev 1.13.0 | 350.34 ms | 453.98 ms |

## Weight and API validation

The LoRA weights occupy approximately 41.34 MiB. The merged safetensors file is 1,706,030,528 bytes, approximately 1.59 GiB. Reloading the merged export and comparing predictions across the test set produced identical selections, with a maximum probability difference of approximately 2.28e-7.

HTTP calls through the official Python SDK 0.7.0 verified the model name, all three response types, and Chinese legend content. At the time of this evaluation, 55 automated tests passed.

## Supporting records

- Base model revision: `2fc06364715b967f1860aea9cf38778875588b17`.
- Test-set SHA-256: `9ddf80506ff04aa7ac0bf520a84c7b4df1acb496b64740f027a259e72480e3e1`.
- Complete local results: `results/improvement/final/`.
- Packaged summaries, judgments, and validation records: `evaluation/`.
- [Training record](../process/improvement-2026-09-20.md) and [sources and licensing](../../THIRD_PARTY_NOTICES.md).
