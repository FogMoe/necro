# Unified retraining cloud execution, 2026-09-21

This run follows the [unified retraining plan](unified-retraining-2026-09-21.md), creating a fresh LoRA from the official post-trained Qwen3.5-0.8B. The workflow separates development assessment, candidate freezing, independent testing, and export validation.

The primary run completed training and development assessment, then failed promotion on condition behavior and a task-retention check. Measurements and the decision are in the [primary evaluation report](../reports/unified-primary-2026-09-21.md).

## Environment and materials

The prepared ZIP supplied the cloud workspace. Its SHA-256 was `df811fe8722867f42726164a448afad2d71739d6efee469cfc1744129ae9db47`, and cloud file verification passed. The original package and raw logs are retained.

The measured environment used an RTX 4090 D, driver 580.105.08, Ubuntu 22.04.5, Python 3.12.14, PyTorch 2.10.0+cu128, Transformers 5.17.0, and PEFT 0.21.0, with BF16 computation. The container had a 16-core CPU quota and PyTorch defaulted to 64 threads. Saved environment logs and `run_config.json` record the full configuration.

The first model download failed with an Xet 401 response before GPU profiling. Disabling Xet allowed all ten runtime files at the specified revision to download and pass verification. Subsequent runs used the offline cache. The public download used anonymous access.

## Performance measurements

The original locked environment completed forward/backward profiles and largest-padded-batch checks at microbatch sizes 4, 8, and 16. It selected microbatch 16 and accumulation 1, retaining effective batch 16 and gradient checkpointing. Peak allocated profile memory was 9.07 GiB. The profile estimated approximately 33 minutes for two epochs.

The two earlier complete local repair runs each took approximately 19–20 minutes for about 910,000 tokens. This run contains 6,098,918 unpadded tokens, approximately 6.7 times that volume. During the first hundred steps, local eight-record steps took approximately 1.63–2.17 seconds each, while cloud sixteen-record steps averaged approximately 1.38 seconds. These batch sizes and token volumes define the timing comparison. The earlier 0.22-second local profile disabled checkpointing and failed on longer batches, so complete local runs retained checkpointing.

Optional acceleration kernels were probed in a separate environment. On 135 exposed examples, answers changed on four records for the official model and three for the Phase 4 reference. The instruction parent's answers were unchanged. Because the kernels changed decisions, that exploration stopped and the primary recipe retained the original locked environment. Exploration logs and predictions are stored separately from candidate-selection evidence.

## Execution and validation

The primary recipe retained its registered training parameters. The three baseline evaluations completed after training and before candidate assessment. Completed official-model development results were reused after checking the data, model provenance, and original row mapping. Each stage saved logs and exit codes in separate outputs.

The preparation package omitted the `examples/` required for export, and the earlier freeze entry point depended on historical data outside the package. A supplement provides those examples and [`unified_release`](../../src/necro/training/release/unified_release.py), retaining the original ZIP and checksum manifest. Its stages are:

- `freeze`: verify development acceptance, training parameters, weights, and calibration provenance, then freeze the decision, validation rules, code, and reference weights.
- `test`: evaluate the frozen candidate and two references, recording eight-task results, language and condition slices, and paired source-group intervals. A failed test closes selection on that cohort.
- `package`: after independent acceptance passes, export merged weights, check reload consistency on the complete independent test and exposed regression data, and run real SDK/HTTP validation.

Each stage records its own outcome and evidence.

## Post-run investigation

The [evaluation report](../reports/unified-primary-2026-09-21.md#diagnostic-evidence) records the relation-coverage audit, equal-value representation audit, and two fixed-weight diagnostics. They used exposed records, preserved the original evaluation outputs, and did not open the sealed test.

The preparation audit checked deduplication, labels, source isolation, language coverage, and semantic category counts. It omitted the cross-product of operators with all three operand relations, and it omitted equivalent integer/float representations. Training and new development reused similar offset logic, so their omissions coincided. The preflight review accepted that audit without independently checking these dimensions. This was a data-design and review failure that should have been detected before training.

Further training is deferred until a revised data and validation proposal is reviewed. The proposal should address three requirements:

1. Audit each operator against `a < b`, `a = b`, and `a > b`, numeric representations, and their language, polarity, gate, and field-presence combinations.
2. Design development and acceptance coverage independently of the training generator. Preserve existing frozen evidence and register changed data separately.
3. Register per-epoch checkpoints and development inspection points before future training, so comparisons of update size and training duration have preserved intermediate evidence. Independent tests remain reserved for frozen candidates.

No lower-rate run, additional epoch, or additional seed started after the primary assessment. The failed candidate was not frozen, independently tested, exported, or published.
