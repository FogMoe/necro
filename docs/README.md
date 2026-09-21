# Documentation

Start with the [project README](../README.md) for a first run. Use the guides below for specific tasks.

| Task | Guide |
|---|---|
| Install dependencies, load models, and resolve startup errors | [Setup and configuration](running.md) |
| Connect to the HTTP API and interpret probabilities | [API](api.md) |
| Prepare evaluation data, compare results, and calibrate probabilities | [Evaluation](evaluation.md) |
| Prepare data and train or continue a LoRA adapter | [Training](training.md) |
| Export a model, assemble its files, and upload to the Hub | [Export and publishing](publishing.md) |
| Change the project and run checks | [Development](development.md) |
| Write, organize, and review project documentation | [Documentation standard](documentation.md) |
| Check upstream sources and licenses | [Licensing and third-party notices](../THIRD_PARTY_NOTICES.md) |

## Evaluation report

The [2026-09-21 unified primary review](reports/unified-primary-2026-09-21.md) reports the fresh-LoRA results against Phase 4, failed stability checks, and the numeric coverage omissions identified in the follow-up audit.

The [2026-09-21 pre-release review](reports/pre-release-2026-09-21.md) compares the original checkpoint, the Phase 4 candidate, and Jev on identical requests, including the condition regression that failed release review.

The [2026-09-20 ScarletKc-Necro-0.8b evaluation report](reports/improvement-2026-09-20.md) covers the exported weights, test results, calibration, observed errors, and latency.

## Process records

Training runs, selection rules, design reviews, and follow-up plans are kept in [process records](process/README.md).
