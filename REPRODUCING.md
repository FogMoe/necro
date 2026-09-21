# Reproducing the archived experiments

The project is abandoned. The [retrospective](EXPERIMENT_RETROSPECTIVE_2026-09-21.md) explains the failed release objective and the experimental findings. This archive preserves the data, LoRA adapters, configurations, calibration, predictions, source snapshots, and logs needed to inspect or reproduce those findings.

## Download and verify

The archive is stored directly in Git, including datasets, adapters, tokenizers, and the two original cloud archives. Clone the repository and verify the downloaded files:

```sh
git clone https://github.com/scarletkc/necro.git
cd necro
python scripts/verify_experiment_backup.py
```

The verifier reads `EXPERIMENT_BACKUP_MANIFEST.json`, checks every archived file's SHA-256 and byte count, and fails on missing or corrupted files. It uses the Python standard library and does not load model weights or run evaluation.

The original cloud ZIP retains its checksum `df811fe8722867f42726164a448afad2d71739d6efee469cfc1744129ae9db47`. The complete cloud evidence archive retains `7dfd74727498d6ab08979c858845c3b1496b2588d3983a8838cc201394be334c`. Frozen evidence is preserved byte for byte, including historical paths and records of failed experiments.

## Contents

| Location | Contents |
|---|---|
| `data/` | Registered training, development, calibration, regression and test files, source snapshots, manifests, and generator snapshots |
| `results/lora-pilot/`, `results/improvement/` | Early adapters, training records, calibration, and evaluation |
| `results/phase2/` through `results/phase5/` | Later adapters, selections, source snapshots, diagnostics, comparisons, and failed assessments |
| `results/phase6/` | Unified preparation evidence |
| `results/cloud-primary-2026-09-21/` | Original cloud environment records, primary adapter, baselines, predictions, and diagnostics |
| `results/cloud-ops-local/` | Preserved local transfer and diagnostic records |
| `artifacts/necro-unified-cloud-2026-09-21.zip` | Original self-contained cloud preparation package with two reference adapters |
| `artifacts/necro-unified-primary-cloud-evidence-2026-09-21.tar.gz` | Original cloud evidence archive |

Merged full-model weights and their export directories are excluded. Every adapter in those local exports also exists in `results/`. Download the official base checkpoint at revision `2fc06364715b967f1860aea9cf38778875588b17` and apply the relevant saved adapter. Network download caches and Python bytecode are excluded from the archive.

Every published safetensors weight file is named `adapter_model.safetensors`. It contains the LoRA parameters, approximately 43 MB per adapter, and is required alongside the adapter configuration. The approximately 1.7 GB merged `model.safetensors` files are not included.

A failed candidate remains failed even though its weights are available. Historical test sets exposed during development are regression evidence for a new experiment. The final unified test was not evaluated by this project. Its publication provides reproducibility, not a new blind acceptance cohort for a reader who inspects it.

## Reproduce the unified cloud recipe

Extract the original cloud preparation ZIP to a new directory. Read `CLOUD_RUN.md` and `cloud-plan.json` inside it. Install the locked environment and verify its materials:

```sh
uv sync --locked --python 3.12 --extra training
uv run --extra training python -m necro.training.cloud verify
uv run --extra training python -m necro.training.cloud profile
uv run --extra training python -m necro.training.cloud baselines
uv run --extra training python -m necro.training.cloud train primary
uv run --extra training python -m necro.training.cloud assess primary
```

These commands spend GPU time. They reproduce the failed registered recipe. Use a fresh extracted workspace so the runner does not overwrite archived experiment directories. The package pins the checkpoint revision, dataset fingerprints, initialization, rank, objective, epochs, learning rate, effective batch, and seed. Its measured Linux environment is documented in the [cloud execution record](docs/process/unified-cloud-run-2026-09-21.md).

For earlier runs, use the corresponding `run_config.json`, parent adapter, dataset manifest, and saved source snapshot. The [training guide](docs/training.md) describes the entry points. Historical absolute machine paths identify original provenance and must be mapped to the matching archived relative paths on a different machine. A configuration from one phase must not be substituted for another merely because both use rank 16.

The cloud report compares baselines rerun in that environment. Historical Windows and cloud percentages differ slightly. Preserve the runtime, batch settings, calibration and request identity when comparing a reproduction. Derive new results into a separate directory and keep the archived raw predictions intact.

## Data terms and attribution

Dataset records and derived examples retain their upstream terms. They are not relicensed under the repository's code license. See [licensing and third-party notices](THIRD_PARTY_NOTICES.md#training-data) for source attribution, dataset licenses, transformations, and the distinction between training and evaluation sources. This includes XNLI's noncommercial terms and the share-alike terms of the named reading and retrieval datasets. Per-experiment manifests identify the sources actually used.

The repository's original rule examples, code and fine-tuning contributions follow the scopes in [the project notices](THIRD_PARTY_NOTICES.md). Credentials and private environment files are not part of the backup.
