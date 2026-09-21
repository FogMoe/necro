"""Generate reports and model cards from recorded experiment artifacts."""

# Keep Markdown table rows and paragraphs on complete lines for output review.
# ruff: noqa: E501

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from necro.evaluation import read_examples
from necro.export import copy_model_licenses


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def percentage(value):
    return f"{100 * value:.2f}%"


def release_report(selected: Path, package: Path):
    final = Path("results/improvement/final")
    calibration = read(final / "calibration.json")
    manifest = read("data/improvement/manifest.json")
    config = read(selected / "run_config.json")
    export = read(package / "export.json")
    local_speed = read(final / "benchmark-local.json")
    jev_speed = read(final / "benchmark-jev.json")
    summaries = {
        name: read(final / name / "summary.json")
        for name in (
            "test",
            "test-raw",
            "pilot-test",
            "jev-test",
            "regression",
            "smoke",
            "probes",
            "jev-probes",
        )
    }
    final_accuracy = summaries["test"]["overall"]["accuracy"]
    jev_accuracy = summaries["jev-test"]["overall"]["accuracy"]
    groups = {}
    examples = read_examples(Path("data/improvement/test-sealed.jsonl"))
    ours = read_examples(final / "test" / "judgments.jsonl")
    theirs = read_examples(final / "jev-test" / "judgments.jsonl")
    assert (
        [row["id"] for row in examples]
        == [row["id"] for row in ours]
        == [row["id"] for row in theirs]
    )
    for example, a, b in zip(examples, ours, theirs, strict=True):
        groups.setdefault(example["group_id"], []).append(int(a["correct"]) - int(b["correct"]))
    differences = np.array([np.mean(values) for values in groups.values()])
    rng = np.random.default_rng(2026)
    bootstrap = rng.choice(differences, size=(5000, len(differences)), replace=True).mean(axis=1)
    ci = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    comparison = {
        "selected_model": export["model_id"],
        "test_accuracy": final_accuracy,
        "jev_test_accuracy": jev_accuracy,
        "gap_percentage_points": 100 * (jev_accuracy - final_accuracy),
        "paired_group_bootstrap_necro_minus_jev_95_ci": ci,
        "source_groups": len(groups),
        "selection": str(selected),
        "choice_temperature": calibration["temperature"],
    }
    (final / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    trials = [("pilot-v1", Path("results/lora-pilot/run1"), Path("results/lora-pilot/after"))]
    trials += [
        (
            f"round{i}",
            Path(f"results/improvement/round{i}"),
            Path(f"results/improvement/round{i}/dev"),
        )
        for i in range(2, 10)
        if Path(f"results/improvement/round{i}/dev/summary.json").is_file()
    ]
    trial_rows = []
    for name, run, evaluation in trials:
        cfg, history, metrics = (
            read(run / "run_config.json"),
            read(run / "training_summary.json"),
            read(evaluation / "summary.json")["overall"],
        )
        trial_rows.append(
            f"| {name} | {cfg['examples']} | {cfg['learning_rate']:g} | "
            f"{history['training_seconds'] / 60:.2f} minutes | {percentage(metrics['accuracy'])} | {metrics['nll']:.4f} |"
        )
    details = "\n".join(
        f"| {group} | {values['count']} | {percentage(values['accuracy'])} | "
        f"{percentage(summaries['jev-test']['groups'][group]['accuracy'])} |"
        for group, values in summaries["test"]["groups"].items()
    )
    raw = summaries["test-raw"]["overall"]
    calibrated = summaries["test"]["overall"]
    model_id = export["model_id"]
    report = f"""# ScarletKc-Necro-0.8b evaluation report, 2026-09-20

This report evaluates the exported `{model_id}` weights. On a test slice of {manifest["test_examples"]} XNLI and MASSIVE questions, accuracy was {percentage(final_accuracy)}, compared with {percentage(jev_accuracy)} for Jev 1.13.0 on the same questions, a gap of {100 * (jev_accuracy - final_accuracy):.2f} percentage points.

The weights come from `{selected.name}`. See the [training record](../process/improvement-2026-09-20.md) for their lineage and development-set comparisons, and [setup and configuration](../running.md) to run the model.

## Test data and results

The test data uses the first 100 rows per language from each dataset's test split. Removing 13 source groups whose contexts overlapped candidate training, development, or calibration data left {manifest["test_examples"]} questions. English and Chinese include records from the same sources. Test results were opened after model selection.

| Group | Questions | Necro | Jev 1.13.0 |
|---|---:|---:|---:|
{details}

A paired bootstrap over source groups with 5,000 resamples gives a 95% interval of [{ci[0] * 100:.2f}, {ci[1] * 100:.2f}] percentage points for the accuracy difference, Necro minus Jev. See [evaluation metrics](../evaluation.md#metrics) for definitions.

## Probability calibration

A Choice temperature of {calibration["temperature"]:.6f} was selected by NLL on {manifest["calibration_examples"]} questions from a separate validation slice after overlap removal. On the held-out test set, NLL decreased from {raw["nll"]:.4f} to {calibrated["nll"]:.4f}, and ECE changed from {raw["ece_10_bins"]:.4f} to {calibrated["ece_10_bins"]:.4f}.

| Metric | Necro | Jev API output |
|---|---:|---:|
| NLL | {calibrated["nll"]:.4f} | {summaries["jev-test"]["overall"]["nll"]:.4f} |
| Brier | {calibrated["brier"]:.4f} | {summaries["jev-test"]["overall"]["brier"]:.4f} |
| ECE | {calibrated["ece_10_bins"]:.4f} | {summaries["jev-test"]["overall"]["ece_10_bins"]:.4f} |

Jev returned some zero probabilities. NLL uses `max(p, 1e-12)`, so output precision and clipping affect the contribution of low-probability answers.

The calibration value is included in the export. See [applying exported calibration](../running.md#applying-exported-calibration) to use it.

## Boundary questions and errors

The 64 boundary questions cover Noul amount and status checks, four-level Score questions, and exact matching across different candidate counts. Necro answered {round(summaries["probes"]["overall"]["accuracy"] * 64)}/64 correctly, and Jev answered {round(summaries["jev-probes"]["overall"]["accuracy"] * 64)}/64 correctly.

Errors comprised seven Noul amount comparisons and one Chinese Score threshold question. For example, the model judged a payment settled when the amount paid was 77, the amount due was 78, and the status field was settled. In another case, it assigned two failed checks to the level for three to four failures.

Per-question requests and errors are included in `evaluation/probe-failures.jsonl`.

## Latency

The same short Choice request was run serially 20 times after warmup on an RTX 5070 Ti Laptop GPU. Local measurements use loopback HTTP. Jev measurements use remote HTTPS and include network and service overhead.

| Service | p50 | p95 |
|---|---:|---:|
| Necro | {local_speed["p50_ms"]:.2f} ms | {local_speed["p95_ms"]:.2f} ms |
| Jev 1.13.0 | {jev_speed["p50_ms"]:.2f} ms | {jev_speed["p95_ms"]:.2f} ms |

## Weight and API validation

The LoRA weights occupy approximately 41.34 MiB. The merged safetensors file is 1,706,030,528 bytes, approximately 1.59 GiB. Reloading the merged export and comparing predictions across the test set produced identical selections, with a maximum probability difference of approximately 2.28e-7.

HTTP calls through the official Python SDK 0.7.0 verified the model name, all three response types, and Chinese legend content. At the time of this evaluation, 55 automated tests passed.

## Supporting records

- Base model revision: `{config["revision"]}`.
- Test-set SHA-256: `{manifest["test_sha256"]}`.
- Complete local results: `results/improvement/final/`.
- Packaged summaries, judgments, and validation records: `evaluation/`.
- [Training record](../process/improvement-2026-09-20.md) and [sources and licensing](../../THIRD_PARTY_NOTICES.md).
"""
    report_path = Path("docs/reports/improvement-2026-09-20.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    history = f"""# Improvement training record, 2026-09-20

These experiments continued from the [first LoRA run](lora-pilot-2026-09-20.md). Model selection followed the [predefined protocol](improvement-protocol.md). Results for the exported weights are in the [evaluation report](../reports/improvement-2026-09-20.md).

## Training and development comparison

All runs used the same base model revision, rank 16, and BF16, updating only LoRA parameters. Continued training created a fresh optimizer. The development set remained the pilot validation set.

| Experiment | Records in this run | Learning rate | Training time | Development accuracy | Development NLL |
|---|---:|---:|---:|---:|---:|
{chr(10).join(trial_rows[1:])}

round2 repeated the pilot data. round3 continued from pilot-v1, adding public training examples, balancing NLI classes, and adding insufficient-evidence triplets. round4 trained for another epoch from round3 at a lower learning rate.

This round selected `{selected.name}`. The decision is recorded in `results/improvement/selection.json`. The selection method was later examined in the [design review](design-review-2026-09-20.md#calibrated-development-comparison).

## Data

The pilot and expanded datasets contained 3,330 training records in total, or 2,432 unique contexts across 1,302 source groups. English and Chinese included corresponding translations, and continued training reused existing examples. Labels came from public annotations and rules implemented in code.

Expanded XNLI Choice data was balanced by language and class, with 176 examples per group. See the [training guide](../training.md#continuing-training-and-expanding-data) for sampling, transformations, and exclusions.

## Regression

| Data | Exported model |
|---|---:|
| Earlier diagnostic set, 400 questions | {percentage(summaries["regression"]["overall"]["accuracy"])} |
| Custom development examples, 24 questions | {round(summaries["smoke"]["overall"]["accuracy"] * 24)}/24 |

Earlier results are recorded in the [base model baseline](baseline-2026-09-20.md) and [first LoRA run](lora-pilot-2026-09-20.md). The original pilot reached {percentage(summaries["pilot-test"]["overall"]["accuracy"])} accuracy on this round's test slice.

## Reproducing continued training

Prepare the pilot data and weights using the [training guide](../training.md), then run:

```powershell
uv run --extra training python -m necro.training.data.experiments
uv run --extra training python -m necro.training.data.probes
uv run --extra training python -m necro.training --data data/improvement/expanded --output results/improvement/round3 --initial-adapter results/lora-pilot/run1/adapter --learning-rate 0.00005 --seed 29 --model-id necro-qwen3.5-0.8b-r3
```

Training configurations, logs, and per-question results are stored in `results/improvement/`. Data fingerprints are in `data/improvement/manifest.json`.
"""
    history_path = Path("docs/process/improvement-2026-09-20.md")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(history, encoding="utf-8")
    evaluation_dir = package / "evaluation"
    evaluation_dir.mkdir(exist_ok=True)
    for name, summary in summaries.items():
        (evaluation_dir / f"{name}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    for name in (
        "calibration.json",
        "comparison.json",
        "benchmark-local.json",
        "benchmark-jev.json",
        "export-verification.json",
        "http-sdk.json",
    ):
        shutil.copy2(final / name, evaluation_dir / name)
    shutil.copy2("results/improvement/selection.json", evaluation_dir / "selection.json")
    shutil.copy2("data/improvement/manifest.json", evaluation_dir / "data-manifest.json")
    probes = {row["id"]: row for row in read_examples(Path("data/improvement/probes-sealed.jsonl"))}
    failures = [
        {**row, "request": probes[row["id"]]["request"]}
        for row in read_examples(final / "probes/judgments.jsonl")
        if not row["correct"]
    ]
    (evaluation_dir / "probe-failures.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures), encoding="utf-8"
    )
    for name in ("test", "jev-test", "probes", "jev-probes"):
        shutil.copy2(final / name / "judgments.jsonl", evaluation_dir / f"{name}-judgments.jsonl")
    for name, run, evaluation in trials:
        destination = evaluation_dir / name
        destination.mkdir(exist_ok=True)
        shutil.copy2(run / "run_config.json", destination / "run_config.json")
        history = read(run / "training_summary.json")
        history.pop("adapter", None)
        (destination / "training_summary.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        shutil.copy2(evaluation / "summary.json", destination / "dev.json")
    for kind in ("adapter", "merged"):
        card = f"""---
language:
- en
- zh
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
base_model_relation: {"adapter" if kind == "adapter" else "finetune"}
library_name: {"peft" if kind == "adapter" else "transformers"}
tags:
- lora
- classification
- candidate-scoring
datasets:
- facebook/xnli
- mteb/amazon_massive_intent
---

# {model_id}

A bilingual decision model for candidate selection, yes/no judgments, and ordered scores. The included runtime reads answer-label probabilities and returns structured API responses.

This is the {kind} distribution. {"It loads alongside the pinned Qwen base model." if kind == "adapter" else "It includes the complete model with LoRA merged into the weights."} Training and evaluation used text inputs.

## Run

Open the included `runtime` directory and run:

```powershell
uv sync --extra {"training" if kind == "adapter" else "inference"}
$env:NECRO_MODEL='{"Qwen/Qwen3.5-0.8B" if kind == "adapter" else ".."}'
$env:NECRO_ADAPTER='{".." if kind == "adapter" else ""}'
$env:NECRO_TEMPERATURE='1.0'
$env:NECRO_CHOICE_TEMPERATURE='{calibration["temperature"]}'
uv run --extra {"training" if kind == "adapter" else "inference"} necro serve
```

The endpoint is `http://127.0.0.1:8000/v1/systemone`. The example configuration uses bearer key `necro-local`. See [runtime configuration](runtime/docs/running.md) and [API documentation](runtime/docs/api.md) for loading options and response semantics.

## Evaluation

The evaluation report covers XNLI and MASSIVE classification, authored boundary questions, probability calibration, and serial HTTP latency. Results, measurement conditions, and observed errors are in the [evaluation report](runtime/docs/reports/improvement-2026-09-20.md).

## Training

Training used Qwen3.5-0.8B with answer-token cross entropy and LoRA on the language backbone. The [training records](runtime/docs/process/improvement-2026-09-20.md) link the data preparation, configurations, and model selection. Machine-readable records are included in `evaluation/`.

## License

Model weights use [Apache-2.0](LICENSE). Training code uses [MIT](LICENSE-MIT). File scope and upstream attribution are recorded in [licensing and third-party notices](THIRD_PARTY_NOTICES.md).
"""
        directory = package / kind
        (directory / "README.md").write_text(card, encoding="utf-8")
        copy_model_licenses(directory)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("selected", type=Path)
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    release_report(args.selected, args.package)
