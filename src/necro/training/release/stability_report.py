"""Package measured stability results, model cards and reproducibility records."""

# Markdown paragraphs and tables occupy complete lines.
# ruff: noqa: E501

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

from necro.evaluation import read_examples
from necro.experiment_guard import digest, verify
from necro.training.release.stability_assessment import read, write


def percent(value):
    return f"{100 * value:.2f}%"


def interval(values):
    return f"[{100 * values[0]:.2f}, {100 * values[1]:.2f}]"


def checksums(package):
    files = {
        p.relative_to(package).as_posix(): digest(p)
        for p in sorted(package.rglob("*"))
        if p.is_file() and p.name != "SHA256SUMS"
    }
    (package / "SHA256SUMS").write_text(
        "".join(f"{value}  {name}\n" for name, value in files.items()), encoding="utf-8"
    )
    return len(files)


def copy_reproduction(selected, data, target):
    audit = read(selected / "lineage-audit.json")
    completed = {}
    for summary in Path("results").rglob("training_summary.json"):
        weights = summary.parent / "adapter/adapter_model.safetensors"
        if weights.exists():
            completed[digest(weights)] = summary.parent
    for row in audit["selected"]["lineage"]:
        source = Path(row["data"]).parent
        folder = target / row["data_sha256"][:12]
        folder.mkdir(parents=True, exist_ok=True)
        for filename in ("experiment.json", "builder.py", "build-provenance.json"):
            if (source / filename).exists():
                shutil.copy2(source / filename, folder / filename)
        run = completed.get(row["weights_sha256"], Path(row["adapter"]).parent)
        run_folder = folder / "runs" / row["weights_sha256"][:12]
        run_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(row["adapter"]) / "necro_adapter.json", run_folder / "necro_adapter.json")
        for filename in ("run_config.json", "training_summary.json", "timing-audit.json"):
            if (run / filename).exists():
                shutil.copy2(run / filename, run_folder / filename)
    shutil.copy2(data / "experiment.json", target / "condition-data-manifest.json")
    shutil.copy2(
        "data/phase3/coverage-v1/experiment.json", target / "regression-data-manifest.json"
    )
    for source in Path("data/phase3/snapshots").iterdir():
        if source.is_dir():
            folder = target / "snapshots" / source.name
            folder.mkdir(parents=True)
            for filename in ("manifest.json", "revision.json"):
                shutil.copy2(source / filename, folder / filename)


def generate(data, selected, final, package):
    selection = read(selected / "selection.json")
    test = verify(data, "test", Path(selection["adapter"]))
    capability = read(final / "capability-review.json")
    if not capability["passed"] or not all(capability["checks"].values()):
        raise ValueError("Condition repair and task retention have not passed")
    runtime = read(final / "package-verification.json")
    exported = read(package / "export.json")
    if not runtime["passed"] or not runtime["official_sdk"]["real_model_http"]:
        raise ValueError("Runtime validation has not passed")
    for value in (capability, runtime, exported):
        if value["adapter_weights_sha256"] != selection["weights_sha256"]:
            raise ValueError("Release evidence refers to different weights")
    for key, expected in (
        ("reload", digest(test)),
        ("regression_reload", read(selected / "validation-plan.json")["task_regression"]["sha256"]),
    ):
        evidence = runtime.get(key)
        if (
            not evidence
            or evidence["dataset_sha256"] != expected
            or not evidence["full_dataset"]
            or evidence["changed_argmax"]
            or evidence["max_probability_difference"] > 1e-5
        ):
            raise ValueError("Both complete reload cohorts must pass")
    for name, expected in capability["evidence_sha256"].items():
        if digest(final / name) != expected:
            raise ValueError("Capability evidence changed after review")
    if (
        digest(package / "adapter/adapter_model.safetensors") != selection["weights_sha256"]
        or exported["temperatures"] != selection["temperatures"]
    ):
        raise ValueError("Packaged adapter or calibration differs from selection")
    for name, evidence in exported["merged_weights"].items():
        if digest(package / exported["merged_path"] / name) != evidence["sha256"]:
            raise ValueError("Merged weights changed after validation")
    if (package / "README.md").exists() or (package / "evaluation").exists():
        raise ValueError("Release report or model card already exists")
    comparison = read(final / "regression-jev-comparison.json")
    predecessor = read(final / "regression-predecessor-comparison.json")
    parent = read(final / "regression-instruction-comparison.json")
    independent = read(final / "conditions-jev-comparison.json")
    conditions = capability["condition_slices"]
    legacy = capability["original_condition_slices"]
    date = datetime.now().date().isoformat()
    model = selection["model_id"]
    audit = read(selected / "lineage-audit.json")["selected"]
    registry = read(data / "experiment.json")
    failures = read_examples(final / "failures.jsonl")
    failure_counts = Counter((r["cohort"], r["family"]) for r in failures)
    tasks = "\n".join(
        f"| {task} | {v['local']['count']} | {percent(predecessor['families'][task]['reference']['accuracy'])} | {percent(parent['families'][task]['reference']['accuracy'])} | {percent(v['local']['accuracy'])} | {percent(v['reference']['accuracy'])} |"
        for task, v in sorted(comparison["families"].items())
    )
    condition_table = "\n".join(
        f"| {key} | {conditions['local'][key]['count']} | {percent(conditions['predecessor'][key]['accuracy'])} | {percent(conditions['local'][key]['accuracy'])} | {percent(conditions['jev'][key]['accuracy'])} |"
        for key in ("all", "complete", "gate_false", "missing", "en", "zh")
    )
    legacy_table = "\n".join(
        f"| {key} | {legacy['local'][key]['count']} | {legacy['predecessor'][key]['correct']} | {legacy['instruction'][key]['correct']} | {legacy['local'][key]['correct']} | {legacy['jev'][key]['correct']} |"
        for key in ("all", "complete", "gate_false", "missing")
    )
    errors = "\n".join(
        f"| {cohort} | {family} | {count} |"
        for (cohort, family), count in sorted(failure_counts.items())
    )
    languages = "\n".join(
        f"| {language} | {v['task_count']} | {v['local']['count']} | {percent(v['macro_accuracy']['local'])} | {percent(v['macro_accuracy']['reference'])} |"
        for language, v in comparison["languages"].items()
    )
    latency = "\n".join(
        f"| {a['questions_per_request']} | {a['p50_ms']:.2f} / {a['p95_ms']:.2f} | {b['p50_ms']:.2f} / {b['p95_ms']:.2f} |"
        for a, b in zip(runtime["latency_local"], runtime["latency_jev"], strict=True)
    )
    reloads = "\n".join(
        f"| {name} | {runtime[key]['examples']} | {runtime[key]['changed_argmax']} | {runtime[key]['max_probability_difference']:.9g} |"
        for name, key in (
            ("Independent conditions", "reload"),
            ("Multi-task regression", "regression_reload"),
        )
    )
    dynamic = runtime["dynamic_choice"]
    dynamic_errors = [r for r in dynamic if not r["exact_match"]]
    diagnostic = (
        "; ".join(
            f"{r['candidates']} candidates, target `{r['target']}`, selected `{r['choice']}`"
            for r in dynamic_errors
        )
        or "All diagnostic targets were selected correctly."
    )
    numeric_errors = [r for r in failures if r["family"] == "numeric_rule"]
    numeric_types = Counter(
        (r["cohort"], r["condition"]["case"], r["language"], r["condition"]["operator"])
        for r in numeric_errors
    )
    numeric_breakdown = (
        "; ".join(
            f"{cohort}/{case}/{lang}/{operator}: {count}"
            for (cohort, case, lang, operator), count in sorted(numeric_types.items())
        )
        or "No numeric errors were observed."
    )
    temperatures = selection["temperatures"]
    report = f"""# {model}: stability evaluation, {date}

The selected release passed the registered condition-repair, task-retention, export, reload and API checks. Independent condition accuracy was **{percent(conditions["local"]["all"]["accuracy"])}**, versus **{percent(conditions["jev"]["all"]["accuracy"])}** for Jev 1.13.0. On the exposed eight-task regression cohort, task-macro accuracy was **{percent(comparison["macro_accuracy"]["local"])}**, versus **{percent(comparison["macro_accuracy"]["reference"])}** for Jev.

## Independent condition transfer

This cohort contains {registry["partitions"]["test"]["rows"]} questions from {registry["partitions"]["test"]["source_groups"]} source groups. Numerical states, field names and two wording styles are held out from the repair training. English and Chinese translations and paired conditions remain grouped. The candidate, calibration and [validation plan]({{EVIDENCE}}selection/validation-plan.json) were frozen before inference on this cohort.

| Slice | Questions | Predecessor | Release | Jev 1.13.0 |
|---|---:|---:|---:|---:|
{condition_table}

The paired source-group bootstrap 95% interval for release minus Jev accuracy is {interval(independent["paired_uncertainty"]["macro_accuracy_delta_95ci"])} percentage points. Brier scores are {independent["macro_brier"]["local"]:.6f} and {independent["macro_brier"]["reference"]:.6f}, respectively. Twelve source groups provide the independent sampling units. See [the paired results]({{EVIDENCE}}conditions-jev-comparison.json) for uncertainty and probability metrics, and [condition slices]({{EVIDENCE}}capability-review.json) for gate, missing-field, polarity, operator and language results.

## Multi-task regression

These 1,790 questions were exposed during Phase 3 and subsequent numeric diagnosis. They measure retention after the repair. The independent results above measure transfer on new conditions. The predecessor is the retained Phase 2 adapter; the instruction parent is the Phase 3 model from which repair training started. All models answered identical requests with their recorded calibration.

| Task | Questions | Predecessor | Instruction parent | Release | Jev 1.13.0 |
|---|---:|---:|---:|---:|---:|
{tasks}

Every other task remained within the registered two-percentage-point margin of both local baselines. Task-language slices were also reviewed. Source-group 95% intervals for release minus predecessor and release minus instruction-parent macro accuracy are {interval(predecessor["paired_uncertainty"]["macro_accuracy_delta_95ci"])} and {interval(parent["paired_uncertainty"]["macro_accuracy_delta_95ci"])} points. Full retention results are in the [predecessor comparison]({{EVIDENCE}}regression-predecessor-comparison.json) and [instruction-parent comparison]({{EVIDENCE}}regression-instruction-comparison.json).

Against Jev, the macro accuracy interval is {interval(comparison["paired_uncertainty"]["macro_accuracy_delta_95ci"])} points. Macro Brier is {comparison["macro_brier"]["local"]:.6f}, versus {comparison["macro_brier"]["reference"]:.6f}. Normalized Score MAE differs by {comparison["normalized_score_mae_delta"]:.6f}. [The Jev comparison]({{EVIDENCE}}regression-jev-comparison.json) includes NLL, ECE, clipping sensitivity and label provenance.

| Language | Tasks | Questions | Release macro | Jev macro |
|---|---:|---:|---:|---:|
{languages}

English includes BoolQ and SciFact; Chinese covers the other six tasks. Intent evaluation covers 58 classes; `cooking_query` and `general_greet` are absent from this cohort. Candidate extraction chooses among supplied answers, including a no-match option. Retrieval uses sampled candidate documents and unjudged negatives. Public-benchmark overlap in base-model pretraining is unknown.

## Confirmed defect and remaining errors

The original condition defect involved ignoring a disabled or missing gate under a guard-first rule. Correct counts on the exposed original-wording cases are:

| Condition | Questions | Predecessor | Instruction parent | Release | Jev |
|---|---:|---:|---:|---:|---:|
{legacy_table}

Remaining numeric errors by cohort, condition, language and operator: {numeric_breakdown}. Their generated requests, expected answers and predictions are included in [the failure record]({{EVIDENCE}}failures.jsonl). Other failures are recorded by source ID and judgment so they can be joined to the licensed source datasets.

| Cohort | Task | Wrong answers |
|---|---|---:|
{errors}

The large-choice exact-match diagnostic answered {sum(r["exact_match"] for r in dynamic)}/{len(dynamic)} requests correctly. Remaining cases: {diagnostic}. All requests passed response-schema, candidate-cardinality and probability-normalization checks. Candidate selection quality remains dependent on the supplied labels, descriptions and number of options.

The runtime accepts text or JSON context and returns Noul probabilities, candidate choices and ordered scores. It scores candidate labels without generating explanations. Input limits are defined by `Settings` in [config.py]({{CODE}}config.py) and `Choice` and `Score` in [schema.py]({{CODE}}schema.py). The [API guide]({{DOCS}}api.md) defines supported fields and the local confidence calculation.

## Training and calibration

The base is Qwen3.5-0.8B at revision `{exported["base_revision"]}`. Rank-16 LoRA updates use answer-token cross entropy with prompts masked. Labels come from public annotations and generated rules. Jev supplies evaluation responses. The selected ancestry contains {audit["training_presentations"]:,} presentations and {audit["unique_training_requests"]:,} unique normalized requests.

The final repair uses the complete condition matrix and retained task replay. The initial numeric-development retention check failed: 157/176 correct, compared with 161/176 for the instruction parent. Selection accepted this four-question tradeoff before opening the independent holdout, while retaining the checks for the other seven tasks. The [frozen decision]({{EVIDENCE}}selection/decision.json) preserves the amendment and links the original [preliminary review]({{EVIDENCE}}selection/preliminary-review.json). Training history, failed candidates and recipe amendments are in the [repair process record]({{DOCS}}process/condition-repair-2026-09-21.md).

Per-primitive Brier calibration uses a separate registered cohort: Choice {temperatures["choice"]:.9f}, Noul {temperatures["noul"]:.9f}, Score {temperatures["score"]:.9f}. Exact values are in [calibration.json]({{EVIDENCE}}selection/calibration.json).

## Runtime validation

The exported model contains {exported["total_parameters"]:,} parameters. Merged safetensors occupy {sum(v["bytes"] for v in exported["merged_weights"].values()):,} bytes. Both complete cohorts were rerun from merged weights, using probability tolerance 1e-5:

| Cohort | Questions | Changed decisions | Maximum probability difference |
|---|---:|---:|---:|
{reloads}

Official Python SDK {runtime["official_sdk"]["version"]} calls through a real loopback HTTP server verified the model identity, mixed primitives and structured Score legends. Dynamic Choice requests covered 1, 26, 27, 60, 90, 91 and 255 candidates at first, middle and last positions.

Device: {runtime["gpu"]}. Loaded PyTorch allocation was {runtime["loaded_allocated_gib"]:.3f} GiB; peak allocation during inference was {runtime["peak_inference_allocated_gib"]:.3f} GiB.

| Questions per request | Local p50 / p95 (ms) | Jev p50 / p95 (ms) |
|---|---:|---:|
{latency}

Each timing uses 20 serial requests after one warmup, persistent connections and the same short Choice payload. The eight-question case repeats that question. Client caching is disabled and provider caching is unknown. Local timings use loopback; Jev includes internet HTTPS and service overhead. Raw timings and verification results are in [package-verification.json]({{EVIDENCE}}package-verification.json).

## Reproduction and licenses

Adapter SHA-256: `{selection["weights_sha256"]}`. Independent condition data SHA-256: `{digest(test)}`. The package includes [selection and lineage evidence]({{EVIDENCE}}selection/selection.json), dataset manifests, archived builders, source revisions, training configurations and predictions in [reproduction records]({{EVIDENCE}}reproduction/). `SHA256SUMS` covers every distributed file. Use the included runtime and calibrated temperatures to reproduce inference.

Project and fine-tuning contributions use Apache-2.0. The declared training files use MIT. Third-party datasets retain their individual licenses, including XNLI's CC BY-NC 4.0. Full terms and attribution are in [licensing and third-party notices]({{LICENSE}}THIRD_PARTY_NOTICES.md).
"""
    evaluation = package / "evaluation"
    shutil.copytree(final, evaluation, ignore=shutil.ignore_patterns("remote-cache", "*.log"))
    shutil.copytree(selected, evaluation / "selection", ignore=shutil.ignore_patterns("adapter"))
    copy_reproduction(selected, data, evaluation / "reproduction")

    def render(evidence, docs, code, licenses):
        return (
            report.replace("{EVIDENCE}", evidence)
            .replace("{DOCS}", docs)
            .replace("{CODE}", code)
            .replace("{LICENSE}", licenses)
        )

    (evaluation / "report.md").write_text(
        render("", "../runtime/docs/", "../runtime/src/necro/", "../"), encoding="utf-8"
    )
    report_path = Path(f"docs/reports/stability-{date}.md")
    report_path.write_text(
        render("../../" + package.as_posix() + "/evaluation/", "../", "../../src/necro/", "../../"),
        encoding="utf-8",
    )
    bundled_report = package / "runtime" / report_path
    bundled_report.parent.mkdir(parents=True, exist_ok=True)
    bundled_report.write_text(
        render("../../../evaluation/", "../", "../../src/necro/", "../../"), encoding="utf-8"
    )
    card = f"""---
language: [en, zh]
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
base_model_relation: finetune
library_name: transformers
tags: [lora, classification, candidate-scoring]
datasets:
- facebook/xnli
- mteb/amazon_massive_intent
- google-research-datasets/paws-x
- google/boolq
- rajpurkar/squad
- hfl/cmrc2018
- BeIR/scifact
---

# {model}

A bilingual judgment model based on Qwen3.5-0.8B. The included runtime accepts TypeSafe v1 Noul, Choice and Score requests and scores candidate labels to return structured judgments with probabilities.

Independent condition accuracy is {percent(conditions["local"]["all"]["accuracy"])}, versus {percent(conditions["jev"]["all"]["accuracy"])} for Jev 1.13.0. On an exposed eight-task regression cohort, macro accuracy is {percent(comparison["macro_accuracy"]["local"])}, versus {percent(comparison["macro_accuracy"]["reference"])}. The [evaluation report](evaluation/report.md) contains cohort definitions, task results, remaining errors and runtime measurements.

## Run the local API

From the downloaded repository root:

```powershell
cd runtime
uv sync --extra inference
Copy-Item .env.example .env
uv run --extra inference necro serve
```

Skip the copy step if `.env` exists. The included example loads the merged weights in the parent directory and their frozen calibration. Send `POST /v1/systemone` to `http://127.0.0.1:8000`, with the API key configured in `.env`. The response model is `{model}`. See [sending a request](runtime/docs/api.md#sending-a-request) and [setup](runtime/docs/running.md).

Merged weights are at the repository root; `adapter/` contains the LoRA. Runtime code, provenance, predictions and checksums are included. Input constraints and supported fields are in the [API guide](runtime/docs/api.md).

## Licensing

Project and fine-tuning contributions use [Apache-2.0](LICENSE). The declared training files use [MIT](LICENSE-MIT). Training sources retain their own terms, including XNLI's CC BY-NC 4.0. See [licensing and third-party notices](THIRD_PARTY_NOTICES.md).
"""
    (package / "README.md").write_text(card, encoding="utf-8")
    (package / "adapter/README.md").write_text(
        f"""# {model}: LoRA adapter

Load this adapter with the Qwen3.5-0.8B revision in `necro_adapter.json`. The [model overview](../README.md) introduces the runtime, and [the evaluation report](../evaluation/report.md) records its results.

From the repository root:

```powershell
cd runtime
uv sync --extra training
Copy-Item .env.example .env
$env:NECRO_MODEL = 'Qwen/Qwen3.5-0.8B'
$env:NECRO_ADAPTER = '../adapter'
uv run --extra training necro serve
```

Skip the copy step if `.env` exists. Retain the packaged calibration and clear conflicting temperature environment variables. Configuration precedence is defined in [the setup guide](../runtime/docs/running.md#configuration-sources).

Fine-tuning contributions use [Apache-2.0](LICENSE). [Licensing and third-party notices](THIRD_PARTY_NOTICES.md) describe training code and source terms.
""",
        encoding="utf-8",
    )
    write(
        package / "release.json",
        {
            "model_id": model,
            "date": date,
            "adapter_weights_sha256": selection["weights_sha256"],
            "capability_review_sha256": digest(evaluation / "capability-review.json"),
            "runtime_verification_sha256": digest(evaluation / "package-verification.json"),
            "checks_passed": True,
        },
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "package": str(package),
                "files_hashed": checksums(package),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("data", "selected", "final", "package"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    generate(args.data, args.selected, args.final, args.package)
