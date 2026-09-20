"""从冻结的真实结果生成最终报告、模型卡和可核验发布清单；不上传。"""

# Markdown tables and paragraphs intentionally occupy complete lines.
# ruff: noqa: E501

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from necro.experiment_guard import digest, verify
from necro.export import sha256


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def percent(value):
    return f"{100 * value:.2f}%"


def generate(data, selected, final, package):
    selection = read(selected / "selection.json")
    test = verify(data, "test", Path(selection["adapter"]))
    comparison = read(final / "comparison.json")
    predecessor = read(final / "predecessor-comparison.json")
    checks = read(final / "package-verification.json")
    exported = read(package / "export.json")
    registry = read(data / "experiment.json")
    regression = read(final / "regression-comparison.json")
    robustness = read(final / "robustness-summary.json")
    if comparison["dataset_sha256"] != digest(test) or predecessor["dataset_sha256"] != digest(
        test
    ):
        raise ValueError("最终比较不是登记的独立测试。")
    if not checks.get("passed") or not checks["reload"]["full_dataset"]:
        raise ValueError("真实发布验证尚未完成。")
    for artifact in (checks, exported):
        if artifact["adapter_weights_sha256"] != selection["weights_sha256"]:
            raise ValueError("发布验证、导出包和冻结选择的权重不一致。")
    if checks["dataset_sha256"] != digest(test):
        raise ValueError("重载没有验证完整最终题集。")
    if not checks.get("latency_jev") or not checks.get("official_sdk", {}).get("real_model_http"):
        raise ValueError("缺少远程同题延迟或真实 SDK 验证。")
    if (package / "README.md").exists():
        raise ValueError("最终模型卡已存在，不能覆盖。")
    local = read(final / "local-test/summary.json")
    if local["metadata"].get("adapter_weights_sha256") != selection["weights_sha256"]:
        raise ValueError("独立测试成绩不是冻结权重的成绩。")
    status = (
        "The selected model met all predefined point-estimate performance margins on this independent suite."
        if comparison["performance_pass"]
        else "The selected model did not meet the predefined Jev performance margins."
    )
    task_rows = "\n".join(
        f"| {family} | {values['local']['count']} | {percent(predecessor['families'][family]['local']['accuracy'])} | {percent(values['local']['accuracy'])} | {percent(values['reference']['accuracy'])} |"
        for family, values in sorted(comparison["families"].items())
    )
    language_rows = "\n".join(
        f"| {language} | {values['task_count']} | {values['local']['count']} | {percent(values['macro_accuracy']['local'])} | {percent(values['macro_accuracy']['reference'])} |"
        for language, values in comparison["languages"].items()
    )
    robustness_rows = "\n".join(
        f"| {key} | {values['count']} | {percent(values['accuracy'])} | {percent(robustness['jev'][key]['accuracy'])} | {percent(values['consistency'])} | {percent(robustness['jev'][key]['consistency'])} |"
        for key, values in sorted(robustness["local"].items())
    )
    speed_rows = "\n".join(
        f"| {n['questions_per_request']} | {n['p50_ms']:.2f} / {n['p95_ms']:.2f} | {j['p50_ms']:.2f} / {j['p95_ms']:.2f} |"
        for n, j in zip(checks["latency_local"], checks["latency_jev"], strict=True)
    )
    gates = "\n".join(
        f"| {name} | {'pass' if passed else 'fail'} |"
        for name, passed in comparison["performance_gates"].items()
    )
    ci = comparison["paired_uncertainty"]["macro_accuracy_delta_95ci"]
    missing = registry["protocol"]["intent_coverage"]["test"]["en"]["missing"]
    audit = read(selected / "lineage-audit.json")["selected"]
    temperatures = selection["temperatures"]
    model_id = exported["model_id"]
    dynamic_correct = sum(row.get("exact_match", False) for row in checks["dynamic_choice"])
    report_date = datetime.now().date().isoformat()
    report = f"""# {model_id}: independent evaluation, {report_date}

{status}

Task-macro accuracy was **{percent(comparison["macro_accuracy"]["local"])}**, versus **{percent(comparison["macro_accuracy"]["reference"])}** for fixed Jev 1.13.0, on the same {registry["partitions"]["test"]["rows"]} records and {registry["partitions"]["test"]["source_groups"]} source components. The paired source-group bootstrap 95% interval for Necro minus Jev was [{100 * ci[0]:.2f}, {100 * ci[1]:.2f}] percentage points.

## Independent results

The predecessor is the frozen Phase 2 operator-contrast model. Both local models and Jev answered identical requests. Each local model's calibration was frozen before this test.

| Task | Questions | Predecessor | Selected Necro | Jev 1.13.0 |
|---|---:|---:|---:|---:|
{task_rows}

Task-macro Brier was {comparison["macro_brier"]["local"]:.6f} versus {comparison["macro_brier"]["reference"]:.6f}. The normalized Score MAE difference, Necro minus Jev, was {comparison["normalized_score_mae_delta"]:.6f}. Full NLL, ECE, clipping sensitivity, label provenance and per-language results are in [the comparison record]({{EVIDENCE}}comparison.json). Jev probabilities include zeros, so clipped NLL depends on output precision.

| Predefined performance check | Result |
|---|---|
{gates}

Margins remain: task-macro accuracy within 3 percentage points, every core task within 5 points, normalized Score MAE no more than Jev +0.05, and task-macro Brier no more than Jev +0.03.

## Language and task coverage

| Language | Tasks | Questions | Necro task macro | Jev task macro |
|---|---:|---:|---:|---:|
{language_rows}

English includes BoolQ and SciFact. Local and Jev results use matching tasks within each language. Intent training, development and calibration cover all 60 classes in both languages. This independent intent test covers 58 classes. Missing classes are `{", ".join(missing)}`, and rare-class counts are small.

Candidate extraction chooses among supplied answers, including a no-match option. SciFact selects among sampled candidate documents with unjudged negatives. Human, derived-human, exact-oracle and qrels-with-unjudged-negatives records are reported separately. Synthetic rule templates are shared across partitions. Public-benchmark overlap in the base model's pretraining remains unknown.

## Paired robustness

These variants are paired with their original final-test examples. Consistency measures the expected answer relationship after a perturbation, including unchanged answers for irrelevant context. A consistently wrong answer still counts as consistent. Numeric parents also include missing-field and native-negation cases.

| Perturbation / task / language | Pairs | Necro accuracy | Jev accuracy | Necro consistency | Jev consistency |
|---|---:|---:|---:|---:|---:|
{robustness_rows}

## Historical regression

On the exposed Phase 2 regression cohort, the selected model scored {percent(regression["macro_accuracy"]["local"])} versus {percent(regression["macro_accuracy"]["reference"])} for Jev. This regression was evaluated after Phase 3 selection. [The regression record]({{EVIDENCE}}regression-comparison.json) contains the full results. The earlier independent failure and sampling audit are retained in the [Phase 2 process record]({{DOCS}}process/phase2-development.md#results-at-the-end-of-phase-2) and [coverage audit]({{DOCS}}process/phase3-coverage-audit.md).

## Training and calibration

The base is Qwen3.5-0.8B, revision `{exported["base_revision"]}`. Training updated rank-16 LoRA parameters with answer-token cross-entropy and masked prompt tokens. Labels came from public annotations and constructed rules. The selected ancestry contains {audit["training_presentations"]} example presentations and {audit["unique_training_requests"]} unique normalized requests. Translations and counterfactuals retain their source-group relationships.

The model was selected using development results and predefined development perturbations. Calibration minimized Brier on a separate registered split, per primitive, with source-group and task weighting. Temperatures are Choice {temperatures["choice"]:.9f}, Noul {temperatures["noul"]:.9f}, Score {temperatures["score"]:.9f}. Exact values are in [the frozen calibration]({{EVIDENCE}}selection/calibration.json).

The [development record]({{DOCS}}process/phase3-development.md) contains training times, candidate comparisons, seed replication and the calibration-objective experiment. [The frozen decision]({{EVIDENCE}}selection/decision.json) records the selection evidence.

## Runtime validation and resources

The package contains {exported["total_parameters"]:,} merged parameters. Merged safetensors occupy {sum(v["bytes"] for v in exported["merged_weights"].values()):,} bytes. The full independent suite was rerun from the merged export: {checks["reload"]["changed_argmax"]} changed argmax decisions and maximum probability difference {checks["reload"]["max_probability_difference"]:.9g}, with a tolerance of {checks["reload"]["tolerance"]:g}.

Official Python SDK {checks["official_sdk"]["version"]} calls through a real loopback HTTP server verified all three primitives, model identity and structured Score legends. Real inference checked 1, 26, 27, 60, 90, 91 and 255 choices with first, middle and last target positions. The exact-match diagnostic was correct on {dynamic_correct}/{len(checks["dynamic_choice"])} requests. The [API guide]({{DOCS}}api.md#probabilities-and-scores) defines the local `confidence` formula.

Device: {checks["gpu"]}. Loaded model allocation was {checks["loaded_allocated_gib"]:.3f} GiB. Peak PyTorch allocation during full-test inference was {checks["peak_inference_allocated_gib"]:.3f} GiB. These measurements cover PyTorch process allocations.

| Questions per request | Local p50 / p95 (ms) | Jev p50 / p95 (ms) |
|---|---:|---:|
{speed_rows}

Each timing uses 20 serial repetitions after warmup, persistent connections, and the same short Choice payload. The eight-question case repeats that question. Client caching is disabled, and provider caching is unknown. Local timings use loopback HTTP. Jev timings include internet HTTPS and service overhead. Raw samples and checks are in [the package verification record]({{EVIDENCE}}package-verification.json).

## Reproduction and licenses

Frozen adapter SHA-256: `{selection["weights_sha256"]}`. Independent data SHA-256: `{digest(test)}`. The package includes [selection evidence]({{EVIDENCE}}selection/selection.json), [the data manifest]({{EVIDENCE}}reproduction/independent-data-manifest.json), predictions and checksums. Reproduction files retain source revisions, hashes and the original data-builder snapshot. Retrieve source datasets at those recorded revisions.

Project and fine-tuning contributions use Apache-2.0. The declared training files use MIT. Third-party terms and attribution are listed in [licensing and third-party notices]({{LICENSE}}THIRD_PARTY_NOTICES.md).
"""
    evaluation = package / "evaluation"
    evaluation.mkdir(exist_ok=False)

    def render(evidence, docs, licenses):
        return (
            report.replace("{EVIDENCE}", evidence)
            .replace("{DOCS}", docs)
            .replace("{LICENSE}", licenses)
        )

    (evaluation / "report.md").write_text(render("", "../runtime/docs/", "../"), encoding="utf-8")
    report_path = Path(f"docs/reports/phase3-{report_date}.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path = "../../" + package.as_posix() + "/evaluation/"
    report_path.write_text(render(evidence_path, "../", "../../"), encoding="utf-8")
    target = package / "runtime/docs/reports" / report_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render("../../../evaluation/", "../", "../../"), encoding="utf-8")
    for filename in (
        "comparison.json",
        "predecessor-comparison.json",
        "regression-comparison.json",
        "robustness-summary.json",
        "package-verification.json",
    ):
        shutil.copy2(final / filename, evaluation / filename)
    for run in (
        "local-test",
        "baseline-test",
        "jev-test",
        "local-robustness",
        "jev-robustness",
        "local-regression",
    ):
        shutil.copytree(
            final / run, evaluation / run, ignore=shutil.ignore_patterns("remote-cache")
        )
    shutil.copytree(selected, evaluation / "selection", ignore=shutil.ignore_patterns("adapter"))
    shutil.copy2(
        Path("results/phase2/v3/final/comparison.json"),
        evaluation / "phase2-original-comparison.json",
    )
    recipes = evaluation / "reproduction"
    recipes.mkdir()
    completed_runs = {}
    for summary_path in Path("results").rglob("training_summary.json"):
        weight_file = summary_path.parent / "adapter/adapter_model.safetensors"
        if weight_file.exists():
            completed_runs[digest(weight_file)] = summary_path.parent
    for record in audit["lineage"]:
        folder = recipes / record["data_sha256"][:12]
        folder.mkdir(exist_ok=True)
        source = Path(record["data"])
        for filename in ("experiment.json", "builder.py", "build-provenance.json"):
            if (source.parent / filename).exists():
                shutil.copy2(source.parent / filename, folder / filename)
        run_folder = folder / "runs" / record["weights_sha256"][:12]
        run_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            Path(record["adapter"]) / "necro_adapter.json", run_folder / "necro_adapter.json"
        )
        run = completed_runs.get(record["weights_sha256"], Path(record["adapter"]).parent)
        for filename in ("run_config.json", "training_summary.json", "timing-audit.json"):
            if (run / filename).exists():
                record_data = read(run / filename)
                record_data.pop("adapter", None)
                (run_folder / filename).write_text(
                    json.dumps(record_data, indent=2), encoding="utf-8"
                )
    shutil.copy2(data / "experiment.json", recipes / "independent-data-manifest.json")
    for source in Path("data/phase3/snapshots").glob("*"):
        folder = recipes / "snapshots" / source.name
        folder.mkdir(parents=True)
        for filename in ("manifest.json", "revision.json"):
            shutil.copy2(source / filename, folder / filename)
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

# {model_id}

A small bilingual judgment model based on Qwen3.5-0.8B. The included runtime supports TypeSafe v1 Noul, Choice and Score request/response formats by scoring candidate labels. It uses text inputs.

On the frozen eight-task independent suite, task-macro accuracy is {percent(comparison["macro_accuracy"]["local"])}, versus {percent(comparison["macro_accuracy"]["reference"])} for Jev 1.13.0 on identical requests. The [evaluation report](evaluation/report.md) contains per-task results, source-group uncertainty, observed failures and measurement conditions.

## Run the local API

From the downloaded repository root, use the included runtime. Its `.env.example` points to these merged weights and includes the frozen calibration:

```powershell
cd runtime
uv sync --extra inference
Copy-Item .env.example .env
uv run --extra inference necro serve
```

Skip the copy step if `.env` already exists. The example address is `http://127.0.0.1:8000`. Send `POST /v1/systemone` with `Authorization: Bearer necro-local`, matching the key in your configuration. The returned model identity is `{model_id}`. Follow the [API guide](runtime/docs/api.md#sending-a-request) to send a request, or the [setup guide](runtime/docs/running.md) to configure model loading.

This repository contains merged safetensors at its root, the LoRA in `adapter/`, source/runtime code, provenance, results and checksums. Use the supplied runtime and calibration for the reported behavior.

## Licensing

Project and fine-tuning contributions use [Apache-2.0](LICENSE). The declared training files use [MIT](LICENSE-MIT). See [licensing and third-party notices](THIRD_PARTY_NOTICES.md) for file scope, training sources and upstream terms.
"""
    (package / "README.md").write_text(card, encoding="utf-8")
    adapter_metadata = (
        card.split("\n# ", 1)[0]
        .replace("base_model_relation: finetune", "base_model_relation: adapter")
        .replace("library_name: transformers", "library_name: peft")
    )
    adapter_card = (
        adapter_metadata
        + f"""
# {model_id}: LoRA adapter

This adapter loads with the Qwen3.5-0.8B revision recorded in `necro_adapter.json`. The [model overview](../README.md) describes the judgment model, and the [evaluation report](../evaluation/report.md) records its measurements.

From the downloaded repository root:

```powershell
cd runtime
uv sync --extra training
Copy-Item .env.example .env
$env:NECRO_MODEL = 'Qwen/Qwen3.5-0.8B'
$env:NECRO_ADAPTER = '../adapter'
uv run --extra training necro serve
```

Skip the copy step if `.env` exists. Retain the frozen calibration from the included example, and clear any conflicting temperature overrides in the terminal. See [configuration sources](../runtime/docs/running.md#configuration-sources) for precedence.

Fine-tuning contributions use [Apache-2.0](LICENSE). Training code scope and third-party terms are recorded in [licensing and third-party notices](THIRD_PARTY_NOTICES.md).
"""
    )
    (package / "adapter/README.md").write_text(adapter_card, encoding="utf-8")
    manifest = {
        str(path.relative_to(package)).replace("\\", "/"): sha256(path)
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    }
    (package / "SHA256SUMS").write_text(
        "".join(f"{value}  {path}\n" for path, value in manifest.items()), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "package": str(package),
                "files_hashed": len(manifest),
                "performance_pass": comparison["performance_pass"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("data", "selected", "final", "package"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    generate(args.data, args.selected, args.final, args.package)
