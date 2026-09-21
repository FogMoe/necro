"""从冻结的真实结果生成最终报告、模型卡和可核验发布清单；不上传。"""

# Markdown tables and paragraphs intentionally occupy complete lines.
# ruff: noqa: E501

import argparse
import json
import shutil
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
    old = read(Path("results/phase2/v3/final/comparison.json"))
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
    trial_rows = []
    for name in ("coverage-seed2026", "instruction-seed2026", "instruction-seed2027"):
        run = Path("results/phase3") / name
        if not (run / "training_summary.json").exists():
            continue
        timing, config = read(run / "training_summary.json"), read(run / "run_config.json")
        audit_path = run / "timing-audit.json"
        timing_note = "optimizer-step wall timing"
        if audit_path.exists():
            timing_note = f"includes long sleep/standby intervals; other step intervals total {read(audit_path)['sum_intervals_under_60_seconds'] / 60:.2f} min"
        trial_rows.append(
            f"| {name} | {config['examples']} | {config['seed']} | {timing['training_seconds'] / 60:.2f} | {timing['peak_allocated_gib']:.3f} | {timing_note} |"
        )
    dynamic_correct = sum(row.get("exact_match", False) for row in checks["dynamic_choice"])
    report = f"""# {model_id}: independent evaluation report

{status}

Task-macro accuracy was **{percent(comparison["macro_accuracy"]["local"])}**, versus **{percent(comparison["macro_accuracy"]["reference"])}** for fixed Jev 1.13.0, on the same {registry["partitions"]["test"]["rows"]} records and {registry["partitions"]["test"]["source_groups"]} source components. The paired source-group bootstrap 95% interval for Necro minus Jev was [{100 * ci[0]:.2f}, {100 * ci[1]:.2f}] percentage points.

## Independent results

The predecessor is the frozen Phase 2 operator-contrast model. Both local models and Jev answered identical requests; each local model's calibration was frozen before this test.

| Task | Questions | Predecessor | Selected Necro | Jev 1.13.0 |
|---|---:|---:|---:|---:|
{task_rows}

Task-macro Brier was {comparison["macro_brier"]["local"]:.6f} versus {comparison["macro_brier"]["reference"]:.6f}. Normalized Score MAE exceeded Jev by {comparison["normalized_score_mae_delta"]:.6f}. Full NLL, ECE, clipping-sensitivity, label-quality and per-language results are in `comparison.json`. Jev probabilities can be quantized and contain zeros; clipped NLL is sensitive to output precision.

| Predefined performance check | Result |
|---|---|
{gates}

Margins remain: task-macro accuracy within 3 percentage points, every core task within 5 points, normalized Score MAE no more than Jev +0.05, and task-macro Brier no more than Jev +0.03.

## Language and task coverage

| Language | Tasks | Questions | Necro task macro | Jev task macro |
|---|---:|---:|---:|---:|
{language_rows}

English includes BoolQ and SciFact. Language comparisons use matching tasks. Intent training/development/calibration cover all 60 classes in both languages. This independent intent test covers 58 classes; missing classes are `{", ".join(missing)}`. Rare-class counts are small.

Candidate extraction chooses among supplied answers, including a no-match option. SciFact selects among sampled candidate documents with unjudged negatives. Human, derived-human, exact-oracle and qrels-with-unjudged-negatives records are reported separately. Synthetic rule templates are shared across partitions. Public-benchmark overlap in the base model's pretraining remains unknown.

## Paired robustness

These variants are paired with their original final-test examples. Consistency means the answer transforms as expected when candidate order or question polarity changes; a consistently wrong answer still counts as consistent. Numeric parents also include missing-field and native-negation cases.

| Perturbation / task / language | Pairs | Necro accuracy | Jev accuracy | Necro consistency | Jev consistency |
|---|---:|---:|---:|---:|---:|
{robustness_rows}

## Historical failure and regression

The Phase 2 independent result was {percent(old["macro_accuracy"]["local"])} versus {percent(old["macro_accuracy"]["reference"])}; its accuracy and Brier gates failed. The subsequent audit found missing intent classes and prefix/block sampling bias. Those old results remain part of the record.

The selected model's result on that **exposed historical regression** cohort is {percent(regression["macro_accuracy"]["local"])} versus {percent(regression["macro_accuracy"]["reference"])}. This regression was evaluated after Phase 3 selection. See `regression-comparison.json`.

## Training and calibration

The base is Qwen3.5-0.8B, revision `{exported["base_revision"]}`. Training updated rank-16 LoRA parameters with answer-token cross-entropy; prompt tokens were masked. No Jev training labels or Jev distillation were used. The selected ancestry contains {audit["training_presentations"]} example presentations and {audit["unique_training_requests"]} unique normalized requests. Translations and counterfactuals retain their source-group relationships.

The model was selected using development results and predefined development perturbations. Calibration used a separate registered split, per primitive, with source-group and family weighting. Temperatures are Choice {temperatures["choice"]:.9f}, Noul {temperatures["noul"]:.9f}, Score {temperatures["score"]:.9f}. A preregistered NLL/Brier fitting comparison found a negligible average Brier difference; the adopted objective is recorded in the selection materials. Temperature does not change argmax accuracy.

Selection rationale is recorded in `selection/decision.json`.

| Phase 3 trial | Records | Seed | Recorded wall minutes | Peak training GiB | Timing scope |
|---|---:|---:|---:|---:|---|
{chr(10).join(trial_rows)}

The two instruction trials use identical data and the same coverage-model parent. Long interrupted intervals are retained in the raw record. Step timings measure wall time rather than GPU kernel time.

## Runtime validation and resources

The package contains {exported["total_parameters"]:,} merged parameters. Merged safetensors occupy {sum(v["bytes"] for v in exported["merged_weights"].values()):,} bytes. The full independent suite was rerun from the merged export: {checks["reload"]["changed_argmax"]} changed argmax decisions and maximum probability difference {checks["reload"]["max_probability_difference"]:.9g}, with a tolerance of {checks["reload"]["tolerance"]:g}.

Official Python SDK {checks["official_sdk"]["version"]} calls through a real loopback HTTP server verified all three primitives, model identity and structured Score legends. Real inference checked 1, 26, 27, 60, 90, 91 and 255 choices with first/middle/last target positions. The exact-match diagnostic was correct on {dynamic_correct}/{len(checks["dynamic_choice"])} requests. The runtime API documentation defines the local `confidence` formula.

Device: {checks["gpu"]}. Loaded model allocation was {checks["loaded_allocated_gib"]:.3f} GiB; peak PyTorch allocation during full-test inference was {checks["peak_inference_allocated_gib"]:.3f} GiB. The memory measurements report PyTorch process allocations.

| Questions per request | Local p50 / p95 (ms) | Jev p50 / p95 (ms) |
|---|---:|---:|
{speed_rows}

Each timing uses 20 serial repetitions after warmup, persistent connections, and the same short Choice payload. The eight-question case repeats that question. Client caching is disabled; provider caching is unknown. Local timings use loopback HTTP; Jev timings include internet HTTPS and service overhead.

## Reproduction and licenses

Frozen adapter SHA-256: `{selection["weights_sha256"]}`. Independent data SHA-256: `{digest(test)}`. See the attached selection, source snapshots, data manifests, training records, predictions and checksums. The original data-builder snapshot is retained separately from later code changes. Source datasets are not bundled as raw data; their revisions and fingerprints are supplied for retrieval and validation.

Code and fine-tuning contributions follow the licenses in the package. The base Qwen weights are Apache-2.0. Dataset terms remain distinct, including XNLI's CC BY-NC 4.0 and share-alike sources. See `THIRD_PARTY_NOTICES.md` and the included license texts.
"""
    evaluation = package / "evaluation"
    evaluation.mkdir(exist_ok=False)
    (evaluation / "report.md").write_text(report, encoding="utf-8")
    report_path = Path("docs/reports/phase3-2026-09-20.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    target = package / "runtime/docs/reports" / report_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(report_path, target)
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

{status}

On the frozen eight-task independent suite, task-macro accuracy is {percent(comparison["macro_accuracy"]["local"])}, versus {percent(comparison["macro_accuracy"]["reference"])} for Jev 1.13.0 on identical requests. See the [full report](evaluation/report.md) for per-task results, source-group uncertainty, historical failures, language coverage and the precise acceptance criteria.

## Run the local API

From the downloaded repository root, use the included runtime. Its `.env.example` points to these merged weights and includes the frozen calibration:

```powershell
cd runtime
uv sync --extra inference
Copy-Item .env.example .env
uv run --extra inference necro serve
```

The default address is `http://127.0.0.1:8000`; send `POST /v1/systemone` with `Authorization: Bearer necro-local` and the documented TypeSafe request body. The returned model identity is `{model_id}`. Setup and supported fields are documented in [runtime/docs/running.md](runtime/docs/running.md). Tested hardware and measured latency are in the report.

This repository contains merged safetensors at its root, the LoRA in `adapter/`, source/runtime code, provenance, results and checksums. Use the supplied runtime and calibration for the reported behavior.

## Coverage and licensing

Candidate extraction and document selection are bounded-choice tasks. BoolQ and SciFact evaluation is English-only; the intent test omits two rare classes. Public-benchmark overlap in the base model's pretraining remains unknown.

Fine-tuning contributions and model metadata use Apache-2.0. Source datasets retain their own terms, including XNLI CC BY-NC 4.0 and share-alike sources. Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the license files. No Jev distillation was used.
"""
    (package / "README.md").write_text(card, encoding="utf-8")
    adapter_card = card.replace(
        "base_model_relation: finetune", "base_model_relation: adapter"
    ).replace("library_name: transformers", "library_name: peft")
    adapter_card += "\nFor LoRA loading, use the pinned Qwen base revision from `necro_adapter.json`. In the sibling runtime, set `NECRO_MODEL=Qwen/Qwen3.5-0.8B`, `NECRO_ADAPTER=../adapter`, retain the calibration values, and install the `training` extra for PEFT.\n"
    adapter_card = adapter_card.replace("](evaluation/", "](../evaluation/").replace(
        "](runtime/", "](../runtime/"
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
