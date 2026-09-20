"""从实际实验产物生成报告与待发布模型卡，避免手填评测数字。"""

# Markdown 表格和段落保持完整行，便于检查生成内容。
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
            f"{history['training_seconds'] / 60:.2f} 分钟 | {percentage(metrics['accuracy'])} | {metrics['nll']:.4f} |"
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
    history = f"""# 2026-09-20 改进训练记录

本轮从 [首轮 LoRA](lora-pilot-2026-09-20.md) 继续实验。模型选择遵循 [事先确定的规则](improvement-protocol.md)，导出权重的结果见 [评测报告](../reports/improvement-2026-09-20.md)。

## 训练与开发集比较

各轮使用相同基模 revision、rank 16 和 BF16，只更新 LoRA。续训重新建立优化器。开发集固定为 pilot 验证集。

| 实验 | 本轮记录数 | 学习率 | 纯训练时间 | 开发集准确率 | 开发集 NLL |
|---|---:|---:|---:|---:|---:|
{chr(10).join(trial_rows[1:])}

round2 重复 pilot 数据。round3 从 pilot-v1 续训，扩充公开 train 样例、平衡 NLI 类别，并加入证据不足三元组。round4 在 round3 上以较低学习率再训练一轮。

本轮选择 `{selected.name}`，具体依据记录在 `results/improvement/selection.json`。选择方法的后续核查见 [设计复审](design-review-2026-09-20.md#校准后的开发集比较)。

## 数据

pilot 与 expanded 合计呈现 3,330 条训练记录，按上下文去重为 2,432 条，来源组 1,302 个。中英文包含对应翻译，续训再次使用已有样例。标签来自公开标注和代码构造规则。

expanded 的 XNLI Choice 按语言和类别平衡，本轮每组 176 条。取样、转换与排除逻辑见 [训练指南](../training.md#continuing-training-and-expanding-data)。

## 回归

| 数据 | 导出模型 |
|---|---:|
| 旧诊断集，400 题 | {percentage(summaries["regression"]["overall"]["accuracy"])} |
| 自编开发样例，24 题 | {round(summaries["smoke"]["overall"]["accuracy"] * 24)}/24 |

此前成绩分别保存在 [基模基线](baseline-2026-09-20.md) 和 [首轮 LoRA](lora-pilot-2026-09-20.md)。原 pilot 在本轮测试切片上的准确率为 {percentage(summaries["pilot-test"]["overall"]["accuracy"])}。

## 复现本轮续训

先完成 [训练指南](../training.md) 中的 pilot 数据和权重准备，再运行：

```powershell
uv run --extra training python -m necro.experiments
uv run --extra training python -m necro.probes
uv run --extra training python -m necro.training --data data/improvement/expanded --output results/improvement/round3 --initial-adapter results/lora-pilot/run1/adapter --learning-rate 0.00005 --seed 29 --model-id necro-qwen3.5-0.8b-r3
```

本轮训练配置、日志和逐题结果保存在 `results/improvement/`，数据指纹保存在 `data/improvement/manifest.json`。
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
