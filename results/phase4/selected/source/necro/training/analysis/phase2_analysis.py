"""八任务的配对比较；按来源组 bootstrap，不把翻译当独立样本。"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from necro.evaluation import metrics, read_examples, summarize
from necro.experiment_guard import canonical_request, digest
from necro.schema import EvaluationResponse
from necro.training.data.training_data import write_jsonl

FAMILIES = frozenset(
    {
        "inference",
        "intent",
        "paraphrase",
        "reading_boolean",
        "numeric_rule",
        "ordinal_rule",
        "candidate_extraction",
        "candidate_retrieval",
    }
)


def project_report(original_dataset, dataset, predictions_path, output):
    """已有同题预测只做来源清理和重新分组；不再次推理或改变答案。"""
    original = {row["id"]: row for row in read_examples(original_dataset)}
    examples = read_examples(dataset)
    predictions = {row["id"]: row for row in read_examples(predictions_path)}
    metadata = json.loads((predictions_path.parent / "summary.json").read_text(encoding="utf-8"))[
        "metadata"
    ]
    if metadata["dataset_sha256"] != digest(original_dataset):
        raise ValueError("原始预测与原始数据哈希不符。")
    selected, responses = [], []
    for row in examples:
        old = original[row["id"]]
        if canonical_request(row) != canonical_request(old) or row["expected"] != old["expected"]:
            raise ValueError("来源重分组不可修改请求或标签。")
        record = predictions[row["id"]]
        selected.append(record)
        responses.append(EvaluationResponse.model_validate(record["response"]))
    summary, judgments = summarize(examples, responses)
    output.mkdir(parents=True, exist_ok=False)
    summary["metadata"] = {
        **metadata,
        "dataset_sha256": digest(dataset),
        "evaluated_examples": len(examples),
        "full_dataset_examples": len(examples),
        "full_dataset_evaluated": True,
        "evaluated_questions": len(judgments),
        "projected_from": str(predictions_path.parent),
        "source_predictions_sha256": digest(predictions_path),
        "timing_scope": "elapsed_seconds describes original full evaluation before projection",
    }
    write_jsonl(output / "predictions.jsonl", selected)
    write_jsonl(output / "judgments.jsonl", judgments)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def cluster_intervals(left, right, repetitions=2000, seed=20260920, metric="correct"):
    name = {
        "correct": "accuracy",
        "brier": "brier",
        "score_normalized_error": "score_normalized_mae",
    }[metric]
    if [(row["id"], row["question_id"]) for row in left] != [
        (row["id"], row["question_id"]) for row in right
    ]:
        raise ValueError("比较必须使用相同顺序的同题结果。")
    grouped = defaultdict(lambda: defaultdict(list))
    for a, b in zip(left, right, strict=True):
        if (a["family"], a["group_id"], a["expected"]) != (
            b["family"],
            b["group_id"],
            b["expected"],
        ):
            raise ValueError("同题的来源或标签不一致。")
        if (metric in a) != (metric in b):
            raise ValueError("比较指标的覆盖不一致。")
        if metric in a:
            grouped[a["family"]][a["group_id"]].append(float(a[metric]) - float(b[metric]))
    if not grouped:
        raise ValueError("没有可比较的指标记录。")
    rng = np.random.default_rng(seed)
    simulations, families = [], {}
    for family, groups in sorted(grouped.items()):
        sums = np.array([sum(values) for values in groups.values()])
        counts = np.array([len(values) for values in groups.values()])
        indices = rng.integers(0, len(sums), size=(repetitions, len(sums)))
        samples = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
        simulations.append(samples)
        families[family] = {
            "source_groups": len(groups),
            f"{name}_delta": float(sums.sum() / counts.sum()),
            f"{name}_delta_95ci": np.quantile(samples, [0.025, 0.975]).tolist(),
        }
    return {
        "method": "paired, stratified by family, resampling source groups with replacement",
        "repetitions": repetitions,
        "seed": seed,
        "families": families,
        f"macro_{name}_delta_95ci": np.quantile(
            np.mean(simulations, axis=0), [0.025, 0.975]
        ).tolist(),
    }


def compare(local: Path, reference: Path):
    left_report = json.loads((local / "summary.json").read_text(encoding="utf-8"))
    right_report = json.loads((reference / "summary.json").read_text(encoding="utf-8"))
    if left_report["metadata"]["dataset_sha256"] != right_report["metadata"]["dataset_sha256"]:
        raise ValueError("不能比较不同测试文件。")
    left, right = (
        read_examples(local / "judgments.jsonl"),
        read_examples(reference / "judgments.jsonl"),
    )
    paired = cluster_intervals(left, right)
    paired["brier"] = cluster_intervals(left, right, metric="brier")
    if any("score_normalized_error" in row for row in left):
        paired["score_normalized_mae"] = cluster_intervals(
            left, right, metric="score_normalized_error"
        )
    a, b = left_report["families"], right_report["families"]
    if set(a) != set(b):
        raise ValueError("任务覆盖不一致。")

    def macro(report, key):
        return float(np.mean([value[key] for value in report.values()]))

    accuracy_delta = macro(a, "accuracy") - macro(b, "accuracy")
    brier_delta = macro(a, "brier") - macro(b, "brier")
    score_delta = a.get("ordinal_rule", {}).get("score_normalized_mae", float("inf")) - (
        b.get("ordinal_rule", {}).get("score_normalized_mae", 0)
    )
    gates = {
        "complete_same_dataset": all(
            report["metadata"].get("full_dataset_evaluated") is True
            and report["metadata"].get("evaluated_questions") == len(rows)
            for report, rows in ((left_report, left), (right_report, right))
        )
        and len(left) == len(right),
        "fixed_jev_reference": right_report["metadata"].get("backend") == "jev"
        and right_report["metadata"].get("model") == ["jev-1.13.0"],
        "all_eight_tasks": set(a) == FAMILIES,
        "macro_accuracy_gap_at_most_3pp": accuracy_delta >= -0.03 - 1e-12,
        "each_task_gap_at_most_5pp": all(
            a[key]["accuracy"] >= b[key]["accuracy"] - 0.05 - 1e-12 for key in a
        ),
        "normalized_score_mae_excess_at_most_005": score_delta <= 0.05 + 1e-12,
        "macro_brier_excess_at_most_003": brier_delta <= 0.03 + 1e-12,
    }
    languages = {}
    for language in sorted({row["language"] for row in left}):
        local_rows = [row for row in left if row["language"] == language]
        reference_rows = [row for row in right if row["language"] == language]
        family_metrics = {
            family: {
                "local": metrics([row for row in local_rows if row["family"] == family]),
                "reference": metrics([row for row in reference_rows if row["family"] == family]),
            }
            for family in sorted({row["family"] for row in local_rows})
        }
        languages[language] = {
            "local": metrics(local_rows),
            "reference": metrics(reference_rows),
            "families": family_metrics,
            "macro_accuracy": {
                side: float(np.mean([value[side]["accuracy"] for value in family_metrics.values()]))
                for side in ("local", "reference")
            },
            "task_count": len(family_metrics),
        }
    label_quality = {
        quality: {
            "local": metrics(
                [row for row in left if row.get("label_quality", "unspecified") == quality]
            ),
            "reference": metrics(
                [row for row in right if row.get("label_quality", "unspecified") == quality]
            ),
        }
        for quality in sorted({row.get("label_quality", "unspecified") for row in left})
    }
    return {
        "local": str(local),
        "reference": str(reference),
        "dataset_sha256": left_report["metadata"]["dataset_sha256"],
        "macro_accuracy": {
            "local": macro(a, "accuracy"),
            "reference": macro(b, "accuracy"),
            "delta": accuracy_delta,
        },
        "macro_brier": {
            "local": macro(a, "brier"),
            "reference": macro(b, "brier"),
            "delta": brier_delta,
        },
        "normalized_score_mae_delta": score_delta,
        "families": {key: {"local": a[key], "reference": b[key]} for key in a},
        "languages": languages,
        "label_quality": label_quality,
        "language_coverage_note": "Languages may cover different task sets; compare within tasks. "
        "BoolQ and SciFact are English only in this experiment.",
        "paired_uncertainty": paired,
        "performance_gates": gates,
        "performance_pass": all(gates.values()),
        "notes": "Predefined point-estimate acceptance margins. "
        "Protocol/reload/robustness/resource checks are separately required.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("local", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.local, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: result[key] for key in ("macro_accuracy", "macro_brier", "performance_gates")},
            indent=2,
        )
    )
