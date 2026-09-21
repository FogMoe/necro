"""分别报告语义资格、问题极性和前置条件，防止总准确率掩盖误拒。"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from necro.diagnostics.numeric_regression import rule_metadata
from necro.evaluation import read_examples
from necro.experiment_guard import digest
from necro.training.data.condition_balance import semantic_outcome


def minimum_accuracy(slices, keys, minimum):
    """缺少任何必检切片也算失败，不能由总准确率补足。"""
    return all(
        key in slices and slices[key]["count"] > 0 and slices[key]["accuracy"] >= minimum - 1e-12
        for key in keys
    )


def condition_checks(development, regression, fixture):
    return {
        "known_false_rejections_resolved": minimum_accuracy(fixture, ["all"], 1.0),
        "complete_and_eligible_retained": minimum_accuracy(
            regression,
            [
                f"{language}/{case}"
                for language in ("en", "zh")
                for case in ("complete", "eligible")
            ],
            0.98,
        ),
        "exposed_preconditions_repaired": minimum_accuracy(
            regression,
            [
                f"{language}/{case}"
                for language in ("en", "zh")
                for case in ("gate_false", "missing")
            ],
            0.95,
        ),
        "development_transfer_repaired": minimum_accuracy(
            development,
            [
                f"transfer/{language}/{case}"
                for language in ("en", "zh")
                for case in ("eligible", "gate_false", "missing")
            ],
            0.95,
        ),
    }


def condition_slices(dataset, predictions):
    examples = {r["id"]: r for r in read_examples(dataset)}
    metadata = json.loads((predictions / "summary.json").read_text())["metadata"]
    if metadata["dataset_sha256"] != digest(dataset) or not metadata["full_dataset_evaluated"]:
        raise ValueError("Condition analysis requires complete predictions of the same dataset")
    judgments = read_examples(predictions / "judgments.jsonl")
    if len(judgments) != len(examples) or {r["id"] for r in judgments} != set(examples):
        raise ValueError("Condition judgments must cover every example exactly once")
    groups = defaultdict(list)
    for row in judgments:
        example = examples[row["id"]]
        if example["family"] != "numeric_rule":
            continue
        meta = rule_metadata(example)
        eligible = semantic_outcome(example)
        semantic = "eligible" if eligible else "ineligible"
        language, case = meta["language"], meta["case"]
        style = example.get("condition_style", "legacy")
        state = example["request"]["state"]
        negative = any(state.get(field, 0) < 0 for field in meta["fields"][:2])
        prefix = "transfer" if example["source"] == "constructed-condition-balance" else "existing"
        for key in (
            "all",
            semantic,
            case,
            language,
            f"{language}/{semantic}",
            f"{language}/{case}",
            f"style-{style}/{language}/{case}",
            f"operator-{meta['operator']}/{semantic}",
            f"negated-{meta['negated']}/{semantic}",
            f"{prefix}/all",
            f"{prefix}/{case}",
            f"{prefix}/{language}/{semantic}",
            f"{prefix}/{language}/{case}",
            f"negative-values-{negative}/{language}/{semantic}",
            f"negative-values-{negative}/{language}/{case}",
        ):
            groups[key].append(row)
    return {
        key: {
            "count": len(rows),
            "correct": sum(r["correct"] for r in rows),
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "brier": sum(r["brier"] for r in rows) / len(rows),
            "source_groups": len({r["group_id"] for r in rows}),
        }
        for key, rows in sorted(groups.items())
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    print(json.dumps(condition_slices(args.dataset, args.predictions), indent=2))
