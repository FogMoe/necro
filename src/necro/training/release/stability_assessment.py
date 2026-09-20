"""Assess condition repair and task retention without Jev percentage gates."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from necro.diagnostics.numeric_regression import rule_metadata
from necro.evaluation import metrics, read_examples
from necro.experiment_guard import digest, verify
from necro.training.analysis.phase2_analysis import cluster_intervals, compare


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def paired_comparison(candidate, reference, scope):
    for path in (candidate, reference):
        report = read(path / "summary.json")
        if not report["metadata"].get("full_dataset_evaluated"):
            raise ValueError("A stability comparison requires complete cohorts")
    result = compare(candidate, reference)
    for key in ("performance_gates", "performance_pass", "notes", "language_coverage_note"):
        result.pop(key)
    if "ordinal_rule" not in result["families"]:
        result["normalized_score_mae_delta"] = None
    result["scope"] = scope
    result["notes"] = "Jev is a descriptive reference. Release checks concern repair and retention."
    return result


def condition_slices(dataset, result):
    examples = {r["id"]: r for r in read_examples(dataset) if r["family"] == "numeric_rule"}
    grouped = defaultdict(list)
    for row in read_examples(result / "judgments.jsonl"):
        if row["id"] not in examples:
            continue
        meta = rule_metadata(examples[row["id"]])
        for key in (
            "all",
            meta["case"],
            meta["language"],
            f"{meta['language']}/{meta['case']}",
            f"{meta['case']}/{'negative' if meta['negated'] else 'positive'}",
            f"operator/{meta['operator']}",
            f"{meta['case']}/operator/{meta['operator']}",
        ):
            grouped[key].append(row)
    return {
        key: {**metrics(rows), "correct": sum(r["correct"] for r in rows)}
        for key, rows in grouped.items()
    }


def language_alerts(candidate, reference):
    left = read_examples(candidate / "judgments.jsonl")
    right = read_examples(reference / "judgments.jsonl")
    alerts = []
    for family, language in sorted({(r["family"], r["language"]) for r in left}):
        if family == "numeric_rule":
            continue
        a = [r for r in left if (r["family"], r["language"]) == (family, language)]
        b = [r for r in right if (r["family"], r["language"]) == (family, language)]
        interval = cluster_intervals(a, b)["families"][family]
        if interval["accuracy_delta"] < -0.05 - 1e-12 and interval["accuracy_delta_95ci"][1] < 0:
            alerts.append({"family": family, "language": language, **interval})
    return alerts


def assess(data, selected, final):
    output = final / "capability-review.json"
    if output.exists():
        raise ValueError("Capability review already exists")
    selection = read(selected / "selection.json")
    contract = read(Path(selection["adapter"]) / "necro_adapter.json")
    plan = read(selected / "validation-plan.json")
    test = verify(data, "test", Path(selection["adapter"]))
    regression = Path(plan["task_regression"]["source"])
    if (
        digest(test) != plan["condition_test"]["sha256"]
        or digest(regression) != plan["task_regression"]["sha256"]
    ):
        raise ValueError("Validation cohorts do not match the frozen plan")
    for name, dataset in (("conditions", test), ("regression", regression)):
        for model in (
            ("local", "predecessor", "jev")
            if name == "conditions"
            else ("local", "predecessor", "instruction", "jev")
        ):
            metadata = read(final / f"{model}-{name}/summary.json")["metadata"]
            if (
                metadata["dataset_sha256"] != digest(dataset)
                or not metadata["full_dataset_evaluated"]
            ):
                raise ValueError("Result does not cover the complete registered cohort")
            if model == "local" and (
                metadata.get("adapter_weights_sha256") != selection["weights_sha256"]
                or any(
                    metadata.get(f"{p}_temperature") != t
                    for p, t in selection["temperatures"].items()
                )
            ):
                raise ValueError("Local results do not match selected weights and calibration")
            if (
                model == "predecessor"
                and metadata.get("adapter_weights_sha256")
                not in selection["reference_weights_sha256"]
            ):
                raise ValueError("Predecessor results do not match the frozen reference")
            if (
                model == "instruction"
                and metadata.get("adapter_weights_sha256") != contract["initial_adapter_sha256"]
            ):
                raise ValueError("Instruction results do not match the training parent")
            if model == "jev" and (
                metadata["backend"] != "jev" or metadata["model"] != ["jev-1.13.0"]
            ):
                raise ValueError("Jev results must use the fixed reference model")
    comparisons = {}
    for cohort, references in (
        ("conditions", ("predecessor", "jev")),
        ("regression", ("predecessor", "instruction", "jev")),
    ):
        scope = (
            "independent condition transfer"
            if cohort == "conditions"
            else "exposed multi-task regression"
        )
        for model in references:
            key = f"{cohort}-{model}"
            comparisons[key] = paired_comparison(
                final / f"local-{cohort}", final / f"{model}-{cohort}", scope
            )
            write(final / f"{key}-comparison.json", comparisons[key])
    conditions = {
        m: condition_slices(test, final / f"{m}-conditions")
        for m in ("local", "predecessor", "jev")
    }
    legacy = {
        m: condition_slices(regression, final / f"{m}-regression")
        for m in ("local", "predecessor", "instruction", "jev")
    }
    alerts = {
        m: language_alerts(final / "local-regression", final / f"{m}-regression")
        for m in ("predecessor", "instruction")
    }
    condition_plan = plan["condition_test"]
    checks = {
        "independent_gate_and_missing": all(
            conditions["local"][k]["accuracy"]
            >= condition_plan["disabled_gate_and_missing_accuracy_minimum"]
            for k in ("gate_false", "missing")
        ),
        "independent_gate_and_missing_each_language": all(
            conditions["local"][f"{lang}/{case}"]["accuracy"]
            >= condition_plan["each_language_disabled_gate_and_missing_accuracy_minimum"]
            for lang in ("en", "zh")
            for case in ("gate_false", "missing")
        ),
        "independent_condition_accuracy_retained": conditions["local"]["all"]["accuracy"]
        >= conditions["predecessor"]["all"]["accuracy"]
        - condition_plan["overall_accuracy_margin_against_predecessor"]
        - 1e-12,
        "other_task_accuracy_retained": all(
            v["local"]["accuracy"]
            >= v["reference"]["accuracy"]
            - plan["task_regression"]["other_task_accuracy_margin_against_both_baselines"]
            - 1e-12
            for model in ("predecessor", "instruction")
            for task, v in comparisons[f"regression-{model}"]["families"].items()
            if task != "numeric_rule"
        ),
        "no_material_language_regression": not any(alerts.values()),
        "original_condition_defect_repaired": legacy["local"]["all"]["accuracy"]
        >= legacy["predecessor"]["all"]["accuracy"]
        and all(legacy["local"][k]["accuracy"] >= 0.9 for k in ("gate_false", "missing")),
    }
    failures = []
    for name, dataset in (("conditions", test), ("regression", regression)):
        examples = {r["id"]: r for r in read_examples(dataset)}
        reference = {
            (r["id"], r["question_id"]): r
            for r in read_examples(final / f"jev-{name}/judgments.jsonl")
        }
        for row in read_examples(final / f"local-{name}/judgments.jsonl"):
            if row["correct"]:
                continue
            record = {"cohort": name, **row, "jev": reference[(row["id"], row["question_id"])]}
            if row["family"] == "numeric_rule":
                record["request"] = examples[row["id"]]["request"]
                record["condition"] = rule_metadata(examples[row["id"]])
            failures.append(record)
    with (final / "failures.jsonl").open("w", encoding="utf-8") as stream:
        for row in failures:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    result = {
        "adapter_weights_sha256": selection["weights_sha256"],
        "checks": checks,
        "passed": all(checks.values()),
        "condition_slices": conditions,
        "original_condition_slices": legacy,
        "language_alerts": alerts,
        "evidence_sha256": {
            str(p.relative_to(final)): digest(p)
            for p in final.rglob("*")
            if p.is_file() and "remote-cache" not in p.parts
        },
    }
    write(output, result)
    print(json.dumps({"checks": checks, "passed": result["passed"]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("data", "selected", "final"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    assess(args.data, args.selected, args.final)
