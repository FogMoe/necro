"""冻结统一重训候选，验证独立测试和最终导出；拒绝覆盖已有结果。"""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from necro.engine import prompt_fingerprint
from necro.evaluation import run_evaluation
from necro.experiment_guard import digest, verify
from necro.training.analysis.condition_balance import condition_slices, minimum_accuracy
from necro.training.cloud import (
    DATA,
    EVALUATION,
    OUTPUT,
    REFERENCES,
    acceptance_checks,
    read,
    release_memory,
    settings,
    validate_report,
    verify_bundle,
)
from necro.training.release.freeze_phase3 import reject_exposed_tests
from necro.training.release.stability_assessment import language_alerts, paired_comparison

SELECTED = OUTPUT / "selected"
FINAL = OUTPUT / "final"


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def validate_recipe(contract, plan, recipe, train_hash):
    expected = {
        "checkpoint": plan["checkpoint"],
        "revision": plan["revision"],
        "initial_adapter": None,
        "rank": plan["rank"],
        "epochs": plan["epochs"],
        "seed": plan["seed"],
        "objective": plan["objective"],
        "gradient_checkpointing": plan["gradient_checkpointing"],
        "learning_rate": plan["recipes"][recipe]["learning_rate"],
        "train_sha256": train_hash,
    }
    if any(key not in contract or contract[key] != value for key, value in expected.items()):
        raise ValueError("训练记录与冻结方案不符，不能冻结候选。")
    if contract["batch_size"] * contract["gradient_accumulation"] != plan["effective_batch_size"]:
        raise ValueError("训练的有效 batch 与方案不符。")


def phase4_gain(comparisons, keys):
    return all(
        key in comparisons and comparisons[key]["macro_accuracy"]["delta"] > 1e-12 for key in keys
    )


def freeze(recipe, decision_path):
    verify_bundle()
    if SELECTED.exists() or (DATA / "selection.json").exists() or FINAL.exists():
        raise ValueError("候选或最终测试目录已存在；保留并检查已有结果。")
    run = OUTPUT / recipe
    plan = read(Path("cloud-plan.json"))
    decision = read(decision_path)
    if (
        decision.get("recipe") != recipe
        or not decision.get("rationale")
        or decision.get("independent_test_opened") is not False
    ):
        raise ValueError("选择依据必须注明方案、开发阶段理由和测试未开启。")
    review = read(run / "assessment.json")
    adapter = run / "adapter"
    weights = digest(adapter / "adapter_model.safetensors")
    fit_path = run / "calibration-brier-fit.json"
    fit = read(fit_path)
    if (
        review.get("eligible_for_freeze") is not True
        or review.get("independent_test_opened") is not False
        or review["weights_sha256"] != weights
        or review["calibration_sha256"] != digest(fit_path)
        or review["data_manifest_sha256"] != digest(DATA / "experiment.json")
    ):
        raise ValueError("开发验收未通过，或权重、校准、数据已变更。")
    parent = condition_slices(DATA / "validation.jsonl", OUTPUT / "baseline/parent/development")
    checks = acceptance_checks(
        read(run / "slices.json"), parent, review["comparisons"], review["language_alerts"]
    )
    if checks != review["checks"] or not all(checks.values()):
        raise ValueError("无法从开发证据复核全部验收条件。")
    if not phase4_gain(review["comparisons"], ("retention-repaired", "regression-repaired")):
        raise ValueError("相对 Phase 4 尚无八任务整体净提升，不能冻结为发布候选。")
    validate_recipe(
        read(adapter / "necro_adapter.json"), plan, recipe, digest(verify(DATA, "train"))
    )
    calibration = verify(DATA, "calibration")
    validate_report(run / "calibration", calibration, adapter, None)
    if (
        fit["dataset_sha256"] != digest(calibration)
        or fit["predictions_sha256"] != digest(run / "calibration/predictions.jsonl")
        or fit.get("objective") != "brier"
        or set(fit["temperatures"]) != {"choice", "noul", "score"}
    ):
        raise ValueError("校准拟合与该候选的冻结校准集预测不符。")
    manifest = read(DATA / "experiment.json")
    reject_exposed_tests({manifest["partitions"]["test"]["sha256"]})
    SELECTED.mkdir(parents=True)
    shutil.copytree(adapter, SELECTED / "adapter")
    contract = read(SELECTED / "adapter/necro_adapter.json")
    contract["training_model_id"] = contract["model_id"]
    contract["model_id"] = "ScarletKc-Necro-0.8b"
    contract["recommended_temperatures"] = fit["temperatures"]
    (SELECTED / "adapter/necro_adapter.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    for source, name in (
        (fit_path, "calibration.json"),
        (decision_path, "decision.json"),
        (run / "assessment.json", "development-assessment.json"),
        (Path("cloud-plan.json"), "cloud-plan.json"),
        (DATA / "audit.json", "data-audit.json"),
        (DATA / "experiment.json", "data-manifest.json"),
    ):
        shutil.copy2(source, SELECTED / name)
    validation = {
        "independent_test": manifest["partitions"]["test"],
        "task_regression": {
            "source": str(EVALUATION / "regression.jsonl"),
            "sha256": digest(EVALUATION / "regression.jsonl"),
        },
        "condition_minimum": 0.95,
        "complete_eligible_max_drop": 0.02,
        "family_max_drop": 0.02,
        "macro_max_drop_vs_repaired": 0.005,
        "release_target": "Positive eight-family macro gain over Phase 4; Jev is not a gate",
        "language_rule": "Existing language_alerts, paired source-group confidence intervals",
        "test_failure": "Stop; no learning-rate, weight or calibration changes using this test",
        "runtime": "Full independent and regression reload; real SDK and HTTP; no Hub upload",
    }
    write_new(SELECTED / "validation-plan.json", validation)
    shutil.copytree(
        "src/necro", SELECTED / "source/necro", ignore=shutil.ignore_patterns("__pycache__")
    )
    reference_weights, reference_temperatures = [], {}
    external = {}
    for name in REFERENCES:
        root = Path("reference") / name
        weight_hash = digest(root / "adapter/adapter_model.safetensors")
        reference_weights.append(weight_hash)
        reference_temperatures[weight_hash] = read(root / "calibration.json")["temperatures"]
        for path in root.rglob("*"):
            if path.is_file():
                external[str(path)] = digest(path)
    hashes = {str(p): digest(p) for p in SELECTED.rglob("*") if p.is_file()}
    hashes.update(external)
    hashes.update({str(p): digest(p) for p in (Path("uv.lock"), Path("bundle.json"))})
    hashes.update({str(p): digest(p) for p in Path("src/necro").rglob("*.py")})
    for path in (
        run / "assessment.json",
        run / "slices.json",
        run / "run_config.json",
        run / "calibration/summary.json",
        run / "calibration/predictions.jsonl",
    ):
        hashes[str(path)] = digest(path)
    selection = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "adapter": str(SELECTED / "adapter"),
        "weights_sha256": weights,
        "temperatures": fit["temperatures"],
        "reference_weights_sha256": reference_weights,
        "reference_temperatures": reference_temperatures,
        "prompt_sha256": prompt_fingerprint(),
        "data_manifest_sha256": digest(DATA / "experiment.json"),
        "artifact_hashes": hashes,
        "final_predictions_available_at_selection": False,
    }
    write_new(SELECTED / "selection.json", selection)
    write_new(DATA / "selection.json", selection)
    verify(DATA, "test", SELECTED / "adapter")
    print(json.dumps({"frozen": recipe, "weights_sha256": weights}), flush=True)


def independent_checks(slices, comparisons, alerts, plan):
    required = set(REFERENCES)
    keys = [f"{language}/{case}" for language in ("en", "zh") for case in ("complete", "eligible")]
    prerequisite_keys = [
        f"{language}/{case}" for language in ("en", "zh") for case in ("gate_false", "missing")
    ]
    return {
        "independent_preconditions": minimum_accuracy(
            slices["selected"], prerequisite_keys, plan["condition_minimum"]
        ),
        "independent_complete_eligible_retained": all(
            key in slices["parent"]
            and slices["parent"][key]["count"] > 0
            and minimum_accuracy(
                slices["selected"],
                [key],
                slices["parent"][key]["accuracy"] - plan["complete_eligible_max_drop"],
            )
            for key in keys
        ),
        "independent_families_retained": set(comparisons) == required
        and all(
            len(report["families"]) == 8
            and all(
                values["local"]["accuracy"]
                >= values["reference"]["accuracy"] - plan["family_max_drop"] - 1e-12
                for values in report["families"].values()
            )
            for report in comparisons.values()
        ),
        "independent_macro_retained": "repaired" in comparisons
        and comparisons["repaired"]["macro_accuracy"]["delta"]
        >= -plan["macro_max_drop_vs_repaired"] - 1e-12,
        "independent_phase4_net_gain": phase4_gain(comparisons, ("repaired",)),
        "independent_languages_retained": set(alerts) == required and not any(alerts.values()),
    }


def test():
    verify_bundle()
    selection = read(DATA / "selection.json")
    adapter = Path(selection["adapter"])
    dataset = verify(DATA, "test", adapter)
    if FINAL.exists():
        raise ValueError("最终测试目录已存在；保留并检查已有结果，不能覆盖。")
    FINAL.mkdir(parents=True)
    write_new(FINAL / "started.json", {"selection_sha256": digest(DATA / "selection.json")})
    slices = {}
    for name in ("selected", *REFERENCES):
        weights = adapter if name == "selected" else Path("reference") / name / "adapter"
        temperatures = (
            selection["temperatures"]
            if name == "selected"
            else read(weights.parent / "calibration.json")["temperatures"]
        )
        target = FINAL / name
        run_evaluation(dataset, target, settings(weights, temperatures))
        validate_report(target, dataset, weights, temperatures)
        slices[name] = condition_slices(dataset, target)
        release_memory()
    comparisons = {
        name: paired_comparison(FINAL / "selected", FINAL / name, "independent unified test")
        for name in REFERENCES
    }
    alerts = {name: language_alerts(FINAL / "selected", FINAL / name) for name in REFERENCES}
    checks = independent_checks(
        slices, comparisons, alerts, read(SELECTED / "validation-plan.json")
    )
    write_new(
        FINAL / "assessment.json",
        {
            "checks": checks,
            "passed": all(checks.values()),
            "slices": slices,
            "comparisons": comparisons,
            "language_alerts": alerts,
            "selection_sha256": digest(DATA / "selection.json"),
            "independent_test_opened": True,
        },
    )
    print(
        json.dumps({"independent_test_passed": all(checks.values()), "checks": checks}), flush=True
    )


def package():
    from necro.export import export
    from necro.training.release.package_verification import verify_package

    verify_bundle()
    selection = read(DATA / "selection.json")
    adapter = Path(selection["adapter"])
    verify(DATA, "test", adapter)
    review = read(FINAL / "assessment.json")
    if not review["passed"] or review["selection_sha256"] != digest(DATA / "selection.json"):
        raise ValueError("独立测试未通过或冻结记录发生变更，不能进入最终导出。")
    target = FINAL / "selected-regression"
    package_path = OUTPUT / "export"
    if target.exists() or package_path.exists():
        raise ValueError("导出或回归结果已存在；保留并检查已有结果。")
    run_evaluation(
        EVALUATION / "regression.jsonl", target, settings(adapter, selection["temperatures"])
    )
    release_memory()
    export(adapter, package_path, calibration_file=SELECTED / "calibration.json")
    release_memory()
    verify_package(
        package_path,
        DATA,
        FINAL / "selected",
        FINAL / "package-verification.json",
        regression_data=EVALUATION / "regression.jsonl",
        regression_predictions=target,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "test", "package"))
    parser.add_argument("--recipe", choices=("primary", "lower-rate"))
    parser.add_argument("--decision", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        if args.recipe is None or args.decision is None:
            parser.error("freeze 需要 --recipe 和 --decision")
        freeze(args.recipe, args.decision)
    elif args.action == "test":
        test()
    else:
        package()


if __name__ == "__main__":
    main()
