"""准备、校验和运行冻结的云端重训包；准备阶段不连接云主机。"""
# ruff: noqa: E501

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from necro.config import Settings
from necro.diagnostics.numeric_regression import write_json
from necro.evaluation import read_examples, run_evaluation
from necro.experiment_guard import canonical_request, digest, register, verify
from necro.training.analysis.assess_candidate import assess
from necro.training.analysis.condition_balance import condition_slices, minimum_accuracy
from necro.training.analysis.phase2_analysis import project_report
from necro.training.release.stability_assessment import language_alerts, paired_comparison
from necro.training.trainer import train

DATA = Path("data/phase6/unified-v1")
EVALUATION = Path("data/phase6/regression-v1")
OUTPUT = Path("results/cloud")
BASE = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
REFERENCES = {
    "parent": (
        Path("results/phase3/instruction-seed2026/adapter"),
        Path("results/phase3/instruction-seed2026/calibration-brier-fit.json"),
    ),
    "repaired": (
        Path("results/phase4/selected/adapter"),
        Path("results/phase4/selected/calibration.json"),
    ),
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def safe_file(root, relative):
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Bundle path escapes its root: {relative}")
    return target


def verify_bundle(root=Path(".")):
    manifest = read(root / "bundle.json")
    for name, expected in manifest["files"].items():
        path = safe_file(root, name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    return manifest


def build_bundle(output):
    if output.exists() or output.with_suffix(".zip").exists():
        raise ValueError(f"Bundle destination already exists: {output}")
    verify(DATA, "train")
    if (DATA / "selection.json").exists():
        raise ValueError("Preparation requires an unselected, unopened experiment")
    from necro.training.release.freeze_phase3 import reject_exposed_tests

    reject_exposed_tests(
        {digest(DATA / "test.jsonl"), digest(Path("data/phase5/condition-balance-v1/test.jsonl"))}
    )
    output.mkdir(parents=True)
    for name in ("src", "tests", "docs", "licenses", "examples"):
        shutil.copytree(name, output / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in (
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "LICENSE",
        "LICENSE-MIT",
        "THIRD_PARTY_NOTICES.md",
        ".env.example",
    ):
        shutil.copy2(name, output / name)
    shutil.copytree(DATA, output / DATA)
    evaluation = output / EVALUATION
    evaluation.mkdir(parents=True)
    files = {}
    for role, source in (
        ("retention", "data/phase3/coverage-v1/validation.jsonl"),
        ("regression", "data/phase4/phase3-regression-v1/regression.jsonl"),
        ("conditions", "data/phase4/condition-regression-v1/regression.jsonl"),
    ):
        files[role] = evaluation / f"{role}.jsonl"
        shutil.copy2(source, files[role])
    register(
        evaluation, files, {"scope": "Exposed regression cohorts; never independent acceptance"}
    )

    # 参考模型的温度必须来自同一校准题集；允许行顺序不同。
    def cohort(path):
        return sorted(
            (canonical_request(r), json.dumps(r["expected"], sort_keys=True))
            for r in read_examples(path)
        )

    if cohort(DATA / "calibration.jsonl") != cohort(
        Path("data/phase3/coverage-v1/calibration.jsonl")
    ):
        raise ValueError("Reference calibration questions differ from the unified cohort")
    refs = {}
    for name, (adapter, calibration) in REFERENCES.items():
        target = output / "reference" / name
        target.mkdir(parents=True)
        shutil.copytree(adapter, target / "adapter")
        shutil.copy2(calibration, target / "calibration.json")
        refs[name] = {
            "weights_sha256": digest(adapter / "adapter_model.safetensors"),
            "calibration_sha256": digest(calibration),
        }
    plan = {
        "checkpoint": BASE,
        "revision": REVISION,
        "initial_adapter": None,
        "rank": 16,
        "objective": "answer-ce",
        "epochs": 2,
        "effective_batch_size": 16,
        "seed": 2026,
        "gradient_checkpointing": True,
        "recipes": {"primary": {"learning_rate": 1e-4}, "lower-rate": {"learning_rate": 5e-5}},
        "budget": "At most two fresh recipes. Assess primary before lower-rate. No automatic continuation, Base run or repair-only follow-up. A passing recipe may receive one separately registered seed confirmation.",
        "selection": {
            "known_38_failures": "all corrected",
            "exposed_conditions": "complete and eligible >=98% per language; disabled and missing >=95% per language",
            "retention": "each original development and regression family within2pp of both references; original-task macro not below repaired reference by more than0.5pp",
            "new_development": "disabled and missing >=95% per language; complete and eligible within2pp of parent, with negative arithmetic reported separately",
            "stopping": "A failed recipe is not promoted. If both fail, stop and review the overall model/data approach. Freeze weights/calibration before independent testing; no test-driven recipe changes.",
        },
        "final_validation": "Independent eight-family test, followed by merged export, full reload and real SDK/HTTP checks; not run automatically by this preparation or development runner.",
        "data_manifest_sha256": digest(output / DATA / "experiment.json"),
        "references": refs,
    }
    write_json(output / "cloud-plan.json", plan)
    audit = read(DATA / "audit.json")
    instructions = f"""# Unified LoRA retraining on a Linux GPU host

This bundle starts a new LoRA from `{BASE}` at revision `{REVISION}`. It contains frozen training/evaluation data and two reference adapters. No cloud job has been started by packaging it.

Use a Linux GPU environment with Python/uv and a driver compatible with the locked CUDA 12.8 PyTorch wheel. Install the project in this extracted directory:

```bash
uv sync --locked --python 3.12 --extra training
uv run --extra training python -m necro.training.cloud verify
uv run --extra training python -m necro.training.cloud profile
uv run --extra training python -m necro.training.cloud baselines
uv run --extra training python -m necro.training.cloud train primary
uv run --extra training python -m necro.training.cloud assess primary
```

The profile action measures checkpointed microbatches 4, 8 and 16, including the largest padded batch. It records the selected microbatch while keeping effective batch size 16. Training rejects a changed model revision or an existing output directory. No previous project adapter is loaded for training.

The dataset has {audit["data"]["train"]["rows"]:,} training records, {audit["train_tokens_per_epoch"]:,} unpadded tokens per epoch, {audit["data"]["development"]["rows"]:,} exposed development records and {audit["data"]["test"]["rows"]:,} sealed test records. Actual padding, memory and elapsed-time estimates come from the GPU profile. The standard installation uses the project's locked runtime; no untested kernel packages are installed automatically.

If the primary result justifies the registered lower-rate comparison, run the same train/assess commands with `lower-rate`. Do not open `test.jsonl` to choose a recipe. Independent acceptance and export verification follow an explicit candidate freeze. Keep `results/cloud/` in full when returning evidence.

`cloud-plan.json` owns the recipe, acceptance criteria and stopping budget. `data/phase6/unified-v1/audit.json` records source protection, task/language counts and limits. Intent test coverage is partial; BoolQ and SciFact are English-only. Historical evidence and reference models remain unchanged. The builder requires the original source snapshots only when rebuilding data; cloud training uses the supplied frozen JSONL files.

No `.env`, access token, SSH key, Git metadata or raw source cache is included. Public base weights download from Hugging Face on first use. This package neither uploads a model nor rents a server.
"""
    (output / "CLOUD_RUN.md").write_text(instructions, encoding="utf-8")
    hashes = {
        p.relative_to(output).as_posix(): digest(p)
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    if any(Path(name).name == ".env" or ".git" in Path(name).parts for name in hashes):
        raise ValueError("Unexpected private workspace file in bundle")
    write_json(output / "bundle.json", {"format": 1, "files": hashes})
    verify_bundle(output)
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(output).as_posix())
    with zipfile.ZipFile(archive) as handle:
        if handle.testzip() is not None:
            raise ValueError("Archive CRC verification failed")
    print(
        json.dumps(
            {
                "bundle": str(output.resolve()),
                "archive": str(archive.resolve()),
                "bytes": archive.stat().st_size,
                "sha256": digest(archive),
            }
        ),
        flush=True,
    )


def release_memory():
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def settings(adapter=None, temperatures=None):
    return Settings(
        checkpoint=BASE,
        adapter=str(adapter) if adapter else None,
        device="cuda",
        **{
            f"{key}_temperature": (temperatures or {}).get(key, 1.0)
            for key in ("choice", "noul", "score")
        },
    )


def profile(plan):
    import torch

    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use a GPU-enabled host and compatible driver")
    paths = []
    for batch in (4, 8, 16):
        path = OUTPUT / f"profile-b{batch}"
        if path.exists():
            raise ValueError(f"Profile exists; preserve it and inspect before rerunning: {path}")
        command = [
            sys.executable,
            "-m",
            "necro.training",
            "--data",
            str(DATA),
            "--output",
            str(path),
            "--batch-size",
            str(batch),
            "--accumulation",
            str(16 // batch),
            "--epochs",
            str(plan["epochs"]),
            "--seed",
            str(plan["seed"]),
            "--expected-revision",
            plan["revision"],
            "--profile-only",
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode == 0:
            paths.append(path)
        else:
            print(
                json.dumps({"failed_profile": str(path), "returncode": completed.returncode}),
                flush=True,
            )
    memory = torch.cuda.get_device_properties(0).total_memory / 2**30
    feasible = [
        p for p in paths if read(p / "profile.json")["peak_allocated_gib"] + 1 < memory * 0.8
    ]
    if not feasible:
        raise ValueError("No profile left sufficient memory headroom")
    best = min(feasible, key=lambda p: read(p / "profile.json")["estimated_training_seconds"])
    batch = read(best / "run_config.json")["batch_size"]
    write_json(
        OUTPUT / "performance.json",
        {
            "gpu": torch.cuda.get_device_name(),
            "total_memory_gib": memory,
            "batch_size": batch,
            "accumulation": 16 // batch,
            "profile": str(best),
            "data_manifest_sha256": digest(DATA / "experiment.json"),
            "torch": torch.__version__,
        },
    )


def baselines():
    for name in ("official", *REFERENCES):
        adapter = Path("reference") / name / "adapter" if name != "official" else None
        temps = (
            read(Path("reference") / name / "calibration.json")["temperatures"] if adapter else None
        )
        for cohort, path in (
            ("development", DATA / "validation.jsonl"),
            ("regression", EVALUATION / "regression.jsonl"),
            ("conditions", EVALUATION / "conditions.jsonl"),
        ):
            target = OUTPUT / "baseline" / name / cohort
            if target.exists():
                raise ValueError(f"Baseline results already exist: {target}")
            run_evaluation(path, target, settings(adapter, temps))
            validate_report(target, path, adapter, temps)
            release_memory()
        project_report(
            DATA / "validation.jsonl",
            EVALUATION / "retention.jsonl",
            OUTPUT / "baseline" / name / "development/predictions.jsonl",
            OUTPUT / "baseline" / name / "retention",
        )


def assess_run(name):
    run = OUTPUT / name
    for reference in REFERENCES:
        for cohort in ("development", "regression", "conditions", "retention"):
            if not (OUTPUT / "baseline" / reference / cohort / "summary.json").exists():
                raise ValueError("Run both reference baselines before assessing a candidate")
    assess(DATA, run / "adapter", run, objective="brier")
    temps = read(run / "calibration-brier-fit.json")["temperatures"]
    for cohort in ("regression", "conditions"):
        target = run / cohort
        if target.exists():
            raise ValueError(f"Results already exist: {target}")
        run_evaluation(EVALUATION / f"{cohort}.jsonl", target, settings(run / "adapter", temps))
        release_memory()
    project_report(
        DATA / "validation.jsonl",
        EVALUATION / "retention.jsonl",
        run / "development-brier/predictions.jsonl",
        run / "retention",
    )
    fixture = Path("docs/process/evidence/condition-false-rejections-2026-09-21.jsonl")
    project_report(
        EVALUATION / "conditions.jsonl",
        fixture,
        run / "conditions/predictions.jsonl",
        run / "fixture",
    )
    slices = {
        "conditions": condition_slices(EVALUATION / "conditions.jsonl", run / "conditions"),
        "development": condition_slices(DATA / "validation.jsonl", run / "development-brier"),
        "fixture": condition_slices(fixture, run / "fixture"),
    }
    write_json(run / "slices.json", slices)
    comparisons, alerts = {}, {}
    for reference in REFERENCES:
        reference_adapter = Path("reference") / reference / "adapter"
        reference_temperatures = read(reference_adapter.parent / "calibration.json")["temperatures"]
        for cohort, filename in (
            ("development", DATA / "validation.jsonl"),
            ("conditions", EVALUATION / "conditions.jsonl"),
        ):
            validate_report(
                OUTPUT / "baseline" / reference / cohort,
                filename,
                reference_adapter,
                reference_temperatures,
            )
        for cohort in ("retention", "regression"):
            key = f"{cohort}-{reference}"
            baseline = OUTPUT / "baseline" / reference / cohort
            validate_report(
                baseline, EVALUATION / f"{cohort}.jsonl", reference_adapter, reference_temperatures
            )
            validate_report(run / cohort, EVALUATION / f"{cohort}.jsonl", run / "adapter", temps)
            comparisons[key] = paired_comparison(
                run / cohort, baseline, "exposed development selection"
            )
            alerts[key] = language_alerts(run / cohort, baseline)
    parent = condition_slices(DATA / "validation.jsonl", OUTPUT / "baseline/parent/development")
    checks = acceptance_checks(slices, parent, comparisons, alerts)
    write_json(
        run / "assessment.json",
        {
            "checks": checks,
            "eligible_for_freeze": all(checks.values()),
            "independent_test_opened": False,
            "comparisons": comparisons,
            "language_alerts": alerts,
            "weights_sha256": digest(run / "adapter/adapter_model.safetensors"),
            "calibration_sha256": digest(run / "calibration-brier-fit.json"),
            "data_manifest_sha256": digest(DATA / "experiment.json"),
        },
    )
    print(
        json.dumps(
            {
                "run": name,
                "known_failures_corrected": slices["fixture"]["all"],
                "eligible_for_freeze": all(checks.values()),
                "checks": checks,
                "independent_test_opened": False,
            }
        ),
        flush=True,
    )


def validate_report(result, dataset, adapter, temperatures):
    metadata = read(result / "summary.json")["metadata"]
    if (
        metadata.get("dataset_sha256") != digest(dataset)
        or not metadata.get("full_dataset_evaluated")
        or metadata.get("revision") != REVISION
        or metadata.get("adapter_weights_sha256")
        != (digest(adapter / "adapter_model.safetensors") if adapter else None)
        or any(
            metadata.get(f"{p}_temperature") != (temperatures or {}).get(p, 1)
            for p in ("choice", "noul", "score")
        )
    ):
        raise ValueError(f"Evaluation provenance differs from the frozen comparison: {result}")


def acceptance_checks(slices, parent, comparisons, alerts):
    def keys(cases, prefix=""):
        return [f"{prefix}{language}/{case}" for language in ("en", "zh") for case in cases]

    retention = comparisons.values()
    required = {
        f"{cohort}-{reference}"
        for cohort in ("retention", "regression")
        for reference in REFERENCES
    }
    return {
        "known_38_resolved": slices["fixture"].get("all", {}).get("count") == 38
        and minimum_accuracy(slices["fixture"], ["all"], 1),
        "complete_and_eligible": minimum_accuracy(
            slices["conditions"], keys(("complete", "eligible")), 0.98
        ),
        "exposed_preconditions": minimum_accuracy(
            slices["conditions"], keys(("gate_false", "missing")), 0.95
        ),
        "new_preconditions": minimum_accuracy(
            slices["development"], keys(("gate_false", "missing"), "transfer/"), 0.95
        ),
        "new_complete_and_eligible_retained": all(
            key in parent
            and parent[key]["count"] > 0
            and minimum_accuracy(slices["development"], [key], parent[key]["accuracy"] - 0.02)
            for key in keys(("complete", "eligible"), "transfer/")
        ),
        "all_original_families_retained": set(comparisons) == required
        and all(
            len(result["families"]) == 8
            and all(
                values["local"]["accuracy"] >= values["reference"]["accuracy"] - 0.02 - 1e-12
                for values in result["families"].values()
            )
            for result in retention
        ),
        "original_task_macro_retained": all(
            key in comparisons and comparisons[key]["macro_accuracy"]["delta"] >= -0.005 - 1e-12
            for key in ("retention-repaired", "regression-repaired")
        ),
        "no_material_language_regression": set(alerts) == required and not any(alerts.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("bundle", "verify", "profile", "baselines", "train", "assess")
    )
    parser.add_argument("recipe", nargs="?", choices=("primary", "lower-rate"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/necro-unified-cloud-2026-09-21")
    )
    args = parser.parse_args()
    if args.action == "bundle":
        build_bundle(args.output)
        return
    verify_bundle()
    plan = read(Path("cloud-plan.json"))
    verify(DATA, "train")
    if args.action == "verify":
        print(
            json.dumps(
                {
                    "verified": True,
                    "training_rows": read(DATA / "audit.json")["data"]["train"]["rows"],
                    "test_opened": False,
                }
            )
        )
        return
    os.environ.update(NECRO_MODEL=BASE, NECRO_ADAPTER="", NECRO_DEVICE="cuda")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.action == "profile":
        profile(plan)
    elif args.action == "baselines":
        baselines()
    elif args.action == "train":
        if (DATA / "selection.json").exists():
            raise ValueError("Candidate is frozen; no further training in this experiment")
        if args.recipe is None:
            parser.error("train requires a recipe")
        if args.recipe == "lower-rate" and not (OUTPUT / "primary/assessment.json").exists():
            raise ValueError("Assess the primary recipe before starting the second recipe")
        performance = read(OUTPUT / "performance.json")
        if performance["data_manifest_sha256"] != digest(DATA / "experiment.json"):
            raise ValueError("GPU profile does not match the frozen dataset")
        train(
            DATA,
            OUTPUT / args.recipe,
            batch_size=performance["batch_size"],
            accumulation=performance["accumulation"],
            learning_rate=plan["recipes"][args.recipe]["learning_rate"],
            rank=plan["rank"],
            seed=plan["seed"],
            epochs=plan["epochs"],
            expected_revision=plan["revision"],
            model_id=f"necro-unified-{args.recipe}",
        )
    elif args.action == "assess":
        if args.recipe is None:
            parser.error("assess requires a recipe")
        assess_run(args.recipe)


if __name__ == "__main__":
    main()
