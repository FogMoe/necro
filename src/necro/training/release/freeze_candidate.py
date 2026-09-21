"""Freeze a release candidate after its registered development checks pass."""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from necro.engine import prompt_fingerprint
from necro.evaluation import read_examples
from necro.experiment_guard import canonical_request, digest, verify
from necro.training.data.full_snapshots import rows as raw_rows
from necro.training.data.phase3_data import SourceIndex
from necro.training.release.freeze_phase3 import lineage, reject_exposed_tests


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def freeze(
    data,
    run,
    output,
    decision_path=None,
    *,
    reference=Path("results/phase2/v3/selected/adapter"),
    reference_calibration=Path("results/phase3/baseline/calibration-brier-fit.json"),
    plan_files=None,
):
    if output.exists() or (data / "selection.json").exists():
        raise ValueError("Candidate selection already exists")
    preliminary = run / "repair-assessment.json"
    decision_path = decision_path or preliminary
    review = read(decision_path)
    if decision_path != preliminary and (
        review.get("supersedes_review_sha256") != digest(preliminary)
        or review.get("original_checks") != read(preliminary)["checks"]
        or not review.get("accepted_development_tradeoff")
    ):
        raise ValueError(
            "Amended selection must preserve the original checks and explain its tradeoff"
        )
    if not review.get("eligible") or not all(review["checks"].values()):
        raise ValueError("Registered development checks did not pass")
    reject_exposed_tests({digest(data / "test.jsonl")})
    for role in ("train", "development", "calibration"):
        verify(data, role)
    if digest(data / "builder.py") != read(data / "experiment.json")["protocol"]["builder_sha256"]:
        raise ValueError("The archived builder does not match registration")
    adapter = run / "adapter"
    fit = read(run / "calibration-brier-fit.json")
    raw = read(run / "calibration/summary.json")["metadata"]
    weights = digest(adapter / "adapter_model.safetensors")
    if review.get("adapter_weights_sha256") != weights or review.get(
        "data_manifest_sha256"
    ) != digest(data / "experiment.json"):
        raise ValueError("Development checks do not match these weights and data")
    if (
        fit["dataset_sha256"] != digest(data / "calibration.jsonl")
        or fit["predictions_sha256"] != digest(run / "calibration/predictions.jsonl")
        or raw.get("adapter_weights_sha256") != weights
        or not raw.get("full_dataset_evaluated")
        or any(raw.get(f"{p}_temperature") != 1 for p in ("choice", "noul", "score"))
    ):
        raise ValueError("Calibration evidence does not match the candidate")
    predecessor = reference
    reference_fit_path = reference_calibration
    reference_fit = read(reference_fit_path)
    reference_metadata = read(reference_fit_path.parent / "calibration/summary.json")["metadata"]
    if (
        reference_fit["dataset_sha256"] != digest(Path("data/phase3/coverage-v1/calibration.jsonl"))
        or reference_fit["predictions_sha256"]
        != digest(reference_fit_path.parent / "calibration/predictions.jsonl")
        or reference_metadata.get("adapter_weights_sha256")
        != digest(predecessor / "adapter_model.safetensors")
        or not reference_metadata.get("full_dataset_evaluated")
    ):
        raise ValueError("Reference calibration does not match the predecessor")
    old_calibration = read_examples(Path("data/phase3/coverage-v1/calibration.jsonl"))
    current_calibration = read_examples(data / "calibration.jsonl")

    def cohort(rows):
        return {r["id"]: (canonical_request(r), r["expected"]) for r in rows}

    if cohort(old_calibration) != cohort(current_calibration):
        raise ValueError("Reference calibration must contain the same requests and labels")
    datasets = {digest(path): path for path in Path("data").rglob("train.jsonl")}
    records, training = lineage(adapter, datasets)
    reference_records, reference_training = lineage(predecessor, datasets)
    holdouts = {
        role: read_examples(data / filename)
        for role, filename in (
            ("development", "validation.jsonl"),
            ("calibration", "calibration.jsonl"),
            ("test", "test.jsonl"),
        )
    }
    english = {
        (split, str(r["id"])): r
        for split in ("train", "validation", "test")
        for _, r in raw_rows("paws", "en", split)
    }
    index = SourceIndex(
        [*training, *reference_training, *(r for rows in holdouts.values() for r in rows)],
        english_paws=english,
    )
    for source in (training, reference_training):
        if any(index.all_keys(source) & index.all_keys(rows) for rows in holdouts.values()):
            raise ValueError("An ancestor overlaps a held-out source")
    output.mkdir(parents=True)
    selected = output / "adapter"
    shutil.copytree(adapter, selected)
    contract = read(selected / "necro_adapter.json")
    contract["training_model_id"] = contract["model_id"]
    contract["model_id"] = "ScarletKc-Necro-0.8b"
    contract["recommended_temperatures"] = fit["temperatures"]
    (selected / "necro_adapter.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    if plan_files is None:
        plan_files = [
            (Path("results/phase4/condition-repair-plan.json"), "repair-plan.json"),
            (Path("results/phase4/condition-followup-plan.json"), "followup-plan.json"),
            (Path("results/phase4/condition-matrix-plan.json"), "matrix-plan.json"),
            (Path("results/phase4/condition-retention-plan.json"), "retention-plan.json"),
            (Path("results/phase4/final-validation-plan.json"), "validation-plan.json"),
        ]
    for source, name in (
        (run / "calibration-brier-fit.json", "calibration.json"),
        (reference_fit_path, "reference-calibration.json"),
        (decision_path, "decision.json"),
        (preliminary, "preliminary-review.json"),
        (data / "experiment.json", "data-manifest.json"),
        (data / "builder.py", "builder.py"),
        *plan_files,
        (Path("docs/process/release-criteria-2026-09-21.md"), "release-criteria.md"),
    ):
        shutil.copy2(source, output / name)
    audit = {
        "selected": {
            "lineage": records,
            "training_presentations": len(training),
            "unique_training_requests": len({canonical_request(r) for r in training}),
        },
        "reference": {
            "lineage": reference_records,
            "training_presentations": len(reference_training),
        },
        "source_overlap_with_holdouts": 0,
        "reference_calibration": "Same requests and labels, reordered in the new data directory",
    }
    (output / "lineage-audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    shutil.copytree(
        "src/necro", output / "source/necro", ignore=shutil.ignore_patterns("__pycache__")
    )
    reference_weights = digest(predecessor / "adapter_model.safetensors")
    hashes = {str(p): digest(p) for p in output.rglob("*") if p.is_file()}
    hashes.update({r["data"]: r["data_sha256"] for r in [*records, *reference_records]})
    hashes[str(predecessor / "adapter_model.safetensors")] = reference_weights
    selection = {
        "model_id": contract["model_id"],
        "frozen_at": datetime.now(UTC).isoformat(),
        "adapter": str(selected),
        "weights_sha256": weights,
        "temperatures": fit["temperatures"],
        "reference": "jev-1.13.0",
        "reference_adapter": str(predecessor),
        "reference_weights_sha256": [reference_weights],
        "reference_temperatures": {reference_weights: reference_fit["temperatures"]},
        "prompt_sha256": prompt_fingerprint(),
        "data_manifest_sha256": digest(data / "experiment.json"),
        "artifact_hashes": hashes,
        "final_predictions_available_at_selection": False,
    }
    for path in (data / "selection.json", output / "selection.json"):
        path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    verify(data, "test", selected)
    print(json.dumps({"selected": str(selected), "weights_sha256": weights}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--decision", type=Path)
    args = parser.parse_args()
    freeze(args.data, args.run, args.output, args.decision)
