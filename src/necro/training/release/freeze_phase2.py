"""第二阶段模型选择封存；只读取开发记录，不读取最终预测。"""

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from necro.engine import prompt_fingerprint
from necro.evaluation import read_examples
from necro.experiment_guard import canonical_request, digest, verify
from necro.training.data.source_isolation import Sources


def freeze():
    results = Path("results/phase2/v3")
    strict = Path("data/phase2/strict-v1")
    robustness = Path("data/phase2/robustness-v5")
    output = results / "selected"
    if output.exists() or any((root / "selection.json").exists() for root in (strict, robustness)):
        raise ValueError("选择已经封存，不可覆盖。")
    manifest = json.loads((robustness / "experiment.json").read_text(encoding="utf-8"))
    if manifest["protocol"]["parent_test_sha256"] != digest(strict / "test.jsonl"):
        raise ValueError("稳健性题与最终题不对应。")
    dataset_paths = [
        Path(path)
        for path in (
            "data/lora-pilot/train.jsonl",
            "data/improvement/expanded/train.jsonl",
            "data/phase2/experiment-v3/train.jsonl",
            "data/phase2/refinement-v2/train.jsonl",
            "data/phase2/operator-contrast-v1/train.jsonl",
        )
    ]
    datasets = {digest(path): path for path in dataset_paths}
    adapter = results / "operator-contrast/adapter"
    lineage, training = [], []
    current = adapter
    while current:
        contract = json.loads((current / "necro_adapter.json").read_text(encoding="utf-8"))
        path = datasets[contract["train_sha256"]]
        rows = read_examples(path)
        if len(rows) != contract["examples"]:
            raise ValueError("祖先训练量与数据不符。")
        training.extend(rows)
        lineage.append(
            {
                "adapter": str(current),
                "model_id": contract["model_id"],
                "weights_sha256": digest(current / "adapter_model.safetensors"),
                "training_file": str(path),
                "train_sha256": digest(path),
                "presentations": len(rows),
                "seed": contract["seed"],
            }
        )
        parent = contract.get("initial_adapter")
        current = Path(parent) if parent else None
    protected = {
        role: read_examples(strict / f"{role}.jsonl")
        for role in ("development", "calibration", "test")
    }
    sources = Sources(
        [
            *training,
            *(row for rows in protected.values() for row in rows),
            *read_examples(Path("data/improvement/source-expanded.jsonl")),
        ]
    )
    keys = sources.all_keys(training)
    requests = {canonical_request(row) for row in training}
    for role, rows in protected.items():
        if keys & sources.all_keys(rows) or requests & {canonical_request(row) for row in rows}:
            raise ValueError(f"完整训练祖先链与 {role} 重叠。")
    output.mkdir(parents=True)
    copied = output / "adapter"
    shutil.copytree(adapter, copied)
    calibration = output / "calibration.json"
    shutil.copy2(results / "operator-contrast/calibration-wide.json", calibration)
    temperatures = json.loads(calibration.read_text(encoding="utf-8"))["temperatures"]
    contract = json.loads((copied / "necro_adapter.json").read_text(encoding="utf-8"))
    contract["training_model_id"] = contract["model_id"]
    contract["model_id"] = "ScarletKc-Necro-0.8b"
    contract["recommended_temperatures"] = temperatures
    (copied / "necro_adapter.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    frozen = output / "frozen-materials"
    shutil.copytree(
        "src/necro", frozen / "necro", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for source in (
        Path("docs/process/phase2-acceptance.md"),
        Path("docs/process/phase2-source-audit.md"),
        results / "ablation-plan.json",
        results / "calibration-amendment.json",
        results / "operator-plan.json",
        results / "seed-replication-plan.json",
        results / "reading-plan.json",
    ):
        shutil.copy2(source, frozen / source.name)
    audit = {
        "lineage": lineage,
        "training_presentations": len(training),
        "unique_training_requests": len(requests),
        "training_source_keys": len(keys),
        "source_overlap_with_development_calibration_test": 0,
        "notes": "Whole ancestor training chain checked against development, calibration "
        "and test sources.",
    }
    (output / "lineage-audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    hashes = {str(path): digest(path) for path in frozen.rglob("*") if path.is_file()}
    for path in [
        calibration,
        copied / "necro_adapter.json",
        output / "lineage-audit.json",
        *dataset_paths,
    ]:
        hashes[str(path)] = digest(path)
    selection = {
        "version": 2,
        "frozen_at": datetime.now(UTC).isoformat(),
        "model_id": contract["model_id"],
        "adapter": str(copied),
        "weights_sha256": digest(copied / "adapter_model.safetensors"),
        "prompt_sha256": prompt_fingerprint(),
        "temperatures": temperatures,
        "artifact_hashes": hashes,
        "reference": "jev-1.13.0",
        "final_predictions_available_at_selection": False,
        "decision": "Select operator seed 2026 (highest task-macro development accuracy, 89.13%). "
        "Matched seed 2027 reproduces 89.10%. Parameter averaging reaches 88.81%, with better "
        "Brier but lower accuracy. Added 1000 reading questions reach 89.09%, only one more "
        "reading answer correct, with numeric/ordinal regression. Stop further lightweight "
        "training for diminishing returns; final performance gates remain unchanged.",
    }
    for root in (strict, robustness):
        (root / "selection.json").write_text(
            json.dumps(
                {
                    **selection,
                    "data_manifest_sha256": digest(root / "experiment.json"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        verify(root, "test", copied)
    (results / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected": str(copied),
                "weights_sha256": selection["weights_sha256"],
                "temperatures": temperatures,
                "training_presentations": len(training),
                "unique_training_requests": len(requests),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    freeze()
