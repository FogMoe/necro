"""第三阶段选择冻结；同时固定旧模型对照及各自的独立校准。"""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from necro.engine import prompt_fingerprint
from necro.evaluation import read_examples
from necro.experiment_guard import canonical_request, digest, verify
from necro.training.data.full_snapshots import rows as raw_rows
from necro.training.data.phase3_data import SourceIndex, historical


def reject_exposed_tests(hashes, results=Path("results")):
    for path in results.rglob("summary.json"):
        metadata = json.loads(path.read_text(encoding="utf-8")).get("metadata", {})
        if metadata.get("dataset_sha256") in hashes:
            raise ValueError(f"发现选择冻结前的最终预测，不能将此测试当作盲测：{path}")


def lineage(adapter, datasets):
    records, training, visited = [], [], set()
    while adapter:
        identity = str(adapter.resolve())
        if identity in visited:
            raise ValueError("训练祖先出现循环。")
        visited.add(identity)
        contract = json.loads((adapter / "necro_adapter.json").read_text())
        source = datasets[contract["train_sha256"]]
        rows = read_examples(source)
        if len(rows) != contract["examples"]:
            raise ValueError("祖先训练量与真实数据不符。")
        training.extend(rows)
        records.append(
            {
                "adapter": str(adapter),
                "weights_sha256": digest(adapter / "adapter_model.safetensors"),
                "data": str(source),
                "data_sha256": digest(source),
                "presentations": len(rows),
                "seed": contract["seed"],
            }
        )
        parent = contract.get("initial_adapter")
        if (
            parent
            and digest(Path(parent) / "adapter_model.safetensors")
            != contract["initial_adapter_sha256"]
        ):
            raise ValueError("祖先权重与训练时记录不符。")
        adapter = Path(parent) if parent else None
    return records, training


def freeze(
    data, robustness, adapter, calibration, reference, reference_calibration, output, decision
):
    roots = (data, robustness)
    if output.exists() or any((root / "selection.json").exists() for root in roots):
        raise ValueError("第三阶段已经冻结，不可覆盖。")
    test_hashes = {digest(root / "test.jsonl") for root in roots}
    reject_exposed_tests(test_hashes)
    robust = json.loads((robustness / "experiment.json").read_text())
    if robust["protocol"]["parent_test_sha256"] != digest(data / "test.jsonl"):
        raise ValueError("稳健性数据与最终测试不对应。")
    calibration_hash = digest(verify(data, "calibration"))
    fits = [json.loads(p.read_text()) for p in (calibration, reference_calibration)]
    for fit, fit_path, weights_path in zip(
        fits, (calibration, reference_calibration), (adapter, reference), strict=True
    ):
        if fit["dataset_sha256"] != calibration_hash or set(fit["temperatures"]) != {
            "choice",
            "noul",
            "score",
        }:
            raise ValueError("校准不是登记的独立校准集或缺少判断类型。")
        raw_predictions = fit_path.parent / "calibration/predictions.jsonl"
        raw_metadata = json.loads((raw_predictions.parent / "summary.json").read_text())["metadata"]
        if (
            digest(raw_predictions) != fit["predictions_sha256"]
            or raw_metadata.get("adapter_weights_sha256")
            != digest(weights_path / "adapter_model.safetensors")
            or raw_metadata["dataset_sha256"] != calibration_hash
            or not raw_metadata.get("full_dataset_evaluated")
        ):
            raise ValueError("校准预测与指定权重或完整校准集不匹配。")
    evidence = json.loads(decision.read_text(encoding="utf-8"))
    if evidence.get("new_test_exposed") is not False or not evidence.get("rationale"):
        raise ValueError("缺少明确的开发阶段选模依据。")
    datasets = {digest(path): path for path in Path("data").rglob("train.jsonl")}
    selected_lineage, selected_training = lineage(adapter, datasets)
    reference_lineage, reference_training = lineage(reference, datasets)
    past, _ = historical()
    holdouts = {role: read_examples(verify(data, role)) for role in ("development", "calibration")}
    holdouts["test"] = read_examples(data / "test.jsonl")
    english_paws = {
        (split, str(r["id"])): r
        for split in ("train", "validation", "test")
        for _, r in raw_rows("paws", "en", split)
    }
    index = SourceIndex(
        [
            *past,
            *selected_training,
            *reference_training,
            *(row for rows in holdouts.values() for row in rows),
        ],
        english_paws=english_paws,
    )
    audits = {}
    for name, records, training in (
        ("selected", selected_lineage, selected_training),
        ("reference", reference_lineage, reference_training),
    ):
        keys = index.all_keys(training)
        requests = {canonical_request(row) for row in training}
        for role, rows in holdouts.items():
            if keys & index.all_keys(rows) or requests & {canonical_request(row) for row in rows}:
                raise ValueError(f"{name} 完整祖先链与 {role} 重叠。")
        audits[name] = {
            "lineage": records,
            "training_presentations": len(training),
            "unique_training_requests": len(requests),
            "source_overlap_with_holdouts": 0,
        }
    output.mkdir(parents=True)
    selected_adapter = output / "adapter"
    shutil.copytree(adapter, selected_adapter)
    shutil.copy2(calibration, output / "calibration.json")
    shutil.copy2(reference_calibration, output / "reference-calibration.json")
    shutil.copy2(decision, output / "decision.json")
    contract = json.loads((selected_adapter / "necro_adapter.json").read_text())
    contract.setdefault("training_model_id", contract["model_id"])
    contract["model_id"] = "ScarletKc-Necro-0.8b"
    contract["recommended_temperatures"] = fits[0]["temperatures"]
    (selected_adapter / "necro_adapter.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    (output / "lineage-audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    frozen = output / "frozen-materials"
    shutil.copytree(
        "src/necro", frozen / "necro", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    build = json.loads((data / "build-provenance.json").read_text())
    for source, expected in build["source_hashes"].items():
        if digest(data / "build-code" / Path(source).name) != expected:
            raise ValueError("原始数据生成源码快照与登记哈希不符。")
    shutil.copytree(data / "build-code", frozen / "original-data-builder")
    for source in (
        Path("docs/process/phase2-acceptance.md"),
        Path("docs/process/phase3-coverage-audit.md"),
        *sorted(Path("results/phase3").glob("*plan.json")),
        data / "build-provenance.json",
        Path("pyproject.toml"),
        Path("uv.lock"),
    ):
        shutil.copy2(source, frozen / source.name)
    weights = digest(selected_adapter / "adapter_model.safetensors")
    reference_weights = digest(reference / "adapter_model.safetensors")
    if weights == reference_weights and fits[0]["temperatures"] != fits[1]["temperatures"]:
        raise ValueError("同一份权重的冻结校准不一致。")
    hashes = {str(path): digest(path) for path in output.rglob("*") if path.is_file()}
    for record in [*selected_lineage, *reference_lineage]:
        hashes[record["data"]] = record["data_sha256"]
    hashes[str(reference / "necro_adapter.json")] = digest(reference / "necro_adapter.json")
    hashes[str(reference / "adapter_model.safetensors")] = reference_weights
    selection = {
        "version": 3,
        "frozen_at": datetime.now(UTC).isoformat(),
        "model_id": contract["model_id"],
        "adapter": str(selected_adapter),
        "weights_sha256": weights,
        "prompt_sha256": prompt_fingerprint(),
        "temperatures": fits[0]["temperatures"],
        "reference_adapter": str(reference),
        "reference_weights_sha256": [reference_weights],
        "reference_temperatures": {reference_weights: fits[1]["temperatures"]},
        "artifact_hashes": hashes,
        "reference": "jev-1.13.0",
        "final_predictions_available_at_selection": False,
        "decision": evidence,
    }
    for root in roots:
        (root / "selection.json").write_text(
            json.dumps(
                {**selection, "data_manifest_sha256": digest(root / "experiment.json")}, indent=2
            ),
            encoding="utf-8",
        )
        verify(root, "test", selected_adapter)
        verify(root, "test", reference)
    (output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected": str(selected_adapter),
                "weights_sha256": weights,
                "reference_weights_sha256": reference_weights,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in (
        "data",
        "robustness",
        "adapter",
        "calibration",
        "reference",
        "reference-calibration",
        "output",
        "decision",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    freeze(
        a.data,
        a.robustness,
        a.adapter,
        a.calibration,
        a.reference,
        a.reference_calibration,
        a.output,
        a.decision,
    )
