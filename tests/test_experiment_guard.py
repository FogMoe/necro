import json
from pathlib import Path

import pytest

from necro.experiment_guard import audit_partitions, digest, register, verify, verify_temperatures
from necro.training.data.training_data import write_jsonl


def row(identifier, group, language="en"):
    return {
        "id": identifier,
        "group_id": group,
        "source": "source",
        "request": {
            "model": "necro-qwen3.5-0.8b",
            "state": language,
            "questions": {"q": {"type": "noul", "instructions": "Is it true?"}},
        },
        "expected": {"q": True},
    }


def test_translations_must_stay_in_one_partition():
    with pytest.raises(ValueError, match="来源组"):
        audit_partitions({"train": [row("a", "same", "en")], "test": [row("b", "same", "zh")]})


def test_prediction_partitions_reject_reused_ids_with_different_requests():
    with pytest.raises(ValueError, match="重复样本 ID"):
        audit_partitions({"test": [row("same-id", "a", "a"), row("same-id", "b", "b")]})


def test_record_id_cannot_disguise_duplicate_rule():
    a, b = row("a", "a"), row("b", "b")
    for record, identifier in ((a, "001"), (b, "999")):
        record["source"] = "constructed-rules"
        record["request"]["state"] = {"record_id": identifier, "enabled": True}
    with pytest.raises(ValueError, match="请求跨划分|内容跨划分"):
        audit_partitions({"train": [a], "test": [b]})


def test_modified_data_and_unfrozen_test_are_rejected(tmp_path):
    train, test = tmp_path / "train.jsonl", tmp_path / "test.jsonl"
    write_jsonl(train, [row("a", "a", "train context")])
    write_jsonl(test, [row("b", "b", "test context")])
    register(tmp_path, {"train": train, "test": test}, {})
    with pytest.raises(ValueError, match="尚未冻结"):
        verify(tmp_path, "test", Path("unused"))
    assert verify(tmp_path, "train") == train
    train.write_text(json.dumps(row("c", "c")), encoding="utf-8")
    with pytest.raises(ValueError, match="发生变更"):
        verify(tmp_path, "train")


def test_frozen_calibration_and_protocol_cannot_be_changed(tmp_path):
    test = tmp_path / "test.jsonl"
    write_jsonl(test, [row("a", "a")])
    register(tmp_path, {"test": test}, {})
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture-weights")
    calibration = tmp_path / "calibration.json"
    calibration.write_text('{"temperature": 1}')
    selected = {
        "weights_sha256": digest(adapter / "adapter_model.safetensors"),
        "data_manifest_sha256": digest(tmp_path / "experiment.json"),
        "artifact_hashes": {str(calibration): digest(calibration)},
    }
    (tmp_path / "selection.json").write_text(json.dumps(selected))
    assert verify(tmp_path, "test", adapter) == test
    calibration.write_text('{"temperature": 2}')
    with pytest.raises(ValueError, match="校准或协议材料"):
        verify(tmp_path, "test", adapter)
    calibration.write_text('{"temperature": 1}')
    registry = tmp_path / "experiment.json"
    manifest = json.loads(registry.read_text())
    manifest["protocol"]["changed"] = True
    registry.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="实验登记"):
        verify(tmp_path, "test", adapter)


def test_frozen_reference_uses_its_own_calibration():
    selected_t = dict(choice=2.0, noul=4.0, score=1.5)
    reference_t = dict(choice=3.0, noul=5.0, score=2.5)
    selection = {
        "weights_sha256": "chosen",
        "temperatures": selected_t,
        "reference_temperatures": {"baseline": reference_t},
    }
    verify_temperatures(selection, "chosen", selected_t)
    verify_temperatures(selection, "baseline", reference_t)
    verify_temperatures(selection, "baseline", dict(choice=1, noul=1, score=1))
    with pytest.raises(ValueError, match="该权重冻结"):
        verify_temperatures(selection, "baseline", selected_t)
    with pytest.raises(ValueError, match="该权重冻结"):
        verify_temperatures(selection, "unregistered", reference_t)
