import json
from pathlib import Path

import pytest

from necro.experiment_guard import audit_partitions, register, verify
from necro.training_data import write_jsonl


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
