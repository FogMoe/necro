# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ScarletKc-Necro contributors

import copy
import json
import string
from types import SimpleNamespace

import pytest

from necro.backend import TransformersScorer
from necro.config import Settings
from necro.engine import prompt_fingerprint
from necro.training.data.training_data import audit_disjoint, rule_pairs
from necro.training.trainer import answer_loss, collate, encode_example


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages, ensure_ascii=False) + "answer:"


def test_correct_choice_only_changes_target_and_teacher_forced_suffix():
    row = {
        "id": "test",
        "request": {
            "model": "necro-qwen3.5-0.8b",
            "state": "example",
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": "Select an item.",
                    "criteria": {f"item{i}": str(i) for i in range(60)},
                }
            },
        },
        "expected": {"decision": "item0"},
    }
    tokenizer = CharacterTokenizer()
    first = encode_example(tokenizer, string.ascii_uppercase, row, max_length=10000)
    row["expected"]["decision"] = "item59"
    last = encode_example(tokenizer, string.ascii_uppercase, row, max_length=10000)
    assert first["input_ids"][:-1] == last["input_ids"][:-1]
    assert first["target_ids"] == list(map(ord, "10"))
    assert last["target_ids"] == list(map(ord, "69"))
    assert last["input_ids"][-1] == ord("6")
    with pytest.raises(ValueError, match="不进行截断"):
        encode_example(tokenizer, string.ascii_uppercase, row, max_length=5)


def test_mixed_answer_lengths_are_aligned_and_only_answers_have_loss():
    torch = pytest.importorskip("torch")
    rows = [
        {"input_ids": [1, 2, 3], "target_ids": [4, 5]},
        {"input_ids": [6, 7], "target_ids": [8]},
    ]
    inputs, targets = collate(rows, 0, "cpu")
    assert inputs["attention_mask"].tolist() == [[1, 1, 1], [0, 1, 1]]
    assert targets.tolist() == [[4, 5], [-100, 8]]
    logits = torch.zeros(2, 2, 10, requires_grad=True)

    def forward(**kwargs):
        assert kwargs["logits_to_keep"] == 2
        assert kwargs["use_cache"] is False
        return SimpleNamespace(logits=logits)

    loss = answer_loss(forward, inputs, targets)
    assert float(loss.detach()) == pytest.approx(1.5 * torch.log(torch.tensor(10.0)).item())
    loss.backward()
    assert torch.count_nonzero(logits.grad[1, 0]) == 0
    assert torch.count_nonzero(logits.grad[0]) > 0


def test_candidate_loss_ignores_non_candidates_and_keeps_multi_token_supervision():
    torch = pytest.importorskip("torch")
    logits = torch.zeros(2, 2, 10, requires_grad=True)
    targets = torch.tensor([[-100, 3], [2, 4]])

    def forward(**kwargs):
        return SimpleNamespace(logits=logits)

    loss = answer_loss(forward, {}, targets, [[3, 7], None])
    expected = (torch.log(torch.tensor(2.0)) + 2 * torch.log(torch.tensor(10.0))) / 2
    assert float(loss.detach()) == pytest.approx(float(expected))
    loss.backward()
    assert logits.grad[0, -1, 3] < 0
    assert logits.grad[0, -1, 7] > 0
    assert torch.count_nonzero(logits.grad[0, -1, [0, 1, 2, 4, 5, 6, 8, 9]]) == 0
    assert torch.count_nonzero(logits.grad[1, 0]) == 10
    assert torch.count_nonzero(logits.grad[1, 1]) == 10


def test_split_audit_catches_translated_groups_and_identical_contexts():
    first = rule_pairs()[0]
    translated = copy.deepcopy(first)
    translated["request"]["state"] = "translated example"
    with pytest.raises(ValueError, match="来源组"):
        audit_disjoint([first], [translated])
    duplicated = copy.deepcopy(first)
    duplicated["group_id"] = "different-source"
    with pytest.raises(ValueError, match="上下文"):
        audit_disjoint([first], [duplicated])


@pytest.mark.parametrize("candidate_ids", [None, [[0, 1], [0, 1], None]])
def test_weighted_loss_preserves_gradients_across_unequal_microbatches(candidate_ids):
    torch = pytest.importorskip("torch")
    targets = torch.tensor([[0], [1], [2]])
    weights = [2.0, 0.5, 0.5]
    full = torch.tensor(
        [[[0.3, 0.2, 0.1]], [[0.2, 0.7, 0.4]], [[0.9, 0.8, 0.1]]], requires_grad=True
    )
    loss = answer_loss(
        lambda **kw: SimpleNamespace(logits=full), {}, targets, candidate_ids, weights
    )
    loss.backward()
    plain = full.detach().clone().requires_grad_()
    answer_loss(lambda **kw: SimpleNamespace(logits=plain), {}, targets, candidate_ids).backward()
    torch.testing.assert_close(full.grad, plain.grad * torch.tensor(weights).view(-1, 1, 1))
    split = full.detach().clone().requires_grad_()
    for start, end in ((0, 1), (1, 3)):
        part = answer_loss(
            lambda start=start, end=end, **kw: SimpleNamespace(logits=split[start:end]),
            {},
            targets[start:end],
            candidate_ids[start:end] if candidate_ids else None,
            weights[start:end],
        )
        (part * (end - start) / 3).backward()
    torch.testing.assert_close(split.grad, full.grad)


@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf"), True, "1"])
def test_invalid_training_weight_rejected_before_encoding(weight):
    with pytest.raises(ValueError, match="training_weight"):
        encode_example(None, None, {"id": "bad", "training_weight": weight})


def test_adapter_rejects_wrong_prompt_before_loading_weights(tmp_path):
    (tmp_path / "necro_adapter.json").write_text(
        json.dumps({"checkpoint": Settings().checkpoint, "prompt_sha256": "wrong"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="提示代码不一致"):
        TransformersScorer(Settings(adapter=str(tmp_path)))


def test_merged_model_uses_manifest_identity_and_rejects_second_adapter(tmp_path):
    (tmp_path / "necro_model.json").write_text(
        json.dumps({"prompt_sha256": prompt_fingerprint(), "model_id": "necro-export-test"}),
        encoding="utf-8",
    )
    assert TransformersScorer(Settings(checkpoint=str(tmp_path))).model_id == "necro-export-test"
    with pytest.raises(ValueError, match="再次叠加"):
        TransformersScorer(Settings(checkpoint=str(tmp_path), adapter="another-adapter"))
