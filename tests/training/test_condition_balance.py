import copy

import pytest

from necro.experiment_guard import audit_partitions, canonical_request
from necro.training.data.condition_balance import (
    audit_balance,
    audit_sampling,
    balance,
    resample,
    semantic_outcome,
    transfer_rows,
)
from necro.training.data.condition_refinement import complete_condition_matrix


def test_balance_changes_exposure_without_changing_requests_labels_or_order():
    original = complete_condition_matrix()
    weighted = balance(original)
    assert [canonical_request(r) for r in weighted] == [canonical_request(r) for r in original]
    assert [r["expected"] for r in weighted] == [r["expected"] for r in original]
    assert [r["id"] for r in weighted] == [r["id"] for r in original]
    audit = audit_balance(weighted)
    assert audit["semantic_presentations"] == {"True": 320, "False": 2880}
    assert audit["semantic_weight"] == pytest.approx({"True": 1600, "False": 1600})
    assert audit["answer_weight"] == pytest.approx({"True": 1600, "False": 1600})
    assert all(r.get("training_weight") is None for r in original)


def test_semantic_audit_catches_bias_hidden_by_answer_balance():
    with pytest.raises(ValueError, match="Eligibility"):
        audit_balance(complete_condition_matrix())
    weighted = balance(complete_condition_matrix())
    changed = copy.deepcopy(weighted)
    for row in changed:
        if row["language"] == "zh":
            row["training_weight"] *= 1.2 if semantic_outcome(row) else 0.8
    with pytest.raises(ValueError, match="Eligibility"):
        audit_balance(changed)


def test_new_transfer_partitions_are_disjoint_and_have_both_outcomes():
    parts = {role: transfer_rows(role) for role in ("development", "test")}
    audit_partitions(parts)
    for rows in parts.values():
        assert {semantic_outcome(r) for r in rows} == {False, True}
        assert len({r["condition_matrix_case"] for r in rows}) == 5
        assert {r["language"] for r in rows} == {"en", "zh"}


def test_sampling_balances_presentations_and_keeps_all_failure_combinations():
    original = complete_condition_matrix()
    sampled = resample(original)
    audit = audit_sampling(sampled)
    assert len(sampled) == len(original)
    assert len({r["id"] for r in sampled}) == len(sampled)
    assert audit["semantic_presentations"] == {"True": 1600, "False": 1600}
    assert audit["answer_presentations"] == {"True": 1600, "False": 1600}
    assert audit["combinations"] == 800
    assert audit["minimum_presentations_per_combination"] >= 2
    assert {canonical_request(r) for r in sampled} <= {canonical_request(r) for r in original}
    assert sampled == resample(original)
