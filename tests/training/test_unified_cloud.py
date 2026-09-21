import copy
from collections import Counter

import pytest

from necro.diagnostics.numeric_regression import rule_metadata, write_json
from necro.experiment_guard import digest
from necro.training.cloud import acceptance_checks, verify_bundle
from necro.training.data.condition_balance import semantic_outcome
from necro.training.data.unified import condition_training, deduplicate
from necro.training.trainer import epoch_batches, make_batches


def test_unified_matrix_balances_complete_outcomes_and_prerequisites():
    rows = condition_training()
    assert len(rows) == len(deduplicate(rows)) == 2400
    assert Counter(r["semantic_category"] for r in rows) == {
        "eligible": 800,
        "comparison_failure": 800,
        "precondition_failure": 800,
    }
    complete = [r for r in rows if rule_metadata(r)["case"] == "complete"]
    assert Counter(semantic_outcome(r) for r in complete) == {True: 800, False: 800}
    assert {rule_metadata(r)["case"] for r in rows} == {"complete", "gate_false", "missing"}
    assert len({r["group_id"] for r in rows}) == 200
    assert Counter(r["expected"]["decision"] for r in rows) == {True: 1200, False: 1200}
    conflict = copy.deepcopy(rows[0])
    conflict["expected"]["decision"] = not conflict["expected"]["decision"]
    with pytest.raises(ValueError, match="Conflicting labels"):
        deduplicate([rows[0], conflict])


def test_epoch_batches_preserve_each_record_and_seeded_order():
    rows = [{"id": i, "input_ids": list(range(i % 12 + 1))} for i in range(131)]
    first = make_batches(rows, 8, 2026)
    assert epoch_batches(rows, 8, 2026, 1) == first
    batches = epoch_batches(rows, 8, 2026, 2)
    assert batches[: len(first)] == first
    assert batches[len(first) :] == make_batches(rows, 8, 2027)
    assert Counter(r["id"] for batch in batches for r in batch) == dict.fromkeys(range(131), 2)
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            epoch_batches(rows, 8, 2026, invalid)


def test_bundle_detects_corruption_and_path_escape(tmp_path):
    payload = tmp_path / "payload.txt"
    payload.write_text("frozen", encoding="utf-8")
    write_json(tmp_path / "bundle.json", {"files": {"payload.txt": digest(payload)}})
    verify_bundle(tmp_path)
    payload.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        verify_bundle(tmp_path)
    write_json(tmp_path / "bundle.json", {"files": {"../outside": "bad"}})
    with pytest.raises(ValueError, match="escapes"):
        verify_bundle(tmp_path)


def test_acceptance_cannot_hide_missing_slices_or_family_regression():
    valid = {"count": 100, "accuracy": 1.0}
    conditions = {
        f"{language}/{case}": dict(valid)
        for language in ("en", "zh")
        for case in ("complete", "eligible", "gate_false", "missing")
    }
    development = {f"transfer/{k}": dict(v) for k, v in conditions.items()}
    slices = {
        "conditions": conditions,
        "development": development,
        "fixture": {"all": {"count": 38, "accuracy": 1.0}},
    }
    comparisons = {
        f"{cohort}-{reference}": {
            "families": {
                str(i): {"local": {"accuracy": 0.95}, "reference": {"accuracy": 0.95}}
                for i in range(8)
            },
            "macro_accuracy": {"delta": 0},
        }
        for cohort in ("retention", "regression")
        for reference in ("parent", "repaired")
    }
    alerts = {k: [] for k in comparisons}
    assert all(acceptance_checks(slices, development, comparisons, alerts).values())
    comparisons["regression-parent"]["families"]["0"]["local"]["accuracy"] = 0.92
    assert not acceptance_checks(slices, development, comparisons, alerts)[
        "all_original_families_retained"
    ]
    del conditions["zh/eligible"]
    assert not acceptance_checks(slices, development, comparisons, alerts)["complete_and_eligible"]
    assert not acceptance_checks(slices, {}, {}, {})["new_complete_and_eligible_retained"]
