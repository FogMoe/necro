from collections import Counter

from necro.diagnostics.numeric_regression import rule_metadata
from necro.training.data.condition_refinement import FIELDS, audit_condition_matrix, construct


def test_condition_partitions_balance_gates_and_polarity():
    parts = construct()
    for role, rows in parts.items():
        assert len(rows) == len(FIELDS[role]) * 3 * 60
        assert len(set(Counter(r["condition_case"] for r in rows).values())) == 1
        assert Counter(r["expected"]["decision"] for r in rows)[True] == len(rows) // 2
        for row in rows:
            rule_metadata(row)
    assert not {r["condition_style"] for r in parts["train"]} & {
        r["condition_style"] for r in parts["development"]
    }
    assert not {r["condition_style"] for r in parts["test"]} & {
        r["condition_style"] for r in parts["development"]
    }
    assert not set(FIELDS["train"]) & set(FIELDS["development"])
    assert not set(FIELDS["train"]) & set(FIELDS["test"])


def test_followup_preserves_holdouts_and_pairs_identical_rules():
    first, second = construct(), construct(2)
    assert first["development"] == second["development"]
    assert first["test"] == second["test"]
    paired = {}
    for row in second["train"]:
        rule_metadata(row)
        group = (
            row["group_id"],
            row["language"],
            row["id"].split("/")[-4],
            row["id"].split("/")[-1],
        )
        paired.setdefault(group, set()).add(row["request"]["questions"]["decision"]["instructions"])
    assert all(len(rules) == 1 for rules in paired.values())


def test_complete_matrix_covers_hard_missing_cases_and_preserves_holdouts():
    original, revised = construct(), construct(3)
    assert revised["development"] == original["development"]
    assert revised["test"] == original["test"]
    assert audit_condition_matrix(revised["train"]) == {
        "combinations": 800,
        "examples_per_combination": 4,
    }
