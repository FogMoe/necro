import copy

import pytest

from necro.diagnostics.numeric_regression import iff_instruction, rule_metadata
from necro.training.data.phase2_data import numeric_rules


def test_diagnostic_oracle_checks_all_operators_and_missing_gates():
    rows = numeric_rules("test", 30)
    cases = {rule_metadata(row)["case"] for row in rows}
    assert cases == {"complete", "gate_false", "missing"}
    assert {rule_metadata(row)["operator"] for row in rows} == {"ge", "gt", "le", "lt", "eq"}
    assert {rule_metadata(row)["negated"] for row in rows} == {False, True}
    broken = copy.deepcopy(rows[0])
    broken["expected"]["decision"] = not broken["expected"]["decision"]
    with pytest.raises(ValueError, match="disagrees with label"):
        rule_metadata(broken)


def test_wording_and_field_changes_preserve_numeric_truth():
    for row in numeric_rules("test", 30):
        original = rule_metadata(row)
        renamed = copy.deepcopy(row)
        fields = ["value_a", "value_b", "gate"]
        mapping = dict(zip(original["fields"], fields, strict=True))
        renamed["request"]["state"] = {mapping[k]: v for k, v in row["request"]["state"].items()}
        renamed["request"]["questions"]["decision"]["instructions"] = iff_instruction(
            fields, original["operator"], row["language"], original["negated"]
        )
        transformed = rule_metadata(renamed)
        for key in ("operator", "case", "relation", "negated"):
            assert transformed[key] == original[key]
