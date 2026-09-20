from necro.training.data.operator_contrast import construct


def test_same_state_has_opposite_labels_when_only_operator_changes():
    rows = {row["id"]: row for row in construct()}
    for language in ("en", "zh"):
        for group, first, second in ((0, "lt", "ge"), (1, "eq", "gt"), (2, "gt", "le")):
            a = rows[f"operator-contrast/{group}/{first}/{language}/compound/0"]
            b = rows[f"operator-contrast/{group}/{second}/{language}/compound/0"]
            assert a["request"]["state"] == b["request"]["state"]
            assert a["group_id"] == b["group_id"]
            assert a["expected"]["decision"] is True
            assert b["expected"]["decision"] is False
            atomic = rows[f"operator-contrast/{group}/{first}/{language}/atomic"]
            negated = rows[f"operator-contrast/{group}/{first}/{language}/compound/1"]
            assert atomic["expected"] == a["expected"]
            assert negated["expected"]["decision"] is False
