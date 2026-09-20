from necro.training.data.phase2_refinement import contrast_rules


def test_numeric_truth_tables_and_negation_cover_missing_fields():
    rows = {row["id"]: row for row in contrast_rules()}
    # 前五个组分别是 >=、>、<=、<、==；前 3 个状态是阈值下、等于、阈值上。
    expected = (
        (False, True, True),
        (False, False, True),
        (True, True, False),
        (True, False, False),
        (False, True, False),
    )
    for group, truth in enumerate(expected):
        for language in ("en", "zh"):
            for side, gold in enumerate([*truth, False, False]):
                positive = rows[f"contrast-rule/{group}/{language}/0/{side}"]
                negative = rows[f"contrast-rule/{group}/{language}/1/{side}"]
                assert positive["expected"]["decision"] is gold
                assert negative["expected"]["decision"] is not gold
                assert positive["group_id"] == negative["group_id"]
                assert positive["request"]["state"] == negative["request"]["state"]
            assert len(rows[f"contrast-rule/{group}/{language}/0/3"]["request"]["state"]) == 2
