import pytest

from necro.training.analysis.phase2_analysis import cluster_intervals


def test_source_group_bootstrap_keeps_translations_paired():
    left, right = [], []
    for group in range(4):
        for language in ("en", "zh"):
            common = {
                "id": f"{group}/{language}",
                "question_id": "q",
                "family": "test",
                "group_id": str(group),
                "expected": True,
            }
            left.append({**common, "correct": True})
            right.append({**common, "correct": False})
    report = cluster_intervals(left, right)
    assert report["families"]["test"]["source_groups"] == 4
    assert report["macro_accuracy_delta_95ci"] == [1, 1]
    with pytest.raises(ValueError, match="同题结果"):
        cluster_intervals(left, right[::-1])


def test_probability_metric_intervals_use_paired_errors():
    left = [
        {
            "id": str(i),
            "question_id": "q",
            "family": "ordinal_rule",
            "group_id": str(i // 2),
            "expected": "0",
            "brier": 0.3,
            "score_normalized_error": 0.1,
        }
        for i in range(6)
    ]
    right = [{**row, "brier": 0.1, "score_normalized_error": 0.4} for row in left]
    brier = cluster_intervals(left, right, metric="brier")
    score = cluster_intervals(left, right, metric="score_normalized_error")
    assert brier["macro_brier_delta_95ci"] == pytest.approx([0.2, 0.2])
    assert score["macro_score_normalized_mae_delta_95ci"] == pytest.approx([-0.3, -0.3])
