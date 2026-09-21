from necro.training.analysis.condition_balance import condition_checks, minimum_accuracy


def test_missing_slice_fails_and_high_aggregate_cannot_hide_false_rejections():
    assert not minimum_accuracy({"all": {"count": 1000, "accuracy": 1}}, ["zh/eligible"], 0.98)
    regression = {"all": {"count": 1000, "accuracy": 0.99}}
    for language in ("en", "zh"):
        for case in ("complete", "eligible", "gate_false", "missing"):
            regression[f"{language}/{case}"] = {"count": 100, "accuracy": 1.0}
    regression["zh/eligible"]["accuracy"] = 0.74
    development = {
        f"transfer/{language}/{case}": {"count": 100, "accuracy": 1}
        for language in ("en", "zh")
        for case in ("eligible", "gate_false", "missing")
    }
    checks = condition_checks(development, regression, {"all": {"count": 38, "accuracy": 1}})
    assert not checks["complete_and_eligible_retained"]
    assert all(value for name, value in checks.items() if name != "complete_and_eligible_retained")
