import copy

import pytest

from necro.training.release.unified_release import (
    independent_checks,
    phase4_gain,
    validate_recipe,
    write_new,
)


def test_cloud_training_stops_after_candidate_freeze(monkeypatch, tmp_path):
    from necro.training import cloud

    monkeypatch.setattr(cloud, "DATA", tmp_path)
    monkeypatch.setattr(cloud, "OUTPUT", tmp_path / "output")
    monkeypatch.setattr(cloud, "verify_bundle", lambda: None)
    monkeypatch.setattr(cloud, "verify", lambda *_: None)
    monkeypatch.setattr(cloud, "read", lambda _: {})
    monkeypatch.setattr("sys.argv", ["cloud", "train", "primary"])
    (tmp_path / "selection.json").write_text("{}")
    with pytest.raises(ValueError, match="frozen"):
        cloud.main()


def recipe():
    plan = {
        "checkpoint": "Qwen/Qwen3.5-0.8B",
        "revision": "frozen",
        "rank": 16,
        "epochs": 2,
        "seed": 2026,
        "objective": "answer-ce",
        "gradient_checkpointing": True,
        "recipes": {"primary": {"learning_rate": 1e-4}},
        "effective_batch_size": 16,
    }
    contract = {
        **{k: v for k, v in plan.items() if k not in {"recipes", "effective_batch_size"}},
        "initial_adapter": None,
        "learning_rate": 1e-4,
        "train_sha256": "data",
        "batch_size": 8,
        "gradient_accumulation": 2,
    }
    return plan, contract


@pytest.mark.parametrize(
    "field,value",
    [
        ("initial_adapter", "old-adapter"),
        ("revision", "new-main"),
        ("train_sha256", "changed"),
        ("learning_rate", 5e-5),
        ("epochs", 3),
        ("gradient_accumulation", 1),
    ],
)
def test_freeze_rejects_recipe_drift(field, value):
    plan, contract = recipe()
    validate_recipe(contract, plan, "primary", "data")
    contract[field] = value
    with pytest.raises(ValueError):
        validate_recipe(contract, plan, "primary", "data")


def test_release_records_cannot_be_overwritten(tmp_path):
    path = tmp_path / "selection.json"
    write_new(path, {"weights": "first"})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_new(path, {"weights": "second"})
    assert path.read_bytes() == before


def test_failed_development_cannot_create_selection_or_open_test(monkeypatch, tmp_path):
    from necro.training.release import unified_release as release

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(release, "verify_bundle", lambda: None)
    monkeypatch.setattr(release, "DATA", tmp_path / "data")
    monkeypatch.setattr(release, "OUTPUT", tmp_path / "runs")
    monkeypatch.setattr(release, "SELECTED", tmp_path / "selected")
    monkeypatch.setattr(release, "FINAL", tmp_path / "final")
    monkeypatch.setattr(release, "verify", lambda *_: pytest.fail("Opened data before rejection"))
    run = release.OUTPUT / "primary"
    write_new(tmp_path / "cloud-plan.json", {})
    decision = tmp_path / "decision.json"
    write_new(
        decision,
        {
            "recipe": "primary",
            "rationale": "Test development decision",
            "independent_test_opened": False,
        },
    )
    write_new(run / "assessment.json", {"eligible_for_freeze": False})
    write_new(run / "calibration-brier-fit.json", {})
    (run / "adapter").mkdir()
    (run / "adapter/adapter_model.safetensors").write_bytes(b"test weights")
    with pytest.raises(ValueError, match="开发验收未通过"):
        release.freeze("primary", decision)
    assert not release.SELECTED.exists()
    assert not (release.DATA / "selection.json").exists()
    assert not release.FINAL.exists()


def test_independent_checks_require_both_references_and_all_language_slices():
    plan = {
        "condition_minimum": 0.95,
        "complete_eligible_max_drop": 0.02,
        "family_max_drop": 0.02,
        "macro_max_drop_vs_repaired": 0.005,
    }
    slices = {
        name: {
            f"{lang}/{case}": {"count": 100, "accuracy": 0.99}
            for lang in ("en", "zh")
            for case in ("complete", "eligible", "gate_false", "missing")
        }
        for name in ("selected", "parent")
    }
    comparisons = {
        name: {
            "families": {
                str(i): {"local": {"accuracy": 0.95}, "reference": {"accuracy": 0.95}}
                for i in range(8)
            },
            "macro_accuracy": {"delta": 0.01},
        }
        for name in ("parent", "repaired")
    }
    alerts = {name: [] for name in comparisons}
    assert all(independent_checks(slices, comparisons, alerts, plan).values())
    broken = copy.deepcopy(slices)
    del broken["selected"]["zh/missing"]
    assert not independent_checks(broken, comparisons, alerts, plan)["independent_preconditions"]
    broken = copy.deepcopy(comparisons)
    broken["parent"]["families"]["3"]["local"]["accuracy"] = 0.92
    assert not independent_checks(slices, broken, alerts, plan)["independent_families_retained"]
    del broken["parent"]
    assert not independent_checks(slices, broken, alerts, plan)["independent_families_retained"]
    comparisons["repaired"]["macro_accuracy"]["delta"] = -0.006
    assert not independent_checks(slices, comparisons, alerts, plan)["independent_macro_retained"]
    alerts["parent"] = [{"language": "zh", "accuracy_delta": -0.06}]
    assert not independent_checks(slices, comparisons, alerts, plan)[
        "independent_languages_retained"
    ]


def test_retention_margin_or_tie_is_not_a_phase4_gain():
    for delta in (-0.004, 0):
        assert not phase4_gain({"repaired": {"macro_accuracy": {"delta": delta}}}, ("repaired",))
    assert phase4_gain({"repaired": {"macro_accuracy": {"delta": 0.01}}}, ("repaired",))
    assert not phase4_gain({}, ("repaired",))
