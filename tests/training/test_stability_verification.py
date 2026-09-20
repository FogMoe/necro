import json
from types import SimpleNamespace

import pytest

from necro.evaluation import summarize
from necro.experiment_guard import digest
from necro.schema import EvaluationResponse
from necro.training.data.training_data import write_jsonl
from necro.training.release.package_verification import verify_reload
from necro.training.release.stability_assessment import paired_comparison


def evidence(tmp_path):
    example = {
        "id": "one",
        "family": "numeric_rule",
        "group_id": "one",
        "language": "en",
        "request": {
            "model": "test",
            "state": "1",
            "questions": {"q": {"type": "noul", "instructions": "Is the value one?"}},
        },
        "expected": {"q": True},
    }
    response = EvaluationResponse.model_validate(
        {
            "model": "test",
            "answers": {"q": {"type": "noul", "noul": 0.75}},
            "usage": {"input_tokens": 1},
        }
    )
    dataset = tmp_path / "cohort.jsonl"
    write_jsonl(dataset, [example])
    run = tmp_path / "result"
    run.mkdir()
    selected = {"weights_sha256": "weights", "temperatures": {"noul": 1.0}}
    report, judgments = summarize([example], [response])
    report["metadata"] = {
        "dataset_sha256": digest(dataset),
        "full_dataset_evaluated": True,
        "evaluated_questions": 1,
        "backend": "local",
        "adapter_weights_sha256": "weights",
        "noul_temperature": 1.0,
    }
    (run / "summary.json").write_text(json.dumps(report))
    write_jsonl(run / "judgments.jsonl", judgments)
    write_jsonl(run / "predictions.jsonl", [{"id": "one", "response": response.model_dump()}])
    return dataset, run, selected, response


def test_reload_rejects_probability_drift_with_unchanged_decision(tmp_path):
    dataset, run, selected, response = evidence(tmp_path)
    engine = SimpleNamespace(evaluate_many=lambda requests: [response])
    assert verify_reload(engine, dataset, run, selected)["max_probability_difference"] == 0
    response.answers["q"].noul = 0.7528
    with pytest.raises(ValueError, match="0.0028"):
        verify_reload(engine, dataset, run, selected)
    selected["weights_sha256"] = "different"
    with pytest.raises(ValueError, match="weights and calibration"):
        verify_reload(engine, dataset, run, selected)


def test_single_task_comparison_has_finite_json_and_no_jev_gate(tmp_path):
    _, run, _, _ = evidence(tmp_path)
    result = paired_comparison(run, run, "independent condition transfer")
    assert result["normalized_score_mae_delta"] is None
    assert "performance_gates" not in result
    assert result["macro_accuracy"]["delta"] == 0
    json.dumps(result, allow_nan=False)
    path = run / "summary.json"
    summary = json.loads(path.read_text())
    summary["metadata"]["full_dataset_evaluated"] = False
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="complete cohorts"):
        paired_comparison(run, run, "independent condition transfer")
