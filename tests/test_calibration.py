import json

import numpy as np
import pytest

from necro.calibration import fit_primitive_temperatures, fit_temperature, temper, transform_report
from necro.config import Settings
from necro.engine import ScoredTask, answer_for, prepare_task
from necro.experiment_guard import digest, register
from necro.schema import EvaluationRequest
from necro.training.data.training_data import write_jsonl


def test_temperature_matches_logit_scaling_without_changing_ranking():
    logits = np.array([1.0, -3.0, 2.0])
    original = np.exp(logits) / np.exp(logits).sum()
    expected = np.exp(logits / 2) / np.exp(logits / 2).sum()
    result = temper(original, 2)
    assert result == pytest.approx(expected)
    assert np.argmax(result) == np.argmax(original)
    assert max(result) < max(original)


def test_calibration_reduces_overconfidence_on_mixed_correctness(tmp_path):
    path = tmp_path / "predictions.jsonl"
    rows = []
    for i in range(10):
        rows.append(
            {
                "id": str(i),
                "expected": {"q": "a" if i < 7 else "b"},
                "response": {
                    "model": "test",
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                    "answers": {
                        "q": {
                            "type": "choice",
                            "choice": "a",
                            "confidence": 0.98,
                            "probabilities": {"a": 0.99, "b": 0.01},
                        }
                    },
                },
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    result = fit_temperature(path)
    assert result["temperature"] > 1
    assert result["after_nll"] < result["before_nll"]
    assert result["questions"] == 10


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        temper([0.2, 0.8], temperature)


def test_primitive_temperature_overrides_do_not_change_other_primitive():
    settings = Settings(temperature=1.5, noul_temperature=2.0, score_temperature=0.5)
    assert settings.temperature_for("choice") == 1.5
    assert settings.temperature_for("noul") == 2.0
    assert settings.temperature_for("score") == 0.5
    with pytest.raises(ValueError):
        Settings(score_temperature=float("nan"))


@pytest.mark.parametrize("objective", ["nll", "brier"])
def test_mixed_calibration_preserves_decisions_and_recomputes_expected_score(tmp_path, objective):
    examples, predictions = [], []
    for primitive in ("noul", "choice", "score"):
        for i in range(20):
            question = {"type": primitive, "instructions": "Judge using the supplied rule."}
            if primitive == "choice":
                question["criteria"] = {"a": "low", "b": "high"}
            elif primitive == "score":
                question["criteria"] = ["low", "high"]
            gold = (
                (i < 14)
                if primitive == "noul"
                else (("a" if i < 14 else "b") if primitive == "choice" else (0 if i < 14 else 1))
            )
            identifier = f"{primitive}/{i}"
            example = {
                "id": identifier,
                "group_id": identifier,
                "family": primitive,
                "language": "en",
                "source": "fixture",
                "request": {"model": "test", "state": identifier, "questions": {"q": question}},
                "expected": {"q": gold},
            }
            request = EvaluationRequest.model_validate(example["request"])
            answer = answer_for(
                prepare_task(request.state, request.questions["q"]), ScoredTask([0.99, 0.01], 0)
            )
            examples.append(example)
            predictions.append(
                {
                    "id": identifier,
                    "expected": example["expected"],
                    "response": {
                        "model": "test",
                        "answers": {"q": answer.model_dump()},
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    },
                }
            )
    dataset = tmp_path / "calibration.jsonl"
    write_jsonl(dataset, examples)
    register(tmp_path, {"calibration": dataset}, {})
    raw = tmp_path / "raw"
    raw.mkdir()
    write_jsonl(raw / "predictions.jsonl", predictions)
    (raw / "summary.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "temperature": 1,
                    "dataset_sha256": digest(dataset),
                }
            }
        )
    )
    fit = fit_primitive_temperatures(dataset, raw / "predictions.jsonl", objective=objective)
    assert set(fit["temperatures"]) == {"noul", "choice", "score"}
    assert all(
        item[f"after_{objective}"] < item[f"before_{objective}"]
        for item in fit["primitives"].values()
    )
    report = transform_report(
        dataset, raw / "predictions.jsonl", tmp_path / "scaled", fit["temperatures"]
    )
    assert report["overall"]["accuracy"] == pytest.approx(0.7)
    records = [
        json.loads(line)
        for line in (tmp_path / "scaled/predictions.jsonl").read_text().splitlines()
    ]
    score = records[-1]["response"]["answers"]["q"]
    assert score["score"] == pytest.approx(score["probabilities"]["1"])
    assert score["score"] > 0.01
    with pytest.raises(ValueError, match="原始预测"):
        fit_primitive_temperatures(dataset, tmp_path / "scaled/predictions.jsonl")
