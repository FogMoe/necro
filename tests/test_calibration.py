import json

import numpy as np
import pytest

from necro.calibration import fit_temperature, temper


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
