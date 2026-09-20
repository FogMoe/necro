import pytest

from necro.evaluation import metrics, summarize
from necro.schema import EvaluationResponse


def test_metrics_use_probabilities_not_api_confidence():
    examples = [{"id": "x", "expected": {"binary": False, "score": 2}, "language": "zh"}]
    responses = [
        EvaluationResponse.model_validate(
            {
                "model": "local",
                "usage": {"input_tokens": 10, "output_tokens": 0},
                "answers": {
                    "binary": {"type": "noul", "noul": 0.25},
                    "score": {
                        "type": "score",
                        "score": 1.5,
                        "legend": {"0": "a", "1": "b", "2": "c"},
                        "probabilities": {"0": 0.2, "1": 0.1, "2": 0.7},
                        "confidence": 0.1,
                    },
                },
            }
        )
    ]
    summary, rows = summarize(examples, responses)
    assert summary["overall"]["accuracy"] == 1
    assert rows[0]["top_probability"] == 0.75
    assert rows[1]["top_probability"] == 0.7
    assert summary["score_mae"] == 0.5
    assert summary["overall"]["ece_10_bins"] == pytest.approx(0.275)


def test_empty_metrics():
    assert metrics([]) == {"count": 0}
