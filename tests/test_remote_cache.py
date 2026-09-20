import json

import httpx

from necro.evaluation import evaluate_remote
from necro.schema import EvaluationRequest


def test_remote_resume_reuses_completed_answers_and_binds_candidate_order(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "fixture-not-a-real-key")
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://fixture.invalid")
    monkeypatch.setattr("necro.evaluation.time.sleep", lambda seconds: None)
    fail = True
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload["state"])
        if payload["state"] == "second" and fail:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "usage": {"input_tokens": 1, "output_tokens": 0},
                "answers": {
                    "q": {
                        "type": "choice",
                        "choice": "a",
                        "confidence": 1,
                        "probabilities": {"a": 1, "b": 0},
                    }
                },
            },
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        "necro.evaluation.httpx.Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )

    def make(state, reverse=False):
        criteria = {"b": "B", "a": "A"} if reverse else {"a": "A", "b": "B"}
        return EvaluationRequest.model_validate(
            {
                "model": "jev-1.13.0",
                "state": state,
                "questions": {
                    "q": {"type": "choice", "instructions": "Pick A", "criteria": criteria}
                },
            }
        )

    import pytest

    with pytest.raises(httpx.HTTPStatusError):
        evaluate_remote([make("first"), make("second")], tmp_path)
    assert len(list(tmp_path.glob("*.json"))) == 1
    fail = False
    calls.clear()
    stats = {}
    answers = evaluate_remote([make("first"), make("second")], tmp_path, stats)
    assert len(answers) == 2
    assert calls == ["second"]
    assert stats == {"hits": 1, "requests": 2}
    evaluate_remote([make("first", reverse=True)], tmp_path)
    assert calls == ["second", "first"]
