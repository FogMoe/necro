import socket
import threading
import time

import pytest
import uvicorn
from fastapi.testclient import TestClient
from pydantic import ValidationError

from necro.api import create_app
from necro.config import MODEL_ID, Settings
from necro.engine import DecisionEngine, ScoredTask, answer_for, prepare_task
from necro.schema import Choice, EvaluationRequest, Score


class FixedScorer:
    def __init__(self):
        self.tasks = []

    def score(self, tasks):
        self.tasks = tasks
        results = []
        for task in tasks:
            count = len(task.keys)
            p = [1.0] if count == 1 else [0.8] + [0.2 / (count - 1)] * (count - 1)
            results.append(ScoredTask(p, 42))
        return results


def request_body():
    return {
        "state": {"text": "请退款", "nested": ["evidence", {"count": 2}]},
        "model": "jev-latest",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": {"task": "选择部门"},
                "criteria": {"billing": {"examples": ["refunds"]}, "support": None},
            },
            "refund": {
                "type": "noul",
                "instructions": "是否要求退款？",
                "criteria": {"true": ["明确要求退款"], "false": "未要求退款"},
            },
            "severity": {
                "type": "score",
                "instructions": "严重程度？",
                "criteria": [{"meaning": "轻微"}, "中等", ["严重"]],
            },
        },
    }


@pytest.fixture
def setup():
    scorer = FixedScorer()
    app = create_app(Settings(api_key="test-key"), DecisionEngine(scorer))
    with TestClient(app) as client:
        client.headers["Authorization"] = "Bearer test-key"
        yield client, scorer


def test_mixed_types_preserve_ids_and_structured_legend(setup):
    client, _ = setup
    result = client.post("/v1/systemone", json=request_body())
    assert result.status_code == 200
    body = result.json()
    assert body["model"] == MODEL_ID
    assert body["answers"]["route"]["choice"] == "billing"
    assert body["answers"]["refund"] == {"type": "noul", "noul": 0.8}
    score = body["answers"]["severity"]
    assert score["score"] == pytest.approx(0.3)
    assert score["legend"] == {"0": {"meaning": "轻微"}, "1": "中等", "2": ["严重"]}
    assert body["usage"] == {"input_tokens": 126, "output_tokens": 0}


def test_question_ids_are_not_sent_to_model(setup):
    client, scorer = setup
    body = request_body()
    body["questions"] = {"SECRET_QUESTION_ID": body["questions"]["route"]}
    client.post("/v1/systemone", json=body).raise_for_status()
    assert "SECRET_QUESTION_ID" not in str(scorer.tasks[0].messages(["A", "B"]))


@pytest.mark.parametrize("header", [None, "Bearer wrong", "Basic test-key"])
def test_authentication_returns_401(setup, header):
    client, _ = setup
    client.headers.pop("Authorization")
    headers = {} if header is None else {"Authorization": header}
    assert client.get("/v1/models", headers=headers).status_code == 401
    assert client.post("/v1/systemone", json=request_body(), headers=headers).status_code == 401


@pytest.mark.parametrize("count", [1, 26, 60, 255])
def test_dynamic_choice_sizes(setup, count):
    client, _ = setup
    body = request_body()
    body["questions"] = {
        "x": {
            "type": "choice",
            "instructions": "Select.",
            "criteria": {f"option-{i}": None for i in range(count)},
        }
    }
    response = client.post("/v1/systemone", json=body)
    assert response.status_code == 200
    answer = response.json()["answers"]["x"]
    assert len(answer["probabilities"]) == count
    assert sum(answer["probabilities"].values()) == pytest.approx(1)


@pytest.mark.parametrize(
    "question",
    [
        {"type": "choice", "instructions": "x", "criteria": {}},
        {"type": "choice", "instructions": "x", "criteria": {str(i): None for i in range(256)}},
        {"type": "score", "instructions": "x", "criteria": ["one"]},
        {"type": "score", "instructions": "x", "criteria": ["x"] * 11},
        {"type": "score", "instructions": "x", "criteria": [1, 2]},
        {"type": "noul", "instructions": "x", "criteria": {"maybe": "x"}},
    ],
)
def test_invalid_questions_return_422(setup, question):
    client, _ = setup
    body = request_body()
    body["questions"] = {"x": question}
    assert client.post("/v1/systemone", json=body).status_code == 422


def test_unknown_model_rejected_before_inference(setup):
    client, scorer = setup
    body = request_body()
    body["model"] = "nonexistent"
    assert client.post("/v1/systemone", json=body).status_code == 422
    assert not scorer.tasks


def test_noul_optional_and_partial_criteria():
    body = request_body()
    body["questions"] = {"x": {"type": "noul", "instructions": "yes?", "criteria": {"true": None}}}
    request = EvaluationRequest.model_validate(body)
    task = prepare_task(request.state, request.questions["x"])
    assert task.keys == ["yes", "no"]
    assert all(task.descriptions)


@pytest.mark.parametrize("probabilities", [[float("nan"), 0], [-0.2, 1.2], [0.2, 0.2], [0.9]])
def test_bad_inference_probabilities_fail(probabilities):
    question = Choice(type="choice", instructions="x", criteria={"a": None, "b": None})
    with pytest.raises(RuntimeError):
        answer_for(prepare_task("x", question), ScoredTask(probabilities, 1))


def test_score_uses_mean_not_winning_level():
    question = Score(type="score", instructions="x", criteria=["low", "medium", "high"])
    answer = answer_for(prepare_task("x", question), ScoredTask([0.2, 0.1, 0.7], 1))
    assert answer.score == pytest.approx(1.5)


def test_multiple_requests_keep_distinct_budgets_and_answers():
    scorer = FixedScorer()
    engine = DecisionEngine(scorer)
    first = EvaluationRequest.model_validate(request_body())
    second = first.model_copy(update={"questions": {"different-id": first.questions["refund"]}})
    responses = engine.evaluate_many([first, second])
    assert len(responses) == 2
    assert list(responses[1].answers) == ["different-id"]
    assert [task.request_group for task in scorer.tasks] == [0, 0, 0, 1]


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_invalid_temperatures_rejected(temperature):
    with pytest.raises(ValueError):
        Settings(temperature=temperature)


def test_numeric_state_rejected():
    body = request_body()
    body["state"] = 123
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(body)


def test_official_python_sdk_over_http():
    from typesafe_sdk import TypeSafeClient

    app = create_app(Settings(api_key="test-key"), DecisionEngine(FixedScorer()))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while not server.started and time.monotonic() < deadline:
                time.sleep(0.01)
            assert server.started
            with TypeSafeClient(api_key="test-key", base_url=f"http://127.0.0.1:{port}") as client:
                body = request_body()
                response = client.system_one(state=body["state"], questions=body["questions"])
                assert response.answers["route"].choice == "billing"
                assert response.answers["refund"].noul == pytest.approx(0.8)
                # SDK 将 JSON 对象里的数字字符串键转换成整数。
                assert response.answers["severity"].legend[0] == {"meaning": "轻微"}
                assert client.models.list().models[0].name == MODEL_ID
        finally:
            server.should_exit = True
            thread.join(5)
