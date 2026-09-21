"""验证最终导出包：完整题集重载一致性、真实 SDK/HTTP 与同口径延迟。"""

import argparse
import json
import os
import secrets
import socket
import threading
import time
from importlib.metadata import version
from pathlib import Path

import httpx
import numpy as np
import uvicorn
from dotenv import load_dotenv

from necro.api import create_app
from necro.config import MODEL_ID, Settings
from necro.engine import DecisionEngine
from necro.evaluation import read_examples
from necro.experiment_guard import digest, verify
from necro.export import sha256
from necro.schema import EvaluationRequest, EvaluationResponse


def probabilities(answer):
    return (
        [answer.noul, 1 - answer.noul]
        if answer.type == "noul"
        else list(answer.probabilities.values())
    )


def mixed_body():
    return {
        "model": MODEL_ID,
        "state": {"text": "我被扣了两次款，请退还重复付款。", "failed_checks": 2},
        "questions": {
            "route": {
                "type": "choice",
                "instructions": {"task": "选择负责部门"},
                "criteria": {"billing": {"examples": ["付款", "退款"]}, "support": "软件故障"},
            },
            "refund": {
                "type": "noul",
                "instructions": ["是否请求退款？"],
                "criteria": {"true": "明确请求退款", "false": "没有请求退款"},
            },
            "level": {
                "type": "score",
                "instructions": "根据 failed_checks 的值选择对应等级。",
                "criteria": [{"rule": "等于 0"}, "等于 1", ["不少于 2"]],
            },
        },
    }


def latency(client, model, questions, repeats=20):
    question = {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
            "billing": "Payments and refunds",
            "technical": "Software bugs",
            "sales": "New product purchases",
        },
    }
    body = {
        "model": model,
        "state": "I was charged twice. Please refund the duplicate payment.",
        "questions": {f"q{i}": question for i in range(questions)},
    }
    measurements, tokens = [], []
    for i in range(repeats + 1):
        start = time.perf_counter()
        response = client.post("/v1/systemone", json=body)
        response.raise_for_status()
        result = EvaluationResponse.model_validate(response.json())
        elapsed = time.perf_counter() - start
        if result.model != model or len(result.answers) != questions:
            raise ValueError("计时响应的模型或问题数不符。")
        if i:
            measurements.append(elapsed)
            tokens.append(result.usage.input_tokens)
    return {
        "questions_per_request": questions,
        "repeats": repeats,
        "warmup_requests": 1,
        "p50_ms": float(np.median(measurements) * 1000),
        "p95_ms": float(np.quantile(measurements, 0.95) * 1000),
        "latencies_seconds": measurements,
        "input_tokens": tokens,
        "request": body,
        "scope": "Serial HTTP, connection reused, no client response cache; provider caching is "
        "unknown. Batch case repeats one "
        "short Choice question.",
    }


def verify_reload(engine, dataset, predictions, selected):
    """Compare every response against the frozen adapter's calibrated predictions."""
    metadata = json.loads((predictions / "summary.json").read_text())["metadata"]
    if (
        metadata["dataset_sha256"] != digest(dataset)
        or metadata.get("adapter_weights_sha256") != selected["weights_sha256"]
        or not metadata.get("full_dataset_evaluated")
        or any(metadata.get(f"{p}_temperature") != t for p, t in selected["temperatures"].items())
    ):
        raise ValueError("Reload reference must match the complete cohort, weights and calibration")
    examples, originals = read_examples(dataset), read_examples(predictions / "predictions.jsonl")
    if [r["id"] for r in examples] != [r["id"] for r in originals]:
        raise ValueError("重载参考题目不一致。")
    requests = [EvaluationRequest.model_validate(r["request"]) for r in examples]
    reloaded = engine.evaluate_many(requests)
    max_delta, changed = 0.0, 0
    for old, actual in zip(originals, reloaded, strict=True):
        expected = EvaluationResponse.model_validate(old["response"])
        if actual.model != expected.model or actual.answers.keys() != expected.answers.keys():
            raise ValueError("重载模型身份或问题映射改变。")
        for key, answer in actual.answers.items():
            previous = expected.answers[key]
            if answer.type != previous.type:
                raise ValueError("重载改变判断类型。")
            if answer.type != "noul" and list(answer.probabilities) != list(previous.probabilities):
                raise ValueError("重载改变候选概率键或其顺序。")
            a, b = probabilities(answer), probabilities(previous)
            if len(a) != len(b):
                raise ValueError("重载改变概率数量。")
            max_delta = max(max_delta, max(abs(x - y) for x, y in zip(a, b, strict=True)))
            changed += int(np.argmax(a) != np.argmax(b))
            if answer.type == "choice" and answer.choice != previous.choice:
                raise ValueError("重载改变 Choice 返回的候选键。")
            if answer.type == "score" and answer.legend != previous.legend:
                raise ValueError("重载改变等级说明。")
    if max_delta > 1e-5 or changed:
        raise ValueError(f"重载不一致：最大概率误差 {max_delta}，改变 {changed} 个选择。")
    return {
        "dataset_sha256": digest(dataset),
        "examples": len(examples),
        "full_dataset": True,
        "max_probability_difference": max_delta,
        "changed_argmax": changed,
        "tolerance": 1e-5,
    }


def verify_package(
    package,
    data,
    predictions,
    output,
    benchmark_jev=False,
    regression_data=None,
    regression_predictions=None,
):
    import torch
    from typesafe_sdk import TypeSafeClient

    from necro.backend import TransformersScorer

    if output.exists():
        raise ValueError("导出验证已记录，不可覆盖。")
    if (regression_data is None) != (regression_predictions is None):
        raise ValueError("Regression data and predictions must be supplied together")
    selected = json.loads((data / "selection.json").read_text())
    dataset = verify(data, "test", Path(selected["adapter"]))
    exported = json.loads((package / "export.json").read_text())
    if exported["adapter_weights_sha256"] != selected["weights_sha256"]:
        raise ValueError("导出包不是冻结的权重。")
    if sha256(package / "adapter/adapter_model.safetensors") != selected["weights_sha256"]:
        raise ValueError("导出包内的 LoRA 权重校验失败。")
    if exported["temperatures"] != selected["temperatures"]:
        raise ValueError("导出包不是冻结的校准参数。")
    merged = package / exported["merged_path"]
    for name, info in exported["merged_weights"].items():
        if sha256(merged / name) != info["sha256"]:
            raise ValueError("合并权重校验失败。")
    metadata = json.loads((predictions / "summary.json").read_text())["metadata"]
    if (
        metadata["dataset_sha256"] != digest(dataset)
        or metadata.get("adapter_weights_sha256") != selected["weights_sha256"]
        or not metadata.get("full_dataset_evaluated")
        or any(metadata.get(f"{p}_temperature") != t for p, t in selected["temperatures"].items())
    ):
        raise ValueError("参考预测必须是全量、已冻结权重和校准的最终同题结果。")
    settings = Settings(
        checkpoint=str(merged.resolve()),
        device="cuda",
        api_key=secrets.token_urlsafe(24),
        **{f"{p}_temperature": t for p, t in selected["temperatures"].items()},
    )
    scorer = TransformersScorer(settings)
    scorer.load()
    engine = DecisionEngine(scorer)
    loaded_allocated_gib = torch.cuda.memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    reload = verify_reload(engine, dataset, predictions, selected)
    regression_reload = None
    if regression_data is not None:
        registered = verify(regression_data.parent, "regression")
        plan = json.loads((Path(selected["adapter"]).parent / "validation-plan.json").read_text())
        if (
            registered.resolve() != regression_data.resolve()
            or digest(registered) != plan["task_regression"]["sha256"]
        ):
            raise ValueError("Regression reload does not match the frozen validation plan")
        regression_reload = verify_reload(engine, registered, regression_predictions, selected)
    checks = {
        "model": scorer.model_id,
        "dataset_sha256": digest(dataset),
        "adapter_weights_sha256": selected["weights_sha256"],
        "merged_weights": exported["merged_weights"],
        "reload": {**reload, "load_seconds": scorer.load_seconds},
        "regression_reload": regression_reload,
        "gpu": torch.cuda.get_device_name(),
        "loaded_allocated_gib": loaded_allocated_gib,
        "peak_inference_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    print(json.dumps({"reload": checks["reload"]}), flush=True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(create_app(settings, engine), log_level="error"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started and time.monotonic() < deadline:
                time.sleep(0.01)
            if not server.started:
                raise RuntimeError("真实 HTTP 服务启动失败。")
            base_url = f"http://127.0.0.1:{port}"
            with TypeSafeClient(api_key=settings.api_key, base_url=base_url) as client:
                body = mixed_body()
                result = client.system_one(state=body["state"], questions=body["questions"])
                if (
                    result.model != scorer.model_id
                    or set(result.answers) != set(body["questions"])
                    or result.answers["route"].choice not in body["questions"]["route"]["criteria"]
                    or not 0 <= result.answers["refund"].noul <= 1
                    or not 0 <= result.answers["level"].score <= 2
                    or result.answers["level"].legend[0] != {"rule": "等于 0"}
                    or client.models.list().models[0].name != scorer.model_id
                ):
                    raise ValueError("真实官方 SDK 的模型、问题或等级说明不匹配。")
                checks["official_sdk"] = {
                    "package": "typesafe-sdk",
                    "version": version("typesafe-sdk"),
                    "mixed_primitives": True,
                    "structured_legend": True,
                    "real_model_http": True,
                }
            with httpx.Client(
                base_url=base_url,
                timeout=120,
                headers={"Authorization": f"Bearer {settings.api_key}"},
            ) as client:
                dynamic = []
                for count, target in (
                    (candidate_count, position)
                    for candidate_count in (1, 26, 27, 60, 90, 91, 255)
                    for position in sorted({0, candidate_count // 2, candidate_count - 1})
                ):
                    body = {
                        "model": scorer.model_id,
                        "state": {"target": f"option_{target}"},
                        "questions": {
                            "dynamic": {
                                "type": "choice",
                                "instructions": "Select the option named by target.",
                                "criteria": {f"option_{i}": None for i in range(count)},
                            }
                        },
                    }
                    response = client.post("/v1/systemone", json=body)
                    response.raise_for_status()
                    answer = EvaluationResponse.model_validate(response.json()).answers["dynamic"]
                    if (
                        len(answer.probabilities) != count
                        or abs(sum(answer.probabilities.values()) - 1) > 1e-6
                    ):
                        raise ValueError("真实多候选推理的概率数量或归一化错误。")
                    dynamic.append(
                        {
                            "candidates": count,
                            "target": f"option_{target}",
                            "choice": answer.choice,
                            "exact_match": answer.choice == f"option_{target}",
                            "probability_sum": sum(answer.probabilities.values()),
                        }
                    )
                checks["dynamic_choice"] = dynamic
                checks["latency_local"] = [latency(client, scorer.model_id, n) for n in (1, 8)]
        finally:
            server.should_exit = True
            thread.join(10)
    if benchmark_jev:
        load_dotenv(override=False)
        key = os.getenv("TYPESAFE_API_KEY")
        if not key:
            raise ValueError("缺少 Jev API key。")
        with httpx.Client(
            base_url=os.getenv("TYPESAFE_BASE_URL", "https://api.typesafe.ai").rstrip("/"),
            headers={"Authorization": f"Bearer {key}"},
            timeout=120,
        ) as client:
            checks["latency_jev"] = [latency(client, "jev-1.13.0", n) for n in (1, 8)]
        checks["latency_caveat"] = (
            "Jev measurements include remote HTTPS network and service overhead; "
            "local measurements use loopback."
        )
    checks["passed"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "output": str(output)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--benchmark-jev", action="store_true")
    parser.add_argument("--regression-data", type=Path)
    parser.add_argument("--regression-predictions", type=Path)
    args = parser.parse_args()
    verify_package(
        args.package,
        args.data,
        args.predictions,
        args.output,
        args.benchmark_jev,
        args.regression_data,
        args.regression_predictions,
    )
