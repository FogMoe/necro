import hashlib
import json
import math
import os
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np

from necro.config import Settings
from necro.engine import PROMPT_VERSION, DecisionEngine
from necro.schema import EvaluationRequest, EvaluationResponse, NoulAnswer, ScoreAnswer


def read_examples(path: Path, limit: int | None = None):
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return records[:limit] if limit is not None else records


def metrics(rows):
    if not rows:
        return {"count": 0}
    correct = np.array([row["correct"] for row in rows], dtype=float)
    confidence = np.array([row["top_probability"] for row in rows])
    ece = 0.0
    for i in range(10):
        selected = (confidence >= i / 10) & (
            (confidence < (i + 1) / 10) if i < 9 else (confidence <= 1)
        )
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    result = {
        "count": len(rows),
        "accuracy": float(correct.mean()),
        "nll": float(np.mean([row["nll"] for row in rows])),
        "brier": float(np.mean([row["brier"] for row in rows])),
        "ece_10_bins": float(ece),
    }
    scored = [row for row in rows if "score_absolute_error" in row]
    if scored:
        result["score_mae"] = float(np.mean([row["score_absolute_error"] for row in scored]))
        if all("score_normalized_error" in row for row in scored):
            result["score_normalized_mae"] = float(
                np.mean([row["score_normalized_error"] for row in scored])
            )
    if all("gold_probability" in row for row in rows):
        result["nll_floor_sensitivity"] = {
            str(floor): float(
                np.mean([-math.log(max(row["gold_probability"], floor)) for row in rows])
            )
            for floor in (1e-12, 1e-3, 0.005, 0.01)
        }
    return result


def summarize(examples, responses):
    rows = []
    for example, response in zip(examples, responses, strict=True):
        for key, expected in example["expected"].items():
            answer = response.answers[key]
            if isinstance(answer, NoulAnswer):
                keys, probabilities = [True, False], [answer.noul, 1 - answer.noul]
            else:
                keys = list(answer.probabilities)
                probabilities = list(answer.probabilities.values())
                expected = str(expected)
            target = keys.index(expected)
            prediction = keys[int(np.argmax(probabilities))]
            row = {
                "id": example["id"],
                "question_id": key,
                "language": example.get("language", "unknown"),
                "source": example.get("source", "custom"),
                "family": example.get("family", example.get("source", "custom")),
                "group_id": example.get("group_id", example["id"]),
                "label_quality": example.get("label_quality", "unspecified"),
                "type": answer.type,
                "expected": expected,
                "predicted": prediction,
                "correct": prediction == expected,
                "top_probability": max(probabilities),
                "nll": -math.log(max(probabilities[target], 1e-12)),
                "gold_probability": probabilities[target],
                "brier": sum((p - (i == target)) ** 2 for i, p in enumerate(probabilities)),
            }
            if isinstance(answer, ScoreAnswer):
                row["score_absolute_error"] = abs(answer.score - int(expected))
                row["score_normalized_error"] = row["score_absolute_error"] / (len(keys) - 1)
            rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        groups[f"{row['source']}/{row['language']}/{row['type']}"].append(row)
    summary = {
        "overall": metrics(rows),
        "groups": {key: metrics(value) for key, value in groups.items()},
    }
    families = defaultdict(list)
    for row in rows:
        families[row["family"]].append(row)
    summary["families"] = {family: metrics(items) for family, items in families.items()}
    summary["macro_family_accuracy"] = float(
        np.mean([value["accuracy"] for value in summary["families"].values()])
    )
    scored = [row["score_absolute_error"] for row in rows if "score_absolute_error" in row]
    if scored:
        summary["score_mae"] = float(np.mean(scored))
    return summary, rows


def evaluate_remote(requests: list[EvaluationRequest], cache_dir: Path | None = None, stats=None):
    key = os.getenv("TYPESAFE_API_KEY", "")
    if not key:
        raise ValueError("请先在 .env 设置 TYPESAFE_API_KEY。")
    base_url = os.getenv("TYPESAFE_BASE_URL", "https://api.typesafe.ai").rstrip("/")
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(
        base_url=base_url, headers={"Authorization": f"Bearer {key}"}, timeout=60
    ) as client:

        def call(request):
            payload = request.model_dump()
            payload["model"] = "jev-1.13.0"
            # 保留 criteria 顺序；选项重排是不同的实验请求。
            fingerprint = hashlib.sha256(
                json.dumps(payload, ensure_ascii=False).encode()
            ).hexdigest()
            cache = cache_dir / f"{fingerprint}.json" if cache_dir else None
            if cache and cache.exists():
                saved = json.loads(cache.read_text(encoding="utf-8"))
                if saved["request_sha256"] != fingerprint:
                    raise ValueError("Jev 缓存请求指纹不一致。")
                response = EvaluationResponse.model_validate(saved["response"])
                if response.model != "jev-1.13.0" or set(response.answers) != set(
                    request.questions
                ):
                    raise ValueError("Jev 缓存模型或问题不一致。")
                return response, True
            for attempt in range(4):
                response = client.post("/v1/systemone", json=payload)
                if response.status_code in {429, 529, 503} and attempt < 3:
                    try:
                        delay = max(2**attempt, float(response.headers.get("Retry-After", "0")))
                    except ValueError:
                        delay = 2**attempt
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                answer = EvaluationResponse.model_validate(response.json())
                if answer.model != "jev-1.13.0" or set(answer.answers) != set(request.questions):
                    raise ValueError("Jev 返回的模型或问题与固定评测协议不一致。")
                if cache:
                    temporary = cache.with_suffix(f".{uuid.uuid4().hex}.tmp")
                    temporary.write_text(
                        json.dumps(
                            {
                                "request_sha256": fingerprint,
                                "response": answer.model_dump(),
                                "created_at": datetime.now(UTC).isoformat(),
                            },
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    temporary.replace(cache)
                return answer, False
            raise RuntimeError("远程请求重试失败。")

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(call, requests))
        if stats is not None:
            stats.update({"hits": sum(hit for _, hit in results), "requests": len(results)})
        return [response for response, _ in results]


def run_evaluation(path: Path, output: Path, settings: Settings, backend="local", limit=None):
    registry = path.parent / "experiment.json"
    if registry.is_file():
        from necro.experiment_guard import digest, verify, verify_temperatures

        manifest = json.loads(registry.read_text(encoding="utf-8"))
        for role, partition in manifest["partitions"].items():
            if (path.parent / partition["path"]).resolve() == path.resolve():
                verify(
                    path.parent,
                    role,
                    Path(settings.adapter) if settings.adapter else None,
                    remote_reference=backend == "jev",
                )
                if role == "test" and backend == "local":
                    selected = json.loads(
                        (path.parent / "selection.json").read_text(encoding="utf-8")
                    )
                    actual = {
                        primitive: settings.temperature_for(primitive)
                        for primitive in ("choice", "noul", "score")
                    }
                    verify_temperatures(
                        selected,
                        digest(Path(settings.adapter) / "adapter_model.safetensors"),
                        actual,
                    )
                break
        else:
            raise ValueError("实验目录中的评测文件尚未登记。")
    all_examples = read_examples(path)
    examples = all_examples[:limit] if limit is not None else all_examples
    if not examples:
        raise ValueError("评测集为空。")
    requests = [EvaluationRequest.model_validate(example["request"]) for example in examples]
    scorer = None
    if backend == "local":
        from necro.backend import TransformersScorer

        scorer = TransformersScorer(settings)
        scorer.load()
        engine = DecisionEngine(scorer)
        engine.evaluate(requests[0])  # 单独预热，计时不含加载和首轮初始化。
    started = time.perf_counter()
    cache_stats = {}
    responses = (
        engine.evaluate_many(requests)
        if backend == "local"
        else evaluate_remote(requests, output / "remote-cache", cache_stats)
    )
    elapsed = time.perf_counter() - started
    summary, rows = summarize(examples, responses)
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "backend": backend,
        "model": sorted({response.model for response in responses}),
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "evaluated_examples": len(examples),
        "full_dataset_examples": len(all_examples),
        "full_dataset_evaluated": len(examples) == len(all_examples),
        "evaluated_questions": len(rows),
        "elapsed_seconds": elapsed,
        "questions_per_second": len(rows) / elapsed,
        "input_tokens": sum(response.usage.input_tokens for response in responses),
        "notes": "Evaluation results for the supplied dataset and recorded model configuration.",
    }
    if backend == "jev":
        metadata["remote_cache"] = cache_stats
        if cache_stats.get("hits"):
            metadata["questions_per_second"] = None
            metadata["timing_scope"] = (
                "Elapsed time includes cached response reuse and remaining remote requests."
            )
    if scorer is not None:
        import torch
        import transformers

        metadata.update(
            {
                "checkpoint": settings.checkpoint,
                "adapter": settings.adapter,
                "prompt_version": PROMPT_VERSION,
                "revision": getattr(scorer.model.config, "_commit_hash", None),
                "temperature": settings.temperature,
                "choice_temperature": settings.choice_temperature,
                "noul_temperature": settings.noul_temperature,
                "score_temperature": settings.score_temperature,
                "batch_size": settings.batch_size,
                "batch_tokens": settings.batch_tokens,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "device": str(scorer.device),
                "candidate_mass_mean": float(np.mean(scorer.last_candidate_masses)),
                "candidate_mass_min": min(scorer.last_candidate_masses),
            }
        )
        if settings.adapter:
            metadata["adapter_weights_sha256"] = hashlib.sha256(
                (Path(settings.adapter) / "adapter_model.safetensors").read_bytes()
            ).hexdigest()
        if scorer.device.type == "cuda":
            metadata["gpu"] = torch.cuda.get_device_name()
            metadata["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    output.mkdir(parents=True, exist_ok=True)
    report = {"metadata": metadata, **summary}
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    predictions = [
        {"id": example["id"], "response": response.model_dump(), "expected": example["expected"]}
        for example, response in zip(examples, responses, strict=True)
    ]
    for filename, data in (("predictions.jsonl", predictions), ("judgments.jsonl", rows)):
        (output / filename).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data),
            encoding="utf-8",
        )
    return report
