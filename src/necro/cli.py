import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

from necro.config import MODEL_ID, Settings
from necro.schema import EvaluationRequest


def print_json(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-0.8B 本地判断与评测")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="启动 TypeSafe 格式的本地 HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    score = commands.add_parser("score", help="读取一个 JSON 请求并输出响应")
    score.add_argument("input", type=Path)
    prepare = commands.add_parser("prepare-eval", help="下载中英文 XNLI/MASSIVE 验证切片")
    prepare.add_argument("--output", type=Path, default=Path("data/baseline.jsonl"))
    prepare.add_argument("--per-language", type=int, default=100)
    evaluate = commands.add_parser("evaluate", help="在本地或 Jev 上运行同一份评测")
    evaluate.add_argument("dataset", type=Path)
    evaluate.add_argument("--backend", choices=["local", "jev"], default="local")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--output", type=Path, default=Path("results/local"))
    benchmark = commands.add_parser("benchmark", help="测量预热后的本机请求延迟")
    benchmark.add_argument("--repeats", type=int, default=10)
    benchmark.add_argument("--questions", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = Settings.from_env()
    try:
        if args.command == "prepare-eval":
            from necro.datasets import prepare_public_eval

            print_json(prepare_public_eval(args.output, args.per_language))
            return
        if args.command == "evaluate":
            from necro.evaluation import run_evaluation

            if args.limit is not None and args.limit < 1:
                parser.error("--limit 必须大于 0")
            print_json(
                run_evaluation(args.dataset, args.output, settings, args.backend, args.limit)
            )
            return
        from necro.backend import TransformersScorer
        from necro.engine import DecisionEngine

        scorer = TransformersScorer(settings)
        scorer.load()
        engine = DecisionEngine(scorer)
        if args.command == "serve":
            import uvicorn

            from necro.api import create_app

            uvicorn.run(create_app(settings, engine), host=args.host, port=args.port)
        elif args.command == "score":
            request = EvaluationRequest.model_validate_json(args.input.read_text(encoding="utf-8"))
            print_json(engine.evaluate(request).model_dump())
        elif args.command == "benchmark":
            import torch

            if args.repeats < 1 or args.questions < 1:
                parser.error("--repeats 和 --questions 必须大于 0")
            question = {
                "type": "choice",
                "instructions": "Which team should handle this request?",
                "criteria": {
                    "billing": "Payments and refunds",
                    "technical": "Software bugs",
                    "sales": "New product purchases",
                },
            }
            request = EvaluationRequest.model_validate(
                {
                    "state": "I was charged twice. Please refund the duplicate payment.",
                    "model": MODEL_ID,
                    "questions": {f"q{i}": question for i in range(args.questions)},
                }
            )
            engine.evaluate(request)
            latencies = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                response = engine.evaluate(request)
                latencies.append(time.perf_counter() - start)
            p95 = (
                statistics.quantiles(latencies, n=100, method="inclusive")[94]
                if len(latencies) > 1
                else latencies[0]
            )
            result = {
                "repeats": args.repeats,
                "questions_per_request": args.questions,
                "p50_ms": statistics.median(latencies) * 1000,
                "p95_ms": p95 * 1000,
                "input_tokens_per_request": response.usage.input_tokens,
                "load_seconds": scorer.load_seconds,
                "latencies_seconds": latencies,
                "answer": response.answers["q0"].model_dump(),
                "candidate_mass": scorer.last_candidate_masses[0],
            }
            if scorer.device.type == "cuda":
                result["gpu"] = torch.cuda.get_device_name()
                result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            print_json(result)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"错误：{error}\n")


if __name__ == "__main__":
    main()
