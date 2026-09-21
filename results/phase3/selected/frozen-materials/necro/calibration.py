"""仅在校准切片上选择温度，再把相同变换应用于已有预测。"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from necro.engine import ScoredTask, answer_for, prepare_task
from necro.evaluation import read_examples, summarize
from necro.schema import ChoiceAnswer, EvaluationRequest, EvaluationResponse, NoulAnswer


def temper(probabilities, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("温度必须是大于零的有限数。")
    values = np.asarray(probabilities, dtype=float)
    scores = np.log(np.maximum(values, 1e-30)) / temperature
    weights = np.exp(scores - scores.max())
    return (weights / weights.sum()).tolist()


def fit_temperature(predictions_path):
    rows = read_examples(predictions_path)
    pairs = []
    for row in rows:
        response = EvaluationResponse.model_validate(row["response"])
        for key, gold in row["expected"].items():
            answer = response.answers[key]
            if not isinstance(answer, ChoiceAnswer):
                raise ValueError("此校准器只校准 Choice，不能混入 Noul/Score。")
            probabilities = list(answer.probabilities.values())
            index = list(answer.probabilities).index(str(gold))
            pairs.append((probabilities, index))

    def nll(temperature):
        return float(np.mean([-np.log(max(temper(p, temperature)[i], 1e-30)) for p, i in pairs]))

    candidates = sorted({1.0, *np.exp(np.linspace(np.log(0.5), np.log(4.0), 181)).tolist()})
    temperature = min(candidates, key=nll)
    return {
        "temperature": temperature,
        "before_nll": nll(1.0),
        "after_nll": nll(temperature),
        "questions": len(pairs),
        "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "method": "one Choice temperature; minimum mean NLL on a separate calibration split",
        "bounds": [0.5, 4.0],
    }


def fit_primitive_temperatures(dataset, predictions_path, upper=4.0, objective="nll"):
    """相同来源组先取均值，再按任务等权拟合；校准集之外不选择温度。"""
    from necro.experiment_guard import verify

    if upper not in (4.0, 8.0):
        raise ValueError("校准网格只接受登记过的上界 4 或 8。")
    if objective not in {"nll", "brier"}:
        raise ValueError("校准目标必须是 nll 或 brier。")
    if verify(dataset.parent, "calibration") != dataset.resolve():
        raise ValueError("必须使用登记的校准切片。")
    examples, records = read_examples(dataset), read_examples(predictions_path)
    if [row["id"] for row in examples] != [row["id"] for row in records]:
        raise ValueError("预测记录与校准行不一致。")
    metadata = json.loads((predictions_path.parent / "summary.json").read_text(encoding="utf-8"))[
        "metadata"
    ]
    if metadata.get("dataset_sha256") != hashlib.sha256(dataset.read_bytes()).hexdigest():
        raise ValueError("预测数据哈希与校准切片不一致。")
    if metadata.get("postprocessed") or any(
        metadata.get(key) not in (None, 1, 1.0)
        for key in ("temperature", "choice_temperature", "noul_temperature", "score_temperature")
    ):
        raise ValueError("校准输入必须是温度为 1 的原始预测。")
    pairs = defaultdict(list)
    for example, record in zip(examples, records, strict=True):
        if example["expected"] != record["expected"]:
            raise ValueError("预测中的标签与校准切片不一致。")
        response = EvaluationResponse.model_validate(record["response"])
        for key, gold in record["expected"].items():
            answer = response.answers[key]
            if isinstance(answer, NoulAnswer):
                p, target = [answer.noul, 1 - answer.noul], 0 if gold else 1
            else:
                p = list(answer.probabilities.values())
                target = list(answer.probabilities).index(str(gold))
            pairs[answer.type].append((p, target, example["family"], example["group_id"]))

    result = {}
    grid = sorted({1.0, *np.exp(np.linspace(np.log(0.5), np.log(4.0), 181)).tolist()})
    if upper == 8.0:
        grid = sorted({*grid, *np.exp(np.linspace(np.log(4.0), np.log(8.0), 61)).tolist()})
    for primitive, values in pairs.items():
        groups = {group for _, _, _, group in values}
        if len(groups) < 20:
            raise ValueError(f"{primitive} 独立校准组不足 20。")

        def loss_at(temperature, values=values):
            grouped = defaultdict(list)
            for p, target, family, group in values:
                q = temper(p, temperature)
                loss = (
                    -np.log(max(q[target], 1e-30))
                    if objective == "nll"
                    else sum(value * value for value in q) - 2 * q[target] + 1
                )
                grouped[family, group].append(loss)
            families = defaultdict(list)
            for (family, _), losses in grouped.items():
                families[family].append(float(np.mean(losses)))
            return float(np.mean([np.mean(losses) for losses in families.values()]))

        temperature = min(grid, key=loss_at)
        result[primitive] = {
            "temperature": temperature,
            f"before_{objective}": loss_at(1.0),
            f"after_{objective}": loss_at(temperature),
            "questions": len(values),
            "source_groups": len(groups),
            "at_search_boundary": temperature in (grid[0], grid[-1]),
        }
    return {
        "temperatures": {key: value["temperature"] for key, value in result.items()},
        "primitives": result,
        "method": f"per primitive; mean family {objective.upper()} "
        "after averaging within source groups",
        "objective": objective,
        "bounds": [0.5, upper],
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
    }


def transform_report(dataset, predictions_path, output, temperature):
    examples = read_examples(dataset)
    records = read_examples(predictions_path)
    if [row["id"] for row in examples] != [row["id"] for row in records]:
        raise ValueError("预测记录与数据行不一致。")
    responses = []
    temperatures = temperature if isinstance(temperature, dict) else {"choice": temperature}
    metadata = json.loads((predictions_path.parent / "summary.json").read_text(encoding="utf-8"))[
        "metadata"
    ]
    if metadata.get("dataset_sha256") != hashlib.sha256(dataset.read_bytes()).hexdigest():
        raise ValueError("预测数据哈希与变换目标不一致。")
    if metadata.get("postprocessed") or any(
        metadata.get(key) not in (None, 1, 1.0)
        for key in ("temperature", "choice_temperature", "noul_temperature", "score_temperature")
    ):
        raise ValueError("概率变换必须以温度 1 的原始预测为输入，不能重复校准。")
    registry = dataset.parent / "experiment.json"
    if registry.exists():
        manifest = json.loads(registry.read_text(encoding="utf-8"))
        if (
            "test" in manifest["partitions"]
            and (dataset.parent / manifest["partitions"]["test"]["path"]).resolve()
            == dataset.resolve()
        ):
            from necro.experiment_guard import verify

            verify(dataset.parent, "test", remote_reference=True)
            selected = json.loads((dataset.parent / "selection.json").read_text(encoding="utf-8"))
            if selected.get("temperatures") != temperatures:
                raise ValueError("测试集只能应用已冻结的温度。")
            if metadata.get("adapter_weights_sha256") != selected["weights_sha256"]:
                raise ValueError("测试预测不是冻结的模型权重。")
    for example, record in zip(examples, records, strict=True):
        if example["expected"] != record["expected"]:
            raise ValueError("预测标签与变换目标不一致。")
        request = EvaluationRequest.model_validate(example["request"])
        response = EvaluationResponse.model_validate(record["response"])
        for key, question in request.questions.items():
            original = response.answers[key]
            if original.type not in temperatures:
                continue
            probabilities = (
                [original.noul, 1 - original.noul]
                if isinstance(original, NoulAnswer)
                else list(original.probabilities.values())
            )
            response.answers[key] = answer_for(
                prepare_task(request.state, question),
                ScoredTask(temper(probabilities, temperatures[original.type]), 0),
            )
        responses.append(response)
    report, rows = summarize(examples, responses)
    metadata = {
        **metadata,
        **{f"{primitive}_temperature": value for primitive, value in temperatures.items()},
        "postprocessed": True,
        "source_predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
    }
    report = {"metadata": metadata, **report}
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "judgments.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    (output / "predictions.jsonl").write_text(
        "".join(
            json.dumps(
                {"id": ex["id"], "response": response.model_dump(), "expected": ex["expected"]},
                ensure_ascii=False,
            )
            + "\n"
            for ex, response in zip(examples, responses, strict=True)
        ),
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = fit_temperature(args.predictions)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
