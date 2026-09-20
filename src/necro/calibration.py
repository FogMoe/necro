"""仅在校准切片上选择温度，再把相同变换应用于已有预测。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from necro.engine import ScoredTask, answer_for, prepare_task
from necro.evaluation import read_examples, summarize
from necro.schema import ChoiceAnswer, EvaluationRequest, EvaluationResponse


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


def transform_report(dataset, predictions_path, output, temperature):
    examples = read_examples(dataset)
    records = read_examples(predictions_path)
    if [row["id"] for row in examples] != [row["id"] for row in records]:
        raise ValueError("预测记录与数据行不一致。")
    responses = []
    for example, record in zip(examples, records, strict=True):
        request = EvaluationRequest.model_validate(example["request"])
        response = EvaluationResponse.model_validate(record["response"])
        for key, question in request.questions.items():
            original = response.answers[key]
            if not isinstance(original, ChoiceAnswer):
                continue
            probabilities = list(original.probabilities.values())
            response.answers[key] = answer_for(
                prepare_task(request.state, question),
                ScoredTask(temper(probabilities, temperature), 0),
            )
        responses.append(response)
    report, rows = summarize(examples, responses)
    metadata = json.loads((predictions_path.parent / "summary.json").read_text(encoding="utf-8"))[
        "metadata"
    ]
    metadata = {
        **metadata,
        "choice_temperature": temperature,
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
