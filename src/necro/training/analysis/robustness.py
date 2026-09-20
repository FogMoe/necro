"""最终题的配对扰动，只用于冻结后的稳健性报告。"""

import argparse
import copy
import json
import random
from collections import defaultdict
from pathlib import Path

from necro.evaluation import metrics, read_examples
from necro.experiment_guard import digest, register
from necro.training.data.training_data import write_jsonl


def negate_instructions(family, language, instruction):
    """直接改变所问条件，避免把相反的回答指令嵌在同一个提示中。"""
    if family == "paraphrase":
        return (
            "两个句子的含义是否不同？注意实体与关系的方向。"
            if language == "zh"
            else (
                "Do the sentences differ in meaning? Check the entities and the direction "
                "of relationships."
            )
        )
    if family == "reading_boolean":
        return (
            "Based only on state.passage, would 'no' be the correct answer to state.question? "
            "Use the facts as described in the passage, even if they are historical."
        )
    if family == "numeric_rule":
        endings = (
            ("现在是否合格？", "现在是否不合格？")
            if language == "zh"
            else ("Is this record eligible?", "Is this record ineligible?")
        )
        for i, ending in enumerate(endings):
            if instruction.endswith(ending):
                return instruction[: -len(ending)] + endings[1 - i]
    raise ValueError("此题没有经过定义的自然否定变换。")


def prepare(source: Path, output: Path, *, role="test"):
    if role not in {"test", "development"}:
        raise ValueError("配对扰动仅支持开发或最终评测。")
    if output.exists():
        raise ValueError("稳健性数据已存在，不可覆盖。")
    groups = defaultdict(list)
    for row in read_examples(source):
        groups[row["family"], row["language"]].append(row)
    rng, variants = random.Random(99172026), []
    for (family, language), rows in sorted(groups.items()):
        rng.shuffle(rows)
        for original in rows[:10]:
            row = copy.deepcopy(original)
            key, question = next(iter(row["request"]["questions"].items()))
            primitive = question["type"]
            if primitive == "choice":
                question["criteria"] = dict(reversed(list(question["criteria"].items())))
                variant = "candidate_order"
            elif primitive == "noul":
                question["instructions"] = negate_instructions(
                    family, language, question["instructions"]
                )
                row["expected"][key] = not row["expected"][key]
                variant = "question_negation"
            else:
                # Score 的等级顺序具有语义，不做候选反转。
                state = row["request"]["state"]
                row["request"]["state"] = {
                    **state,
                    "irrelevant_archive_notes": "Routine archive entry; no measurements. " * 150,
                }
                variant = "irrelevant_long_context"
            row["parent_id"] = original["id"]
            row["id"] = original["id"] + "/" + variant
            row["variant"] = variant
            variants.append(row)
    output.mkdir(parents=True)
    path = output / f"{role}.jsonl"
    write_jsonl(path, variants)
    register(
        output,
        {role: path},
        {
            f"parent_{role}_sha256": digest(source),
            "seed": 99172026,
            "notes": "Perturbations are paired with their original final-test examples. "
            "Numeric-rule parents already include native negation and explicit missing-field "
            "policy.",
        },
    )
    print(json.dumps({"examples": len(variants), "output": str(path)}))


def summarize(dataset, parent_results, variant_results):
    examples = {row["id"]: row for row in read_examples(dataset)}
    parents = {row["id"]: row for row in read_examples(parent_results / "judgments.jsonl")}
    changed = read_examples(variant_results / "judgments.jsonl")
    groups = defaultdict(list)
    for row in changed:
        ex = examples[row["id"]]
        parent = parents[ex["parent_id"]]
        target = (
            not parent["predicted"] if ex["variant"] == "question_negation" else parent["predicted"]
        )
        groups[f"{ex['variant']}/{ex['family']}/{ex['language']}"].append(
            {
                **row,
                "consistent": row["predicted"] == target,
                "parent_correct": parent["correct"],
            }
        )
    return {
        key: {
            **metrics(rows),
            "consistency": sum(r["consistent"] for r in rows) / len(rows),
            "parent_accuracy": sum(r["parent_correct"] for r in rows) / len(rows),
        }
        for key, rows in groups.items()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=["test", "development"], default="test")
    args = parser.parse_args()
    prepare(args.source, args.output, role=args.role)
