# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ScarletKc-Necro contributors

"""准备首轮小 LoRA 数据；保持公开 split、翻译组与对比样本的边界。"""

import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from necro.config import MODEL_ID
from necro.datasets import prepare_public_eval
from necro.evaluation import read_examples


def context_hash(example):
    value = json.dumps(example["request"]["state"], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(value.encode()).hexdigest()


def write_jsonl(path, examples):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples), encoding="utf-8"
    )


def rule_pairs(seed=17):
    rng = random.Random(seed)
    examples = []
    policies = [
        ("refund", "payment_confirmed", "within_return_window", "退款", "refund"),
        ("library", "active_membership", "book_available", "借书", "borrow the book"),
        ("lab", "training_completed", "access_permit_valid", "进入实验室", "enter the lab"),
        ("shipping", "payment_received", "parcel_ready", "发货", "ship the parcel"),
        ("deploy", "tests_passed", "review_approved", "部署", "deploy the change"),
    ]
    for pair in range(100):
        language = "zh" if pair % 2 == 0 else "en"
        if pair < 50:
            domain, first, second, zh_action, en_action = policies[(pair // 2) % len(policies)]
            positive = {first: True, second: True}
            negative = dict(positive)
            negative[rng.choice([first, second])] = False
            question = {
                "type": "noul",
                "instructions": (
                    f"仅当 {first} 和 {second} 都为 true 时允许{zh_action}。现在是否允许？"
                    if language == "zh"
                    else f"Allow {en_action} only when both {first} and {second} are true. "
                    "Is it allowed?"
                ),
            }
            states, expected = [positive, negative], [True, False]
        else:
            domain = "service-impact"
            if pair % 4 < 2:
                states = [
                    {"working": True, "workaround": True},
                    {"working": False, "workaround": True},
                ]
                expected = [0, 1]
            else:
                states = [
                    {"working": False, "workaround": True},
                    {"working": False, "workaround": False},
                ]
                expected = [1, 2]
            question = {
                "type": "score",
                "instructions": (
                    "根据 working（功能正常）和 workaround（存在替代方法）的布尔值评估影响。"
                    if language == "zh"
                    else "Rate impact using the boolean working and workaround fields."
                ),
                "criteria": (
                    [
                        "working 为 true，功能正常",
                        "working 为 false，workaround 为 true",
                        "working 和 workaround 都为 false",
                    ]
                    if language == "zh"
                    else [
                        "working is true",
                        "working is false and workaround is true",
                        "working and workaround are both false",
                    ]
                ),
            }
        for side, (state, answer) in enumerate(zip(states, expected, strict=True)):
            # 不提供标签暗示；无关标识使来源记录可追溯，决策仍只依赖规则中的两个字段。
            state = {**state, "record_id": f"record-{pair:03d}"}
            examples.append(
                {
                    "id": f"rules/{pair}/{side}",
                    "group_id": f"rules/{pair}",
                    "source": "constructed-rules",
                    "language": language,
                    "domain": domain,
                    "request": {
                        "model": MODEL_ID,
                        "state": state,
                        "questions": {"decision": question},
                    },
                    "expected": {"decision": answer},
                }
            )
    return examples


def audit_disjoint(training, validation):
    hashes = {context_hash(row) for row in validation}
    groups = {row.get("group_id", row["id"]) for row in validation}
    if any(context_hash(row) in hashes for row in training):
        raise ValueError("训练与验证上下文重叠。")
    if any(row.get("group_id", row["id"]) in groups for row in training):
        raise ValueError("训练与验证来源组重叠。")


def prepare(output: Path, seed=17):
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "source-train.jsonl"
    val_path = output / "validation.jsonl"
    if not raw_path.exists():
        prepare_public_eval(raw_path, 200, split="train")
    if not val_path.exists():
        prepare_public_eval(val_path, 50, split="validation", start=100)
    raw, validation = read_examples(raw_path), read_examples(val_path)
    rng = random.Random(seed)
    training = []
    for i, source in enumerate(raw):
        row = copy.deepcopy(source)
        question = row["request"]["questions"]["decision"]
        expected = row["expected"]["decision"]
        if "xnli" in row["source"] and i % 4 == 0:
            row["request"]["questions"]["decision"] = {
                "type": "noul",
                "instructions": (
                    "仅凭 premise，是否可以推出 hypothesis？未得到支持时回答否，包括证据不足。"
                    if row["language"] == "zh"
                    else "Does the premise logically entail the hypothesis? "
                    "Answer no when it is not entailed, including when evidence is insufficient."
                ),
            }
            row["expected"]["decision"] = expected == "entailment"
        else:
            keys = list(question["criteria"])
            if "massive" in row["source"] and i % 2 == 0:
                negatives = [key for key in keys if key != expected]
                keys = [expected, *rng.sample(negatives, 7)]
            rng.shuffle(keys)
            question["criteria"] = {key: question["criteria"][key] for key in keys}
        training.append(row)
    training.extend(rule_pairs(seed))
    # 排除验证和旧诊断集的相同文本，避免近乎免费的记忆收益。
    protected = validation[:]
    if Path("data/baseline.jsonl").exists():
        protected.extend(read_examples(Path("data/baseline.jsonl")))
    protected_hashes = {context_hash(row) for row in protected}
    rejected_groups = {
        row.get("group_id", row["id"]) for row in training if context_hash(row) in protected_hashes
    }
    training = [row for row in training if row.get("group_id", row["id"]) not in rejected_groups]
    audit_disjoint(training, validation)
    rng.shuffle(training)
    train_path = output / "train.jsonl"
    write_jsonl(train_path, training)
    manifest = {
        "seed": seed,
        "training_examples": len(training),
        "validation_examples": len(validation),
        "removed_overlap_groups": len(rejected_groups),
        "types": dict(
            Counter(next(iter(row["request"]["questions"].values()))["type"] for row in training)
        ),
        "languages": dict(Counter(row["language"] for row in training)),
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(val_path.read_bytes()).hexdigest(),
        "sources": ["facebook/xnli train", "mteb/amazon_massive_intent train", "constructed-rules"],
        "notes": "Validation uses rows 100–149 in each language/source. No Jev labels.",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/lora-pilot"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), ensure_ascii=False, indent=2))
