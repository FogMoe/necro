# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ScarletKc-Necro contributors

"""小规模改进的数据准备；封存测试集不参与训练和模型选择。"""

import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from necro.config import MODEL_ID
from necro.datasets import prepare_public_eval
from necro.evaluation import read_examples
from necro.training_data import audit_disjoint, context_hash, rule_pairs, write_jsonl


def download_sources(root=Path("data/improvement")):
    root.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("source-expanded.jsonl", 600, "train", 0),
        ("calibration.jsonl", 50, "validation", 200),
        ("test.jsonl", 100, "test", 0),
    ]

    def fetch(job):
        name, count, split, start = job
        path = root / name
        if not path.exists():
            return prepare_public_eval(path, count, split, start)
        return {"cached": str(path)}

    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(fetch, jobs):
            print(json.dumps(result), flush=True)


def uncertainty_triples():
    examples = []
    templates = [
        (
            "owns a red bicycle",
            "does not own a red bicycle",
            "owns a car",
            "有一辆红色自行车",
            "没有红色自行车",
            "有一辆汽车",
        ),
        (
            "works at the museum",
            "does not work at the museum",
            "works on Sundays",
            "在博物馆工作",
            "不在博物馆工作",
            "周日上班",
        ),
        ("ordered tea", "did not order tea", "ordered a cake", "点了茶", "没有点茶", "点了蛋糕"),
        (
            "speaks Spanish",
            "does not speak Spanish",
            "speaks German",
            "会说西班牙语",
            "不会说西班牙语",
            "会说德语",
        ),
        ("has a dog", "does not have a dog", "has a cat", "养了一只狗", "没有养狗", "养了一只猫"),
        (
            "visited Rome last year",
            "did not visit Rome last year",
            "visited Paris last year",
            "去年去过罗马",
            "去年没去过罗马",
            "去年去过巴黎",
        ),
    ]
    criteria = {
        "entailment": "The premise supports the hypothesis.",
        "neutral": "The premise neither supports nor contradicts the hypothesis.",
        "contradiction": "The premise contradicts the hypothesis.",
    }
    for i in range(60):
        language = "zh" if i % 2 == 0 else "en"
        template = templates[(i // 2) % len(templates)]
        name = (
            ["安琪", "小林", "文森", "李明", "瑞秋"][i // 12]
            if language == "zh"
            else ["Mira", "Leo", "Nora", "Sam", "Robin"][i // 12]
        )
        facts = template[3:] if language == "zh" else template[:3]

        def sentence(fact, name=name, language=language):
            return f"{name}{fact}。" if language == "zh" else f"{name} {fact}."

        for relation, fact in zip(("entailment", "contradiction", "neutral"), facts, strict=True):
            examples.append(
                {
                    "id": f"uncertainty/{i}/{relation}",
                    "group_id": f"uncertainty/{i}",
                    "source": "constructed-uncertainty",
                    "language": language,
                    "request": {
                        "model": MODEL_ID,
                        "state": {"premise": sentence(facts[0]), "hypothesis": sentence(fact)},
                        "questions": {
                            "decision": {
                                "type": "choice",
                                "criteria": criteria,
                                "instructions": "仅根据前提判断假设。未提及的事实不一定为假。"
                                if language == "zh"
                                else "Judge the hypothesis using only the premise. "
                                "Unmentioned facts are not necessarily false.",
                            }
                        },
                    },
                    "expected": {"decision": relation},
                }
            )
    return examples


def prepare_improvement(root=Path("data/improvement"), seed=29):
    rng = random.Random(seed)
    raw = read_examples(root / "source-expanded.jsonl")
    pilot = read_examples(Path("data/lora-pilot/train.jsonl"))
    dev = read_examples(Path("data/lora-pilot/validation.jsonl"))
    old = read_examples(Path("data/baseline.jsonl"))
    calibration = read_examples(root / "calibration.jsonl")
    test = read_examples(root / "test.jsonl")
    # 既有 adapter 已见过 pilot，因此先剔除测试集中与训练候选、开发/校准相同的来源组。
    seen = {context_hash(row) for row in raw + pilot + dev + old + calibration}
    excluded_test = {row["group_id"] for row in test if context_hash(row) in seen}
    test = [row for row in test if row["group_id"] not in excluded_test]
    seen_pilot = {context_hash(row) for row in pilot}
    excluded_calibration = {
        row["group_id"] for row in calibration if context_hash(row) in seen_pilot
    }
    calibration = [row for row in calibration if row["group_id"] not in excluded_calibration]
    protected = {context_hash(row) for row in dev + old + calibration + test}
    excluded_train = {row["group_id"] for row in raw if context_hash(row) in protected}
    raw = [row for row in raw if row["group_id"] not in excluded_train]
    nli = defaultdict(list)
    intent = defaultdict(list)
    for row in raw:
        if "xnli" in row["source"]:
            nli[row["language"], row["expected"]["decision"]].append(row)
        else:
            intent[row["language"]].append(row)
    per_class = min(180, min(map(len, nli.values())))
    training = []
    for rows in nli.values():
        rng.shuffle(rows)
        training.extend(copy.deepcopy(rows[:per_class]))
    # 保留少量真实语义 Noul，训练否与信息不足的关系。
    for source in rng.sample(training, 100):
        row = copy.deepcopy(source)
        row["id"] += "/noul"
        row["request"]["questions"]["decision"] = {
            "type": "noul",
            "instructions": "前提是否足以推出假设？证据不足时回答否。"
            if row["language"] == "zh"
            else "Does the premise entail the hypothesis? Answer no if evidence is insufficient.",
        }
        row["expected"]["decision"] = row["expected"]["decision"] == "entailment"
        training.append(row)
    for rows in intent.values():
        rng.shuffle(rows)
        for i, source in enumerate(rows[:400]):
            row = copy.deepcopy(source)
            question = row["request"]["questions"]["decision"]
            if i % 4:
                target = row["expected"]["decision"]
                others = [key for key in question["criteria"] if key != target]
                keys = [target, *rng.sample(others, 7)]
                question["criteria"] = {key: question["criteria"][key] for key in keys}
            training.append(row)
    training.extend(rule_pairs(seed))
    training.extend(uncertainty_triples())
    for row in training:
        question = row["request"]["questions"]["decision"]
        if question["type"] == "choice":
            keys = list(question["criteria"])
            rng.shuffle(keys)
            question["criteria"] = {key: question["criteria"][key] for key in keys}
    for holdout in (dev, calibration, test):
        audit_disjoint(training, holdout)
    audit_disjoint(pilot, test)
    rng.shuffle(training)
    output = root / "expanded"
    output.mkdir(exist_ok=False)
    for path, rows in [
        (output / "train.jsonl", training),
        (output / "validation.jsonl", dev),
        (root / "calibration-clean.jsonl", calibration),
        (root / "test-sealed.jsonl", test),
    ]:
        write_jsonl(path, rows)
    manifest = {
        "seed": seed,
        "training_examples": len(training),
        "nli_per_language_class": per_class,
        "types": dict(
            Counter(next(iter(row["request"]["questions"].values()))["type"] for row in training)
        ),
        "languages": dict(Counter(row["language"] for row in training)),
        "calibration_examples": len(calibration),
        "test_examples": len(test),
        "excluded_test_groups": sorted(excluded_test),
        "excluded_train_groups": sorted(excluded_train),
        "train_sha256": hashlib.sha256((output / "train.jsonl").read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256((root / "test-sealed.jsonl").read_bytes()).hexdigest(),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    download_sources()
    print(json.dumps(prepare_improvement(), ensure_ascii=False, indent=2))
