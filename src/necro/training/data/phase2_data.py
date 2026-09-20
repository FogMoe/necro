"""第二阶段多任务数据；按内容来源组拆分，登记后不可原地修改。"""

import argparse
import bisect
import hashlib
import json
import operator
import random
from collections import Counter, defaultdict
from pathlib import Path

from necro.config import MODEL_ID
from necro.evaluation import read_examples
from necro.experiment_guard import canonical_request, normalized_text, register
from necro.training.data.training_data import write_jsonl

RAW = Path("data/phase2")
CLEANING = {}


def load(name):
    return [
        item["row"]
        for item in json.loads((RAW / f"{name}.json").read_text(encoding="utf-8"))["rows"]
    ]


def hash_text(text):
    return hashlib.sha256(normalized_text(text).encode()).hexdigest()


def example(
    identifier, group, family, language, state, question, expected, source, quality="human"
):
    return {
        "id": identifier,
        "group_id": group,
        "content_group": group,
        "source": source,
        "family": family,
        "language": language,
        "label_quality": quality,
        "request": {"model": MODEL_ID, "state": state, "questions": {"decision": question}},
        "expected": {"decision": expected},
    }


def grouped_sample(rows, count, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    chosen = []
    for key in keys:
        if len(chosen) >= count:
            break
        chosen.extend(groups[key])
    return chosen


def partition_groups(rows, quotas, seed):
    remaining = rows
    result = {}
    for role, count in quotas.items():
        selected = grouped_sample(remaining, count, seed)
        result[role] = selected
        groups = {row["group_id"] for row in selected}
        remaining = [row for row in remaining if row["group_id"] not in groups]
    return result


def choice(values, gold, rng):
    # 候选先打乱，再赋不含正确性暗示的 key，避免 gold 永远叫 option_0。
    values = list(values)
    rng.shuffle(values)
    criteria = {f"option_{i}": value for i, value in enumerate(values)}
    expected = next(key for key, value in criteria.items() if value == gold)
    keys = list(criteria)
    rng.shuffle(keys)
    return {key: criteria[key] for key in keys}, expected


def paraphrases(split, raw=None):
    raw = (
        raw
        if raw is not None
        else {language: load(f"paws-{language}-{split}") for language in ("en", "zh")}
    )
    english = {row["id"]: row for row in raw["en"]}
    translated = raw["zh"]
    invalid = {
        row["id"]
        for row in [*english.values(), *translated]
        if any(row[key].strip() in {"", "NS"} for key in ("sentence1", "sentence2"))
    }
    CLEANING[f"paws-{split}"] = {"excluded_placeholder_ids": sorted(invalid)}
    rows = []
    for language in ("en", "zh"):
        for row in raw[language]:
            if row["id"] in invalid:
                continue
            original = english[row["id"]]
            group = "paws/" + hash_text(sorted([original["sentence1"], original["sentence2"]]))
            rows.append(
                example(
                    f"paws/{split}/{language}/{row['id']}",
                    group,
                    "paraphrase",
                    language,
                    {"sentence1": row["sentence1"], "sentence2": row["sentence2"]},
                    {
                        "type": "noul",
                        "instructions": "两个句子的含义是否相同？注意实体与关系的方向。"
                        if language == "zh"
                        else "Do both sentences express the same meaning? "
                        "Check the entities and the direction of relationships.",
                    },
                    bool(row["label"]),
                    "google-research-datasets/paws-x",
                )
            )
    return rows


def reading(split, raw=None):
    return [
        example(
            f"boolq/{split}/{hash_text(row['question'])[:16]}",
            "passage/" + hash_text(row["passage"]),
            "reading_boolean",
            "en",
            {"passage": row["passage"], "question": row["question"]},
            {
                "type": "noul",
                "instructions": "Answer the question in state using only the passage. "
                "Use the facts as described in that passage, even if they are historical.",
            },
            row["answer"],
            "google/boolq",
        )
        for row in (raw if raw is not None else load(f"boolq-{split}"))
    ]


def extraction(training, raw=None, max_per_passage=1):
    rng = random.Random(735)
    raw = (
        raw
        if raw is not None
        else {
            "en": load("squad-train" if training else "xquad-en"),
            **({} if training else {"zh": load("xquad-zh")}),
        }
    )
    english = raw["en"]
    by_id = {row["id"]: row for row in english}
    result = []
    for language in ("en",) if training else ("en", "zh"):
        rows = raw[language]
        groups = defaultdict(list)
        for row in rows:
            groups[row["context"]].append(row)
        for context, questions in groups.items():
            if len(context) > 4500:
                continue
            candidates = list(
                dict.fromkeys(
                    text
                    for row in questions
                    for text in row["answers"]["text"]
                    if 0 < len(text) < 160
                )
            )
            produced = 0
            for row in questions:
                aliases = row["answers"]["text"]
                if not aliases:
                    continue
                gold = aliases[0]
                negatives = [
                    text
                    for text in candidates
                    if not any(
                        normalized_text(text) in normalized_text(alias)
                        or normalized_text(alias) in normalized_text(text)
                        for alias in aliases
                    )
                ]
                if len(negatives) < 2 or len(gold) >= 160:
                    continue
                absent = int(hash_text(row["id"])[:8], 16) % 4 == 0
                none_option = (
                    "以上候选均不是正确答案"
                    if language == "zh"
                    else "None of the listed answers is correct"
                )
                real_count = min(3, len(negatives))
                values = (
                    rng.sample(negatives, real_count)
                    if absent
                    else [gold, *rng.sample(negatives, real_count - 1)]
                )
                criteria, expected = choice(
                    [*values, none_option], none_option if absent else gold, rng
                )
                group = "passage/" + hash_text(by_id[row["id"]]["context"])
                result.append(
                    example(
                        f"extract/{language}/{row['id']}",
                        group,
                        "candidate_extraction",
                        language,
                        {"passage": context, "question": row["question"]},
                        {
                            "type": "choice",
                            "instructions": "根据 passage 回答 question，选择正确的候选答案。"
                            if language == "zh"
                            else "Answer the question using the passage. "
                            "Select the correct candidate answer.",
                            "criteria": criteria,
                        },
                        expected,
                        "rajpurkar/squad" if training else "google/xquad",
                        "derived-human",
                    )
                )
                produced += 1
                # 多题仍保留同一段落组，不能虚增独立样本数。
                if max_per_passage is not None and produced >= max_per_passage:
                    break
    return result


def retrieval(split):
    rng = random.Random(815)
    corpus = {str(row["_id"]): row for row in load("scifact-corpus")}
    queries = {str(row["_id"]): row["text"] for row in load("scifact-queries")}
    positives = defaultdict(set)
    for row in load(f"scifact-{split}"):
        if row["score"] > 0:
            positives[str(row["query-id"])].add(str(row["corpus-id"]))
    short_docs = [key for key, row in corpus.items() if len(row["text"]) < 2100]
    result = []
    for qid, relevant in positives.items():
        acceptable = [key for key in sorted(relevant) if key in short_docs]
        if not acceptable:
            continue
        gold = acceptable[0]
        negatives = rng.sample([key for key in short_docs if key not in relevant], 2)
        keys = [gold, *negatives]
        rng.shuffle(keys)
        criteria = {
            key: {"title": corpus[key]["title"], "abstract": corpus[key]["text"]} for key in keys
        }
        if sum(len(value["abstract"]) for value in criteria.values()) > 4700:
            continue
        result.append(
            example(
                f"scifact/{split}/{qid}",
                "scifact-query/" + hash_text(queries[qid]),
                "candidate_retrieval",
                "en",
                {"claim": queries[qid]},
                {
                    "type": "choice",
                    "instructions": "Which candidate scientific document is relevant "
                    "evidence for assessing the claim? Select the most relevant document.",
                    "criteria": criteria,
                },
                gold,
                "BeIR/scifact",
                "qrels-with-unjudged-negatives",
            )
        )
    return result


OPS = {
    "ge": (operator.ge, "不少于", "at least"),
    "gt": (operator.gt, "大于", "greater than"),
    "le": (operator.le, "不多于", "at most"),
    "lt": (operator.lt, "小于", "less than"),
    "eq": (operator.eq, "等于", "equal to"),
}


def numeric_rules(role, groups, *, seed=None, namespace="numeric"):
    rng = random.Random(
        seed
        if seed is not None
        else {"train": 101, "development": 202, "calibration": 303, "test": 404}[role]
    )
    fields = {
        "train": ("amount_paid", "amount_due", "confirmed"),
        "development": ("available_slots", "required_slots", "enabled"),
        "calibration": ("balance", "threshold", "verified"),
        "test": ("stock", "requested", "warehouse_open"),
    }[role]
    rows = []
    for i in range(groups):
        code = list(OPS)[i % len(OPS)]
        fn, zh, en = OPS[code]
        threshold = rng.randrange(10, 800)
        step = 0.5 if i % 3 == 0 else 1
        positive, negative = {
            "ge": (0, -step),
            "gt": (step, 0),
            "le": (0, step),
            "lt": (-step, 0),
            "eq": (0, -step),
        }[code]
        cases = [
            (threshold + positive, threshold, True),
            (threshold + negative, threshold, True),
            (threshold + 10 + positive, threshold + 10, True),
            (threshold + 10 + positive, threshold + 10, False),
        ]
        for language in ("zh", "en"):
            a, b, flag = fields
            negated = i % 6 == 0
            target = (
                ("不合格" if negated else "合格")
                if language == "zh"
                else ("ineligible" if negated else "eligible")
            )
            instruction = (
                f"当且仅当 `{a}` {zh} `{b}` 且 `{flag}` 为 true 时合格；其余或缺少字段时不合格。"
                f"现在是否{target}？"
                if language == "zh"
                else f"The record is eligible if and only if `{a}` is {en} `{b}` and `{flag}` "
                f"is true. Otherwise, or if required fields are missing, it is ineligible. "
                f"Is this record {target}?"
            )
            if role == "test":
                instruction = (
                    f"缺少字段或 `{flag}` 不为 true 时不合格。其余情况下，`{a}` {zh} `{b}` "
                    f"则合格，否则不合格。现在是否{target}？"
                    if language == "zh"
                    else f"Missing fields or `{flag}` not true means ineligible. For all other "
                    f"cases, the record is eligible when `{a}` is {en} `{b}`, and ineligible "
                    f"otherwise. Is this record {target}?"
                )
            variants = cases + ([(threshold, threshold, None)] if i % 4 == 0 else [])
            for side, (left, right, gate) in enumerate(variants):
                state = {a: left, b: right}
                if gate is not None:
                    state[flag] = gate
                eligible = bool(fn(left, right) and gate)
                rows.append(
                    example(
                        f"{namespace}/{role}/{i}/{language}/{side}",
                        f"{namespace}/{role}/{i}",
                        "numeric_rule",
                        language,
                        state,
                        {"type": "noul", "instructions": instruction},
                        not eligible if negated else eligible,
                        "constructed-numeric-v2",
                        "exact-oracle",
                    )
                )
    return rows


def ordinal_rules(role, groups, *, seed=None, namespace="ordinal"):
    rng = random.Random(
        seed
        if seed is not None
        else {"train": 511, "development": 612, "calibration": 713, "test": 814}[role]
    )
    rows = []
    field = {
        "train": "delay",
        "development": "defect_count",
        "calibration": "queue_length",
        "test": "measurement",
    }[role]
    for i in range(groups):
        levels = [3, 4, 5, 7][i % 4]
        thresholds = sorted(rng.sample(range(3, 70), levels - 1))
        boundary = thresholds[i % len(thresholds)]
        extra = thresholds[-1] + 2 if boundary != thresholds[-1] else 0
        values = sorted({boundary - 1, boundary, extra})
        for language in ("en", "zh"):
            criteria = []
            for level in range(levels):
                if level == 0:
                    text = (
                        f"{field} 小于 {thresholds[0]}"
                        if language == "zh"
                        else f"{field} is below {thresholds[0]}"
                    )
                elif level == levels - 1:
                    text = (
                        f"{field} 不少于 {thresholds[-1]}"
                        if language == "zh"
                        else f"{field} is at least {thresholds[-1]}"
                    )
                else:
                    text = (
                        f"{field} 不少于 {thresholds[level - 1]} 且小于 {thresholds[level]}"
                        if language == "zh"
                        else f"{field} is at least {thresholds[level - 1]} "
                        f"and below {thresholds[level]}"
                    )
                criteria.append({"rule": text} if i % 3 == 0 else text)
            for value in values:
                rows.append(
                    example(
                        f"{namespace}/{role}/{i}/{language}/{value}",
                        f"{namespace}/{role}/{i}",
                        "ordinal_rule",
                        language,
                        {field: value},
                        {
                            "type": "score",
                            "instructions": "按给定区间规则选择当前值的等级，注意端点。"
                            if language == "zh"
                            else "Rate the current value using the supplied interval rules. "
                            "Respect the boundaries.",
                            "criteria": criteria,
                        },
                        bisect.bisect_right(thresholds, value),
                        "constructed-ordinal-v2",
                        "exact-oracle",
                    )
                )
    return rows


def prepare(output=Path("data/phase2/experiment-v3")):
    if (output / "experiment.json").exists():
        raise ValueError("实验数据已经登记，不可覆盖。")
    output.mkdir(parents=True, exist_ok=True)
    parts = {role: [] for role in ("train", "development", "calibration", "test", "regression")}
    paws_train = paraphrases("train")
    parts["train"] += grouped_sample([row for row in paws_train if row["language"] == "en"], 160, 1)
    parts["train"] += grouped_sample([row for row in paws_train if row["language"] == "zh"], 480, 2)
    for role, rows in partition_groups(
        paraphrases("validation"), {"development": 100, "calibration": 80}, 3
    ).items():
        parts[role] += rows
    parts["test"] += grouped_sample(paraphrases("test"), 120, 4)
    parts["train"] += grouped_sample(reading("train"), 320, 5)
    for role, rows in partition_groups(
        reading("validation"), {"test": 100, "calibration": 80, "development": 80}, 6
    ).items():
        parts[role] += rows
    parts["train"] += grouped_sample(extraction(True), 160, 7)
    for role, rows in partition_groups(
        extraction(False), {"test": 100, "calibration": 80, "development": 80}, 8
    ).items():
        parts[role] += rows
    sci_train = retrieval("train")
    for role, rows in partition_groups(
        sci_train, {"development": 60, "calibration": 50, "train": 128}, 9
    ).items():
        parts[role] += rows
    parts["test"] += grouped_sample(retrieval("test"), 80, 10)
    for role, count in (("train", 40), ("development", 10), ("calibration", 8), ("test", 16)):
        parts[role] += numeric_rules(role, count)
    for role, count in (("train", 60), ("development", 20), ("calibration", 20), ("test", 40)):
        parts[role] += ordinal_rules(role, count)
    replay = [
        row
        for row in read_examples(Path("data/improvement/expanded/train.jsonl"))
        if row["source"] in {"facebook/xnli", "mteb/amazon_massive_intent"}
    ]
    parts["train"] += grouped_sample(replay, 640, 11)
    parts["development"] += read_examples(Path("data/lora-pilot/validation.jsonl"))
    parts["regression"] += read_examples(Path("data/improvement/test-sealed.jsonl"))
    for rows in parts.values():
        for row in rows:
            if row["source"] in {"facebook/xnli", "mteb/amazon_massive_intent"}:
                row["family"] = "inference" if "xnli" in row["source"] else "intent"
                row["label_quality"] = "human"
                row["content_group"] = "legacy/" + hash_text(row["request"]["state"])
    # 内容重复时优先保留新封存测试，其次校准/开发，剔除较低优先级的整个来源组。
    seen_content, seen_groups, seen_requests, removed = set(), set(), set(), {}
    for role in ("test", "calibration", "development", "regression", "train"):
        rejected = {
            row["group_id"]
            for row in parts[role]
            if row["group_id"] in seen_groups
            or row["content_group"] in seen_content
            or canonical_request(row) in seen_requests
        }
        parts[role] = [row for row in parts[role] if row["group_id"] not in rejected]
        # 同一请求去重；不靠无关 ID 虚增数量。
        unique = {}
        for row in parts[role]:
            unique.setdefault(canonical_request(row), row)
        parts[role] = list(unique.values())
        removed[role] = sorted(rejected)
        seen_groups.update(row["group_id"] for row in parts[role])
        seen_content.update(row["content_group"] for row in parts[role])
        seen_requests.update(canonical_request(row) for row in parts[role])
    from transformers import AutoTokenizer

    from necro.backend import single_token_labels
    from necro.training.trainer import encode_example

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        revision="2fc06364715b967f1860aea9cf38778875588b17",
        local_files_only=True,
    )
    alphabet, _ = single_token_labels(tokenizer)
    too_long = {
        row["group_id"]
        for row in parts["train"]
        if len(encode_example(tokenizer, alphabet, row, max_length=32768)["input_ids"]) > 2048
    }
    parts["train"] = [row for row in parts["train"] if row["group_id"] not in too_long]
    CLEANING["training_length"] = {
        "excluded_groups": sorted(too_long),
        "max_tokens": 2048,
        "policy": "drop whole group; never truncate labeled evidence",
    }
    random.Random(2026).shuffle(parts["train"])
    paths = {}
    for role, rows in parts.items():
        filename = "validation.jsonl" if role == "development" else f"{role}.jsonl"
        paths[role] = output / filename
        write_jsonl(paths[role], rows)
    protocol = {
        "phase": 2,
        "model": "Qwen/Qwen3.5-0.8B",
        "rank": 16,
        "training_ablation": {
            "objectives": ["answer-ce", "candidate-ce"],
            "seed": 2026,
            "learning_rate": 3e-5,
            "batch_size": 4,
            "accumulation": 2,
        },
        "max_initial_training_runs": 2,
        "promotion_min_macro_gain": 0.02,
        "max_legacy_family_regression": 0.03,
        "remote_reference": "jev-1.13.0",
        "notes": "Known phase1 test is regression only. "
        "Qrels negatives are unjudged, not verified irrelevant. "
        "Candidate retrieval is top-1 on sampled documents, not full-corpus NDCG. Synthetic oracle "
        "templates have distinct fields and test wording but share logical task families.",
    }
    manifest = register(output, paths, protocol)
    manifest["cleaning"] = CLEANING
    manifest["data_audit"] = {
        role: {
            "families": dict(Counter(row["family"] for row in rows)),
            "languages": dict(Counter(row["language"] for row in rows)),
            "label_quality": dict(Counter(row["label_quality"] for row in rows)),
            "removed_overlap_groups": removed[role],
        }
        for role, rows in parts.items()
    }
    manifest["sources"] = {
        p.name: {
            key: value
            for key, value in json.loads(p.read_text(encoding="utf-8")).items()
            if key not in {"rows", "features"}
        }
        | {"snapshot_sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in RAW.glob("*.json")
    }
    (output / "experiment.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest["data_audit"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/phase2/experiment-v3"))
    prepare(parser.parse_args().output)
