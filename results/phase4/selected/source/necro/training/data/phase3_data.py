"""覆盖审计后的第三阶段：完整快照抽样，保护所有历史题，保持八任务口径。"""

import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from necro.datasets import INTENTS
from necro.evaluation import read_examples
from necro.experiment_guard import canonical_request, digest, register
from necro.training.data.full_snapshots import ROOT as SNAPSHOTS
from necro.training.data.full_snapshots import rows as raw_rows
from necro.training.data.phase2_data import (
    example,
    extraction,
    numeric_rules,
    ordinal_rules,
    paraphrases,
    reading,
)
from necro.training.data.source_isolation import Sources, signature, unseen_retrieval
from necro.training.data.training_data import write_jsonl

ROOT = Path("data/phase3/coverage-v1")
NLI = {
    "entailment": "The premise supports the hypothesis.",
    "neutral": "The premise neither supports nor contradicts the hypothesis.",
    "contradiction": "The premise contradicts the hypothesis.",
}
# 修正训练原文已能证伪的人工描述；不读取旧测试错题来修改标准。
INTENT_CRITERIA = {
    **INTENTS,
    "cooking_query": "Ask a general cooking question, such as meal ideas, ingredient "
    "substitutions or cooking methods",
}


def historical():
    paths = sorted(p for p in Path("data").rglob("*.jsonl") if "phase3" not in p.parts)
    result = []
    for path in paths:
        for row in read_examples(path):
            if "group_id" not in row:
                parts = row["id"].split("/")
                if row["source"] == "facebook/xnli":
                    row["group_id"] = f"xnli/{parts[-2]}/{parts[-1]}"
                elif row["source"] == "mteb/amazon_massive_intent":
                    row["group_id"] = f"massive/{parts[-1]}"
                else:
                    row["group_id"] = row["id"]
            result.append(row)
    return result, {str(p): digest(p) for p in paths}


class SourceIndex(Sources):
    """补充跨 ID 重复话语、问题、翻译关系；旧格式保持可审计。"""

    def keys(self, row):
        keys = set(row.get("source_keys", ())) or super().keys(row)
        source = row["source"]
        state = row["request"]["state"]
        if source == "mteb/amazon_massive_intent":
            keys.add("massive-utterance/" + signature(state))
        elif source == "facebook/xnli":
            keys.add("xnli-local-premise/" + signature(state["premise"]))
        elif source in {"google/boolq", "rajpurkar/squad", "google/xquad"}:
            keys.add("passage/" + signature(state["passage"]))
            keys.add("reading-question/" + signature(state["question"]))
        elif source == "BeIR/scifact":
            keys.add("scifact-query/" + signature(state["claim"]))
        keys.add("request/" + signature(canonical_request(row)))
        return keys


def paired_public(name, split, pool_limit=None):
    """XNLI 行号平行；MASSIVE 按上游 ID 平行。抽样单位是原始双语组。"""
    selected = None
    if pool_limit:
        manifest = json.loads((SNAPSHOTS / name / "manifest.json").read_text())
        count = sum(
            f["rows"] for f in manifest["files"] if f["config"] == "en" and f["split"] == split
        )
        selected = set(random.Random(92031).sample(range(count), min(count, pool_limit)))
    english = {
        i: row for i, row in raw_rows(name, "en", split) if selected is None or i in selected
    }
    translated = {
        i: row
        for i, row in raw_rows(name, "zh-CN" if name == "massive" else "zh", split)
        if selected is None or i in selected
    }
    if english.keys() != translated.keys():
        raise ValueError("平行快照行数不匹配。")
    result = []
    for i, en in english.items():
        zh = translated[i]
        if en["label"] != zh["label"] or (name == "massive" and en["id"] != zh["id"]):
            raise ValueError("双语来源或标签不匹配。")
        group = f"massive/{en['id']}" if name == "massive" else f"xnli/{split}/{i}"
        for language, row in (("en", en), ("zh", zh)):
            if name == "massive":
                state, criteria, gold = row["text"], INTENT_CRITERIA, row["label"]
                instruction = (
                    "选择用户最主要的意图。"
                    if language == "zh"
                    else "Select the user's primary intent."
                )
                family, source = "intent", "mteb/amazon_massive_intent"
                keys = [group]
            else:
                state = {"premise": row["premise"], "hypothesis": row["hypothesis"]}
                criteria, gold = NLI, list(NLI)[row["label"]]
                instruction = (
                    "根据 premise 判断 hypothesis 是否成立，仅依据给定前提，不补充外部事实。"
                    if language == "zh"
                    else "Does the premise support, contradict, or leave the "
                    "hypothesis undetermined? Use only the given premise."
                )
                family, source = "inference", "facebook/xnli"
                keys = ["xnli-premise/" + signature(en["premise"])]
            item = example(
                f"p3/{name}/{split}/{i}/{language}",
                group,
                family,
                language,
                state,
                {"type": "choice", "instructions": instruction, "criteria": criteria},
                gold,
                source,
            )
            item["source_keys"] = keys
            item["official_split"] = split
            result.append(item)
    return result


def groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    return grouped


def take(rows, index, blocked, count, seed, *, per_class=False):
    """随机打散单个来源组，并在选中后排除该组的所有共享内容。"""
    candidates = list(groups(rows).values())
    random.Random(seed).shuffle(candidates)
    result, counts = [], Counter()
    for items in candidates:
        key = items[0]["expected"]["decision"] if per_class else "all"
        if counts[key] >= count:
            continue
        keys = index.all_keys(items)
        if keys & blocked:
            continue
        result.extend(items)
        blocked.update(keys)
        counts[key] += 1
    if not per_class and counts["all"] != count:
        raise ValueError(f"独立来源组不足：{counts['all']} / {count}")
    return result


def coverage(rows, required=None):
    result = {}
    for language in ("en", "zh"):
        counts = Counter(
            row["expected"]["decision"]
            for row in rows
            if row.get("family") == "intent" and row["language"] == language
        )
        missing = sorted(set(INTENTS) - set(counts))
        result[language] = {
            "classes": len(counts),
            "positive_counts": dict(sorted(counts.items())),
            "missing": missing,
        }
        if required is not None and set(required) - set(counts):
            raise ValueError(f"{language} 意图正例缺类：{missing}")
    return result


def intent_partition(pool, index, blocked):
    # 稀有类先确保每个划分至少一个独立组，再补到开发 3 / 校准 2 / 训练 12。
    selected = {role: [] for role in ("train", "development", "calibration")}
    for role, seed in (("train", 92040), ("development", 92041), ("calibration", 92042)):
        selected[role] += take(pool, index, blocked, 1, seed, per_class=True)
        coverage(selected[role], INTENTS)
    for role, count, seed in (
        ("train", 11, 92043),
        ("development", 2, 92044),
        ("calibration", 1, 92045),
    ):
        selected[role] += take(pool, index, blocked, count, seed, per_class=True)
    return selected


def training_intent_variants(rows):
    rng = random.Random(92051)
    result = copy.deepcopy(rows)
    # 每个双语来源两种语言使用相同候选集；随机候选顺序避免位置记忆。
    for i, items in enumerate(groups(result).values()):
        gold = items[0]["expected"]["decision"]
        keys = list(INTENT_CRITERIA)
        if i % 2:
            keys = [gold, *rng.sample([k for k in keys if k != gold], 7)]
        rng.shuffle(keys)
        for row in items:
            row["request"]["questions"]["decision"]["criteria"] = {
                key: INTENT_CRITERIA[key] for key in keys
            }
    return result


def prepare():
    if ROOT.exists():
        raise ValueError("第三阶段数据已登记，不可覆盖。")
    history, history_hashes = historical()
    paws_raw = {
        split: {lang: [r for _, r in raw_rows("paws", lang, split)] for lang in ("en", "zh")}
        for split in ("train", "validation", "test")
    }
    english_paws = {
        (split, str(r["id"])): r for split, data in paws_raw.items() for r in data["en"]
    }
    index = SourceIndex(history, english_paws=english_paws)
    historical_keys = index.all_keys(history)
    blocked = set(historical_keys)
    intent_pool = paired_public("massive", "train") + paired_public("massive", "validation")
    parts = intent_partition(intent_pool, index, blocked)
    intent_test = paired_public("massive", "test")
    parts["test"] = take(intent_test, index, blocked, 4, 92046, per_class=True)
    intent_coverage = {role: coverage(rows) for role, rows in parts.items()}
    # 正例覆盖来自标签，与模型预测无关。完整官方 test 本身缺 cooking_query。
    official_intent_coverage = coverage(intent_test)
    parts["train"] = training_intent_variants(parts["train"])

    for split, specifications in (
        ("validation", (("development", 60), ("calibration", 35))),
        ("test", (("test", 80),)),
        ("train", (("train", 150),)),
    ):
        pool = paired_public("xnli", split, pool_limit=10000 if split == "train" else None)
        for n, (role, count) in enumerate(specifications):
            chosen = take(pool, index, blocked, count, 92061 + n, per_class=True)
            if Counter(r["expected"]["decision"] for r in chosen) != dict.fromkeys(NLI, count * 2):
                raise ValueError("XNLI 类别配额不足。")
            parts[role] += chosen

    for role, split, count in (
        ("development", "validation", 70),
        ("calibration", "validation", 50),
        ("test", "test", 90),
        ("train", "train", 150),
    ):
        parts[role] += take(paraphrases(split, paws_raw[split]), index, blocked, count, 92070)

    boolq = {
        split: reading(split, [r for _, r in raw_rows("boolq", "default", split)])
        for split in ("train", "validation")
    }
    for role, split, count in (
        ("development", "validation", 120),
        ("calibration", "validation", 80),
        ("test", "validation", 150),
        ("train", "train", 400),
    ):
        parts[role] += take(boolq[split], index, blocked, count, 92071)

    xquad = extraction(False)
    for role, count in (("test", 40), ("development", 30), ("calibration", 20)):
        parts[role] += take(xquad, index, blocked, count, 92072)
    squad = extraction(
        True, {"en": [r for _, r in raw_rows("squad", "plain_text", "train")]}, max_per_passage=2
    )
    parts["train"] += take(squad, index, blocked, 160, 92073)

    for i, (role, count) in enumerate((("test", 50), ("development", 35), ("calibration", 25))):
        chosen = unseen_retrieval(
            blocked, count, seed=92080 + i, namespace=f"p3-{role}", split="all"
        )
        if index.all_keys(chosen) & blocked:
            raise ValueError("SciFact 来源重叠。")
        parts[role] += chosen
        blocked.update(index.all_keys(chosen))

    for i, role in enumerate(("development", "calibration", "test")):
        for builder, count in ((numeric_rules, 20), (ordinal_rules, 35)):
            pool = builder(role, count + 10, seed=92100 + i, namespace=f"p3-{builder.__name__}")
            parts[role] += take(pool, index, blocked, count, 92110 + i)

    # 回放仅来自实际训练材料；所有新划分在回放之前已经排除了这些历史来源。
    replay = read_examples(Path("data/phase2/operator-contrast-v1/train.jsonl"))
    replay += read_examples(Path("data/phase2/refinement-v2/train.jsonl"))
    replay = list({canonical_request(r): r for r in replay}.values())
    replay_blocked = index.all_keys(
        [r for role, data in parts.items() if role != "train" for r in data]
    )
    for family, count in (("numeric_rule", 18), ("ordinal_rule", 20), ("candidate_retrieval", 35)):
        pool = [r for r in replay if r["family"] == family]
        parts["train"] += take(pool, index, replay_blocked, count, 92120)

    # 去重、重新计算包含翻译及共享内容的连通分量，并审计历史暴露。
    for role, data in parts.items():
        parts[role] = index.regroup(list({canonical_request(r): r for r in data}.values()))
        random.Random(92130).shuffle(parts[role])
        if role != "train" and index.all_keys(parts[role]) & historical_keys:
            raise ValueError(f"{role} 与历史暴露重叠。")
    all_keys = {role: index.all_keys(data) for role, data in parts.items()}
    for role, keys in all_keys.items():
        for other, other_keys in all_keys.items():
            if role != other and keys & other_keys:
                raise ValueError(f"第三阶段来源交叉：{role} / {other}")
    from transformers import AutoTokenizer

    from necro.backend import single_token_labels
    from necro.training.trainer import encode_example

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        revision="2fc06364715b967f1860aea9cf38778875588b17",
        local_files_only=True,
    )
    alphabet, _ = single_token_labels(tokenizer)
    lengths = [len(encode_example(tokenizer, alphabet, r)["input_ids"]) for r in parts["train"]]
    ROOT.mkdir(parents=True)
    files = {}
    for role, data in parts.items():
        files[role] = ROOT / ("validation.jsonl" if role == "development" else f"{role}.jsonl")
        write_jsonl(files[role], data)
    protocol = {
        "parent": "results/phase2/v3/selected/adapter",
        "reference": "jev-1.13.0",
        "acceptance": "docs/process/phase2-acceptance.md",
        "acceptance_sha256": digest(Path("docs/process/phase2-acceptance.md")),
        "historical_exclusion_files": history_hashes,
        "history_rows_including_duplicates_and_conservative_unused_material": len(history),
        "full_snapshots": {str(p): digest(p) for p in SNAPSHOTS.glob("*/manifest.json")},
        "families": {
            role: dict(Counter(r["family"] for r in data)) for role, data in parts.items()
        },
        "intent_coverage": intent_coverage,
        "official_intent_test_coverage": official_intent_coverage,
        "max_train_input_tokens": max(lengths),
        "sampling": "Whole-split individual/source-group sampling; intent capped class balance, "
        "XNLI balanced relations; remaining tasks random source groups. All previous "
        "train/dev/calibration/test and unused historical registrations protected.",
        "limitations": [
            "Rare intent classes have very few independent sources.",
            "Intent development and calibration use fresh pooled official train/validation rows.",
            "MASSIVE official test lacks cooking_query; historical exclusions may remove other "
            "rare classes. Class coverage is recorded for each split.",
            "Intent criterion cooking_query corrected using official training rows only.",
            "Synthetic rule templates are shared across partitions.",
            "Candidate extraction and sampled candidate retrieval are bounded choice tasks.",
            "Fresh SciFact holds out never-used queries and all candidate documents from pooled "
            "official train/test qrels, evaluated as sampled candidate selection.",
        ],
    }
    manifest = register(ROOT, files, protocol)
    print(
        json.dumps(
            {
                "partitions": manifest["partitions"],
                "families": protocol["families"],
                "intent_coverage": {
                    k: {lang: v[lang]["classes"] for lang in v} for k, v in intent_coverage.items()
                },
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    prepare()
