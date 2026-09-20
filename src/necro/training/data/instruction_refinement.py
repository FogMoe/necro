"""用同一证据的相反判断及候选有/无答案对照，修复开发阶段的指令忽略。"""

import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from necro.evaluation import read_examples
from necro.experiment_guard import canonical_request, digest, normalized_text, register
from necro.training.data.full_snapshots import rows as raw_rows
from necro.training.data.phase2_data import choice, example, reading
from necro.training.data.phase3_data import SourceIndex, coverage, historical, paired_public, take
from necro.training.data.source_isolation import signature
from necro.training.data.training_data import write_jsonl

ROOT = Path("data/phase3/instruction-v2")


def reading_pairs(rows):
    positive = (
        "Using the passage, is an affirmative answer to the supplied question warranted?",
        "Does the passage support answering the question in state with yes?",
        "Should the supplied question receive the answer 'yes', judged only from the passage?",
    )
    negative = (
        "Using the passage, is a negative answer to the supplied question warranted?",
        "Does the passage support answering the question in state with no?",
        "Should the supplied question receive the answer 'no', judged only from the passage?",
    )
    result = []
    for i, original in enumerate(rows):
        for polarity, prompts in ((True, positive), (False, negative)):
            row = copy.deepcopy(original)
            row["id"] += f"/instruction-{int(polarity)}"
            row["request"]["questions"]["decision"]["instructions"] = prompts[i % len(prompts)]
            row["expected"]["decision"] = (
                original["expected"]["decision"]
                if polarity
                else not original["expected"]["decision"]
            )
            row["label_quality"] = "derived-human"
            row["derivation"] = (
                "Positive/negative answer proposition over the same human-labeled evidence"
            )
            result.append(row)
    return result


def extraction_pairs(raw, language, source):
    rng = random.Random(920301)
    paragraphs = defaultdict(list)
    for row in raw:
        paragraphs[row["context"]].append(row)
    result = []
    for context, questions in paragraphs.items():
        if len(context) > 4500:
            continue
        possible = list(
            dict.fromkeys(t for q in questions for t in q["answers"]["text"] if 0 < len(t) < 160)
        )
        produced = 0
        for row in questions:
            aliases = row["answers"]["text"]
            if not aliases or len(aliases[0]) >= 160:
                continue
            negative = [
                t
                for t in possible
                if not any(
                    normalized_text(t) in normalized_text(a)
                    or normalized_text(a) in normalized_text(t)
                    for a in aliases
                )
            ]
            if len(negative) < 3:
                continue
            negatives = rng.sample(negative, 3)
            none = (
                "以上候选均不是正确答案"
                if language == "zh"
                else "None of the listed answers is correct"
            )
            group = "passage/" + signature(context)
            for present in (True, False):
                gold = aliases[0] if present else none
                options = [aliases[0], *negatives[:2], none] if present else [*negatives, none]
                criteria, expected = choice(options, gold, rng)
                item = example(
                    f"paired-extraction/{language}/{row['id']}/{int(present)}",
                    group,
                    "candidate_extraction",
                    language,
                    {"passage": context, "question": row["question"]},
                    {
                        "type": "choice",
                        "instructions": "仅按 passage 回答 question，选择正确候选；"
                        "若候选均不正确，则选择对应的无匹配选项。"
                        if language == "zh"
                        else "Answer question using only passage. Select the correct "
                        "candidate, or the no-match option when none of the candidates is correct.",
                        "criteria": criteria,
                    },
                    expected,
                    source,
                    "derived-human",
                )
                item["source_keys"] = [group, "reading-question/" + signature(row["question"])]
                item["derivation"] = (
                    "Same question and passage; gold answer included or replaced "
                    "by a same-passage distractor"
                )
                result.append(item)
            produced += 1
            if produced == 2:
                break
    return result


def prepare():
    if ROOT.exists():
        raise ValueError("指令对照数据已存在，不可覆盖。")
    base = Path("data/phase3/coverage-v1")
    past, historical_files = historical()
    trained_hashes = {digest(base / "train.jsonl")} | {
        json.loads(p.read_text())["train_sha256"]
        for p in Path("results/phase3").glob("*/adapter/necro_adapter.json")
    }
    prior_paths = sorted(
        p for p in Path("data/phase3").glob("*/train.jsonl") if digest(p) in trained_hashes
    )
    prior = [r for p in prior_paths for r in read_examples(p)]
    heldout_paths = [
        base / name for name in ("validation.jsonl", "calibration.jsonl", "test.jsonl")
    ]
    heldout = [r for p in heldout_paths for r in read_examples(p)]
    english = {
        (split, str(r["id"])): r
        for split in ("train", "validation", "test")
        for _, r in raw_rows("paws", "en", split)
    }
    index = SourceIndex([*past, *prior, *heldout], english_paws=english)
    blocked = index.all_keys([*past, *prior, *heldout])
    fresh_reading = take(
        reading("train", [r for _, r in raw_rows("boolq", "default", "train")]),
        index,
        blocked,
        300,
        920302,
    )
    train = reading_pairs(fresh_reading)
    for language, name, config, source in (
        ("en", "squad", "plain_text", "rajpurkar/squad"),
        ("zh", "cmrc", "default", "hfl/cmrc2018"),
    ):
        pool = extraction_pairs([r for _, r in raw_rows(name, config, "train")], language, source)
        train += take(pool, index, blocked, 100, 920303)
    nli = take(
        paired_public("xnli", "train", pool_limit=10000), index, blocked, 50, 920304, per_class=True
    )
    if len(nli) != 300:
        raise ValueError("新的平衡 NLI 来源不足。")
    train += nli
    replay = read_examples(base / "train.jsonl")
    replay_blocked = index.all_keys([*heldout, *train])
    intents = take(
        [r for r in replay if r["family"] == "intent"],
        index,
        replay_blocked,
        2,
        920305,
        per_class=True,
    )
    from necro.datasets import INTENTS

    coverage(intents, INTENTS)
    train += intents
    train += take(
        [r for r in replay if r["family"] == "paraphrase"], index, replay_blocked, 60, 920306
    )
    rule_paths = [
        Path("data/phase2/operator-contrast-v1/train.jsonl"),
        Path("data/phase2/refinement-v2/train.jsonl"),
    ]
    rule_pool = list(
        {canonical_request(r): r for p in rule_paths for r in read_examples(p)}.values()
    )
    for family, count in (("numeric_rule", 35), ("ordinal_rule", 20), ("candidate_retrieval", 35)):
        train += take(
            [r for r in rule_pool if r["family"] == family], index, replay_blocked, count, 920307
        )
    train = list({canonical_request(r): r for r in train}.values())
    if index.all_keys(train) & index.all_keys(heldout):
        raise ValueError("指令训练与保护划分重叠。")
    train = index.regroup(train)
    random.Random(2026).shuffle(train)
    from transformers import AutoTokenizer

    from necro.backend import single_token_labels
    from necro.training.trainer import encode_example

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        revision="2fc06364715b967f1860aea9cf38778875588b17",
        local_files_only=True,
    )
    alphabet, _ = single_token_labels(tokenizer)
    lengths = [len(encode_example(tokenizer, alphabet, r)["input_ids"]) for r in train]
    ROOT.mkdir(parents=True)
    write_jsonl(ROOT / "train.jsonl", train)
    write_jsonl(ROOT / "validation.jsonl", read_examples(base / "validation.jsonl"))
    result = register(
        ROOT,
        {"train": ROOT / "train.jsonl", "development": ROOT / "validation.jsonl"},
        {
            "hypothesis": "Paired instruction polarity and answer presence force decisions to "
            "depend on the requested predicate/candidates; stronger exact-rule replay counters "
            "the measured numeric regression. Add native Chinese human QA instead of training "
            "English extraction only.",
            "parent": "results/phase3/coverage-seed2026/adapter",
            "protected_manifest_sha256": digest(base / "experiment.json"),
            "historical_files": historical_files,
            "prior_phase3_training": {str(p): digest(p) for p in prior_paths},
            "rule_replay": {str(p): digest(p) for p in rule_paths},
            "families": dict(Counter(r["family"] for r in train)),
            "intent_coverage": coverage(intents, INTENTS),
            "new_reading_source_groups": len({r["group_id"] for r in fresh_reading}),
            "max_train_input_tokens": max(lengths),
            "cmrc_snapshot_sha256": digest(Path("data/phase3/snapshots/cmrc/manifest.json")),
            "builder_sha256": digest(Path(__file__)),
            "notes": "No Jev labels, no old final failures, no new final predictions used for "
            "training. Paired variants retain their shared source groups. "
            "All evaluation cohorts unchanged.",
        },
    )
    print(
        json.dumps(
            {
                "partitions": result["partitions"],
                "families": result["protocol"]["families"],
                "max_tokens": max(lengths),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    prepare()
