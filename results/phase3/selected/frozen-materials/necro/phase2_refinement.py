"""针对开发集证据扩充规则对比与回放；禁止使用封存测试反馈。"""

import copy
import json
import random
from collections import Counter
from pathlib import Path

from necro.evaluation import read_examples
from necro.experiment_guard import canonical_request, digest, normalized_text, register
from necro.phase2_data import OPS, example, grouped_sample, paraphrases, reading
from necro.training_data import write_jsonl


def contrast_rules():
    rng = random.Random(92102026)
    fields = [
        ("credit", "minimum", "account_active"),
        ("weight", "limit", "inspection_passed"),
        ("duration", "target", "license_valid"),
        ("quantity", "quota", "manager_agreed"),
        ("distance", "cutoff", "sensor_ready"),
        ("temperature", "reference", "power_on"),
        ("points", "benchmark", "entry_valid"),
        ("height", "bar", "route_clear"),
    ]
    rows = []
    for i in range(40):
        op = list(OPS)[i % 5]
        fn, zh, en = OPS[op]
        a, b, flag = fields[(i // 5 + i) % len(fields)]
        threshold, step = rng.randrange(5, 950), 0.5 if i % 2 else 1
        states = [{a: threshold + delta, b: threshold, flag: True} for delta in (-step, 0, step)]
        missing = {a: threshold, b: threshold, flag: True}
        missing.pop((a, b, flag)[i % 3])
        states.extend([missing, {a: threshold, b: threshold, flag: False}])
        for language in ("en", "zh"):
            for negated in (False, True):
                rule = (
                    f"判断规则：所有字段 `{a}`、`{b}`、`{flag}` 必须存在，且 `{flag}` 为 true，"
                    f"且 `{a}` {zh} `{b}`。满足全部条件则通过，否则不通过。"
                    + ("当前记录是否不通过？" if negated else "当前记录是否通过？")
                    if language == "zh"
                    else f"A record passes exactly when all fields `{a}`, `{b}`, `{flag}` are "
                    f"present, `{flag}` is true, and `{a}` is {en} `{b}`. It fails otherwise. "
                    + ("Does this record fail?" if negated else "Does this record pass?")
                )
                for side, state in enumerate(states):
                    passes = all(key in state for key in (a, b, flag)) and (
                        state[flag] is True and fn(state[a], state[b])
                    )
                    rows.append(
                        example(
                            f"contrast-rule/{i}/{language}/{int(negated)}/{side}",
                            f"contrast-rule/{i}",
                            "numeric_rule",
                            language,
                            state,
                            {"type": "noul", "instructions": rule},
                            not passes if negated else passes,
                            "constructed-contrast-rule-v3",
                            "exact-oracle",
                        )
                    )
    return rows


def prepare():
    root = Path("data/phase2/refinement-v1")
    if root.exists():
        raise ValueError("改进数据已经存在，不可覆盖。")
    base = Path("data/phase2/experiment-v3")
    development = read_examples(base / "validation.jsonl")
    calibration = read_examples(Path("data/phase2/calibration-v3/calibration.jsonl"))
    final = read_examples(Path("data/phase2/final-v3/test.jsonl"))
    protected = [*development, *calibration, *final]
    protected_groups = {row["group_id"] for row in protected}
    protected_states = {
        normalized_text(row["request"]["state"])
        for row in protected
        if not row["source"].startswith("constructed-")
    }
    protected_requests = {canonical_request(row) for row in protected}

    def clean(rows):
        blocked = {
            row["group_id"]
            for row in rows
            if row["group_id"] in protected_groups
            or normalized_text(row["request"]["state"]) in protected_states
            or canonical_request(row) in protected_requests
        }
        return [row for row in rows if row["group_id"] not in blocked]

    current = read_examples(base / "train.jsonl")
    old_groups = {row["group_id"] for row in current}
    legacy = [
        copy.deepcopy(row)
        for row in read_examples(Path("data/improvement/expanded/train.jsonl"))
        if row["source"] in {"facebook/xnli", "mteb/amazon_massive_intent"}
    ]
    for row in legacy:
        row["family"] = "inference" if "xnli" in row["source"] else "intent"
        row["label_quality"] = "human"
    train = []
    for family in ("inference", "intent"):
        for language in ("en", "zh"):
            candidates = clean(
                [row for row in legacy if row["family"] == family and row["language"] == language]
            )
            train += grouped_sample(candidates, 250, 629)
    for family in (
        "paraphrase",
        "reading_boolean",
        "ordinal_rule",
        "candidate_extraction",
        "candidate_retrieval",
    ):
        # 保留新增能力的回放，避免只修一项又损伤其他任务。
        train += grouped_sample([row for row in current if row["family"] == family], 120, 720)
    train += grouped_sample(
        clean([row for row in reading("train") if row["group_id"] not in old_groups]), 200, 831
    )
    paws = clean([row for row in paraphrases("train") if row["group_id"] not in old_groups])
    train += grouped_sample(paws, 200, 932)
    train += contrast_rules()
    train = clean(train)
    unique = {canonical_request(row): row for row in train}
    train = list(unique.values())
    random.Random(2026).shuffle(train)
    from transformers import AutoTokenizer

    from necro.backend import single_token_labels
    from necro.training import encode_example

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        revision="2fc06364715b967f1860aea9cf38778875588b17",
        local_files_only=True,
    )
    alphabet, _ = single_token_labels(tokenizer)
    lengths = [len(encode_example(tokenizer, alphabet, row)["input_ids"]) for row in train]
    root.mkdir(parents=True)
    files = {}
    for role, rows, name in (
        ("train", train, "train"),
        ("development", development, "validation"),
        ("calibration", calibration, "calibration"),
        ("test", final, "test"),
    ):
        files[role] = root / f"{name}.jsonl"
        write_jsonl(files[role], rows)
    manifest = register(
        root,
        files,
        {
            "hypothesis": "More varied paired numeric rules and legacy replay reduce "
            "missing/negation "
            "errors and retention loss observed on development only; no final predictions used.",
            "families": dict(Counter(row["family"] for row in train)),
            "max_input_tokens": max(lengths),
            "new_to_phase2_training_groups": len({row["group_id"] for row in train} - old_groups),
            "final_parent_sha256": digest(Path("data/phase2/final-v3/test.jsonl")),
            "parent_train_sha256": digest(base / "train.jsonl"),
            "notes": "New to phase2 does not mean new to the ancestor: "
            "legacy rows are deliberate replay.",
        },
    )
    print(json.dumps(manifest["protocol"], indent=2), flush=True)


if __name__ == "__main__":
    prepare()
