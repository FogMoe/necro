"""在预测前登记八任务最终集；旧数据只参与去重，不用于挑选难度。"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from necro.data_catalog import snapshot
from necro.datasets import INTENTS
from necro.evaluation import read_examples
from necro.experiment_guard import canonical_request, digest, normalized_text, register
from necro.phase2_data import example
from necro.training_data import write_jsonl

ROOT = Path("data/phase2/experiment-v3")
OUTPUT = Path("data/phase2/final-v3")


def public_pool():
    rows = []
    for family, dataset in (
        ("inference", "facebook/xnli"),
        ("intent", "mteb/amazon_massive_intent"),
    ):
        for language in ("en", "zh"):
            config = "zh-CN" if family == "intent" and language == "zh" else language
            name = f"final-{family}-{language}"
            snapshot(name, dataset, config, "test", 500, 20260920)
            raw = json.loads(Path(f"data/phase2/{name}.json").read_text(encoding="utf-8"))
            for item in raw["rows"]:
                row = item["row"]
                if family == "inference":
                    group = f"xnli/test/{item['row_idx']}"
                    state = {"premise": row["premise"], "hypothesis": row["hypothesis"]}
                    criteria = {
                        "entailment": "The premise supports the hypothesis.",
                        "neutral": "The premise neither supports nor contradicts the hypothesis.",
                        "contradiction": "The premise contradicts the hypothesis.",
                    }
                    instruction = (
                        "根据 premise 判断 hypothesis 是否成立，仅依据给定前提，不补充外部事实。"
                        if language == "zh"
                        else "Does the premise support, contradict, or leave the hypothesis "
                        "undetermined? Use only the given premise."
                    )
                    gold = list(criteria)[row["label"]]
                else:
                    group = f"massive/{row['id']}"
                    state, criteria, gold = row["text"], INTENTS, row["label"]
                    instruction = (
                        "选择用户最主要的意图。"
                        if language == "zh"
                        else ("Select the user's primary intent.")
                    )
                rows.append(
                    example(
                        f"final/{family}/{language}/{item['row_idx']}",
                        group,
                        family,
                        language,
                        state,
                        {"type": "choice", "instructions": instruction, "criteria": criteria},
                        gold,
                        dataset,
                    )
                )
    return rows


def clean_groups(candidates, protected):
    groups = {row["group_id"] for row in protected}
    states = {normalized_text(row["request"]["state"]) for row in protected}
    requests = {canonical_request(row) for row in protected}
    rejected = {
        row["group_id"]
        for row in candidates
        if row["group_id"] in groups
        or normalized_text(row["request"]["state"]) in states
        or canonical_request(row) in requests
    }
    return [row for row in candidates if row["group_id"] not in rejected], sorted(rejected)


def prepare():
    if OUTPUT.exists():
        raise ValueError("最终集目录已存在，不可覆盖。")
    pool = public_pool()
    historical = []
    sources = {}
    # 包括祖先实际训练、旧测试和先前查看的开发记录；不把仅缓存的原始池视为训练。
    paths = [
        Path("data/lora-pilot/train.jsonl"),
        Path("data/baseline.jsonl"),
        Path("data/lora-pilot/validation.jsonl"),
        Path("data/improvement/expanded/train.jsonl"),
        Path("data/improvement/test-sealed.jsonl"),
    ]
    paths += [
        ROOT / filename
        for filename in ("train.jsonl", "validation.jsonl", "calibration.jsonl", "regression.jsonl")
    ]
    for path in paths:
        historical.extend(read_examples(path))
        sources[str(path)] = digest(path)
    for row in historical:
        row.setdefault("group_id", row["id"])
    pool, excluded = clean_groups(pool, historical)
    chosen = []
    for family in ("inference", "intent"):
        groups = defaultdict(list)
        for row in pool:
            if row["family"] == family:
                groups[row["group_id"]].append(row)
        keys = [
            key
            for key, items in groups.items()
            if {row["language"] for row in items} == {"en", "zh"}
        ]
        random.Random(5202026).shuffle(keys)
        # 同义重复话语也去重，不能仅换 ID 充数。
        seen_states, count = set(), 0
        for key in keys:
            states = {normalized_text(row["request"]["state"]) for row in groups[key]}
            if states & seen_states:
                continue
            chosen.extend(groups[key])
            seen_states.update(states)
            count += 1
            if count == 100:
                break
        if count != 100:
            raise ValueError(f"{family} 独立双语组不足：{count}")
    new_tasks = read_examples(ROOT / "test.jsonl")
    # 历史规则有时共享简短 state，跨问题 state 相同不等于题目泄漏；
    # 新任务已有自己的组和规范化请求去重，不能用 state 单独丢掉规则难题。
    old_groups = {row["group_id"] for row in historical}
    old_requests = {canonical_request(row) for row in historical}
    if any(
        row["group_id"] in old_groups or canonical_request(row) in old_requests for row in new_tasks
    ):
        raise ValueError("新任务测试与历史数据存在重叠，请审计，不可静默清除。")
    OUTPUT.mkdir(parents=True)
    write_jsonl(OUTPUT / "test.jsonl", [*new_tasks, *chosen])
    manifest = register(
        OUTPUT,
        {"test": OUTPUT / "test.jsonl"},
        {
            "reference": "jev-1.13.0",
            "acceptance": "docs/process/phase2-acceptance.md",
            "acceptance_sha256": digest(Path("docs/process/phase2-acceptance.md")),
            "historical_exclusion_sources": sources,
            "six_task_test_sha256": digest(ROOT / "test.jsonl"),
            "fresh_public_groups_per_family": 100,
            "families": dict(Counter(row["family"] for row in [*new_tasks, *chosen])),
            "removed_public_groups": excluded,
            "raw_snapshots": {
                str(path): digest(path) for path in Path("data/phase2").glob("final-*.json")
            },
        },
    )
    print(json.dumps(manifest["partitions"], indent=2), flush=True)


def prepare_calibration():
    """为共同的温度拟合补齐原任务；不更改已登记的训练或测试。"""
    output = Path("data/phase2/calibration-v3")
    if output.exists():
        raise ValueError("校准目录已存在，不可覆盖。")
    protected_paths = [
        Path("data/lora-pilot/train.jsonl"),
        Path("data/improvement/expanded/train.jsonl"),
        ROOT / "train.jsonl",
        ROOT / "validation.jsonl",
        OUTPUT / "test.jsonl",
    ]
    protected = [row for path in protected_paths for row in read_examples(path)]
    legacy_path = Path("data/improvement/calibration-clean.jsonl")
    legacy = read_examples(legacy_path)
    for row in legacy:
        row["family"] = "inference" if "xnli" in row["source"] else "intent"
        row["label_quality"] = "human"
    legacy, removed = clean_groups(legacy, protected)
    combined = [*read_examples(ROOT / "calibration.jsonl"), *legacy]
    protected_groups = {row["group_id"] for row in protected}
    protected_requests = {canonical_request(row) for row in protected}
    if any(
        row["group_id"] in protected_groups or canonical_request(row) in protected_requests
        for row in combined
    ):
        raise ValueError("校准与训练、开发或封存测试重叠。")
    output.mkdir(parents=True)
    write_jsonl(output / "calibration.jsonl", combined)
    manifest = register(
        output,
        {"calibration": output / "calibration.jsonl"},
        {
            "method": "same per-primitive temperature procedure for every candidate",
            "protected": {str(path): digest(path) for path in protected_paths},
            "legacy_calibration_sha256": digest(legacy_path),
            "new_calibration_sha256": digest(ROOT / "calibration.jsonl"),
            "removed_legacy_groups": removed,
            "families": dict(Counter(row["family"] for row in combined)),
        },
    )
    print(json.dumps(manifest["partitions"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", action="store_true")
    args = parser.parse_args()
    prepare_calibration() if args.calibration else prepare()
