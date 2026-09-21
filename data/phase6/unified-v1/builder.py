"""统一多任务重训数据：清理历史配方，保护来源，并冻结新的自然任务测试。"""

# 完整规则保留在同一行，便于核对双语语义。
# ruff: noqa: E501

import argparse
import copy
import json
import random
import shutil
from collections import Counter
from itertools import product
from pathlib import Path

from necro.diagnostics.numeric_regression import rule_metadata
from necro.evaluation import read_examples
from necro.experiment_guard import canonical_request, digest, register
from necro.training.data.condition_balance import transfer_rows
from necro.training.data.condition_refinement import FIELDS, instructions
from necro.training.data.full_snapshots import rows as raw_rows
from necro.training.data.instruction_refinement import extraction_pairs
from necro.training.data.phase2_data import OPS, example, ordinal_rules, paraphrases, reading
from necro.training.data.phase3_data import (
    INTENT_CRITERIA,
    NLI,
    SourceIndex,
    coverage,
    paired_public,
    take,
)
from necro.training.data.source_isolation import unseen_retrieval
from necro.training.data.training_data import write_jsonl

ROOT = Path("data/phase6/unified-v1")
SOURCES = (
    "data/lora-pilot/train.jsonl",
    "data/improvement/expanded/train.jsonl",
    "data/phase2/experiment-v3/train.jsonl",
    "data/phase2/refinement-v2/train.jsonl",
    "data/phase2/operator-contrast-v1/train.jsonl",
    "data/phase3/coverage-v1/train.jsonl",
    "data/phase3/instruction-v2/train.jsonl",
)


def write(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def with_group(row):
    row = copy.deepcopy(row)
    if "group_id" not in row:
        pieces = row["id"].split("/")
        row["group_id"] = (
            f"xnli/{pieces[-2]}/{pieces[-1]}"
            if row["source"] == "facebook/xnli"
            else f"massive/{pieces[-1]}"
            if row["source"] == "mteb/amazon_massive_intent"
            else row["id"]
        )
    return row


def normalize(row):
    row = with_group(row)
    source = row["source"]
    question = row["request"]["questions"]["decision"]
    if source in {"constructed-rules", "constructed-uncertainty"}:
        return None
    if row.get("family") == "numeric_rule":
        return None
    if source == "mteb/amazon_massive_intent":
        row["family"] = "intent"
        if not set(question["criteria"]) <= INTENT_CRITERIA.keys():
            raise ValueError("Historical intent uses unknown class keys")
        question["criteria"] = {key: INTENT_CRITERIA[key] for key in question["criteria"]}
        question["instructions"] = (
            "选择用户最主要的意图。"
            if row["language"] == "zh"
            else "Select the user's primary intent."
        )
    elif source == "facebook/xnli":
        row["family"] = "inference"
        if question["type"] == "choice":
            if row["expected"]["decision"] not in NLI:
                raise ValueError("Historical NLI label is not a canonical relation")
            question["criteria"] = {key: NLI[key] for key in question["criteria"]}
    if not row.get("family"):
        raise ValueError(f"Unclassified training source: {source}")
    row.pop("training_weight", None)
    return row


def deduplicate(rows):
    unique = {}
    for row in rows:
        key = canonical_request(row)
        if key in unique and unique[key]["expected"] != row["expected"]:
            raise ValueError(f"Conflicting labels for a normalized request: {row['id']}")
        unique.setdefault(key, row)
    return list(unique.values())


def protected_material():
    paths = set()
    for registry in Path("data").rglob("experiment.json"):
        manifest = json.loads(registry.read_text(encoding="utf-8"))
        for role, partition in manifest["partitions"].items():
            path = registry.parent / Path(partition["path"].replace("\\", "/"))
            if digest(path) != partition["sha256"]:
                raise ValueError(f"Registered data changed: {path}")
            if role not in {"train", "exposure"}:
                paths.add(path)
    paths.update(Path("data/improvement").glob("*.jsonl"))
    paths.discard(Path("data/improvement/source-expanded.jsonl"))
    paths.update(
        map(
            Path,
            (
                "data/baseline.jsonl",
                "data/lora-pilot/validation.jsonl",
                "data/improvement/expanded/validation.jsonl",
            ),
        )
    )
    hashes, seen, rows = {}, set(), []
    for path in sorted(paths):
        sha = digest(path)
        hashes[path.as_posix()] = sha
        if sha not in seen:
            rows.extend(with_group(r) for r in read_examples(path))
            seen.add(sha)
    return rows, hashes


def condition_training():
    """三类原因等额；完整记录内比较真/假各半，失效前提占三分之一。"""
    rng = random.Random(2026092161)
    rows = []
    for i in range(200):
        code = list(OPS)[i % 5]
        style = (i // 5) % 8
        fields = FIELDS["train"][(i + i // 40) % len(FIELDS["train"])]
        a, b, gate = fields
        value = rng.randrange(11, 9900) * (-1 if rng.random() < 0.3 else 1)
        step = rng.choice((0.25, 0.5, 1, 7))
        positive, negative = {
            "ge": (0, -step),
            "gt": (step, 0),
            "le": (0, step),
            "lt": (-step, 0),
            "eq": (0, step),
        }[code]
        complete = {a: value + positive, b: value, gate: True}
        failure = {a: value + negative, b: value, gate: True}
        invalid_case = (i // 40 + style + i % 5) % 4
        invalid = (
            {**complete, gate: False}
            if invalid_case == 0
            else {key: v for key, v in complete.items() if key != fields[invalid_case - 1]}
        )
        group = f"unified-condition/train/{i}"
        for category, state, eligible in (
            ("eligible", complete, True),
            ("comparison_failure", failure, False),
            ("precondition_failure", invalid, False),
        ):
            for language, negated in product(("en", "zh"), (False, True)):
                # 旧训练措辞保留四种；新措辞独立编写，不使用已封存的 transfer 模板。
                prompt = (
                    instructions(fields, code, language, negated, style, 2)
                    if style < 4
                    else broad_instruction(fields, code, language, negated, style - 4)
                )
                row = example(
                    f"{group}/{category}/{language}/{int(negated)}",
                    group,
                    "numeric_rule",
                    language,
                    copy.deepcopy(state),
                    {"type": "noul", "instructions": prompt},
                    not eligible if negated else eligible,
                    "constructed-unified-condition",
                    "exact-oracle",
                )
                row.update(semantic_category=category, training_style=style)
                rule_metadata(row)
                rows.append(row)
    if len(deduplicate(rows)) != len(rows):
        raise ValueError("Constructed training contains duplicate observable requests")
    return rows


def broad_instruction(fields, code, language, negated, style):
    a, b, gate = fields
    _, zh, en = OPS[code]
    names = f"`{a}`, `{b}`, `{gate}`"
    if language == "zh":
        relation = f"`{a}` {zh} `{b}`"
        texts = [
            f"合格判定由三个要求共同决定：{names} 不能缺项；`{gate}` 的值是 true；{relation}。三个要求均成立就合格，有任何一个不成立就不合格。",
            f"完整性和开关是数值审核的前提。记录须提供 {names} 且 `{gate}` 为 true，再满足 {relation} 才合格。其他记录不合格。",
            f"这是一条同时满足规则：字段 {names} 齐全 AND `{gate}` 为 true AND {relation}。规则为真表示合格，为假表示不合格。",
            f"合格记录的定义如下：{relation}，同时提供全部字段 {names} 并将 `{gate}` 设为 true。全部满足就合格，缺少任意条件即不合格。",
        ]
        return texts[style] + ("该记录是否不合格？" if negated else "该记录是否合格？")
    relation = f"`{a}` is {en} `{b}`"
    texts = [
        f"Three requirements jointly determine eligibility: none of {names} may be absent; `{gate}` has value true; {relation}. The record is eligible when all three hold and ineligible when any one fails. ",
        f"Completeness and the switch are prerequisites for the numerical check. The record must supply {names}, have `{gate}` true, and satisfy that {relation} to be eligible. All other records are ineligible. ",
        f"This rule is a conjunction: all fields {names} are present AND `{gate}` is true AND {relation}. A true rule means eligible; a false rule means ineligible. ",
        f"An eligible record is defined as one where {relation}, all fields {names} are supplied, and `{gate}` is set to true. Satisfying all requirements means eligible; lacking any requirement means ineligible. ",
    ]
    return texts[style] + ("Is the record ineligible?" if negated else "Is the record eligible?")


def inventory(rows):
    return {
        "rows": len(rows),
        "unique_requests": len({canonical_request(r) for r in rows}),
        "source_groups": len({r["group_id"] for r in rows}),
        "families": dict(Counter(r["family"] for r in rows)),
        "family_language": dict(Counter(f"{r['family']}/{r['language']}" for r in rows)),
        "numeric_categories": dict(
            Counter(r["semantic_category"] for r in rows if "semantic_category" in r)
        ),
    }


def prepare(root=ROOT):
    if root.exists():
        raise ValueError(f"Output exists: {root}")
    protected, protected_hashes = protected_material()
    historical = [
        with_group(r) for p in sorted(Path("data").rglob("train.jsonl")) for r in read_examples(p)
    ]
    paws = {
        split: {
            language: [r for _, r in raw_rows("paws", language, split)] for language in ("en", "zh")
        }
        for split in ("train", "validation", "test")
    }
    english = {
        (split, str(r["id"])): r for split, languages in paws.items() for r in languages["en"]
    }
    index = SourceIndex([*historical, *protected], english_paws=english)
    blocked = index.all_keys(protected)
    curated = [
        new
        for p in SOURCES
        for row in read_examples(Path(p))
        if (new := normalize(row)) is not None
    ]
    curated = deduplicate(curated)
    curated, removed = index.exclude(curated, blocked)
    # 在源码定义更新后再去重，避免保留过时的类别描述。
    training = []
    for family in sorted({r["family"] for r in curated}):
        pool = [r for r in curated if r["family"] == family]
        if family == "intent":
            pool = take(pool, index, set(blocked), 24, 2026092162, per_class=True)
        training.extend(pool)
    coverage([r for r in training if r["family"] == "intent"])
    training.extend(condition_training())
    training = deduplicate(training)
    training = index.regroup(training)
    if index.all_keys(training) & blocked:
        raise ValueError("Unified training overlaps protected evaluation sources")
    for position, row in enumerate(training):
        row["source_keys"] = sorted(index.keys(row))
        row["original_id"] = row["id"]
        row["id"] = f"unified/train/{position}"

    # 新自然任务测试排除所有历史训练与所有旧评测，不仅排除这轮训练。
    test_blocked = blocked | index.all_keys(historical) | index.all_keys(training)
    natural = []
    natural += take(
        paired_public("xnli", "test"), index, test_blocked, 40, 2026092163, per_class=True
    )
    natural += take(
        paired_public("massive", "test"), index, test_blocked, 1, 2026092164, per_class=True
    )
    natural += take(paraphrases("test", paws["test"]), index, test_blocked, 60, 2026092165)
    natural += take(
        reading("validation", [r for _, r in raw_rows("boolq", "default", "validation")]),
        index,
        test_blocked,
        80,
        2026092166,
    )
    for language, name, config, source in (
        ("en", "squad", "plain_text", "rajpurkar/squad"),
        ("zh", "cmrc", "default", "hfl/cmrc2018"),
    ):
        pool = extraction_pairs(
            [r for _, r in raw_rows(name, config, "validation")], language, source
        )
        natural += take(pool, index, test_blocked, 30, 2026092167)
    natural += take(
        ordinal_rules("test", 40, seed=2026092168, namespace="unified-ordinal"),
        index,
        test_blocked,
        20,
        2026092169,
    )
    available = unseen_retrieval(
        test_blocked, None, seed=2026092170, namespace="unified-test", split="all"
    )
    natural += take(available, index, test_blocked, 30, 2026092171)
    natural = index.regroup(deduplicate(natural))
    if len({r["family"] for r in natural}) != 7:
        raise ValueError("Fresh natural test does not cover all seven nonnumeric families")
    test = [*natural, *transfer_rows("test")]
    if index.all_keys(test) & index.all_keys([*historical, *training]):
        raise ValueError("Fresh test overlaps a training source")

    parts = {
        "train": training,
        "development": read_examples(Path("data/phase5/condition-balance-v1/validation.jsonl")),
        "calibration": read_examples(Path("data/phase5/condition-balance-v1/calibration.jsonl")),
        "test": test,
    }
    from transformers import AutoTokenizer

    from necro.backend import single_token_labels
    from necro.training.trainer import encode_example

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        revision="2fc06364715b967f1860aea9cf38778875588b17",
        local_files_only=True,
    )
    alphabet, _ = single_token_labels(tokenizer)
    lengths = [len(encode_example(tokenizer, alphabet, r)["input_ids"]) for r in training]
    root.mkdir(parents=True)
    files = {}
    for role, rows in parts.items():
        if role == "train":
            random.Random(2026092172).shuffle(rows)
        files[role] = root / ("validation.jsonl" if role == "development" else f"{role}.jsonl")
        write_jsonl(files[role], rows)
    shutil.copy2(__file__, root / "builder.py")
    write_jsonl(root / "natural-test.jsonl", natural)
    manifest = register(
        root,
        files,
        {
            "initialization": "Fresh rank-16 LoRA from official post-trained Qwen/Qwen3.5-0.8B; no project adapter",
            "training_sources": {p: digest(Path(p)) for p in SOURCES},
            "protected_evaluation": protected_hashes,
            "historical_training": {
                p.as_posix(): digest(p)
                for p in Path("data").rglob("train.jsonl")
                if p.parent != root
            },
            "builder_sha256": digest(root / "builder.py"),
            "removed_training_source_groups": len(removed),
            "data": {role: inventory(rows) for role, rows in parts.items()},
            "max_train_tokens": max(lengths),
            "train_tokens_per_epoch": sum(lengths),
            "condition_design": "Equal eligible/comparison-failure/precondition-failure strata; complete comparisons balanced; unit weights; no hidden counterfactual duplication",
            "test_scope": "Fresh natural sources plus the still-unopened 4800-question condition stress cohort; natural-test.jsonl is an audit subset, not a separate selection set",
            "intent_training_coverage": coverage([r for r in training if r["family"] == "intent"]),
            "intent_test_coverage": coverage([r for r in natural if r["family"] == "intent"]),
        },
    )
    write(root / "audit.json", manifest["protocol"])
    print(json.dumps({role: inventory(rows) for role, rows in parts.items()}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT)
    prepare(parser.parse_args().output)
