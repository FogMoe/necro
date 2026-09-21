"""Balanced precondition contrasts with held-out wording and numeric source groups."""

# Keep complete prompt variants together for wording review.
# ruff: noqa: E501

import argparse
import copy
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from necro.diagnostics.numeric_regression import rule_metadata
from necro.evaluation import read_examples
from necro.experiment_guard import canonical_request, digest, register
from necro.training.data.full_snapshots import rows as raw_rows
from necro.training.data.phase2_data import OPS, example
from necro.training.data.phase3_data import SourceIndex, coverage, groups, take
from necro.training.data.training_data import write_jsonl

ROOT = Path("data/phase4/condition-v1")
FIELDS = {
    "train": [
        ("credit", "minimum", "account_active"),
        ("weight", "limit", "inspection_passed"),
        ("duration", "target", "license_valid"),
        ("quantity", "quota", "manager_agreed"),
        ("distance", "cutoff", "sensor_ready"),
        ("temperature", "reference", "power_on"),
        ("points", "benchmark", "entry_valid"),
        ("alpha", "beta", "flag"),
    ],
    "development": [
        ("daily_budget", "cost", "policy_enabled"),
        ("reservoir", "demand", "valve_ready"),
        ("credits", "requirement", "enrollment_open"),
        ("reading_a", "reading_b", "check_ready"),
    ],
    "test": [
        ("supply", "order_size", "dispatch_ready"),
        ("elapsed", "deadline", "session_valid"),
        ("score_a", "score_b", "round_active"),
        ("measurement_x", "measurement_y", "test_enabled"),
    ],
}


def instructions(fields, code, language, negated, style, revision=1):
    a, b, gate = fields
    _, zh, en = OPS[code]
    comparison = f"`{a}` {zh} `{b}`" if language == "zh" else f"`{a}` is {en} `{b}`"
    names = f"`{a}`, `{b}`, `{gate}`"
    if revision == 2 and style in (0, 1):
        if language == "zh":
            rule = (
                f"当且仅当 {comparison} 且 `{gate}` 为 true 时合格；其余或缺少字段时不合格。"
                if style == 0
                else f"缺少字段或 `{gate}` 不为 true 时不合格。其余情况下，{comparison} 则合格，否则不合格。"
            )
            return rule + ("现在是否不合格？" if negated else "现在是否合格？")
        rule = (
            f"The record is eligible if and only if {comparison} and `{gate}` is true. Otherwise, or if required fields are missing, it is ineligible. "
            if style == 0
            else f"Missing fields or `{gate}` not true means ineligible. For all other cases, the record is eligible when {comparison}, and ineligible otherwise. "
        )
        return rule + ("Is this record ineligible?" if negated else "Is this record eligible?")
    if language == "zh":
        variants = [
            f"当且仅当所有必需字段 {names} 存在、{comparison}，且 `{gate}` 为 true 时合格。其他情况均不合格。",
            f"缺少字段 {names} 中任意一项或 `{gate}` 不为 true 时不合格。其余情况下，{comparison} 则合格，否则不合格。",
            f"必须同时满足三项要求才合格：字段 {names} 全部存在；`{gate}` 为 true；{comparison}。任何一项不满足就不合格。",
            f"检查资格：字段 {names} 必须齐全，`{gate}` 必须为 true，并且必须满足 {comparison}。通过全部检查即合格，否则不合格。",
            f"在字段 {names} 齐全并且 `{gate}` 为 true 的记录中，{comparison} 的记录合格，其余不合格。缺少字段或开关为 false 的记录都不合格。",
            f"先验证记录含有 {names} 并且 `{gate}` 为 true，验证失败即不合格。验证通过后，合格与否由 {comparison} 是否成立决定。",
            f"依次应用规则：若字段 {names} 未全部提供，判为不合格。若 `{gate}` 不是 true，也判为不合格。其他记录在 {comparison} 成立时合格，否则不合格。",
            f"任意必需字段 {names} 缺失，或 `{gate}` 为 false，均直接判为不合格，无论数值如何。其他记录恰在 {comparison} 时合格。",
        ]
        return variants[style] + ("当前记录是否不合格？" if negated else "当前记录是否合格？")
    variants = [
        f"The record is eligible if and only if all fields {names} exist, {comparison}, and `{gate}` is true. All other records are ineligible. ",
        f"Missing any field in {names}, or `{gate}` not true, means ineligible. For all other cases, the record is eligible when {comparison}, and ineligible otherwise. ",
        f"Eligibility requires all three conditions: every field in {names} exists, `{gate}` is true, and {comparison}. Failure of any condition means ineligible. ",
        f"Check eligibility: fields {names} must be present, `{gate}` must be true, and {comparison} must hold. Passing every check means eligible, otherwise ineligible. ",
        f"Among records containing all fields {names} with `{gate}` true, those where {comparison} are eligible and the rest are ineligible. Missing fields or a false gate always make a record ineligible. ",
        f"First validate that the record contains {names} and `{gate}` is true. Failed validation means ineligible. After successful validation, eligibility is determined by whether {comparison}. ",
        f"Apply these rules in order: if any field in {names} is absent, mark ineligible. If `{gate}` is not true, also mark ineligible. Remaining records are eligible when {comparison}, otherwise ineligible. ",
        f"Any missing required field in {names}, or a false `{gate}`, makes the record ineligible regardless of its numbers. Every other record is eligible exactly when {comparison}. ",
    ]
    return variants[style] + (
        "Is this record ineligible?" if negated else "Is this record eligible?"
    )


def construct(revision=1):
    rng = random.Random(2026092101)
    thresholds = iter(rng.sample(range(17, 987), 48))
    partitions = {}
    for role, styles in (("train", range(4)), ("development", (4, 5)), ("test", (6, 7))):
        effective_revision = revision if role == "train" else 1
        rows = []
        for i in range(len(FIELDS[role]) * 3):
            fields = FIELDS[role][i % len(FIELDS[role])]
            a, b, gate = fields
            threshold = next(thresholds)
            delta = (-1, 0, 1)[i // len(FIELDS[role])] * (0.5 if i % 2 else 1)
            state = {a: threshold + delta, b: threshold, gate: True}
            variants = [
                state,
                {**state, gate: False},
                {k: v for k, v in state.items() if k != fields[i % 3]},
            ]
            group = f"condition-v1/{role}/{i}"
            for op_index, (code, (fn, _, _)) in enumerate(OPS.items()):
                for case, current in enumerate(variants):
                    eligible = (
                        all(k in current for k in fields)
                        and current[gate] is True
                        and fn(current[a], current[b])
                    )
                    for language in ("en", "zh"):
                        for negated in (False, True):
                            style = styles[
                                (i + op_index + (case if effective_revision == 1 else 0))
                                % len(styles)
                            ]
                            row = example(
                                f"{group}/{code}/{case}/{language}/{int(negated)}",
                                group,
                                "numeric_rule",
                                language,
                                copy.deepcopy(current),
                                {
                                    "type": "noul",
                                    "instructions": instructions(
                                        fields, code, language, negated, style, effective_revision
                                    ),
                                },
                                not eligible if negated else bool(eligible),
                                "constructed-condition-v1",
                                "exact-oracle",
                            )
                            row.update(condition_style=style, condition_case=case)
                            rule_metadata(row)
                            rows.append(row)
            partitions[role] = rows
    return partitions


def prepare(revision=1):
    root = ROOT if revision == 1 else ROOT.with_name("condition-v2")
    if root.exists():
        raise ValueError("Condition refinement is already registered")
    parts = construct(revision)
    base = Path("data/phase3/coverage-v1")
    replay = read_examples(Path("data/phase3/instruction-v2/train.jsonl"))
    heldout_paths = [
        base / name for name in ("validation.jsonl", "calibration.jsonl", "test.jsonl")
    ]
    heldout = [r for p in heldout_paths for r in read_examples(p)]
    english = {
        (split, str(r["id"])): r
        for split in ("train", "validation", "test")
        for _, r in raw_rows("paws", "en", split)
    }
    index = SourceIndex(
        [*replay, *heldout, *[r for rows in parts.values() for r in rows]], english_paws=english
    )
    blocked = index.all_keys(heldout + parts["development"] + parts["test"])
    retained = []
    for family, count in (
        ("intent", 2),
        ("inference", 100),
        ("reading_boolean", 80),
        ("candidate_extraction", 40),
        ("paraphrase", 50),
        ("ordinal_rule", 20),
        ("candidate_retrieval", 35),
    ):
        pool = [r for r in replay if r["family"] == family]
        limit = count if family == "intent" else min(count, len(groups(pool)))
        retained += take(pool, index, blocked, limit, 2026092102, per_class=family == "intent")
    coverage(retained)
    if revision == 2:
        numeric_replay = [
            row for row in read_examples(base / "train.jsonl") if row["family"] == "numeric_rule"
        ]
        if index.all_keys(numeric_replay) & index.all_keys(heldout):
            raise ValueError("Numeric replay overlaps previous holdouts")
        retained += numeric_replay
    parts["train"] += retained
    # Existing development tasks remain part of capability-retention checks.
    parts["development"] += read_examples(base / "validation.jsonl")
    parts["calibration"] = read_examples(base / "calibration.jsonl")
    all_keys = {role: index.all_keys(rows) for role, rows in parts.items()}
    for role, keys in all_keys.items():
        for other, other_keys in all_keys.items():
            if role != other and keys & other_keys:
                raise ValueError(f"Partition source overlap: {role}/{other}")
    prior = {
        canonical_request(r)
        for p in Path("data").rglob("*.jsonl")
        if revision == 1 or p.name != "train.jsonl"
        for r in read_examples(p)
    }
    new_rows = [
        r
        for role, rows in parts.items()
        if revision == 1 or role == "train"
        for r in rows
        if r["source"] == "constructed-condition-v1"
    ]
    if any(canonical_request(r) in prior for r in new_rows):
        raise ValueError("A new condition record repeats historical material")
    root.mkdir(parents=True)
    files = {}
    for role, rows in parts.items():
        random.Random(2026092103).shuffle(rows)
        files[role] = root / ("validation.jsonl" if role == "development" else f"{role}.jsonl")
        write_jsonl(files[role], rows)
        if (
            revision == 2
            and role != "train"
            and digest(files[role]) != digest(ROOT / files[role].name)
        ):
            raise ValueError("The follow-up must preserve development, calibration and test bytes")
    shutil.copy2(__file__, root / "builder.py")
    manifest = register(
        root,
        files,
        {
            "scope": "Balanced precondition repair with eight-task development retention and independent condition-rule holdout.",
            "parent": "results/phase3/instruction-seed2026/adapter",
            "revision": revision,
            "numeric_replay": "data/phase3/coverage-v1/train.jsonl" if revision == 2 else None,
            "retained_baseline": "results/phase2/v3/selected/adapter",
            "style_partition": {"train": [0, 1, 2, 3], "development": [4, 5], "test": [6, 7]},
            "replay": "data/phase3/instruction-v2/train.jsonl",
            "protected_previous_holdouts": {str(p): digest(p) for p in heldout_paths},
            "families": {
                role: dict(Counter(r["family"] for r in rows)) for role, rows in parts.items()
            },
            "builder_sha256": digest(root / "builder.py"),
            "independent_scope": "The new test measures condition-rule transfer across source values, field sets and wording. Existing natural-task cohorts are regression evidence.",
        },
    )
    print(json.dumps(manifest["partitions"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", type=int, choices=(1, 2), default=1)
    prepare(parser.parse_args().revision)
