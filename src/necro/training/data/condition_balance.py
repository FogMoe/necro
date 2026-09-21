"""控制条件记录的语义权重，并登记与历史材料隔离的新条件测试。"""

# 让完整措辞保持在同一行，便于复核中英文规则。
# ruff: noqa: E501

import argparse
import copy
import math
import random
import shutil
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

from necro.diagnostics.numeric_regression import rule_metadata, write_json
from necro.evaluation import read_examples
from necro.experiment_guard import canonical_request, digest, register
from necro.training.data.condition_refinement import audit_condition_matrix
from necro.training.data.phase2_data import OPS, example
from necro.training.data.training_data import write_jsonl


def semantic_outcome(row):
    meta = rule_metadata(row)
    answer = row["expected"]["decision"]
    return not answer if meta["negated"] else answer


def balance(rows):
    result = copy.deepcopy(rows)
    matrix = [r for r in result if "condition_matrix_case" in r]
    audit_condition_matrix(matrix)
    outcomes = Counter(semantic_outcome(r) for r in matrix)
    for row in matrix:
        row["training_weight"] = len(matrix) / (2 * outcomes[semantic_outcome(row)])
    audit_balance(result)
    return result


def audit_balance(rows):
    cells = defaultdict(lambda: defaultdict(float))
    cases = defaultdict(lambda: {"presentations": 0, "weight": 0.0})
    for row in rows:
        weight = row.get("training_weight", 1.0)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Training weights must be finite and positive")
        if "condition_matrix_case" not in row:
            if weight != 1:
                raise ValueError("Replay weight must remain one")
            continue
        meta = rule_metadata(row)
        key = (row["condition_style"], meta["operator"], row["language"], meta["negated"])
        cells[key][semantic_outcome(row)] += weight
        case = row["condition_matrix_case"]
        cases[case]["presentations"] += 1
        cases[case]["weight"] += weight
    matrix = [r for r in rows if "condition_matrix_case" in r]
    audit_condition_matrix(matrix)
    if any(not math.isclose(v[True], v[False]) for v in cells.values()):
        raise ValueError("Eligibility must be balanced within style/operator/language/polarity")
    total = sum(r.get("training_weight", 1.0) for r in rows)
    if not math.isclose(total, len(rows)):
        raise ValueError("Total training weight must preserve the presentation budget")
    return {
        "presentations": len(rows),
        "matrix_presentations": len(matrix),
        "semantic_presentations": dict(Counter(str(semantic_outcome(r)) for r in matrix)),
        "semantic_weight": {
            str(outcome): sum(
                r["training_weight"] for r in matrix if semantic_outcome(r) == outcome
            )
            for outcome in (False, True)
        },
        "answer_weight": {
            str(answer): sum(
                r["training_weight"] for r in matrix if r["expected"]["decision"] == answer
            )
            for answer in (False, True)
        },
        "joint_cells": len(cells),
        "cases": dict(cases),
        "replay_weight": len(rows) - len(matrix),
        "total_weight": total,
    }


def resample(rows):
    """固定总量，按真实资格分配呈现次数；每个无效条件仍有覆盖。"""
    matrix = [r for r in rows if "condition_matrix_case" in r]
    audit_condition_matrix(matrix)
    cells = defaultdict(list)
    for row in matrix:
        meta = rule_metadata(row)
        cells[row["condition_style"], meta["operator"], row["language"], meta["negated"]].append(
            row
        )
    result = []
    for (style, code, _language, _negated), pool in sorted(cells.items()):
        eligible = sorted((r for r in pool if semantic_outcome(r)), key=lambda r: r["id"])
        negatives = defaultdict(list)
        for row in pool:
            if not semantic_outcome(row):
                negatives[row["condition_matrix_case"], row["condition_relation_true"]].append(row)
        # 各语言与问法使用相同的来源抽样，保留翻译/极性配对。
        rng = random.Random(2026092153 + style * 5 + list(OPS).index(code))
        selected = eligible * 5
        remainder = []
        for _, candidates in sorted(negatives.items()):
            candidates = sorted(candidates, key=lambda r: r["id"])
            rng.shuffle(candidates)
            selected.extend(candidates[:2])
            remainder.extend(candidates[2:])
        rng.shuffle(remainder)
        selected.extend(remainder[:2])
        for position, source in enumerate(selected):
            row = copy.deepcopy(source)
            row.pop("training_weight", None)
            row["id"] += f"/presentation-{position}"
            result.append(row)
    result += copy.deepcopy([r for r in rows if "condition_matrix_case" not in r])
    random.Random(2026092154).shuffle(result)
    audit_sampling(result)
    return result


def audit_sampling(rows):
    cells, combinations = defaultdict(Counter), Counter()
    for row in rows:
        if row.get("training_weight", 1) != 1:
            raise ValueError("Resampled training uses unit weights")
        if "condition_matrix_case" not in row:
            continue
        meta = rule_metadata(row)
        cell = (row["condition_style"], meta["operator"], row["language"], meta["negated"])
        cells[cell][semantic_outcome(row)] += 1
        combinations[*cell, row["condition_matrix_case"], row["condition_relation_true"]] += 1
    required = set(
        product(
            range(4),
            OPS,
            ("en", "zh"),
            (False, True),
            ("complete", "disabled", "missing-0", "missing-1", "missing-2"),
            (False, True),
        )
    )
    if set(combinations) != required or min(combinations.values()) < 2:
        raise ValueError("Resampling must retain every condition combination")
    if any(counts != {False: 20, True: 20} for counts in cells.values()):
        raise ValueError(
            "Each joint cell must contain twenty eligible and twenty ineligible records"
        )
    matrix = [r for r in rows if "condition_matrix_case" in r]
    return {
        "presentations": len(rows),
        "matrix_presentations": len(matrix),
        "semantic_presentations": dict(Counter(str(semantic_outcome(r)) for r in matrix)),
        "answer_presentations": dict(Counter(str(r["expected"]["decision"]) for r in matrix)),
        "unique_matrix_requests": len({canonical_request(r) for r in matrix}),
        "joint_cells": len(cells),
        "combinations": len(combinations),
        "minimum_presentations_per_combination": min(combinations.values()),
        "cases": dict(Counter(r["condition_matrix_case"] for r in matrix)),
        "replay_presentations": len(rows) - len(matrix),
    }


def transfer_instructions(fields, code, language, negated, style):
    a, b, gate = fields
    _, zh, en = OPS[code]
    comparison = f"`{a}` {zh} `{b}`" if language == "zh" else f"`{a}` is {en} `{b}`"
    names = f"`{a}`, `{b}`, `{gate}`"
    if language == "zh":
        variants = [
            f"合格需要全部满足：提供 {names}、`{gate}` 为 true，以及 {comparison}。否则不合格。",
            f"请按照这些要求决定资格：{names} 缺一不可；`{gate}` 必须为 true；还须 {comparison}。全部满足才合格。",
            f"资格审核有两步。第一步检查 {names} 是否齐全且 `{gate}` 为 true；不满足即不合格。通过第一步后，若 {comparison} 则合格，否则不合格。",
            f"只有同时具备以下条件的记录才合格，满足全部条件的记录均合格：{comparison}；`{gate}` 为 true；{names} 全部提供。",
        ]
        return variants[style] + ("这条记录是否不合格？" if negated else "这条记录是否合格？")
    variants = [
        f"To qualify, all requirements must hold: {names} are supplied, `{gate}` is true, and {comparison}. Otherwise the record is ineligible. ",
        f"Decide eligibility using these requirements: none of {names} may be absent; `{gate}` must be true; and {comparison} must hold. Meeting every requirement makes the record eligible. ",
        f"Eligibility review has two steps. First check that {names} are all present and `{gate}` is true; failure means ineligible. After passing this step, the record is eligible if {comparison}, and ineligible otherwise. ",
        f"Only records meeting every condition below are eligible, and every record meeting all of them is eligible: {comparison}; `{gate}` is true; all of {names} are supplied. ",
    ]
    return variants[style] + ("Is the record ineligible?" if negated else "Is the record eligible?")


def transfer_rows(role):
    if role not in {"development", "test"}:
        raise ValueError("Transfer role must be development or test")
    rng = random.Random(2026092151 if role == "development" else 2026092152)
    count = 8 if role == "development" else 24
    rows = []
    prefix = "dev" if role == "development" else "holdout"
    for i, value in enumerate(rng.sample(range(1100, 9900), count)):
        fields = (
            f"{prefix}_observed_{i % 4}",
            f"{prefix}_required_{i % 4}",
            f"{prefix}_allowed_{i % 4}",
        )
        a, b, gate = fields
        value *= -1 if i % 3 == 0 else 1
        step = (0.25, 1, 7)[i % 3]
        group = f"condition-balance/{role}/{i}"
        style = i % 2 + (0 if role == "development" else 2)
        for code, (fn, _, _) in OPS.items():
            offsets = {
                "ge": (0, -step),
                "gt": (step, 0),
                "le": (0, step),
                "lt": (-step, 0),
                "eq": (0, step),
            }
            for truth, delta in zip((True, False), offsets[code], strict=True):
                state = {a: value + delta, b: value, gate: True}
                assert bool(fn(state[a], state[b])) == truth
                cases = [("complete", state), ("disabled", {**state, gate: False})]
                cases += [
                    (f"missing-{j}", {k: v for k, v in state.items() if k != field})
                    for j, field in enumerate(fields)
                ]
                for case, current in cases:
                    eligible = case == "complete" and truth
                    for language, negated in product(("en", "zh"), (False, True)):
                        row = example(
                            f"{group}/{code}/{case}-{int(truth)}/{language}/{int(negated)}",
                            group,
                            "numeric_rule",
                            language,
                            copy.deepcopy(current),
                            {
                                "type": "noul",
                                "instructions": transfer_instructions(
                                    fields, code, language, negated, style
                                ),
                            },
                            not eligible if negated else eligible,
                            "constructed-condition-balance",
                            "exact-oracle",
                        )
                        row.update(
                            condition_style=style + 8,
                            condition_matrix_case=case,
                            condition_relation_true=truth,
                        )
                        rule_metadata(row)
                        rows.append(row)
    return rows


def prepare(root, method="weight", reuse_evaluation=None):
    if root.exists():
        raise ValueError("Choose a new condition-balance directory")
    previous = Path("data/phase4/condition-v3")
    training = read_examples(previous / "train.jsonl")
    parts = {
        "train": balance(training) if method == "weight" else resample(training),
        "development": read_examples(previous / "validation.jsonl") + transfer_rows("development"),
        "calibration": read_examples(previous / "calibration.jsonl"),
        "test": transfer_rows("test"),
    }
    # 同时检查历史训练、校准和已暴露测试，避免新生成材料撞到任一旧请求。
    if reuse_evaluation is None:
        historical = {
            canonical_request(r)
            for path in Path("data").rglob("*.jsonl")
            for r in read_examples(path)
        }
        if any(
            canonical_request(r) in historical
            for role in ("development", "test")
            for r in parts[role]
            if r["source"] == "constructed-condition-balance"
        ):
            raise ValueError("New transfer requests overlap historical material")
    else:
        from necro.experiment_guard import verify

        verify(reuse_evaluation, "train")
        if (reuse_evaluation / "selection.json").exists():
            raise ValueError("Evaluation reuse must precede candidate selection")
        for role, filename in (
            ("development", "validation.jsonl"),
            ("calibration", "calibration.jsonl"),
            ("test", "test.jsonl"),
        ):
            parts[role] = read_examples(reuse_evaluation / filename)
    root.mkdir(parents=True)
    files = {}
    for role, rows in parts.items():
        files[role] = root / ("validation.jsonl" if role == "development" else f"{role}.jsonl")
        write_jsonl(files[role], rows)
        if (
            reuse_evaluation is not None
            and role != "train"
            and digest(files[role]) != digest(reuse_evaluation / files[role].name)
        ):
            raise ValueError("Evaluation bytes must remain unchanged between trials")
    shutil.copy2(__file__, root / "builder.py")
    audit = audit_balance(parts["train"]) if method == "weight" else audit_sampling(parts["train"])
    register(
        root,
        files,
        {
            "scope": "Semantic condition repair; exposed cohorts are regression evidence; test is sealed until selection.",
            "method": method,
            "parent": "results/phase3/instruction-seed2026/adapter",
            "source_training_sha256": digest(previous / "train.jsonl"),
            "builder_sha256": digest(root / "builder.py"),
            "balance": audit,
            "transfer_styles": {"development": [8, 9], "test": [10, 11]},
        },
    )
    write_json(root / "balance-audit.json", audit)
    print(audit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/phase5/condition-balance-v1"))
    parser.add_argument("--method", choices=("weight", "sample"), default="weight")
    parser.add_argument("--reuse-evaluation", type=Path)
    args = parser.parse_args()
    prepare(args.output, args.method, args.reuse_evaluation)
