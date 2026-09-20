"""Separate numeric-rule training regressions from wording and field-name effects."""

import argparse
import copy
import gc
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from necro.config import Settings
from necro.evaluation import read_examples, run_evaluation
from necro.experiment_guard import digest, register
from necro.training.analysis.phase2_analysis import cluster_intervals
from necro.training.data.phase2_data import OPS
from necro.training.data.training_data import write_jsonl

DATA = Path("data/phase3/numeric-regression-v1")
OUTPUT = Path("results/phase3/numeric-regression-v1")
MODELS = {
    "predecessor": ("results/phase2/v3/selected/adapter", "results/phase3/baseline"),
    "coverage": ("results/phase3/coverage-seed2026/adapter", "results/phase3/coverage-seed2026"),
    "instruction2026": (
        "results/phase3/instruction-seed2026/adapter",
        "results/phase3/instruction-seed2026",
    ),
    "instruction2027": (
        "results/phase3/instruction-seed2027/adapter",
        "results/phase3/instruction-seed2027",
    ),
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def rule_metadata(row):
    instruction = row["request"]["questions"]["decision"]["instructions"]
    words = {word: code for code, (_, zh, en) in OPS.items() for word in (zh, en)}
    match = re.search(
        r"`([^`]+)`\s*(?:is\s+|是否)?(" + "|".join(map(re.escape, words)) + r")\s*`([^`]+)`",
        instruction,
    )
    if not match:
        raise ValueError(f"Unrecognized numeric predicate: {row['id']}")
    a, word, b = match.groups()
    fields = set(re.findall(r"`([^`]+)`", instruction))
    remaining = fields - {a, b}
    if len(remaining) > 1:
        raise ValueError(f"Ambiguous gate: {row['id']}")
    gate = next(iter(remaining), None)
    state = row["request"]["state"]
    negated = bool(re.search(r"(?:ineligible|a failure|fail|不合格|不通过)[?？]$", instruction))
    missing = [key for key in (a, b, gate) if key and key not in state]
    case = (
        "missing" if missing else ("gate_false" if gate and state[gate] is not True else "complete")
    )
    relation = (
        "missing"
        if a not in state or b not in state
        else ("lt" if state[a] < state[b] else "gt" if state[a] > state[b] else "eq")
    )
    eligible = (
        not missing
        and (gate is None or state[gate] is True)
        and OPS[words[word]][0](state[a], state[b])
    )
    expected = not eligible if negated else bool(eligible)
    if expected != row["expected"]["decision"]:
        raise ValueError(f"Independent rule calculation disagrees with label: {row['id']}")
    return {
        "operator": words[word],
        "fields": [a, b, gate],
        "negated": negated,
        "case": case,
        "missing_fields": missing,
        "relation": relation,
        "language": row["language"],
        "atomic": gate is None,
    }


def iff_instruction(fields, code, language, negated):
    a, b, gate = fields
    _, zh, en = OPS[code]
    if language == "zh":
        target = "不合格" if negated else "合格"
        return (
            f"当且仅当 `{a}` {zh} `{b}` 且 `{gate}` 为 true 时合格；"
            f"其余或缺少字段时不合格。现在是否{target}？"
        )
    target = "ineligible" if negated else "eligible"
    return (
        f"The record is eligible if and only if `{a}` is {en} `{b}` and `{gate}` is true. "
        "Otherwise, or if required fields are missing, it is ineligible. "
        f"Is this record {target}?"
    )


def prepare():
    if DATA.exists() or OUTPUT.exists():
        raise ValueError("Diagnostic cohort already exists")
    parent = Path("data/phase3/coverage-v1/test.jsonl")
    original = [r for r in read_examples(parent) if r["family"] == "numeric_rule"]
    variants = []
    for row in original:
        meta = rule_metadata(row)
        if meta["fields"] != ["stock", "requested", "warehouse_open"]:
            raise ValueError("Unexpected test field names")
        for fields_name, fields in (
            ("original", meta["fields"]),
            ("development", ["available_slots", "required_slots", "enabled"]),
            ("neutral", ["value_a", "value_b", "gate"]),
        ):
            mapping = dict(zip(meta["fields"], fields, strict=True))
            for wording in ("guard_first", "iff"):
                new = copy.deepcopy(row)
                new["parent_id"] = row["id"]
                new["diagnostic_condition"] = f"{fields_name}/{wording}"
                new["id"] += f"/diagnostic/{fields_name}/{wording}"
                new["request"]["state"] = {
                    mapping[k]: v for k, v in row["request"]["state"].items()
                }
                instruction = row["request"]["questions"]["decision"]["instructions"]
                for old, current in mapping.items():
                    instruction = instruction.replace(f"`{old}`", f"`{current}`")
                if wording == "iff":
                    instruction = iff_instruction(
                        fields, meta["operator"], row["language"], meta["negated"]
                    )
                new["request"]["questions"]["decision"]["instructions"] = instruction
                rule_metadata(new)
                variants.append(new)
    DATA.mkdir(parents=True)
    OUTPUT.mkdir(parents=True)
    write_jsonl(DATA / "regression.jsonl", variants)
    manifest = register(
        DATA,
        {"regression": DATA / "regression.jsonl"},
        {
            "scope": "Post-test diagnostic only. Original test is exposed. "
            "No independent acceptance.",
            "source_test": str(parent),
            "source_test_sha256": digest(parent),
            "factors": {
                "fields": ["original", "development", "neutral"],
                "wording": ["guard_first", "iff"],
            },
            "original_numeric_rows": len(original),
            "label_audit": "Recompute comparisons, required fields, gate and question polarity "
            "in every condition.",
            "models": MODELS,
        },
    )
    write_json(OUTPUT / "plan.json", manifest)
    inventories = {}
    for source in (
        "data/phase2/operator-contrast-v1/train.jsonl",
        "data/phase2/refinement-v2/train.jsonl",
        "data/phase3/coverage-v1/train.jsonl",
        "data/phase3/instruction-v2/train.jsonl",
        "data/phase3/coverage-v1/validation.jsonl",
        "data/phase3/coverage-v1/calibration.jsonl",
        str(parent),
    ):
        rows = [r for r in read_examples(Path(source)) if r["family"] == "numeric_rule"]
        meta = [rule_metadata(r) for r in rows]
        inventories[source] = {
            "sha256": digest(Path(source)),
            "rows": len(rows),
            "source_groups": len({r["group_id"] for r in rows}),
            **{
                key: dict(Counter(str(m[key]) for m in meta))
                for key in (
                    "operator",
                    "fields",
                    "negated",
                    "case",
                    "relation",
                    "language",
                    "atomic",
                )
            },
            "independent_label_mismatches": 0,
        }
    write_json(OUTPUT / "data-audit.json", inventories)
    print(
        json.dumps(
            {
                "diagnostic_rows": len(variants),
                "original_rows": len(original),
                "audited_rows": sum(x["rows"] for x in inventories.values()),
            }
        ),
        flush=True,
    )


def run(name):
    import torch

    adapter, run_path = MODELS[name]
    temperatures = json.loads((Path(run_path) / "calibration-brier-fit.json").read_text())[
        "temperatures"
    ]
    target = OUTPUT / name
    if (target / "summary.json").exists():
        raise ValueError("Diagnostic model results already exist")
    settings = Settings(
        adapter=adapter, device="cuda", **{f"{k}_temperature": v for k, v in temperatures.items()}
    )
    report = run_evaluation(DATA / "regression.jsonl", target, settings)
    print(json.dumps({"model": name, "accuracy": report["overall"]["accuracy"]}), flush=True)
    gc.collect()
    torch.cuda.empty_cache()


def analyze():
    examples = {r["id"]: r for r in read_examples(DATA / "regression.jsonl")}
    result = {}
    paired = {}
    for name in MODELS:
        groups = defaultdict(list)
        rows = read_examples(OUTPUT / name / "judgments.jsonl")
        conditions = defaultdict(list)
        for row in rows:
            example = examples[row["id"]]
            meta = rule_metadata(example)
            condition = example["diagnostic_condition"]
            conditions[condition].append({**row, "id": example["parent_id"]})
            groups[condition].append(row)
            for field in ("language", "operator", "case", "negated", "relation"):
                groups[f"{condition}/{field}/{meta[field]}"].append(row)
        result[name] = {
            key: {
                "count": len(values),
                "correct": sum(r["correct"] for r in values),
                "accuracy": sum(r["correct"] for r in values) / len(values),
            }
            for key, values in sorted(groups.items())
        }
        paired[name] = conditions
    write_json(OUTPUT / "analysis.json", result)
    contrasts = {}
    for name, current, previous, condition_a, condition_b in (
        (
            "coverage_minus_predecessor",
            "coverage",
            "predecessor",
            "original/guard_first",
            "original/guard_first",
        ),
        (
            "instruction2026_minus_coverage",
            "instruction2026",
            "coverage",
            "original/guard_first",
            "original/guard_first",
        ),
        (
            "instruction2026_minus_predecessor",
            "instruction2026",
            "predecessor",
            "original/guard_first",
            "original/guard_first",
        ),
        (
            "instruction2026_wording_effect",
            "instruction2026",
            "instruction2026",
            "original/iff",
            "original/guard_first",
        ),
        (
            "instruction2026_development_fields_effect",
            "instruction2026",
            "instruction2026",
            "development/guard_first",
            "original/guard_first",
        ),
    ):
        left = paired[current][condition_a]
        right = paired[previous][condition_b]
        if [r["id"] for r in left] != [r["id"] for r in right]:
            raise ValueError("Diagnostic pairs do not align")
        contrasts[name] = {
            "accuracy_delta": sum(
                int(a["correct"]) - int(b["correct"]) for a, b in zip(left, right, strict=True)
            )
            / len(left),
            "lost_correct_answers": sum(
                b["correct"] and not a["correct"] for a, b in zip(left, right, strict=True)
            ),
            "gained_correct_answers": sum(
                a["correct"] and not b["correct"] for a, b in zip(left, right, strict=True)
            ),
            "uncertainty": cluster_intervals(left, right),
        }
    write_json(OUTPUT / "paired-comparisons.json", contrasts)
    print(
        json.dumps(
            {
                name: {k: v for k, v in values.items() if k.count("/") == 1}
                for name, values in result.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "analyze", *MODELS])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "analyze":
        analyze()
    else:
        run(args.action)
