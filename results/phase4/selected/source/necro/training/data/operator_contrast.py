"""相同状态下切换规则，避免从字段名或状态值猜测比较操作。"""

import json
import random
from collections import Counter
from pathlib import Path

from necro.evaluation import read_examples
from necro.experiment_guard import canonical_request, digest, register
from necro.training.data.phase2_data import OPS, example, grouped_sample
from necro.training.data.source_isolation import Sources
from necro.training.data.training_data import write_jsonl

FIELDS = [
    ("credit", "minimum", "account_active"),
    ("weight", "limit", "inspection_passed"),
    ("duration", "target", "license_valid"),
    ("quantity", "quota", "manager_agreed"),
    ("distance", "cutoff", "sensor_ready"),
    ("temperature", "reference", "power_on"),
    ("points", "benchmark", "entry_valid"),
    ("height", "bar", "route_clear"),
]


def rule_text(fields, code, language, negated, style=0):
    a, b, gate = fields
    _, zh, en = OPS[code]
    if language == "zh":
        prefix = (
            f"当且仅当 `{a}` {zh} `{b}` 且 `{gate}` 为 true 时合格；其余或缺少字段时不合格。"
            if style == 0
            else f"通过条件：`{a}` {zh} `{b}`，并且 `{gate}` 为 true；"
            "两项条件必须都满足。缺少任意必需字段或条件未满足则不通过。"
        )
        target = (
            ("不合格" if negated else "合格") if style == 0 else ("不通过" if negated else "通过")
        )
        return prefix + f"当前记录是否{target}？"
    prefix = (
        f"The record is eligible if and only if `{a}` is {en} `{b}` and `{gate}` is true. "
        "Otherwise, or if required fields are missing, it is ineligible. "
        if style == 0
        else f"A record passes when both conditions hold: `{a}` is {en} `{b}`, "
        f"and `{gate}` is true. "
        "If either condition fails or a required field is missing, the record fails. "
    )
    target = (
        ("ineligible" if negated else "eligible")
        if style == 0
        else ("a failure" if negated else "a pass")
    )
    return prefix + f"Is this record {target}?"


def construct():
    rng = random.Random(20260922)
    thresholds = rng.sample(range(5, 950), 60)
    rows = []
    for i, threshold in enumerate(thresholds):
        a, b, gate = fields = FIELDS[i % len(FIELDS)]
        delta = (-1, 0, 1)[i % 3] * (0.5 if i % 2 else 1)
        state = {a: threshold + delta, b: threshold, gate: True}
        group = f"operator-contrast/{i}"
        for code, (fn, zh, en) in OPS.items():
            relation = bool(fn(state[a], state[b]))
            for language in ("en", "zh"):
                # 完全相同的状态支持所有 5 个比较算子，只有规则文本发生变化。
                for negated in (False, True):
                    rows.append(
                        example(
                            f"{group}/{code}/{language}/compound/{int(negated)}",
                            group,
                            "numeric_rule",
                            language,
                            state,
                            {
                                "type": "noul",
                                "instructions": rule_text(fields, code, language, negated, i % 2),
                            },
                            not relation if negated else relation,
                            "constructed-operator-contrast",
                            "exact-oracle",
                        )
                    )
                atomic = (
                    f"`{a}` 是否{zh} `{b}`？" if language == "zh" else (f"Is `{a}` {en} `{b}`?")
                )
                rows.append(
                    example(
                        f"{group}/{code}/{language}/atomic",
                        group,
                        "numeric_rule",
                        language,
                        state,
                        {"type": "noul", "instructions": atomic},
                        relation,
                        "constructed-operator-contrast",
                        "exact-oracle",
                    )
                )
        if i % 3 == 0:
            code = list(OPS)[(i // 3) % 5]
            for kind in ("gate_false", "missing"):
                altered = dict(state)
                if kind == "gate_false":
                    altered[gate] = False
                else:
                    altered.pop((a, b, gate)[(i // 3) % 3])
                for language in ("en", "zh"):
                    for negated in (False, True):
                        rows.append(
                            example(
                                f"{group}/{code}/{language}/{kind}/{int(negated)}",
                                group,
                                "numeric_rule",
                                language,
                                altered,
                                {
                                    "type": "noul",
                                    "instructions": rule_text(
                                        fields, code, language, negated, i % 2
                                    ),
                                },
                                negated,
                                "constructed-operator-contrast",
                                "exact-oracle",
                            )
                        )
    return rows


def prepare():
    root = Path("data/phase2/operator-contrast-v1")
    if root.exists():
        raise ValueError("算子对照数据已登记，不可覆盖。")
    strict = Path("data/phase2/strict-v1")
    protected = [
        row
        for role in ("development", "calibration", "test")
        for row in read_examples(strict / f"{role}.jsonl")
    ]
    pool = read_examples(Path("data/phase2/refinement-v2/train.jsonl"))
    sources = Sources(
        [*pool, *protected, *read_examples(Path("data/improvement/source-expanded.jsonl"))]
    )
    pool, removed = sources.exclude(pool, sources.all_keys(protected))
    counts = {
        "inference": 300,
        "intent": 300,
        "paraphrase": 100,
        "reading_boolean": 120,
        "candidate_extraction": 60,
        "candidate_retrieval": 40,
        "ordinal_rule": 80,
        "numeric_rule": 80,
    }
    replay = [
        row
        for family, count in counts.items()
        for row in grouped_sample([row for row in pool if row["family"] == family], count, 834)
    ]
    constructed = construct()
    if sources.all_keys(constructed) & sources.all_keys(protected):
        raise ValueError("新规则与保护材料存在来源重叠。")
    train = list({canonical_request(row): row for row in [*replay, *constructed]}.values())
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
    lengths = [len(encode_example(tokenizer, alphabet, row)["input_ids"]) for row in train]
    root.mkdir(parents=True)
    write_jsonl(root / "train.jsonl", train)
    write_jsonl(root / "validation.jsonl", read_examples(strict / "development.jsonl"))
    manifest = register(
        root,
        {"train": root / "train.jsonl", "development": root / "validation.jsonl"},
        {
            "hypothesis": "Operator-switch counterfactuals share identical states; "
            "atomic comparison "
            "and compound judgments supervise the relationship instead of field-name priors.",
            "protected_manifest_sha256": digest(strict / "experiment.json"),
            "replay_manifest_sha256": digest(Path("data/phase2/refinement-v2/experiment.json")),
            "removed_replay_source_groups": removed,
            "families": dict(Counter(row["family"] for row in train)),
            "max_input_tokens": max(lengths),
            "constructed_source_groups": len({row["group_id"] for row in constructed}),
            "notes": "Each group contains related counterfactuals from a shared source. "
            "Development, calibration and test text/labels are not used to generate training rows.",
        },
    )
    print(json.dumps(manifest["protocol"], indent=2), flush=True)


if __name__ == "__main__":
    prepare()
