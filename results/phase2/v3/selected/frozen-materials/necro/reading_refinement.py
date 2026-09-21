"""只增加未见的阅读判断段落，其他任务保持回放，供一次定向对照使用。"""

import json
import random
from collections import Counter
from pathlib import Path

from necro.evaluation import read_examples
from necro.experiment_guard import canonical_request, digest, normalized_text, register
from necro.phase2_data import grouped_sample, reading
from necro.source_isolation import Sources
from necro.training_data import write_jsonl


def prepare():
    output = Path("data/phase2/reading-v1")
    if output.exists():
        raise ValueError("阅读改进数据已登记，不可覆盖。")
    strict = Path("data/phase2/strict-v1")
    protected = [
        row
        for role in ("development", "calibration", "test")
        for row in read_examples(strict / f"{role}.jsonl")
    ]
    exposure = read_examples(strict / "exposure.jsonl")
    pool = read_examples(Path("data/phase2/refinement-v2/train.jsonl"))
    numeric = [
        row
        for row in read_examples(Path("data/phase2/operator-contrast-v1/train.jsonl"))
        if row["family"] == "numeric_rule"
    ]
    new = reading("train-expanded")
    sources = Sources(
        [
            *protected,
            *exposure,
            *pool,
            *new,
            *read_examples(Path("data/improvement/source-expanded.jsonl")),
        ]
    )
    new, blocked_sources = sources.exclude(new, sources.all_keys([*protected, *exposure]))
    known_questions = {
        normalized_text(row["request"]["state"]["question"])
        for row in [*protected, *exposure]
        if row["source"] == "google/boolq"
    }
    duplicate_questions = {
        row["group_id"]
        for row in new
        if normalized_text(row["request"]["state"]["question"]) in known_questions
    }
    new = [row for row in new if row["group_id"] not in duplicate_questions]
    # 先检查长度，再抽样；不截断支持标签的证据。
    from transformers import AutoTokenizer

    from necro.backend import single_token_labels
    from necro.training import encode_example

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        revision="2fc06364715b967f1860aea9cf38778875588b17",
        local_files_only=True,
    )
    alphabet, _ = single_token_labels(tokenizer)
    too_long = {
        row["group_id"]
        for row in new
        if len(encode_example(tokenizer, alphabet, row, max_length=32768)["input_ids"]) > 2048
    }
    new = grouped_sample([row for row in new if row["group_id"] not in too_long], 1000, 2028)
    if len(new) < 1000:
        raise ValueError("去除来源重叠后，新增阅读材料不足 1000 条。")
    quotas = {
        "inference": 300,
        "intent": 300,
        "paraphrase": 200,
        "reading_boolean": 200,
        "numeric_rule": 320,
        "ordinal_rule": 120,
        "candidate_extraction": 100,
        "candidate_retrieval": 60,
    }
    replay_pool, replay_removed = sources.exclude([*pool, *numeric], sources.all_keys(protected))
    replay = [
        row
        for family, count in quotas.items()
        for row in grouped_sample(
            [row for row in replay_pool if row["family"] == family], count, 824
        )
    ]
    train = list({canonical_request(row): row for row in [*new, *replay]}.values())
    random.Random(2026).shuffle(train)
    if sources.all_keys(train) & sources.all_keys(protected):
        raise ValueError("阅读训练仍存在来源重叠。")
    lengths = [len(encode_example(tokenizer, alphabet, row)["input_ids"]) for row in train]
    output.mkdir(parents=True)
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "validation.jsonl", read_examples(strict / "development.jsonl"))
    manifest = register(
        output,
        {"train": output / "train.jsonl", "development": output / "validation.jsonl"},
        {
            "hypothesis": "Additional unseen human-labeled BoolQ passages improve reading judgment "
            "while replay retains the other seven tasks.",
            "families": dict(Counter(row["family"] for row in train)),
            "new_reading_rows": len(new),
            "new_reading_source_groups": len({row["group_id"] for row in new}),
            "protected_manifest_sha256": digest(strict / "experiment.json"),
            "snapshot_sha256": digest(Path("data/phase2/boolq-train-expanded.json")),
            "replay_manifests": {
                str(path): digest(path)
                for path in (
                    Path("data/phase2/refinement-v2/experiment.json"),
                    Path("data/phase2/operator-contrast-v1/experiment.json"),
                )
            },
            "excluded_sources": blocked_sources,
            "excluded_duplicate_questions": sorted(duplicate_questions),
            "excluded_length_groups": sorted(too_long),
            "excluded_replay_groups": replay_removed,
            "max_input_tokens": max(lengths),
            "notes": "Prepared pool does not mean it has been trained. "
            "Labels are original human BoolQ "
            "labels; no Jev labels or final predictions are used.",
        },
    )
    print(
        json.dumps(
            {
                key: manifest["protocol"][key]
                for key in (
                    "families",
                    "new_reading_rows",
                    "new_reading_source_groups",
                    "max_input_tokens",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    prepare()
