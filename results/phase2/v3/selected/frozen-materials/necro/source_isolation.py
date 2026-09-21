"""按原句、前提和候选文档隔离；只依据来源关系，完全不读取模型预测。"""

import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from necro.evaluation import read_examples
from necro.experiment_guard import digest, normalized_text, register
from necro.phase2_data import example, load
from necro.training_data import write_jsonl


def signature(value):
    return hashlib.sha256(normalized_text(value).encode()).hexdigest()


class Sources:
    def __init__(self, rows, english_paws=None):
        self.english_paws = (
            english_paws
            if english_paws is not None
            else {
                (split, str(row["id"])): row
                for split in ("train", "validation", "test")
                for row in load(f"paws-en-{split}")
            }
        )
        self.premises = {
            row.get("source_group_id", row["group_id"]): row["request"]["state"]["premise"]
            for row in rows
            if row.get("source") == "facebook/xnli" and row["language"] == "en"
        }

    def keys(self, row):
        source = row["source"]
        if source == "google-research-datasets/paws-x":
            _, split, _language, identifier = row["id"].split("/")[:4]
            original = self.english_paws[split, identifier]
            return {
                "paws-sentence/" + signature(original[key]) for key in ("sentence1", "sentence2")
            }
        if source == "facebook/xnli":
            source_group = row.get("source_group_id", row["group_id"])
            premise = self.premises.get(source_group)
            return {"xnli-premise/" + signature(premise)} if premise else {source_group}
        if source == "BeIR/scifact":
            return {
                "scifact-doc/" + str(key)
                for key in row["request"]["questions"]["decision"]["criteria"]
            }
        return {row.get("source_group_id", row["group_id"])}

    def all_keys(self, rows):
        return set().union(*(self.keys(row) for row in rows)) if rows else set()

    def exclude(self, rows, protected):
        blocked = {row["group_id"] for row in rows if self.keys(row) & protected}
        return [row for row in rows if row["group_id"] not in blocked], sorted(blocked)

    def regroup(self, rows):
        """合并共享来源的连通分量，bootstrap 使用同一个来源组。"""
        parent = list(range(len(rows)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        seen = {}
        for i, row in enumerate(rows):
            for key in self.keys(row):
                if key in seen:
                    parent[find(i)] = find(seen[key])
                seen[key] = i
        components = defaultdict(set)
        for i, row in enumerate(rows):
            components[find(i)].update(self.keys(row))
        result = copy.deepcopy(rows)
        for i, row in enumerate(result):
            row.setdefault("source_group_id", row["group_id"])
            row["group_id"] = "source-component/" + signature(sorted(components[find(i)]))
            row["content_group"] = row["group_id"]
        return result


def unseen_retrieval(blocked, count=80):
    rng = random.Random(20260921)
    corpus = {str(row["_id"]): row for row in load("scifact-corpus")}
    queries = {str(row["_id"]): row["text"] for row in load("scifact-queries")}
    positive = defaultdict(set)
    for row in load("scifact-test"):
        if row["score"] > 0:
            positive[str(row["query-id"])].add(str(row["corpus-id"]))
    available = sorted(
        key
        for key, row in corpus.items()
        if "scifact-doc/" + key not in blocked and len(row["text"]) < 2100
    )
    qids = sorted(positive)
    rng.shuffle(qids)
    result = []
    for qid in qids:
        relevant = positive[qid]
        if any("scifact-doc/" + key in blocked for key in relevant):
            continue
        acceptable = sorted(relevant.intersection(available))
        if not acceptable:
            continue
        gold = acceptable[0]
        keys = [gold, *rng.sample([key for key in available if key not in relevant], 2)]
        rng.shuffle(keys)
        criteria = {
            key: {"title": corpus[key]["title"], "abstract": corpus[key]["text"]} for key in keys
        }
        if sum(len(value["abstract"]) for value in criteria.values()) > 4700:
            continue
        result.append(
            example(
                f"scifact/strict-test/{qid}",
                f"scifact-query/{signature(queries[qid])}",
                "candidate_retrieval",
                "en",
                {"claim": queries[qid]},
                {
                    "type": "choice",
                    "instructions": "Which candidate scientific document is relevant "
                    "evidence for assessing the claim? Select the most relevant document.",
                    "criteria": criteria,
                },
                gold,
                "BeIR/scifact",
                "qrels-with-unjudged-negatives",
            )
        )
        if len(result) == count:
            return result
    raise ValueError("独立文档的检索题数量不足。")


def prepare():
    output = Path("data/phase2/strict-v1")
    refinement = Path("data/phase2/refinement-v2")
    if output.exists() or refinement.exists():
        raise ValueError("严格来源快照已存在，不可覆盖。")
    paths = {
        "pilot": Path("data/lora-pilot/train.jsonl"),
        "expanded": Path("data/improvement/expanded/train.jsonl"),
        "current": Path("data/phase2/experiment-v3/train.jsonl"),
        "proposed": Path("data/phase2/refinement-v1/train.jsonl"),
        "development": Path("data/phase2/experiment-v3/validation.jsonl"),
        "calibration": Path("data/phase2/calibration-v3/calibration.jsonl"),
        "test": Path("data/phase2/final-v3/test.jsonl"),
    }
    data = {key: read_examples(path) for key, path in paths.items()}
    all_rows = [row for rows in data.values() for row in rows]
    sources = Sources([*all_rows, *read_examples(Path("data/improvement/source-expanded.jsonl"))])
    textual_holdout = [
        row
        for role in ("development", "calibration", "test")
        for row in data[role]
        if row["source"] != "BeIR/scifact"
    ]
    proposed, removed_train = sources.exclude(data["proposed"], sources.all_keys(textual_holdout))
    exposure = [*data["pilot"], *data["expanded"], *data["current"], *proposed]
    exposure_keys = sources.all_keys(exposure)
    dev, removed_dev = sources.exclude(data["development"], exposure_keys)
    calib_blocked = exposure_keys | sources.all_keys(data["development"])
    calib, removed_calib = sources.exclude(data["calibration"], calib_blocked)
    test_blocked = calib_blocked | sources.all_keys(data["calibration"])
    test, removed_test = sources.exclude(
        [row for row in data["test"] if row["source"] != "BeIR/scifact"],
        test_blocked,
    )
    test += unseen_retrieval(test_blocked)
    parts = {"exposure": exposure, "development": dev, "calibration": calib, "test": test}
    keys = {role: sources.all_keys(rows) for role, rows in parts.items()}
    for role, values in keys.items():
        for other, other_values in keys.items():
            if role != other and values & other_values:
                raise ValueError(f"仍有跨划分来源：{role}/{other}")
    output.mkdir(parents=True)
    files = {}
    for role, rows in parts.items():
        files[role] = output / f"{role}.jsonl"
        write_jsonl(files[role], sources.regroup(rows))
    manifest = register(
        output,
        files,
        {
            "reason": "Pre-prediction source audit found shared retrieval documents "
            "and PAWS sentences.",
            "selection_change": "Only structural source exclusions; no test predictions available. "
            "Development keeps the original rows except sources already present in training.",
            "upstream_files": {
                key: {"path": str(path), "sha256": digest(path)} for key, path in paths.items()
            },
            "removed_groups": {
                "proposed_train": removed_train,
                "development": removed_dev,
                "calibration": removed_calib,
                "test_non_retrieval": removed_test,
            },
            "families": {
                role: dict(Counter(row.get("family", row["source"]) for row in rows))
                for role, rows in parts.items()
                if role != "exposure"
            },
            "retrieval": "80 freshly sampled official test queries with all candidate documents "
            "unseen in training, original development and original calibration. "
            "Negatives remain unjudged.",
            "notes": "Exposure includes ancestors and the prospective refinement replay. "
            "Connected source components determine statistical groups.",
        },
    )
    refinement.mkdir(parents=True)
    write_jsonl(refinement / "train.jsonl", proposed)
    write_jsonl(refinement / "validation.jsonl", sources.regroup(dev))
    register(
        refinement,
        {"train": refinement / "train.jsonl", "development": refinement / "validation.jsonl"},
        {
            "strict_holdouts_manifest_sha256": digest(output / "experiment.json"),
            "source": str(paths["proposed"]),
            "removed_groups": removed_train,
        },
    )
    print(
        json.dumps(
            {"partitions": manifest["partitions"], "families": manifest["protocol"]["families"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    prepare()
