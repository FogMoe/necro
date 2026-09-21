"""可执行的数据划分与封存测试约束。"""

import hashlib
import json
import re
import unicodedata
from pathlib import Path

from necro.evaluation import read_examples
from necro.schema import Choice, EvaluationRequest, Noul, Score


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_text(value):
    text = (
        value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    )
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def canonical_request(row):
    # 只删除项目生成器明确声明的无关记录 ID，不删除用户数据中可能有语义的 ID。
    request = json.loads(json.dumps(row["request"]))
    if row.get("source", "").startswith("constructed-") and isinstance(request["state"], dict):
        request["state"].pop("record_id", None)
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", json.dumps(request, ensure_ascii=False, sort_keys=True)),
    ).strip()


def audit_partitions(partitions):
    seen_groups, seen_content, seen_requests = {}, {}, {}
    report = {}
    for role, rows in partitions.items():
        # exposure 是祖先训练记录的审计并集，可重复；实际数据切片的 ID 必须唯一。
        if role != "exposure" and len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"{role} 存在重复样本 ID，无法可靠对应预测。")
        local_groups, local_content = set(), set()
        for row in rows:
            request = EvaluationRequest.model_validate(row["request"])
            if set(row["expected"]) != set(request.questions):
                raise ValueError(f"标签与问题不匹配：{row['id']}")
            for key, question in request.questions.items():
                gold = row["expected"][key]
                if isinstance(question, Noul) and type(gold) is not bool:
                    raise ValueError(f"Noul 标签必须为布尔值：{row['id']}")
                if isinstance(question, Choice) and gold not in question.criteria:
                    raise ValueError(f"正确 Choice 不在候选内：{row['id']}")
                if isinstance(question, Score) and (
                    type(gold) is not int or not 0 <= gold < len(question.criteria)
                ):
                    raise ValueError(f"Score 标签超出等级：{row['id']}")
            group = row["group_id"]
            content = row.get("content_group", canonical_request(row))
            signature = canonical_request(row)
            if signature in seen_requests and seen_requests[signature] != role:
                raise ValueError(f"规范化请求跨划分：{row['id']}")
            if group in seen_groups and seen_groups[group] != role:
                raise ValueError(f"来源组跨划分：{group}，{seen_groups[group]} / {role}")
            if content in seen_content and seen_content[content] != role:
                raise ValueError(f"内容跨划分：{row['id']}，{seen_content[content]} / {role}")
            seen_groups[group] = role
            seen_content[content] = role
            seen_requests[signature] = role
            local_groups.add(group)
            local_content.add(content)
        report[role] = {
            "rows": len(rows),
            "source_groups": len(local_groups),
            "content_groups": len(local_content),
            "unique_requests": len({canonical_request(row) for row in rows}),
        }
    return report


def register(root: Path, files: dict[str, Path], protocol: dict):
    manifest_path = root / "experiment.json"
    if manifest_path.exists():
        raise ValueError("实验已登记；变更数据必须建立新实验目录。")
    audit = audit_partitions({role: read_examples(path) for role, path in files.items()})
    manifest = {
        "protocol": protocol,
        "partitions": {
            role: {
                "path": str(path.resolve().relative_to(root.resolve())),
                "sha256": digest(path),
                **audit[role],
            }
            for role, path in files.items()
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def verify_temperatures(selection, weights_sha256, actual):
    """测试允许原始概率，或该份冻结权重自己的校准；禁止借用另一模型的温度。"""
    if not selection.get("temperatures"):
        return  # 兼容尚未登记温度的历史实验；新实验始终登记。
    approved = (
        selection["temperatures"]
        if weights_sha256 == selection["weights_sha256"]
        else selection.get("reference_temperatures", {}).get(weights_sha256)
    )
    if actual != dict.fromkeys(("choice", "noul", "score"), 1.0) and actual != approved:
        raise ValueError("测试只允许原始温度 1 或该权重冻结的校准温度。")


def verify(root: Path, role: str, adapter: Path | None = None, remote_reference=False):
    manifest = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    for partition in manifest["partitions"].values():
        if digest(root / partition["path"]) != partition["sha256"]:
            raise ValueError("登记后数据已发生变更，拒绝执行。")
    if role == "test":
        selected_path = root / "selection.json"
        if not selected_path.is_file():
            raise ValueError("尚未冻结模型选择，不能打开封存测试。")
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        if selected.get("data_manifest_sha256") and (
            selected["data_manifest_sha256"] != digest(root / "experiment.json")
        ):
            raise ValueError("冻结后实验登记发生变更。")
        for artifact, expected in selected.get("artifact_hashes", {}).items():
            if digest(Path(artifact)) != expected:
                raise ValueError(f"冻结后的校准或协议材料已变更：{artifact}")
        if selected.get("prompt_sha256"):
            from necro.engine import prompt_fingerprint

            if selected["prompt_sha256"] != prompt_fingerprint():
                raise ValueError("冻结后的提示代码已变更。")
        allowed = {selected["weights_sha256"], *selected.get("reference_weights_sha256", [])}
        if not remote_reference and (
            adapter is None or digest(adapter / "adapter_model.safetensors") not in allowed
        ):
            raise ValueError("封存测试只能评测已冻结的选中权重。")
    return (root / manifest["partitions"][role]["path"]).resolve()
