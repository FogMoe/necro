"""生成本地 adapter 与合并权重发布目录，不连接 Hub 上传。"""

import argparse
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

from necro.backend import TransformersScorer
from necro.config import Settings


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_model_licenses(directory):
    for source, filename in (
        ("LICENSE", "LICENSE"),
        ("LICENSE-MIT", "LICENSE-MIT"),
        ("THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"),
        ("licenses/QWEN_APACHE_LICENSE", "QWEN_APACHE_LICENSE"),
        ("licenses/XNLI_CC_BY_NC_4.0", "XNLI_CC_BY_NC_4.0"),
    ):
        shutil.copy2(source, directory / filename)


def export(adapter: Path, output: Path, temperature: float = 1.0):
    if output.exists():
        raise ValueError("导出目录已存在，不覆盖已有发布包。")
    settings = replace(Settings.from_env(), adapter=str(adapter), choice_temperature=temperature)
    scorer = TransformersScorer(settings)
    scorer.load()
    output.mkdir(parents=True)
    adapter_dir = output / "adapter"
    shutil.copytree(adapter, adapter_dir)
    merged = output / "merged"
    scorer.model.save_pretrained(merged, safe_serialization=True, max_shard_size="2GB")
    scorer.tokenizer.save_pretrained(merged)
    contract = {
        **scorer.adapter_contract,
        "recommended_choice_temperature": temperature,
        "adapter_weights_sha256": sha256(adapter / "adapter_model.safetensors"),
    }
    for directory, name in ((adapter_dir, "necro_adapter.json"), (merged, "necro_model.json")):
        (directory / name).write_text(json.dumps(contract, indent=2), encoding="utf-8")
        copy_model_licenses(directory)
    runtime = output / "runtime"
    runtime.mkdir()
    shutil.copytree(
        Path("src") / "necro",
        runtime / "src" / "necro",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for filename in (
        "pyproject.toml",
        "uv.lock",
        "README.md",
        ".env.example",
        "LICENSE",
        "LICENSE-MIT",
        "THIRD_PARTY_NOTICES.md",
    ):
        shutil.copy2(filename, runtime / filename)
    shutil.copytree("licenses", runtime / "licenses")
    shutil.copytree("examples", runtime / "examples")
    shutil.copytree("docs", runtime / "docs")
    shutil.copytree("tests", runtime / "tests", ignore=shutil.ignore_patterns("__pycache__"))
    metadata = {
        "model_id": scorer.model_id,
        "choice_temperature": temperature,
        "base_model": scorer.settings.checkpoint,
        "base_revision": scorer.adapter_contract["revision"],
        "adapter_weights_sha256": contract["adapter_weights_sha256"],
        "merged_weights": {
            p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in merged.glob("*.safetensors")
        },
    }
    (output / "export.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    export(args.adapter, args.output, args.temperature)
