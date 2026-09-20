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


def export(
    adapter: Path,
    output: Path,
    temperature: float = 1.0,
    calibration_file: Path | None = None,
    hub_layout=False,
):
    if output.exists():
        raise ValueError("导出目录已存在，不覆盖已有发布包。")
    temperatures = {"choice": temperature, "noul": 1.0, "score": 1.0}
    if calibration_file:
        temperatures.update(
            json.loads(calibration_file.read_text(encoding="utf-8"))["temperatures"]
        )
    if set(temperatures) != {"choice", "noul", "score"}:
        raise ValueError("校准文件包含未知判断类型。")
    settings = replace(
        Settings.from_env(),
        checkpoint=json.loads((adapter / "necro_adapter.json").read_text(encoding="utf-8"))[
            "checkpoint"
        ],
        adapter=str(adapter),
        temperature=1.0,
        **{f"{primitive}_temperature": value for primitive, value in temperatures.items()},
    )
    scorer = TransformersScorer(settings)
    scorer.load()
    output.mkdir(parents=True)
    adapter_dir = output / "adapter"
    shutil.copytree(adapter, adapter_dir)
    merged = output if hub_layout else output / "merged"
    scorer.model.save_pretrained(merged, safe_serialization=True, max_shard_size="2GB")
    scorer.tokenizer.save_pretrained(merged)
    contract = {
        **scorer.adapter_contract,
        "recommended_choice_temperature": temperatures["choice"],
        "recommended_temperatures": temperatures,
        "adapter_weights_sha256": sha256(adapter / "adapter_model.safetensors"),
    }
    for directory, name in ((adapter_dir, "necro_adapter.json"), (merged, "necro_model.json")):
        (directory / name).write_text(json.dumps(contract, indent=2), encoding="utf-8")
        copy_model_licenses(directory)
        if calibration_file:
            shutil.copy2(calibration_file, directory / "calibration.json")
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
    # Paths are relative to the bundled runtime directory.
    env_path = runtime / ".env.example"
    environment = {
        "NECRO_MODEL": ".." if hub_layout else "../merged",
        "NECRO_ADAPTER": "",
        "NECRO_TEMPERATURE": "1.0",
        **{
            f"NECRO_{primitive.upper()}_TEMPERATURE": str(value)
            for primitive, value in temperatures.items()
        },
    }
    lines = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        key = line.split("=", 1)[0]
        lines.append(f"{key}={environment[key]}" if key in environment else line)
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shutil.copytree("licenses", runtime / "licenses")
    shutil.copytree("examples", runtime / "examples")
    shutil.copytree("docs", runtime / "docs")
    shutil.copytree("tests", runtime / "tests", ignore=shutil.ignore_patterns("__pycache__"))
    metadata = {
        "model_id": scorer.model_id,
        "layout": "hub-root" if hub_layout else "split-directories",
        "merged_path": "." if hub_layout else "merged",
        "choice_temperature": temperatures["choice"],
        "temperatures": temperatures,
        "base_model": scorer.settings.checkpoint,
        "base_revision": scorer.adapter_contract["revision"],
        "total_parameters": sum(parameter.numel() for parameter in scorer.model.parameters()),
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
    parser.add_argument("--calibration-file", type=Path)
    parser.add_argument("--hub-layout", action="store_true")
    args = parser.parse_args()
    export(args.adapter, args.output, args.temperature, args.calibration_file, args.hub_layout)
