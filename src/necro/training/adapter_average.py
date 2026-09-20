"""对共同起点、同配方 LoRA 的 A、B 因子分别计算参数均值。"""

import argparse
import json
import shutil
from pathlib import Path

from necro.experiment_guard import digest


def average(left: Path, right: Path, output: Path, model_id):
    import torch
    from safetensors.torch import load_file, save_file

    if output.exists():
        raise ValueError("均值 adapter 输出已存在，不可覆盖。")
    manifests = [
        json.loads((path / "necro_adapter.json").read_text(encoding="utf-8"))
        for path in (left, right)
    ]
    required = (
        "checkpoint",
        "revision",
        "prompt_sha256",
        "rank",
        "initial_adapter_sha256",
        "train_sha256",
        "learning_rate",
        "batch_size",
        "gradient_accumulation",
        "objective",
    )
    if any(key not in manifest for key in required for manifest in manifests):
        raise ValueError("缺少共同起点或训练配方的必要记录。")
    if any(manifests[0].get(key) != manifests[1].get(key) for key in required):
        raise ValueError("只允许共同起点、相同数据和主要配方的 LoRA 做参数均值。")
    if not manifests[0].get("initial_adapter_sha256"):
        raise ValueError("必须有共同父权重的可核查哈希。")
    configs = [
        json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
        for path in (left, right)
    ]
    # PEFT 将这些模块名视为集合，保存时的列表顺序可能随进程变化。
    for config in configs:
        if isinstance(config.get("target_modules"), list):
            config["target_modules"] = sorted(config["target_modules"])
    if configs[0] != configs[1]:
        raise ValueError("PEFT 配置不一致。")
    tensors = [
        load_file(str(path / "adapter_model.safetensors"), device="cpu") for path in (left, right)
    ]
    if tensors[0].keys() != tensors[1].keys():
        raise ValueError("Adapter 参数名不一致。")
    merged = {}
    for key, a in tensors[0].items():
        b = tensors[1][key]
        if not key.endswith(("lora_A.weight", "lora_B.weight")):
            raise ValueError("发现 LoRA 因子以外的权重，不自动进行平均。")
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError("Adapter 参数形状或 dtype 不一致。")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("Adapter 包含非有限权重。")
        merged[key] = ((a.float() + b.float()) * 0.5).to(a.dtype).contiguous()
    composition = {
        "method": "arithmetic mean of aligned LoRA A and B factor parameters",
        "coefficients": [0.5, 0.5],
        "parents": [
            {"path": str(path), "weights_sha256": digest(path / "adapter_model.safetensors")}
            for path in (left, right)
        ],
        "common_parent_sha256": manifests[0]["initial_adapter_sha256"],
        "construction_source_sha256": digest(Path(__file__)),
        "notes": "Arithmetic averaging of existing factors. Rank and merged model size "
        "are unchanged; the construction performs no gradient updates.",
    }
    shutil.copytree(left, output)
    source_dir = output.parent / "source"
    source_dir.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), source_dir / "adapter_average.py")
    save_file(merged, str(output / "adapter_model.safetensors"), metadata={"format": "pt"})
    contract = {
        **manifests[0],
        "model_id": model_id,
        "initial_adapter": None,
        "initial_adapter_sha256": None,
        "seed": None,
        "examples": 0,
        "optimizer_steps": 0,
        "epochs": 0,
        "objective": "parameter-average",
        "objective_description": "Arithmetic mean of aligned LoRA factor parameters; "
        "no gradient updates.",
        "composition": composition,
    }
    (output / "necro_adapter.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    result = {**composition, "output_weights_sha256": digest(output / "adapter_model.safetensors")}
    (output.parent / "composition.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()
    print(json.dumps(average(args.left, args.right, args.output, args.model_id), indent=2))
