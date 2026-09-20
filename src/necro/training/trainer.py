# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ScarletKc-Necro contributors

"""一次小 LoRA 实验：只监督答案 token，复用服务提示，不生成训练推理文本。"""

import argparse
import hashlib
import json
import math
import random
import shutil
import time
from dataclasses import replace
from pathlib import Path

from necro.backend import TransformersScorer, numeric_labels
from necro.config import Settings
from necro.engine import PROMPT_VERSION, prepare_task, prompt_fingerprint
from necro.evaluation import read_examples
from necro.schema import EvaluationRequest, Noul
from necro.training.data.training_data import audit_disjoint


def encode_example(tokenizer, alphabet, example, max_length=2048):
    request = EvaluationRequest.model_validate(example["request"])
    if len(request.questions) != 1:
        raise ValueError("训练行应包含一题。")
    key, question = next(iter(request.questions.items()))
    task = prepare_task(request.state, question)
    labels = (
        ["Yes", "No"]
        if isinstance(question, Noul)
        else numeric_labels(len(task.keys))
        if len(task.keys) > len(alphabet)
        else alphabet[: len(task.keys)]
    )
    expected = example["expected"][key]
    if isinstance(question, Noul):
        if type(expected) is not bool:
            raise ValueError("Noul 训练标签必须是布尔值。")
        index = 0 if expected else 1
    else:
        index = task.keys.index(str(expected))
    prompt = tokenizer.apply_chat_template(
        task.messages(labels), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    target = tokenizer.encode(labels[index], add_special_tokens=False)
    full = tokenizer.encode(prompt + labels[index], add_special_tokens=False)
    if full != prefix + target:
        raise ValueError("训练答案的 token 边界与推理不一致。")
    inputs = prefix + target[:-1]
    if len(inputs) > max_length:
        raise ValueError(f"训练行 {example['id']} 超过 {max_length} tokens，不进行截断。")
    candidates = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
    return {
        "id": example["id"],
        "input_ids": inputs,
        "target_ids": target,
        "candidate_ids": [tokens[0] for tokens in candidates]
        if all(len(tokens) == 1 for tokens in candidates)
        else None,
    }


def collate(rows, pad_id, device):
    import torch

    length = max(len(row["input_ids"]) for row in rows)
    answers = max(len(row["target_ids"]) for row in rows)
    inputs, masks, targets = [], [], []
    for row in rows:
        pad = length - len(row["input_ids"])
        inputs.append([pad_id] * pad + row["input_ids"])
        masks.append([0] * pad + [1] * len(row["input_ids"]))
        targets.append([-100] * (answers - len(row["target_ids"])) + row["target_ids"])
    return {
        "input_ids": torch.tensor(inputs, device=device),
        "attention_mask": torch.tensor(masks, device=device),
    }, torch.tensor(targets, device=device)


def answer_loss(model, inputs, targets, candidate_ids=None):
    import torch
    import torch.nn.functional as F

    output = model(**inputs, logits_to_keep=targets.shape[1], use_cache=False)
    # 每题完整答案的 token NLL 之和；prompt 位置不进入 loss。
    logits = output.logits.float()
    full_losses = F.cross_entropy(
        logits.transpose(1, 2), targets, ignore_index=-100, reduction="none"
    ).sum(dim=1)
    if candidate_ids is None:
        return full_losses.mean()
    if len(candidate_ids) != len(targets):
        raise ValueError("候选监督与 batch 数量不一致。")
    losses = []
    for i, candidates in enumerate(candidate_ids):
        if candidates is None:
            losses.append(full_losses[i])
            continue
        if int((targets[i] != -100).sum()) != 1:
            raise ValueError("候选内损失仅用于单 token 答案。")
        ids = torch.tensor(candidates, device=logits.device)
        if ids.unique().numel() != ids.numel():
            raise ValueError("候选 token 不可重复。")
        gold = (ids == targets[i, -1]).nonzero().flatten()
        if gold.numel() != 1:
            raise ValueError("正确标签必须恰好属于一个候选。")
        losses.append(F.cross_entropy(logits[i, -1, ids].unsqueeze(0), gold))
    return torch.stack(losses).mean()


def make_batches(records, batch_size, seed):
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    batches = []
    for start in range(0, len(shuffled), 64):
        bucket = sorted(shuffled[start : start + 64], key=lambda row: len(row["input_ids"]))
        batches.extend(bucket[i : i + batch_size] for i in range(0, len(bucket), batch_size))
    rng.shuffle(batches)
    return batches


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def snapshot_sources(output: Path):
    package = Path(__file__).resolve().parents[1]
    source_dir = output / "source"
    source_dir.mkdir()
    hashes = {}
    for filename in (
        "training/trainer.py",
        "training/__init__.py",
        "training/__main__.py",
        "engine.py",
        "backend.py",
        "schema.py",
        "config.py",
    ):
        path = package / filename
        target = source_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def train(
    data_dir: Path,
    output: Path,
    batch_size=2,
    accumulation=4,
    learning_rate=1e-4,
    rank=16,
    seed=17,
    initial_adapter: Path | None = None,
    model_id="necro-qwen3.5-0.8b-lora-pilot-v1",
    objective="answer-ce",
):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model

    if output.exists():
        raise ValueError("输出目录已存在；请为新实验选择新目录。")
    if batch_size < 1 or accumulation < 1:
        raise ValueError("批量与梯度累积必须大于 0。")
    if objective not in {"answer-ce", "candidate-ce"}:
        raise ValueError("未知训练目标。")
    if (data_dir / "experiment.json").is_file():
        from necro.experiment_guard import verify

        verify(data_dir, "train")
        verify(data_dir, "development")
    training = read_examples(data_dir / "train.jsonl")
    validation = read_examples(data_dir / "validation.jsonl")
    audit_disjoint(training, validation)
    torch.manual_seed(seed)
    settings = replace(Settings.from_env(), adapter=None)
    scorer = TransformersScorer(settings)
    scorer.load()
    if scorer.device.type != "cuda":
        raise ValueError("本次训练要求 CUDA。")
    encoded = [encode_example(scorer.tokenizer, scorer.labels, row) for row in training]
    batches = make_batches(encoded, batch_size, seed)
    base = scorer.model
    revision = base.config._commit_hash
    probe_inputs, _ = collate([encoded[0]], scorer.tokenizer.pad_token_id, scorer.device)
    with torch.inference_mode():
        baseline_probe = (
            base(**probe_inputs, logits_to_keep=1, use_cache=False).logits.float().cpu()
        )
    targets = [
        name
        for name, module in base.named_modules()
        if "language_model.layers." in name and isinstance(module, torch.nn.Linear)
    ]
    if not targets:
        raise RuntimeError("没有找到语言骨干中的 LoRA 目标层。")
    base.requires_grad_(False)
    if initial_adapter:
        contract = TransformersScorer(
            replace(settings, adapter=str(initial_adapter))
        ).adapter_contract
        if contract["revision"] != revision or contract["rank"] != rank:
            raise ValueError("续训 adapter 的基模 revision 或 rank 不一致。")
        model = PeftModel.from_pretrained(base, initial_adapter, is_trainable=True)
    else:
        model = get_peft_model(
            base,
            LoraConfig(
                r=rank,
                lora_alpha=rank * 2,
                lora_dropout=0.05,
                bias="none",
                target_modules=targets,
                task_type="CAUSAL_LM",
                revision=revision,
            ),
        )
    trainable = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("检测到 LoRA 之外的可训练权重。")
    parameters = [parameter for _, parameter in trainable]
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    output.mkdir(parents=True)
    total_steps = math.ceil(len(batches) / accumulation)
    config = {
        "checkpoint": settings.checkpoint,
        "revision": revision,
        "model_id": model_id,
        "initial_adapter": str(initial_adapter) if initial_adapter else None,
        "initial_adapter_sha256": hashlib.sha256(
            (initial_adapter / "adapter_model.safetensors").read_bytes()
        ).hexdigest()
        if initial_adapter
        else None,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_fingerprint(),
        "seed": seed,
        "rank": rank,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation": accumulation,
        "epochs": 1,
        "optimizer_steps": total_steps,
        "examples": len(encoded),
        "trainable_parameters": sum(p.numel() for p in parameters),
        "target_modules": targets,
        "max_input_tokens": max(len(row["input_ids"]) for row in encoded),
        "training_tokens": sum(len(row["input_ids"]) for row in encoded),
        "objective": objective,
        "objective_description": "single-token candidate CE; multi-token full-vocabulary CE"
        if objective == "candidate-ce"
        else "full-vocabulary answer-token CE; prompts masked",
        "train_sha256": hashlib.sha256((data_dir / "train.jsonl").read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(
            (data_dir / "validation.jsonl").read_bytes()
        ).hexdigest(),
    }
    config["source_sha256"] = snapshot_sources(output)
    (output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    emit({"event": "setup", **{k: v for k, v in config.items() if k != "target_modules"}})
    profile_times, profile_tokens = [], 0
    torch.cuda.reset_peak_memory_stats()
    for i, rows in enumerate(batches[:6]):
        inputs, labels = collate(rows, scorer.tokenizer.pad_token_id, scorer.device)
        start = time.perf_counter()
        loss = answer_loss(
            model,
            inputs,
            labels,
            [row["candidate_ids"] for row in rows] if objective == "candidate-ce" else None,
        )
        loss.backward()
        torch.cuda.synchronize()
        if not torch.isfinite(loss):
            raise RuntimeError("训练损失非有限值。")
        if i >= 2:
            profile_times.append(time.perf_counter() - start)
            profile_tokens += inputs["input_ids"].numel()
        model.zero_grad(set_to_none=True)
    padded_tokens = sum(len(rows) * max(len(row["input_ids"]) for row in rows) for rows in batches)
    profile = {
        "event": "profile",
        "seconds_per_microbatch": sum(profile_times) / len(profile_times),
        "estimated_training_seconds": padded_tokens * sum(profile_times) / profile_tokens,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    emit(profile)
    (output / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.01, fused=True)
    warmup = max(1, math.ceil(total_steps * 0.1))
    started = time.perf_counter()
    history = []
    for step, start in enumerate(range(0, len(batches), accumulation), 1):
        group = batches[start : start + accumulation]
        losses = []
        group_size = sum(len(rows) for rows in group)
        for rows in group:
            inputs, labels = collate(rows, scorer.tokenizer.pad_token_id, scorer.device)
            loss = answer_loss(
                model,
                inputs,
                labels,
                [row["candidate_ids"] for row in rows] if objective == "candidate-ce" else None,
            )
            (loss * len(rows) / group_size).backward()
            losses.append(float(loss.detach()) * len(rows))
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        scale = (
            step / warmup
            if step <= warmup
            else max(0.0, (total_steps - step + 1) / (total_steps - warmup))
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate * scale
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        record = {
            "step": step,
            "loss": sum(losses) / group_size,
            "grad_norm": float(norm),
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(record)
        if step == 1 or step % 10 == 0 or step == total_steps:
            emit(
                {
                    "event": "training",
                    **record,
                    "eta_seconds": record["elapsed_seconds"] / step * (total_steps - step),
                }
            )
    elapsed = time.perf_counter() - started
    model.eval()
    model.gradient_checkpointing_disable()
    with model.disable_adapter(), torch.inference_mode():
        disabled_probe = (
            model(**probe_inputs, logits_to_keep=1, use_cache=False).logits.float().cpu()
        )
    max_base_drift = float((disabled_probe - baseline_probe).abs().max())
    torch.testing.assert_close(disabled_probe, baseline_probe, atol=1e-5, rtol=1e-5)
    adapter = output / "adapter"
    model.save_pretrained(adapter)
    scorer.tokenizer.save_pretrained(adapter)
    (adapter / "necro_adapter.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary = {
        "training_seconds": elapsed,
        "optimizer_steps": total_steps,
        "base_logits_max_difference_with_adapter_disabled": max_base_drift,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "adapter": str(adapter.resolve()),
        "history": history,
    }
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    emit({"event": "complete", **{k: v for k, v in summary.items() if k != "history"}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/lora-pilot"))
    parser.add_argument("--output", type=Path, default=Path("results/lora-pilot/run1"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--initial-adapter", type=Path)
    parser.add_argument("--model-id", default="necro-qwen3.5-0.8b-lora-pilot-v1")
    parser.add_argument("--objective", choices=["answer-ce", "candidate-ce"], default="answer-ce")
    args = parser.parse_args()
    train(
        args.data,
        args.output,
        args.batch_size,
        args.accumulation,
        learning_rate=args.learning_rate,
        seed=args.seed,
        initial_adapter=args.initial_adapter,
        model_id=args.model_id,
        objective=args.objective,
    )


if __name__ == "__main__":
    main()
