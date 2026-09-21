"""只评测登记的开发/校准划分；使用同一温度程序比较候选。"""

import argparse
import gc
import json
from pathlib import Path

from necro.calibration import fit_primitive_temperatures, transform_report
from necro.config import Settings
from necro.evaluation import run_evaluation
from necro.experiment_guard import digest, verify


def assess(data, adapter, output, objective="nll"):
    import torch

    settings = Settings(
        adapter=str(adapter),
        device="cuda",
        temperature=1,
        choice_temperature=1,
        noul_temperature=1,
        score_temperature=1,
    )
    for role in ("development", "calibration"):
        path = verify(data, role)
        target = output / role
        summary = target / "summary.json"
        if summary.exists():
            metadata = json.loads(summary.read_text())["metadata"]
            if (
                metadata["dataset_sha256"] != digest(path)
                or metadata["adapter_weights_sha256"]
                != digest(adapter / "adapter_model.safetensors")
                or not metadata["full_dataset_evaluated"]
                or any(metadata.get(f"{p}_temperature") != 1 for p in ("choice", "noul", "score"))
            ):
                raise ValueError("已有结果与候选或原始温度不匹配。")
        else:
            report = run_evaluation(path, target, settings)
            print(json.dumps({"role": role, "families": report["families"]}), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    fit_path = output / (
        "calibration-fit.json" if objective == "nll" else "calibration-brier-fit.json"
    )
    if not fit_path.exists():
        fit = fit_primitive_temperatures(
            verify(data, "calibration"),
            output / "calibration/predictions.jsonl",
            upper=8.0,
            objective=objective,
        )
        fit_path.write_text(json.dumps(fit, indent=2), encoding="utf-8")
    fit = json.loads(fit_path.read_text())
    target = output / ("development-calibrated" if objective == "nll" else "development-brier")
    if not target.exists():
        transform_report(
            verify(data, "development"),
            output / "development/predictions.jsonl",
            target,
            fit["temperatures"],
        )
    report = json.loads((target / "summary.json").read_text())
    print(
        json.dumps(
            {
                "candidate": str(adapter),
                "macro_accuracy": report["macro_family_accuracy"],
                "macro_brier": sum(v["brier"] for v in report["families"].values())
                / len(report["families"]),
                "temperatures": fit["temperatures"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--objective", choices=["nll", "brier"], default="nll")
    args = parser.parse_args()
    assess(args.data, args.adapter, args.output, args.objective)
