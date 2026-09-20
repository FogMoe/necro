"""同一开发集上的本地候选与基线比较。"""

import argparse
import json
from pathlib import Path

from necro.training.analysis.phase2_analysis import compare


def compare_development(candidate, baseline):
    for path in (candidate, baseline):
        metadata = json.loads((path / "summary.json").read_text())["metadata"]
        if metadata["backend"] != "local" or not metadata.get("full_dataset_evaluated"):
            raise ValueError("开发比较要求同一完整数据集上的本地模型结果。")
    result = compare(candidate, baseline)
    result.pop("performance_gates")
    result.pop("performance_pass")
    result["scope"] = "paired local-model comparison on the development dataset"
    result["notes"] = (
        "Selection uses paired source-group uncertainty and per-task retention. "
        "Scalar temperature preserves argmax decisions."
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_development(args.candidate, args.baseline)
    if args.output.exists():
        raise ValueError("比较已记录，不能覆盖。")
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("macro_accuracy", "macro_brier")}, indent=2))
