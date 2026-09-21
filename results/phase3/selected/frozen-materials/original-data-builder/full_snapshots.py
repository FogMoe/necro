"""固定 Hub 列式转换修订，读取完整 split 后再抽样，避免顺序区块偏差。"""

import argparse
import json
from pathlib import Path

from necro.experiment_guard import digest

ROOT = Path("data/phase3/snapshots")
SPECS = {
    "massive": ("mteb/amazon_massive_intent", ("en", "zh-CN")),
    "xnli": ("facebook/xnli", ("en", "zh")),
    "paws": ("google-research-datasets/paws-x", ("en", "zh")),
    "boolq": ("google/boolq", ("default",)),
    "squad": ("rajpurkar/squad", ("plain_text",)),
}


def acquire(name):
    from huggingface_hub import HfApi, hf_hub_download

    dataset, configs = SPECS[name]
    root = ROOT / name
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for item in manifest["files"]:
            if digest(root / item["path"]) != item["sha256"]:
                raise ValueError("完整快照校验失败。")
        return manifest
    api = HfApi()
    # 保存精确转换分支 SHA；main 的 SHA 不等于 Viewer 数据修订。
    pin = root / "revision.json"
    if not pin.exists():
        revision = api.dataset_info(dataset, revision="refs/convert/parquet").sha
        pin.write_text(json.dumps({"dataset": dataset, "revision": revision}), encoding="utf-8")
    revision = json.loads(pin.read_text(encoding="utf-8"))["revision"]
    files = []
    for filename in sorted(api.list_repo_files(dataset, repo_type="dataset", revision=revision)):
        parts = filename.split("/")
        if len(parts) != 3 or parts[0] not in configs or not filename.endswith(".parquet"):
            continue
        local = Path(
            hf_hub_download(
                dataset, filename, repo_type="dataset", revision=revision, local_dir=root
            )
        )
        import pyarrow.parquet as pq

        count = pq.ParquetFile(local).metadata.num_rows
        files.append(
            {
                "path": filename,
                "config": parts[0],
                "split": parts[1],
                "rows": count,
                "bytes": local.stat().st_size,
                "sha256": digest(local),
            }
        )
        print(json.dumps({"dataset": name, "file": filename, "rows": count}), flush=True)
    if {item["config"] for item in files} != set(configs):
        raise ValueError("转换快照缺少所需语言。")
    manifest = {
        "dataset": dataset,
        "revision": revision,
        "files": files,
        "sampling": "Complete split; deterministic individual/source-group sampling later",
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def rows(name, config, split):
    import pyarrow.parquet as pq

    root = ROOT / name
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    files = [
        item for item in manifest["files"] if item["config"] == config and item["split"] == split
    ]
    if not files:
        raise ValueError(f"缺少完整快照：{name}/{config}/{split}")
    offset = 0
    for item in files:
        path = root / item["path"]
        if digest(path) != item["sha256"]:
            raise ValueError(f"原始数据已变更：{path}")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
            for row in batch.to_pylist():
                yield offset, row
                offset += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=list(SPECS))
    for name in parser.parse_args().names:
        acquire(name)
