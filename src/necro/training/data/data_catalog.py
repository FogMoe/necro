"""缓存公开 Dataset Viewer 切片；训练池使用分散区块抽样。"""

import hashlib
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

CACHE = Path("data/phase2/raw-cache")
REQUEST_LOCK = threading.Lock()
NEXT_REQUEST = 0.0


def block(client, dataset, config, split, offset, length):
    global NEXT_REQUEST
    params = dict(dataset=dataset, config=config, split=split, offset=offset, length=length)
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    for attempt in range(4):
        try:
            with REQUEST_LOCK:
                time.sleep(max(0.0, NEXT_REQUEST - time.monotonic()))
                NEXT_REQUEST = time.monotonic() + 1.0
            response = client.get("https://datasets-server.huggingface.co/rows", params=params)
            if response.status_code == 429:
                try:
                    delay = max(60.0, float(response.headers.get("Retry-After", "60")))
                except ValueError:
                    delay = 60.0
                with REQUEST_LOCK:
                    NEXT_REQUEST = max(NEXT_REQUEST, time.monotonic() + delay)
            response.raise_for_status()
            payload = response.json()
            if any(row.get("truncated_cells") for row in payload["rows"]):
                raise ValueError("Dataset Viewer 返回截断内容。")
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return payload
        except (httpx.TransportError, httpx.HTTPStatusError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("请求未完成。")


def snapshot(name, dataset, config, split, count, seed, whole=False):
    path = CACHE.parent / f"{name}.json"
    if path.exists():
        return {"cached": name}
    with httpx.Client(timeout=45, follow_redirects=True) as client:
        head = block(client, dataset, config, split, 0, 1)
        total = head["num_rows_total"]
        if whole:
            offsets = list(range(0, total, 100))
            size = 100
        else:
            size = 50
            offsets = random.Random(seed).sample(
                list(range(0, total - size + 1, size)), (count + size - 1) // size
            )
        rows = []
        for offset in offsets:
            rows.extend(
                block(client, dataset, config, split, offset, min(size, total - offset))["rows"]
            )
        info = client.get(f"https://huggingface.co/api/datasets/{dataset}")
        info.raise_for_status()
        metadata = info.json()
        result = {
            "dataset": dataset,
            "config": config,
            "split": split,
            "observed_repository_revision": metadata.get("sha"),
            "license": metadata.get("cardData", {}).get("license"),
            "sampling": "whole split" if whole else "seeded dispersed 50-row blocks",
            "seed": seed,
            "offsets": offsets,
            "total_rows": total,
            "features": head["features"],
            "rows": rows,
        }
        path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return {
            "snapshot": name,
            "rows": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }


def download():
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = []
    for language in ("en", "zh"):
        for split, count in (("train", 800), ("validation", 300), ("test", 300)):
            jobs.append(
                (
                    f"paws-{language}-{split}",
                    "google-research-datasets/paws-x",
                    language,
                    split,
                    count,
                    611,
                )
            )
        jobs.append(
            (f"xquad-{language}", "google/xquad", f"xquad.{language}", "validation", 0, 611, True)
        )
    jobs += [
        ("boolq-train", "google/boolq", "default", "train", 600, 712),
        ("boolq-validation", "google/boolq", "default", "validation", 800, 813),
        ("squad-train", "rajpurkar/squad", "plain_text", "train", 1200, 914),
        ("scifact-corpus", "BeIR/scifact", "corpus", "corpus", 0, 615, True),
        ("scifact-queries", "BeIR/scifact", "queries", "queries", 0, 615, True),
        ("scifact-train", "BeIR/scifact-qrels", "default", "train", 0, 615, True),
        ("scifact-test", "BeIR/scifact-qrels", "default", "test", 0, 615, True),
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(lambda args: snapshot(*args), jobs):
            print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    download()
