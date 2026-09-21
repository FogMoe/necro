"""Verify the published experiment archive without loading a model."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "EXPERIMENT_BACKUP_MANIFEST.json").read_text(encoding="utf-8"))
    failures = []
    total = 0
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            failures.append(f"Missing: {entry['path']}")
            continue
        size = path.stat().st_size
        if size != entry["bytes"]:
            with path.open("rb") as stream:
                pointer = stream.read(80).startswith(b"version https://git-lfs.github.com/spec/v1")
            reason = "Git LFS object not downloaded" if pointer else "Size mismatch"
            failures.append(f"{reason}: {entry['path']}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != entry["sha256"]:
            failures.append(f"SHA-256 mismatch: {entry['path']}")
        total += size
    if failures:
        for failure in failures:
            print(failure)
        raise SystemExit(1)
    print(f"Verified {len(manifest['files'])} files, {total:,} bytes.")


if __name__ == "__main__":
    main()
