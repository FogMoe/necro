#!/bin/bash
set -eu
cd /root/autodl-tmp/necro-unified-2026-09-21
while kill -0 "$(cat ops/fast-profile.pid)" 2>/dev/null; do sleep 2; done
source ops/environment-fast.sh
cd workspace-fast
uv run --no-sync python ../ops/kernel_probe.py conv-only
uv run --no-sync python ../ops/kernel_probe.py fla-only
cd ../workspace
uv run --no-sync python ../ops/kernel_probe.py fp32
