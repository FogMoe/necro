#!/bin/bash
set -eu
cd /root/autodl-tmp/necro-unified-2026-09-21
source ops/environment.sh
cd workspace
uv run --no-sync python ../ops/kernel_probe.py reference
cd ../workspace-fast
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
uv run --no-sync python ../ops/kernel_probe.py fast
