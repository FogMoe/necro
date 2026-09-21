#!/bin/bash
set -eu
cd /root/autodl-tmp/necro-unified-2026-09-21
source ops/environment-fast.sh
stage="$1"
shift
test ! -e "ops/$stage.started"
date -u +%FT%TZ > "ops/$stage.started"
cd workspace-fast
trap 'status=$?; printf "%s\n" "$status" > "../ops/$stage.exit"' EXIT
uv run --no-sync python -m necro.training.cloud "$@"
