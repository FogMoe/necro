#!/bin/bash
set -eu
cd /root/autodl-tmp/necro-unified-2026-09-21
source ops/environment.sh
stage="$1"
shift
test ! -e "ops/$stage.started"
date -u +%FT%TZ > "ops/$stage.started"
trap 'status=$?; printf "%s\n" "$status" > "../ops/$stage.exit"' EXIT
cd workspace
uv run --extra training python -m necro.training.cloud "$@"
