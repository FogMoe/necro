#!/bin/bash
set -eu
cd /root/autodl-tmp/necro-unified-2026-09-21
while test ! -f ops/primary.exit; do
  if ! kill -0 "$(cat ops/primary.pid)" 2>/dev/null; then echo 'Primary ended without exit receipt'; exit 1; fi
  sleep 10
done
test "$(cat ops/primary.exit)" = 0
source ops/environment.sh
cd workspace
test -f results/cloud/primary/adapter/adapter_model.safetensors
date -u +%FT%TZ > ../ops/baselines-resume.started
set +e
uv run --extra training python ../ops/resume_baselines.py > ../ops/baselines-resume.log 2>&1
status=$?
printf '%s\n' "$status" > ../ops/baselines-resume.exit
set -e
test "$status" = 0
cd ..
bash ops/run-stage.sh primary-assessment assess primary > ops/primary-assessment.log 2>&1
