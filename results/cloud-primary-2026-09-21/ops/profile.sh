#!/bin/bash
set -eu
cd /root/autodl-tmp/necro-unified-2026-09-21
source ops/environment.sh
source /etc/network_turbo >/dev/null
cd workspace
uv run --extra training python -m necro.training.cloud profile
