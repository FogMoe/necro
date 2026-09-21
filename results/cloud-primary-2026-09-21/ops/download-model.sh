#!/bin/bash
set -eu
cd /root/autodl-tmp/necro-unified-2026-09-21
source ops/environment.sh
source /etc/network_turbo >/dev/null
cd workspace
uv run --extra training hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --include '*.json' --include '*.safetensors' --include '*.txt' --include '*.jinja' --max-workers 2
uv run --extra training hf cache verify Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17
