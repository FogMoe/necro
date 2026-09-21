#!/bin/bash
set -eu
cd /root/autodl-tmp/necro-unified-2026-09-21
source ops/environment.sh
source /etc/network_turbo >/dev/null
cd workspace
uv sync --locked --python 3.12 --extra training
uv run --extra training python -m necro.training.cloud verify
uv run --extra training python -c 'import torch,transformers,peft,sys; print(dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,transformers=transformers.__version__,peft=peft.__version__,gpu=torch.cuda.get_device_name(),bf16=torch.cuda.is_bf16_supported()))'
