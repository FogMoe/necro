import json
import sys
from collections import defaultdict
from pathlib import Path

from necro.evaluation import read_examples, run_evaluation
from necro.training.cloud import release_memory, settings

root = Path('/root/autodl-tmp/necro-unified-2026-09-21/ops')
mode = sys.argv[1]
if mode in ('conv-only', 'fla-only'):
    import inspect
    from transformers.models.qwen3_5 import modeling_qwen3_5 as model_module
    names = ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule') if mode == 'conv-only' else ('causal_conv1d_fn', 'causal_conv1d_update')
    for name in names:
        setattr(model_module, name, inspect.unwrap(getattr(model_module, name)))
if mode == 'fp32':
    import torch
    from necro.backend import TransformersScorer
    torch.set_float32_matmul_precision('highest')
    original_load = TransformersScorer._load
    def load_fp32(self):
        original_load(self)
        if not getattr(self, '_fp32_probe', False):
            self.model.float()
            self._fp32_probe = True
    TransformersScorer._load = load_fp32
dataset = root / 'kernel-parity.jsonl'
if not dataset.exists():
    rows = read_examples(Path('data/phase6/unified-v1/validation.jsonl'))
    groups = defaultdict(list)
    for row in rows:
        groups[(row['family'], row['language'])].append(row)
    selected = {}
    for values in groups.values():
        for row in [*values[:6], max(values, key=lambda r: len(json.dumps(r['request'])))]:
            selected[row['id']] = row
    for row in read_examples(Path('docs/process/evidence/condition-false-rejections-2026-09-21.jsonl')):
        selected[row['id']] = row
    with dataset.open('x', encoding='utf-8') as f:
        for row in selected.values():
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
for name in ('official', 'parent', 'repaired'):
    adapter = None if name == 'official' else Path('reference') / name / 'adapter'
    output = root / ('kernel-' + mode) / name
    if output.exists():
        raise ValueError('Preserve existing kernel probe: ' + str(output))
    result = run_evaluation(dataset, output, settings(adapter))
    print(json.dumps({'mode': mode, 'model': name, 'seconds': result['metadata']['elapsed_seconds'],
                      'examples': result['metadata']['evaluated_examples']}), flush=True)
    release_memory()
