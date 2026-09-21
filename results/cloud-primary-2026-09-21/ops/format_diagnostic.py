"""已暴露等值记录的表示诊断；不修改正式评测数据或结果。"""
import copy
import json
from pathlib import Path

from necro.diagnostics.numeric_regression import rule_metadata
from necro.evaluation import read_examples, run_evaluation
from necro.training.cloud import settings

root = Path('../ops/format-diagnostic')
root.mkdir()
rows = []
for old in read_examples(Path('data/phase6/regression-v1/conditions.jsonl')):
    meta = rule_metadata(old)
    if meta['case'] != 'complete':
        continue
    a, b = meta['fields'][:2]
    original = old['request']['state']
    if original[a] != original[b] or type(original[a]) == type(original[b]):
        continue
    row = copy.deepcopy(old)
    for key in (a, b):
        value = row['request']['state'][key]
        if type(value) is float and value.is_integer():
            row['request']['state'][key] = int(value)
    assert row['request']['state'] == original
    assert row['request']['questions'] == old['request']['questions']
    rows.append(row)
dataset = root / 'same-values-integer-format.jsonl'
dataset.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
(root / 'plan.json').write_text(json.dumps({
    'scope': 'Post-hoc diagnosis on exposed equal-value regression; not acceptance',
    'rows': len(rows), 'change': 'Represent integral float operands as equal-valued integers',
    'unchanged': ['weights','calibration','mathematical values','instructions','labels'],
    'training': False, 'sealed_test_opened': False,
}, indent=2), encoding='utf-8')
fit = json.loads(Path('results/cloud/primary/calibration-brier-fit.json').read_text())
run_evaluation(dataset, root / 'predictions', settings(Path('results/cloud/primary/adapter'), fit['temperatures']))
original = {r['id']: r for r in read_examples(Path('results/cloud/primary/conditions/judgments.jsonl'))}
actual = read_examples(root / 'predictions/judgments.jsonl')
result = {'count': len(actual), 'before_correct': sum(original[r['id']]['correct'] for r in actual),
          'after_correct': sum(r['correct'] for r in actual),
          'corrected': sum(not original[r['id']]['correct'] and r['correct'] for r in actual),
          'new_errors': sum(original[r['id']]['correct'] and not r['correct'] for r in actual)}
(root/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result),flush=True)
