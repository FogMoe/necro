"""只重写已暴露完整记录的措辞；不训练，不打开封存测试。"""
import copy
import json
from pathlib import Path

from necro.diagnostics.numeric_regression import rule_metadata
from necro.evaluation import read_examples, run_evaluation
from necro.training.cloud import settings
from necro.training.data.condition_refinement import instructions

root = Path('../ops/wording-diagnostic')
root.mkdir()
source = Path('data/phase6/regression-v1/conditions.jsonl')
rows = []
for old in read_examples(source):
    meta = rule_metadata(old)
    if meta['case'] != 'complete':
        continue
    row = copy.deepcopy(old)
    row['request']['questions']['decision']['instructions'] = instructions(
        meta['fields'], meta['operator'], meta['language'], meta['negated'], 2, 2)
    new_meta = rule_metadata(row)
    for key in ('fields', 'operator', 'language', 'negated', 'case'):
        assert new_meta[key] == meta[key], key
    assert row['request']['state'] == old['request']['state']
    assert row['expected'] == old['expected']
    rows.append(row)
dataset = root / 'same-states-training-wording.jsonl'
dataset.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
(root / 'plan.json').write_text(json.dumps({
    'scope': 'Post-hoc diagnosis on exposed regression; not independent acceptance',
    'rows': len(rows), 'change': 'Only replace instructions with existing training wording style 2',
    'unchanged': ['weights','calibration','state','operator','question polarity','language','labels'],
    'training': False, 'sealed_test_opened': False,
}, indent=2), encoding='utf-8')
fit = json.loads(Path('results/cloud/primary/calibration-brier-fit.json').read_text())
run_evaluation(dataset, root / 'predictions', settings(Path('results/cloud/primary/adapter'), fit['temperatures']))
original = {r['id']: r for r in read_examples(Path('results/cloud/primary/conditions/judgments.jsonl'))}
actual = read_examples(root / 'predictions/judgments.jsonl')
groups = {}
for r in actual:
    for key in ('all', r['language']):
        g = groups.setdefault(key, {'count':0,'before_correct':0,'after_correct':0,'corrected':0,'new_errors':0})
        before = original[r['id']]['correct']; after = r['correct']
        g['count'] += 1; g['before_correct'] += int(before); g['after_correct'] += int(after)
        g['corrected'] += int(not before and after); g['new_errors'] += int(before and not after)
(root/'comparison.json').write_text(json.dumps(groups,indent=2),encoding='utf-8')
print(json.dumps(groups),flush=True)
