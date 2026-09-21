import json
from collections import Counter
from pathlib import Path
from necro.evaluation import read_examples
from necro.diagnostics.numeric_regression import rule_metadata
from necro.training.data.phase2_data import OPS

out={}
for version in ('v1','v2'):
 rows=read_examples(Path(f'data/phase4/condition-{version}/train.jsonl'))
 counts=Counter()
 for row in rows:
  if row['source']!='constructed-condition-v1': continue
  m=rule_metadata(row); a,b,gate=m['fields']; state=row['request']['state']
  if m['missing_fields']==[gate] and OPS[m['operator']][0](state[a],state[b]):
   counts[f"style-{row['condition_style']}/{m['operator']}/{m['language']}/{m['negated']}"]+=1
 out[version]={'gate_missing_comparison_true':dict(sorted(counts.items()))}
original={r['id']:r for r in read_examples(Path('data/phase4/legacy-conditions-regression-v1/regression.jsonl'))}
failures=[]
for row in read_examples(Path('results/phase4/condition-v2-seed2026/legacy-conditions/judgments.jsonl')):
 ex=original[row['id']]; m=rule_metadata(ex)
 if not row['correct'] and m['case']=='missing':
  failures.append({'id':row['id'],'state':ex['request']['state'],'expected':row['expected'],'predicted':row['predicted'],'metadata':m})
out['remaining_missing_failures']=failures
Path('results/phase4/missing-combination-audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({'v2_implicit_guard_hard_missing':{k:v for k,v in out['v2']['gate_missing_comparison_true'].items() if k.startswith('style-1')},'remaining_failures':failures},indent=2),flush=True)
