import argparse
import json
from collections import defaultdict
from pathlib import Path

from necro.config import Settings
from necro.diagnostics.numeric_regression import rule_metadata
from necro.evaluation import read_examples, run_evaluation
from necro.experiment_guard import digest, register
from necro.training.analysis.development_comparison import compare_development
from necro.training.analysis.phase2_analysis import project_report
from necro.training.data.training_data import write_jsonl

parser=argparse.ArgumentParser()
parser.add_argument('--run',type=Path,default=Path('results/phase4/condition-seed2026'))
parser.add_argument('--data',type=Path,default=Path('data/phase4/condition-v1'))
args=parser.parse_args()
root=args.run
data=args.data
fresh=Path('data/phase4/development-conditions-v1/development.jsonl')
old=Path('data/phase3/coverage-v1/validation.jsonl')
pred=root/'development-brier/predictions.jsonl'
for subset,name in ((fresh,'development-conditions'),(old,'development-retention')):
 if not (root/name).exists(): project_report(data/'validation.jsonl',subset,pred,root/name)
comparisons={}
for name,path in (('predecessor','results/phase3/baseline/development-brier'),('instruction-parent','results/phase3/instruction-seed2026/development-brier')):
 comparisons[name]=compare_development(root/'development-retention',Path(path))
 (root/f'{name}-retention-comparison.json').write_text(json.dumps(comparisons[name],indent=2))

def conditions(dataset,results):
 examples={r['id']:r for r in read_examples(dataset)}
 groups=defaultdict(list)
 for row in read_examples(results/'judgments.jsonl'):
  ex=examples[row['id']]
  meta=rule_metadata(ex)
  for group in ('all',meta['case'],meta['language'],f"style-{ex.get('condition_style','legacy')}/{meta['language']}/{meta['case']}"):
   groups[group].append(row['correct'])
 return {key:{'count':len(values),'correct':sum(values),'accuracy':sum(values)/len(values)} for key,values in groups.items()}

fresh_results=conditions(fresh,root/'development-conditions')
baseline_results={name:conditions(fresh,Path(path)) for name,path in (
 ('predecessor','results/phase4/predecessor/selected/development-conditions'),
 ('instruction-parent','results/phase4/instruction-parent/instruction-seed2026/development-conditions'))}
legacy=Path('data/phase4/legacy-conditions-regression-v1')
if not legacy.exists():
 legacy.mkdir(parents=True)
 source=Path('data/phase3/coverage-v1/test.jsonl')
 write_jsonl(legacy/'regression.jsonl',[r for r in read_examples(source) if r['family']=='numeric_rule'])
 register(legacy,{'regression':legacy/'regression.jsonl'},{'source_test_sha256':digest(source),'scope':'Exposed Phase 3 numeric regression. Excluded from independent acceptance.'})
temps=json.loads((root/'calibration-brier-fit.json').read_text())['temperatures']
if not (root/'legacy-conditions/summary.json').exists():
 run_evaluation(legacy/'regression.jsonl',root/'legacy-conditions',Settings(adapter=str(root/'adapter'),device='cuda',**{f'{p}_temperature':t for p,t in temps.items()}))
legacy_results=conditions(legacy/'regression.jsonl',root/'legacy-conditions')
checks={
 'retain_each_development_task_within_2pp':all(v['local']['accuracy']>=v['reference']['accuracy']-.02-1e-12 for c in comparisons.values() for v in c['families'].values()),
 'retain_new_condition_accuracy':all(fresh_results['all']['accuracy']>=v['all']['accuracy']-1e-12 for v in baseline_results.values()),
 'new_gate_and_missing_each_95pct':all(fresh_results[k]['accuracy']>=.95 for k in ('gate_false','missing')),
 'legacy_accuracy_recovers_predecessor':legacy_results['all']['accuracy']>=146/172,
 'legacy_gate_and_missing_each_90pct':all(legacy_results[k]['accuracy']>=.9 for k in ('gate_false','missing')),
}
result={'adapter_weights_sha256':digest(root/'adapter/adapter_model.safetensors'),'data_manifest_sha256':digest(data/'experiment.json'),'fresh_development':fresh_results,'baselines':baseline_results,'legacy_regression':legacy_results,'checks':checks,'eligible':all(checks.values()),'retention':{k:{'macro':v['macro_accuracy'],'families':{f:{'local':m['local']['accuracy'],'reference':m['reference']['accuracy']} for f,m in v['families'].items()}} for k,v in comparisons.items()}}
(root/'repair-assessment.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'checks':checks,'fresh':fresh_results['all'],'legacy':legacy_results['all'],'retention':result['retention']},indent=2),flush=True)
