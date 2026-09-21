import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from necro.config import Settings
from necro.evaluation import run_evaluation
from necro.experiment_guard import digest, register
from necro.training.release.stability_assessment import paired_comparison, condition_slices

output = Path('results/phase4/quick-release-audit')
output.mkdir(exist_ok=True)
root = Path('data/phase4/condition-regression-v1')
source = Path('data/phase4/condition-v3/test.jsonl')
if not root.exists():
    root.mkdir()
    shutil.copy2(source, root/'regression.jsonl')
    register(root, {'regression':root/'regression.jsonl'}, {
        'source':str(source), 'source_sha256':digest(source),
        'scope':'Previously exposed condition holdout, used only for post-hoc base and parent comparisons.'})
conditions = root/'regression.jsonl'
parent = Path('results/phase3/instruction-seed2026/adapter')
temps=json.loads(Path('results/phase3/instruction-seed2026/calibration-brier-fit.json').read_text())['temperatures']
if not (output/'parent-conditions/summary.json').exists():
    run_evaluation(conditions,output/'parent-conditions',Settings(adapter=str(parent),device='cuda',
        **{f'{p}_temperature':t for p,t in temps.items()}))
print(json.dumps({'completed':'parent-conditions'}),flush=True)
contract=json.loads((parent/'necro_adapter.json').read_text())
base=snapshot_download(contract['checkpoint'],revision=contract['revision'],local_files_only=True)
settings=Settings(checkpoint=base,device='cuda',temperature=1,
    choice_temperature=1,noul_temperature=1,score_temperature=1)
for name,dataset in [('regression',Path('data/phase4/phase3-regression-v1/regression.jsonl')),('conditions',conditions)]:
    target=output/f'base-{name}'
    if not (target/'summary.json').exists():
        run_evaluation(dataset,target,settings)
    comparison=paired_comparison(Path(f'results/phase4/final/local-{name}'),target,
        'Post-hoc comparison with the untrained base on identical exposed requests; base temperatures are 1.')
    (output/f'{name}-base-comparison.json').write_text(json.dumps(comparison,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'completed':f'base-{name}','macro':comparison['macro_accuracy']}),flush=True)
comparison=paired_comparison(Path('results/phase4/final/local-conditions'),output/'parent-conditions',
    'Post-hoc comparison with the direct training parent on the exposed condition cohort.')
(output/'conditions-parent-comparison.json').write_text(json.dumps(comparison,indent=2,allow_nan=False),encoding='utf-8')
details={name:condition_slices(conditions,path) for name,path in [
    ('base',output/'base-conditions'),('parent',output/'parent-conditions'),
    ('trained',Path('results/phase4/final/local-conditions')),('jev',Path('results/phase4/final/jev-conditions'))]}
(output/'condition-slices.json').write_text(json.dumps(details,indent=2,allow_nan=False),encoding='utf-8')
(output/'provenance.json').write_text(json.dumps({'base_checkpoint':contract['checkpoint'],'base_revision':contract['revision'],
    'base_snapshot':base,'base_temperatures':1,'selection_changed':False,
    'condition_sha256':digest(conditions),'scope':'Minimal post-hoc pre-release diagnosis; no training or additional Jev requests.'},indent=2),encoding='utf-8')
print(json.dumps({'completed':'audit'}),flush=True)
