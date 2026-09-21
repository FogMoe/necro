import gc
import json
from pathlib import Path

import torch

from necro.config import Settings
from necro.diagnostics.numeric_regression import write_json
from necro.evaluation import run_evaluation
from necro.experiment_guard import digest
from necro.training.analysis.assess_candidate import assess
from necro.training.analysis.condition_balance import condition_slices
from necro.training.trainer import train
from necro.calibration import transform_report
from necro.training.analysis.phase2_analysis import project_report
from necro.training.analysis.condition_balance import minimum_accuracy

ROOT = Path('results/phase5')
DATA = Path('data/phase5/condition-balance-v1')
PARENT = Path('results/phase3/instruction-seed2026/adapter')


def screen(run):
    path=Path('data/phase4/condition-regression-v1/regression.jsonl')
    fixture=Path('docs/process/evidence/condition-false-rejections-2026-09-21.jsonl')
    target=run/'screen-conditions'
    if not target.exists():
        run_evaluation(path,target,Settings(adapter=str(run/'adapter'),device='cuda'))
        gc.collect()
        torch.cuda.empty_cache()
    metadata=json.loads((target/'summary.json').read_text())['metadata']
    if metadata['adapter_weights_sha256']!=digest(run/'adapter/adapter_model.safetensors') or metadata['dataset_sha256']!=digest(path):
        raise ValueError('Screen predictions do not match current weights/data')
    if not (run/'screen-fixture').exists():
        project_report(path,fixture,target/'predictions.jsonl',run/'screen-fixture')
    s=condition_slices(path,target)
    f=condition_slices(fixture,run/'screen-fixture')
    checks={
        'known_false_rejections_resolved':minimum_accuracy(f,['all'],1),
        'complete_eligible':minimum_accuracy(s,[f'{lang}/{case}' for lang in ['en','zh'] for case in ['complete','eligible']],.98),
        'preconditions':minimum_accuracy(s,[f'{lang}/{case}' for lang in ['en','zh'] for case in ['gate_false','missing']],.95),
    }
    write_json(run/'screen-review.json',{'checks':checks,'passed':all(checks.values()),'conditions':s,'fixture':f,'adapter_weights_sha256':digest(run/'adapter/adapter_model.safetensors')})
    print(json.dumps({'screen':str(run),'checks':checks,'fixture':f['all'],'conditions':{k:v for k,v in s.items() if k in ['all','complete','eligible','zh/complete','en/gate_false','en/missing']}}),flush=True)
    return all(checks.values())


def evaluate(run, data=DATA):
    assess(data, run / 'adapter', run, objective='brier')
    temperatures = json.loads((run / 'calibration-brier-fit.json').read_text())['temperatures']
    settings = Settings(adapter=str(run / 'adapter'), device='cuda', **{f'{k}_temperature':v for k,v in temperatures.items()})
    slices = {'development':condition_slices(data/'validation.jsonl',run/'development-brier')}
    for name,path in (
        ('conditions',Path('data/phase4/condition-regression-v1/regression.jsonl')),
        ('regression',Path('data/phase4/phase3-regression-v1/regression.jsonl')),
        ('fixture',Path('docs/process/evidence/condition-false-rejections-2026-09-21.jsonl')),
    ):
        if not (run/name/'summary.json').exists():
            if (run/f'screen-{name}'/'predictions.jsonl').exists():
                transform_report(path,run/f'screen-{name}'/'predictions.jsonl',run/name,temperatures)
            else:
                run_evaluation(path,run/name,settings)
            gc.collect()
            torch.cuda.empty_cache()
        slices[name] = condition_slices(path,run/name)
    write_json(run/'slices.json',slices)
    print(json.dumps({'run':str(run),'slices':{name:{k:v for k,v in s.items() if k in ['all','complete','eligible','zh/eligible','en/gate_false','en/missing']} for name,s in slices.items()}}),flush=True)


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('name')
    p.add_argument('--rate',type=float,default=2.5e-6)
    p.add_argument('--control',action='store_true')
    p.add_argument('--data',type=Path)
    p.add_argument('--full',action='store_true')
    args=p.parse_args()
    run=ROOT/args.name
    data=args.data or (Path('data/phase4/condition-v3') if args.control else DATA)
    if not (run/'adapter').exists():
        train(data,run,batch_size=4,accumulation=2,learning_rate=args.rate,rank=16,seed=2026,initial_adapter=PARENT,model_id=f'necro-phase5-{args.name}',gradient_checkpointing=True)
    gc.collect()
    torch.cuda.empty_cache()
    if screen(run) or args.full:
        evaluate(run, data)
