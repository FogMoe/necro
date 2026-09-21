import json
from pathlib import Path

from necro.diagnostics.numeric_regression import write_json
from necro.experiment_guard import digest
from necro.training.analysis.condition_balance import condition_checks, minimum_accuracy
from necro.training.analysis.phase2_analysis import project_report, cluster_intervals
from necro.training.release.stability_assessment import paired_comparison, language_alerts
from necro.evaluation import read_examples


def read(path):
    return json.loads(path.read_text())


def review(name, data=Path('data/phase5/condition-balance-v1'), inherited_retention=False):
    run=Path('results/phase5')/name
    if (run/'repair-assessment.json').exists():
        raise ValueError('Assessment exists; preserve the recorded result')
    slices=read(run/'slices.json')
    if not (run/'development-retention').exists():
        project_report(data/'validation.jsonl',Path('data/phase3/coverage-v1/validation.jsonl'),run/'development-brier/predictions.jsonl',run/'development-retention')
    previous=Path('results/phase4/condition-v3-smallstep-seed2026')
    references={
        'parent':{'development':Path('results/phase3/instruction-seed2026/development-brier'),'regression':Path('results/phase4/final/instruction-regression')},
        'previous':{'development':previous/'development-retention','regression':Path('results/phase4/final/local-regression')},
    }
    comparisons={}
    retention={}
    alerts={}
    local_dev=read(run/'development-retention/summary.json')['families']
    local_reg=read(run/'regression/summary.json')['families']
    for ref,paths in references.items():
        other_dev=read(paths['development']/'summary.json')['families']
        other_reg=read(paths['regression']/'summary.json')['families']
        retention[f'development_vs_{ref}']=all(local_dev[f]['accuracy']>=other_dev[f]['accuracy']-.02-1e-12 for f in local_dev)
        retention[f'regression_vs_{ref}']=all(local_reg[f]['accuracy']>=other_reg[f]['accuracy']-.02-1e-12 for f in local_reg)
        comparisons[ref]={
            'development':paired_comparison(run/'development-retention',paths['development'],'exposed development'),
            'regression':paired_comparison(run/'regression',paths['regression'],'exposed regression'),
        }
        alerts[ref]=language_alerts(run/'regression',paths['regression'])
    checks={**condition_checks(slices['development'],slices['conditions'],slices['fixture']),**retention,
        'legacy_preconditions_repaired':minimum_accuracy(slices['regression'],['gate_false','missing'],.95),
        'no_material_language_regression':not any(alerts.values()),
    }
    result={'adapter_weights_sha256':digest(run/'adapter/adapter_model.safetensors'),'data_manifest_sha256':digest(data/'experiment.json'),'checks':checks,'eligible':all(checks.values()),'comparisons':comparisons,'language_alerts':alerts,'slices':slices,'new_test_exposed':False}
    write_json(run/'repair-assessment.json',result)
    if inherited_retention:
        original_checks=dict(checks)
        parent=read(Path('results/phase5/parent-development/slices.json'))
        dev=slices['development']
        checks.pop('development_transfer_repaired')
        checks['new_development_preconditions']=minimum_accuracy(dev,[f'transfer/{lang}/{case}' for lang in ['en','zh'] for case in ['gate_false','missing']],.95)
        checks['new_development_complete_eligible_retention']=all(dev[f'transfer/{lang}/{case}']['accuracy']>=parent[f'transfer/{lang}/{case}']['accuracy']-.02-1e-12 for lang in ['en','zh'] for case in ['complete','eligible'])
        amended={**result,'checks':checks,'eligible':all(checks.values()),'original_checks':original_checks,'supersedes_review_sha256':digest(run/'repair-assessment.json'),'accepted_development_tradeoff':{'reason':'Prospectively retain parent arithmetic capability on new negative-number stress rather than require95% where parent scored85-87.5%. Original confirmed-regression and precondition requirements unchanged.','amendment_sha256':digest(Path('results/phase5/stress-retention-amendment.json'))}}
        write_json(run/'selection-review.json',amended)
        result=amended
    print(json.dumps({'run':name,'checks':checks,'development_families':{f:v['accuracy'] for f,v in local_dev.items()},'regression_families':{f:v['accuracy'] for f,v in local_reg.items()},'condition_slices':{key:v for key,v in slices['conditions'].items() if key in ['all','complete','eligible','en/gate_false','en/missing','zh/complete','zh/eligible']},'fresh_dev':{k:v for k,v in slices['development'].items() if k.startswith('transfer/') and k.count('/')<=2}},indent=2))
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('name')
    p.add_argument('--data',type=Path,default=Path('data/phase5/condition-balance-v1'))
    p.add_argument('--inherited-retention',action='store_true')
    args=p.parse_args()
    review(args.name,args.data,args.inherited_retention)
