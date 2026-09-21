import json
from pathlib import Path
from necro.evaluation import read_examples
from necro.experiment_guard import digest, register
from necro.training.data.training_data import write_jsonl

data=Path('data/phase4/condition-v1')
root=Path('results/phase4')
subset=Path('data/phase4/development-conditions-v1')
subset.mkdir(parents=True,exist_ok=False)
write_jsonl(subset/'development.jsonl',[r for r in read_examples(data/'validation.jsonl') if r['source']=='constructed-condition-v1'])
register(subset,{'development':subset/'development.jsonl'},{'parent_sha256':digest(data/'validation.jsonl'),'scope':'New condition-rule development slice of the registered repair experiment'})
plan={
 'purpose':'Repair wording-sensitive preconditions, retain existing tasks, and resolve export consistency. Ordinary unrelated errors do not expand training scope.',
 'data':str(data),'manifest_sha256':digest(data/'experiment.json'),
 'initial_adapter':'results/phase3/instruction-seed2026/adapter',
 'initial_weights_sha256':digest(Path('results/phase3/instruction-seed2026/adapter/adapter_model.safetensors')),
 'training':{'rank':16,'learning_rate':1e-5,'batch_size':4,'accumulation':2,'epochs':1,'seed':2026,'objective':'answer-ce'},
 'selection':{
  'retention':'On the existing 1464 development records, no family falls more than 2 percentage points below either predecessor or instruction parent. Report language slices and paired source-group intervals.',
  'new_conditions':'Overall new-condition development accuracy must at least retain both baselines. Disabled-gate and missing-field accuracy must each reach 95 percent.',
  'legacy_diagnostic':'Original-field, original-wording accuracy must recover the predecessor; disabled-gate and missing-field accuracy must each reach 90 percent.',
  'blind_holdout':'Do not evaluate the registered 720 new test records until candidate selection and calibration are frozen.',
 },
 'trial_limit':'One initial repair and at most one targeted follow-up trial in this bounded experiment. Assess diminishing returns after capability stability, not from a regressing model.',
 'reference':'Jev is a measured comparison, not a release percentage target.',
 'export':'Preserve exact weight comparisons and investigate the two probability outliers. Do not waive the failed tolerance without explaining and verifying the numerical cause.'
}
path=root/'condition-repair-plan.json'
if path.exists(): raise ValueError('Plan exists')
path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
print(json.dumps({'plan':str(path),'training_rows':len(read_examples(data/'train.jsonl'))},indent=2))
