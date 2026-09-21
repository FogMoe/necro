import json
from pathlib import Path
from necro.experiment_guard import digest

root=Path('results/phase4')
plan={
 'scope':'Correct the verified condition-combination omission and retain existing tasks. No Jev percentage target or new task family.',
 'reason':'The two initial trials failed the confirmed missing-field check. The v2 audit finds zero implicit guard-first equality examples where the gate is missing and the numerical comparison would pass. A complete matrix is required before training again.',
 'audit_sha256':digest(root/'missing-combination-audit.json'),
 'data':'data/phase4/condition-v3',
 'manifest_sha256':digest(Path('data/phase4/condition-v3/experiment.json')),
 'matrix':{'combinations':800,'presentations_per_combination':4,'dimensions':['style','operator','language','question polarity','gate or missing-field case','counterfactual comparison truth']},
 'parent':'results/phase3/instruction-seed2026/adapter',
 'training':{'learning_rate':5e-6,'epochs':1,'batch_size':4,'accumulation':2,'seed':2026,'rank':16,'objective':'answer-ce'},
 'retention':'Include the same other-task replay and 532 original numeric replay records. Lower the learning rate to reduce parameter movement.',
 'unchanged':'Development, calibration, sealed test and all acceptance checks remain unchanged.',
 'stopping':'Accept only after the confirmed defects, task retention, export/reload/API checks and honest reporting requirements pass. Do not add rounds to pursue Jev accuracy.'
}
path=root/'condition-matrix-plan.json'
if path.exists(): raise ValueError('Matrix repair is already registered')
path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
print(json.dumps({'plan':str(path)}))
