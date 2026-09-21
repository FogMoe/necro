import json
from pathlib import Path
from necro.experiment_guard import digest

root=Path('results/phase4')
first=json.loads((root/'condition-seed2026/repair-assessment.json').read_text())
plan={
 'parent':'results/phase3/instruction-seed2026/adapter',
 'data':'data/phase4/condition-v2',
 'data_manifest_sha256':digest(Path('data/phase4/condition-v2/experiment.json')),
 'prior_assessment_sha256':digest(root/'condition-seed2026/repair-assessment.json'),
 'reason':'Initial repair recovered original-wording accuracy and disabled-gate handling but left missing fields at 5/12 and lost four correct numeric development answers.',
 'changes':['Include implicit-required-field formulations in two training styles.','Hold the rule text constant across paired enabled, disabled and missing states.','Add the 532 previously trained numeric replay records to retain the existing numeric task.'],
 'unchanged':['Same parent, seed, rank, optimizer, learning rate 1e-5, batch 4, accumulation 2 and one epoch.','Development, calibration and sealed test files remain byte-identical.','The initial selection criteria remain in force.'],
 'training_trial':'Second and final planned repair trial in this bounded experiment.',
 'first_trial_checks':first['checks'],
}
path=root/'condition-followup-plan.json'
if path.exists(): raise ValueError('Follow-up already registered')
path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
print(json.dumps({'plan':str(path)},indent=2))
