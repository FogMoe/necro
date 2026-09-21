import json
from pathlib import Path
from statistics import mean

from necro.development_comparison import compare_development
from necro.experiment_guard import digest

root = Path('results/phase3')
comparison = compare_development(root / 'instruction-seed2027/development-brier', root / 'instruction-seed2026/development-brier')
(root / 'final-seed-comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
models = {}
for name in ('baseline', 'coverage-seed2026', 'instruction-seed2026', 'instruction-seed2027'):
    summary = json.loads((root / name / 'development-brier/summary.json').read_text())
    record = {'macro_accuracy': summary['macro_family_accuracy'], 'macro_brier': mean(v['brier'] for v in summary['families'].values()), 'families': {k: v['accuracy'] for k, v in summary['families'].items()}}
    audit = {}
    for role in ('development', 'calibration'):
        values = [a['noul'] for line in (root / name / role / 'predictions.jsonl').read_text(encoding='utf-8').splitlines() for a in json.loads(line)['response']['answers'].values() if a['type'] == 'noul']
        audit[role] = {'noul_count': len(values), 'saturated_probabilities': sum(p in (0, 1) for p in values)}
    record['raw_probability_precision'] = audit
    robust = root / name / 'robustness-summary.json'
    if robust.exists():
        record['reading_negation'] = json.loads(robust.read_text())['question_negation/reading_boolean/en']
    models[name] = record
decision = {
    'new_test_exposed': False,
    'selected_run': 'instruction-seed2026',
    'adapter_weights_sha256': digest(root / 'instruction-seed2026/adapter/adapter_model.safetensors'),
    'rationale': 'Instruction seed 2026 meets the registered critical-repair rule: development reading-negation consistency rises from 0/10 to 10/10, macro accuracy improves by 0.613 percentage points over the coverage parent, and the largest task regression is 0.775 percentage points. The final matched seed 2027 replication scores lower on development macro accuracy and Brier and reaches only 8/10 reading-negation consistency. Retain seed 2026. The small aggregate gain and seed variability close lightweight training exploration. Independent performance gates remain unchanged.',
    'calibration_objective': 'brier',
    'further_training': False,
    'development_models': models,
    'seed_comparison': 'results/phase3/final-seed-comparison.json',
}
path = root / 'final-decision.json'
if path.exists():
    raise ValueError('Selection decision already exists')
path.write_text(json.dumps(decision, indent=2), encoding='utf-8')
print(json.dumps({k: {'accuracy': v['macro_accuracy'], 'brier': v['macro_brier'], 'precision': v['raw_probability_precision']} for k,v in models.items()}, indent=2))
