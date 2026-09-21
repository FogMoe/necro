import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
from necro.config import Settings
from necro.evaluation import run_evaluation

p = argparse.ArgumentParser()
p.add_argument('job', choices=['selected', 'baseline', 'robustness', 'regression', 'jev', 'jev-robustness'])
a = p.parse_args()
load_dotenv(override=False)
selected = json.loads(Path('results/phase3/selected/selection.json').read_text())
jobs = {
    'selected': ('data/phase3/coverage-v1/test.jsonl', 'local-test'),
    'baseline': ('data/phase3/coverage-v1/test.jsonl', 'baseline-test'),
    'robustness': ('data/phase3/robustness-v1/test.jsonl', 'local-robustness'),
    'regression': ('data/phase3/phase2-regression/regression.jsonl', 'local-regression'),
    'jev': ('data/phase3/coverage-v1/test.jsonl', 'jev-test'),
    'jev-robustness': ('data/phase3/robustness-v1/test.jsonl', 'jev-robustness'),
}
adapter = selected['reference_adapter'] if a.job == 'baseline' else selected['adapter']
temps = json.loads(Path('results/phase3/selected/reference-calibration.json').read_text())['temperatures'] if a.job == 'baseline' else selected['temperatures']
settings = Settings(adapter=adapter, device='cuda', **{f'{k}_temperature': v for k,v in temps.items()})
data, name = jobs[a.job]
output = Path('results/phase3/final') / name
if (output / 'summary.json').exists():
    raise ValueError(f'Completed output already exists: {output}')
report = run_evaluation(Path(data), output, settings, backend='jev' if a.job.startswith('jev') else 'local')
print(json.dumps({'job': a.job, 'metadata': report['metadata'], 'macro_accuracy': report['macro_family_accuracy']}, indent=2), flush=True)
