import json
from pathlib import Path
from necro.config import Settings
from necro.evaluation import run_evaluation
from necro.diagnostics.numeric_regression import write_json
from necro.training.analysis.condition_balance import condition_slices

data=Path('data/phase5/condition-balance-v1/validation.jsonl')
root=Path('results/phase5/parent-development')
fit=json.loads(Path('results/phase3/instruction-seed2026/calibration-brier-fit.json').read_text())
settings=Settings(adapter='results/phase3/instruction-seed2026/adapter',device='cuda',**{f'{k}_temperature':v for k,v in fit['temperatures'].items()})
run_evaluation(data,root,settings)
slices=condition_slices(data,root)
write_json(root/'slices.json',slices)
print(json.dumps({k:v for k,v in slices.items() if k.startswith('transfer/') and k.count('/')<=2}),flush=True)
