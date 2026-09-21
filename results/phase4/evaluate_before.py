import json
from pathlib import Path
from necro.config import Settings
from necro.evaluation import run_evaluation

for name,adapter,run in (
 ('predecessor','results/phase2/v3/selected/adapter','results/phase3/baseline'),
 ('instruction-parent','results/phase3/instruction-seed2026/adapter','results/phase3/instruction-seed2026'),
):
 temps=json.loads((Path(run)/'calibration-brier-fit.json').read_text())['temperatures']
 settings=Settings(adapter=adapter,device='cuda',**{f'{p}_temperature':t for p,t in temps.items()})
 report=run_evaluation(Path('data/phase4/development-conditions-v1/development.jsonl'),Path('results/phase4')/name/settings.adapter.split('/')[-2]/'development-conditions',settings)
 print(json.dumps({'name':name,'accuracy':report['overall']['accuracy']}),flush=True)
 import gc,torch
 gc.collect(); torch.cuda.empty_cache()
