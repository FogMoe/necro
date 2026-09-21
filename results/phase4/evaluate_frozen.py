import argparse
import json
import shutil
from pathlib import Path

from dotenv import load_dotenv

from necro.config import Settings
from necro.evaluation import run_evaluation
from necro.experiment_guard import digest, register, verify

parser = argparse.ArgumentParser()
parser.add_argument('backend', choices=['local', 'predecessor', 'jev'])
args = parser.parse_args()
data = Path('data/phase4/condition-v3')
selected = Path('results/phase4/selected')
final = Path('results/phase4/final')
selection = json.loads((selected/'selection.json').read_text())
test = verify(data, 'test', Path(selection['adapter']))
load_dotenv(override=False)
final.mkdir(exist_ok=True)
if args.backend == 'jev':
    run_evaluation(test, final/'jev-conditions', Settings(), backend='jev')
else:
    if args.backend == 'local':
        adapter = selection['adapter']
        temperatures = selection['temperatures']
    else:
        adapter = selection['reference_adapter']
        identity = digest(Path(adapter)/'adapter_model.safetensors')
        temperatures = selection['reference_temperatures'][identity]
    settings = Settings(adapter=adapter, device='cuda',
        **{f'{p}_temperature':t for p,t in temperatures.items()})
    run_evaluation(test, final/f'{args.backend}-conditions', settings)
    if args.backend == 'local':
        root = Path('data/phase4/phase3-regression-v1')
        source = Path('data/phase3/coverage-v1/test.jsonl')
        if not root.exists():
            root.mkdir()
            shutil.copy2(source, root/'regression.jsonl')
            register(root, {'regression':root/'regression.jsonl'}, {
                'source':str(source), 'source_sha256':digest(source),
                'scope':'Exposed Phase 3 multi-task regression, preserved byte for byte.'})
        run_evaluation(root/'regression.jsonl', final/'local-regression', settings)
        for name, old in [('predecessor','baseline-test'),('instruction','local-test'),('jev','jev-test')]:
            shutil.copytree(Path('results/phase3/final')/old, final/f'{name}-regression',
                ignore=shutil.ignore_patterns('remote-cache'))
