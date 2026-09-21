import json
from pathlib import Path
from necro.backend import TransformersScorer
from necro.config import Settings
from necro.engine import DecisionEngine
from necro.schema import EvaluationRequest

root=Path('results/phase3')
saved=json.loads((root/'export-outlier-comparison.json').read_text())['rows']
batches=json.loads((root/'export-outlier-batches.json').read_text())['batches']
rows=[r for b in batches for r in b['records']]
selection=json.loads((root/'selected/selection.json').read_text())
scorer=TransformersScorer(Settings(checkpoint=str(Path('artifacts/ScarletKc-Necro-0.8b-phase3').resolve()),device='cuda',**{f'{k}_temperature':v for k,v in selection['temperatures'].items()}))
scorer.load()
before=sum(p.requires_grad for p in scorer.model.parameters())
scorer.model.requires_grad_(False)
responses=DecisionEngine(scorer).evaluate_many([EvaluationRequest.model_validate(r['request']) for r in rows])
differences={}
for row,response in zip(rows,responses,strict=True):
 actual=response.answers['decision'].model_dump()
 def values(x): return list(x['probabilities'].values()) if 'probabilities' in x else [x['noul']]
 differences[row['id']]={k:max(abs(a-b) for a,b in zip(values(actual),values(saved[row['id']][k]))) for k in ('adapter_0','merged_0')}
out={'parameters_requiring_grad_before':before,'after':sum(p.requires_grad for p in scorer.model.parameters()),'differences':differences}
(root/'export-gradflags-diagnosis.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2),flush=True)
