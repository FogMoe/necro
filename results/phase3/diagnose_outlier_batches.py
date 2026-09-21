import json
from pathlib import Path
from necro.backend import TransformersScorer
from necro.config import Settings
from necro.engine import DecisionEngine
from necro.evaluation import read_examples
from necro.schema import EvaluationRequest

root=Path('results/phase3')
sel=json.loads((root/'selected/selection.json').read_text())
batches=json.loads((root/'export-outlier-batches.json').read_text())['batches']
rows=[r for b in batches for r in b['records']]
requests=[EvaluationRequest.model_validate(r['request']) for r in rows]
temps={f'{k}_temperature':v for k,v in sel['temperatures'].items()}
scorers={
 'adapter':TransformersScorer(Settings(adapter=sel['adapter'],device='cuda',**temps)),
 'merged':TransformersScorer(Settings(checkpoint=str(Path('artifacts/ScarletKc-Necro-0.8b-phase3').resolve()),device='cuda',**temps)),
}
original={r['id']:r['response']['answers']['decision'] for r in read_examples(root/'final/local-test/predictions.jsonl')}
output={r['id']:{'original':original[r['id']]} for r in rows}
for kind,scorer in scorers.items():
 scorer.load()
 engine=DecisionEngine(scorer)
 for repetition in range(2):
  values=engine.evaluate_many(requests)
  for row,value in zip(rows,values,strict=True): output[row['id']][f'{kind}_{repetition}']=value.answers['decision'].model_dump()
report={'rows':output,'weight_strides':{kind:{name:list(p.stride()) for name,p in scorer.model.named_parameters() if 'language_model.layers.0' in name} for kind,scorer in scorers.items()}}
(root/'export-outlier-comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'examples':len(rows),'saved':'results/phase3/export-outlier-comparison.json'}),flush=True)
