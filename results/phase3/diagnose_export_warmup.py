import json
from pathlib import Path
from necro.backend import TransformersScorer
from necro.config import Settings
from necro.engine import DecisionEngine
from necro.evaluation import read_examples
from necro.schema import EvaluationRequest, EvaluationResponse

root=Path('results/phase3')
selection=json.loads((root/'selected/selection.json').read_text())
scorer=TransformersScorer(Settings(checkpoint=str(Path('artifacts/ScarletKc-Necro-0.8b-phase3').resolve()),device='cuda',**{f'{k}_temperature':v for k,v in selection['temperatures'].items()}))
scorer.load()
engine=DecisionEngine(scorer)
examples=read_examples(Path('data/phase3/coverage-v1/test.jsonl'))
requests=[EvaluationRequest.model_validate(r['request']) for r in examples]
engine.evaluate(requests[0])
actual=engine.evaluate_many(requests)
original=read_examples(root/'final/local-test/predictions.jsonl')
maximum=0; changed=0; differences=[]
for old,new in zip(original,actual,strict=True):
    old_response=EvaluationResponse.model_validate(old['response'])
    for key,a in new.answers.items():
        b=old_response.answers[key]
        x,y=([a.noul,1-a.noul],[b.noul,1-b.noul]) if a.type=='noul' else (list(a.probabilities.values()),list(b.probabilities.values()))
        delta=max(abs(p-q) for p,q in zip(x,y,strict=True))
        maximum=max(maximum,delta)
        changed+=max(range(len(x)),key=lambda i:x[i])!=max(range(len(y)),key=lambda i:y[i])
        if delta>1e-5: differences.append({'id':old['id'],'type':a.type,'max_difference':delta})
report={'examples':len(requests),'warmup':'same first request as original run_evaluation','max_probability_difference':maximum,'changed_argmax':changed,'differences_over_tolerance':differences}
(root/'export-warmup-diagnosis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='differences_over_tolerance'},indent=2),flush=True)
