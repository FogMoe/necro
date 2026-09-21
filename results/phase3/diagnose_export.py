import gc
import json
from pathlib import Path
import torch
from safetensors import safe_open
from necro.backend import TransformersScorer
from necro.config import Settings
from necro.engine import DecisionEngine
from necro.evaluation import read_examples
from necro.schema import EvaluationRequest

root = Path('results/phase3')
selection = json.loads((root/'selected/selection.json').read_text())
package = Path('artifacts/ScarletKc-Necro-0.8b-phase3')
temps = {f'{k}_temperature':v for k,v in selection['temperatures'].items()}
adapter = TransformersScorer(Settings(adapter=selection['adapter'],device='cuda',**temps))
adapter.load()
state = adapter.model.state_dict()
saved_mismatches = []
with safe_open(package/'model.safetensors',framework='pt',device='cpu') as saved:
    for name in saved.keys():
        current, stored = state[name].detach().cpu(), saved.get_tensor(name)
        if current.dtype != stored.dtype or not torch.equal(current,stored):
            saved_mismatches.append({'name':name,'active_dtype':str(current.dtype),'saved_dtype':str(stored.dtype),'max_difference':float((current.float()-stored.float()).abs().max())})
report = {'saved_weight_tensors':len(state),'saved_mismatches':saved_mismatches}
print(json.dumps(report),flush=True)
merged = TransformersScorer(Settings(checkpoint=str(package.resolve()),device='cuda',**temps))
merged.load()
reloaded_mismatches = []
for name, current in state.items():
    loaded = merged.model.state_dict()[name]
    if current.dtype != loaded.dtype or not torch.equal(current,loaded):
        reloaded_mismatches.append({'name':name,'active_dtype':str(current.dtype),'reloaded_dtype':str(loaded.dtype),'max_difference':float((current.float()-loaded.float()).abs().max())})
report['reloaded_mismatches'] = reloaded_mismatches
print(json.dumps({'reload_mismatches':reloaded_mismatches}),flush=True)
rows = read_examples(Path('data/phase3/coverage-v1/test.jsonl'))
subset=[]
for family in dict.fromkeys(r['family'] for r in rows):
    subset += [r for r in rows if r['family']==family][:4]
requests=[EvaluationRequest.model_validate(r['request']) for r in subset]
ae,me=DecisionEngine(adapter),DecisionEngine(merged)
a1=ae.evaluate_many(requests)
m1=me.evaluate_many(requests)
a2=ae.evaluate_many(requests)
m2=me.evaluate_many(requests)
def difference(left,right):
    largest=0.0; changed=0
    for a,b in zip(left,right,strict=True):
        for key,x in a.answers.items():
            y=b.answers[key]
            p,q=([x.noul,1-x.noul],[y.noul,1-y.noul]) if x.type=='noul' else (list(x.probabilities.values()),list(y.probabilities.values()))
            largest=max(largest,max(abs(v-w) for v,w in zip(p,q,strict=True)))
            changed+=max(range(len(p)),key=lambda i:p[i])!=max(range(len(q)),key=lambda i:q[i])
    return {'maximum_probability_difference':largest,'changed_argmax':changed}
report['same_batch']={'examples':len(requests),'adapter_vs_merged':difference(a1,m1),'adapter_repeat':difference(a1,a2),'merged_repeat':difference(m1,m2)}
(root/'export-diagnosis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report['same_batch'],indent=2),flush=True)
del state,adapter,merged,ae,me
gc.collect()
torch.cuda.empty_cache()
