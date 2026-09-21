import json
from pathlib import Path
import torch
from necro.backend import TransformersScorer
from necro.config import Settings

root=Path('results/phase3')
sel=json.loads((root/'selected/selection.json').read_text())
a=TransformersScorer(Settings(adapter=sel['adapter'],device='cuda'))
b=TransformersScorer(Settings(checkpoint=str(Path('artifacts/ScarletKc-Necro-0.8b-phase3').resolve()),device='cuda'))
a.load(); b.load()
left=dict(a.model.named_buffers()); right=dict(b.model.named_buffers())
differences=[]
for name,x in left.items():
 y=right[name]
 if x.dtype!=y.dtype or not torch.equal(x,y): differences.append({'name':name,'left_dtype':str(x.dtype),'right_dtype':str(y.dtype),'shape':list(x.shape),'max_difference':float((x.float()-y.float()).abs().max())})
ac,bc=a.model.config.to_dict(),b.model.config.to_dict()
configs={k:{'adapter':ac.get(k),'merged':bc.get(k)} for k in ac.keys()|bc.keys() if ac.get(k)!=bc.get(k)}
classes={n:type(m).__module__+'.'+type(m).__name__ for n,m in a.model.named_modules()}
class_differences={n:{'adapter':classes.get(n),'merged':type(m).__module__+'.'+type(m).__name__} for n,m in b.model.named_modules() if classes.get(n)!=type(m).__module__+'.'+type(m).__name__}
report={'buffer_count':len(left),'buffer_differences':differences,'config_differences':configs,'module_class_differences':class_differences}
(root/'export-buffer-diagnosis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2),flush=True)
