import json
from pathlib import Path
from transformers import AutoTokenizer
from necro.backend import numeric_labels, single_token_labels
from necro.engine import prepare_task
from necro.evaluation import read_examples
from necro.schema import EvaluationRequest, Noul

root=Path('results/phase3')
base=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B',revision='2fc06364715b967f1860aea9cf38778875588b17',local_files_only=True,padding_side='left')
saved=AutoTokenizer.from_pretrained('artifacts/ScarletKc-Necro-0.8b-phase3',local_files_only=True,padding_side='left')
labels,_=single_token_labels(base)
rows=read_examples(Path('data/phase3/coverage-v1/test.jsonl'))
encodings=[]; differences=[]; normal=[]
for i,row in enumerate(rows):
 request=EvaluationRequest.model_validate(row['request'])
 question=request.questions['decision']
 task=prepare_task(request.state,question)
 names=['Yes','No'] if isinstance(question,Noul) else labels[:len(task.keys)] if len(task.keys)<=26 else numeric_labels(len(task.keys))
 messages=task.messages(names)
 prompts=[t.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False) for t in (base,saved)]
 ids=[t.encode(p,add_special_tokens=False) for t,p in zip((base,saved),prompts,strict=True)]
 encodings.append(ids[0])
 if ids[0]!=ids[1] or prompts[0]!=prompts[1]: differences.append({'id':row['id'],'lengths':list(map(len,ids)),'prompt_equal':prompts[0]==prompts[1]})
 if isinstance(question,Noul) or len(task.keys)<=26: normal.append(i)
normal.sort(key=lambda i:len(encodings[i]))
targets={'p3/xnli/test/198/zh','p3/xnli/test/1850/zh'}
selected=[]
for start in range(0,len(normal),4):
 batch=normal[start:start+4]
 if targets & {rows[i]['id'] for i in batch}:
  selected.append({'positions':batch,'lengths':[len(encodings[i]) for i in batch],'records':[rows[i] for i in batch]})
result={'tokenization_differences':differences,'batches':selected}
(root/'export-outlier-batches.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'differences':differences,'batches':[[r['id'] for r in b['records']] for b in selected]},ensure_ascii=False,indent=2),flush=True)
