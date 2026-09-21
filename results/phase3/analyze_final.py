import json
from pathlib import Path
from necro.phase2_analysis import compare
from necro.robustness import summarize
from necro.evaluation import read_examples

root = Path('results/phase3/final')
for local, reference, name in (
    ('local-test', root/'jev-test', 'comparison'),
    ('baseline-test', root/'jev-test', 'predecessor-comparison'),
    ('local-regression', Path('results/phase2/v3/final/jev-test'), 'regression-comparison'),
):
    result = compare(root/local, reference)
    if name == 'regression-comparison':
        result['scope'] = 'Exposed historical regression, evaluated after final selection. Excluded from independent acceptance.'
    (root/f'{name}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'comparison':name, **{k:result[k] for k in ('macro_accuracy','macro_brier','performance_gates')}}, indent=2), flush=True)
robust = {key: summarize(Path('data/phase3/robustness-v1/test.jsonl'), root/parent, root/variant) for key,parent,variant in (('local','local-test','local-robustness'),('jev','jev-test','jev-robustness'))}
(root/'robustness-summary.json').write_text(json.dumps(robust, indent=2), encoding='utf-8')
local = read_examples(root/'local-test/judgments.jsonl')
reference = {r['id']:r for r in read_examples(root/'jev-test/judgments.jsonl')}
failures = [{**row, 'jev_prediction': reference[row['id']]['predicted'], 'jev_correct':reference[row['id']]['correct']} for row in local if not row['correct']]
(root/'failures.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in failures),encoding='utf-8')
print(json.dumps({'local_errors':len(failures),'reading_negation': {k:v['question_negation/reading_boolean/en'] for k,v in robust.items()}},indent=2),flush=True)
