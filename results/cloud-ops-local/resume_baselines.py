"""核对已有完整结果，仅补跑缺少的基线评测。"""
from pathlib import Path

from necro.evaluation import read_examples, run_evaluation
from necro.experiment_guard import verify
from necro.training.cloud import (
    DATA, EVALUATION, OUTPUT, REFERENCES, project_report, read,
    release_memory, settings, validate_report, verify_bundle,
)

verify_bundle()
verify(DATA, 'train')

def validate_complete(target, dataset, adapter, temperatures):
    validate_report(target, dataset, adapter, temperatures)
    expected = [r['id'] for r in read_examples(dataset)]
    for filename in ('predictions.jsonl', 'judgments.jsonl'):
        actual = [r['id'] for r in read_examples(target / filename)]
        if actual != expected:
            raise ValueError(f'Incomplete or reordered saved baseline: {target}/{filename}')

for name in ('official', *REFERENCES):
    adapter = Path('reference') / name / 'adapter' if name != 'official' else None
    temperatures = read(adapter.parent / 'calibration.json')['temperatures'] if adapter else None
    for cohort, dataset in (
        ('development', DATA / 'validation.jsonl'),
        ('regression', EVALUATION / 'regression.jsonl'),
        ('conditions', EVALUATION / 'conditions.jsonl'),
    ):
        target = OUTPUT / 'baseline' / name / cohort
        if target.exists():
            validate_complete(target, dataset, adapter, temperatures)
            print(f'Validated existing {name}/{cohort}', flush=True)
        else:
            print(f'Evaluating {name}/{cohort}', flush=True)
            run_evaluation(dataset, target, settings(adapter, temperatures))
            validate_complete(target, dataset, adapter, temperatures)
            release_memory()
    target = OUTPUT / 'baseline' / name / 'retention'
    if not target.exists():
        project_report(DATA / 'validation.jsonl', EVALUATION / 'retention.jsonl',
                       OUTPUT / 'baseline' / name / 'development/predictions.jsonl', target)
    validate_complete(target, EVALUATION / 'retention.jsonl', adapter, temperatures)
print('All baseline cohorts complete and validated', flush=True)
