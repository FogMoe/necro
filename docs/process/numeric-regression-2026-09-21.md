# Numeric-rule regression diagnosis, 2026-09-21

Both training stages increased sensitivity to rule wording. Errors concentrate on handling a disabled gate or missing fields when the rule states rejection conditions first. The same numerical comparisons remain accurate under an equivalent conjunction-based formulation.

## Diagnostic design

The cohort contains all 172 numeric-rule records from the exposed Phase 3 test, grouped into 20 source components. Every variant preserves values, comparison operator, gate state, question polarity and expected answer. Two factors vary independently:

- Field names: the original `stock/requested/warehouse_open`, the development names `available_slots/required_slots/enabled`, or neutral names `value_a/value_b/gate`.
- Wording: rejection conditions first, followed by the numeric rule, or an eligibility rule expressed as “if and only if” the comparison and gate conditions hold.

The six conditions produce 1,032 records per checkpoint and 4,128 judgments across four checkpoints. Comparisons, required fields, gate values and question polarity were recalculated before registration. A separate audit recalculated 4,860 numeric records across training, replay, development, calibration and the original test, finding no label mismatches. This count includes replay records repeated across stages.

These are post-test diagnostics. Variants share their original source groups, and subsequent independent acceptance requires newly registered data. Original tests, selections, weights and predictions remain preserved.

## Where the regression appeared

| Checkpoint | Original fields and wording | Original fields, if-and-only-if wording |
|---|---:|---:|
| Phase 2 predecessor | 146/172, 84.88% | 172/172, 100.00% |
| Coverage refinement | 133/172, 77.33% | 172/172, 100.00% |
| Instruction seed2026 | 122/172, 70.93% | 172/172, 100.00% |
| Instruction seed2027 | 129/172, 75.00% | 172/172, 100.00% |

Coverage refinement lost 13 previously correct answers and gained none. Instruction seed2026 lost another 11 and gained none. The combined decrease was 13.95 percentage points, with a paired source-group bootstrap 95% interval of [-18.07, -9.88] points. Seed2027 also scored below the predecessor.

## Preconditions account for most errors

The table below retains the original fields and wording. Values are correct-answer counts.

| Condition | Questions | Predecessor | Coverage | Instruction 2026 | Instruction 2027 |
|---|---:|---:|---:|---:|---:|
| Complete fields and enabled gate | 120 | 119 | 114 | 114 | 113 |
| Disabled gate | 40 | 23 | 15 | 4 | 12 |
| Missing fields | 12 | 4 | 4 | 4 | 4 |

All 11 additional errors from the final seed2026 instruction stage involve a disabled gate. The numerical relation holds in these records, but the false gate must determine the eligibility result. The model frequently ignores that precondition under the original wording.

Changing only the wording corrected all 50 seed2026 errors. Keeping the original wording and changing only field names produced 76.74% with development fields and 77.33% with neutral fields. Field names influence the result, while wording accounts for the larger change. Original English and Chinese accuracy fell to 72.09% and 69.77%, respectively.

The predecessor already had the same weakness: 23/40 correct with a disabled gate and 4/12 with missing fields. Reverting reduces the regression, but still leaves precondition handling to address.

## Training and selection gaps

Coverage refinement replayed 532 numeric records from 18 source groups. Instruction refinement increased this to 970 records from 35 groups. Only 68 instruction-refinement numeric records had a disabled gate, about 7.0%, compared with 23.3% of the numeric test. Increasing replay volume repeated existing formulations without covering the expression change.

The numeric development cohort contains 20 source groups and uses the if-and-only-if formulation. All four checkpoints scored 40/40 on disabled gates and 16/16 on missing fields there. Development selection checked task aggregates and question negation, but did not combine changes in condition order, gates and field names. A restored development accuracy of 91.48% therefore concealed the regression under the test formulation.

The measured conclusions are that training amplified an existing wording weakness and development checks missed it. Replay balance, limited expression diversity and incomplete condition coverage are the training factors to test next. Their individual contributions have not yet been isolated by controlled training ablations.

## Repair direction

Compare a small repair against the retained baseline. Use paired records that vary gate state, missing fields, operator and question polarity over shared states, with balanced passing and failing outcomes. Include multiple natural formulations and condition orders. Development checks should cover each condition slice alongside retention on other tasks.

Register new development and independent acceptance sources, expression groups and stopping rules before training. The equivalent-wording result is diagnostic evidence. The original test result remains 70.93%. Final selection follows the [release criteria](release-criteria-2026-09-21.md).

## Separate package verification failure

Reloading the merged Phase 3 candidate on all 1,790 test records changed no argmax decisions. The maximum probability difference was 0.00279618, exceeding the registered tolerance of 1e-5, so verification failed. The record is `results/phase3/final-package-verification.log`. Numerical consistency needs separate investigation. SDK and latency checks later in that pipeline did not run.

## Evidence

- Cohort construction, label audit and paired analysis: `prepare`, `rule_metadata` and `analyze` in [numeric_regression.py](../../src/necro/diagnostics/numeric_regression.py).
- Cohort registration: `data/phase3/numeric-regression-v1/experiment.json`.
- Conditional results: `results/phase3/numeric-regression-v1/analysis.json`.
- Paired source-group intervals: `paired-comparisons.json` in the same result directory.
- Data composition and development cases: `data-audit.json` and `development-cases.json` in that directory.
- Requests and checkpoint predictions: the registered data file and each checkpoint's result subdirectory.
