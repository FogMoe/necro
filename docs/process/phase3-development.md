# Phase 3 development record, 2026-09-20 to 2026-09-21

This phase corrects prefix sampling, block sampling, and class coverage defects. This page records comparisons on a development set isolated by source. The data rationale is in the [coverage audit](phase3-coverage-audit.md).

## Coverage correction

Starting from the frozen Phase 2 operator-contrast weights, one epoch of answer-token CE used 4,076 records, rank 16, learning rate 3e-5, batch 4, accumulation 2, and seed 2026. Training took 1,417.85 seconds, with peak memory of 3.50 GiB. Disabling the adapter produced zero maximum difference on the base-model probe. Positive intent examples covered all 60 classes in both English and Chinese across training, development, and calibration.

On the same 1,464 development records:

| Task | Phase 2 weights | Coverage correction | Jev 1.13.0 |
|---|---:|---:|---:|
| inference | 74.44% | 77.50% | 84.17% |
| intent | 74.15% | 79.83% | 85.80% |
| paraphrase | 78.17% | 80.99% | 80.99% |
| reading_boolean | 82.17% | 84.50% | 89.92% |
| numeric_rule | 91.48% | 89.20% | 100.00% |
| ordinal_rule | 99.05% | 98.57% | 100.00% |
| candidate_extraction | 88.33% | 88.33% | 100.00% |
| candidate_retrieval | 100.00% | 100.00% | 100.00% |
| Task macro average | 85.97% | 87.37% | 92.61% |

Macro accuracy improved by 1.39 percentage points, with a source-group paired bootstrap 95% interval of -0.05 to +2.81. Intent improved by 5.68 percentage points, with an interval of +1.42 to +10.23, supporting the coverage correction. Numeric rules regressed by 2.27 percentage points, requiring additional capability retention measures.

Both models answered the same development questions. Gains were calculated from paired same-question results.

## Calibration comparison

Calibration used only a separate set of 1,068 records, averaging by source group and then weighting tasks equally to fit one temperature per judgment type. The grid remained `[0.5, 8]`. A preregistered comparison tested NLL and Brier objectives:

| Fitting objective | Phase 2 development macro Brier | Coverage-corrected development macro Brier | Mean of both |
|---|---:|---:|---:|
| NLL | 0.213797 | 0.188415 | 0.201106 |
| Brier | 0.213520 | 0.188504 | 0.201012 |

Following the comparison plan, subsequent runs used Brier fitting because its mean Brier was slightly lower. The difference was only 0.000094, effectively a tie rather than a material improvement. Fitting preserves argmax and the number of inference calls. Jev's development macro Brier was 0.109576, leaving a substantial probability-quality gap.

The coverage-corrected model's temperatures were Choice 1.430646, Noul 2.378414, and Score 0.5. Exact values are stored in the fitting output. Score reached the predefined lower bound. The boundary flag was retained, with no further search for lower temperatures based on development results. Neither model's development or calibration responses contained Noul probabilities saturated at exactly 0 or 1.

## Development perturbations and follow-up hypotheses

The 140 predefined development perturbations cover candidate reversal, question negation, and irrelevant long text. Statistics use their pairing with the original questions.

A key weakness was negation around nested reading questions. When asked whether the original question's correct answer was no according to the passage, the Phase 2 weights achieved 10% consistency on 10 pairs, the coverage-corrected model 0%, and Jev 100%. Adding ordinary reading examples had not made the model reliably follow the outer instruction. Targeted paired training was used to investigate this issue.

Instruction refinement used 3,254 records in `data/phase3/instruction-v2`, retaining rank 16 and the same major training hyperparameters:

- 668 reading judgments from 300 new passage sources, with paired supervision for affirmative and negative answer propositions using the same evidence.
- 800 candidate-extraction records from 100 new SQuAD English passages and 100 new CMRC 2018 Chinese passages, with at most two questions per passage. The correct answer was either present or replaced by a distractor from the same passage, keeping the candidate count fixed.
- Replay of 970 numeric and 120 ordinal rule records, targeting the observed numeric regression.
- 300 new NLI records, 240 replay records covering all intent classes, and paraphrase and candidate-retrieval replay.

Labels came from human annotations, semantically explicit transformations, and the existing rule oracle. Jev was used only for evaluation and supplied no training labels. All original development, calibration, independent-test, and perturbation sets stayed unchanged. Pretraining checks corrected the ambiguity of `no response`, which can mean an absence of response. v1 received no gradient training. v2 used the explicit wording `the answer 'no'`, retaining selected sources and order.

The preregistered promotion rules required repair of the key instruction defect, reading-negation consistency of at least 70%, a macro decline no greater than 0.5 percentage points, and no core-task decline greater than 3 percentage points. A qualifying candidate would receive one replication with the same recipe and a different seed.

## Instruction refinement and final replication

Both candidates started from coverage-seed2026 and used the same instruction-v2 data, learning rate 3e-5, batch 4, and accumulation 2, with seeds 2026 and 2027. Each run performed 407 optimizer updates. Both produced zero base-model probe difference with the adapter disabled.

| Development metric | Coverage correction | Instruction seed2026 | Instruction seed2027 |
|---|---:|---:|---:|
| Task-macro accuracy | 87.37% | 87.98% | 86.47% |
| Task-macro Brier | 0.188504 | 0.181986 | 0.196695 |
| Reading-negation consistency, 10 pairs | 0% | 100% | 80% |
| Reading-negation accuracy, 10 questions | 20% | 90% | 70% |

seed2026 gained 0.61 percentage points in macro accuracy, with a source-group paired 95% interval of -0.94 to +2.14. The largest task decline was 0.78 percentage points on ordinary reading. It repaired the key negation defect and met the preregistered promotion rules. seed2027 did not reproduce the overall gain. Its macro accuracy, Brier, and reading-negation results were all weaker than seed2026.

The final selection retained instruction-seed2026 and ended lightweight training exploration. Overall gains were small and varied by seed. Final testing retained the original acceptance thresholds. Selection used only development and development-perturbation results. None of the four candidates' raw development or calibration Noul probabilities were saturated at exactly 0 or 1.

## Training time

| Experiment | Records | Recorded wall-clock minutes | Peak training GiB |
|---|---:|---:|---:|
| coverage-seed2026 | 4,076 | 23.63 | 3.496 |
| instruction-seed2026 | 3,254 | 410.30 | 3.577 |
| instruction-seed2027 | 3,254 | 16.88 | 3.609 |

seed2026's wall-clock time includes host sleep or standby. The interval from steps 151 to 152 was 1,805.24 seconds, and from steps 152 to 153 was 22,143.69 seconds. The remaining optimizer-step intervals shorter than 60 seconds totaled 11.13 minutes. The original long intervals are retained. The sum of short intervals describes observable execution time, distinct from GPU kernel time. The detailed audit is in `timing-audit.json` in the corresponding run directory.

## Selection freeze

The frozen-weight SHA-256 is `35421e4cf65a5320ba36159d9160d6bf608b25df5700c071ece3336aad8c8f13`. `results/phase3/selected/` retains the selection rationale, each model's calibration, complete training lineage, source-overlap checks, and source snapshots. Independent-test predictions run after freezing.

## Evidence

Local records in `results/phase3/` include `coverage-plan.json`, `calibration-objective-plan.json`, `calibration-objective-decision.json`, `instruction-plan.json`, `final-replication-plan.json`, `final-decision.json`, `final-seed-comparison.json`, each model's development and calibration predictions, and `development-robustness-comparison.json`.
