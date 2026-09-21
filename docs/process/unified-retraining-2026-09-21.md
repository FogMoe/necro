# Unified retraining preparation

Status: data and a portable Linux GPU runner are prepared. No unified training, cloud upload or server rental has started. The official post-trained `Qwen/Qwen3.5-0.8B` is the selected initialization; a true Base comparison is deferred. This record does not claim the condition defect is fixed.

## Design

The two [local condition-balance trials](condition-balance-repair-2026-09-21.md) failed their repair checks. Their partial gains support imbalance as a contributing factor, not a complete explanation. The historical adapter chain also makes later task proportions and accumulated exposures difficult to reason about. The next experiment replaces that chain with one multi-task recipe and a fresh rank-16 LoRA.

Historical training requests are normalized to current task definitions, deduplicated and checked for conflicting labels. Outdated synthetic numeric and uncertainty templates are excluded. All protected evaluation sources are excluded before selection, including cross-language source relations. Presentation IDs are unique while original IDs and source keys remain recorded.

The frozen training set contains 11,390 unique requests across eight task families. Numeric conditions contribute 2,400 requests: 800 eligible, 800 comparison failures and 800 prerequisite failures. Complete records therefore have equal comparison outcomes. Languages and question polarities are paired, with disabled gates, missing fields, negative values and fractional boundaries represented. All loss weights are one. The design balances these causes rather than treating balanced Yes/No labels as evidence of semantic balance.

The other seven families retain 8,990 source-protected requests. Task counts are intentionally not all equal; the eight-family macro and per-family retention checks prevent the largest family from deciding acceptance. Candidate retrieval remains comparatively small, and its uncertainty must be reported.

The registry at `data/phase6/unified-v1/experiment.json` freezes training, development, calibration and test hashes. Its audit records 3,049,459 unpadded training tokens per epoch, a maximum length of 1,394 tokens, source exclusions and language/task coverage. Development has 3,784 records; calibration has 1,068. The sealed test combines 948 fresh nonnumeric requests and the still-unopened 4,800-question condition stress cohort. Test sources exclude all historical training, including recipes not selected for the new mixture. Intent test coverage is 56/60 classes; BoolQ and SciFact are English-only. Correlated translations and counterfactuals are source groups, not independent trials.

## Cloud procedure

Build from the prepared workspace:

```powershell
uv run --extra training python -m necro.training.cloud bundle
```

The output is `artifacts/necro-unified-cloud-2026-09-21.zip`. It includes frozen data, project source and lockfile, reference adapters and calibration, checksums, `cloud-plan.json`, and `CLOUD_RUN.md` with Linux commands. Public base weights download on the host. Original raw source caches are unnecessary for training the supplied data.

The runner profiles checkpointed microbatches before choosing a size at effective batch 16. The primary recipe uses two epochs and learning rate 1e-4. One lower-rate comparison at 5e-5 is registered; it can start only after primary assessment. Both start fresh, use seed 2026 and reject a changed base revision. This is an initial bounded comparison, not a claim that those hyperparameters are optimal. Base training and further repair chains are outside this run budget.

The host recomputes official, instruction-parent and Phase 4 reference results on the same exposed cohorts. Candidate calibration uses the frozen calibration partition. Assessment writes condition slices, paired source-group uncertainty, language alerts and machine-readable acceptance checks. It requires all 38 known failures corrected, complete/eligible accuracy at least 98% per language on exposed condition regression, and gate/missing accuracy at least 95%. New condition development must retain parent complete/eligible performance within two points and achieve 95% gate/missing accuracy. Negative-value results are reported separately. Original task families must remain within two points of both references, with macro accuracy within half a point of the Phase 4 reference and no material language regression.

A passing development assessment only makes a candidate eligible for freezing. Independent test evaluation, merged export/reload and real SDK/HTTP checks remain required before completion. If both recipes fail, stop and review the model/data approach. If one passes and the second offers no material gain, freeze the stronger result instead of expanding the search. The package verifies hashes before each action and does not open the independent test automatically.
