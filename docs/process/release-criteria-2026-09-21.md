# Release criteria, 2026-09-21

Release selection requires retention of established capabilities, resolution of critical defects, reliable runtime behavior and limited returns from further lightweight training. Jev remains a measured reference in the report. Earlier Jev percentage thresholds remain part of their corresponding historical experiment records.

The [numeric-rule regression](numeric-regression-2026-09-21.md) reopened release selection after Phase 3. Frozen weights, calibration and predictions remain preserved. Aggregate development gains do not offset a clear task regression.

## Acceptance checks

- Compare the candidate with the retained baseline on identical requests, broken down by task and language. Investigate material regressions or retain the previous weights.
- Check each confirmed critical defect across different formulations, missing fields, negation and boundary conditions.
- Verify merged weights, the real API, the official SDK, input limits and probability output. Preserve failed checks and their investigation records.
- Establish capability stability before using controlled lightweight trials to assess diminishing returns. A failed or regressing trial alone does not establish that further training has little value.

Register each recipe and selection rule before training. Development results select candidates. Independent results measure generalization. The exposed Phase 3 cohort is available for diagnosis and regression, and subsequent acceptance uses newly registered independent sources. Model cards and final reports must describe the distributed weights and calibration.
