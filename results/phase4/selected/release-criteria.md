# Release criteria, 2026-09-21

Release selection requires resolution of the confirmed condition-judgment defects, retention of established capabilities, and verified runtime behavior. Jev remains a measured reference in the report. Earlier Jev percentage thresholds remain part of their corresponding historical experiment records.

The [numeric-rule regression](numeric-regression-2026-09-21.md) reopened release selection after Phase 3. Frozen weights, calibration and predictions remain preserved. Aggregate development gains do not offset a clear task regression.

## Acceptance checks

- Resolve the confirmed condition-judgment defects across the registered formulations, missing fields, negation and boundary cases.
- Compare the candidate with the retained baseline on identical requests, broken down by task and language. Investigate material regressions or retain the previous weights.
- Verify export, merged-weight reload, the real API and official SDK. Preserve failed checks and their investigation records.
- Report the remaining ordinary errors, measurement conditions and supported scope accurately.

Once these checks pass, assemble the stable release. Further training to reduce the percentage gap to Jev is outside the closing scope.

Register each recipe and selection rule before training. Development results select candidates. Independent results measure generalization. The exposed Phase 3 cohort is available for diagnosis and regression, and subsequent acceptance uses newly registered independent sources. Model cards and final reports must describe the distributed weights and calibration.
