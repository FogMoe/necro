# False-rejection regression: repair handoff, 2026-09-21

The latest condition repair rejects eligible records when all required fields exist, the gate is enabled and the numerical condition holds. The [pre-release report](../reports/pre-release-2026-09-21.md#remaining-release-blocker) is the canonical measurement record, including the direct-parent comparison, uncertainty and semantic sampling counts. Stable publication remains blocked by this regression.

## Reproduction evidence

The committed [regression fixture](evidence/condition-false-rejections-2026-09-21.jsonl) contains all 38 observed false rejections, with the original requests, expected answers and both models' recorded responses. Its [metadata](evidence/condition-false-rejections-2026-09-21.json) records model identities, calibration, source hashes and the subset analysis. These are known failures from an exposed cohort, not an independent acceptance test.

One English example is `condition-v1/test/11/gt/0/en/0`:

```json
{
  "measurement_x": 288.5,
  "measurement_y": 288,
  "test_enabled": true
}
```

The instruction requires every field, an enabled gate and `measurement_x > measurement_y`, then asks whether the record is eligible. The expected answer is true. The direct parent returns Noul 0.852202; the candidate returns 0.440983 and therefore selects false. The fixture preserves the full instruction verbatim, including its wording and question polarity.

To replay the fixture with the candidate, run from the project root with the local weights present:

```powershell
@'
import json
from pathlib import Path
from necro.config import Settings
from necro.evaluation import run_evaluation

selected = Path("results/phase4/selected")
selection = json.loads((selected / "selection.json").read_text())
output = Path("results/condition-handoff/candidate")
if output.exists():
    raise ValueError("Choose a new output directory")
settings = Settings(
    adapter=str(selected / "adapter"), device="cuda",
    **{f"{key}_temperature": value for key, value in selection["temperatures"].items()},
)
run_evaluation(
    Path("docs/process/evidence/condition-false-rejections-2026-09-21.jsonl"),
    output, settings,
)
'@ | uv run --extra training python -B -
```

Recorded probabilities come from the original full-cohort inference. The fixture provides a short decision regression check; cohort-level accuracy requires the full dataset. For the direct-parent comparison, use `results/phase3/instruction-seed2026/adapter` with the temperatures in its sibling `calibration-brier-fit.json`.

## Likely contributing data problem

`complete_condition_matrix` in [condition_refinement.py](../../src/necro/training/data/condition_refinement.py) crosses two comparison outcomes with five presence/gate cases: complete, disabled, missing-left, missing-right and missing-gate. Its eligibility rule is `case == "complete" and truth`. Only one of those ten combinations is eligible.

Changing the question polarity balances the answer tokens, but does not change the underlying eligibility of the record. `audit_condition_matrix` checks equal coverage of the Cartesian combinations; it does not check semantic outcome balance. The measured imbalance applies to the 3,200 constructed matrix presentations. The complete training set also contains 1,556 replay presentations.

The sampling issue is a concrete repair hypothesis. It has not been isolated as the sole cause of the behavior. Labels were independently checked by `rule_metadata` in [numeric_regression.py](../../src/necro/diagnostics/numeric_regression.py); changing correct labels is not the repair.

## Bounded repair direction

1. Audit and control both underlying eligibility and answer polarity, separately by language, operator, wording and presence/gate case. Preserve coverage of disabled gates and missing fields while restoring enough genuinely eligible examples.
2. Compare a controlled sampling change from the direct parent. Keep the current candidate as the repaired-gate reference, and the parent as the complete-field reference. Record the data distribution and recipe before training.
3. Include complete-field and eligible-record false-rejection checks in selection. Aggregate numeric accuracy can rise while those slices regress. Preserve the other task families and inspect both languages.
4. Keep this fixture and the exposed 720-question cohort for regression only. Freeze selection before opening newly held-out expressions and source states.
5. Before another training run, measure short-run throughput for micro batch size and checkpointing. Change training performance settings independently of the sampling experiment.

The release standard remains [condition repair, task retention, export/reload/API verification and accurate reporting](release-criteria-2026-09-21.md). A lower Jev percentage gap does not resolve this defect.

## Local artifacts and publication state

| Artifact | Location |
|---|---|
| Frozen candidate and calibration | `results/phase4/selected/` |
| Direct-parent adapter and calibration fit | `results/phase3/instruction-seed2026/` |
| Constructed training data and archived builder | `data/phase4/condition-v3/` |
| Candidate and Jev full-cohort predictions | `results/phase4/final/` |
| Base/parent comparisons and semantic sampling audit | `results/phase4/quick-release-audit/` |

The repository includes the small reproduction fixture and its metadata. Full weights, datasets and raw run directories remain local under Git ignore rules. Preserve them before moving this repair to another machine.

No repair training was performed for this handoff. The current candidate has not completed final merged-export and API verification, and no stable model package or Hugging Face publication was produced. The earlier Phase 3 loader fix remains documented in the [export investigation](numeric-regression-2026-09-21.md).
