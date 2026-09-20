import json

import pytest

from necro.experiment_guard import digest
from necro.training.release.freeze_candidate import freeze


def test_failed_review_cannot_freeze_release(tmp_path):
    data, run, output = (tmp_path / name for name in ("data", "run", "selected"))
    run.mkdir()
    (run / "repair-assessment.json").write_text(
        json.dumps({"eligible": False, "checks": {"missing_fields": False}})
    )
    with pytest.raises(ValueError, match="did not pass"):
        freeze(data, run, output)
    assert not output.exists()


def test_amendment_must_preserve_failed_checks_and_explain_tradeoff(tmp_path):
    data, run, output = (tmp_path / name for name in ("data", "run", "selected"))
    run.mkdir()
    preliminary = run / "repair-assessment.json"
    preliminary.write_text(json.dumps({"eligible": False, "checks": {"numeric_retention": False}}))
    decision = run / "selection-review.json"
    decision.write_text(
        json.dumps(
            {
                "eligible": True,
                "checks": {"retention": True},
                "supersedes_review_sha256": digest(preliminary),
                "original_checks": {"numeric_retention": False},
            }
        )
    )
    with pytest.raises(ValueError, match="preserve the original checks"):
        freeze(data, run, output, decision)
    assert not output.exists()
