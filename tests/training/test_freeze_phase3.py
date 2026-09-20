import json

import pytest

from necro.training.release.freeze_phase3 import reject_exposed_tests


def test_even_partial_final_predictions_prevent_blind_freeze(tmp_path):
    run = tmp_path / "partial-run"
    run.mkdir()
    (run / "summary.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "dataset_sha256": "sealed-hash",
                    "full_dataset_evaluated": False,
                    "evaluated_examples": 1,
                }
            }
        )
    )
    with pytest.raises(ValueError, match="选择冻结前的最终预测"):
        reject_exposed_tests({"sealed-hash"}, tmp_path)


def test_development_predictions_do_not_block_final_freeze(tmp_path):
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "dataset_sha256": "development-hash",
                    "full_dataset_evaluated": True,
                }
            }
        )
    )
    reject_exposed_tests({"sealed-hash"}, tmp_path)
