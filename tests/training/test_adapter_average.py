import json

import pytest

from necro.training.adapter_average import average


def test_average_keeps_rank_and_records_both_parents(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file, save_file

    paths = [tmp_path / "left", tmp_path / "right"]
    for i, path in enumerate(paths):
        path.mkdir()
        (path / "adapter_config.json").write_text(
            json.dumps(
                {
                    "r": 2,
                    "lora_alpha": 4,
                    "target_modules": ["x", "y"] if i == 0 else ["y", "x"],
                }
            )
        )
        (path / "necro_adapter.json").write_text(
            json.dumps(
                {
                    "initial_adapter_sha256": "same-parent",
                    "rank": 2,
                    "seed": i,
                    "checkpoint": "fixture",
                    "revision": "fixed",
                    "train_sha256": "same-data",
                    "prompt_sha256": "same-prompt",
                    "learning_rate": 3e-5,
                    "batch_size": 4,
                    "gradient_accumulation": 2,
                    "objective": "answer-ce",
                }
            )
        )
        save_file(
            {
                "layer.lora_A.weight": torch.full((2, 3), float(i)),
                "layer.lora_B.weight": torch.full((4, 2), float(i + 2)),
            },
            str(path / "adapter_model.safetensors"),
        )
    output = tmp_path / "average/adapter"
    result = average(*paths, output, "averaged-fixture")
    weights = load_file(str(output / "adapter_model.safetensors"))
    assert torch.equal(weights["layer.lora_A.weight"], torch.full((2, 3), 0.5))
    assert torch.equal(weights["layer.lora_B.weight"], torch.full((4, 2), 2.5))
    manifest = json.loads((output / "necro_adapter.json").read_text())
    assert manifest["rank"] == 2 and manifest["examples"] == 0
    assert len(result["parents"]) == 2
    changed = json.loads((paths[1] / "necro_adapter.json").read_text())
    changed["initial_adapter_sha256"] = "unrelated-parent"
    (paths[1] / "necro_adapter.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="共同起点"):
        average(*paths, tmp_path / "invalid", "invalid")
