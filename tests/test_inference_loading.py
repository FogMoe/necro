from types import SimpleNamespace

import torch
import transformers

from necro.backend import TransformersScorer
from necro.config import Settings


def test_direct_model_loading_freezes_inference_parameters(monkeypatch):
    model = torch.nn.Linear(2, 2)
    assert all(p.requires_grad for p in model.parameters())
    tokens = {
        **{chr(65 + i): i + 1 for i in range(26)},
        "Yes": 27,
        "No": 28,
        **{str(i): i + 30 for i in range(10)},
    }
    tokenizer = SimpleNamespace(
        all_special_ids=[0],
        encode=lambda text, **kwargs: [tokens[text]] if text else [],
        apply_chat_template=lambda *args, **kwargs: "",
    )
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(
        transformers.AutoModelForImageTextToText, "from_pretrained", lambda *a, **k: model
    )
    scorer = TransformersScorer(Settings(checkpoint="local-merged-fixture", device="cpu"))
    scorer.load()
    assert not model.training
    assert not any(p.requires_grad for p in model.parameters())
