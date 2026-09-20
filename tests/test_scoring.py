from types import SimpleNamespace

import pytest

from necro.backend import (
    TransformersScorer,
    numeric_labels,
    single_token_labels,
    verify_answer_boundary,
)
from necro.config import Settings


@pytest.mark.parametrize("count", [1, 26, 27, 60, 90, 91, 100, 255])
def test_numeric_labels_are_unique_equal_length_without_leading_zero(count):
    labels = numeric_labels(count)
    assert len(set(labels)) == count
    assert len({len(label) for label in labels}) == 1
    assert all(label[0] != "0" for label in labels)


@pytest.mark.parametrize("count", [0, 256])
def test_numeric_labels_reject_invalid_counts(count):
    with pytest.raises(ValueError):
        numeric_labels(count)


def test_labels_reject_special_tokens_and_never_extend_past_z():
    class Tokenizer:
        all_special_ids = [ord("A")]

        def encode(self, text, **kwargs):
            return [ord(letter) for letter in text]

    labels, ids = single_token_labels(Tokenizer(), 2)
    assert labels == ["B", "C"]
    assert ids == [ord("B"), ord("C")]
    with pytest.raises(RuntimeError):
        single_token_labels(Tokenizer(), 26)


def test_sequence_scores_whole_path_and_reuses_prefix():
    torch = pytest.importorskip("torch")

    class Cache:
        def __init__(self):
            self.paths = [()]
            self.reorders = []

        def reorder_cache(self, parents):
            self.reorders.append(parents.tolist())
            self.paths = [self.paths[i] for i in parents.tolist()]

    class Model:
        def __init__(self):
            self.calls = []
            self.cache = Cache()

        def __call__(self, input_ids, past_key_values=None, **kwargs):
            self.calls.append(input_ids.tolist())
            if past_key_values is None:
                distributions = [{1: 0.9, 2: 0.1}]
            else:
                self.cache.paths = [
                    (*prefix, token[0])
                    for prefix, token in zip(self.cache.paths, input_ids.tolist(), strict=True)
                ]
                distributions = [
                    {0: 0.01, 1: 0.01, 9: 0.98} if path == (1,) else {0: 0.95, 9: 0.05}
                    for path in self.cache.paths
                ]
            logits = torch.full((len(distributions), 1, 10), -100.0)
            for i, distribution in enumerate(distributions):
                for token, probability in distribution.items():
                    logits[i, 0, token] = torch.tensor(probability).log()
            return SimpleNamespace(logits=logits, past_key_values=self.cache)

    scorer = TransformersScorer(Settings())
    scorer.device = torch.device("cpu")
    scorer.digit_ids = {str(i): i for i in range(10)}
    scorer.model = Model()
    result = scorer._score_sequence([7, 8], ["10", "11", "20"])
    # 首 token 1 更可能，但其大部分概率去了不合法的 19；完整路径应选择 20。
    assert result.probabilities == pytest.approx([0.009 / 0.113, 0.009 / 0.113, 0.095 / 0.113])
    assert result.candidate_mass == pytest.approx(0.113)
    assert result.input_tokens == 4
    assert scorer.model.calls == [[[7, 8]], [[1], [2]]]
    assert scorer.model.cache.reorders == [[0, 0]]


def test_answer_boundary_rejects_a_token_that_merges_with_prompt():
    class Tokenizer:
        def encode(self, text, **kwargs):
            return {"Answer: ": [1, 2], "A": [3], "Answer: A": [1, 4]}[text]

    with pytest.raises(RuntimeError, match="答案边界"):
        verify_answer_boundary(Tokenizer(), "Answer: ", {"A": 3})
