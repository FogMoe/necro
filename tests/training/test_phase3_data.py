import copy

import pytest

from necro.datasets import INTENTS
from necro.training.data.phase2_data import example
from necro.training.data.phase3_data import SourceIndex, coverage, intent_partition, take


def intent(identifier, text, gold, language="en"):
    return example(
        identifier,
        identifier,
        "intent",
        language,
        text,
        {"type": "choice", "instructions": "Select intent", "criteria": INTENTS},
        gold,
        "mteb/amazon_massive_intent",
    )


def test_same_utterance_cannot_cross_partitions_under_different_source_ids():
    a = intent("a", "Play  Music", "play_music")
    b = intent("b", "play music", "play_music")
    index = SourceIndex([], english_paws={})
    with pytest.raises(ValueError, match="独立来源组不足"):
        take([b], index, index.keys(a), 1, 42)


def test_taxonomy_guard_checks_each_language_independently():
    data = [intent(str(i), "text", key) for i, key in enumerate(INTENTS)]
    with pytest.raises(ValueError, match="zh 意图正例缺类"):
        coverage(data, INTENTS)


def test_rare_class_must_have_distinct_train_development_and_calibration_sources():
    data = []
    for key in INTENTS:
        for n in range(3):
            for language in ("en", "zh"):
                row = intent(
                    f"{key}/{n}/{language}", f"{key} example {n} {language}", key, language
                )
                row["group_id"] = f"{key}/{n}"
                data.append(row)
    index = SourceIndex([], english_paws={})
    result = intent_partition(data, index, set())
    assert all(len(rows) == 120 for rows in result.values())
    assert not index.all_keys(result["train"]) & index.all_keys(result["development"])
    # 两个来源无法冒充三个独立划分；也不能用同组的两个语言补配额。
    insufficient = copy.deepcopy(data)
    insufficient = [row for row in insufficient if not row["group_id"].endswith("/2")]
    with pytest.raises(ValueError, match="正例缺类"):
        intent_partition(insufficient, index, set())


def test_question_reuse_is_protected_even_when_passage_changes():
    row = example(
        "a",
        "passage/a",
        "reading_boolean",
        "en",
        {"passage": "Text A", "question": "Is this true?"},
        {"type": "noul", "instructions": "Read the passage"},
        True,
        "google/boolq",
    )
    other = copy.deepcopy(row)
    other["group_id"] = "passage/b"
    other["request"]["state"]["passage"] = "Different text"
    index = SourceIndex([], english_paws={})
    assert index.keys(row) & index.keys(other)
