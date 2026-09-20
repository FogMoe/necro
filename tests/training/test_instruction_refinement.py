from collections import defaultdict

from necro.training.data.instruction_refinement import extraction_pairs, reading_pairs
from necro.training.data.phase2_data import reading


def test_reading_polarity_pairs_keep_evidence_and_invert_both_gold_classes():
    original = reading(
        "train",
        [
            {"passage": "A is true.", "question": "Is A true?", "answer": True},
            {"passage": "B is false.", "question": "Is B true?", "answer": False},
        ],
    )
    pairs = reading_pairs(original)
    for i, source in enumerate(original):
        positive, negative = pairs[2 * i : 2 * i + 2]
        assert (
            positive["request"]["state"]
            == negative["request"]["state"]
            == source["request"]["state"]
        )
        assert positive["group_id"] == negative["group_id"] == source["group_id"]
        assert positive["expected"]["decision"] is source["expected"]["decision"]
        assert negative["expected"]["decision"] is not source["expected"]["decision"]
        assert positive["request"]["questions"] != negative["request"]["questions"]


def test_no_match_pair_keeps_candidate_count_and_excludes_gold_alias():
    context = "Ada lives in Rome, Italy, and is thirty years old."
    raw = [
        {
            "id": str(i),
            "context": context,
            "question": question,
            "answers": {"text": [answer], "answer_start": [context.index(answer)]},
        }
        for i, (question, answer) in enumerate(
            [
                ("Who lives there?", "Ada"),
                ("Which city?", "Rome"),
                ("Which country?", "Italy"),
                ("How old?", "thirty"),
            ]
        )
    ]
    grouped = defaultdict(list)
    for row in extraction_pairs(raw, "en", "fixture"):
        grouped[row["request"]["state"]["question"]].append(row)
    assert len(grouped) == 2
    for present, absent in grouped.values():
        a = present["request"]["questions"]["decision"]["criteria"]
        b = absent["request"]["questions"]["decision"]["criteria"]
        assert len(a) == len(b) == 4
        assert a[present["expected"]["decision"]] not in b.values()
        assert b[absent["expected"]["decision"]] == "None of the listed answers is correct"
        assert present["source_keys"] == absent["source_keys"]
