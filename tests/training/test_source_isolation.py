from necro.training.data.source_isolation import Sources


def test_shared_original_sentence_excludes_translation_with_different_pair_id():
    sources = Sources(
        [],
        english_paws={
            ("train", "1"): {"sentence1": "A acts on B.", "sentence2": "B receives action from A."},
            ("test", "2"): {"sentence1": "A acts on B.", "sentence2": "B acts on A."},
        },
    )

    def row(identifier, group):
        return {"id": identifier, "group_id": group, "source": "google-research-datasets/paws-x"}

    training = row("paws/train/en/1", "pair1")
    heldout = row("paws/test/zh/2", "pair2")
    clean, removed = sources.exclude([heldout], sources.keys(training))
    assert clean == []
    assert removed == ["pair2"]


def test_candidate_document_components_are_transitive():
    sources = Sources([], english_paws={})
    rows = [
        {
            "id": str(i),
            "group_id": str(i),
            "source": "BeIR/scifact",
            "request": {"questions": {"decision": {"criteria": dict.fromkeys(keys, "doc")}}},
        }
        for i, keys in enumerate((("a", "b"), ("b", "c"), ("c", "d"), ("e", "f")))
    ]
    result = sources.regroup(rows)
    assert len({row["group_id"] for row in result[:3]}) == 1
    assert result[3]["group_id"] != result[0]["group_id"]
    assert [row["source_group_id"] for row in result] == ["0", "1", "2", "3"]


def test_regrouping_keeps_original_identity_for_future_source_audits():
    rows = [{"id": "r", "group_id": "original-passage", "source": "google/boolq"}]
    sources = Sources(rows, english_paws={})
    regrouped = sources.regroup(rows)
    assert sources.regroup(regrouped) == regrouped
    assert sources.all_keys(regrouped) == sources.all_keys(rows)
