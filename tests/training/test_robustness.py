import pytest

from necro.training.analysis.robustness import negate_instructions


@pytest.mark.parametrize(
    "language, ending",
    [
        ("en", "Is this record eligible?"),
        ("en", "Is this record ineligible?"),
        ("zh", "现在是否合格？"),
        ("zh", "现在是否不合格？"),
    ],
)
def test_numeric_negation_preserves_rule_and_is_reversible(language, ending):
    original = "A fixed rule mentions eligible and ineligible. " + ending
    changed = negate_instructions("numeric_rule", language, original)
    assert changed != original
    assert changed.startswith("A fixed rule mentions eligible and ineligible. ")
    assert negate_instructions("numeric_rule", language, changed) == original


def test_unknown_numeric_wording_is_rejected_instead_of_guessing():
    with pytest.raises(ValueError):
        negate_instructions("numeric_rule", "en", "Something unrelated.")
