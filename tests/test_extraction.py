import pytest

from visiontune.evaluation.extraction import extract_answer, extract_explanation

CHOICES = ["Protein synthesis", "Energy production", "Photosynthesis", "Cell division"]


@pytest.mark.parametrize("text,index,method", [
    ("Answer: B. Energy production\nExplanation: ATP.", 1, "strict"),
    ("answer: (C)", 2, "strict"),
    ("Answer: a cell makes energy production happen", 1, "text"),   # lower-case 'a' is a word
    ("**Answer:** D", 3, "strict"),
    ("B. Energy production", 1, "lenient"),
    ("The correct answer is A because ribosomes...", 0, "lenient"),
    ("(C)", 2, "lenient"),
    ("Mitochondria are responsible for energy production.", 1, "text"),
    ("I am not sure.", None, "none"),
    ("Answer: Because cells need energy", None, "none"),   # 'B' of 'Because' is not a letter answer
])
def test_extract_answer(text, index, method):
    ext = extract_answer(text, CHOICES)
    assert (ext.index, ext.method) == (index, method)


def test_letter_out_of_range_is_rejected():
    assert extract_answer("Answer: E", CHOICES).index is None


def test_ambiguous_option_text_is_not_guessed():
    assert extract_answer("It could be photosynthesis or cell division.", CHOICES).index is None


def test_explanation_text_does_not_leak_into_answer():
    out = "The answer: unsure\nExplanation: unlike photosynthesis, this is energy production."
    assert extract_answer(out, CHOICES).index is None


def test_substring_options_prefer_longest():
    choices = ["heat", "heat and light"]
    assert extract_answer("It gives off heat and light.", choices).index == 1


def test_extract_explanation():
    assert extract_explanation("Answer: B\nExplanation:  Cells make\nATP.") == "Cells make ATP."
