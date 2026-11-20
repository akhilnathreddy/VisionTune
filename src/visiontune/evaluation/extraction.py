"""Turn free-form model output into a predicted option index.

The same procedure is used for every model (blueprint 4.7 / 7.1). It returns the
index and *how* it was found, so format compliance can be reported separately
from accuracy: a base model that answers correctly but ignores the format should
not be scored as wrong, and a fine-tuned model should not get credit just for
following the format.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from visiontune.data.records import LETTERS, normalize_text

STRICT = "strict"      # "Answer: B" as instructed
LENIENT = "lenient"    # a bare letter, "(B)", "The answer is B", ...
TEXT = "text"          # exactly one option's text appears in the answer
NONE = "none"          # could not be parsed

# Keywords are case-insensitive; option letters must be upper case, so that
# "Answer: a cell..." is not read as option A.
_STRICT_RE = re.compile(r"^[ \t*_]*(?i:answer)[ \t*_]*[:：][ \t*_]*\(?([A-J])\)?(?![A-Za-z])", re.MULTILINE)
_LENIENT_RES = [
    re.compile(r"^\s*\(?([A-J])\)?\s*(?:[.:)\-]|$)", re.MULTILINE),
    re.compile(r"\b(?i:answer|option|choice)\s+(?i:is\s+)?\(?([A-J])\)?(?![A-Za-z])"),
    re.compile(r"\b(?i:correct (?:answer|option|choice) is)\s*[:：]?\s*\(?([A-J])\)?(?![A-Za-z])"),
]


@dataclass(frozen=True)
class Extraction:
    index: int | None
    method: str


def _answer_part(text: str) -> str:
    """Text before any 'Explanation:' section, so explanation wording can't leak in."""
    return re.split(r"\n\s*\**explanation\**\s*[:：]", text, maxsplit=1, flags=re.IGNORECASE)[0]


def _letter_to_index(letter: str, n_choices: int) -> int | None:
    idx = LETTERS.index(letter.upper())
    return idx if idx < n_choices else None


def extract_answer(output: str, choices: list[str]) -> Extraction:
    text = output or ""
    n = len(choices)

    m = _STRICT_RE.search(text)
    if m:
        idx = _letter_to_index(m.group(1), n)
        if idx is not None:
            return Extraction(idx, STRICT)

    head = _answer_part(text)
    for pattern in _LENIENT_RES:
        m = pattern.search(head)
        if m:
            idx = _letter_to_index(m.group(1), n)
            if idx is not None:
                return Extraction(idx, LENIENT)

    # Fall back to option text. Prefer the longest match, and require it to be
    # unambiguous: if two different options both appear, give up.
    lowered = normalize_text(head).lower()
    hits = [i for i, c in enumerate(choices) if c and normalize_text(c).lower() in lowered]
    if hits:
        longest = max(hits, key=lambda i: len(choices[i]))
        others = [i for i in hits if i != longest
                  and normalize_text(choices[i]).lower() not in normalize_text(choices[longest]).lower()]
        if not others:
            return Extraction(longest, TEXT)
    return Extraction(None, NONE)


def extract_explanation(output: str) -> str:
    m = re.search(r"\**explanation\**\s*[:：]\s*(.*)", output or "", flags=re.IGNORECASE | re.DOTALL)
    return normalize_text(m.group(1)) if m else ""
