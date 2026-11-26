"""Coralscapes decision rule, strict A/B/C decoding, and asymmetric cost."""

import re
from typing import Mapping


def health_state(live: float, bleached: float) -> str:
    if live > 30.0 and bleached < 10.0:
        return "healthy"
    if live < 10.0 or bleached > 25.0:
        return "degraded"
    return "stressed"

STATES = ("healthy", "stressed", "degraded")


LETTERS = ("A", "B", "C")


def strict_letter(
    raw_output,
) -> str | None:
    """
    Primary parser.

    Valid only when the stripped model output is exactly one letter:
    A, B, or C. Punctuation or explanatory text is invalid.
    """
    text = str(
        raw_output
        if raw_output is not None
        else ""
    ).strip()

    match = re.fullmatch(
        r"[ABCabc]",
        text,
    )

    if match is None:
        return None

    return text.upper()


def state_from_letter(
    row: Mapping,
    letter: str | None,
) -> str | None:
    if letter is None:
        return None

    mapping = {
        "A": str(row["option_A_state"]),
        "B": str(row["option_B_state"]),
        "C": str(row["option_C_state"]),
    }

    return mapping[letter]

