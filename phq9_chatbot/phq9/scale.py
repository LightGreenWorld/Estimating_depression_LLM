"""The PHQ-9 instrument itself: items, scoring range and severity bands.

Kroenke K, Spitzer RL, Williams JB. The PHQ-9. J Gen Intern Med 2001. https://pmc.ncbi.nlm.nih.gov/articles/PMC1495268/
Nothing here talks to a model; it is pure clinical knowledge, so it is the easiest file to test.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class SeverityBand:
    low: int
    high: int
    label: str


class PHQ9Scale:
    ITEMS = (
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Trouble falling or staying asleep, or sleeping too much",
        "Feeling tired or having little energy",
        "Poor appetite or overeating",
        "Feeling bad about yourself, or that you are a failure, or have let yourself or your family down",
        "Trouble concentrating on things, such as reading the newspaper or watching television",
        "Moving or speaking so slowly that other people could have noticed, or being fidgety or restless "
        "that you have been moving around a lot more than usual",
        "Thoughts that you would be better off dead, or of hurting yourself in some way",
    )
    BANDS = (
        SeverityBand(0, 4, "None to Minimal"),
        SeverityBand(5, 9, "Mild"),
        SeverityBand(10, 14, "Moderate"),
        SeverityBand(15, 19, "Moderately severe"),
        SeverityBand(20, 27, "Severe"),
    )
    N_ITEMS = 9
    MAX_ITEM_SCORE = 3
    MAX_TOTAL = N_ITEMS * MAX_ITEM_SCORE          # 27
    SELF_HARM_ITEM = 9                             # item 9 is handled specially in the report

    @classmethod
    def severity(cls, total: int) -> str:
        for band in cls.BANDS:
            if band.low <= total <= band.high:
                return band.label
        return "Unknown"

    @classmethod
    def validate_scores(cls, scores: list) -> list[int]:
        """Return a clean list of 9 ints in 0..3, or raise ValueError."""
        if len(scores) != cls.N_ITEMS:
            raise ValueError(f"expected {cls.N_ITEMS} scores, got {len(scores)}")
        clean = []
        for s in scores:
            s = int(s)
            if not 0 <= s <= cls.MAX_ITEM_SCORE:
                raise ValueError(f"item score {s} outside 0..{cls.MAX_ITEM_SCORE}")
            clean.append(s)
        return clean
