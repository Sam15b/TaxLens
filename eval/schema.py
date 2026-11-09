"""The golden-set row format (docs/06-evaluation.md § 6.1).

Each row of ``eval/golden_set.jsonl`` is one ``GoldenItem``. Rows are checked on their own here,
and against each other and the corpus in ``eval/validate_golden.py``.
"""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

GOLDEN_VERSION = "v1"

Category = Literal[
    "act_section",
    "act_mapping",
    "circular",
    "gst_notification",
    "calc",
    "multi_hop",
    "out_of_scope",
]
Difficulty = Literal["easy", "medium", "hard"]


class ExpectedSource(BaseModel):
    """One source a good answer should cite: an Act section, a circular or a notification."""

    model_config = ConfigDict(extra="forbid")

    act: str | None = None
    section: str | None = None
    circular_no: str | None = None
    notification_no: str | None = None

    @model_validator(mode="after")
    def _one_kind(self) -> "ExpectedSource":
        kinds = [bool(self.act), bool(self.circular_no), bool(self.notification_no)]
        if sum(kinds) != 1:
            raise ValueError("give exactly one of act, circular_no or notification_no")
        if self.section and not self.act:
            raise ValueError("section needs act")
        return self

    def as_filter(self) -> dict[str, str]:
        """The non-empty fields, as matched against hit/citation metadata."""
        return {k: v for k, v in self.model_dump().items() if v}


def normalise(text: str) -> str:
    """For ``must_include`` checks: case, ₹/Rs., commas and spacing don't matter."""
    text = re.sub(r"\brs\b\.?", " ", text.lower().replace("₹", " "))  # not the "rs" in "years"
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    return re.sub(r"\s+", " ", text).strip()


class GoldenItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^q\d{3}$")
    category: Category
    difficulty: Difficulty
    question: str = Field(min_length=10)
    expected_answer: str | None
    must_include: list[str] = Field(default_factory=list)
    expected_sources: list[ExpectedSource] = Field(default_factory=list)
    answerable: bool
    expected_total_tax: int | None = None  # calc questions: the exact total tax in rupees
    notes: str | None = None  # where the answer is: page, wording, caveats

    @model_validator(mode="after")
    def _consistent(self) -> "GoldenItem":
        if not self.answerable:
            if self.expected_answer or self.must_include or self.expected_sources:
                raise ValueError("unanswerable rows have no answer, must_include or sources")
            return self
        if not self.expected_answer:
            raise ValueError("answerable rows need expected_answer")
        if self.category == "calc":
            if self.expected_total_tax is None:
                raise ValueError("calc rows need expected_total_tax")
        elif not self.expected_sources:
            raise ValueError("answerable rows need expected_sources")
        if self.category == "out_of_scope":
            raise ValueError("out_of_scope rows must be unanswerable")
        answer = normalise(self.expected_answer)
        missing = [m for m in self.must_include if normalise(m) not in answer]
        if missing:
            raise ValueError(f"must_include not in expected_answer: {missing}")
        return self
