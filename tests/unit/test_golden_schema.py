"""The golden set validates (this is how CI runs the validator), and the checks catch mistakes."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from eval.schema import ExpectedSource, GoldenItem, normalise
from eval.validate_golden import check, validate

ROW: dict[str, Any] = {
    "id": "q001",
    "category": "act_section",
    "difficulty": "easy",
    "question": "What is the maximum deduction under section 80C?",
    "expected_answer": "Rs 1,50,000 in aggregate.",
    "must_include": ["1,50,000"],
    "expected_sources": [{"act": "Income-tax Act, 1961", "section": "80C"}],
    "answerable": True,
}


def test_the_committed_golden_set_is_valid() -> None:
    items, problems = validate()
    assert problems == []
    assert items


def test_a_good_row() -> None:
    item = GoldenItem.model_validate(ROW)
    assert item.expected_sources[0].as_filter() == {"act": "Income-tax Act, 1961", "section": "80C"}


@pytest.mark.parametrize(
    "change",
    [
        {"id": "Q1"},
        {"category": "trivia"},
        {"expected_answer": None},
        {"expected_sources": []},
        {"must_include": ["2,00,000"]},  # not in the expected answer
        {"answerable": False},  # but it has an answer
        {"category": "out_of_scope"},
        {"category": "calc"},  # no expected_total_tax
        {"surprise": 1},
    ],
)
def test_bad_rows(change: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        GoldenItem.model_validate({**ROW, **change})


def test_unanswerable_and_calc_rows() -> None:
    GoldenItem.model_validate(
        {**ROW, "category": "out_of_scope", "answerable": False, "expected_answer": None,
         "must_include": [], "expected_sources": []}
    )  # fmt: skip
    GoldenItem.model_validate(
        {**ROW, "category": "calc", "expected_sources": [], "must_include": ["81,900"],
         "expected_answer": "Rs 81,900", "expected_total_tax": 81900}
    )  # fmt: skip


@pytest.mark.parametrize(
    "source",
    [{}, {"section": "80C"}, {"act": "A", "circular_no": "3/2025"}, {"act": "A", "x": 1}],
)
def test_bad_sources(source: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        ExpectedSource.model_validate(source)


def test_must_include_ignores_rupee_signs_commas_and_case() -> None:
    assert (
        normalise("₹1,50,000 Per Year") == normalise("Rs. 150000 per   year") == "150000 per year"
    )


def test_cross_row_checks() -> None:
    a = GoldenItem.model_validate(ROW)
    b = GoldenItem.model_validate(
        {
            **ROW,
            "expected_sources": [{"circular_no": "99/2030"}, {"act": "Some Act", "section": "1"}],
        }
    )
    corpus = {"act": {"Income-tax Act, 1961"}, "circular_no": {"3/2025"}, "notification_no": set()}
    problems = check([a, b], corpus, {"ci": ["q001", "q404"], "sources": {"ci": ["x"]}})
    assert "q001: duplicate id (2 rows)" in problems
    assert "q001: circular_no '99/2030' is not in data/sources.yaml" in problems
    assert "q001: act 'Some Act' is not in data/sources.yaml" in problems
    assert "subset ci: unknown ids ['q404']" in problems


def test_validate_reports_bad_json(tmp_path: Path) -> None:
    golden = tmp_path / "g.jsonl"
    golden.write_text(json.dumps(ROW) + "\n{not json\n", encoding="utf-8")
    subsets = tmp_path / "s.yaml"
    subsets.write_text("ci: [q001]\n", encoding="utf-8")
    items, problems = validate(golden=golden, subsets=subsets)
    assert len(items) == 1 and len(problems) == 1 and problems[0].startswith("line 2:")
