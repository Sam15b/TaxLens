"""The calc node with a fake extractor: no LLM, real calculator."""

from decimal import Decimal
from typing import Any

import pytest

from taxlens.agent.nodes import calc
from taxlens.agent.nodes.calc import CalcExtraction, inr


class FakeExtractor:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> Any:
        self.prompts.append(prompt)
        return self.result


def run(monkeypatch: pytest.MonkeyPatch, extracted: Any, question: str = "q") -> dict[str, Any]:
    fake = FakeExtractor(extracted)
    monkeypatch.setattr(calc, "get_extractor", lambda: fake)
    out = calc.run({"question": question, "masked_question": question})
    assert question in fake.prompts[0] and "Do not compute any tax" in fake.prompts[0]
    return out


def test_salary_14_lakh_new_regime(monkeypatch: pytest.MonkeyPatch) -> None:
    out = run(
        monkeypatch,
        CalcExtraction(gross_income=Decimal(1_400_000)),
        "Salary ₹14 lakh, new regime, FY 2025-26: how much tax?",
    )
    calc_result = out["calculation"]
    assert calc_result["total_tax"] == "81900" and calc_result["taxable_income"] == "1325000"
    assert calc_result["rules_source"] == "Finance Act, 2025" and len(calc_result["breakdown"]) == 4
    answer = out["answer"]
    assert "**₹81,900**" in answer and "₹13,25,000" in answer and "Finance Act, 2025" in answer
    assert "₹12,00,000 to ₹13,25,000 at 15%: ₹18,750" in answer


def test_dict_output_and_old_regime_deductions(monkeypatch: pytest.MonkeyPatch) -> None:
    extracted = {"gross_income": "1000000", "regime": "old", "deductions": {"80C": "200000"}}
    out = run(monkeypatch, extracted)
    assert out["calculation"]["deductions"] == "150000" and out["calculation"]["regime"] == "old"
    assert "Deductions allowed: ₹1,50,000" in out["answer"]


def test_missing_income_asks_instead_of_guessing(monkeypatch: pytest.MonkeyPatch) -> None:
    out = run(monkeypatch, CalcExtraction(clarification="Is ₹1.2 lakh your monthly salary?"))
    assert out == {"answer": "Is ₹1.2 lakh your monthly salary?", "calculation": None}
    out = run(monkeypatch, CalcExtraction())
    assert out == {"answer": calc.DEFAULT_CLARIFICATION, "calculation": None}


def test_unsupported_year(monkeypatch: pytest.MonkeyPatch) -> None:
    out = run(monkeypatch, CalcExtraction(gross_income=Decimal(900_000), tax_year="2019-20"))
    assert out["calculation"] is None and "2025-26" in out["answer"]


@pytest.mark.parametrize(
    ("amount", "text"),
    [(0, "₹0"), (999, "₹999"), (1000, "₹1,000"), (124800, "₹1,24,800"), (12345678, "₹1,23,45,678")],
)
def test_indian_digit_grouping(amount: int, text: str) -> None:
    assert inr(Decimal(amount)) == text
    assert inr(Decimal(-amount)) == (f"₹-{text[1:]}" if amount else text)
