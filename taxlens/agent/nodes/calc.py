"""Calc route: the LLM extracts TaxInput, ``compute_tax`` does the arithmetic.

If the income is missing or ambiguous the node asks a clarifying question instead of guessing a
number. The answer names its source (the Finance Act the rules come from) and the full result
goes back as ``calculation`` so clients can show the slab breakdown.
"""

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field

from taxlens.agent.prompts import CALC_PROMPT
from taxlens.agent.state import TaxLensState
from taxlens.tools.tax_calculator import (
    DEFAULT_TAX_YEAR,
    Regime,
    TaxInput,
    TaxResult,
    UnsupportedTaxYearError,
    compute_tax,
)

DEFAULT_CLARIFICATION = (
    "What is your total annual income in rupees, and is it salary or business income?"
)


class CalcExtraction(BaseModel):
    """What the LLM fills in. Like TaxInput, but the income may be missing."""

    gross_income: Decimal | None = Field(default=None, description="Annual income in rupees")
    regime: Regime = "new"
    tax_year: str = DEFAULT_TAX_YEAR
    is_salaried: bool = True
    deductions: dict[str, Decimal] = Field(default_factory=dict)
    clarification: str | None = Field(default=None, description="Question if income is missing")


def get_extractor() -> Any:
    from taxlens.llm.providers import get_structured_model

    return get_structured_model(CalcExtraction)


def inr(amount: Decimal) -> str:
    """Indian digit grouping: 1424800 -> "₹14,24,800"."""
    sign, digits = ("-" if amount < 0 else ""), str(abs(int(amount)))
    head, tail = digits[:-3], digits[-3:]
    groups: list[str] = []
    while len(head) > 2:
        head, groups = head[:-2], [head[-2:], *groups]
    return f"₹{sign}{','.join([g for g in [head, *groups] if g] + [tail])}"


def format_answer(r: TaxResult) -> str:
    lines = [
        f"Estimated income tax for FY {r.tax_year} under the {r.regime} regime: "
        f"**{inr(r.total_tax)}** (including 4% cess).",
        "",
        f"- Gross income: {inr(r.gross_income)}",
    ]
    if r.standard_deduction:
        lines.append(f"- Standard deduction: {inr(r.standard_deduction)}")
    if r.deductions:
        lines.append(f"- Deductions allowed: {inr(r.deductions)}")
    lines.append(f"- Taxable income: {inr(r.taxable_income)}")
    for row in r.breakdown:
        rate = f"{float(row.rate) * 100:g}%"
        lines.append(f"  - {inr(row.lower)} to {inr(row.upper)} at {rate}: {inr(row.tax)}")
    lines.append(f"- Tax before rebate: {inr(r.tax_before_rebate)}")
    if r.rebate:
        lines.append(f"- Rebate under section 87A: {inr(r.rebate)}")
    lines.append(f"- Health and education cess: {inr(r.cess)}")
    lines += [
        "",
        f"Source: slab rates and rebate per the {r.rules_source}, for a resident individual "
        "below 60. Surcharge (income above ₹50 lakh) and special-rate income such as capital "
        "gains are not included.",
    ]
    return "\n".join(lines)


def run(state: TaxLensState) -> dict[str, Any]:
    question = state.get("masked_question") or state["question"]
    raw = get_extractor().invoke(CALC_PROMPT.format(q=question))
    ext = raw if isinstance(raw, CalcExtraction) else CalcExtraction.model_validate(raw)
    if ext.gross_income is None:
        return {"answer": ext.clarification or DEFAULT_CLARIFICATION, "calculation": None}
    try:
        result = compute_tax(TaxInput(**ext.model_dump(exclude={"clarification"})))
    except UnsupportedTaxYearError:
        return {
            "answer": f"I can only compute tax for FY {DEFAULT_TAX_YEAR} so far.",
            "calculation": None,
        }
    return {"answer": format_answer(result), "calculation": result.model_dump(mode="json")}
