"""Deterministic income-tax computation: the LLM extracts the inputs, Python does the arithmetic.

Rules are versioned data (``tax_rules/fy<year>.yaml``), so a new Finance Act means a new YAML file
and tests, not new code. Money is ``Decimal`` throughout. Covered: resident individuals below 60,
both regimes, standard deduction, capped old-regime deductions, the 87A rebate (with marginal
relief where the rules say so) and the 4% cess. Not covered yet (stretch issues): surcharge above
₹50 lakh, special-rate income such as capital gains, senior-citizen slabs.
"""

from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

RULES_DIR = Path(__file__).parent / "tax_rules"
DEFAULT_TAX_YEAR = "2025-26"

Regime = Literal["new", "old"]


class UnsupportedTaxYearError(ValueError):
    """There are no rules for this tax year."""


class TaxInput(BaseModel):
    gross_income: Decimal = Field(
        ge=0, description="Total annual income in rupees, before deductions"
    )
    regime: Regime = Field(default="new", description="'new' (section 115BAC, default) or 'old'")
    tax_year: str = Field(
        default=DEFAULT_TAX_YEAR, pattern=r"^\d{4}-\d{2}$", description="e.g. 2025-26"
    )
    is_salaried: bool = Field(
        default=True, description="Salary/pension income gets the standard deduction"
    )
    deductions: dict[str, Decimal] = Field(
        default_factory=dict,
        description='Old regime only, e.g. {"80C": 150000, "80D_self": 25000}',
    )


class SlabRow(BaseModel):
    lower: Decimal
    upper: Decimal
    rate: Decimal
    tax: Decimal


class TaxResult(BaseModel):
    tax_year: str
    regime: Regime
    gross_income: Decimal
    standard_deduction: Decimal
    deductions: Decimal
    taxable_income: Decimal
    tax_before_rebate: Decimal
    rebate: Decimal
    cess: Decimal
    total_tax: Decimal
    rules_source: str
    breakdown: list[SlabRow]


@lru_cache
def load_rules(tax_year: str) -> dict[str, Any]:
    """The rules YAML for ``tax_year`` ("2025-26"). Raises ``UnsupportedTaxYearError``."""
    path = RULES_DIR / f"fy{tax_year.replace('-', '_')}.yaml"
    if not path.is_file():
        supported = sorted(p.stem[2:].replace("_", "-") for p in RULES_DIR.glob("fy*.yaml"))
        raise UnsupportedTaxYearError(
            f"no tax rules for {tax_year}; supported: {', '.join(supported)}"
        )
    rules: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return rules


def _rupees(d: Decimal) -> Decimal:
    return d.quantize(Decimal("1"), rounding=ROUND_HALF_UP)


def _slab_tax(income: Decimal, slabs: list[list[Any]]) -> tuple[Decimal, list[SlabRow]]:
    """Tax on ``income`` slab by slab, and one breakdown row per slab the income reaches."""
    tax, lower, rows = Decimal(0), Decimal(0), []
    for upper, rate in slabs:
        if income <= lower:
            break
        top = income if upper is None else min(income, Decimal(upper))
        part = (top - lower) * Decimal(str(rate))
        rows.append(SlabRow(lower=lower, upper=top, rate=Decimal(str(rate)), tax=_rupees(part)))
        tax += part
        lower = top
    return tax, rows


def _rebate(tax: Decimal, taxable: Decimal, rule: dict[str, Any]) -> Decimal:
    """Section 87A. Up to the threshold: the tax, capped at ``max_rebate``. Just above it, with
    marginal relief: whatever keeps the tax from exceeding the income above the threshold."""
    threshold = Decimal(rule["max_taxable_income"])
    if taxable <= threshold:
        return min(tax, Decimal(rule["max_rebate"]))
    excess = taxable - threshold
    if rule.get("marginal_relief") and tax > excess:
        return tax - excess
    return Decimal(0)


def compute_tax(inp: TaxInput) -> TaxResult:
    rules = load_rules(inp.tax_year)
    r = rules["regimes"][inp.regime]
    standard = Decimal(r["standard_deduction_salaried"]) if inp.is_salaried else Decimal(0)
    standard = min(standard, inp.gross_income)
    allowed = Decimal(0)
    if inp.regime == "old":  # the new regime allows no Chapter VI-A deductions here
        caps = r.get("deduction_caps", {})
        allowed = sum(
            (min(max(amount, Decimal(0)), Decimal(caps.get(name, amount)))
             for name, amount in inp.deductions.items()),
            Decimal(0),
        )  # fmt: skip
    taxable = max(inp.gross_income - standard - allowed, Decimal(0))
    tax, rows = _slab_tax(taxable, r["slabs"])
    rebate = _rebate(tax, taxable, r["rebate"])
    after = tax - rebate
    cess = after * Decimal(str(rules["cess_rate"]))
    return TaxResult(
        tax_year=inp.tax_year,
        regime=inp.regime,
        gross_income=_rupees(inp.gross_income),
        standard_deduction=_rupees(standard),
        deductions=_rupees(allowed),
        taxable_income=_rupees(taxable),
        tax_before_rebate=_rupees(tax),
        rebate=_rupees(rebate),
        cess=_rupees(cess),
        total_tax=_rupees(after + cess),
        rules_source=str(rules["source"]),
        breakdown=rows,
    )
