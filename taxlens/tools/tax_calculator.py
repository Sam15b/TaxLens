"""Deterministic income-tax computation: the LLM extracts the inputs, Python does the arithmetic.

Rules are versioned data (``tax_rules/fy<year>.yaml``), so a new Finance Act means a new YAML file
and tests, not new code. Money is ``Decimal`` throughout. Covered: resident individuals below 60,
both regimes, standard deduction, capped old-regime deductions, the 87A rebate (with marginal
relief where the rules say so) and the 4% cess. Not covered yet (stretch issues): surcharge above
₹50 lakh, special-rate income such as capital gains, senior-citizen slabs.
"""

from decimal import Decimal
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
