from decimal import Decimal

import pytest
from pydantic import ValidationError

from taxlens.tools.tax_calculator import TaxInput, UnsupportedTaxYearError, load_rules


def test_loads_2025_26_rules() -> None:
    rules = load_rules("2025-26")
    assert rules["source"] == "Finance Act, 2025" and rules["cess_rate"] == 0.04
    assert rules["regimes"]["new"]["rebate"]["max_taxable_income"] == 1_200_000


def test_unsupported_tax_year() -> None:
    with pytest.raises(UnsupportedTaxYearError, match="2019-20.*supported: 2025-26"):
        load_rules("2019-20")


def test_input_defaults_and_validation() -> None:
    inp = TaxInput(gross_income=Decimal(1_400_000))
    assert (inp.regime, inp.tax_year, inp.is_salaried, inp.deductions) == (
        "new",
        "2025-26",
        True,
        {},
    )
    with pytest.raises(ValidationError):
        TaxInput(gross_income=Decimal(-1))
    with pytest.raises(ValidationError):
        TaxInput(gross_income=Decimal(1), tax_year="FY25")


def test_salaried_12_75_lakh_new_regime_pays_nothing() -> None:
    from taxlens.tools.tax_calculator import compute_tax

    r = compute_tax(TaxInput(gross_income=Decimal(1_275_000)))
    assert r.standard_deduction == 75_000 and r.taxable_income == 1_200_000
    assert r.tax_before_rebate == 60_000 and r.rebate == 60_000 and r.total_tax == 0
    assert [(row.lower, row.upper, row.rate, row.tax) for row in r.breakdown] == [
        (0, 400_000, Decimal("0.00"), 0),
        (400_000, 800_000, Decimal("0.05"), 20_000),
        (800_000, 1_200_000, Decimal("0.10"), 40_000),
    ]
    assert r.rules_source == "Finance Act, 2025"
