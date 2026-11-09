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
