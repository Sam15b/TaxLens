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


# (name, gross, regime, salaried, deductions, taxable, total_tax)
CASES = [
    # ── new regime, slab boundaries (not salaried: taxable = gross) ──
    ("zero income", 0, "new", False, {}, 0, 0),
    ("4L: top of nil slab", 400_000, "new", False, {}, 400_000, 0),
    ("8L: rebate covers 20k", 800_000, "new", False, {}, 800_000, 0),
    ("12L exactly: full rebate", 1_200_000, "new", False, {}, 1_200_000, 0),
    ("12L + 1: marginal relief", 1_200_001, "new", False, {}, 1_200_001, 1),
    ("12.1L: tax capped at the 10k excess + cess", 1_210_000, "new", False, {}, 1_210_000, 10_400),
    ("12,70,588: relief about to end", 1_270_588, "new", False, {}, 1_270_588, 73_412),
    ("13L: no relief, full slab tax", 1_300_000, "new", False, {}, 1_300_000, 78_000),
    # official cumulative slab tax (incometax.gov.in) + 4% cess
    ("oracle: 16L", 1_600_000, "new", False, {}, 1_600_000, 124_800),
    ("oracle: 20L", 2_000_000, "new", False, {}, 2_000_000, 208_000),
    ("oracle: 24L", 2_400_000, "new", False, {}, 2_400_000, 312_000),
    ("25L: 30% slab", 2_500_000, "new", False, {}, 2_500_000, 343_200),
    # ── new regime, salaried ──
    ("oracle: salaried 12.75L pays nothing", 1_275_000, "new", True, {}, 1_200_000, 0),
    ("salaried 12,75,001", 1_275_001, "new", True, {}, 1_200_001, 1),
    ("salaried 14L (day-8 demo)", 1_400_000, "new", True, {}, 1_325_000, 81_900),
    ("standard deduction capped at income", 50_000, "new", True, {}, 0, 0),
    ("new regime ignores 80C", 2_000_000, "new", True, {"80C": 150_000}, 1_925_000, 192_400),
    # ── old regime ──
    ("old 2.5L", 250_000, "old", False, {}, 250_000, 0),
    ("old 5L: rebate", 500_000, "old", False, {}, 500_000, 0),
    ("old 5L + 1: no marginal relief", 500_001, "old", False, {}, 500_001, 13_000),
    ("oracle: old 10L", 1_000_000, "old", False, {}, 1_000_000, 117_000),
    ("old 15L", 1_500_000, "old", False, {}, 1_500_000, 273_000),
    ("old salaried 50k standard deduction", 550_000, "old", True, {}, 500_000, 0),
    (
        "old 80C capped at 1.5L, 80D_self at 25k",
        1_000_000, "old", True, {"80C": 200_000, "80D_self": 30_000}, 775_000, 70_200,
    ),
    ("old uncapped deduction counts in full", 600_000, "old", False, {"80G": 100_000}, 500_000, 0),
    ("old negative deduction ignored", 600_000, "old", False, {"80C": -50_000}, 600_000, 33_800),
    ("old deductions above income", 300_000, "old", True, {"80C": 150_000, "80G": 500_000}, 0, 0),
]  # fmt: skip


@pytest.mark.parametrize(
    ("name", "gross", "regime", "salaried", "deductions", "taxable", "total"),
    CASES,
    ids=[c[0] for c in CASES],
)
def test_cases(
    name: str,
    gross: int,
    regime: str,
    salaried: bool,
    deductions: dict[str, int],
    taxable: int,
    total: int,
) -> None:
    from taxlens.tools.tax_calculator import compute_tax

    r = compute_tax(
        TaxInput(
            gross_income=Decimal(gross),
            regime=regime,  # type: ignore[arg-type]
            is_salaried=salaried,
            deductions={k: Decimal(v) for k, v in deductions.items()},
        )
    )
    assert (r.taxable_income, r.total_tax) == (taxable, total), name
    # each part is rounded to the rupee on its own, so the parts add up to within ₹1
    assert abs(r.total_tax - (r.tax_before_rebate - r.rebate + r.cess)) <= 1
    assert abs(sum(row.tax for row in r.breakdown) - r.tax_before_rebate) <= 1


def test_unknown_tax_year_in_input() -> None:
    from taxlens.tools.tax_calculator import compute_tax

    with pytest.raises(UnsupportedTaxYearError):
        compute_tax(TaxInput(gross_income=Decimal(1), tax_year="2030-31"))
