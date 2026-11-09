"""Calc route: income-tax computation. Stub until the calculator lands on day 8."""

from typing import Any

from taxlens.agent.state import TaxLensState

NOT_AVAILABLE = "Income-tax computation is not available yet."


def run(state: TaxLensState) -> dict[str, Any]:
    return {"answer": NOT_AVAILABLE, "calculation": None, "cited": []}
