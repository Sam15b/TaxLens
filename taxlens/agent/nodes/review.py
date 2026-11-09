"""Review mode: findings on an uploaded client document. Stub until week 3."""

from typing import Any

from taxlens.agent.state import TaxLensState

NOT_AVAILABLE = "Document review is not available yet."


def run(state: TaxLensState) -> dict[str, Any]:
    return {"answer": NOT_AVAILABLE, "findings": [], "cited": []}
