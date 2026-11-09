"""One real call to the judge model. Runs only in the eval job: pytest -m eval."""

import pytest

from eval import evaluators as ev
from taxlens.config import settings

KEYS = {"gemini": settings.google_api_key, "groq": settings.groq_api_key, "ollama": "local"}

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(KEYS.get(settings.judge_provider) is None, reason="judge API key not set"),
]


def test_judge_scores_a_correct_and_a_wrong_answer() -> None:
    ref = {
        "answerable": True,
        "question": "What is the maximum deduction under section 80C of the Income-tax Act, 1961?",
        "expected_answer": "Rs 1,50,000 in aggregate.",
    }
    good = ev.answer_correctness({"answer": "Up to ₹1,50,000 a year under section 80C [S1]."}, ref)
    bad = ev.answer_correctness({"answer": "Up to ₹5,00,000 under section 80C [S1]."}, ref)
    assert good["score"] == 1.0, good
    assert bad["score"] == 0.0, bad
