import math

import pytest

from taxlens.config import settings
from taxlens.llm.embeddings import embed_query, embed_texts

pytestmark = pytest.mark.slow  # downloads the ONNX model (~65 MB) on first use


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_vectors_are_384_dim_unit_and_deterministic() -> None:
    texts = ["Section 194J: fees for professional or technical services.", "Section 80C deduction."]
    first, again = embed_texts(texts), embed_texts(texts)
    assert len(first) == 2
    assert all(len(v) == settings.embedding_dim == 384 for v in first)
    assert all(math.isclose(math.sqrt(_dot(v, v)), 1.0, abs_tol=1e-6) for v in first)
    assert first == again


def test_query_is_closest_to_the_matching_passage() -> None:
    fees, insurance = embed_texts(
        ["Tax is deducted at source on fees for professional services.", "Life insurance premia."]
    )
    query = embed_query("TDS on professional fees")
    assert math.isclose(math.sqrt(_dot(query, query)), 1.0, abs_tol=1e-6)
    assert _dot(query, fees) > _dot(query, insurance)


def test_no_texts_needs_no_model() -> None:
    assert embed_texts([]) == []
