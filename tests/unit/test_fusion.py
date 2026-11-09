import pytest

from taxlens.retrieval.fusion import RRF_K, rrf_fuse


def test_consensus_beats_a_single_first_place() -> None:
    fused = rrf_fuse(["a", "b", "c"], ["d", "b", "a"])
    ids = [i for i, _ in fused]
    assert ids == ["a", "b", "d", "c"]  # 1st + 3rd and 2nd + 2nd beat d's lone 1st place


def test_scores_follow_the_formula() -> None:
    fused = dict(rrf_fuse(["x", "y"], ["y"]))
    assert fused["x"] == pytest.approx(1 / (RRF_K + 1))
    assert fused["y"] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))


def test_single_ranking_keeps_its_order_and_ties_are_stable() -> None:
    assert [i for i, _ in rrf_fuse(["c", "a", "b"])] == ["c", "a", "b"]
    assert [i for i, _ in rrf_fuse(["a"], ["b"])] == ["a", "b"]  # equal scores: first seen first


def test_empty_input_and_duplicates() -> None:
    assert rrf_fuse() == []
    assert rrf_fuse([], []) == []
    assert rrf_fuse(["a", "a"]) == [("a", pytest.approx(1 / (RRF_K + 1)))]


def test_smaller_k_rewards_top_ranks_more() -> None:
    assert dict(rrf_fuse(["a"], k=1))["a"] == pytest.approx(0.5)
