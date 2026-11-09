"""Reciprocal Rank Fusion: merge ranked lists without comparing their scores.

Used to merge the main hybrid query with the section-boost query (docs/01-architecture.md
§ 1.3 B), and by the Qdrant and Oracle backends later. Each list contributes ``1 / (k + rank)``
per id, so an id ranked well in several lists beats one ranked first in only one.
"""

from collections import defaultdict
from collections.abc import Sequence

RRF_K = 60  # the constant from the original RRF paper (Cormack et al., 2009)


def rrf_fuse(*rankings: Sequence[str], k: int = RRF_K) -> list[tuple[str, float]]:
    """Each ranking is a list of ids, best first. Returns ``(id, score)``, best first.

    Ties keep the order in which ids were first seen, so the result is deterministic.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(dict.fromkeys(ranking), start=1):  # a repeat counts once
            scores[item] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
