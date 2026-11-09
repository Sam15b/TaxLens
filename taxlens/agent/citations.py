"""Check that an answer cites only real sources, and cites them everywhere.

The graph retries an answer that fails this check, and falls back to NOT_FOUND when it keeps
failing, so the LLM can't skip citations (docs/01-architecture.md § 1.4).
"""

import re

from taxlens.agent.prompts import NOT_FOUND

CITE_RE = re.compile(r"\[S(\d+)\]")
MIN_PARAGRAPH_CHARS = 40  # shorter lines (headings, "In short:") need no citation of their own


def is_not_found(answer: str) -> bool:
    return answer.strip().strip('"').strip() == NOT_FOUND


def validate(answer: str, n_hits: int) -> tuple[bool, list[int]]:
    """``(valid, cited source numbers)``. NOT_FOUND is valid with no citations.

    Invalid: no citations at all, a number outside 1..n_hits, or a substantial paragraph
    without a citation.
    """
    if is_not_found(answer):
        return True, []
    refs = [int(x) for x in CITE_RE.findall(answer)]
    if not refs or any(r < 1 or r > n_hits for r in refs):
        return False, []
    paragraphs = [p for p in re.split(r"\n\s*\n", answer) if len(p.strip()) > MIN_PARAGRAPH_CHARS]
    if any(not CITE_RE.search(p) for p in paragraphs):
        return False, []
    return True, sorted(set(refs))
