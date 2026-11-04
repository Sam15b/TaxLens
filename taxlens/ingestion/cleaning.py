"""Remove headers/footers, fix hyphenation, strip Hindi portions of bilingual notices.

The embedding model is English-only (docs/01-architecture.md § 1.7), so Hindi lines are dropped.
Gazette PDFs carry Hindi in two forms: real Unicode Devanagari, and text typed in a legacy
"Kruti Dev" font that extracts as Latin gibberish (``jftLVªh lañ``). Both are detected.

Everything here is pure string processing so it can be tested on snippets copied from real PDFs.
Use ``clean_pages`` for a whole document (header/footer detection and the hyphenation vocabulary
need every page); ``clean_text`` for a single string.
"""

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence

HEADER_FOOTER_RATIO = 0.6  # a line on more than 60% of page edges is a running header/footer
EDGE_LINES = 3  # only the first/last N non-empty lines of a page can be a header/footer
MIN_PAGES_FOR_HEADERS = 3
SHORT_LINE_WORDS = 5  # digits are ignored when comparing lines this short (page numbers)
DEVANAGARI_RATIO = 0.5  # drop a line when more than half its characters are Devanagari

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
# Legacy-font Hindi. Characters that the Kruti Dev mapping produces but English legal text
# doesn't, plus very common Hindi words as they come out of that font.
_KRUTI_CHARS = re.compile(r"[ª¼½ñ@`]")
_KRUTI_WORDS = frozenset(
    "ds esa dh dk dks gS gSa ls fd ;g bl ij us Hkh tk dj fy, rFkk vkSj ljdkj Hkkjr".split()
)
_KRUTI_MASTHEAD = frozenset(["vlk/kkj.k", "izkf/kdkj", "izdkf'kr", "[k.M", "Hkkx", "lañ"])
_GAZETTE_MARKER = re.compile(r"^x{3}GID[A-Z]x{3}$")  # "xxxGIDHxxx" markers in e-Gazette PDFs

_HYPHENS = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "−": "-"})
_SOFT_HYPHEN_BREAK = re.compile(r"­[ \t]*\n[ \t]*")
# A word broken across a line: letters, a hyphen, the line break, then a lower-case letter.
_HYPHEN_BREAK = re.compile(r"\b([^\W\d_]+)-[ \t]*\n[ \t]*([a-z]\w*)")
_WORDS = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")
_INLINE_SPACE = re.compile(r"[^\S\n]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_DIGITS = re.compile(r"\d+")


# ── Hindi ────────────────────────────────────────────────────────────────────


def is_devanagari_line(line: str) -> bool:
    chars = [c for c in line if not c.isspace()]
    if not chars:
        return False
    return len(_DEVANAGARI.findall(line)) / len(chars) > DEVANAGARI_RATIO


def is_legacy_hindi_line(line: str) -> bool:
    tokens = [t.strip("]A,") for t in line.split()]
    score = len(_KRUTI_CHARS.findall(line))
    score += sum(t in _KRUTI_WORDS for t in tokens)
    score += 2 * sum(t in _KRUTI_MASTHEAD for t in tokens)
    return score >= 2 or bool(_GAZETTE_MARKER.match(line.strip()))


def drop_hindi_lines(text: str) -> str:
    return "\n".join(
        line
        for line in text.split("\n")
        if not (is_devanagari_line(line) or is_legacy_hindi_line(line))
    )


# ── Unicode + whitespace ─────────────────────────────────────────────────────


def normalise_unicode(text: str) -> str:
    """NFKC (NBSP → space, ligatures and maths letters → ASCII), Unicode hyphens → ``-``.

    En/em dashes are kept: notification numbers use them (``12/2025 – Central Tax``).
    """
    text = _SOFT_HYPHEN_BREAK.sub("", text).replace("­", "")
    return unicodedata.normalize("NFKC", text).translate(_HYPHENS)


def collapse_whitespace(text: str) -> str:
    """One space between words, no trailing spaces, at most one blank line in a row."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = (_INLINE_SPACE.sub(" ", line).strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


# ── Running headers and footers ──────────────────────────────────────────────


def _line_key(line: str) -> str:
    """Page numbers change from page to page, so digits don't count in short lines
    (``Page 2 of 3``, ``SEC. 1]``). Longer lines must repeat exactly: otherwise body lines that
    differ only in a number (``... section 57;`` / ``... section 58;``) would look the same."""
    words = line.split()
    key = " ".join(words).lower()
    return _DIGITS.sub("#", key) if len(words) <= SHORT_LINE_WORDS else key


def _edge_keys(text: str) -> set[str]:
    lines = [line for line in text.split("\n") if line.strip()]
    edges = lines[:EDGE_LINES] + lines[-EDGE_LINES:]
    return {_line_key(line) for line in edges}


def find_repeated_lines(pages: Sequence[str]) -> set[str]:
    """Keys of lines repeated at the top/bottom of most pages.

    Odd and even pages are also counted separately, because gazettes alternate their running
    headers (``[PART II—`` on one side, ``SEC. 1]`` on the other).
    """
    if len(pages) < MIN_PAGES_FOR_HEADERS:
        return set()
    edges = [_edge_keys(p) for p in pages]
    repeated: set[str] = set()
    for group in (edges, edges[0::2], edges[1::2]):
        counts = Counter(key for keys in group for key in keys)
        repeated |= {k for k, n in counts.items() if n > HEADER_FOOTER_RATIO * len(group)}
    return {k for k in repeated if _could_be_header(k)}


def _could_be_header(key: str) -> bool:
    """Page numbers, or lines of 2+ tokens. A lone word that repeats at a page edge is usually
    a table cell (``Nil`` closes most pages of the exempt-services notification), not a header."""
    return len(key.split()) >= 2 or not re.search(r"[^\W\d_]", key)


def strip_repeated_lines(text: str, repeated: set[str]) -> str:
    if not repeated:
        return text
    lines = text.split("\n")
    nonblank = [i for i, line in enumerate(lines) if line.strip()]
    edge = set(nonblank[:EDGE_LINES] + nonblank[-EDGE_LINES:])
    return "\n".join(
        line for i, line in enumerate(lines) if i not in edge or _line_key(line) not in repeated
    )


# ── Hyphenation ──────────────────────────────────────────────────────────────


def build_vocabulary(texts: Iterable[str]) -> set[str]:
    """Lower-cased words and hyphenated compounds that occur within a line."""
    return {w.lower() for t in texts for w in _WORDS.findall(t)}


def fix_hyphenation(text: str, vocab: set[str] | None = None) -> str:
    """Rejoin words broken across lines, keeping the hyphen of real compounds.

    Statutes break lines at real compounds far more than mid-word (``sub-\\nclause``,
    ``non-\\nresident``, ``short-\\nterm``), so the hyphen is dropped only when the document
    suggests it is a line-break hyphen:

    1. ``a-b`` occurs elsewhere           → keep the hyphen
    2. ``ab`` occurs elsewhere            → join
    3. ``a`` and ``b`` are both words     → keep (a compound)
    4. otherwise                          → join (``deduc-\\ntion``)
    """
    vocab = build_vocabulary([text]) if vocab is None else vocab

    def join(m: re.Match[str]) -> str:
        a, b = m[1], m[2]
        hyphenated, joined = f"{a}-{b}".lower(), f"{a}{b}".lower()
        if hyphenated in vocab:
            return f"{a}-{b}"
        if joined in vocab:
            return f"{a}{b}"
        if a.lower() in vocab and b.lower() in vocab:
            return f"{a}-{b}"
        return f"{a}{b}"

    return _HYPHEN_BREAK.sub(join, text)


# ── Entry points ─────────────────────────────────────────────────────────────


def clean_pages(pages: Sequence[str]) -> list[str]:
    """Clean every page of one document. Returns one string per input page (maybe empty)."""
    texts = [normalise_unicode(drop_hindi_lines(p)) for p in pages]
    repeated = find_repeated_lines(texts)
    texts = [strip_repeated_lines(t, repeated) for t in texts]
    vocab = build_vocabulary(texts)
    return [collapse_whitespace(fix_hyphenation(t, vocab)) for t in texts]


def clean_text(text: str) -> str:
    """Clean one string on its own (no header/footer detection: that needs several pages)."""
    return collapse_whitespace(fix_hyphenation(normalise_unicode(drop_hindi_lines(text))))
