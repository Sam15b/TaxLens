"""Validate the golden set: every row, unique ids, sources that exist, subsets that resolve.

    uv run python -m eval.validate_golden

Prints the row count per category. Exits 1 with one line per problem otherwise. CI runs the same
checks as a unit test (tests/unit/test_golden_schema.py).
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from eval.schema import GoldenItem

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "eval" / "golden_set.jsonl"
SUBSETS = ROOT / "eval" / "subsets.yaml"
SOURCES = ROOT / "data" / "sources.yaml"


def load_golden(path: Path = GOLDEN) -> tuple[list[GoldenItem], list[str]]:
    items, problems = [], []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            items.append(GoldenItem.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            problems.append(f"line {n}: {exc}".replace("\n", " "))
    return items, problems


def corpus_index(path: Path = SOURCES) -> dict[str, set[str]]:
    """What the corpus can be cited as: Act names, circular numbers, notification numbers."""
    entries: list[dict[str, Any]] = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return {
        "act": {e["act"] for e in entries if e.get("act")},
        "circular_no": {e["circular_no"] for e in entries if e.get("circular_no")},
        "notification_no": {e["notification_no"] for e in entries if e.get("notification_no")},
    }


def check(
    items: list[GoldenItem], corpus: dict[str, set[str]], subsets: dict[str, Any]
) -> list[str]:
    problems = []
    ids = Counter(i.id for i in items)
    problems += [f"{i}: duplicate id ({n} rows)" for i, n in ids.items() if n > 1]
    for item in items:
        for src in item.expected_sources:
            for key, value in src.as_filter().items():
                if key in corpus and value not in corpus[key]:
                    problems.append(f"{item.id}: {key} {value!r} is not in data/sources.yaml")
    for name, subset in subsets.items():
        if name == "sources":
            continue
        unknown = sorted(set(subset or []) - set(ids))
        if unknown:
            problems.append(f"subset {name}: unknown ids {unknown}")
    return problems


def check_subset_sources(
    items: list[GoldenItem], subsets: dict[str, Any], entries: list[dict[str, Any]]
) -> list[str]:
    """Each named subset's corpus (``sources: {name: [...]}``) must download without a manual
    step, and every answerable question in the subset must cite at least one source in it."""
    problems = []
    by_id = {e["id"]: e for e in entries}
    by_item = {i.id: i for i in items}
    for name, ids in (subsets.get("sources") or {}).items():
        docs = [by_id[i] for i in ids or [] if i in by_id]
        problems += [f"sources.{name}: unknown id {i!r}" for i in ids or [] if i not in by_id]
        problems += [
            f"sources.{name}: {d['id']} needs a manual download" for d in docs if d.get("manual")
        ]
        have = corpus_index_of(docs)
        for qid in subsets.get(name) or []:
            item = by_item.get(qid)
            if item is None or not item.expected_sources:
                continue
            if not any(
                all(v in have.get(k, {v}) for k, v in s.as_filter().items() if k != "section")
                for s in item.expected_sources
            ):
                problems.append(f"subset {name}: {qid} cites no source in sources.{name}")
    return problems


def corpus_index_of(entries: list[dict[str, Any]]) -> dict[str, set[str]]:
    return {
        "act": {e["act"] for e in entries if e.get("act") and e.get("doc_type") == "act"},
        "circular_no": {e["circular_no"] for e in entries if e.get("circular_no")},
        "notification_no": {e["notification_no"] for e in entries if e.get("notification_no")},
    }


def validate(
    golden: Path = GOLDEN, sources: Path = SOURCES, subsets: Path = SUBSETS
) -> tuple[list[GoldenItem], list[str]]:
    items, problems = load_golden(golden)
    subset_map = yaml.safe_load(subsets.read_text(encoding="utf-8")) or {}
    entries = yaml.safe_load(sources.read_text(encoding="utf-8")) or []
    problems += check(items, corpus_index(sources), subset_map)
    return items, problems + check_subset_sources(items, subset_map, entries)


def main() -> int:
    items, problems = validate()
    for p in problems:
        print(p, file=sys.stderr)
    if problems:
        return 1
    by_category = Counter(i.category for i in items)
    print(f"{len(items)} valid rows across {len(by_category)} categories")
    for category, n in sorted(by_category.items()):
        print(f"  {category:18} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
