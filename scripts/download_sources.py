"""Download the public corpus listed in data/sources.yaml and verify sha256.

    uv run python scripts/download_sources.py                   # download + verify everything
    uv run python scripts/download_sources.py --update-hashes   # record hashes in sources.yaml
    uv run python scripts/download_sources.py --subset ci --dest var/ci-raw

Rules:
- Files that are already present with the recorded hash are skipped, so a second run does nothing.
- Without ``--update-hashes`` a hash mismatch is an error (the file is not overwritten).
- Entries marked ``manual: true`` are never fetched: their hosts block scripted clients. Save
  them by hand as ``<dest>/<id>.pdf``; the hash check still applies.

See docs/03-local-setup.md § 3.7.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import ssl
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
import truststore
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data" / "sources.yaml"
SUBSETS = ROOT / "eval" / "subsets.yaml"
DEFAULT_DEST = ROOT / "data" / "raw"

# Identify honestly. Hosts that refuse this client are handled with `manual: true`.
USER_AGENT = "TaxLens-corpus/0.1 (+https://github.com/Sam15b/TaxLens)"
TIMEOUT = httpx.Timeout(60.0, connect=15.0)
RETRIES = 3
BACKOFF_SECONDS = 2.0
CHUNK_BYTES = 1 << 16

Status = Literal["skipped", "downloaded", "hashed", "manual", "mismatch", "failed"]


@dataclass(frozen=True)
class Result:
    id: str
    status: Status
    sha256: str | None = None
    detail: str = ""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK_BYTES), b""):
            h.update(block)
    return h.hexdigest()


def load_sources(path: Path = SOURCES) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    ids = [s["id"] for s in data]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"duplicate ids in {path}")
    return data


def subset_ids(name: str, path: Path = SUBSETS) -> set[str]:
    """Source ids of a named subset: ``eval/subsets.yaml`` → ``sources: {<name>: [...]}``."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    ids = (data.get("sources") or {}).get(name)
    if not ids:
        raise SystemExit(f"no source subset {name!r} in {path} (want sources: {{{name}: [...]}})")
    return set(ids)


def _retryable(e: httpx.HTTPError) -> bool:
    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        return code >= 500 or code == 429  # 403/404 won't fix themselves: don't hammer the host
    return isinstance(e, httpx.TransportError)


def fetch(
    client: httpx.Client,
    url: str,
    target: Path,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Stream ``url`` to ``target`` atomically, retrying transient failures with backoff."""
    part = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, RETRIES + 1):
        try:
            with client.stream("GET", url) as r:
                r.raise_for_status()
                with part.open("wb") as f:
                    for block in r.iter_bytes(CHUNK_BYTES):
                        f.write(block)
            with part.open("rb") as f:
                if f.read(5) != b"%PDF-":
                    raise ValueError(f"not a PDF (got {r.headers.get('content-type')!r})")
            part.replace(target)
            return
        except httpx.HTTPError as e:
            if attempt == RETRIES or not _retryable(e):
                raise
        finally:
            part.unlink(missing_ok=True)  # no-op after a successful replace()
        sleep(BACKOFF_SECONDS * 2 ** (attempt - 1))


def process(
    source: dict[str, Any],
    dest: Path,
    client: httpx.Client,
    update_hashes: bool,
    sleep: Callable[[float], None] = time.sleep,
) -> Result:
    sid = source["id"]
    expected: str | None = source.get("sha256")
    target = dest / f"{sid}.pdf"

    if target.exists():
        actual = sha256_of(target)
        if expected is None or update_hashes:
            return Result(sid, "hashed" if actual != expected else "skipped", actual)
        if actual == expected:
            return Result(sid, "skipped", actual)
        if source.get("manual"):
            return Result(sid, "mismatch", actual, f"expected {expected[:12]}…, got {actual[:12]}…")
        # A partial or stale file: fetch again and verify below.

    if source.get("manual"):
        return Result(sid, "manual", expected, f"save by hand from {source['url']} as {target}")

    try:
        fetch(client, source["url"], target, sleep=sleep)
    except (httpx.HTTPError, ValueError) as e:
        return Result(sid, "failed", expected, str(e))

    actual = sha256_of(target)
    if expected is not None and actual != expected and not update_hashes:
        target.unlink()
        return Result(sid, "mismatch", actual, f"expected {expected[:12]}…, got {actual[:12]}…")
    return Result(sid, "downloaded", actual)


_ID_LINE = re.compile(r"^- id:\s*(\S+)\s*$")
_SHA_LINE = re.compile(r"^(\s+sha256:\s*)(\S+)\s*$")


def write_hashes(path: Path, hashes: dict[str, str]) -> int:
    """Rewrite ``sha256:`` values in place, keeping comments and layout. Returns lines changed."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    current: str | None = None
    changed = 0
    for i, line in enumerate(lines):
        if m := _ID_LINE.match(line):
            current = m[1]
        elif (m := _SHA_LINE.match(line)) and current in hashes and m[2] != hashes[current]:
            lines[i] = f"{m[1]}{hashes[current]}\n"
            changed += 1
    if changed:
        path.write_text("".join(lines), encoding="utf-8", newline="")  # keep LF on Windows
    return changed


def run(
    sources: Iterable[dict[str, Any]],
    dest: Path,
    update_hashes: bool,
    client: httpx.Client,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Result]:
    dest.mkdir(parents=True, exist_ok=True)
    results = []
    for s in sources:
        r = process(s, dest, client, update_hashes, sleep=sleep)
        print(f"{r.status:<10} {r.id}  {r.detail}".rstrip())
        results.append(r)
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--update-hashes", action="store_true", help="write sha256 values back")
    p.add_argument("--subset", help="only the ids under sources.<name> in eval/subsets.yaml")
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="download folder")
    p.add_argument("--sources", type=Path, default=SOURCES, help=argparse.SUPPRESS)
    a = p.parse_args(argv)

    sources = load_sources(a.sources)
    if a.subset:
        wanted = subset_ids(a.subset)
        unknown = wanted - {s["id"] for s in sources}
        if unknown:
            raise SystemExit(f"subset {a.subset!r} names unknown ids: {sorted(unknown)}")
        sources = [s for s in sources if s["id"] in wanted]

    headers = {"User-Agent": USER_AGENT, "Accept": "application/pdf"}
    # Verify TLS against the OS trust store: some government hosts send an incomplete chain that
    # browsers/OSes complete but certifi alone rejects. Verification stays on.
    tls = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    client = httpx.Client(headers=headers, timeout=TIMEOUT, follow_redirects=True, verify=tls)
    with client:
        results = run(sources, a.dest, a.update_hashes, client)

    if a.update_hashes:
        hashes = {r.id: r.sha256 for r in results if r.sha256 and r.status != "mismatch"}
        print(f"updated {write_hashes(a.sources, hashes)} hash(es) in {a.sources}")

    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    print(", ".join(f"{n} {k}" for k, n in sorted(counts.items())))
    if counts.get("manual"):
        print("manual entries: their hosts block scripted downloads, so save those PDFs by hand.")
    return 1 if counts.get("mismatch") or counts.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
