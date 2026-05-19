#!/usr/bin/env python3
"""
Derive a notable-people name list from the local Esperanto Wikipedia dump.

VERSION: v2.x (DuckDB era)
COMPATIBLE WITH: build_synthetic_who_test_set.py (--notable consumer)
DEPENDENCIES: stdlib only (bz2, xml.etree.ElementTree)
STAGE: Acquire

Description:
    Offline alternative to acquire_wikidata_notable_people.py (which
    requires the Wikidata Query Service — currently rate-limited /
    outage-affected). The eo Wikipedia dump on disk is the canonical
    source: every person article carries a biographical category
    ([[Kategorio:Naskiĝintoj…]] / [[Kategorio:Mortintoj…]] /
    [[Kategorio:Vivantaj personoj]]). Stream the bz2 XML, keep titles
    whose wikitext matches, write a flat name list in the same JSON
    format the generator already consumes. Fully deterministic and
    reproducible from the dump; no network.

Pipeline Position:
    data/raw/eo/wikipedia/eo_wikipedia.xml.bz2
       → [THIS SCRIPT] → data/eo_wikipedia_notable_people.json
       → build_synthetic_who_test_set.py --notable ...

Usage:
    python scripts/acquire/acquire_eo_wikipedia_notable_people.py

Inputs:
    data/raw/eo/wikipedia/eo_wikipedia.xml.bz2 (348 MB compressed)

Outputs:
    data/eo_wikipedia_notable_people.json — {"names": [...], "meta": {...}}

Quality Checks:
    - Filters to namespace 0 (main articles, not Category:/Help:/...).
    - Requires a biographical category marker in the page wikitext.
    - Drops titles whose first token isn't uppercase (defensive).
    - Atomic write (.tmp -> rename) so a partial run never corrupts the
      cache.

Last Updated: 2026-05-19
Author: Claude Code (with Marc Jones)
See Also: scripts/eval/build_synthetic_who_test_set.py,
          scripts/acquire/acquire_wikidata_notable_people.py
"""
from __future__ import annotations

import argparse
import bz2
import json
import re
import sys
import time
from pathlib import Path
from xml.etree.ElementTree import iterparse

DEFAULT_SRC = 'data/raw/eo/wikipedia/eo_wikipedia.xml.bz2'
DEFAULT_OUT = 'data/eo_wikipedia_notable_people.json'


def _localname(tag: str) -> str:
    """Strip whatever MediaWiki export-X.YY namespace is present so the
    parser works against any dump version (this dump is 0.11; the previous
    bug was a hardcoded 0.10 prefix that silently matched zero tags)."""
    return tag.rsplit('}', 1)[-1]

# Esperanto-Wikipedia biographical category families. Confirmed against
# a sample of the dump (Naskiĝintoj/Mortintoj come in -en YYYY / -la D-an
# de MONATO / -en la N-a jarcento variants; Vivantaj personoj is the
# living-people catch-all).
BIO_RE = re.compile(
    r'\[\[Kategorio:\s*(?:Naskiĝintoj|Mortintoj|Vivantaj\s+personoj)',
    re.IGNORECASE,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', default=DEFAULT_SRC)
    ap.add_argument('--output', default=DEFAULT_OUT)
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"ERROR: dump not found at {src}", file=sys.stderr)
        return 1
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    pages = bio = kept = 0
    names: list[str] = []
    seen_lower: set[str] = set()

    title: str | None = None
    page_ns: str | None = None
    in_page = False

    print(f"Streaming {src} ...")
    with bz2.open(src, 'rb') as f:
        for ev, el in iterparse(f, events=('end',)):
            tag = _localname(el.tag)
            if tag == 'title':
                title = (el.text or '').strip()
            elif tag == 'ns':
                page_ns = (el.text or '').strip()
            elif tag == 'text':
                text = el.text or ''
                if (page_ns == '0' and title
                        and BIO_RE.search(text)):
                    bio += 1
                    t = title
                    # Defensive: drop disambiguation pages and titles
                    # whose first character is not a letter.
                    if t.endswith(')') and ' (' in t:
                        t = t.rsplit(' (', 1)[0]
                    if t and t[:1].isalpha() and t[:1].isupper():
                        key = t.lower()
                        if key not in seen_lower:
                            seen_lower.add(key)
                            names.append(t)
                            kept += 1
                el.clear()                       # free memory
            elif tag == 'page':
                pages += 1
                title = page_ns = None
                el.clear()
                if pages % 50000 == 0:
                    print(f"  pages={pages:,} bio={bio:,} kept={kept:,} "
                          f"({time.time()-t0:.0f}s)")

    print(f"\nTotal: pages={pages:,} biographical={bio:,} unique={kept:,}")
    tmp = out.with_suffix('.tmp')
    with open(tmp, 'w') as fh:
        json.dump({
            'names': names,
            'meta': {'src': str(src), 'pages': pages, 'biographical': bio,
                     'unique': kept,
                     'fetched': time.strftime('%Y-%m-%d')},
        }, fh, ensure_ascii=False)
    tmp.rename(out)
    print(f"Wrote {kept} unique notable names -> {out}")
    print(f"Total time: {time.time() - t0:.0f}s")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
