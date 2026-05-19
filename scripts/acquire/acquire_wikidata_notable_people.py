#!/usr/bin/env python3
"""
Acquire a Wikidata notable-people name list (for trivia prioritisation).

VERSION: v2.x (DuckDB era)
COMPATIBLE WITH: build_synthetic_who_test_set.py (--notable consumer)
DEPENDENCIES: SPARQLWrapper (already in requirements.txt)
STAGE: Acquire

Description:
    One-time network fetch. Queries the Wikidata Query Service for
    "notable" humans — instance-of human (Q5) with many Wikipedia
    sitelinks — and writes a flat set of their labels (incl. the
    Esperanto label when present) to a local JSON cache. This is a
    PRIORITISATION oracle only: the trivia generator stays correct and
    fully Pure-Esperanto / corpus-grounded without it; the cache merely
    biases candidate selection toward famous, well-attested creators so
    the residual weak/obscure pairs shrink.

Pipeline Position:
    Wikidata SPARQL → [THIS SCRIPT] → data/wikidata_notable_people.json
                    → build_synthetic_who_test_set.py --notable ...

Usage:
    python scripts/acquire/acquire_wikidata_notable_people.py
    python scripts/acquire/acquire_wikidata_notable_people.py \
        --min-sitelinks 40 --limit 60000

Inputs:
    Wikidata Query Service (network).

Outputs:
    data/wikidata_notable_people.json — {"names": [...], "meta": {...}}

Quality Checks:
    - Skips empty/single-char labels.
    - De-duplicates case-insensitively; keeps original-case form.
    - Atomic write (.tmp then rename) so a partial fetch never corrupts
      an existing cache.

Last Updated: 2026-05-19
Author: Claude Code (with Marc Jones)
See Also: scripts/eval/build_synthetic_who_test_set.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

OUT = Path('data/wikidata_notable_people.json')

# Humans (Q5) ranked by Wikipedia sitelink count — a robust, language
# -agnostic notability proxy. Pull the English + Esperanto labels.
SPARQL = """
SELECT ?personLabel ?personEoLabel WHERE {{
  ?person wdt:P31 wd:Q5 ;
          wikibase:sitelinks ?sl .
  FILTER(?sl >= {min_sitelinks})
  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "en" .
    ?person rdfs:label ?personLabel .
  }}
  OPTIONAL {{ ?person rdfs:label ?personEoLabel .
             FILTER(LANG(?personEoLabel) = "eo") }}
}}
LIMIT {limit}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--min-sitelinks', type=int, default=40,
                    help='Notability threshold (Wikipedia sitelink count)')
    ap.add_argument('--limit', type=int, default=80000)
    ap.add_argument('--output', default=str(OUT))
    args = ap.parse_args()

    try:
        from SPARQLWrapper import SPARQLWrapper, JSON
    except ImportError:
        print("SPARQLWrapper not installed (pip install SPARQLWrapper)",
              file=sys.stderr)
        return 1

    sparql = SPARQLWrapper(
        "https://query.wikidata.org/sparql",
        agent="KlarecoTriviaBuilder/1.0 (research; contact via repo)")
    sparql.setQuery(SPARQL.format(min_sitelinks=args.min_sitelinks,
                                  limit=args.limit))
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(600)

    print(f"Querying Wikidata: humans with >= {args.min_sitelinks} "
          f"sitelinks, limit {args.limit} ...")
    t0 = time.time()
    res = sparql.query().convert()
    rows = res["results"]["bindings"]
    print(f"  {len(rows)} rows in {time.time() - t0:.0f}s")

    seen, names = set(), []
    for r in rows:
        for key in ('personEoLabel', 'personLabel'):
            v = (r.get(key) or {}).get('value', '').strip()
            if len(v) < 2 or v.startswith('Q') and v[1:].isdigit():
                continue
            if v.lower() in seen:
                continue
            seen.add(v.lower())
            names.append(v)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump({'names': names,
                   'meta': {'min_sitelinks': args.min_sitelinks,
                            'rows': len(rows), 'unique': len(names),
                            'fetched': time.strftime('%Y-%m-%d')}},
                  f, ensure_ascii=False)
    tmp.rename(out)
    print(f"Wrote {len(names)} unique notable names -> {out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
