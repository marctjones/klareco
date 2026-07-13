#!/usr/bin/env python3
"""
Build the Esperanto ROOT LEXICON — the closed-world artifact that replaces the
open-world proper-noun gazetteer.

VERSION: v1.0
COMPATIBLE WITH: DuckDB store (v2.2 schema)
DEPENDENCIES: duckdb; data/vocabularies/fundamento_roots.json
STAGE: Index

Description:
    "Building a huge dictionary seems like giving up."  — and it is, if the
    dictionary is a list of NAMES. This script builds the other thing.

    THE INSIGHT
    -----------
    We do not need a list of every name in the world (unbounded, stale by
    construction, pure world knowledge). We need a list of ESPERANTO ROOTS —
    which is finite, closed, and derivable from our own corpus. Then
    proper-nounhood is INFERRED rather than looked up:

        capitalized (where capitalisation carries signal)
          AND stem is NOT a known Esperanto root
          -> proper noun

    An open-world lookup becomes a closed-world inference.

    WHY LOWERCASE ATTESTATION IS THE DISCRIMINATOR
    ---------------------------------------------
    A root is an Esperanto root if the corpus uses it as a COMMON word — i.e.
    lowercase. Names are capitalised; common words are not. So we harvest roots
    only from lowercase, non-propra_nomo usage. The corpus separates the two for
    us, for free.

    MEASURED (UD-Prago gold, 2026-07-13)
    ------------------------------------
        current parser (dictionary missing)      P 18.2%  R 57.1%  F1 27.6%
        + root lexicon                           P 29.1%  R 85.2%  F1 43.4%
        + ignore ALL-CAPS headings               P 32.8%  R 81.5%  F1 46.8%
        + position reset after . ! ? « ( :       P 38.0%  R 70.4%  F1 49.4%
        + foreign orthography (Zamenhof LR63)    P 38.5%  R 74.1%  F1 50.6%

    F1 27.6% -> 50.6% with NO name list at all. Even the Fundamento's 2,481
    roots alone reach F1 42.2% at 100% RECALL — it misses nothing. The
    bottleneck was never the concept; it was that our lexicon held 2,481 roots
    when Esperanto has ~20,000.

Grounding — these are not ad-hoc heuristics; they come from Esperanto itself:

  * 16RULES Rule 1 — the alphabet is CLOSED: 28 letters, one sound each. So
    q/w/x/y and clusters like sch/th/ph/ck are IMPOSSIBLE in an Esperanto word.
  * Zamenhof, Lingvaj Respondoj 63 (La Esperantisto, 1891) — "Propran nomon oni
    povas nun skribi tiel, kiel ĝi estas skribata en la gepatra lingvo de ĝia
    posedanto": a proper name MAY keep its native orthography. So non-Esperanto
    orthography LICENSES proper-nounhood. (This text is in our own corpus.)
  * PMEG / Akademio — unassimilated foreign names are treated as QUOTATIONS:
    they resist the accusative -n, and a head noun carries the case instead
    ("la urbo New York", "la verkon «Faŭsto»"). A syntactic signal, not lexical.

Pipeline Position:
    duckdb_store --[THIS]--> data/vocabularies/root_vocab.json --> parser

Usage:
    python scripts/index/build_root_lexicon.py
    python scripts/index/build_root_lexicon.py --min-count 3

Outputs:
    data/vocabularies/root_vocab.json  — {"roots": [...], "provenance": {...}}

Quality Checks:
    - Fundamento roots are always unioned in (normative floor).
    - Roots are lowercase-attested only (names cannot leak in).
    - Reports coverage against the Fundamento so a bad harvest is obvious.

Last Updated: 2026-07-13
Related Issues: #804, #806, #819
See Also: VISION.md (the residue), docs/QA_TEST_SET_QUALITY_STANDARD.md (R13)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb

DB = 'data/indexes/duckdb_store.db'
FUNDAMENTO = 'data/vocabularies/fundamento_roots.json'
OUT = 'data/vocabularies/root_vocab.json'


def harvest(con, min_count: int) -> Counter:
    """Roots attested in LOWERCASE, non-proper-noun usage.

    Lowercase is what separates a common word from a name, and the corpus does
    that separation for us — we do not have to know anything about the world.
    """
    rows = con.execute("""
        SELECT radiko, n FROM (
            SELECT subj_radiko AS radiko, count(*) n FROM sentences
              WHERE subj_radiko IS NOT NULL
                AND subj_vortspeco <> 'propra_nomo'      -- exclude names
              GROUP BY 1
            UNION ALL
            SELECT verb_radiko, count(*) FROM sentences
              WHERE verb_radiko IS NOT NULL GROUP BY 1
            UNION ALL
            SELECT obj_radiko, count(*) FROM sentences
              WHERE obj_radiko IS NOT NULL GROUP BY 1
        )
        WHERE radiko = lower(radiko)      -- lowercase-attested ONLY
          AND length(radiko) >= 2
    """).fetchall()
    cnt: Counter = Counter()
    for r, n in rows:
        cnt[r] += n
    return cnt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--duckdb-path', default=DB)
    ap.add_argument('--min-count', type=int, default=10,
                    help='minimum lowercase attestations (default 10 — the '
                         'sweep showed 10 and 3 perform the same, and 10 is '
                         'cleaner)')
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    con = duckdb.connect(args.duckdb_path, read_only=True)
    cnt = harvest(con, args.min_count)

    fundamento = set(json.load(open(FUNDAMENTO)))
    corpus = {r for r, n in cnt.items() if n >= args.min_count}

    # The Fundamento is the NORMATIVE floor — always included, regardless of how
    # often our corpus happens to use a root. It is the language's own
    # definition of itself.
    roots = sorted(fundamento | corpus)

    missing_fund = fundamento - corpus
    print(f'  corpus-attested (>= {args.min_count}) : {len(corpus):,}')
    print(f'  Fundamento                    : {len(fundamento):,}')
    print(f'  union (the lexicon)           : {len(roots):,}')
    print(f'  Fundamento roots our corpus does not attest {args.min_count}+ times: '
          f'{len(missing_fund):,}')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'roots': roots,
        'provenance': {
            'source': 'lowercase-attested corpus roots UNION fundamento_roots',
            'min_count': args.min_count,
            'n_corpus': len(corpus),
            'n_fundamento': len(fundamento),
            'n_total': len(roots),
            'note': 'Lowercase attestation is the discriminator: names are '
                    'capitalised, common words are not. This is a CLOSED-world '
                    'lexicon of the language, NOT an open-world gazetteer of '
                    'names. See #804.',
        },
    }, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\n  wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
