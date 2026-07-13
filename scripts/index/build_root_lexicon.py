#!/usr/bin/env python3
"""
Build the Esperanto ROOT LEXICON — ReVo-first, tiered, and no longer laundered.

VERSION: v2.0
COMPATIBLE WITH: v2.2 DuckDB store
DEPENDENCIES: data/raw/eo/dictionaries/revo_roots.json (scripts/acquire/acquire_revo.py);
              data/vocabularies/fundamento_roots.json; duckdb (optional, tier 2)
STAGE: Index

Description:
    v1 of this script harvested roots from `subj_radiko` / `verb_radiko` /
    `obj_radiko` — i.e. FROM PARSER OUTPUT. When the parser was wrong, the
    harvest wrote the error down as evidence and fed it straight back:

        degraded parser splits   organo -> org + an
        harvest records          `org` as a ROOT
        parser reloads,  splits  organo -> org + an       <-- self-reinforcing

    **A corpus-harvested lexicon can never be more correct than the parser that
    harvested it**, and no `--min-count` escapes it: the parser makes the same
    mis-split on EVERY occurrence, so the bad root has a HIGH count. That is
    failure mode F13 in the data pipeline (#806).

    THE FIX IS NOT A SMALLER LEXICON. IT IS AN INDEPENDENT ONE.
    ----------------------------------------------------------
    TIER 1 — ReVo + Fundamento. Curated, external, and nothing of ours in it.
             ReVo's source is one XML article per root and the FILENAME IS THE
             ROOT, so the inventory is a directory listing. It separates every
             contaminated case exactly:

                 organ  YES / org       no        amerik YES / amerikan  no
                 banan  YES / mak       no        (esperant: correctly ABSENT —
                                                   it is LEXICALIZED, not a root)

    TIER 2 — the corpus harvest. KEPT, because ReVo lacks ~7,900 roots the corpus
             attests (neologisms, technical and geographic vocabulary), and
             dropping it COSTS us: measured on UD gold, curated-only scores
             proper-noun F1 44.9% vs 48.9% for the union. Coverage matters.

    SO WHAT STOPS THE CONTAMINATION? — PROTECTION, NOT EXCLUSION.
    ------------------------------------------------------------
    The damage a laundered root does is that it ENABLES A BAD SPLIT. `org` is
    harmless sitting in the lexicon; it is harmful only when it lets `organo`
    become org+an. So we do not have to remove it — we have to make the correct
    reading WIN:

        **ReVo says X is a root => X is ATOMIC => never split X.**

    Every tier-1 root that merely LOOKS decomposable is emitted as PROTECTED
    (2,364 of them: `organ`, `banan`, `milit`, `regul`, `kalkul`, `postul` …).
    `amerikan` and `kristan` are NOT in ReVo, so they are NOT protected, and
    `amerikano` correctly stays amerik+an. This is `protected_roots` — derived
    from a dictionary, not hand-listed.

    Lexicalization (`esperant`) is a DIFFERENT fact — about usage, not about the
    dictionary — and comes from scripts/index/build_surface_lexical_facts.py.
    The parser unions the two.

Pipeline Position:
    ReVo + Fundamento --[THIS]--> root_vocab.json {roots, protected} --> parser
    duckdb (tier 2)   --^

Usage:
    python scripts/acquire/acquire_revo.py        # once, or when ReVo updates
    python scripts/index/build_root_lexicon.py
    python scripts/index/build_root_lexicon.py --no-corpus   # tier 1 only

Inputs:
    - data/raw/eo/dictionaries/revo_roots.json   (REQUIRED — tier 1)
    - data/vocabularies/fundamento_roots.json    (REQUIRED — tier 1, normative)
    - data/indexes/duckdb_store.db               (optional — tier 2 coverage)

Outputs:
    - data/vocabularies/root_vocab.json
      {"roots": [...], "tier1": [...], "protected": [...], "provenance": {...}}

Quality Checks:
    - Asserts the anchors #806 rests on: organ/banan/amerik present in tier 1;
      org/amerikan/mak absent from it. Fails loudly if ReVo ever stops
      separating them.
    - Reports how many tier-2 roots ReVo contradicts (the laundering estimate).

Last Updated: 2026-07-13
Author: Claude (with Marc Jones)
Related Issues: #806, #804, #819, #821
See Also: docs/PROPER_NOUNS.md, scripts/acquire/acquire_revo.py,
          scripts/index/build_surface_lexical_facts.py (lexicalization)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

REVO_TYPED = Path('data/raw/eo/dictionaries/revo_typed_roots.json')   # voko-akrido: COMMON roots only
REVO_NAMES = Path('data/raw/eo/dictionaries/revo_name_roots.json')    # voko-akrido: PROPER-NAME roots
REVO = Path('data/raw/eo/dictionaries/revo_roots.json')               # legacy: filename harvest (CONTAMINATED)
FUNDAMENTO = Path('data/vocabularies/fundamento_roots.json')
DB = 'data/indexes/duckdb_store.db'
OUT = Path('data/vocabularies/root_vocab.json')

# The closed suffix inventory (the 16 rules). Used ONLY to decide which tier-1
# roots merely LOOK decomposable and therefore need protecting.
_SUFFIXES = ('ant', 'int', 'ont', 'at', 'it', 'ot',
             'ism', 'ist', 'ind', 'em', 'ec', 'aĵ', 'ul', 'in', 'et', 'eg',
             'ar', 'er', 'uj', 'ej', 'estr', 'ad', 'aĝ', 'an', 'ig', 'iĝ',
             'il', 'obl', 'op', 'um', 'id')

_MUST_BE_TIER1 = ('organ', 'banan', 'amerik', 'hund', 'milit', 'regul')
_MUST_NOT_BE_TIER1 = ('org', 'amerikan', 'mak')


def harvest_corpus(min_count: int) -> Counter:
    """TIER 2 — lowercase-attested roots from the store.

    ⚠️ This is PARSER OUTPUT and is therefore contaminated. It is kept for
    COVERAGE (ReVo lacks ~7,900 roots the corpus attests), and its contamination
    is neutralised by PROTECTION rather than by exclusion — see the module
    docstring. Do not treat these as authoritative.
    """
    import duckdb
    con = duckdb.connect(DB, read_only=True)
    rows = con.execute("""
        SELECT radiko, sum(n) FROM (
            SELECT subj_radiko AS radiko, count(*) n FROM sentences
              WHERE subj_radiko IS NOT NULL AND subj_vortspeco <> 'propra_nomo'
              GROUP BY 1
            UNION ALL
            SELECT verb_radiko, count(*) FROM sentences
              WHERE verb_radiko IS NOT NULL GROUP BY 1
            UNION ALL
            SELECT obj_radiko, count(*) FROM sentences
              WHERE obj_radiko IS NOT NULL GROUP BY 1
        )
        WHERE radiko = lower(radiko) AND length(radiko) >= 2
        GROUP BY 1
    """).fetchall()
    return Counter({r: int(n) for r, n in rows})


def main() -> int:
    ap = argparse.ArgumentParser(description='Build the ReVo-first root lexicon')
    ap.add_argument('--min-count', type=int, default=10)
    ap.add_argument('--no-corpus', action='store_true',
                    help='tier 1 only (curated). Costs coverage: proper-noun F1 '
                         '44.9%% vs 48.9%% for the union.')
    args = ap.parse_args()

    # voko-akrido separates COMMON roots from PROPER-NAME roots. Prefer it.
    #
    # The legacy acquire_revo.py harvested ReVo by FILENAME, and ~1,835 ReVo
    # articles have a PROPER NOUN as their root (zamenhof.xml, varsovio.xml).
    # They went straight into the common-word lexicon, so `Zamenhof` and
    # `Varsovio` began "decomposing" to known roots and stopped being detectable
    # as names. That was a real bug, and this is the fix.
    names: set[str] = set()
    if REVO_TYPED.exists():
        revo = set(json.loads(REVO_TYPED.read_text(encoding='utf-8'))['roots'])
        if REVO_NAMES.exists():
            names = {n.lower() for n in
                     json.loads(REVO_NAMES.read_text(encoding='utf-8'))['roots']}
        print(f'  voko-akrido: {len(revo):,} COMMON roots, '
              f'{len(names):,} NAME roots held OUT of the lexicon')
    elif REVO.exists():
        revo = set(json.loads(REVO.read_text(encoding='utf-8'))['roots'])
        print('  ⚠️  using the LEGACY filename harvest — it mixes ~1,835 proper-noun '
              'roots into the common lexicon. Run acquire_voko_akrido.py.')
    else:
        raise FileNotFoundError(
            f'No ReVo lexicon: {REVO_TYPED} / {REVO}\n'
            '  Acquire it:  python scripts/acquire/acquire_voko_akrido.py\n'
            'Refusing to build a lexicon out of parser output alone — that is '
            'the laundering loop this script exists to break (#806).')
    fund = set(json.loads(FUNDAMENTO.read_text(encoding='utf-8')).keys())
    # A NAME root is still a ROOT — `amerik` (Ameriko) must be in the lexicon or
    # `amerikano` cannot decompose. What it is NOT is a COMMON WORD. So carry two
    # INDEPENDENT flags rather than one list:
    #
    #   is_root  -> may participate in morphology        (common ∪ names)
    #   is_name  -> the BARE root, with only a grammatical ending, is a PROPER NOUN
    #
    #     Zamenhof   = zamenhof + ø        -> name root, no derivation  -> PROPER NOUN
    #     Varsovio   = varsovi  + o        -> name root, no derivation  -> PROPER NOUN
    #     amerikano  = amerik + AN + o     -> name root + DERIVATION    -> common noun
    #
    # This replaces a pile of capitalisation heuristics with a lexical fact.
    tier1 = revo | fund | names

    missing = [r for r in _MUST_BE_TIER1 if r not in tier1]
    present = [r for r in _MUST_NOT_BE_TIER1 if r in tier1]
    if missing or present:
        raise SystemExit(
            f'TIER-1 FAILED THE ANCHORS that #806 rests on.\n'
            f'  expected but ABSENT : {missing}\n'
            f'  unexpected, PRESENT : {present}')

    tier2: set[str] = set()
    if not args.no_corpus:
        cnt = harvest_corpus(args.min_count)
        tier2 = {r for r, n in cnt.items() if n >= args.min_count} - tier1

    # ReVo says it is a root => it is ATOMIC => never split it.
    protected = sorted(
        r for r in tier1
        if any(r.endswith(s) and len(r) - len(s) >= 2 for s in _SUFFIXES))

    roots = sorted(tier1 | tier2)
    contradicted = sorted(t for t in tier2
                          if any(t.endswith(s) and t[:-len(s)] in tier1
                                 for s in _SUFFIXES))

    print(f'  tier 1  ReVo                : {len(revo):,}')
    print(f'  tier 1  Fundamento          : {len(fund):,}')
    print(f'  tier 1  union (AUTHORITATIVE): {len(tier1):,}')
    print(f'  tier 2  corpus (contaminated): {len(tier2):,}')
    print(f'  LEXICON                     : {len(roots):,}')
    print(f'  PROTECTED (ReVo says atomic): {len(protected):,}')
    print(f'  NAME roots (bare form = a PROPER NOUN): {len(names):,}')
    print(f'\n  tier-2 roots ReVo CONTRADICTS (look like parser mis-splits): '
          f'{len(contradicted):,}')
    print(f'    e.g. {contradicted[:8]}')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix('.tmp')
    tmp.write_text(json.dumps({
        'roots': roots,
        'tier1': sorted(tier1),
        'protected': protected,
        'name_roots': sorted(names),
        'provenance': {
            'tier1': 'ReVo (github.com/revuloj/revo-fonto, GPL-2.0) UNION Fundamento '
                     '— CURATED and INDEPENDENT of our parser',
            'tier2': ('corpus-harvested from parser output — kept for COVERAGE only, '
                      'NOT authoritative' if tier2 else 'excluded (--no-corpus)'),
            'protected': 'tier-1 roots that merely LOOK decomposable. ReVo says they '
                         'are roots, so they are ATOMIC and must never be split: '
                         'organ, banan, milit, regul. `amerikan`/`kristan` are NOT in '
                         'ReVo and so correctly stay decomposable.',
            'n_tier1': len(tier1), 'n_tier2': len(tier2),
            'n_protected': len(protected),
            'n_tier2_contradicted_by_revo': len(contradicted),
            'note': 'v1 harvested from subj_radiko and LAUNDERED the parser\'s own '
                    'mis-splits back in as roots (org<-organo, amerikan<-amerikano, '
                    'mak<-Makita). Contamination is now neutralised by PROTECTION, '
                    'not exclusion — removing tier 2 costs more coverage than it '
                    'buys purity. See #806.',
        },
    }, ensure_ascii=False, indent=1), encoding='utf-8')
    tmp.rename(OUT)
    print(f'\n  wrote {OUT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
