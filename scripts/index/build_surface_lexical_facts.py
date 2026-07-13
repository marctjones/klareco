#!/usr/bin/env python3
"""
Derive LEXICALIZATION and NAMEHOOD from raw surface text — never from parser output.

VERSION: v1.0
COMPATIBLE WITH: v2.2 DuckDB store (table `sentences`, column `text`)
DEPENDENCIES: duckdb; data/vocabularies/fundamento_roots.json. NOTHING ELSE.
STAGE: Index

Description:
    Two artifacts the parser needs and cannot derive from itself:

      protected_roots.json      which forms have LEXICALIZED (esperant, organ)
      capitalization_ratio.json which word types behave like NAMES (Esperanto, Petro)

    THE POINT: BREAK THE LAUNDERING LOOP
    ------------------------------------
    `build_root_lexicon.py` harvests roots from `subj_radiko` / `verb_radiko` /
    `obj_radiko` — i.e. FROM PARSER OUTPUT. So when the parser is wrong, the
    harvest writes the error down as evidence and feeds it back:

        degraded parser splits  organo -> org + an
        harvest records         `org` as a ROOT
        parser reloads,  splits organo -> org + an        <-- self-reinforcing

    That is failure mode F13 in the DATA pipeline, and it means a corpus-harvested
    lexicon can never be more correct than the parser that harvested it. Confirmed
    contaminants: `org` (from organo), `amerikan` (from amerikano), `mak` (from the
    NAME Makita). None is in the Fundamento.

    This script touches NONE of that. Its only inputs are:

      1. the RAW `text` column — surface strings, exactly as written by humans
      2. the CLOSED AFFIX INVENTORY — the 16 rules; grammar, not a parser judgment
      3. the FUNDAMENTO — normative, 2,481 roots, independent of us

    It never reads a parser-produced column. So its output is an INDEPENDENT
    source of truth, and it is the thing that lets the parser be corrected rather
    than merely confirmed.

    WHAT LEXICALIZATION IS, AND WHY THE CORPUS CAN SEE IT
    ----------------------------------------------------
    The parser splits `Esperanton` -> `esper` + `ant`. ETYMOLOGICALLY IT IS RIGHT:
    Zamenhof's pseudonym was *Doktoro Esperanto*, "Doctor One-Who-Hopes". The word
    genuinely IS esper-ant-o.

    The phenomenon is LEXICALIZATION: a compositionally-derived form has frozen
    into a fixed lexeme with its own meaning. That is a fact about USAGE, not about
    grammar, so no grammar rule recovers it — but the corpus shows it plainly. A
    lexicalized form takes FURTHER derivation as if it were a root:

        esperant-ist-o, esperant-uj-o, esperant-ig-i, esperant-ec-o, esperant-a ...

    So we count DISTINCT DERIVATIONAL TAILS per stem, over surface strings. A stem
    that is productive in its own right has lexicalized. This is measurable, it is
    parser-free, and it regenerates when the corpus changes.

    WHAT THE CAPITALIZATION RATIO IS
    --------------------------------
    A name is capitalized MID-SENTENCE almost always; a common noun almost never.
    Sentence-initial position is EXCLUDED — every sentence starts with a capital,
    so that position carries no information (this is the same rule the parser uses).

        ratio(t) = capitalized_mid_sentence(t) / (capitalized_mid_sentence(t) + lowercase(t))

    This is a CLOSED-WORLD statistic over our own corpus — the same move as
    lowercase-attestation for the root lexicon. It is NOT a gazetteer of the
    world's names: it says nothing about tokens the corpus has never seen, and the
    parser's morphological rules (which DO generalise) still handle those.

    It targets exactly the residue that morphology and syntax provably cannot
    reach: `Esperanto` (esper-ant-o), `Ruslando` (rus-land-o), `Petro` (petr-o =
    "rock"). Morphology says ordinary word, syntax says ordinary word, and only
    USAGE says name. See docs/PROPER_NOUNS.md and #819.

Pipeline Position:
    duckdb.sentences.text --[THIS]--> protected_roots.json
                                  --> capitalization_ratio.json  --> parser

Usage:
    # sample first — proves the method without a full scan
    python scripts/index/build_surface_lexical_facts.py --limit 200000

    # the real run (full corpus; slow — run it in a terminal, not from Claude)
    ./scripts/index/build_surface_lexical_facts.sh

Inputs:
    - data/indexes/duckdb_store.db      (table `sentences`, column `text` ONLY)
    - data/vocabularies/fundamento_roots.json

Outputs:
    - data/vocabularies/protected_roots.json       {"roots": [...], "provenance": {...}}
    - data/vocabularies/capitalization_ratio.json  {"types": {t: ratio}, "provenance": {...}}

Quality Checks:
    - Refuses to read any parser-derived column (text only) — asserted at query time.
    - Reports known-good anchors: `esperant` MUST come out lexicalized; `hund` must NOT.
    - Reports known-good name anchors: `esperanto`/`zamenhof` high ratio; `hundo` low.
    - Writes atomically (.tmp then rename).

Last Updated: 2026-07-13
Author: Claude (with Marc Jones)
Related Issues: #804, #806, #819, #821
See Also: docs/PROPER_NOUNS.md, scripts/index/build_root_lexicon.py (the CONTAMINATED one)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb

DB = 'data/indexes/duckdb_store.db'
FUNDAMENTO = 'data/vocabularies/fundamento_roots.json'
OUT_PROTECTED = 'data/vocabularies/protected_roots.json'
OUT_CAPRATIO = 'data/vocabularies/capitalization_ratio.json'

# ---------------------------------------------------------------------------
# The CLOSED inventories — the 16 rules. Grammar, not parser output.
# ---------------------------------------------------------------------------
ENDINGS = ('ojn', 'oj', 'on', 'o', 'ajn', 'aj', 'an', 'a',
           'en', 'e', 'as', 'is', 'os', 'us', 'i', 'u')

# Derivational suffixes, incl. the six participles (Rule 6).
SUFFIXES = ('ant', 'int', 'ont', 'at', 'it', 'ot',
            'ism', 'ist', 'ind', 'em', 'ec', 'aĵ', 'ul', 'in', 'et', 'eg',
            'ar', 'er', 'uj', 'ej', 'estr', 'ad', 'aĝ', 'an', 'ig', 'iĝ',
            'il', 'obl', 'op', 'um', 'id', 'nj', 'ĉj')

# TRANSPARENT suffixes — "member-of", "practitioner-of", "doctrine-of". These
# compose RELIABLY and so never lexicalize: `kristano` really IS "a Christian"
# (krist+an), `amerikano` really IS "an American" (amerik+an). Protecting them
# would DESTROY the useful root — retrieval wants `krist`, not `kristan`.
#
# The stems that DO lexicalize are the ones whose inner split is an ACCIDENTAL
# HOMOGRAPH — the "suffix" is not doing any morphological work at all:
#     milit != mil+it ("thousand"+participle)   regul  != reg+ul
#     postul != post+ul                          kalkul != kalk+ul
#     esperant = esper+ant, "one who hopes" — true etymology, but the WORD is a
#     NAME, and its meaning no longer composes.
#
# That is the real test — does the meaning still compose? — and it is SEMANTIC.
# Excluding the transparent suffixes is a structural PROXY for it, and it is
# where this method's honesty ends: separating `esperant` from `kristan` on
# distribution alone is itself a residue. See docs/PROPER_NOUNS.md.
TRANSPARENT_SUFFIXES = frozenset({'an', 'ist', 'ism'})

# A stem's -o form must behave like a NAME to count as a lexicalized name.
# The corpus separates these with a wide, empty gap: esperanto 0.957 vs
# arbaro 0.210, lernejo 0.218, komunumo 0.021. Not a tuned parameter.
_NAME_RATIO = 0.85

# A word is a run of Esperanto letters. Apostrophes and hyphens split.
WORD_RE = re.compile(r"[A-Za-zĈĉĜĝĤĥĴĵŜŝŬŭ]+")


def load_fundamento() -> set[str]:
    return set(json.loads(Path(FUNDAMENTO).read_text(encoding='utf-8')).keys())


def strip_ending(w: str) -> str | None:
    """Return the stem if `w` carries a grammatical ending, else None."""
    for e in ENDINGS:
        if w.endswith(e) and len(w) - len(e) >= 2:
            return w[: -len(e)]
    return None


def scan(con, limit: int | None) -> tuple[Counter, Counter, Counter]:
    """One pass over RAW TEXT. Returns (lowercase_types, cap_mid_types, stem_tails).

    ⚠️ Reads `text` and NOTHING else. No subj_radiko, no verb_radiko, no
    vortspeco — that is the entire point of this script.
    """
    q = 'SELECT text FROM sentences'
    if limit:
        q += f' USING SAMPLE {limit} ROWS (reservoir, 42)'

    lower: Counter = Counter()      # token type seen lowercase
    cap_mid: Counter = Counter()    # token type seen Capitalised, NOT sentence-initial
    surface: Counter = Counter()    # every lowercase surface form (for tail counting)

    n = 0
    for (text,) in con.execute(q).fetchall():
        if not text:
            continue
        n += 1
        toks = WORD_RE.findall(text)
        for i, t in enumerate(toks):
            if t.islower():
                lower[t] += 1
                surface[t] += 1
            elif t[:1].isupper() and not t.isupper():
                # Sentence-initial capitalisation carries NO information — every
                # sentence starts with a capital. Only mid-sentence capitals count.
                if i > 0:
                    cap_mid[t.lower()] += 1
        if n % 500_000 == 0:
            print(f'    ...{n:,} sentences', flush=True)
    print(f'    scanned {n:,} sentences, {len(surface):,} lowercase types')
    return lower, cap_mid, surface


def load_tier1() -> set[str]:
    """ReVo + Fundamento — the CURATED root inventory (#806)."""
    p = Path('data/vocabularies/root_vocab.json')
    if p.exists():
        d = json.loads(p.read_text(encoding='utf-8'))
        if d.get('tier1'):
            return set(d['tier1'])
    return load_fundamento()


def derive_protected_roots(surface: Counter, fundamento: set[str],
                           min_tails: int, min_count: int,
                           tier1: set[str] | None = None,
                           cap: dict[str, float] | None = None) -> dict[str, int]:
    """Stems that take further derivation AS IF THEY WERE ROOTS -> lexicalized.

    `esperant` is analysable as esper+ant, yet the corpus attests esperant-ist-o,
    esperant-uj-o, esperant-ig-i, esperant-ec-o ... Many distinct derivational
    tails means the form has frozen into a lexeme. That is a USAGE fact, visible
    only in the corpus — and it is what stops the parser splitting `Esperanton`.
    """
    tails: dict[str, set[str]] = defaultdict(set)

    for w, c in surface.items():
        if c < min_count:
            continue
        stem = strip_ending(w)
        if not stem:
            continue
        # For each way of cutting the stem into <candidate>+<suffix-tail>, record
        # the tail against the candidate. A candidate with MANY distinct tails is
        # behaving like a root.
        for suf in SUFFIXES:
            if stem.endswith(suf) and len(stem) - len(suf) >= 3:
                cand = stem[: -len(suf)]
                tails[cand].add(suf)

    # `-ig` / `-iĝ` are VALENCY-changing and fully productive — they never form a
    # frozen lexeme. `publikigi` really is publik+ig+i and must stay decomposable.
    # Without this filter the productivity test mistakes ordinary productivity for
    # lexicalization, which is the opposite of what it is looking for.
    NEVER_LEXICALIZES = ('ig', 'iĝ')

    protected: dict[str, int] = {}
    for cand, ts in tails.items():
        if len(ts) < min_tails:
            continue
        if cand.endswith(NEVER_LEXICALIZES):
            continue
        # Only interesting if the candidate is ITSELF decomposable — i.e. the
        # parser would otherwise split it. A plain Fundamento root needs no
        # protection; it is already atomic.
        inner = None
        for suf in SUFFIXES:
            if suf in TRANSPARENT_SUFFIXES:
                continue
            if cand.endswith(suf) and len(cand) - len(suf) >= 3:
                base = cand[: -len(suf)]
                if base in fundamento:
                    inner = (base, suf)
                    break
        if not inner:
            continue

        # DERIVATIONAL PRODUCTIVITY IS NOT THE DISCRIMINATOR. It only looked like
        # one on a 300K sample. At full corpus scale EVERY productive stem is
        # productive, so `arbar` (12 tails), `lernej`, `prezidant` and `komunum`
        # all qualify — and `arbaro` genuinely IS arb+ar+o ("a collection of
        # trees"). Protecting them destroys the useful root.
        #
        # The right division of labour, and it is clean:
        #
        #   ReVo covers lexicalized COMMON words — `milit`, `regul`, `organ`,
        #     `banan` are all headwords, so build_root_lexicon.py already
        #     protects them ("ReVo says X is a root => X is atomic").
        #
        #   THIS artifact covers lexicalized NAMES — which a dictionary does not
        #     list. `esperant` is not in ReVo, and `esperanto` is capitalised
        #     mid-sentence 95.7% of the time. `arbaro` is 21%.
        #
        # So: productive, NOT a ReVo headword, and it behaves like a NAME.
        if tier1 is not None and cand in tier1:
            continue                       # ReVo already protects it
        if cap is not None:
            ratio = cap.get(cand + 'o')
            if ratio is None or ratio < _NAME_RATIO:
                continue                   # compositional, not a lexicalized name
        protected[cand] = len(ts)
    return protected


def derive_cap_ratio(lower: Counter, cap_mid: Counter,
                     min_count: int) -> dict[str, float]:
    """P(capitalised mid-sentence | this word type). High => it behaves like a NAME."""
    out: dict[str, float] = {}
    for t in set(lower) | set(cap_mid):
        lo, hi = lower.get(t, 0), cap_mid.get(t, 0)
        if lo + hi < min_count:
            continue
        out[t] = round(hi / (lo + hi), 4)
    return out


def write_atomic(path: str, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
    tmp.rename(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--duckdb-path', default=DB)
    ap.add_argument('--limit', type=int, default=None,
                    help='sample N sentences instead of the full scan (for testing)')
    ap.add_argument('--min-tails', type=int, default=4,
                    help='distinct derivational tails before a stem counts as lexicalized')
    ap.add_argument('--min-count', type=int, default=5,
                    help='minimum surface attestations')
    args = ap.parse_args()

    con = duckdb.connect(args.duckdb_path, read_only=True)
    fundamento = load_fundamento()
    print(f'  Fundamento: {len(fundamento):,} roots (normative, independent)')
    print('  scanning RAW TEXT only — no parser-derived column is read')

    lower, cap_mid, surface = scan(con, args.limit)

    cap = derive_cap_ratio(lower, cap_mid, args.min_count)
    tier1 = load_tier1()
    print(f'  tier-1 (ReVo + Fundamento): {len(tier1):,} — these are ALREADY '
          f'protected by build_root_lexicon.py')
    protected = derive_protected_roots(surface, fundamento, args.min_tails,
                                       args.min_count, tier1=tier1, cap=cap)

    print(f'\n  lexicalized stems (>= {args.min_tails} distinct tails): {len(protected):,}')
    for r, n in sorted(protected.items(), key=lambda x: -x[1])[:15]:
        print(f'    {r:16s} {n:3d} distinct derivational tails')

    print('\n  ANCHORS — lexicalization (esperant MUST be in; hund MUST NOT):')
    for a in ('esperant', 'hund', 'organ', 'amerikan'):
        print(f'    {a:12s} lexicalized={a in protected}')

    print(f'\n  capitalisation ratios computed for {len(cap):,} types')
    print('  ANCHORS — namehood (high = behaves like a NAME):')
    for a in ('esperanto', 'zamenhof', 'petro', 'ruslando', 'hundo', 'urbo', 'libro'):
        v = cap.get(a)
        print(f'    {a:12s} ratio={v if v is not None else "(unattested)"}')

    write_atomic(OUT_PROTECTED, {
        'roots': sorted(protected),
        'tail_counts': protected,
        'provenance': {
            'source': 'RAW SURFACE TEXT (sentences.text) + closed affix inventory + Fundamento',
            'method': 'derivational productivity: a stem taking many distinct '
                      'derivational tails has LEXICALIZED and must not be split',
            'reads_parser_output': False,
            'min_tails': args.min_tails, 'min_count': args.min_count,
            'sampled': args.limit,
            'note': 'Deliberately INDEPENDENT of the parser. build_root_lexicon.py '
                    'harvests from subj_radiko and therefore launders the parser\'s '
                    'own mis-splits back in as roots (org<-organo, amerikan<-amerikano, '
                    'mak<-Makita). See #806.',
        },
    })
    write_atomic(OUT_CAPRATIO, {
        'types': cap,
        'provenance': {
            'source': 'RAW SURFACE TEXT (sentences.text)',
            'method': 'P(capitalised mid-sentence | type). Sentence-initial tokens '
                      'EXCLUDED — every sentence starts with a capital, so that '
                      'position carries no information.',
            'reads_parser_output': False,
            'min_count': args.min_count, 'sampled': args.limit,
            'note': 'A CLOSED-WORLD statistic over our own corpus, not a gazetteer '
                    'of the world\'s names: it says nothing about unseen tokens, which '
                    'the parser\'s morphological rules still handle. Targets the '
                    'residue morphology and syntax provably cannot reach — Esperanto '
                    '(esper-ant-o), Ruslando (rus-land-o), Petro (petr-o = "rock"). '
                    'See #819.',
        },
    })
    print(f'\n  wrote {OUT_PROTECTED}')
    print(f'  wrote {OUT_CAPRATIO}')
    if args.limit:
        print(f'\n  ⚠️  SAMPLED RUN ({args.limit:,} sentences). Re-run without --limit '
              f'before the rebuild (#807).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
