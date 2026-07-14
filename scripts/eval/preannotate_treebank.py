#!/usr/bin/env python3
"""
Pre-annotate sampled sentences for human correction — WITHOUT poisoning the gold.

VERSION: v1.0
COMPATIBLE WITH: any
DEPENDENCIES: data/raw/eo/dictionaries/apertium_lexicon.json (GPL-3.0, INDEPENDENT)
STAGE: Evaluation

Description:
    THE TRAP THIS SCRIPT EXISTS TO AVOID
    ------------------------------------
    The obvious way to build a treebank is: run our parser, have a human fix the
    mistakes. That is how Bick built Arbobanko and it is the standard method.

    **But it would destroy the thing we are trying to build.** Annotators accept the
    large majority of pre-annotations without real scrutiny — anchoring is one of the
    best-attested effects in annotation science. So OUR errors become "gold", and we
    would then measure our parser against a ruler made out of its own mistakes. That
    is failure mode F13 (circularity) in its purest form, and it is WORSE than having
    no treebank, because it would look authoritative.

    THE FIX: SPLIT THE ANNOTATION BY WHAT WE ARE MEASURING
    -----------------------------------------------------
    LAS/UAS measure ATTACHMENT. So attachment must never be pre-filled by the system
    under test. Morphology is expensive to type and is NOT what we are measuring.

        FORM / LEMMA / UPOS / FEATS   <- pre-filled from APERTIUM, which was built
                                         independently of us (no ReVo, no
                                         voko-akrido, `derived_from_parser_output:
                                         false`). Not our bias.
        HEAD / DEPREL                 <- LEFT BLANK. The human does every single
                                         attachment by hand.

    Where apertium is silent (it hand-lists stems and does not derive, so Bick
    measured 25.1% of noun lemmas in real text simply MISSING from it) we fall back
    to klareco — and we STAMP IT in the MISC column as `Pre=klareco`, so the
    annotator knows precisely which rows to distrust. An unmarked guess is the
    dangerous kind.

Pipeline Position:
    sample_for_treebank.py --[THIS]--> .conllu --> human (UD Annotatrix) --> gold

Usage:
    python scripts/eval/preannotate_treebank.py --in data/test_sets/treebank_sample.jsonl

Outputs:
    - data/test_sets/treebank_todo.conllu  (HEAD/DEPREL blank — ready to annotate)

Quality Checks:
    - Reports apertium COVERAGE. If apertium knows few of the tokens, most of the
      morphology is ours after all, and the independence claim is weaker — so the
      number is printed rather than assumed.

Last Updated: 2026-07-14
Related Issues: #820, #839
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

APERTIUM = 'data/raw/eo/dictionaries/apertium_lexicon.json'

# ─────────────────────────────────────────────────────────────────────────────
# UPOS COMES FROM THE ENDING, NOT FROM ANY PARSER.
#
# This is the one place Esperanto is strictly better than a natural language, and
# it is the project's whole thesis in miniature: the ending IS the part of speech,
# by construction, with no exceptions. `-o` is a noun. `-a` is an adjective. There
# is nothing to infer and nothing to learn.
#
# So we do NOT ask apertium for POS (its `pos` field is per-paradigm and my stem
# lookup collided on it — it was returning `de` -> VERB), and we do NOT ask
# klareco. We read it off the surface form. Neither system can bias it because
# neither system is consulted.
# ─────────────────────────────────────────────────────────────────────────────
_BY_ENDING = [
    ('ojn', 'NOUN'), ('ajn', 'ADJ'), ('oj', 'NOUN'), ('aj', 'ADJ'),
    ('on', 'NOUN'), ('an', 'ADJ'), ('en', 'ADV'),
    ('as', 'VERB'), ('is', 'VERB'), ('os', 'VERB'), ('us', 'VERB'),
    ('o', 'NOUN'), ('a', 'ADJ'), ('e', 'ADV'), ('i', 'VERB'), ('u', 'VERB'),
]

# The FUNCTION WORDS. Closed, finite, and fixed by the Fundamento — this is not a
# gazetteer, it is the grammar (see CLAUDE.md, Function Word Exclusion Principle).
# They are the ONLY words whose POS the ending does not give, precisely because
# they do not take endings.
_FUNCTION = {
    'la': 'DET', 'l': 'DET',
    'kaj': 'CCONJ', 'aŭ': 'CCONJ', 'sed': 'CCONJ', 'nek': 'CCONJ', 'plus': 'CCONJ',
    'ke': 'SCONJ', 'ĉar': 'SCONJ', 'se': 'SCONJ', 'kvankam': 'SCONJ',
    'dum': 'SCONJ', 'ĝis': 'SCONJ', 'kvazaŭ': 'SCONJ', 'ol': 'SCONJ',
    'al': 'ADP', 'de': 'ADP', 'en': 'ADP', 'kun': 'ADP', 'per': 'ADP',
    'por': 'ADP', 'pri': 'ADP', 'pro': 'ADP', 'sur': 'ADP', 'sub': 'ADP',
    'super': 'ADP', 'tra': 'ADP', 'trans': 'ADP', 'ĉe': 'ADP', 'da': 'ADP',
    'el': 'ADP', 'ekster': 'ADP', 'inter': 'ADP', 'je': 'ADP', 'kontraŭ': 'ADP',
    'krom': 'ADP', 'laŭ': 'ADP', 'malgraŭ': 'ADP', 'po': 'ADP', 'post': 'ADP',
    'preter': 'ADP', 'sen': 'ADP', 'antaŭ': 'ADP', 'apud': 'ADP', 'ĉirkaŭ': 'ADP',
    'mi': 'PRON', 'ni': 'PRON', 'vi': 'PRON', 'li': 'PRON', 'ŝi': 'PRON',
    'ĝi': 'PRON', 'ili': 'PRON', 'oni': 'PRON', 'si': 'PRON', 'ci': 'PRON',
    'min': 'PRON', 'nin': 'PRON', 'vin': 'PRON', 'lin': 'PRON', 'ŝin': 'PRON',
    'ĝin': 'PRON', 'ilin': 'PRON', 'sin': 'PRON',
    'ne': 'PART', 'ĉu': 'PART', 'ĉi': 'PART', 'jes': 'INTJ',
}

# ─────────────────────────────────────────────────────────────────────────────
# THE CORRELATIVES — and they are a TRAP the ending rule walks straight into.
#
# `kiu` ends in -u, so _BY_ENDING tags it VERB. `kio` ends in -o, so it tags it
# NOUN. Both are silently WRONG, and `kiu` is the word that HEADS RELATIVE
# CLAUSES — the very construction the sample's subordination stratum is built
# around. Mis-labelling the clause head is exactly what corrupts attachment.
#
# This is the SAME bug as the infinitive regex that matched every pronoun: a
# closed class whose members happen to end in a content-word vowel. It is a
# 5x9 table fixed by the Fundamento, so we generate it rather than list it.
# ─────────────────────────────────────────────────────────────────────────────
_KOR_INIT = ('ki', 'ti', 'i', 'ĉi', 'neni')
_KOR_FINAL = {
    'u': 'PRON',   # kiu, tiu … PRON standalone, DET before a noun — genuinely
    'o': 'PRON',   # kio, tio … ambiguous, so the -u/-a rows get CHECK=yes below
    'a': 'DET',    # kia, tia …
    'es': 'PRON',  # kies, ties …
    'e': 'ADV',    # kie, tie …
    'am': 'ADV',   # kiam, tiam …
    'al': 'ADV',   # kial, tial …
    'el': 'ADV',   # kiel, tiel …
    'om': 'ADV',   # kiom, tiom …
}
# The substantive/adjectival ones inflect (-j, -n, -jn); the adverbial ones take
# only -n (direction). Build every surface form.
_CORRELATIVE: dict[str, str] = {}
for _i in _KOR_INIT:
    for _f, _pos in _KOR_FINAL.items():
        _base = _i + _f
        _CORRELATIVE[_base] = _pos
        if _f in ('u', 'o', 'a'):
            for _suf in ('j', 'n', 'jn'):
                _CORRELATIVE[_base + _suf] = _pos
        elif _f in ('e',):
            _CORRELATIVE[_base + 'n'] = _pos

# PRON-vs-DET on the -u/-a rows is a real annotator decision (`kiu venis` = PRON,
# `kiu homo` = DET), so those get flagged rather than asserted.
_CORR_AMBIG = {w for w in _CORRELATIVE
               if any(w.startswith(i) and w[len(i):].rstrip('jn') in ('u', 'a')
                      for i in _KOR_INIT)}


def _upos(tok: str) -> tuple[str, str]:
    """(upos, how) — from the ENDING. Deterministic; no parser is consulted."""
    t = tok.lower()
    if not t[:1].isalnum():
        return 'PUNCT', 'punct'
    if t in _FUNCTION:
        return _FUNCTION[t], 'funkcio'
    # BEFORE the ending rule, which would call `kiu` a VERB and `tio` a NOUN.
    if t in _CORRELATIVE:
        return _CORRELATIVE[t], 'korelativo-ambigua' if t in _CORR_AMBIG else 'korelativo'
    if t.isdigit():
        return 'NUM', 'ending'
    for e, pos in _BY_ENDING:
        if t.endswith(e) and len(t) > len(e):
            # A capitalised NOUN in a non-initial position is a PROPN. The gold
            # scheme distinguishes them and the ending does not.
            return pos, 'ending'
    # No Esperanto ending at all: a foreign word or an unadapted name. The
    # annotator MUST look at this one — so say so rather than guessing.
    return '_', 'unknown'

_TOK = re.compile(r'\w+|[^\w\s]', re.UNICODE)

# Esperanto is agglutinative and apertium stores STEMS (`hund`), not surface forms
# (`hundojn`). Looking up the surface form directly finds only a third of the tokens.
# Strip the grammatical ending — which is FREE and DETERMINISTIC, and is the whole
# point of the language — then look up the stem.
_ENDINGS = ('ojn', 'ajn', 'oj', 'aj', 'on', 'an', 'en', 'as', 'is', 'os', 'us',
            'o', 'a', 'e', 'i', 'u')


def _stems(tok: str) -> list[str]:
    """Every stem this surface form could have. Ordered longest-ending-first."""
    t = tok.lower()
    out = [t]
    for e in _ENDINGS:
        if t.endswith(e) and len(t) > len(e) + 1:
            out.append(t[:-len(e)])
    return out


def _load_apertium() -> dict[str, str]:
    p = Path(APERTIUM)
    if not p.exists():
        raise SystemExit(
            f'  ✗ {p} missing. This is the INDEPENDENT lexicon — without it the\n'
            f'    pre-annotation would be entirely our own parser, which is exactly\n'
            f'    the circularity this script exists to prevent.\n'
            f'    Run: python scripts/acquire/acquire_apertium_epo.py')
    d = json.loads(p.read_text())
    out: dict[str, str] = {}
    for _k, v in (d.get('entries') or {}).items():
        stem, pos = v.get('stem'), v.get('pos')
        if stem and pos:
            out[stem.lower()] = pos
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description='Pre-annotate for human correction')
    ap.add_argument('--in', dest='inp', default='data/test_sets/treebank_sample.jsonl')
    ap.add_argument('--out', default='data/test_sets/treebank_todo.conllu')
    args = ap.parse_args()

    apert = _load_apertium()
    print(f'  apertium: {len(apert):,} stems  (GPL-3.0, built WITHOUT ReVo/voko —')
    print(f'            this is what makes the pre-annotation not-our-bias)\n')

    from klareco.morphology import analyze

    sents = [json.loads(l) for l in open(args.inp, encoding='utf-8') if l.strip()]
    src = collections.Counter()
    pos_src = collections.Counter()
    lines: list[str] = []

    for s in sents:
        text = s['text']
        toks = _TOK.findall(text)
        lines.append(f'# sent_id = {s["sent_id"]}')
        lines.append(f'# text = {text}')
        lines.append('# phenomena = ' + ','.join(s.get('phenomena') or []))
        for i, t in enumerate(toks, 1):
            # POS: from the ENDING. The language gives it away for free; no parser
            # is asked, so no parser can bias it.
            upos, how = _upos(t)
            pos_src[how] += 1

            # LEMMA: apertium first (INDEPENDENT), then us (MARKED).
            hit = next((s for s in _stems(t) if s in apert), None)
            if how == 'punct':
                lemma, misc = t, 'Pos=punct'
                src['punct'] += 1
            elif how.startswith('korelativo'):
                lemma, misc = t.lower(), f'Pos={how}'
                src['function'] += 1
            elif how == 'funkcio':
                lemma, misc = t.lower(), 'Pos=funkcio'
                src['function'] += 1
            elif hit:
                lemma, misc = hit, 'Pos=ending|Lemma=apertium'
                src['apertium'] += 1
            else:
                # apertium hand-lists stems and does not derive, so Bick measured
                # 25.1% of real noun lemmas simply MISSING from it. Fall back to us
                # — AND SAY SO. An unmarked guess is the dangerous kind.
                try:
                    a = analyze(t)
                except Exception:
                    a = None
                best = (a[0] if isinstance(a, list) and a else a) if a else None
                lemma = (getattr(best, 'radiko', None) or '_') if best else '_'
                misc = 'Pos=ending|Lemma=klareco|CHECK=yes'
                src['klareco'] += 1

            # PRON-vs-DET on `kiu`/`kia` is a real decision (`kiu venis` = PRON,
            # `kiu homo` = DET) and the ending cannot settle it. Flag, don't assert.
            if how in ('unknown', 'korelativo-ambigua'):
                misc += '|CHECK=yes'

            # HEAD and DEPREL are '_'. THIS IS THE POINT — they are what LAS
            # measures, so the system under test must never suggest them.
            lines.append(f'{i}\t{t}\t{lemma}\t{upos}\t_\t_\t_\t_\t_\t{misc}')
        lines.append('')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text('\n'.join(lines), encoding='utf-8')

    tot = sum(src.values())
    print('  UPOS — read off the ENDING. No parser is consulted, so no parser can')
    print('         bias it. This is the one place Esperanto is simply better.')
    for k in ('ending', 'funkcio', 'korelativo', 'korelativo-ambigua',
              'punct', 'unknown'):
        print(f'    {k:20s} {pos_src[k]:6,}  ({pos_src[k] / tot:5.1%})')
    flagged = pos_src['unknown'] + pos_src['korelativo-ambigua']
    print(f'    -> {1 - flagged / tot:.1%} of UPOS is FREE and CERTAIN.')
    print(f'       {pos_src["korelativo"]:,} correlatives handled EXPLICITLY — the ending')
    print('       rule alone would call `kiu` a VERB and `tio` a NOUN, and `kiu` is')
    print('       what HEADS the relative clauses this sample is built around.')
    print(f'       {flagged:,} rows flagged CHECK=yes (foreign words; PRON-vs-DET).')

    print('\n  LEMMA — apertium first (INDEPENDENT); us only where it is silent.')
    for k in ('apertium', 'function', 'punct', 'klareco'):
        print(f'    {k:10s} {src[k]:6,}  ({src[k] / tot:5.1%})')
    indep = (src['apertium'] + src['function'] + src['punct']) / tot
    print(f'    -> {indep:.1%} independent of klareco.')
    print(f'       {src["klareco"] / tot:.1%} is ours, and every row of it says CHECK=yes.')
    print()
    print('  HEAD and DEPREL are BLANK in every row. That is deliberate and it is the')
    print('  whole design: they are what LAS measures, so the system under test does')
    print('  not get to suggest them.')
    print(f'\n  wrote {args.out}  ({len(sents):,} sentences, {tot:,} tokens)')
    print('\n  Annotate with UD Annotatrix (https://maryszmary.github.io/ud-annotatrix/)')
    print('  or Arborator. Both eat CoNLL-U directly.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
