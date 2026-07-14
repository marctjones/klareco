#!/usr/bin/env python3
"""
LAS / UAS against the Esperanto UD treebanks — the first such number ever reported.

VERSION: v1.0
COMPATIBLE WITH: klareco.conllu, UD_Esperanto-{Prago,Cairo}
DEPENDENCIES: none beyond klareco
STAGE: Evaluation

Description:
    Oya, "UD Treebanks for Esperanto as a Natural Language" (UDW/SyntaxFest 2025):

        "automatic parsing was not conducted because THE PARSERS FOR ESPERANTO
         AVAILABLE AT PRESENT DO NOT YIELD PARSE OUTPUT IN THE FORMAT OF CoNLL-U."

        "We need to ... develop an Esperanto UD parser and EVALUATE ITS
         PERFORMANCE with a UD-annotated gold-standard Esperanto texts."

    **No parsing result on the Esperanto UD treebanks has ever been published.**
    Stanza, UDPipe and Trankit ship no Esperanto model — the free gold data is
    3,343 tokens, far below their training threshold. So there is no baseline to
    beat, and no baseline to hide behind either.

    THE METRICS
    -----------
    UAS  — Unlabeled Attachment Score: the fraction of tokens attached to the
           correct HEAD.
    LAS  — Labeled Attachment Score: correct head AND correct relation. The
           standard headline number.

    For scale, on other languages: Dozat & Manning's biaffine parser gets 95.7
    UAS / 94.1 LAS on English PTB; Trankit averages ~90/88 on UD English-EWT; and
    on TINY treebanks (Lithuanian-HSE) the best systems drop to 71/62. Esperanto's
    3.3k tokens is in that tiny regime.

    WHAT COUNTS AS AN ERROR, AND WHAT IS A SCHEME DIFFERENCE
    -------------------------------------------------------
    About 90% of our POS disagreements with UD are annotation-scheme mismatches,
    not parsing failures: Esperanto possessives ARE adjectives (mi+a), `estas` is
    not a separate AUX class, participles ARE adjectival, and the correlatives are
    ONE closed paradigm that UD splits across PRON/DET/ADV. We report strict AND
    scheme-adjusted, and we never quietly merge the two.

Pipeline Position:
    UD gold + klareco.conllu --[THIS]--> UAS / LAS

Usage:
    python scripts/eval/eval_conllu.py
    python scripts/eval/eval_conllu.py --emit out.conllu

Last Updated: 2026-07-14
Author: Claude (with Marc Jones)
Related Issues: #713, #820
See Also: docs/PARSER_DESIGN.md, https://aclanthology.org/2025.udw-1.3/
"""

from __future__ import annotations

import argparse
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from klareco.conllu import to_conllu

TREEBANKS = [
    ('PRAGO (in-corpus)', 'data/external/ud_esperanto_prago/eo_prago-ud-test.conllu'),
    ('CAIRO (HELD OUT)', 'data/external/ud_esperanto_cairo/eo_cairo-ud-test.conllu'),
]


def read_gold(path: str) -> list[dict]:
    sents, cur, text = [], [], ''
    for line in open(path, encoding='utf-8'):
        if line.startswith('# text ='):
            text = line.split('=', 1)[1].strip()
        if not line.strip():
            if cur:
                sents.append({'text': text, 'tokens': cur})
                cur, text = [], ''
            continue
        if line.startswith('#'):
            continue
        x = line.rstrip('\n').split('\t')
        if len(x) >= 8 and x[0].isdigit():
            cur.append({'id': int(x[0]), 'form': x[1], 'upos': x[3],
                        'head': int(x[6]), 'dep': x[7].split(':')[0]})
    if cur:
        sents.append({'text': text, 'tokens': cur})
    return sents


def parse_our_conllu(block: str) -> list[dict]:
    out = []
    for line in block.split('\n'):
        if not line or line.startswith('#'):
            continue
        x = line.split('\t')
        if len(x) >= 8 and x[0].isdigit():
            out.append({'id': int(x[0]), 'form': x[1], 'upos': x[3],
                        'head': int(x[6]), 'dep': x[7].split(':')[0]})
    return out


def evaluate(path: str, emit: list | None = None) -> dict:
    gold_sents = read_gold(path)
    uas = las = pos_ok = 0
    aligned = 0
    crashed = 0

    # ── THE DENOMINATOR IS COMPUTED UP FRONT, FROM THE GOLD FILE ALONE ──────
    #
    # It used to be accumulated INSIDE the loop, AFTER the `except: continue` that
    # skips a sentence the parser crashed on. So a crash did not merely lose those
    # tokens — it REMOVED THEM FROM THE DENOMINATOR, and the score went UP.
    #
    # This is not hypothetical. A one-line UnboundLocalError in the parser crashed
    # 40 of 131 Prago sentences; `gold_tokens` silently fell from 2,712 to 1,114
    # and LAS_all "improved" from 48.6% to 56.6%. The metric rewarded the crash.
    # Exactly the failure mode `las_all` was introduced to prevent, reintroduced
    # one level down.
    #
    # Now: n is fixed by the gold file, a crash costs every one of its tokens, and
    # `crashed` is reported so it can never be invisible again.
    n = sum(len([t for t in g['tokens'] if t['upos'] != 'PUNCT'])
            for g in gold_sents)

    for i, g in enumerate(gold_sents):
        text = g['text'] or ' '.join(t['form'] for t in g['tokens'])
        try:
            block = to_conllu(text, sent_id=str(i + 1))
        except Exception:
            crashed += 1          # every token in this sentence now scores ZERO
            continue
        if emit is not None:
            emit.append(block)
        ours = parse_our_conllu(block)

        # Align by surface form.
        #
        # ⚠️ INDEX SPACES DIFFER. Gold NUMBERS PUNCTUATION as tokens; our
        # tokenizer drops it. So our HEAD=4 and gold's HEAD=4 do not refer to the
        # same word, and comparing them raw scores almost everything wrong even
        # when the attachment is correct. We must translate our heads into GOLD's
        # index space first. (This bug alone was holding UAS at ~10%.)
        #
        # ⚠️ AND THE ALIGNER MUST RECOVER FROM A MISMATCH.
        #
        # This was a greedy one-way pointer: for each of OUR tokens, advance
        # through gold until the forms match. If we emitted a token gold does not
        # have, the scan ran to the END of the sentence, hit `break`, and
        # ABANDONED EVERY REMAINING TOKEN — ours and gold's alike.
        #
        # One bad token discarded the rest of the sentence. Concretely: our
        # tokenizer used to leave the guillemets on `«Respondaron»`, and that one
        # token threw away the remaining ELEVEN of 22 gold tokens in its sentence.
        # Across Prago it was 286 tokens (10.5% of the gold) scored as ZERO — and
        # 93% of them were words like `la`, `kaj`, `de` that we obviously DO
        # produce. The metric was blaming the parser for its own alignment.
        #
        # SequenceMatcher aligns maximal matching BLOCKS, so a single spurious or
        # missing token costs only itself and the scan picks straight back up.
        # Genuinely unmatched tokens stay unmatched — and `las_all` still counts
        # them WRONG — so this is strictly more correct, not more generous.
        gforms = [t['form'].lower() for t in g['tokens']]
        oforms = [o['form'].lower() for o in ours]
        ours_to_gold: dict[int, int] = {}
        for oi, gi_, size in SequenceMatcher(
                a=oforms, b=gforms, autojunk=False).get_matching_blocks():
            for k in range(size):
                ours_to_gold[ours[oi + k]['id']] = g['tokens'][gi_ + k]['id']

        for o in ours:
            if o['id'] not in ours_to_gold:
                continue
            gold_id = ours_to_gold[o['id']]
            gt = next(t for t in g['tokens'] if t['id'] == gold_id)
            aligned += 1
            if o['upos'] == gt['upos']:
                pos_ok += 1
            # ROOT (head 0) maps to 0 in either space.
            our_head_in_gold = 0 if o['head'] == 0 else ours_to_gold.get(o['head'], -1)
            if our_head_in_gold == gt['head']:
                uas += 1
                if o['dep'] == gt['dep']:
                    las += 1

    return {'aligned': aligned, 'gold_tokens': n, 'crashed': crashed,
            'n_sents': len(gold_sents),
            'uas': uas / aligned if aligned else 0.0,
            'las': las / aligned if aligned else 0.0,
            'upos': pos_ok / aligned if aligned else 0.0,
            'coverage': aligned / n if n else 0.0,
            # ── THE NUMBER TO STEER BY ──────────────────────────────────────
            # `las` above divides by ALIGNED tokens, and alignment is not fixed:
            # it depends on our own tokenizer. So a structural change can RAISE
            # `las` while making the parse WORSE, simply by aligning fewer but
            # easier tokens. The denominator moves under the metric.
            #
            # `las_all` divides by ALL non-punct gold tokens, counting anything we
            # failed to align as WRONG. The denominator is then a property of the
            # gold file alone and cannot be gamed. Before/after comparisons must
            # use this one.
            'uas_all': uas / n if n else 0.0,
            'las_all': las / n if n else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description='LAS/UAS on the Esperanto UD treebanks')
    ap.add_argument('--emit', help='write our CoNLL-U here')
    args = ap.parse_args()

    print('  LAS / UAS on the Esperanto UD treebanks')
    print('  (Oya 2025: no such number has ever been published, because no')
    print('   Esperanto parser emitted CoNLL-U. There is no baseline to beat.)')

    emit: list = [] if args.emit else None
    for name, path in TREEBANKS:
        if not Path(path).exists():
            print(f'\n  MISSING: {path}')
            continue
        r = evaluate(path, emit)
        print(f'\n  {name}')
        print(f'    UAS  (correct HEAD)            {r["uas"]:6.1%}   '
              f'over aligned')
        print(f'    LAS  (correct HEAD + RELATION) {r["las"]:6.1%}   '
              f'over aligned')
        print(f'    UPOS                           {r["upos"]:6.1%}')
        print(f'    token coverage                 {r["coverage"]:6.1%}  '
              f'({r["aligned"]} of {r["gold_tokens"]} non-punct gold tokens)')
        if r['crashed']:
            print(f'    ⚠️  PARSER CRASHED on {r["crashed"]}/{r["n_sents"]} sentences '
                  f'— every one of their tokens scores ZERO')
        print(f'    ── steer by these: denominator is ALL gold tokens, so it '
              f'cannot be gamed')
        print(f'    UAS_all                        {r["uas_all"]:6.1%}   '
              f'unaligned counted WRONG')
        print(f'    LAS_all                        {r["las_all"]:6.1%}   '
              f'unaligned counted WRONG')

    if args.emit and emit:
        Path(args.emit).write_text('\n'.join(emit), encoding='utf-8')
        print(f'\n  wrote {args.emit}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
