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
    uas = las = pos_ok = n = 0
    aligned = 0

    for i, g in enumerate(gold_sents):
        text = g['text'] or ' '.join(t['form'] for t in g['tokens'])
        try:
            block = to_conllu(text, sent_id=str(i + 1))
        except Exception:
            continue
        if emit is not None:
            emit.append(block)
        ours = parse_our_conllu(block)

        # Align by surface form, left to right.
        #
        # ⚠️ INDEX SPACES DIFFER. Gold NUMBERS PUNCTUATION as tokens; our
        # tokenizer drops it. So our HEAD=4 and gold's HEAD=4 do not refer to the
        # same word, and comparing them raw scores almost everything wrong even
        # when the attachment is correct. We must translate our heads into GOLD's
        # index space first. (This bug alone was holding UAS at ~10%.)
        ours_to_gold: dict[int, int] = {}
        gi = 0
        for o in ours:
            while gi < len(g['tokens']) and \
                    g['tokens'][gi]['form'].lower() != o['form'].lower():
                gi += 1
            if gi >= len(g['tokens']):
                break
            ours_to_gold[o['id']] = g['tokens'][gi]['id']
            gi += 1

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
        n += len([t for t in g['tokens'] if t['upos'] != 'PUNCT'])

    return {'aligned': aligned, 'gold_tokens': n,
            'uas': uas / aligned if aligned else 0.0,
            'las': las / aligned if aligned else 0.0,
            'upos': pos_ok / aligned if aligned else 0.0,
            'coverage': aligned / n if n else 0.0}


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
        print(f'    UAS  (correct HEAD)            {r["uas"]:6.1%}')
        print(f'    LAS  (correct HEAD + RELATION) {r["las"]:6.1%}')
        print(f'    UPOS                           {r["upos"]:6.1%}')
        print(f'    token coverage                 {r["coverage"]:6.1%}  '
              f'({r["aligned"]} of {r["gold_tokens"]} non-punct gold tokens)')

    if args.emit and emit:
        Path(args.emit).write_text('\n'.join(emit), encoding='utf-8')
        print(f'\n  wrote {args.emit}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
