#!/usr/bin/env python3
"""
Role-labelling benchmark — the number that was never measured, and should have been.

VERSION: v1.0
COMPATIBLE WITH: klareco.parser (any), UD_Esperanto-* treebanks
DEPENDENCIES: none beyond klareco.parser
STAGE: Evaluation

Description:
    WHY THIS EXISTS
    ---------------
    We have measured POS accuracy for months. POS is at 80.9% strict / 94.5%
    scheme-adjusted, and roughly 90% of the residual "errors" are annotation-scheme
    mismatches, not parsing failures (Esperanto possessives ARE adjectives; `estas`
    is not a separate AUX class; participles ARE adjectival). The deterministic
    parser is close to its morphological ceiling.

    Meanwhile the field the pipeline ACTUALLY READS — `subj_radiko`, the column
    DuckDBRetriever and every reranker key on — had never been measured at all.
    When first measured (2026-07-13):

        SUBJECT (nsubj)   P 53.8%   R 33.7%   F1 41.4%
        OBJECT  (obj)     P 49.1%   R 18.9%   F1 27.3%

    We were polishing a number that was already near ceiling while the one that
    matters sat at 41%. That is the exact failure the merge gate exists to stop,
    and it went unnoticed because there was no benchmark. This is the benchmark.

    THE STRUCTURAL CEILING — READ THIS BEFORE OPTIMISING
    ---------------------------------------------------
    The AST is a SINGLE-CLAUSE FRAME: one `subjekto`, one `verbo`, one `objekto`
    per sentence. Real text is not:

        1.64 gold subjects per sentence
        35.8% of sentences have 2+ subjects (i.e. multiple clauses)

    So a single-slot AST has a HARD RECALL CEILING of 42.5%, no matter how good
    the rules get. We measure 33.7% — already 79% of the maximum the SHAPE allows.

    **Rule improvements cannot break 42.5%. Only a shape change can.** This script
    reports the ceiling alongside the score so that fact stays visible, and so
    nobody spends another month optimising against a wall.

Pipeline Position:
    UD treebank + klareco.parser --[THIS]--> subject/object P/R/F1 + the ceiling

Usage:
    python scripts/eval/eval_ud_roles.py
    python scripts/eval/eval_ud_roles.py --json results/roles.json

Inputs:
    - data/external/ud_esperanto_prago/eo_prago-ud-test.conllu
    - data/external/ud_esperanto_cairo/eo_cairo-ud-test.conllu   (HELD OUT)

Outputs:
    - stdout report; optional JSON for data/perf/bench_history.jsonl

Quality Checks:
    - Reports Prago and Cairo SEPARATELY. Prago is the Prago Manifesto /
      Homaranismo — texts almost certainly IN our corpus, so anything
      corpus-derived (capitalization_ratio) may have memorised its tokens. Cairo
      is the honest, held-out ruler.
    - Reports the structural recall ceiling, so a score is never read without it.
    - Breaks precision errors down by WHAT WE WRONGLY PICKED, because that says
      whether an error is a rule bug (fixable now) or the shape (needs redesign).

Last Updated: 2026-07-13
Author: Claude (with Marc Jones)
Related Issues: #713, #780, #821
See Also: DESIGN.md (the merge gate), docs/PROPER_NOUNS.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from klareco.parser import parse

TREEBANKS = [
    ('PRAGO  (in-corpus — leakage risk)',
     'data/external/ud_esperanto_prago/eo_prago-ud-test.conllu'),
    ('CAIRO  (HELD OUT — the honest ruler)',
     'data/external/ud_esperanto_cairo/eo_cairo-ud-test.conllu'),
]

SUBJ_DEPS = ('nsubj', 'nsubj:pass')
OBJ_DEPS = ('obj',)


def load_conllu(path: str) -> list[list[dict]]:
    sents, cur = [], []
    for line in open(path, encoding='utf-8'):
        if not line.strip():
            if cur:
                sents.append(cur)
                cur = []
            continue
        if line.startswith('#'):
            continue
        x = line.rstrip('\n').split('\t')
        if len(x) >= 8 and x[0].isdigit():
            cur.append({'id': int(x[0]), 'form': x[1], 'upos': x[3],
                        'head': int(x[6]), 'dep': x[7].split(':')[0]})
    if cur:
        sents.append(cur)
    return sents


def _kerno(node):
    if not isinstance(node, dict):
        return None
    return node.get('kerno', node)


def _surface(node) -> str:
    k = _kerno(node)
    return (k.get('plena_vorto') or '').lower() if k else ''


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(sents: list[list[dict]]) -> dict:
    counts = {'subj': [0, 0, 0], 'obj': [0, 0, 0]}      # tp, fp, fn
    wrong_pick: Counter = Counter()
    examples = defaultdict(list)
    n_gold_subj = 0
    n_sent_with_subj = 0

    for s in sents:
        text = ' '.join(t['form'] for t in s)
        upos_of = {t['form'].lower(): t['upos'] for t in s}
        gold = {
            'subj': {t['form'].lower() for t in s if t['dep'] in SUBJ_DEPS},
            'obj': {t['form'].lower() for t in s if t['dep'] in OBJ_DEPS},
        }
        n_gold_subj += len(gold['subj'])
        n_sent_with_subj += 1 if gold['subj'] else 0

        try:
            ast = parse(text)
        except Exception:
            continue

        # Walk the TREE: every clause contributes its own subject and object.
        # Falls back to the legacy top-level slots when `propozicioj` is absent,
        # so this script also scores the old single-slot AST and the two are
        # directly comparable.
        got = {'subj': set(), 'obj': set()}
        frames = ast.get('propozicioj') or [ast]
        for frame in frames:
            for key, slot in (('subj', 'subjekto'), ('obj', 'objekto')):
                sf = _surface(frame.get(slot))
                if sf:
                    got[key].add(sf)

        for key in ('subj', 'obj'):
            c = counts[key]
            c[0] += len(got[key] & gold[key])
            c[1] += len(got[key] - gold[key])
            c[2] += len(gold[key] - got[key])

        for sf in got['subj'] - gold['subj']:
            tag = upos_of.get(sf, '?')
            wrong_pick[tag] += 1
            if len(examples[tag]) < 3:
                examples[tag].append((sf, text[:52]))

    # The SHAPE ceiling: one subjekto slot can recover at most one subject per
    # sentence, so recall can never exceed (sentences that have any subject) /
    # (total gold subjects).
    ceiling = n_sent_with_subj / n_gold_subj if n_gold_subj else 0.0

    out = {'n_sentences': len(sents), 'n_gold_subjects': n_gold_subj,
           'subjects_per_sentence': round(n_gold_subj / len(sents), 2) if sents else 0,
           'single_slot_recall_ceiling': round(ceiling, 3),
           'wrong_pick': dict(wrong_pick), 'examples': dict(examples)}
    for key in ('subj', 'obj'):
        p, r, f = prf(*counts[key])
        out[key] = {'precision': round(p, 4), 'recall': round(r, 4),
                    'f1': round(f, 4), 'tp': counts[key][0],
                    'fp': counts[key][1], 'fn': counts[key][2]}
    return out


# What we wrongly picked -> whether it is a RULE bug or the SHAPE.
_DIAGNOSIS = {
    'ADV':   'an ADVERB can never be a subject — RULE BUG',
    'DET':   'a DETERMINER modifies; the noun heads — RULE BUG',
    'ADJ':   'adjective — the agreement pass should catch it — RULE BUG',
    'VERB':  'participle taken as subject — RULE BUG',
    'NOUN':  'usually a VERBLESS HEADING — there is no subject to find — RULE BUG',
    'PROPN': 'usually a VERBLESS HEADING — RULE BUG',
    'NUM':   'numbering in a heading — RULE BUG',
}


def report(name: str, r: dict) -> None:
    print(f'\n  {name}')
    print(f'  {"-" * len(name)}')
    for key, label in (('subj', 'SUBJECT (nsubj)'), ('obj', 'OBJECT  (obj)')):
        d = r[key]
        print(f'    {label:16s} P {d["precision"]:6.1%}  R {d["recall"]:6.1%}  '
              f'F1 {d["f1"]:6.1%}   (tp={d["tp"]} fp={d["fp"]} fn={d["fn"]})')
    ceiling = r['single_slot_recall_ceiling']
    print(f'\n    {r["subjects_per_sentence"]} subjects/sentence  ->  a SINGLE-SLOT '
          f'AST could never exceed {ceiling:.1%} recall')
    if r['subj']['recall'] > ceiling:
        print(f'    we are at {r["subj"]["recall"]:.1%} — PAST that ceiling, because the AST is '
              f'now a TREE\n    (one predicate-argument frame per CLAUSE, not one per sentence)')
    elif r['subj']['recall'] and ceiling:
        print(f'    we are at {r["subj"]["recall"] / ceiling:.0%} of what a flat record allows')
    if r['wrong_pick']:
        print('\n    precision errors, by what we WRONGLY picked:')
        tot = sum(r['wrong_pick'].values())
        for tag, c in sorted(r['wrong_pick'].items(), key=lambda x: -x[1]):
            why = _DIAGNOSIS.get(tag, '')
            print(f'      {tag:6s} {c:3d} ({c / tot:4.0%})  {why}')


def main() -> int:
    ap = argparse.ArgumentParser(description='Role-labelling benchmark')
    ap.add_argument('--json', help='write results here')
    args = ap.parse_args()

    print('  ROLE LABELLING — the field the retriever actually reads')
    print('  (POS is at 94.5% scheme-adjusted; this is what was never measured)')

    results = {}
    for name, path in TREEBANKS:
        if not Path(path).exists():
            print(f'\n  MISSING: {path} — run scripts/acquire/acquire_ud_prago.sh')
            continue
        r = evaluate(load_conllu(path))
        results[name.split()[0]] = r
        report(name, r)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                   encoding='utf-8')
        print(f'\n  wrote {args.json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
