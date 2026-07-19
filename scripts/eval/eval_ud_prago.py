#!/usr/bin/env python3
"""
UD_Esperanto-Prago Parser Accuracy Harness

VERSION: v2.1
COMPATIBLE WITH: Klareco parser (klareco.parser)
DEPENDENCIES: data/external/ud_esperanto_prago/eo_prago-ud-test.conllu
STAGE: Evaluation

Runs Klareco's deterministic parser over the Universal Dependencies Esperanto
Prago treebank (CC-BY-SA 4.0, linguist-curated gold standard) and reports
per-token accuracy on:

  - Part of speech (UD UPOS vs. klareco vortspeco)
  - Lemma (UD lemma vs. klareco radiko)
  - Proper noun detection (UD PROPN tag vs. klareco propra_nomo flag)

This is the first time the parser meets external gold-standard data. We expect
some mismatch on edge cases (PROPN classification of foreign-looking words,
DET vs. PRON distinctions) and want a real accuracy number rather than the
implicit "parser is deterministic so it must be correct" assumption.

Usage:
    python scripts/eval/eval_ud_prago.py
    python scripts/eval/eval_ud_prago.py --conllu PATH
    python scripts/eval/eval_ud_prago.py --limit 50

Last Updated: 2026-05-14
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from klareco.parser import parse


# Map UD universal POS tags → klareco's `vortspeco` strings.
# klareco uses Esperanto terms; UD uses English abbreviations.
UD_TO_KLARECO_POS = {
    'NOUN': 'substantivo',
    'PROPN': 'propra_nomo',
    'VERB': 'verbo',
    'AUX': 'verbo',       # 'esti' as copula — klareco doesn't split AUX
    'ADJ': 'adjektivo',
    'ADV': 'adverbo',
    'PRON': 'pronomo',
    'DET': 'artikolo',    # la, lia, mia — klareco treats most as articles
    'ADP': 'prepozicio',
    'CCONJ': 'konjunkcio',
    'SCONJ': 'konjunkcio',
    'NUM': 'numero',
    'PART': 'particle',   # ne, ĉu — klareco may classify differently
    'INTJ': 'interjekcio',
    'PUNCT': 'punkto',
    'SYM': 'simbolo',
    'X': 'nekonata',
}


class UDToken(NamedTuple):
    word: str
    lemma: str
    upos: str
    feats: str
    head: int
    deprel: str


def iter_sentences(path: Path) -> Iterator[tuple[str, list[UDToken]]]:
    """Yield (text, tokens) pairs from a CoNLL-U file."""
    text = ''
    tokens: list[UDToken] = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('# text = '):
                text = line[len('# text = '):]
            elif line.startswith('#'):
                continue
            elif not line:
                if tokens:
                    yield text, tokens
                text, tokens = '', []
            else:
                fields = line.split('\t')
                if len(fields) < 8:
                    continue
                idx_str = fields[0]
                # Skip multi-word token ranges like "1-2"
                if '-' in idx_str or '.' in idx_str:
                    continue
                try:
                    head = int(fields[6])
                except ValueError:
                    head = 0
                tokens.append(UDToken(
                    word=fields[1],
                    lemma=fields[2],
                    upos=fields[3],
                    feats=fields[5],
                    head=head,
                    deprel=fields[7],
                ))
    if tokens:
        yield text, tokens


def flatten_words(ast: dict) -> list[dict]:
    """Flatten klareco AST → list of word dicts in surface order."""
    out: list[dict] = []
    def walk(node):
        if not isinstance(node, dict):
            return
        if node.get('tipo') == 'vorto':
            out.append(node)
            return
        # vortgrupo
        kerno = node.get('kerno')
        if kerno:
            walk(kerno)
        for child in node.get('priskriboj', []) or []:
            walk(child)
    if not ast or not isinstance(ast, dict):
        return out
    if ast.get('subjekto'):
        walk(ast['subjekto'])
    if ast.get('verbo'):
        walk(ast['verbo'])
    if ast.get('objekto'):
        walk(ast['objekto'])
    for other in ast.get('aliaj', []) or []:
        walk(other)
    return out


def pos_matches(ud_upos: str, klareco_vortspeco: str) -> bool:
    expected = UD_TO_KLARECO_POS.get(ud_upos)
    if expected is None:
        return False
    return expected == klareco_vortspeco


# Known annotation-scheme differences (defensible on both sides, not parser
# bugs). Esperanto possessives are adjectival in form (nia/nian); correlatives
# are a finer Esperanto category than UD DET/PRON; participles are adjectival;
# ĉi/plu-type particles vs ADV.
SCHEME_DIFFS = {
    ('PRON', 'adjektivo'), ('PRON', 'korelativo'),
    ('DET', 'korelativo'), ('DET', 'adjektivo'),
    ('VERB', 'adjektivo'), ('ADV', 'partiklo'),
    ('ADV', 'korelativo'), ('SCONJ', 'konjunkcio'),
    ('AUX', 'verbo'),
}


def evaluate(conllu: Path, limit: int | None = None,
             show_mismatches: int = 0) -> dict:
    """Run the parser over a UD CoNLL-U file and return POS/lemma/PROPN metrics.

    Importable single source of truth (used by main() here and by the pytest
    regression guard tests/test_parser_ud_accuracy.py). Returns raw counts plus
    derived rates so callers can assert on floors without re-deriving.
    """
    sentences = 0
    parse_errors = 0
    total_tokens = 0
    aligned_tokens = 0
    pos_correct = 0
    lemma_correct = 0
    propn_tp = propn_fp = propn_fn = 0
    upos_confusion: Counter[tuple[str, str]] = Counter()
    sample_mismatches: list[tuple[str, str, str, str]] = []

    for sent_idx, (text, ud_tokens) in enumerate(iter_sentences(conllu)):
        if limit and sent_idx >= limit:
            break
        sentences += 1
        try:
            ast = parse(text)
        except Exception:
            parse_errors += 1
            continue
        klareco_words = flatten_words(ast)

        # Filter UD tokens to content (drop PUNCT for fairer comparison —
        # klareco's parser strips punctuation rather than tagging it).
        ud_content = [t for t in ud_tokens if t.upos != 'PUNCT']
        total_tokens += len(ud_content)

        # Align by surface form; skip tokens we can't match (token
        # segmentation may differ).
        klareco_by_word: dict[str, list[dict]] = {}
        for w in klareco_words:
            surface = (w.get('plena_vorto') or w.get('vorto') or '').lower()
            klareco_by_word.setdefault(surface, []).append(w)

        for ud_tok in ud_content:
            kw = klareco_by_word.get(ud_tok.word.lower())
            if not kw:
                continue
            klareco_tok = kw.pop(0)
            aligned_tokens += 1

            if pos_matches(ud_tok.upos, klareco_tok.get('vortspeco', '')):
                pos_correct += 1
            else:
                upos_confusion[(ud_tok.upos, klareco_tok.get('vortspeco', '?'))] += 1
                if len(sample_mismatches) < show_mismatches:
                    sample_mismatches.append((
                        ud_tok.word, ud_tok.upos,
                        klareco_tok.get('vortspeco', '?'),
                        text[:80] + ('…' if len(text) > 80 else ''),
                    ))

            # Lemma vs. radiko — exact match
            if (klareco_tok.get('radiko') or '').lower() == ud_tok.lemma.lower():
                lemma_correct += 1

            # PROPN precision/recall on klareco's propra_nomo flag
            is_propn_ud = ud_tok.upos == 'PROPN'
            is_propn_kl = klareco_tok.get('vortspeco') == 'propra_nomo'
            if is_propn_ud and is_propn_kl:
                propn_tp += 1
            elif is_propn_kl and not is_propn_ud:
                propn_fp += 1
            elif is_propn_ud and not is_propn_kl:
                propn_fn += 1

    scheme_diff_count = sum(n for (u, k), n in upos_confusion.items()
                            if (u, k) in SCHEME_DIFFS)
    adjusted_correct = pos_correct + scheme_diff_count
    al = max(aligned_tokens, 1)
    propn_p = propn_tp / max(propn_tp + propn_fp, 1)
    propn_r = propn_tp / max(propn_tp + propn_fn, 1)
    return {
        'sentences': sentences, 'parse_errors': parse_errors,
        'total_tokens': total_tokens, 'aligned_tokens': aligned_tokens,
        'pos_correct': pos_correct, 'lemma_correct': lemma_correct,
        'adjusted_correct': adjusted_correct,
        'scheme_diff_count': scheme_diff_count,
        'propn_tp': propn_tp, 'propn_fp': propn_fp, 'propn_fn': propn_fn,
        'upos_confusion': upos_confusion, 'sample_mismatches': sample_mismatches,
        # derived rates (fractions in [0,1])
        'pos_strict': pos_correct / al,
        'pos_adjusted': adjusted_correct / al,
        'lemma_rate': lemma_correct / al,
        'align_rate': aligned_tokens / max(total_tokens, 1),
        'propn_p': propn_p, 'propn_r': propn_r,
        'propn_f1': 2 * propn_p * propn_r / max(propn_p + propn_r, 1e-9),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--conllu',
                   default='data/external/ud_esperanto_prago/eo_prago-ud-test.conllu')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--show-mismatches', type=int, default=0,
                   help='Print N sample POS mismatches for inspection')
    args = p.parse_args()

    conllu = Path(args.conllu)
    if not conllu.exists():
        print(f'CoNLL-U file not found: {conllu}', file=sys.stderr)
        return 1

    r = evaluate(conllu, limit=args.limit, show_mismatches=args.show_mismatches)
    sentences = r['sentences']
    parse_errors = r['parse_errors']
    total_tokens = r['total_tokens']
    aligned_tokens = r['aligned_tokens']
    pos_correct = r['pos_correct']
    lemma_correct = r['lemma_correct']
    adjusted_correct = r['adjusted_correct']
    scheme_diff_count = r['scheme_diff_count']
    propn_tp, propn_fp, propn_fn = r['propn_tp'], r['propn_fp'], r['propn_fn']
    upos_confusion = r['upos_confusion']
    sample_mismatches = r['sample_mismatches']

    print(f'Sentences:           {sentences}')
    print(f'Parse errors:        {parse_errors}')
    print(f'UD content tokens:   {total_tokens}')
    print(f'Aligned tokens:      {aligned_tokens}  ({aligned_tokens/max(total_tokens,1)*100:.1f}%)')
    print()
    print(f'POS accuracy (strict):   {pos_correct}/{aligned_tokens}  '
          f'({pos_correct/max(aligned_tokens,1)*100:.1f}%)')
    print(f'POS accuracy (scheme-adjusted): {adjusted_correct}/{aligned_tokens}  '
          f'({adjusted_correct/max(aligned_tokens,1)*100:.1f}%)  '
          f'[+{scheme_diff_count} known UD-vs-Esperanto scheme diffs]')
    print(f'Lemma vs radiko:     {lemma_correct}/{aligned_tokens}  '
          f'({lemma_correct/max(aligned_tokens,1)*100:.1f}%)  '
          f'[NOTE: UD lemma=dictionary form, klareco radiko=morpheme root; '
          f'they only match for morphologically simple words — low % expected]')
    print()
    print('Proper-noun detection (vs UD PROPN):')
    propn_total_ud = propn_tp + propn_fn
    propn_total_kl = propn_tp + propn_fp
    print(f'  TP={propn_tp}  FP={propn_fp}  FN={propn_fn}')
    prec = propn_tp / max(propn_total_kl, 1)
    rec = propn_tp / max(propn_total_ud, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f'  Precision: {prec*100:.1f}%   Recall: {rec*100:.1f}%   F1: {f1*100:.1f}%')
    print()
    print('Top 10 UPOS confusions (ud_upos → klareco_vortspeco):')
    for (u, k), n in upos_confusion.most_common(10):
        print(f'  {u:8s} → {k:14s} {n:4d}')

    if sample_mismatches:
        print()
        print(f'Sample mismatches:')
        for word, ud, kl, ctx in sample_mismatches:
            print(f'  "{word}" UD={ud} klareco={kl}  | {ctx}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
