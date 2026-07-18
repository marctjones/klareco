#!/usr/bin/env python3
"""
RESEARCH SPIKE (#854): why does retrieval miss the 506 deep-band answers?

VERSION: v1.0
COMPATIBLE WITH: qa_gold_v2.jsonl (difficulty_band records), `claude` CLI
STAGE: Evaluation / research spike — produces a DECISION TABLE, not a capability

Description:
    Every deep-band question has a KNOWN answering sentence (source_sentence_id)
    that question-only BM25 misses or buries past rank 50. This script explains WHY,
    per question, and aggregates a category table that decides which retrieval fixes
    get built (#855 aliases, #856 definitional, or the learned-synonymy residue).

    Two layers:
      1. DETERMINISTIC features — production content terms vs gold-sentence text:
         exact-normalized term hits, root-prefix hits (inflection-tolerant), the
         definitional-question signal, unmatched-proper-noun signal.
      2. LLM classification (batched claude -p), grounded in the pair + features,
         into: alias_variant | definitional | synonym_paraphrase | low_overlap |
         common_terms_competition | other.

Outputs:
    data/staging/deep_taxonomy.jsonl  (per-question label + features)
    stdout: the category x count decision table + examples.

Usage:
    python scripts/qa/qa_deep_taxonomy.py [--limit N] [--batch 20]

Last Updated: 2026-07-17
Related Issues: #854, #855, #856, #857
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from klareco.rag.duckdb_retriever import _content_terms

_EO_FOLD = str.maketrans({c: r for c, r in zip('ĉĝĥĵŝŭĈĜĤĴŜŬ', 'cghjsucghjsu')})


def norm(t: str) -> str:
    t = t.translate(_EO_FOLD).lower()
    return ''.join(c for c in unicodedata.normalize('NFD', t)
                   if not unicodedata.combining(c))


def features(question: str, gold_text: str) -> dict:
    qterms = _content_terms(question.lower())
    gnorm = norm(gold_text)
    gwords = set(gnorm.split())
    exact = [t for t in qterms if norm(t) in gnorm]
    # root-prefix match: catches inflection (fondiĝis~fondita) without a parser call
    def root_hit(t):
        r = norm(t)[:max(4, len(norm(t)) - 3)]
        return any(w.startswith(r) for w in gwords)
    rootm = [t for t in qterms if t not in exact and root_hit(t)]
    ql = question.lower()
    return {
        'n_qterms': len(qterms),
        'n_exact': len(exact),
        'n_root': len(rootm),
        'unmatched': [t for t in qterms if t not in exact and t not in rootm],
        'definitional_signal': (ql.startswith(('kio ', 'kion '))
                                or 'signifas' in ql or 'estas la' in ql),
    }


_PROMPT = """You are classifying WHY a BM25 retriever (bag of content words) misses the
gold sentence for each Esperanto question. For each item you get the question, the
expected answer, the GOLD sentence that answers it, and term-overlap features
(exact = question terms found verbatim in gold; root = matched only by word-root;
unmatched = question terms absent from gold).

Categories (pick ONE best):
  alias_variant     a name/abbreviation/spelling variant blocks matching
                    (GMT vs Greniĉo, Kanado vs Kanada, x-system, translated titles)
  definitional      question asks what X is/means; gold states "X estas ..." with
                    little other overlap
  synonym_paraphrase gold expresses the relation with different words (synonyms,
                    paraphrase) — no variant of the SAME word would help
  low_overlap       question simply shares too few content words with gold
                    (question wording adds words gold never had)
  common_terms_competition  the shared terms ARE present but are common words —
                    thousands of sentences match them equally well
  other             none of the above fits

Reply ONLY a JSON array: [{"id":"...","category":"...","note":"<8 words"}]

ITEMS:
"""


def classify_batch(items, timeout=240):
    payload = []
    for it in items:
        payload.append(json.dumps({
            'id': it['id'], 'question': it['question'],
            'answer': it['expected_answer'], 'gold': it['source_sentence_text'],
            'exact_hits': it['feat']['n_exact'], 'root_hits': it['feat']['n_root'],
            'unmatched_terms': it['feat']['unmatched'][:8],
        }, ensure_ascii=False))
    try:
        r = subprocess.run(['claude', '-p'], input=_PROMPT + '\n'.join(payload),
                           text=True, capture_output=True, timeout=timeout)
        i, j = r.stdout.find('['), r.stdout.rfind(']')
        arr = json.loads(r.stdout[i:j + 1]) if i != -1 and j > i else []
    except Exception:
        arr = []
    return {str(v['id']): v for v in arr if isinstance(v, dict) and 'id' in v}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--input', default='data/test_sets/qa_gold_v2.jsonl')
    ap.add_argument('--out', default='data/staging/deep_taxonomy.jsonl')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--batch', type=int, default=20)
    args = ap.parse_args()

    deep = [json.loads(l) for l in open(args.input, encoding='utf-8')
            if l.strip() and json.loads(l).get('difficulty_band') == 'deep']
    if args.limit:
        deep = deep[:args.limit]
    print(f'  {len(deep)} deep-band questions to classify\n')

    for d in deep:
        d['feat'] = features(d['question'], d['source_sentence_text'])

    labeled = []
    for i in range(0, len(deep), args.batch):
        chunk = deep[i:i + args.batch]
        verd = classify_batch(chunk)
        for d in chunk:
            v = verd.get(str(d['id']), {})
            d['category'] = v.get('category', 'unclassified')
            d['category_note'] = v.get('note', '')
            labeled.append(d)
        done = collections.Counter(x['category'] for x in labeled)
        print(f'  …{len(labeled)}/{len(deep)}  {dict(done)}', flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for d in labeled:
            row = {k: d[k] for k in ('id', 'question', 'expected_answer',
                                     'source_sentence_id', 'category',
                                     'category_note')}
            row['features'] = d['feat']
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    print('\n  ══ DECISION TABLE — deep-band misses by cause ══')
    cats = collections.Counter(d['category'] for d in labeled)
    for c, n in cats.most_common():
        print(f'    {c:26s} {n:4d}  ({n/len(labeled):.0%})')
    print('\n  deterministic cross-check (mean term hits per category):')
    for c in cats:
        rows = [d for d in labeled if d['category'] == c]
        me = sum(d['feat']['n_exact'] for d in rows) / len(rows)
        mr = sum(d['feat']['n_root'] for d in rows) / len(rows)
        print(f'    {c:26s} exact={me:.1f} root={mr:.1f}')
    print('\n  examples:')
    seen = set()
    for d in labeled:
        if d['category'] in seen:
            continue
        seen.add(d['category'])
        print(f'    [{d["category"]}] {d["question"][:60]} -> {d["expected_answer"][:30]}')
        print(f'       gold: “{d["source_sentence_text"][:70]}”  ({d["category_note"]})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
