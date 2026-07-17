#!/usr/bin/env python3
"""
Answerability check: attach a REAL corpus sentence that answers each question. (#737)

VERSION: v1.0
COMPATIBLE WITH: trivia_bank.jsonl candidates, whoosh_v2, DuckDB store, `claude` CLI
DEPENDENCIES: duckdb, whoosh, klareco.rag.duckdb_retriever (_content_terms), claude CLI
STAGE: Evaluation / test-set construction

Description:
    The coverage check in qa_gate.py only asks "does the answer WORD
    appear in the corpus" (a LIKE count). That is necessary but NOT sufficient: a
    sentence mentioning `Frago` does not assert that a strawberry is not a berry.
    This is the strong check the user asked for — for each candidate question it:

      1. RETRIEVES the top-K corpus sentences via the PRODUCTION query path
         (_content_terms BM25), the same retrieval the system actually uses.
      2. Has the Claude CLI judge which candidate, if any, ACTUALLY ANSWERS the
         question (asserts the answer), grounded in the sentence text only.
      3. Attaches that sentence as gold with its `source_sentence_id` — real
         provenance into the DuckDB corpus.

    Questions with no answering sentence are dropped (no_answering_sentence). The
    survivors are circularity-free gold: external question, corpus-grounded answer,
    verified by an independent judge, with a citable source sid.

Usage:
    python scripts/qa/qa_answerability.py \
        --input data/staging/trivia_bank.jsonl \
        --out   data/test_sets/qa_gold_v1.jsonl --top-k 6

Last Updated: 2026-07-17
Related Issues: #736, #737
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb
from whoosh import index as whoosh_index
from whoosh.qparser import OrGroup, QueryParser

from klareco.rag.duckdb_retriever import _content_terms

DB = 'data/indexes/duckdb_store.db'
WHOOSH = 'data/indexes/whoosh_v2'

from klareco.eval.qa_schema import band_for

_BATCH_PROMPT = """You are verifying Esperanto QA pairs against real corpus sentences.
For EACH item: given its QUESTION, EXPECTED ANSWER, and its own numbered CANDIDATE
sentences, identify which candidate (if any) actually ANSWERS the question — the
sentence itself asserts the answer (not merely mentions the word). Use only the text.

Reply with ONLY a JSON array, one object per item:
[{"id":"<id>","answering_index":<1-based int into THAT item's candidates, or 0 if none>}]

ITEMS:
"""

_QWORDS = {'kiu': 'KIU', 'kiun': 'KIUN', 'kio': 'KIO', 'kion': 'KION',
           'kie': 'KIE', 'kiam': 'KIAM', 'kiom': 'KIOM', 'kial': 'KIAL', 'kies': 'KIES'}


def _qtype(q: str) -> str:
    return _QWORDS.get((q.split() or [''])[0].strip('¿?').lower(), 'KIU')


def judge_answering_batch(items: list, timeout: int = 300) -> dict:
    """items: [{'id','question','answer','cands':[(sid,text),...]}].
    ONE claude call for the whole batch. Returns {id: answering_index}."""
    blocks = []
    for it in items:
        lst = '\n'.join(f'{i+1}. {t}' for i, (_s, t) in enumerate(it['cands']))
        blocks.append(f"--- id: {it['id']} ---\nQ: {it['question']}\n"
                      f"A: {it['answer']}\ncandidates:\n{lst}")
    prompt = _BATCH_PROMPT + '\n\n'.join(blocks)
    try:
        r = subprocess.run(['claude', '-p'], input=prompt, text=True,
                           capture_output=True, timeout=timeout)
        i, j = r.stdout.find('['), r.stdout.rfind(']')
        arr = json.loads(r.stdout[i:j + 1]) if i != -1 and j > i else []
    except Exception:
        arr = []
    return {str(v['id']): v.get('answering_index')
            for v in arr if isinstance(v, dict) and 'id' in v}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--input', default='data/staging/trivia_bank.jsonl')
    ap.add_argument('--out', default='data/test_sets/qa_gold_v1.jsonl')
    ap.add_argument('--review-out', default='data/staging/qa_needs_review.jsonl',
                    help='questions with no answering sentence found — KEPT for revisit')
    ap.add_argument('--judge-k', type=int, default=25,
                    help='answer-aware candidates shown to the judge (small window; '
                         'the answer-aware query floats the answering sentence up)')
    ap.add_argument('--rank-k', type=int, default=200,
                    help='question-ONLY retrieval depth used to record the TRUE '
                         'difficulty rank (never a filter — low ranks are kept)')
    ap.add_argument('--batch', type=int, default=8, help='questions per judge call')
    ap.add_argument('--workers', type=int, default=3,
                    help='concurrent judge calls (each = one claude process)')
    ap.add_argument('--only-verdict', default='any',
                    help="process candidates with this verdict ('any' = keep all)")
    args = ap.parse_args()

    con = duckdb.connect(DB, read_only=True)
    ix = whoosh_index.open_dir(WHOOSH)
    qp = QueryParser('text', ix.schema, group=OrGroup)

    cands_in = []
    with open(args.input, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if args.only_verdict != 'any' and d.get('verdict') != args.only_verdict:
                continue
            if d.get('eo_question') and d.get('eo_answer'):
                cands_in.append(d)
    print(f'  {len(cands_in)} candidate questions to verify (KEEP ALL policy)')

    # 1) Two retrievals per question (fast, no Claude):
    #    - ANSWER-AWARE (question + answer terms), small window -> what the judge scans
    #      (the answer terms float the answering sentence into this window if it exists).
    #    - QUESTION-ONLY, deep -> a sid->rank map giving the TRUE difficulty (the rank
    #      the retriever achieves WITHOUT knowing the answer). Never used to filter.
    items, no_terms = [], []
    with ix.searcher() as srch:
        for d in cands_in:
            q, a = d['eo_question'], d['eo_answer']
            qterms = _content_terms(q.lower())
            if not qterms:
                no_terms.append(d); continue
            aa = srch.search(qp.parse(' OR '.join(qterms + _content_terms(a.lower()))),
                             limit=args.judge_k)
            cands = [(int(h['id']), h['text']) for h in aa]
            qo = srch.search(qp.parse(' OR '.join(qterms)), limit=args.rank_k)
            qo_rank = {int(h['id']): i + 1 for i, h in enumerate(qo)}
            items.append({'id': f'q{len(items)}', 'question': q, 'answer': a,
                          'cands': cands, 'qo_rank': qo_rank, 'meta': d})

    # 2) BATCH + parallel judge over the small answer-aware window
    batches = [items[i:i + args.batch] for i in range(0, len(items), args.batch)]
    print(f'  judging {len(items)} questions in {len(batches)} batches '
          f'(batch={args.batch}, workers={args.workers}, judge-k={args.judge_k})\n')

    def run_batch(b):
        verd = judge_answering_batch(b)
        for it in b:
            idx = verd.get(it['id'])
            it['idx'] = idx if (isinstance(idx, int) and 1 <= idx <= len(it['cands'])) else None
        return b

    judged = []
    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for b in ex.map(run_batch, batches):
                judged.extend(b)
                print(f'    …{sum(1 for x in judged if x.get("idx"))} answered '
                      f'/ {len(judged)} judged', flush=True)
    else:
        for bi, b in enumerate(batches, 1):
            judged.extend(run_batch(b))
            print(f'    batch {bi}/{len(batches)}', flush=True)

    gold, review = [], []
    for it in judged + [{'idx': None, 'cands': [], 'question': d['eo_question'],
                         'answer': d['eo_answer'], 'meta': d} for d in no_terms]:
        d = it['meta']
        if it.get('idx'):                       # an answering sentence exists
            sid, text = it['cands'][it['idx'] - 1]
            rank = it['qo_rank'].get(sid)       # TRUE difficulty (question-only)
            # rank is None => the answer IS in the corpus but the question-only
            # retriever ranks it beyond top-{rank-k}: the hardest, most valuable
            # kind (first-stage retrieval fails). Keep it, banded 'deep'.
            band = band_for(rank) if rank else 'deep'
            gold.append({
                'id': f'gold-{sid}', 'question': it['question'],
                'question_type': _qtype(it['question']),
                'expected_answer': it['answer'], 'expected_keywords': [it['answer']],
                'source_sentence_id': str(sid), 'source_sentence_text': text,
                'source': 'opentdb', 'bm25_gold_rank': rank, 'difficulty_band': band,
                'en_question': d.get('en_question'), 'category': d.get('category'),
            })
        else:                                    # KEEP for revisit — do not discard
            review.append({
                'question': it['question'], 'expected_answer': it['answer'],
                'question_type': _qtype(it['question']),
                'status': 'needs_review',
                # distinguishes "retriever failed" from "answer likely not stated":
                'gate_verdict': d.get('verdict'),
                'source': 'opentdb', 'en_question': d.get('en_question'),
                'category': d.get('category'),
            })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + '\n')
    Path(args.review_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.review_out, 'w', encoding='utf-8') as f:
        for r in review:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    import collections
    bands = collections.Counter(g['difficulty_band'] for g in gold)
    print(f'\n  ✓ {len(gold)} GOLD (answering sentence found, ANY rank) -> {args.out}')
    print(f'    difficulty: {dict(bands)}')
    print(f'  ⌛ {len(review)} NEEDS-REVIEW (no answering sentence found) -> {args.review_out}')
    rev_by = collections.Counter(r['gate_verdict'] for r in review)
    print(f'    by gate verdict: {dict(rev_by)}  '
          f'(measurable here = retriever-fail or unstated; corpus_gap = answer word absent)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
