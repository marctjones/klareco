#!/usr/bin/env python3
"""
Answerability check: attach a REAL corpus sentence that answers each question. (#737)

VERSION: v1.0
COMPATIBLE WITH: trivia_bank.jsonl candidates, whoosh_v2, DuckDB store, `claude` CLI
DEPENDENCIES: duckdb, whoosh, klareco.rag.duckdb_retriever (_content_terms), claude CLI
STAGE: Evaluation / test-set construction

Description:
    The coverage check in build_trivia_bank.py only asks "does the answer WORD
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
    python scripts/eval/finalize_trivia_gold.py \
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

_PROMPT = """You are verifying an Esperanto QA pair against real corpus sentences.
QUESTION: {q}
EXPECTED ANSWER: {a}

Below are candidate corpus sentences. Identify which one, if ANY, actually ANSWERS
the question — i.e. the sentence itself asserts that the answer is "{a}" (not merely
mentions the word). Use ONLY the sentence text.

CANDIDATES:
{cands}

Reply ONLY JSON: {{"answering_index": <1-based int, or 0 if none answers>, "reason": "..."}}"""


def judge_answering(question: str, answer: str, cands: list, timeout: int = 120):
    listing = '\n'.join(f'{i+1}. {t}' for i, (_sid, t) in enumerate(cands))
    prompt = _PROMPT.format(q=question, a=answer, cands=listing)
    try:
        r = subprocess.run(['claude', '-p'], input=prompt, text=True,
                           capture_output=True, timeout=timeout)
        i, j = r.stdout.find('{'), r.stdout.rfind('}')
        v = json.loads(r.stdout[i:j + 1]) if i != -1 and j > i else {}
    except Exception:
        v = {}
    idx = v.get('answering_index')
    if isinstance(idx, int) and 1 <= idx <= len(cands):
        return cands[idx - 1], v.get('reason', '')
    return None, v.get('reason', '')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--input', default='data/staging/trivia_bank.jsonl')
    ap.add_argument('--out', default='data/test_sets/qa_gold_v1.jsonl')
    ap.add_argument('--top-k', type=int, default=6)
    ap.add_argument('--only-verdict', default='measurable',
                    help="process only candidates with this verdict (or 'any')")
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
    print(f'  {len(cands_in)} candidate questions to verify\n')

    gold, no_answer = [], 0
    with ix.searcher() as srch:
        for d in cands_in:
            q, a = d['eo_question'], d['eo_answer']
            terms = _content_terms(q.lower())
            if not terms:
                no_answer += 1; continue
            hits = srch.search(qp.parse(' OR '.join(terms)), limit=args.top_k)
            cands = [(int(h['id']), h['text']) for h in hits]
            if not cands:
                no_answer += 1; continue
            best, reason = judge_answering(q, a, cands)
            if not best:
                no_answer += 1
                print(f'  ✗ {q[:55]}  (no answering sentence)', flush=True)
                continue
            sid, text = best
            gold.append({
                'id': f'gold-{sid}', 'question': q, 'question_type': 'KIU',
                'expected_answer': a, 'expected_keywords': [a],
                'source_sentence_id': str(sid), 'source_sentence_text': text,
                'source': d.get('source', 'opentdb.com'),
                'en_question': d.get('en_question'), 'category': d.get('category'),
                'answerability_reason': reason[:200],
            })
            print(f'  ✓ {q[:55]}  -> sid {sid}', flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + '\n')
    print(f'\n  ✓ {len(gold)} GOLD pairs (real answering sentence + sid) -> {args.out}')
    print(f'    dropped (no answering sentence): {no_answer}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
