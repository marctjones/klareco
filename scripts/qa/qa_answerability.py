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
    ap.add_argument('--top-k', type=int, default=6)
    ap.add_argument('--batch', type=int, default=8, help='questions per judge call')
    ap.add_argument('--workers', type=int, default=1,
                    help='concurrent judge calls (each = one claude process)')
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
    print(f'  {len(cands_in)} candidate questions to verify')

    # 1) retrieve candidates for every question up front (fast, no Claude)
    items, no_answer = [], 0
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
            items.append({'id': f'q{len(items)}', 'question': q, 'answer': a,
                          'cands': cands, 'meta': d})

    # 2) BATCH the judge (many questions per claude call), optionally in parallel
    batches = [items[i:i + args.batch] for i in range(0, len(items), args.batch)]
    print(f'  judging {len(items)} questions in {len(batches)} batches '
          f'(batch={args.batch}, workers={args.workers})\n')

    def run_batch(b):
        verd = judge_answering_batch(b)
        out = []
        for it in b:
            idx = verd.get(it['id'])
            if isinstance(idx, int) and 1 <= idx <= len(it['cands']):
                out.append((it, idx))
        return out

    results = []
    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(run_batch, batches):
                results.extend(r)
                print(f'    …{len(results)} gold', flush=True)
    else:
        for bi, b in enumerate(batches, 1):
            results.extend(run_batch(b))
            print(f'    batch {bi}/{len(batches)}  {len(results)} gold', flush=True)

    gold = []
    for it, idx in results:
        sid, text = it['cands'][idx - 1]
        d = it['meta']
        gold.append({
            'id': f'gold-{sid}', 'question': it['question'],
            'question_type': _qtype(it['question']),
            'expected_answer': it['answer'], 'expected_keywords': [it['answer']],
            'source_sentence_id': str(sid), 'source_sentence_text': text,
            'source': 'opentdb', 'bm25_gold_rank': idx, 'difficulty_band': band_for(idx),
            'en_question': d.get('en_question'), 'category': d.get('category'),
        })
    no_answer += len(items) - len(results)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + '\n')
    print(f'\n  ✓ {len(gold)} GOLD pairs (real answering sentence + sid) -> {args.out}')
    print(f'    dropped (no answering sentence): {no_answer}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
