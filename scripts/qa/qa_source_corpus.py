#!/usr/bin/env python3
"""
Engine B: corpus sentence -> Claude natural question -> verified gold. (#843)

VERSION: v1.0
COMPATIBLE WITH: v2.3+ store, whoosh_v2, `claude` CLI
DEPENDENCIES: duckdb, whoosh, klareco.parser, klareco.rag.duckdb_retriever,
              klareco.eval.qa_judge, klareco.eval.qa_schema
STAGE: Evaluation / test-set construction

Description:
    The HIGH-YIELD, circularity-free volume engine (complements Engine A / OpenTDB,
    which is diverse but corpus-coverage-limited). Sample a real corpus sentence, have
    Claude write a NATURAL Esperanto question it answers, keep that sentence as gold.

      - Coverage is GUARANTEED (100%): the question is built from a corpus sentence,
        so a real answering sentence always exists (its `sid` is the provenance).
      - NOT parser-circular: the question comes from Claude READING the sentence, not
        from parser (subj,verb,obj) extraction — so it does not flatter structural
        rerankers (unlike the retired parser-derived builders).

    HONEST CAVEAT: question generation is GENERATION (Claude writes the surface), so it
    is fluent rather than corpus-shaped. Standard for QA sets (SQuAD-style) and accepted
    here because it buys the circularity break; quality is enforced downstream:
      1. parser grammar/pureness gate,
      2. qa_schema.validate (answer verbatim, answer-not-in-question, interrogative),
      3. difficulty band from production-path BM25,
      4. qa_judge (accuracy + naturalness, grounded in the source sentence).

Pipeline Position:
    sentences (proper-noun-subject clauses) --[THIS]--> qa_gold candidates (schema-valid)

Usage:
    python scripts/qa/qa_source_corpus.py --n 100 --out data/staging/corpus_gold.jsonl

Last Updated: 2026-07-17
Related Issues: #736, #737, #842, #843
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

from klareco.parser import parse
from klareco.rag.duckdb_retriever import _content_terms
from klareco.eval.qa_judge import judge_batch
from klareco.eval.qa_schema import validate, band_for

DB = 'data/indexes/duckdb_store.db'
WHOOSH = 'data/indexes/whoosh_v2'

_GEN_PROMPT = """You are building an Esperanto question-answering TEST SET from real corpus sentences.
For EACH sentence, write ONE natural Esperanto question that is answered SOLELY by that
sentence, and give the ANSWER as a span copied VERBATIM from the sentence.

Rules:
- question starts with a proper interrogative: Kiu/Kiun/Kio/Kion/Kie/Kiam/Kiom/Kial/Kies.
- the ANSWER must appear VERBATIM (exact substring) in the sentence.
- the question must NOT contain the answer.
- pure, grammatical Esperanto; a question a person would actually ask.
- if the sentence states no clear fact, set "skip": true.

Reply with ONLY a JSON array, one object per sentence:
[{"id":"...","question":"...","answer":"...","question_type":"KIU","skip":false}]

SENTENCES:
"""


def generate_batch(sents, timeout=240):
    prompt = _GEN_PROMPT + '\n'.join(
        json.dumps({'id': str(sid), 'sentence': text}, ensure_ascii=False)
        for sid, text in sents)
    try:
        r = subprocess.run(['claude', '-p'], input=prompt, text=True,
                           capture_output=True, timeout=timeout)
        i, j = r.stdout.find('['), r.stdout.rfind(']')
        arr = json.loads(r.stdout[i:j + 1]) if i != -1 and j > i else []
    except Exception:
        arr = []
    return {str(v['id']): v for v in arr if isinstance(v, dict) and 'id' in v}


def _grammar_ok(question: str) -> bool:
    try:
        ast = parse(question)
    except Exception:
        return False
    unknown = sum(1 for w in (ast.get('vortoj') or [])
                  if isinstance(w, dict)
                  and w.get('vortspeco') in ('substantivo', 'verbo', 'adjektivo', 'adverbo')
                  and w.get('analizstato') == 'unknown_root')
    return unknown <= 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--n', type=int, default=100, help='target gold count')
    ap.add_argument('--oversample', type=int, default=3)
    ap.add_argument('--batch', type=int, default=12)
    ap.add_argument('--out', default='data/staging/corpus_gold.jsonl')
    args = ap.parse_args()

    con = duckdb.connect(DB, read_only=True); con.execute('SET threads=6')
    ix = whoosh_index.open_dir(WHOOSH)
    qp = QueryParser('text', ix.schema, group=OrGroup)

    # factual material: proper-noun-subject clauses (a named entity doing something)
    rows = con.execute(f"""
        SELECT * FROM (
          SELECT DISTINCT s.sid, s.text
          FROM clauses c JOIN sentences s ON s.sid=c.sid
          WHERE c.subj_vortspeco='propra_nomo' AND length(s.text) BETWEEN 45 AND 170
        ) USING SAMPLE {args.n * args.oversample} ROWS (reservoir, 23)
    """).fetchall()
    print(f'  sampled {len(rows)} corpus sentences; generating questions…')

    drop = {'skip/blank': 0, 'grammar': 0, 'schema': 0, 'judge': 0}
    gold = []
    with ix.searcher() as srch:
        for i in range(0, len(rows), args.batch):
            if len(gold) >= args.n:
                break
            chunk = rows[i:i + args.batch]
            gen = generate_batch(chunk)
            cand = []
            for sid, text in chunk:
                v = gen.get(str(sid))
                if not v or v.get('skip') or not v.get('question') or not v.get('answer'):
                    drop['skip/blank'] += 1; continue
                q, a = v['question'].strip(), v['answer'].strip()
                if not _grammar_ok(q):
                    drop['grammar'] += 1; continue
                terms = _content_terms(q.lower())
                rank = None
                if terms:
                    hits = srch.search(qp.parse(' OR '.join(terms)), limit=200)
                    rank = next((k + 1 for k, h in enumerate(hits)
                                 if str(h['id']) == str(sid)), None)
                rec = {
                    'id': f'gold-{sid}', 'question': q, 'expected_answer': a,
                    'expected_keywords': [a], 'question_type': v.get('question_type', 'KIU'),
                    'source_sentence_id': str(sid), 'source_sentence_text': text,
                    'source': 'corpus', 'bm25_gold_rank': rank,
                    'difficulty_band': band_for(rank),
                }
                ok, _ = validate(rec)
                if not ok:
                    drop['schema'] += 1; continue
                cand.append(rec)
            # one judge pass over the batch's schema-valid candidates
            verds = {x['id']: x for x in judge_batch(cand)}
            for rec in cand:
                if verds.get(rec['id'], {}).get('keep'):
                    rec['verified'] = {'grammar': True, 'schema': True,
                                       'answerability': True, 'judge': True}
                    gold.append(rec)
                else:
                    drop['judge'] += 1
            print(f'  …{min(i+args.batch,len(rows))}/{len(rows)} sampled, {len(gold)} gold',
                  flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for g in gold[:args.n]:
            f.write(json.dumps(g, ensure_ascii=False) + '\n')
    print(f'\n  ✓ {min(len(gold),args.n)} gold -> {args.out}')
    print(f'    dropped: {drop}')
    import collections
    bands = collections.Counter(g['difficulty_band'] for g in gold[:args.n])
    print(f'    difficulty: {dict(bands)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
