#!/usr/bin/env python3
"""
Build ONE large, stratified, gate-passed, LLM-judged QA corpus. (#736/#737)

VERSION: v1.0
COMPATIBLE WITH: v2.3+ store (clauses + dependency_arcs), whoosh_v2, `claude` CLI
DEPENDENCIES: duckdb, whoosh, klareco.parser, klareco.rag.duckdb_retriever,
              klareco.eval.qa_judge (claude CLI)
STAGE: Evaluation / test-set construction

Description:
    Replaces the five small fragmented QA sets with ONE corpus. Division of labour:

      GENERATE   subject-gap "Kiu <verb> <object-phrase>?" from real corpus trees
                 (proper-noun subject = the answer; object phrase = the object's
                 SUBTREE, so modifiers come along naturally). Accuracy + presence
                 are automatic — the answer IS a role of the source sentence.
      GATE       DETERMINISTIC only: grammar (the question re-parses cleanly),
                 pureness (every content root is known to the lexicon), presence
                 (answer verbatim), not-negated. The parser and SQL are better
                 judges of these than any LLM.
      DIFFICULTY production-path BM25 rank (via _content_terms) -> a band. This is
                 the stratifier: the hard band IS the reranker-discrimination set.
      JUDGE      the Claude CLI, on survivors only, for the ONE thing determinism
                 cannot check: is the answer correct & unambiguous GIVEN THE SOURCE,
                 and is the Esperanto natural. A filter, never a generator.
      STRATIFY   one file, stratified by (gap_shape, difficulty band); report N per
                 stratum against the power targets (need ~185 for a 0.03 MRR delta).

    Parser data (treebank_sample) is a DIFFERENT deliverable and is never touched.

Usage:
    python scripts/eval/build_qa_corpus.py --target 500 --out data/test_sets/qa_corpus_v1.jsonl
    python scripts/eval/build_qa_corpus.py --target 40 --no-judge   # fast dry run

Last Updated: 2026-07-17
Related Issues: #713, #726, #736, #737
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb
from whoosh import index as whoosh_index
from whoosh.qparser import OrGroup, QueryParser

from klareco.parser import parse, expand_ast
from klareco.rag.duckdb_retriever import _content_terms
from klareco.eval.qa_judge import judge_batch

DB = 'data/indexes/duckdb_store.db'
WHOOSH = 'data/indexes/whoosh_v2'
_NOUN = ('substantivo', 'propra_nomo')
_PROPER = ('propra_nomo',)
_CONTENT_VS = ('substantivo', 'verbo', 'adjektivo', 'adverbo', 'propra_nomo', 'numeralo')


def _by_id(ast):
    return {w['id']: w for w in (ast.get('vortoj') or [])
            if isinstance(w, dict) and w.get('id')}


def _subtree_ids(byid, root_id, maxdepth=6):
    out = set()
    for w in byid.values():
        cur, seen = w, 0
        while cur and seen < maxdepth:
            if cur.get('id') == root_id:
                out.add(w['id']); break
            cur = byid.get(cur.get('kapo')); seen += 1
    return out


def _entity_surface(byid, subj_tok):
    sid_ = subj_tok['id']
    ids = {sid_}
    for w in byid.values():
        if w.get('kapo') == sid_ and w.get('rolo') in ('flat', 'appos', 'flat:name') \
           and w.get('vortspeco') in _NOUN:
            ids.add(w['id'])
    i = sid_ - 1
    while i in byid and byid[i].get('vortspeco') in _PROPER:
        ids.add(i); i -= 1
    i = sid_ + 1
    while i in byid and byid[i].get('vortspeco') in _PROPER:
        ids.add(i); i += 1
    lo, hi = min(ids), max(ids)
    return ' '.join((byid[i].get('plena_vorto') or '')
                    for i in range(lo, hi + 1) if i in byid).strip()


def _is_negated(ast, byid):
    if ast.get('negita'):
        return True
    return any((w.get('radiko') or '').lower() == 'ne'
               and w.get('rolo') in ('advmod', 'neg', 'advmod:neg') for w in byid.values())


def _grammar_ok(question: str) -> bool:
    """Lenient deterministic gate: the question must re-parse without crashing and
    not be dominated by unknown COMMON roots (a foreign-word / garble signal).

    Proper nouns are DELIBERATELY not rejected — names are legitimately absent from
    the root lexicon (the known parser-data gap: `Esperanton`->`esper`+`ant`), and a
    question SHOULD contain names. Naturalness/grammar nuance is the LLM judge's job;
    this gate only screens obvious garble so the judge isn't wasted on it."""
    try:
        ast = parse(question)
    except Exception:
        return False
    unknown_common = sum(
        1 for w in (ast.get('vortoj') or [])
        if isinstance(w, dict)
        and w.get('vortspeco') in ('substantivo', 'verbo', 'adjektivo', 'adverbo')
        and w.get('analizstato') == 'unknown_root')
    return unknown_common <= 1


def generate(con, n_candidates: int):
    """subject-gap questions from proper-noun-subject clauses; object phrase = the
    object subtree (modifiers ride along). Deterministic gates only."""
    rows = con.execute(f"""
        SELECT * FROM (
            SELECT c.sid, c.clause_idx, c.subj_radiko, c.verb_radiko, c.obj_radiko,
                   s.ast_json, s.text
            FROM clauses c JOIN sentences s ON s.sid=c.sid
            WHERE c.subj_vortspeco='propra_nomo' AND c.obj_radiko IS NOT NULL
              AND length(s.text) BETWEEN 40 AND 180
        ) USING SAMPLE {n_candidates} ROWS (reservoir, 17)
    """).fetchall()
    drop = collections.Counter()
    for sid, cidx, subj, verb, obj, aj, text in rows:
        try:
            ast = expand_ast(json.loads(aj))
        except Exception:
            drop['bad ast'] += 1; continue
        byid = _by_id(ast)
        if _is_negated(ast, byid):
            drop['negated'] += 1; continue
        vtok = next((w for w in byid.values()
                     if (w.get('radiko') or '').lower() == (verb or '').lower()
                     and w.get('vortspeco') == 'verbo'), None)
        otok = next((w for w in byid.values()
                     if (w.get('radiko') or '').lower() == (obj or '').lower()
                     and w.get('vortspeco') in _NOUN), None)
        subtok = next((w for w in byid.values()
                       if (w.get('radiko') or '').lower() == (subj or '').lower()
                       and w.get('vortspeco') == 'propra_nomo'), None)
        if not (vtok and otok and subtok):
            drop['missing token'] += 1; continue
        sub_ids = _subtree_ids(byid, otok['id'])
        lo, hi = min(sub_ids), max(sub_ids)
        if vtok['id'] in range(lo, hi + 1):
            drop['verb inside object span'] += 1; continue
        obj_phrase = ' '.join((byid[i].get('plena_vorto') or '')
                              for i in range(lo, hi + 1) if i in sub_ids).strip()
        answer = _entity_surface(byid, subtok)
        if not answer or answer.lower() not in text.lower():
            drop['answer not verbatim'] += 1; continue
        question = f'Kiu {vtok.get("plena_vorto")} {obj_phrase}?'
        if answer.lower() in obj_phrase.lower() or len(answer) < 3:
            drop['degenerate'] += 1; continue
        if not _grammar_ok(question):
            drop['grammar/pureness'] += 1; continue
        has_mod = any(w.get('kapo') == otok['id'] and w.get('rolo') == 'nmod'
                      for w in byid.values())
        yield {
            'id': f'qa-{sid}-{cidx}',
            'question': question, 'question_type': 'KIU',
            'expected_answer': answer, 'expected_keywords': [answer],
            'source_sentence_id': str(sid), 'source_sentence_text': text,
            'clause_idx': cidx, 'verb_surface': vtok.get('plena_vorto'),
            'gap_shape': 'modifier' if has_mod else ('subordinate' if cidx > 0 else 'main'),
        }
    generate.drop = drop


def bm25_rank(srch, qp, question, gold, limit=200):
    terms = _content_terms(question.lower())
    if not terms:
        return None
    hits = srch.search(qp.parse(' OR '.join(terms)), limit=limit)
    return next((i + 1 for i, h in enumerate(hits) if str(h['id']) == str(gold)), None)


def band(rank):
    if rank is None:
        return 'miss'
    if rank == 1:
        return 'trivial(r1)'
    if rank <= 5:
        return 'easy(2-5)'
    if rank <= 50:
        return 'hard(6-50)'
    return 'deep(51+)'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--target', type=int, default=500, help='kept-pair target')
    ap.add_argument('--oversample', type=int, default=8,
                    help='candidates sampled = target * this')
    ap.add_argument('--batch', type=int, default=15, help='pairs per judge call')
    ap.add_argument('--no-judge', action='store_true', help='skip the LLM judge (dry)')
    ap.add_argument('--model', default=None)
    ap.add_argument('--out', default='data/test_sets/qa_corpus_v1.jsonl')
    args = ap.parse_args()

    con = duckdb.connect(DB, read_only=True); con.execute('SET threads=6')
    ix = whoosh_index.open_dir(WHOOSH)
    qp = QueryParser('text', ix.schema, group=OrGroup)

    print(f'GENERATE + deterministic gates (grammar/pureness/presence)…')
    cands = list(generate(con, args.target * args.oversample))
    print(f'  {len(cands):,} passed deterministic gates')
    for k, v in generate.drop.most_common():
        print(f'    dropped {k:26s} {v:,}')

    print('\nDIFFICULTY (production BM25 rank)…')
    with ix.searcher() as srch:
        for c in cands:
            c['bm25_gold_rank'] = bm25_rank(srch, qp, c['question'], c['source_sentence_id'])
            c['difficulty'] = band(c['bm25_gold_rank'])

    # Spend the judge budget on RERANKABLE questions first. gold already at rank 1
    # (trivial) or beyond top-200 (miss) can't be improved by any reranker, so they
    # are low-value for the set's main purpose — keep a few for coverage, not the
    # bulk. Priority: hard > easy > deep(rerankable) >> trivial > miss.
    _PRIO = {'hard(6-50)': 0, 'easy(2-5)': 1, 'deep(51+)': 2, 'trivial(r1)': 3, 'miss': 4}
    cands.sort(key=lambda c: _PRIO.get(c['difficulty'], 5))

    if args.no_judge:
        kept = cands
        print('\nJUDGE skipped (--no-judge)')
    else:
        print(f'\nJUDGE via claude CLI ({args.batch}/batch, filter only)…')
        kept = []
        for i in range(0, len(cands), args.batch):
            chunk = cands[i:i + args.batch]
            verds = {v['id']: v for v in judge_batch(chunk, model=args.model)}
            for c in chunk:
                v = verds.get(c['id'], {})
                c['judge_keep'] = bool(v.get('keep'))
                c['judge_reason'] = v.get('reason', '')
                if c['judge_keep']:
                    kept.append(c)
            print(f'  judged {min(i+args.batch,len(cands)):>5d}/{len(cands)}  '
                  f'kept so far {len(kept)}', flush=True)
            if len(kept) >= args.target:
                break

    # de-dup by source sentence, cap at target
    seen, final = set(), []
    for c in kept:
        if c['source_sentence_id'] in seen:
            continue
        seen.add(c['source_sentence_id']); final.append(c)
        if len(final) >= args.target:
            break

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for c in final:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')

    print(f'\n✓ {len(final)} QA pairs -> {args.out}')
    print('  STRATIFICATION (gap_shape × difficulty):')
    strat = collections.Counter((c['gap_shape'], c['difficulty']) for c in final)
    for (shape, diff), k in sorted(strat.items()):
        print(f'    {shape:11s} {diff:12s} {k}')
    print('\n  POWER (need ~185 to resolve a 0.03 MRR delta, ~67 for 0.05):')
    for shape, k in sorted(collections.Counter(c['gap_shape'] for c in final).items()):
        print(f'    {shape:11s} n={k}  -> resolves ~{1.96*0.21/max(k,1)**0.5:.3f} MRR delta')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
