#!/usr/bin/env python3
"""
A test set that tests ARCS vs FRAMES — questions constrained by a modifier arc. (#736)

VERSION: v1.0
COMPATIBLE WITH: v2.3+ store (clauses + dependency_arcs), whoosh_v2
DEPENDENCIES: duckdb, whoosh, klareco.parser, klareco.rag.duckdb_retriever (_content_terms)
STAGE: Evaluation / test-set construction

Description:
    The clause-discriminating set proved reranking-in-general helps (+0.0528 MRR),
    but J_tree_aware (arcs) tied I_clause_aware (frames) on it — because every
    question was "Kiu VERB OBJ?", exactly the (subj,verb,obj) FRAME the clause
    reranker already reads. Arcs can express one thing a frame cannot: a constraint
    on a MODIFIER — "Kiu verkis la libron DE PETRO?" (the obj has an `nmod` arc to
    Petro). This set is made only of such questions, so arcs finally have something
    a frame does not.

    THE VALIDITY GATE (this is what makes it an ARC test, not a general one):
    a question only counts if a WRONG-STRUCTURE DISTRACTOR exists — a sentence that
    shares the verb and object roots but NOT the modifier arc, and that BM25 (on
    content terms) ranks at or above the gold. If no such distractor exists, the
    verb+object frame already disambiguates and the modifier does no work — the
    question is dropped (or flagged), because it could not distinguish arcs from
    frames.

    QUALITY GATES (learned from the v0 prototype's failures):
      - subject is a FULL proper-noun entity (extend across flat/appos), verbatim.
      - clause is NOT negated (a dropped `ne` inverts the meaning).
      - object phrase = the object's SUBTREE in token order (never the id-range,
        which swallowed the verb → "reasertis … reasertis").
      - the modifier root must actually appear in the object phrase.
      - R3 verb surface from the AST; R9 answer verbatim; R2 object is a noun.
      - R7 gold in content-term BM25 top-50; R16 gold NOT at rank 1 — via the
        PRODUCTION query path (_content_terms), not the raw question.

Pipeline Position:
    clauses + dependency_arcs + whoosh --[THIS]--> data/test_sets/arc_discriminating_qa.jsonl

Usage:
    python scripts/eval/build_arc_discriminating_qa.py --n 60 --pool 3000

Outputs:
    - data/test_sets/arc_discriminating_qa.jsonl
      {question, expected_answer, source_sentence_id, verb_surface, object_radiko,
       modifier_radiko, bm25_gold_rank, has_structural_distractor, distractor_sid}

Last Updated: 2026-07-17
Related Issues: #713, #736, #737
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

from klareco.parser import expand_ast
from klareco.rag.duckdb_retriever import _content_terms

DB = 'data/indexes/duckdb_store.db'
WHOOSH = 'data/indexes/whoosh_v2'
_PROPER = ('propra_nomo',)
_NOUN = ('substantivo', 'propra_nomo')
# tokens that chain a name together into one entity
_ENTITY_GLUE = ('flat', 'appos', 'flat:name', 'nmod', 'conj')


def _by_id(ast):
    return {w['id']: w for w in (ast.get('vortoj') or [])
            if isinstance(w, dict) and w.get('id')}


def _subtree_ids(byid, root_id, maxdepth=6):
    """ids of every token whose head-chain reaches root_id (the token's subtree)."""
    out = set()
    for w in byid.values():
        cur, seen = w, 0
        while cur and seen < maxdepth:
            if cur.get('id') == root_id:
                out.add(w['id'])
                break
            cur = byid.get(cur.get('kapo'))
            seen += 1
    return out


def _entity_surface(byid, subj_tok):
    """full proper-noun entity: the subject token + adjacent name-glue tokens."""
    sid_ = subj_tok['id']
    ids = {sid_}
    # pull in tokens that are children of the subject via name-glue relations,
    # and any immediately-adjacent proper nouns (multi-word names).
    for w in byid.values():
        if w.get('kapo') == sid_ and w.get('rolo') in _ENTITY_GLUE \
           and w.get('vortspeco') in _NOUN:
            ids.add(w['id'])
    # extend to a contiguous run of proper nouns around the subject
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
    return any(isinstance(w, dict) and (w.get('radiko') or '').lower() == 'ne'
               and w.get('rolo') in ('advmod', 'neg', 'advmod:neg')
               for w in byid.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--pool', type=int, default=3000)
    ap.add_argument('--top-k', type=int, default=50)
    ap.add_argument('--out', default='data/test_sets/arc_discriminating_qa.jsonl')
    ap.add_argument('--require-distractor', action='store_true', default=True,
                    help='drop questions with no wrong-structure distractor (default on)')
    ap.add_argument('--keep-nondiscriminating', dest='require_distractor',
                    action='store_false')
    args = ap.parse_args()

    con = duckdb.connect(DB, read_only=True)
    con.execute('SET threads=6')
    ix = whoosh_index.open_dir(WHOOSH)
    qp = QueryParser('text', ix.schema, group=OrGroup)

    print('  sampling proper-noun-subject clauses whose OBJECT has an nmod modifier…')
    # ⚠️ SAMPLE THE FILTERED SET, NOT THE RAW SCAN. `... WHERE ... USING SAMPLE`
    # samples the base table first and filters after — a 3000-row request came back
    # with 53. Wrap the filter in a subquery and sample THAT.
    rows = con.execute(f"""
        SELECT * FROM (
            SELECT c.sid, c.subj_radiko, c.verb_radiko, c.obj_radiko, s.ast_json, s.text
            FROM clauses c JOIN sentences s ON s.sid=c.sid
            WHERE c.subj_vortspeco='propra_nomo' AND c.obj_radiko IS NOT NULL
              AND length(s.text) BETWEEN 45 AND 180
              AND EXISTS (SELECT 1 FROM dependency_arcs a
                          WHERE a.sid=c.sid AND a.rolo='nmod' AND a.kapo_radiko=c.obj_radiko)
        ) USING SAMPLE {args.pool} ROWS (reservoir, 17)
    """).fetchall()
    print(f'  {len(rows):,} candidates\n')

    out, drop, ranks = [], collections.Counter(), []
    n_distractor = 0

    with ix.searcher() as srch:
        for sid, subj, verb, obj, aj, text in rows:
            if len(out) >= args.n:
                break
            try:
                ast = expand_ast(json.loads(aj))
            except Exception:
                drop['bad ast'] += 1; continue
            byid = _by_id(ast)

            if _is_negated(ast, byid):
                drop['negated clause'] += 1; continue

            # verb surface (R3)
            vtok = next((w for w in byid.values()
                         if (w.get('radiko') or '').lower() == (verb or '').lower()
                         and w.get('vortspeco') == 'verbo'), None)
            # object token (R2: a noun)
            otok = next((w for w in byid.values()
                         if (w.get('radiko') or '').lower() == (obj or '').lower()
                         and w.get('vortspeco') in _NOUN), None)
            subtok = next((w for w in byid.values()
                           if (w.get('radiko') or '').lower() == (subj or '').lower()
                           and w.get('vortspeco') == 'propra_nomo'), None)
            if not (vtok and otok and subtok):
                drop['missing verb/obj/subj token'] += 1; continue

            # the object's nmod modifier
            mod = next((w for w in byid.values()
                        if w.get('kapo') == otok['id'] and w.get('rolo') == 'nmod'), None)
            if not mod:
                drop['no nmod on object token'] += 1; continue
            mod_radiko = (mod.get('radiko') or '').lower()

            # object phrase = the object's SUBTREE in token order (natural, incl. det/case)
            sub_ids = _subtree_ids(byid, otok['id'])
            if mod['id'] not in sub_ids:
                drop['modifier not in object subtree'] += 1; continue
            lo, hi = min(sub_ids), max(sub_ids)
            obj_phrase = ' '.join((byid[i].get('plena_vorto') or '')
                                  for i in range(lo, hi + 1) if i in sub_ids).strip()
            if vtok['id'] in range(lo, hi + 1):  # verb inside the span => malformed
                drop['verb inside object span'] += 1; continue

            answer = _entity_surface(byid, subtok)
            if not answer or answer.lower() not in text.lower():
                drop['answer entity not verbatim (R9)'] += 1; continue

            question = f'Kiu {vtok.get("plena_vorto")} {obj_phrase}?'
            if answer.lower() in obj_phrase.lower():
                drop['answer appears in question'] += 1; continue

            # ---- R7 + R16 via the PRODUCTION content-term path ----
            terms = _content_terms(question.lower())
            if not terms:
                drop['no content terms'] += 1; continue
            hits = srch.search(qp.parse(' OR '.join(terms)), limit=args.top_k)
            hit_ids = [int(h['id']) for h in hits]
            rank = next((i + 1 for i, x in enumerate(hit_ids) if x == sid), None)
            if rank is None:
                drop['gold NOT in BM25 top-50 (R7)'] += 1; continue
            if rank == 1:
                drop['gold at BM25 rank 1 (R16)'] += 1; continue

            # ---- ADVERSARIAL DISTRACTOR: a sentence at/above gold with verb+obj
            #      roots but WITHOUT the (obj -nmod-> modifier) arc ----
            above = [x for x in hit_ids[:rank] if x != sid]
            distractor = None
            if above:
                placeholders = ','.join('?' * len(above))
                # candidates that have BOTH verb and obj roots somewhere in the tree
                vo = set(r[0] for r in con.execute(
                    f"SELECT a1.sid FROM dependency_arcs a1 JOIN dependency_arcs a2 "
                    f"ON a1.sid=a2.sid WHERE a1.sid IN ({placeholders}) "
                    f"AND a1.dep_radiko=? AND a2.dep_radiko=?",
                    [*above, obj, verb]).fetchall())
                for cand in above:
                    if cand not in vo:
                        continue
                    # does cand have the SAME modifier arc? if not -> wrong structure
                    same = con.execute(
                        "SELECT 1 FROM dependency_arcs WHERE sid=? AND rolo='nmod' "
                        "AND kapo_radiko=? AND dep_radiko=? LIMIT 1",
                        [cand, obj, mod_radiko]).fetchone()
                    if not same:
                        distractor = cand; break
            has_distractor = distractor is not None
            if args.require_distractor and not has_distractor:
                drop['no wrong-structure distractor (not an ARC test)'] += 1; continue
            if has_distractor:
                n_distractor += 1

            ranks.append(rank)
            out.append({
                'id': f'arc-{sid}',
                'question': question,
                'question_type': 'KIU',
                'expected_answer': answer,
                'expected_keywords': [answer],
                'source_sentence_id': str(sid),
                'source_sentence_text': text,
                'verb_surface': vtok.get('plena_vorto'),
                'verb_root': verb,
                'object_radiko': obj,
                'modifier_radiko': mod_radiko,
                'bm25_gold_rank': rank,
                'has_structural_distractor': has_distractor,
                'distractor_sid': str(distractor) if distractor else None,
            })

    print(f'  {"reason dropped":48s} {"n":>5s}')
    for k, v in drop.most_common():
        print(f'  {k:48s} {v:5,}')
    if not out:
        print('\n  ✗ produced NOTHING. Loosen --pool / --require-distractor.')
        return 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    ranks.sort()
    hist = collections.Counter()
    for r in ranks:
        hist['2-5' if r <= 5 else ('6-20' if r <= 20 else '21-50')] += 1
    print(f'\n  ✓ {len(out)} questions -> {args.out}')
    print(f'    with a wrong-structure distractor: {n_distractor}/{len(out)} '
          f'({n_distractor/len(out):.0%})  <- the ones that test ARCS not FRAMES')
    print(f'    median BM25 gold rank: {ranks[len(ranks)//2]}   (R16 requires > 1)')
    for b in ('2-5', '6-20', '21-50'):
        print(f'    rank {b:6s} {hist[b]:3}  {"█"*hist[b]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
