#!/usr/bin/env python3
"""
Multi-reranker bench: compare N reranking strategies on the same cached
candidate pool to measure which yield real improvement.

VERSION: v2.x (DuckDB)
COMPATIBLE WITH: post-bug-#1/#2/#4 parser, post-refresh DuckDB
DEPENDENCIES: duckdb, klareco.parser, klareco.orchestrator, klareco.rag
STAGE: Evaluation

Description:
    GitHub issue: #733
    Validates the deterministic-indexing work (#728-732) and any future
    reranker proposal. The flow:

      Question → parse →
        BM25 top-100 (cached once per question) →
          ├─ Reranker A: BM25 baseline (identity)
          ├─ Reranker B: phrase-query boost (no index dep)
          ├─ Reranker C: radiko-aware boost (uses #728 indices)
          ├─ Reranker D: entity-postings boost (uses #729 table)
          ├─ Reranker E: verb-class expansion (uses #730 column)
          ├─ Reranker F: negation-aware filter (uses #731 column)
          ├─ Reranker G: AST role-scoring (uses #576 reranker)
          ├─ Reranker H: Reciprocal Rank Fusion over all the above
          → For each reranker: take its top-10, run extractor, record
            answer + rank-of-correct + correct?

    Aggregate: per-reranker Recall@1/5/10, MRR, answer accuracy. The
    differences are the signal.

    Rerankers that need indices not yet built are SKIPPED gracefully —
    the bench logs which were available and which couldn't run.

Pipeline Position:
    Test set JSONL → [THIS SCRIPT] → wide-format per-question report
                                  → aggregate metrics CSV/JSONL
                                  → identifies which reranker to ship

Usage:
    python scripts/eval/multi_reranker_bench.py \\
        --test-set data/test_sets/trivia_bank.jsonl

Inputs:
    --test-set       JSONL with {question, expected_keywords, ...} per line
    --duckdb-path    data/indexes/duckdb_store.db
    --whoosh-dir     data/indexes/whoosh_v2

Outputs:
    Stdout: aggregate metrics table
    --output-jsonl: per-question per-reranker detail
    --output-csv:   wide-format CSV (one row per question, one column per reranker)

Last Updated: 2026-05-20
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb

from klareco.parser import parse
from klareco.orchestrator import build_default_pipeline
from klareco.orchestrator.context import QueryContext, ContextDelta, ParsedPassage
from klareco.orchestrator.stages.parse_question import ParseQuestionStage
from klareco.orchestrator.stages.extract_generate import ExtractAndGenerateStage
from klareco.orchestrator.stages.format_output import FormatOutputStage
from klareco.rag.extractive_answering import ExtractiveAnswerGenerator


# =============================================================================
# Reranker abstraction
# =============================================================================

@dataclass
class RerankerResult:
    name: str
    ranked: list[ParsedPassage]   # top-K passages after reranking
    metadata: dict                # arbitrary diagnostic info


class Reranker:
    """Base class. Subclasses implement `rerank`."""
    name: str = '???'
    requires: list[str] = []  # column / table names required from DuckDB

    def available(self, conn) -> bool:
        """Check that this reranker's required DB assets exist."""
        for req in self.requires:
            try:
                if '.' in req:
                    table, col = req.split('.')
                    conn.execute(f"SELECT {col} FROM {table} LIMIT 1").fetchone()
                else:
                    conn.execute(f"SELECT * FROM {req} LIMIT 1").fetchone()
            except Exception:
                return False
        return True

    def rerank(self, question: str, question_ast: dict,
               candidates: list[ParsedPassage], conn,
               top_k: int = 10) -> RerankerResult:
        raise NotImplementedError


class BaselineReranker(Reranker):
    """Identity reranker — preserve BM25 order."""
    name = 'A_bm25_baseline'

    def rerank(self, question, question_ast, candidates, conn, top_k=10):
        return RerankerResult(name=self.name, ranked=list(candidates[:top_k]),
                              metadata={})


class PhraseQueryReranker(Reranker):
    """Boost passages whose surface text contains an entity from the question
    as an exact substring. No index dependency."""
    name = 'B_phrase_query'

    def _question_entities(self, question: str, question_ast: dict) -> list[str]:
        """Extract candidate phrases to look for: quoted spans + proper nouns."""
        ents = []
        for m in re.finditer(r'[«"„]\s*([^«»"]{2,80}?)\s*[»"]', question):
            ents.append(m.group(1).strip())
        # Multi-token entities from question's AST
        for g in (question_ast.get('multi_token_entities') or []):
            span = ' '.join(g.get('span_tokens') or [])
            if len(span) > 2:
                ents.append(span)
        # Single-token capitalised propra-nomos from question
        for token in re.findall(r'\b[A-ZÀ-ÞĈĜĤĴŜŬ][\wÀ-ſĉĝĥĵŝŭĈĜĤĴŜŬ-]{3,}', question):
            if token in ('Kio', 'Kiu', 'Kie', 'Kiam', 'Kial', 'Kiel',
                         'Kiom', 'Kion', 'Kies', 'Kia'):
                continue
            ents.append(token)
        return list(dict.fromkeys(ents))  # dedupe, preserve order

    def rerank(self, question, question_ast, candidates, conn, top_k=10):
        ents = self._question_entities(question, question_ast)
        if not ents:
            return RerankerResult(name=self.name, ranked=list(candidates[:top_k]),
                                  metadata={'entities': ents})
        scored: list[tuple[float, ParsedPassage]] = []
        for p in candidates:
            text = p.text or ''
            boost = 0.0
            for e in ents:
                if e in text:
                    boost += 5.0 * (len(e.split()) ** 1.5)
            new_score = p.score + boost
            scored.append((new_score, p))
        scored.sort(key=lambda kv: -kv[0])
        ranked = [
            ParsedPassage(sentence_id=p.sentence_id, text=p.text, ast=p.ast,
                          score=ns, source_doc=p.source_doc, source_type=p.source_type)
            for ns, p in scored[:top_k]
        ]
        return RerankerResult(name=self.name, ranked=ranked,
                              metadata={'entities': ents})


class RadikoAwareReranker(Reranker):
    """Boost passages whose subj/verb/obj radiko matches the question's.
    Depends on #728 indices being built (for speed) but works without
    them on small candidate pools too."""
    name = 'C_radiko_aware'
    requires = []  # no hard requirement (uses sentences.* which exists)

    def rerank(self, question, question_ast, candidates, conn, top_k=10):
        q_subj = ((question_ast.get('subjekto') or {}).get('kerno') or {}).get('radiko')
        q_verb = (question_ast.get('verbo') or {}).get('radiko')
        q_obj = ((question_ast.get('objekto') or {}).get('kerno') or {}).get('radiko')
        scored = []
        for p in candidates:
            boost = 0.0
            if p.ast:
                p_subj = ((p.ast.get('subjekto') or {}).get('kerno') or {}).get('radiko')
                p_verb = (p.ast.get('verbo') or {}).get('radiko')
                p_obj = ((p.ast.get('objekto') or {}).get('kerno') or {}).get('radiko')
                # Reward role-aligned radiko matches
                if q_verb and p_verb == q_verb:
                    boost += 3.0
                if q_subj and p_subj == q_subj:
                    boost += 2.0
                if q_obj and p_obj == q_obj:
                    boost += 2.0
                # Modest cross-role boosts
                if q_obj and (p_subj == q_obj or p_obj == q_obj):
                    boost += 1.0
            scored.append((p.score + boost, p))
        scored.sort(key=lambda kv: -kv[0])
        ranked = [
            ParsedPassage(sentence_id=p.sentence_id, text=p.text, ast=p.ast,
                          score=ns, source_doc=p.source_doc, source_type=p.source_type)
            for ns, p in scored[:top_k]
        ]
        return RerankerResult(name=self.name, ranked=ranked,
                              metadata={'q_subj': q_subj, 'q_verb': q_verb, 'q_obj': q_obj})


class NegationAwareReranker(Reranker):
    """Penalise passages whose verb has opposite negation polarity to the
    question. Requires #731's `verb_negated` column."""
    name = 'D_negation_aware'
    requires = ['sentences.verb_negated']

    def rerank(self, question, question_ast, candidates, conn, top_k=10):
        q_negated = bool((question_ast.get('verbo') or {}).get('negita'))
        scored = []
        for p in candidates:
            # Look up verb_negated for the passage. We could pre-cache, but for
            # the bench we fetch per-passage (cheap with index).
            penalty = 0.0
            try:
                row = conn.execute(
                    "SELECT verb_negated FROM sentences WHERE sid = ?",
                    [int(p.sentence_id)]
                ).fetchone()
                if row is not None and row[0] != q_negated and row[0] is not None:
                    penalty = 2.0
            except Exception:
                pass
            scored.append((p.score - penalty, p))
        scored.sort(key=lambda kv: -kv[0])
        ranked = [
            ParsedPassage(sentence_id=p.sentence_id, text=p.text, ast=p.ast,
                          score=ns, source_doc=p.source_doc, source_type=p.source_type)
            for ns, p in scored[:top_k]
        ]
        return RerankerResult(name=self.name, ranked=ranked,
                              metadata={'q_negated': q_negated})


class VerbClassReranker(Reranker):
    """Boost passages whose verb shares a VerbaKlaso with the question's verb.
    Requires #730's `verb_klaso` column."""
    name = 'E_verb_class'
    requires = ['sentences.verb_klaso']

    def rerank(self, question, question_ast, candidates, conn, top_k=10):
        q_verb_radiko = (question_ast.get('verbo') or {}).get('radiko')
        q_klaso = None
        if q_verb_radiko:
            try:
                row = conn.execute(
                    "SELECT class_id FROM ontology_edges "
                    "WHERE rel = 'APARTENAS_AL_VERBA_KLASO' AND radiko = ?",
                    [q_verb_radiko]
                ).fetchone()
                q_klaso = row[0] if row else None
            except Exception:
                pass
        if not q_klaso:
            return RerankerResult(name=self.name, ranked=list(candidates[:top_k]),
                                  metadata={'q_klaso': None})
        scored = []
        for p in candidates:
            boost = 0.0
            try:
                row = conn.execute(
                    "SELECT verb_klaso FROM sentences WHERE sid = ?",
                    [int(p.sentence_id)]
                ).fetchone()
                if row and row[0] == q_klaso:
                    boost += 2.5
            except Exception:
                pass
            scored.append((p.score + boost, p))
        scored.sort(key=lambda kv: -kv[0])
        ranked = [
            ParsedPassage(sentence_id=p.sentence_id, text=p.text, ast=p.ast,
                          score=ns, source_doc=p.source_doc, source_type=p.source_type)
            for ns, p in scored[:top_k]
        ]
        return RerankerResult(name=self.name, ranked=ranked,
                              metadata={'q_klaso': q_klaso})


class EntityPostingsReranker(Reranker):
    """Direct lookup of entity-mentioning sentences via the entity_postings
    table (#729). Promotes those passages to the top regardless of BM25 rank."""
    name = 'F_entity_postings'
    requires = ['entity_postings']

    def rerank(self, question, question_ast, candidates, conn, top_k=10):
        # Same entity extraction as the phrase-query reranker
        pqr = PhraseQueryReranker()
        ents = pqr._question_entities(question, question_ast)
        if not ents:
            return RerankerResult(name=self.name, ranked=list(candidates[:top_k]),
                                  metadata={'entities': ents})
        # Lookup sids for these entities
        promoted_sids: set[int] = set()
        for e in ents:
            try:
                rows = conn.execute(
                    "SELECT sid FROM entity_postings WHERE entity_text = ? LIMIT 50",
                    [e]
                ).fetchall()
                promoted_sids.update(r[0] for r in rows)
            except Exception:
                pass
        # Reorder: promoted passages first (preserving their BM25 score order),
        # then the rest
        promoted, rest = [], []
        for p in candidates:
            if int(p.sentence_id) in promoted_sids:
                promoted.append(p)
            else:
                rest.append(p)
        ranked = (promoted + rest)[:top_k]
        # Boost their scores so downstream metrics see them as "high confidence"
        ranked = [
            ParsedPassage(sentence_id=p.sentence_id, text=p.text, ast=p.ast,
                          score=p.score + (5.0 if int(p.sentence_id) in promoted_sids else 0.0),
                          source_doc=p.source_doc, source_type=p.source_type)
            for p in ranked
        ]
        return RerankerResult(name=self.name, ranked=ranked,
                              metadata={'entities': ents,
                                        'promoted_count': len(promoted)})


class RRFCombo(Reranker):
    """Reciprocal Rank Fusion across all sub-rerankers that ran successfully."""
    name = 'Z_rrf_combo'

    def __init__(self, sub_rerankers: list[Reranker], k: int = 60):
        self.subs = sub_rerankers
        self.k_constant = k

    def rerank(self, question, question_ast, candidates, conn, top_k=10):
        per_sub_scores: dict[int, float] = {}
        for sub in self.subs:
            try:
                sub_result = sub.rerank(question, question_ast, candidates, conn,
                                        top_k=len(candidates))
            except Exception:
                continue
            for rank, p in enumerate(sub_result.ranked, 1):
                sid_i = int(p.sentence_id)
                per_sub_scores[sid_i] = per_sub_scores.get(sid_i, 0.0) + 1.0 / (self.k_constant + rank)
        # Re-sort candidates by combined score
        sid_to_passage = {int(p.sentence_id): p for p in candidates}
        ranked_ids = sorted(per_sub_scores.keys(),
                            key=lambda s: -per_sub_scores[s])[:top_k]
        ranked = [
            ParsedPassage(
                sentence_id=str(sid_i), text=sid_to_passage[sid_i].text,
                ast=sid_to_passage[sid_i].ast,
                score=per_sub_scores[sid_i],
                source_doc='rrf', source_type='rrf',
            )
            for sid_i in ranked_ids if sid_i in sid_to_passage
        ]
        return RerankerResult(name=self.name, ranked=ranked,
                              metadata={'subs_used': [s.name for s in self.subs]})


# =============================================================================
# Bench machinery
# =============================================================================

def get_bm25_candidates(pipeline, question: str, top_k: int = 100
                        ) -> tuple[list[ParsedPassage], dict]:
    """Run only the retrieve stage to get a frozen candidate pool."""
    parse_stage = ParseQuestionStage()
    ctx = QueryContext(question=question)
    delta = parse_stage.run(ctx); ctx = ctx.apply(delta)
    question_ast = ctx.symbolic.question_ast
    for stage in pipeline.stages:
        if stage.name == 'retrieve':
            delta = stage.run(ctx); ctx = ctx.apply(delta)
            break
    return list(ctx.symbolic.passage_asts or []), question_ast


def run_extractor(question: str, passages: list[ParsedPassage]) -> str:
    """Run extract+format on a given passage list. Returns final_text."""
    gen = ExtractiveAnswerGenerator()
    extract = ExtractAndGenerateStage(generator=gen)
    fmt = FormatOutputStage()
    parse_stage = ParseQuestionStage()

    ctx = QueryContext(question=question)
    delta = parse_stage.run(ctx); ctx = ctx.apply(delta)
    ctx = ctx.apply(ContextDelta(symbolic={'passage_asts': tuple(passages)}))
    delta = extract.run(ctx); ctx = ctx.apply(delta)
    delta = fmt.run(ctx); ctx = ctx.apply(delta)
    return ctx.symbolic.final_text or ''


def find_first_relevant_rank(ranked: list[ParsedPassage],
                              expected_keywords: list[str]) -> int | None:
    for rank, p in enumerate(ranked, 1):
        text = (p.text or '').lower()
        if any(kw.lower() in text for kw in expected_keywords):
            return rank
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--test-set', required=True)
    ap.add_argument('--whoosh-dir', default='data/indexes/whoosh_v2')
    ap.add_argument('--duckdb-path', default='data/indexes/duckdb_store.db')
    ap.add_argument('--top-k', type=int, default=10)
    ap.add_argument('--candidate-pool', type=int, default=100)
    ap.add_argument('--output-jsonl', default=None)
    ap.add_argument('--output-csv', default=None)
    args = ap.parse_args()

    test = []
    with open(args.test_set) as f:
        for line in f:
            line = line.strip()
            if line:
                test.append(json.loads(line))
    print(f'Loaded {len(test)} test questions from {args.test_set}\n')

    conn = duckdb.connect(args.duckdb_path, read_only=True)
    pipeline = build_default_pipeline(whoosh_index_dir=args.whoosh_dir,
                                       top_k=args.candidate_pool)

    # Instantiate rerankers, dropping those whose required DB assets aren't present
    candidate_rerankers: list[Reranker] = [
        BaselineReranker(),
        PhraseQueryReranker(),
        RadikoAwareReranker(),
        NegationAwareReranker(),
        VerbClassReranker(),
        EntityPostingsReranker(),
    ]
    enabled: list[Reranker] = []
    skipped: list[str] = []
    for r in candidate_rerankers:
        if r.available(conn):
            enabled.append(r)
        else:
            skipped.append(f'{r.name} (missing: {r.requires})')
    # RRF combo over all available rerankers (excluding baseline to avoid double-count)
    if len(enabled) >= 2:
        enabled.append(RRFCombo([r for r in enabled if r.name != 'A_bm25_baseline']))

    print(f'Enabled rerankers ({len(enabled)}):')
    for r in enabled:
        print(f'  {r.name}')
    if skipped:
        print(f'\nSkipped (DB asset not present):')
        for s in skipped:
            print(f'  {s}')
    print()

    # Per-question per-reranker results
    rows: list[dict] = []
    for q_idx, q in enumerate(test, 1):
        question = q.get('eo_question') or q['question']
        expected = q.get('expected_keywords') or [q.get('eo_answer', '')]
        if not expected[0]:
            continue
        candidates, q_ast = get_bm25_candidates(pipeline, question,
                                                 top_k=args.candidate_pool)
        row = {
            'id':       q.get('id', f'q{q_idx}'),
            'question': question,
            'expected': expected,
        }
        for r in enabled:
            res = r.rerank(question, q_ast, candidates, conn, top_k=args.top_k)
            final = run_extractor(question, res.ranked)
            correct = any(kw.lower() in final.lower() for kw in expected)
            rank = find_first_relevant_rank(res.ranked, expected)
            row[f'{r.name}_rank'] = rank
            row[f'{r.name}_correct'] = correct
            row[f'{r.name}_answer'] = final[:120]
        rows.append(row)
        marks = ' '.join(
            ('✓' if row.get(f'{r.name}_correct') else '·') for r in enabled
        )
        print(f'  [{q_idx:>3}/{len(test)}]  {marks}  {question[:60]}')

    # Aggregate
    print(f'\n\n=== Aggregate ({len(rows)} questions) ===\n')
    print(f'{"reranker":<22s} {"R@1":>5s} {"R@5":>5s} {"R@10":>5s} {"MRR":>6s} {"ans%":>6s}')
    print('-' * 60)
    for r in enabled:
        ranks = [row.get(f'{r.name}_rank') for row in rows]
        n_r1 = sum(1 for x in ranks if x == 1)
        n_r5 = sum(1 for x in ranks if x is not None and x <= 5)
        n_r10 = sum(1 for x in ranks if x is not None and x <= 10)
        mrr = (sum(1.0 / x for x in ranks if x is not None) / len(ranks)
               if ranks else 0)
        n_correct = sum(1 for row in rows if row.get(f'{r.name}_correct'))
        print(f'{r.name:<22s} {n_r1:>5d} {n_r5:>5d} {n_r10:>5d} '
              f'{mrr:>6.3f} {100*n_correct/max(1,len(rows)):>5.1f}%')

    if args.output_jsonl:
        Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_jsonl, 'w') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f'\nPer-question JSONL: {args.output_jsonl}')


if __name__ == '__main__':
    main()
