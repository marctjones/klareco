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
from typing import Callable, Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for perf_history import

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
    """REDESIGNED (v2): focuses on the question's *propra_nomo* entities
    rather than the (often-generic) subjekto.kerno. Most trivia
    questions have the question's named entity in `aliaj` (PP-governed
    in the question's surface form, e.g. `disvolvis la senpagan ludon
    Fortnite?` — Fortnite is in aliaj, not subjekto). The previous v1
    matched on `subjekto.kerno.radiko` which was usually 'kiu' or a
    generic noun, boosting irrelevant passages.

    v2: reward candidates whose subj/obj/aliaj contain a propra_nomo
    from the question's aliaj/objekto. Verb radiko is a tie-breaker."""
    name = 'C_radiko_aware'
    requires = []

    def _question_propra_nomos(self, question_ast: dict) -> set[str]:
        """All propra_nomo plena_vortos from the question AST (any role)."""
        names: set[str] = set()
        for role in ('subjekto', 'objekto'):
            n = question_ast.get(role)
            if isinstance(n, dict):
                k = n.get('kerno') if n.get('tipo') == 'vortgrupo' else n
                if isinstance(k, dict) and k.get('vortspeco') == 'propra_nomo':
                    pv = k.get('plena_vorto')
                    if pv:
                        names.add(pv)
        for item in question_ast.get('aliaj') or []:
            if isinstance(item, dict):
                k = item.get('kerno') if item.get('tipo') == 'vortgrupo' else item
                if isinstance(k, dict) and k.get('vortspeco') == 'propra_nomo':
                    pv = k.get('plena_vorto')
                    if pv:
                        names.add(pv)
        # Also include any multi_token_entity spans
        for g in question_ast.get('multi_token_entities') or []:
            span = g.get('span_tokens') or []
            for tok in span:
                names.add(tok)
            if span:
                names.add(' '.join(span))
        return names

    def _candidate_propra_nomos_anywhere(self, ast: dict) -> set[str]:
        """All propra_nomo strings anywhere in the candidate AST."""
        if not isinstance(ast, dict):
            return set()
        names: set[str] = set()
        for role in ('subjekto', 'objekto'):
            n = ast.get(role)
            if isinstance(n, dict):
                k = n.get('kerno') if n.get('tipo') == 'vortgrupo' else n
                if isinstance(k, dict) and k.get('vortspeco') == 'propra_nomo':
                    pv = k.get('plena_vorto')
                    if pv:
                        names.add(pv)
        for item in ast.get('aliaj') or []:
            if isinstance(item, dict):
                k = item.get('kerno') if item.get('tipo') == 'vortgrupo' else item
                if isinstance(k, dict) and k.get('vortspeco') == 'propra_nomo':
                    pv = k.get('plena_vorto')
                    if pv:
                        names.add(pv)
        for g in ast.get('multi_token_entities') or []:
            span = g.get('span_tokens') or []
            for tok in span:
                names.add(tok)
            if span:
                names.add(' '.join(span))
        return names

    def rerank(self, question, question_ast, candidates, conn, top_k=10):
        q_names = self._question_propra_nomos(question_ast)
        q_verb = (question_ast.get('verbo') or {}).get('radiko')
        scored = []
        for p in candidates:
            boost = 0.0
            if p.ast and q_names:
                p_names = self._candidate_propra_nomos_anywhere(p.ast)
                shared = q_names & p_names
                if shared:
                    # Strong reward — exact entity match anywhere in candidate
                    boost += 5.0 * len(shared)
            if p.ast and q_verb:
                p_verb = (p.ast.get('verbo') or {}).get('radiko')
                if p_verb == q_verb:
                    boost += 1.0  # tie-breaker
            scored.append((p.score + boost, p))
        scored.sort(key=lambda kv: -kv[0])
        ranked = [
            ParsedPassage(sentence_id=p.sentence_id, text=p.text, ast=p.ast,
                          score=ns, source_doc=p.source_doc, source_type=p.source_type)
            for ns, p in scored[:top_k]
        ]
        return RerankerResult(name=self.name, ranked=ranked,
                              metadata={'q_names': list(q_names),
                                        'q_verb': q_verb})


class NegationAwareReranker(Reranker):
    """Penalise passages whose verb has opposite negation polarity to the
    question. Requires #731's `verb_negated` column."""
    name = 'D_negation_aware'
    # ⚠️ THE FRAME LIVES ON `clauses`, NOT ON `sentences`.
    #
    # `verb_klaso` and `verb_negated` moved when the clause table landed (#836):
    # the predicate-argument frame belongs to a CLAUSE, because gold has 1.64
    # subjects per sentence and a one-frame-per-sentence schema silently discards
    # every subordinate clause.
    #
    # This bench still looked for them on `sentences`, so `requires` was never
    # satisfied and FIVE OF THE NINE RERANKERS WERE SKIPPED — D, E, F, G, H. Not
    # tied: never run. CLAUDE.md records "all nine rerankers are tied" as a
    # finding; four of them were tied and five were not present.
    requires = ['clauses.verb_negated']

    def rerank(self, question, question_ast, candidates, conn, top_k=10):
        q_negated = bool((question_ast.get('verbo') or {}).get('negita'))
        scored = []
        for p in candidates:
            # Look up verb_negated for the passage. We could pre-cache, but for
            # the bench we fetch per-passage (cheap with index).
            penalty = 0.0
            try:
                row = conn.execute(
                    "SELECT verb_negated FROM clauses "
                    "WHERE sid = ? AND clause_idx = 0",
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
    requires = ['clauses.verb_klaso']

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
                    "SELECT verb_klaso FROM clauses "
                    "WHERE sid = ? AND clause_idx = 0",
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


class ASTAwareRanker(Reranker):
    """Per-question-type structured score over shredded AST columns.
    Implements the framework from #741. Stage 1: uses only existing
    `sentences` columns + on-the-fly aliaj_json parsing for type flags."""
    name = 'G_ast_aware'
    requires = ['clauses.verb_klaso', 'clauses.verb_negated']

    _COLS_CORE = ('sid', 'text', 'subj_radiko', 'subj_vortspeco',
                  'subj_propranoma_kat', 'subj_kazo',
                  'verb_radiko', 'verb_tempo', 'verb_klaso', 'verb_negated',
                  'obj_radiko', 'obj_kazo', 'aliaj_json')
    _COLS_STAGE2 = ('aliaj_has_loko', 'aliaj_has_jaro', 'aliaj_has_kvant')

    @property
    def _COLS(self) -> tuple[str, ...]:
        """Columns to fetch — include Stage 2 boolean flags if they exist."""
        cols = list(self._COLS_CORE)
        if self._stage2_available:
            cols.extend(self._COLS_STAGE2)
        return tuple(cols)

    def __init__(self):
        from klareco.rag.ast_aware_reranker import ASTAwareScorer
        self._scorer_cls = ASTAwareScorer
        self._scorer = None
        self._stage2_available = False
        self._stage2_probed = False

    def _probe_stage2(self, conn) -> None:
        """Check once whether the Stage-2 aliaj_has_* columns are present."""
        if self._stage2_probed:
            return
        try:
            conn.execute(
                "SELECT aliaj_has_loko, aliaj_has_jaro, aliaj_has_kvant "
                "FROM sentences LIMIT 1").fetchone()
            self._stage2_available = True
        except Exception:
            self._stage2_available = False
        self._stage2_probed = True

    def rerank(self, question, question_ast, candidates, conn, top_k=10):
        if not candidates:
            return RerankerResult(name=self.name, ranked=[], metadata={})
        self._probe_stage2(conn)
        if self._scorer is None:
            self._scorer = self._scorer_cls(conn)
        # Batch-fetch the shredded columns for all candidate sids.
        sids = [int(p.sentence_id) for p in candidates]
        placeholders = ','.join('?' for _ in sids)
        col_list = ', '.join(self._COLS)
        try:
            rows = conn.execute(
                f"SELECT {col_list} FROM sentences "
                f"WHERE sid IN ({placeholders})",
                sids,
            ).fetchall()
        except Exception as e:
            return RerankerResult(name=self.name,
                                  ranked=list(candidates[:top_k]),
                                  metadata={'error': str(e)[:80]})
        row_by_sid = {int(r[0]): dict(zip(self._COLS, r)) for r in rows}
        # Build the candidate dicts the scorer expects.
        bm25_scores = {int(p.sentence_id): float(p.score) for p in candidates}
        cand_dicts = []
        ast_by_sid = {int(p.sentence_id): p.ast for p in candidates}
        for sid in sids:
            r = row_by_sid.get(sid)
            if r is None:
                continue
            d = dict(r)
            d['ast'] = ast_by_sid.get(sid)
            cand_dicts.append(d)
        scored = self._scorer.score_batch(question_ast, cand_dicts,
                                          bm25_scores,
                                          question_text=question)
        # Map back to ParsedPassage list, preserving original ParsedPassage
        # objects so source_doc/source_type carry through.
        pp_by_sid = {int(p.sentence_id): p for p in candidates}
        ranked: list[ParsedPassage] = []
        for new_score, c in scored[:top_k]:
            pp = pp_by_sid.get(int(c['sid']))
            if pp is None:
                continue
            ranked.append(ParsedPassage(
                sentence_id=pp.sentence_id, text=pp.text, ast=pp.ast,
                score=float(new_score),
                source_doc=pp.source_doc, source_type=pp.source_type,
            ))
        from klareco.rag.ast_aware_reranker import detect_question_type
        return RerankerResult(name=self.name, ranked=ranked,
                              metadata={'qtype': detect_question_type(question_ast)})


class HybridReranker(Reranker):
    """Hybrid: G_ast_aware structural filters + B_phrase_query lexical boost.

    Composition (multiplicative):
        candidate's AST-aware score (with hard filters) is multiplied by
        (1 + phrase_boost / 10), where phrase_boost is the same
        entity-in-text boost B_phrase_query computes. Filters still gate
        (mismatched/missing → 0). Among surviving candidates, those whose
        text literally contains the question's entities are amplified.

    Rationale: structural filters remove KNOWN BAD candidates (high
    precision), phrase boost identifies HIGH-CONFIDENCE good ones
    (high recall on the right relevance signal). Multiplying combines
    them without letting either dominate.
    """
    name = 'H_hybrid'
    requires = ['clauses.verb_klaso', 'clauses.verb_negated']

    _COLS = ASTAwareRanker._COLS_CORE
    _COLS_STAGE2 = ASTAwareRanker._COLS_STAGE2

    def __init__(self):
        self._ast_reranker = ASTAwareRanker()
        self._phrase_reranker = PhraseQueryReranker()

    def rerank(self, question, question_ast, candidates, conn, top_k=10):
        if not candidates:
            return RerankerResult(name=self.name, ranked=[], metadata={})
        # v2 composition (after v1 multiplicative-with-AST-filter regressed
        # from R@1=54 to 30): treat AST score as an ADDITIVE BONUS on top
        # of the B_phrase_query baseline. Every BM25 candidate stays in
        # the pool — AST structural filters never eliminate, they only
        # boost compatible candidates. This preserves B_phrase_query's
        # recall while letting structural agreement break ties.
        ast_result = self._ast_reranker.rerank(
            question, question_ast, candidates, conn, top_k=len(candidates))
        ast_score_by_sid = {int(p.sentence_id): float(p.score)
                            for p in ast_result.ranked}
        ents = self._phrase_reranker._question_entities(question, question_ast)

        scored: list[tuple[float, ParsedPassage]] = []
        for p in candidates:
            sid = int(p.sentence_id)
            bm25 = float(p.score)
            text = p.text or ''
            phrase_boost = 0.0
            for e in ents:
                if e and e in text:
                    phrase_boost += 5.0 * (len(e.split()) ** 1.5)
            # B_phrase_query baseline (BM25 + entity-in-text boost)
            base_score = bm25 + phrase_boost
            # AST structural bonus — non-zero only when the candidate
            # passed G_ast_aware's filters. The 5.0 weight matches the
            # per-entity boost magnitude so neither signal dominates.
            ast_s = ast_score_by_sid.get(sid, 0.0)
            ast_bonus = 5.0 * ast_s if ast_s > 0 else 0.0
            scored.append((base_score + ast_bonus, p))
        scored.sort(key=lambda kv: -kv[0])
        ranked = [
            ParsedPassage(sentence_id=p.sentence_id, text=p.text, ast=p.ast,
                          score=ns, source_doc=p.source_doc,
                          source_type=p.source_type)
            for ns, p in scored[:top_k]
        ]
        n_ast_pass = sum(1 for s in ast_score_by_sid.values() if s > 0)
        return RerankerResult(
            name=self.name, ranked=ranked,
            metadata={'entities': ents,
                      'n_ast_filter_pass': n_ast_pass})


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
    ap.add_argument('--output-summary', default=None,
                    help='Optional path to write a per-run summary JSON '
                         '(consumable by perf_history.py append --run-summary)')
    ap.add_argument('--append-history',
                    default=None,
                    help='Optional path to perf history JSONL — appends the '
                         'run summary directly (default: data/perf/bench_history.jsonl)')
    ap.add_argument('--llm-baseline',
                    default=None,
                    help='Ollama model tag (e.g. llama3.2:latest). When set, '
                         'also queries the LLM per question and adds a '
                         'comparable column to the report. Requires Ollama '
                         'running at http://localhost:11434.')
    args = ap.parse_args()

    test = []
    with open(args.test_set) as f:
        for line in f:
            line = line.strip()
            if line:
                test.append(json.loads(line))
    print(f'Loaded {len(test)} test questions from {args.test_set}\n')

    # Set up the optional LLM baseline column
    llm_tag: Optional[str] = None
    llm_chat = None
    if args.llm_baseline:
        # Lazy import to avoid the dep when not used
        from scripts.eval.eval_vs_llm import ollama_chat, check_ollama_alive
        if not check_ollama_alive(args.llm_baseline):
            print(f'ERROR: LLM baseline model {args.llm_baseline!r} not '
                  f'available via Ollama. Run: ollama pull {args.llm_baseline}',
                  file=sys.stderr)
            sys.exit(2)
        llm_tag = args.llm_baseline.replace(':', '_').replace('/', '_')
        llm_chat = ollama_chat
        print(f'LLM baseline column enabled: {args.llm_baseline} '
              f'(tag={llm_tag})')

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
        ASTAwareRanker(),
        HybridReranker(),
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
        # Optional LLM baseline column
        if llm_chat and llm_tag:
            llm_text, llm_lat = llm_chat(args.llm_baseline, question)
            llm_correct = any(kw.lower() in llm_text.lower() for kw in expected)
            row[f'{llm_tag}_answer'] = llm_text[:120]
            row[f'{llm_tag}_correct'] = llm_correct
            row[f'{llm_tag}_latency_s'] = round(llm_lat, 3)
        rows.append(row)
        marks = ' '.join(
            ('✓' if row.get(f'{r.name}_correct') else '·') for r in enabled
        )
        if llm_tag:
            marks += f"  {llm_tag[:10]}{'✓' if row.get(f'{llm_tag}_correct') else '·'}"
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
    # Optional LLM baseline row (no rank metrics — LLM gives a direct
    # answer, not a candidate list). Show '-' for R@K / MRR columns and
    # only the answer accuracy + latency.
    if llm_tag:
        n_correct = sum(1 for row in rows if row.get(f'{llm_tag}_correct'))
        lats = [row.get(f'{llm_tag}_latency_s', 0) for row in rows
                if row.get(f'{llm_tag}_latency_s') is not None]
        avg_lat = sum(lats) / max(1, len(lats))
        n = max(1, len(rows))
        print('-' * 60)
        print(f'{args.llm_baseline:<22s} {"-":>5s} {"-":>5s} {"-":>5s} '
              f'{"-":>6s} {100*n_correct/n:>5.1f}%   (LLM avg_lat={avg_lat:.1f}s)')

    if args.output_jsonl:
        Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_jsonl, 'w') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f'\nPer-question JSONL: {args.output_jsonl}')

    # Emit a per-run summary that perf_history.py can ingest with `append`.
    # Captures: which assets were active, per-reranker metrics, test set, sample size.
    per_reranker_metrics = {}
    for r in enabled:
        ranks = [row.get(f'{r.name}_rank') for row in rows]
        n_r1 = sum(1 for x in ranks if x == 1)
        n_r5 = sum(1 for x in ranks if x is not None and x <= 5)
        n_r10 = sum(1 for x in ranks if x is not None and x <= 10)
        mrr = (sum(1.0 / x for x in ranks if x is not None) / len(ranks)
               if ranks else 0)
        n_correct = sum(1 for row in rows if row.get(f'{r.name}_correct'))
        per_reranker_metrics[r.name] = {
            'recall_at_1':       n_r1,
            'recall_at_5':       n_r5,
            'recall_at_10':      n_r10,
            'mrr':               round(mrr, 4),
            'answer_accuracy':   round(100 * n_correct / max(1, len(rows)), 2),
        }
    # LLM baseline (no rank metrics, just accuracy + latency)
    if llm_tag:
        n_correct = sum(1 for row in rows if row.get(f'{llm_tag}_correct'))
        lats = [row.get(f'{llm_tag}_latency_s', 0) for row in rows
                if row.get(f'{llm_tag}_latency_s') is not None]
        per_reranker_metrics[args.llm_baseline] = {
            'kind':              'llm_baseline',
            'answer_accuracy':   round(100 * n_correct / max(1, len(rows)), 2),
            'avg_latency_s':     round(sum(lats) / max(1, len(lats)), 2),
        }
    run_summary = {
        'test_set':         args.test_set,
        'n_questions':      len(rows),
        'top_k':            args.top_k,
        'candidate_pool':   args.candidate_pool,
        'rerankers':        per_reranker_metrics,
        'skipped':          skipped,
    }
    if args.output_summary:
        Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_summary, 'w') as f:
            json.dump(run_summary, f, ensure_ascii=False, indent=2)
        print(f'Per-run summary:    {args.output_summary}')
    if args.append_history:
        from perf_history import append_run as _append_history
        _append_history(Path(args.append_history), run_summary)


if __name__ == '__main__':
    main()
