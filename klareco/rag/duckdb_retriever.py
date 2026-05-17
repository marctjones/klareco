"""DuckDB-backed retriever — the Kuzu replacement.

Pipeline per query (drop-in for RetrieveStage; same contract as the
retired WhooshRetriever.retrieve_with_ast_roles):

  1. Whoosh BM25 over the question's content terms  -> candidate sids
  2. ONE indexed DuckDB fetch for those sids         -> shredded cols
                                                        + ast_json blob
  3. light question-type bias using the shredded columns
     (deep AST-role boosting stays in DeterministicRerankStage, which
      now gets a real AST for free via json.loads(ast_json))
  4. return [{id, text, ast, score}], ast = json.loads(ast_json)

Measured rationale (2026-05): replaces a 7-query unindexed graph
traversal (~2.6 s warm) + ~17 s/AST KuzuASTReconstructor with one
indexed filter (~8 ms) + ~0.9 ms/AST blob deserialize.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import duckdb
from whoosh import scoring
from whoosh.index import open_dir
from whoosh.qparser import OrGroup, QueryParser

logger = logging.getLogger(__name__)

# Esperanto question/function words dropped from the BM25 query.
_STOP = set('kiu kio kie kiam kiom kial kiel kiuj kion estas estis estos '
            'la de en al el ĉu por kaj aŭ ke ne je da'.split())


class _PhaseTimer:
    """Minimal stand-in for orchestrator.phase_timer.PhaseTimer so
    RetrieveStage's getattr(retriever,'_phase_timer').snapshot() works."""
    def __init__(self):
        self._t: Dict[str, float] = {}

    def reset(self):
        self._t = {}

    def add(self, name: str, ms: float):
        self._t[name] = self._t.get(name, 0.0) + ms

    def snapshot(self) -> Dict[str, float]:
        return dict(self._t)


def _content_terms(q: str) -> List[str]:
    toks = re.findall(r"[\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+", q.lower())
    return [t for t in toks if t not in _STOP and len(t) > 2]


def _kerno(node) -> dict:
    if not isinstance(node, dict):
        return {}
    if node.get('tipo') == 'vortgrupo':
        return node.get('kerno') or {}
    return node


class DuckDBRetriever:
    def __init__(self, whoosh_index_dir: Path, duckdb_path: Path):
        self.whoosh_index_dir = Path(whoosh_index_dir)
        self.duckdb_path = Path(duckdb_path)
        logger.info("Loading Whoosh index from %s", whoosh_index_dir)
        self.ix = open_dir(str(whoosh_index_dir))
        logger.info("Connecting to DuckDB store at %s", duckdb_path)
        # read_only so many eval workers can share the file.
        self.con = duckdb.connect(str(duckdb_path), read_only=True)
        self._phase_timer = _PhaseTimer()

    # --- question analysis -------------------------------------------------
    @staticmethod
    def _question_type(question_ast: Dict) -> str:
        subj = question_ast.get('subjekto') or {}
        k = _kerno(subj)
        if k.get('vortspeco') == 'korelativo':
            return (k.get('radiko') or '').upper()
        # fall back to scanning aliaj for a ki- correlative
        for a in question_ast.get('aliaj') or []:
            w = _kerno(a)
            if w.get('vortspeco') == 'korelativo':
                return (w.get('radiko') or '').upper()
        return 'UNKNOWN'

    @staticmethod
    def _question_text(question_ast: Dict) -> str:
        words = []

        def walk(n):
            if not isinstance(n, dict):
                return
            if n.get('tipo') == 'vorto':
                pv = n.get('plena_vorto')
                if pv:
                    words.append(pv)
                return
            if n.get('kerno'):
                walk(n['kerno'])
            for c in n.get('priskriboj', []) or []:
                walk(c)
        for key in ('subjekto', 'verbo', 'objekto'):
            if question_ast.get(key):
                walk(question_ast[key])
        for a in question_ast.get('aliaj') or []:
            walk(a)
        return ' '.join(words)

    # --- retrieval ---------------------------------------------------------
    def retrieve_with_ast_roles(self, question_ast: Dict,
                                top_k: int = 20) -> List[Dict]:
        self._phase_timer.reset()
        qtext = self._question_text(question_ast)
        qtype = self._question_type(question_ast)
        terms = _content_terms(qtext)
        if not terms:
            return []

        # 1. Whoosh BM25 -> candidate sids (wide net; DuckDB refines).
        t0 = time.time()
        cand_ids: List[int] = []
        with self.ix.searcher(weighting=scoring.BM25F()) as s:
            qp = QueryParser('text', self.ix.schema, group=OrGroup)
            q = qp.parse(' OR '.join(terms))
            for hit in s.search(q, limit=max(top_k * 15, 300)):
                try:
                    cand_ids.append(int(hit['id']))
                except (KeyError, ValueError):
                    continue
        self._phase_timer.add('whoosh_query', (time.time() - t0) * 1000)
        if not cand_ids:
            return []

        # 2. ONE indexed DuckDB fetch for the candidates.
        t0 = time.time()
        placeholders = ','.join('?' * len(cand_ids))
        rows = self.con.execute(
            f"SELECT sid, text, verb_radiko, subj_vortspeco, "
            f"subj_propranoma_kat, obj_radiko, ast_json "
            f"FROM sentences WHERE sid IN ({placeholders})",
            cand_ids,
        ).fetchall()
        self._phase_timer.add('duckdb_fetch', (time.time() - t0) * 1000)

        # rank-position from Whoosh = base relevance signal
        rank_of = {sid: i for i, sid in enumerate(cand_ids)}

        # 3. Light question-type bias on shredded columns. Deep AST-role
        #    boosting is left to DeterministicRerankStage (it now gets a
        #    real AST for free from ast_json).
        t0 = time.time()
        q_obj = _kerno((question_ast.get('objekto') or {})).get('radiko')
        scored = []
        for sid, text, verb_r, subj_vs, subj_kat, obj_r, ast_json in rows:
            base = 1.0 / (1 + rank_of.get(sid, len(cand_ids)))
            boost = 1.0
            if qtype in ('KIU', 'KIE') and subj_vs == 'propra_nomo':
                boost += 0.5
            if q_obj and obj_r == q_obj:
                boost += 0.5
            try:
                ast = json.loads(ast_json) if ast_json else None
            except Exception:
                ast = None
            scored.append({
                'id': str(sid),
                'text': text,
                'ast': ast,
                'score': base * boost * 10.0,
                'source': 'duckdb',
            })
        scored.sort(key=lambda r: r['score'], reverse=True)
        self._phase_timer.add('score', (time.time() - t0) * 1000)
        return scored[:top_k]
