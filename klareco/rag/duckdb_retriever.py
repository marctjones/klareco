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

from klareco.parser import expand_ast

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
        # Cached searcher + parser. Opening a Whoosh searcher re-opens the segment
        # files (~220 ms measured 2026-07-17 — 15% of retrieval wall clock); the
        # index is read-only in this process, so one long-lived searcher is safe.
        self._cached_searcher = None
        self._qp = QueryParser('text', self.ix.schema, group=OrGroup)

    def _searcher(self):
        if self._cached_searcher is None:
            self._cached_searcher = self.ix.searcher(weighting=scoring.BM25F())
        return self._cached_searcher

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
        #    The searcher and parser are CACHED (see _searcher()): opening a
        #    searcher re-opens the segment files (~220 ms measured) and the index
        #    is read-only here, so paying that per query was pure waste.
        t0 = time.time()
        cand_ids: List[int] = []
        s = self._searcher()
        q = self._qp.parse(' OR '.join(terms))
        for hit in s.search(q, limit=max(top_k * 15, 300)):
            try:
                cand_ids.append(int(hit['id']))
            except (KeyError, ValueError):
                continue
        self._phase_timer.add('whoosh_query', (time.time() - t0) * 1000)
        if not cand_ids:
            return []

        # 2. ONE indexed DuckDB fetch for the candidates. Wrapped in a
        # try/except + chunked fallback to survive a corrupt block: if a
        # single batch SELECT hits the bad block we fall back to per-sid
        # fetches and skip the ones that throw. The DB has a known-bad
        # block at file offset ~30GB (2026-05-21 incident); rebuild is
        # pending. This wrap prevents one bad block from blocking ALL
        # retrieval.
        # ⚠️ #864: the candidate fetch reads ONLY the light shredded columns the
        # boost needs. text + ast_json are hauled in a SECOND query for the final
        # top_k only — fetching ast_json for all top_k*15 (=1,500) candidates
        # moved ~3 MB of compact blobs per query to score rows we discard
        # (measured: DuckDB dominated 82.6% of retrieval wall clock).
        t0 = time.time()
        placeholders = ','.join('?' * len(cand_ids))
        try:
            rows = self.con.execute(
                f"SELECT sid, subj_vortspeco, obj_radiko "
                f"FROM sentences WHERE sid IN ({placeholders})",
                cand_ids,
            ).fetchall()
        except Exception as e:
            # Likely corrupt-block IO. Fall back to per-sid fetch, dropping
            # the ones that throw.
            logger.warning(f"Bulk fetch failed ({e}); falling back to per-sid")
            rows = []
            for sid in cand_ids:
                try:
                    r = self.con.execute(
                        "SELECT sid, subj_vortspeco, obj_radiko "
                        "FROM sentences WHERE sid = ?", [sid]).fetchone()
                    if r:
                        rows.append(r)
                except Exception:
                    continue
        self._phase_timer.add('duckdb_fetch', (time.time() - t0) * 1000)

        # rank-position from Whoosh = base relevance signal
        rank_of = {sid: i for i, sid in enumerate(cand_ids)}

        # 3. Light question-type bias on shredded columns. Deep AST-role
        #    boosting is left to DeterministicRerankStage (it now gets a
        #    real AST for free from ast_json).
        t0 = time.time()
        q_obj = _kerno((question_ast.get('objekto') or {})).get('radiko')
        scored = []
        for sid, subj_vs, obj_r in rows:
            base = 1.0 / (1 + rank_of.get(sid, len(cand_ids)))
            boost = 1.0
            if qtype in ('KIU', 'KIE') and subj_vs == 'propra_nomo':
                boost += 0.5
            if q_obj and obj_r == q_obj:
                boost += 0.5
            scored.append({
                'id': str(sid),
                'text': None,             # filled for the top_k below
                'ast': None,
                'score': base * boost * 10.0,
                'source': 'duckdb',
            })
        scored.sort(key=lambda r: r['score'], reverse=True)
        top = scored[:top_k]

        # Second fetch: text + ast_json for the WINNERS only (~top_k rows, not 15x).
        top_ids = [int(r['id']) for r in top]
        heavy: dict = {}
        if top_ids:
            ph = ','.join('?' * len(top_ids))
            try:
                for sid, text, aj in self.con.execute(
                        f"SELECT sid, text, ast_json FROM sentences "
                        f"WHERE sid IN ({ph})", top_ids).fetchall():
                    heavy[sid] = (text, aj)
            except Exception as e:
                logger.warning(f"Top-k heavy fetch failed ({e}); per-sid fallback")
                for sid in top_ids:
                    try:
                        r = self.con.execute(
                            "SELECT sid, text, ast_json FROM sentences "
                            "WHERE sid = ?", [sid]).fetchone()
                        if r:
                            heavy[r[0]] = (r[1], r[2])
                    except Exception:
                        continue
        for r in top:
            text, aj = heavy.get(int(r['id']), (None, None))
            r['text'] = text
            try:
                ast = json.loads(aj) if aj else None
                # ⚠️ The store carries the COMPACT form (compact_ast). Consumers
                # read the EXPANDED shape and silently extract ZERO facts from a
                # compact dict — the answer_accuracy=0.0% bug (#851). Expand here,
                # at the single choke point. Guarded: expand_ast is not idempotent.
                if ast is not None and ('subjekto_id' in ast or 'verbo_id' in ast
                                        or 'objekto_id' in ast):
                    ast = expand_ast(ast)
                r['ast'] = ast
            except Exception:
                r['ast'] = None
        self._phase_timer.add('score', (time.time() - t0) * 1000)
        return top
