#!/usr/bin/env python3
"""
A/B substrate: one store, three tokenizations, same BM25 engine. (#736)

VERSION: v1.1
COMPATIBLE WITH: v2.3+ store (dependency_arcs + ontology_edges populated)
DEPENDENCIES: duckdb (fts extension)
STAGE: Index / experiment

Description:
    WHY BM25 IS NOT THE PROBLEM — THE TOKENS ARE. The live Whoosh index stores
    surface text, so `fondiĝis` (question) and `fondita` (corpus) — same root
    `fond`, different inflection — never match, and 62% of natural-trivia gold
    sentences fall out of the top-200. This builds ONE DuckDB/FTS store carrying
    the SAME sentences under three tokenizations, so BM25 is A/B'd over each with
    everything else held constant:

      surface   the raw sentence text            (BASELINE = what we have today)
      roots     content-word radikoj             (morphology: collapses inflection)
      rootclass roots + ontology class/type ids   (synonymy: `fond` and `kre` both
                                                   emit `kreado-26`, so a class-id
                                                   token matches synonymous verbs
                                                   WITHOUT enumerating synonyms)

    v1.1: built from `dependency_arcs`, NOT by re-expanding ast_json — the arc
    forest already holds every token's root and role. 4.6M sentences in seconds
    instead of ~2.5 h. Function words are dropped BY ROLE: heads are content
    predicates/nouns; a dependent is kept only in a CONTENT relation (nsubj/obj/
    nmod/…), so `case`/`det`/`cc`/`cop`/`aux`/`mark` tokens (la/de/kaj/est-cop)
    never enter — the Function Word Exclusion Principle, read off the tree.

Pipeline Position:
    dependency_arcs + ontology_edges --[THIS]--> retrieval_ab.db (FTS) --> ab_recall.py

Usage:
    python scripts/index/build_retrieval_ab_index.py --apply

Outputs:
    - data/indexes/retrieval_ab.db  (table ab_docs + one 3-field FTS index)

Last Updated: 2026-07-17
Related Issues: #713, #736, #737
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb

SRC = 'data/indexes/duckdb_store.db'
OUT = 'data/indexes/retrieval_ab.db'

# A dependent is a content word only in these relations. Everything else
# (case/det/cc/cop/aux/mark/punct/dep/root/…) is grammatical, not semantic.
CONTENT_ROLES = ('nsubj', 'nsubj:pass', 'obj', 'iobj', 'nmod', 'obl', 'amod',
                 'nummod', 'advmod', 'conj', 'appos', 'acl', 'advcl', 'xcomp',
                 'ccomp', 'parataxis')
CLASS_RELS = ('APARTENAS_AL_VERBA_KLASO', 'HAVAS_ENTECAN_TIPON')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--src', default=SRC)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--threads', type=int, default=6)
    args = ap.parse_args()
    if not args.apply:
        print('  dry run — pass --apply to build'); return 0

    if Path(args.out).exists():
        Path(args.out).unlink()
    con = duckdb.connect(args.out)
    con.execute(f'SET threads={args.threads}')
    con.execute('INSTALL fts; LOAD fts;')
    con.execute(f"ATTACH '{args.src}' AS src (READ_ONLY)")

    roles = ','.join(f"'{r}'" for r in CONTENT_ROLES)
    rels = ','.join(f"'{r}'" for r in CLASS_RELS)

    print('  building ab_docs from the arc forest (surface / roots / rootclass)…')
    t = time.time()
    con.execute(f"""
        CREATE TABLE ab_docs AS
        WITH tok AS (   -- distinct content roots per sentence, function words
                        -- excluded BY ROLE (heads + content-relation dependents)
          SELECT DISTINCT sid, r FROM (
            SELECT sid, kapo_radiko AS r FROM src.dependency_arcs
            UNION ALL
            SELECT sid, dep_radiko AS r FROM src.dependency_arcs
              WHERE rolo IN ({roles})
          ) WHERE r <> ''
        ),
        cls AS (        -- each root's ontology class/type id, as a shared token
          SELECT t.sid, oe.class_id AS r
          FROM tok t JOIN src.ontology_edges oe
            ON oe.radiko = t.r AND oe.rel IN ({rels})
        ),
        roots_agg AS (SELECT sid, string_agg(r, ' ') AS roots FROM tok GROUP BY sid),
        rc_agg AS (SELECT sid, string_agg(r, ' ') AS rootclass
                   FROM (SELECT sid, r FROM tok UNION ALL SELECT sid, r FROM cls)
                   GROUP BY sid)
        SELECT s.sid, s.text AS surface,
               COALESCE(ra.roots, '')     AS roots,
               COALESCE(rc.rootclass, '') AS rootclass
        FROM src.sentences s
        LEFT JOIN roots_agg ra ON ra.sid = s.sid
        LEFT JOIN rc_agg    rc ON rc.sid = s.sid
    """)
    n = con.execute('SELECT count(*) FROM ab_docs').fetchone()[0]
    empty = con.execute("SELECT count(*) FROM ab_docs WHERE roots=''").fetchone()[0]
    print(f'    {n:,} rows in {time.time() - t:.0f}s   ({empty:,} with no content roots)')

    con.execute('DETACH src')
    print('  building FTS index over surface / roots / rootclass…')
    t = time.time()
    # stemmer='none': an English Porter stemmer on Esperanto is wrong, and it
    # would corrupt the already-exact `roots` field. stopwords='none': function
    # words are already excluded by role in `roots`, and we want `surface` raw.
    con.execute("PRAGMA create_fts_index('ab_docs','sid',"
                "'surface','roots','rootclass', "
                "stemmer='none', stopwords='none', overwrite=1)")
    print(f'    fts built in {time.time() - t:.0f}s')

    stats = con.execute("""
        SELECT avg(length(surface)-length(replace(surface,' ',''))+1),
               avg(length(roots)-length(replace(roots,' ',''))+1),
               avg(length(rootclass)-length(replace(rootclass,' ',''))+1)
        FROM ab_docs WHERE roots <> ''""").fetchone()
    print(f'\n  mean tokens/field — surface {stats[0]:.1f}  roots {stats[1]:.1f}  '
          f'rootclass {stats[2]:.1f}')
    print('  ✓ retrieval_ab.db ready.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
