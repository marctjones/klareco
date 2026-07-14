#!/usr/bin/env python3
"""
Load the ontology into the store — ontology_nodes, ontology_edges, verb_klaso.

VERSION: v1.0
COMPATIBLE WITH: v2.3 store; data/raw/eo/dictionaries/revo_ontology.json
DEPENDENCIES: duckdb; klareco.ontology
STAGE: Index

Description:
    CLAUDE.md:

        "`ontology_nodes` and `ontology_edges` are EMPTY and `verb_klaso` is 0%
         populated ... the 'always query the ontology' rule is UNFOLLOWABLE."

    This makes it followable. The data is curated (ReVo, GPL-2.0) and already
    extracted by scripts/acquire/acquire_revo_ontology.py:

        8,709 hypernym edges · 2,984 synonyms · 22,770 domain labels (78 classes)
        133 typed entity lists · 40,230 senses
        persono: 377 members (was FOUR hand-picked roots)
        loko:    706 members (was a hardcoded gazetteer)

    plus voko-akrido's SEMANTIC TYPE HIERARCHY (`best` ⊂ `subst`, `pers` ⊂ `best`,
    `parc` ⊂ `pers`) and its TYPED ROOTS, which are what took morpheme ambiguity
    from 32% to 0.285%.

    ⚠️ THIS IS WHAT UN-TIES THE RERANKERS, AND THE MECHANISM IS EASY TO MUDDLE:

        `subj_radiko` is a SHARED input. Fixing it raises the floor for all nine
        rerankers EQUALLY — it un-ties nothing.

        `verb_klaso` (0%), `verb_negated` (1.1%) and entity-type gating are the
        DIFFERENTIATING inputs, and they are dead. THAT is why all nine score
        identically.

    NO REPARSE NEEDED — this reads the clause table and the curated JSON.

Usage:
    python scripts/index/load_ontology.py

Outputs:
    - tables `ontology_nodes`, `ontology_edges` in the store
    - `clauses.verb_klaso` populated

Last Updated: 2026-07-14
Related Issues: #837, #830, #777, #780, EPIC #713
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb

from klareco.ontology import ENTITY_CLASSES, ontology

DB = 'data/indexes/duckdb_store.db'


def main() -> int:
    ap = argparse.ArgumentParser(description='Load the ontology into the store')
    ap.add_argument('--duckdb-path', default=DB)
    args = ap.parse_args()

    onto = ontology()
    con = duckdb.connect(args.duckdb_path)

    # ---- ontology_nodes: one row per root, with its class and domains ----
    con.execute('DROP TABLE IF EXISTS ontology_nodes')
    con.execute("""
        CREATE TABLE ontology_nodes (
          radiko    VARCHAR,
          klaso     VARCHAR,   -- persono | loko | besto | …  (from ReVo's voko: lists)
          fako      VARCHAR,   -- ZOO | BOT | MED | GEOG | …  (78 of them)
          n_sencoj  INTEGER    -- how many senses. >1 == a SENSE OR-node.
        )""")
    nodes = []
    for root, v in onto.roots.items():
        klasoj = onto.classes(root) or {None}
        fakoj = v.get('domains') or [None]
        for k in klasoj:
            nodes.append((root, k, fakoj[0], len(v.get('senses') or [])))
    con.executemany('INSERT INTO ontology_nodes VALUES (?,?,?,?)', nodes)

    # ---- ontology_edges: the TAXONOMY -----------------------------------
    #
    # ⚠️ THE SCHEMA IS (rel, radiko, class_id). NOT (de, al, rel).
    #
    # This table previously wrote (de, al, rel) — a perfectly reasonable shape, and
    # NOT the one every consumer in the codebase reads. `ast_aware_reranker.py` and
    # `ast_retriever.py` both do:
    #
    #     SELECT class_id FROM ontology_edges
    #     WHERE rel = 'APARTENAS_AL_VERBA_KLASO' AND radiko = ?
    #
    # which raised `BinderException: column "radiko" not found` — and both wrap the
    # call in a BARE `except: return None`. So the verb-class signal came back None
    # on every row, silently, and the nine rerankers stayed tied EVEN AFTER the
    # ontology was "loaded". We would have run the A/B, seen nine identical scores,
    # and concluded that verb class does not help. It was never asked.
    #
    # `build_duckdb_store.ensure_schema` already declares the correct shape. This
    # script was overwriting it.
    con.execute('DROP TABLE IF EXISTS ontology_edges')
    con.execute("""
        CREATE TABLE ontology_edges (
          rel       VARCHAR,   -- APARTENAS_AL_VERBA_KLASO | HAVAS_ENTECAN_TIPON
                               -- | HAVAS_SUPERKLASON | SINONIMO
          radiko    VARCHAR,   -- the root
          class_id  VARCHAR    -- the class / target
        )""")
    edges = []
    for root, v in onto.roots.items():
        for h in v.get('hypernyms') or []:
            edges.append(('HAVAS_SUPERKLASON', root, h))
        for s in v.get('synonyms') or []:
            edges.append(('SINONIMO', root, s))
    for klaso in ENTITY_CLASSES:
        for m in onto.members(klaso):
            edges.append(('HAVAS_ENTECAN_TIPON', m, klaso))

    # ---- APARTENAS_AL_VERBA_KLASO — the DIFFERENTIATING signal -----------
    #
    # This is the edge the rerankers actually read, and it did not exist. The class
    # definitions survive as Python literals in
    # `scripts/index/extend_kuzu_schema_semantic_ontology.py` (CLAUDE.md: "restoring
    # the ontology means emitting a snapshot from those literals, not re-deriving
    # it"). They are HAND-SEEDED AND THIN — 8 classes, 4 roots each, 32 roots total.
    #
    # So we expand them the one way that is not just more hand-listing: through
    # ReVo's CURATED SYNONYM edges. If `fond` is in `kreado-26`, its ReVo synonyms
    # are creation verbs too — that is a lexicographer's judgement, not ours.
    #
    # It is still thin, and the coverage number below says exactly how thin. A null
    # result in the A/B must be read against it: a signal present on 2% of clauses
    # cannot move a metric, and that is a fact about COVERAGE, not about whether
    # verb class is useful.
    VERB_CLASSES = {
        'kreado-26':      ['fond', 'kre', 'produk', 'far'],
        'movo-51':        ['ir', 'ven', 'kur', 'voj'],
        'pensado-29':     ['pens', 'sci', 'kred', 'komprend'],
        'perceptado-30':  ['vid', 'aŭd', 'sent', 'gust'],
        'emocio-31':      ['am', 'ĝoj', 'tim', 'trist'],
        'komunikado-37':  ['dir', 'parol', 'demand', 'respond'],
        'vivo-48':        ['viv', 'mort', 'nask', 'kresk'],
        'profesio-50':    ['labor', 'instru', 'kurac', 'vend'],
    }
    vk: dict[str, str] = {}
    for klaso, roots in VERB_CLASSES.items():
        for r in roots:
            vk.setdefault(r, klaso)
    seeded = len(vk)
    # one hop through ReVo's curated synonyms
    for r, klaso in list(vk.items()):
        for s in onto.synonyms(r) or []:
            vk.setdefault(s, klaso)
    for r, klaso in vk.items():
        edges.append(('APARTENAS_AL_VERBA_KLASO', r, klaso))

    con.executemany('INSERT INTO ontology_edges VALUES (?,?,?)', edges)
    con.execute('CREATE INDEX idx_onodes_radiko ON ontology_nodes(radiko)')
    con.execute('CREATE INDEX idx_oedges_radiko ON ontology_edges(radiko)')

    print(f'  ontology_nodes : {len(nodes):,} rows   (was 0)')
    print(f'  ontology_edges : {len(edges):,} rows   (was 0)')
    for k in ENTITY_CLASSES:
        print(f'    {k:10s} {len(onto.members(k)):5,} members')
    print(f'\n  APARTENAS_AL_VERBA_KLASO : {len(vk)} roots in {len(VERB_CLASSES)} classes')
    print(f'    {seeded} hand-seeded (the surviving literals)')
    print(f'    {len(vk) - seeded} added via ReVo synonyms (curated, not hand-listed)')

    # ---- entity_facts ----------------------------------------------------
    # The table BiographyFormatStage reads and CRASHES on, because it does not
    # exist. Seed it from the ontology: root -> entity class, with the ReVo sense
    # as the fact. That is not a biography, but it is a real table with real rows,
    # and the stage stops crashing — which is the difference between "broken" and
    # "thin".
    con.execute('DROP TABLE IF EXISTS entity_facts')
    con.execute("""
        CREATE TABLE entity_facts (
          entito   VARCHAR,   -- the root
          klaso    VARCHAR,   -- persono | loko | …
          fakto    VARCHAR,   -- the ReVo definition
          fonto    VARCHAR    -- attribution (VISION.md)
        )""")
    facts = []
    for klaso in ENTITY_CLASSES:
        for m in onto.members(klaso):
            senses = onto.senses(m)
            facts.append((m, klaso, senses[0] if senses else None, 'revo'))
    con.executemany('INSERT INTO entity_facts VALUES (?,?,?,?)', facts)
    con.execute('CREATE INDEX idx_efacts_entito ON entity_facts(entito)')
    print(f'  entity_facts   : {len(facts):,} rows   (table did not EXIST)')

    # ---- verb_klaso on the clause table ---------------------------------
    # The verb's SEMANTIC CLASS comes from the typed root lexicon (voko-akrido):
    # tr / ntr / best / pers / parc / subst. This is the DIFFERENTIATING input the
    # rerankers read and that has been 0% populated since the migration.
    have = {t[0] for t in con.execute('SHOW TABLES').fetchall()}
    if 'clauses' not in have:
        print('\n  ⚠️  no `clauses` table — run build_clause_table.py first (#836).')
        return 0

    from klareco.morphology import lexicon
    lex = lexicon()
    pairs = [(r, p) for r, p in lex.roots.items() if p]
    con.execute('DROP TABLE IF EXISTS _root_pos')
    con.execute('CREATE TABLE _root_pos (radiko VARCHAR, pos VARCHAR)')
    con.executemany('INSERT INTO _root_pos VALUES (?,?)', pairs)
    con.execute("""
        UPDATE clauses SET verb_klaso = (
            SELECT pos FROM _root_pos WHERE _root_pos.radiko = clauses.verb_radiko
        )""")
    con.execute('DROP TABLE _root_pos')

    n = con.execute(
        'SELECT count(*) FROM clauses WHERE verb_klaso IS NOT NULL').fetchone()[0]
    tot = con.execute('SELECT count(*) FROM clauses').fetchone()[0]
    print(f'\n  verb_klaso     : {n:,}/{tot:,} = {n / tot:.1%}   (was 0.0%)')
    print('\n  ⚠️  This is the DIFFERENTIATING input the nine rerankers read.')
    print('     `subj_radiko` is a SHARED input — fixing it raises the floor for all')
    print('     nine EQUALLY and un-ties nothing. THIS is what makes them differ.')
    print('     If they still score identically, the ontology did not reach them,')
    print('     and that is a finding worth having.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
