#!/usr/bin/env python3
"""
Build synthetic WHO test set from corpus.

VERSION: v2.1
COMPATIBLE WITH: v2.1 Kuzu DB schema (Frazo / Vorto / Vortgrupo nodes)
DEPENDENCIES: klareco.utils.kuzu_open
STAGE: Evaluation

Description:
    Generates a regression-test-style retrieval benchmark by extracting WHO
    questions FROM the corpus. For each candidate sentence with the active-voice
    WHO shape (propra_nomo subject + action verb + object), we generate
    "Kiu {verb}is {object}?" with the subject as the expected answer. By
    construction the answer is in the corpus and the keyword is non-trivially
    related to the question, eliminating both false-positive substring matches
    and corpus-coverage failures from the eval signal.

Pipeline Position:
    Kuzu corpus → [THIS SCRIPT] → JSONL test set → modal_eval.py

Usage:
    python scripts/eval/build_synthetic_who_test_set.py
    python scripts/eval/build_synthetic_who_test_set.py --target-size 200 --seed 42

Inputs:
    Kuzu DB at data/indexes/v2.1_kuzu_index_full

Outputs:
    JSONL test set at data/test_sets/synthetic_who_active.jsonl, one per line:
      {id, question, expected_answer, expected_keywords,
       source_sentence_id, source_sentence_text, question_type='WHO',
       pattern='active', verb_root, object_radiko}

Quality Checks:
    - Subject is a propra_nomo at least 3 chars, not in stop-word list
    - Object is at least 3 chars and != subject
    - Sentence length is 5–40 words
    - Diversity: balanced sample across verb roots
    - Answer text never appears in the generated question

Last Updated: 2026-05-05
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kuzu

from klareco.utils.kuzu_open import open_kuzu

# Verb roots known to take a propra_nomo agent in WHO questions.
# Drawn from semantic-ontology verb classes for "creation", "authorship",
# "performance", "discovery". Avoid generic verbs (est, hav, far) — those
# fire too broadly and produce ambiguous questions.
ACTIVE_VERBS = [
    # creation / founding
    'fond', 'kre', 'establ', 'invent', 'desegn', 'konstruk', 'edif',
    # writing / publishing
    'verk', 'skrib', 'redakt', 'publik', 'eldon',
    # discovery
    'malkovr', 'eltrov',
    # performance / production
    'pentr', 'kompoz', 'reĝisor', 'kant',
    # action with named agent
    'gajn', 'venk',
]

# Reject "subjects" that aren't real names (parser sometimes tags
# sentence-initial capitalized words as propra_nomo even when they're
# articles, conjunctions, or function words).
GENERIC_SUBJECT_RADIKOS = {
    'la', 'lia', 'ŝia', 'sia', 'tiu', 'iu',
    'kaj', 'sed', 'aŭ', 'do', 'tamen', 'tial',
    'mi', 'vi', 'ni', 'ili', 'li', 'ŝi', 'ĝi',
    'tio', 'tiu', 'iom', 'iam', 'kio', 'kiu',
    'la', 'estas', 'estis', 'estos',
}


def query_active_who_candidates(conn, verbs, limit=10000):
    """Query WHO candidates with strict person-name filter.

    The propranoma_kategorio='person' filter ensures we only get sentences
    whose subject is a dictionary-confirmed person name. This is independent
    of any parser fix to the stored data — it filters at query time on the
    metadata field that's set ONLY when the proper-noun dictionary matched.
    """
    verb_list = ','.join(f"'{v}'" for v in verbs)
    cypher = f"""
        MATCH (ft:Frazoteksto)-[:FRAZOTEKSTO_HAVAS_AST]->(a:AST)-[:AST_HAVAS_FRAZON]->(frazo:Frazo)
        MATCH (frazo)-[:HAVAS_SUBJEKTON_VORTGRUPO]->(svg:Vortgrupo)-[:HAVAS_KERNON]->(subj:Vorto)
        WHERE subj.vortspeco = 'propra_nomo'
          AND subj.propranoma_kategorio = 'person'
        MATCH (frazo)-[:HAVAS_VERBON]->(verb:Vorto)
        WHERE verb.radiko IN [{verb_list}]
        MATCH (frazo)-[:HAVAS_OBJEKTON_VORTGRUPO]->(ovg:Vortgrupo)-[:HAVAS_KERNON]->(obj:Vorto)
        RETURN
            ft.id            AS sentence_id,
            ft.teksto        AS sentence_text,
            subj.plena_vorto AS subject_pv,
            subj.radiko      AS subject_radiko,
            verb.radiko      AS verb_root,
            obj.plena_vorto  AS object_pv,
            obj.radiko       AS object_radiko,
            obj.vortspeco    AS object_vortspeco
        LIMIT {limit}
    """
    res = conn.execute(cypher)
    rows = []
    while res.has_next():
        r = res.get_next()
        rows.append({
            'sentence_id':       r[0],
            'sentence_text':     r[1],
            'subject_pv':        r[2],
            'subject_radiko':    r[3],
            'verb_root':         r[4],
            'object_pv':         r[5],
            'object_radiko':     r[6],
            'object_vortspeco':  r[7],
        })
    return rows


def is_quality_candidate(c) -> bool:
    subj = c.get('subject_pv') or ''
    obj  = c.get('object_pv') or ''
    text = c.get('sentence_text') or ''

    if len(subj) < 3 or len(obj) < 3:
        return False
    if (c.get('subject_radiko') or '').lower() in GENERIC_SUBJECT_RADIKOS:
        return False
    if not subj[0].isupper():
        return False
    if obj.lower() == subj.lower():
        return False
    word_count = len(text.split())
    if word_count < 5 or word_count > 40:
        return False
    if subj not in text:
        return False
    # Don't accept sentences where the kerno propra is the same as the
    # object (rare but happens with patient-as-subject parses).
    if c.get('subject_radiko') == c.get('object_radiko'):
        return False
    return True


def make_question(c) -> str:
    verb_root = c['verb_root']
    obj_pv = c['object_pv']
    if not obj_pv.endswith('n'):
        obj_text = obj_pv + 'n'
    else:
        obj_text = obj_pv
    return f"Kiu {verb_root}is {obj_text}?"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--kuzu-path',   default='data/indexes/v2.1_kuzu_index_full')
    parser.add_argument('--output',      default='data/test_sets/synthetic_who_active.jsonl')
    parser.add_argument('--target-size', type=int, default=200)
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--query-limit', type=int, default=10000,
                        help='Cap on raw candidates fetched from Kuzu')
    args = parser.parse_args()

    print(f"Opening Kuzu DB: {args.kuzu_path}")
    db = open_kuzu(args.kuzu_path)
    conn = kuzu.Connection(db)

    print(f"Querying corpus for active-voice WHO candidates "
          f"(verbs={ACTIVE_VERBS}, limit={args.query_limit})...")
    candidates = query_active_who_candidates(conn, ACTIVE_VERBS,
                                              limit=args.query_limit)
    print(f"  Raw candidates: {len(candidates)}")

    filtered = [c for c in candidates if is_quality_candidate(c)]
    print(f"  After quality filter: {len(filtered)}")

    if not filtered:
        raise SystemExit("ERROR: no candidates passed filter — check verb list / corpus")

    # Diversity: cap per-verb so common verbs don't dominate.
    random.seed(args.seed)
    by_verb = defaultdict(list)
    for c in filtered:
        by_verb[c['verb_root']].append(c)
    n_verbs = len(by_verb)
    per_verb = max(1, args.target_size // n_verbs)

    pool = []
    for verb_root, items in by_verb.items():
        random.shuffle(items)
        pool.extend(items[:per_verb])

    # If under target, top up from leftovers.
    if len(pool) < args.target_size:
        leftovers = [c for c in filtered if c not in pool]
        random.shuffle(leftovers)
        pool.extend(leftovers[: args.target_size - len(pool)])

    random.shuffle(pool)
    pool = pool[: args.target_size]

    # Generate entries; skip any where the answer leaked into the question.
    output = []
    for i, c in enumerate(pool, 1):
        q = make_question(c)
        if c['subject_pv'].lower() in q.lower():
            continue
        output.append({
            'id':                   f'who_active_{i:03d}',
            'question':             q,
            'expected_answer':      c['subject_pv'],
            'expected_keywords':    [c['subject_pv']],
            'source_sentence_id':   c['sentence_id'],
            'source_sentence_text': c['sentence_text'],
            'question_type':        'WHO',
            'pattern':              'active',
            'verb_root':            c['verb_root'],
            'object_radiko':        c['object_radiko'],
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        for entry in output:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\nWrote {len(output)} questions to {out_path}")

    # Per-verb breakdown for sanity
    print(f"\nVerb-root distribution in output:")
    by_verb_out = defaultdict(int)
    for e in output:
        by_verb_out[e['verb_root']] += 1
    for v, n in sorted(by_verb_out.items(), key=lambda kv: -kv[1]):
        print(f"  {v:12s}  {n}")


if __name__ == '__main__':
    main()
