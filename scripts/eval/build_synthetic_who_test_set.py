#!/usr/bin/env python3
"""
Build synthetic WHO test set from corpus.

VERSION: v2.x (DuckDB)
COMPATIBLE WITH: DuckDB store (sentences: shredded cols + ast_json blob)
DEPENDENCIES: duckdb, Whoosh index, klareco.parser
STAGE: Evaluation

Description:
    Generates a regression-test-style retrieval benchmark by extracting WHO
    questions FROM the corpus. For each candidate sentence with the active-voice
    WHO shape (propra_nomo subject + action verb + object), we generate
    "Kiu {verb}is {object}?" with the subject as the expected answer.

    Two correctness gates make every pair measurable:
      1. parser-AST agent verification (the propra is the true subject of
         the templated verb; full proper-name answer) — kills incoherent
         and name-split pairs.
      2. EMPIRICAL DISCRIMINABILITY (added 2026-05-19): a pair is kept
         only if a raw BM25 query on its question terms surfaces the
         source sentence within a generous top-K of the full 5.4M index.
         The gold_anchor_50 autopsy proved templated questions over
         high-frequency verbs/objects ("Kiu kreis verkojn?") are
         information-theoretically unretrievable — no pipeline change can
         fix an under-specified question. This gate excludes the
         impossible class while keeping hard-but-possible pairs, so the
         resulting set measures the pipeline, not test-set pathology.

Pipeline Position:
    DuckDB store + Whoosh → [THIS SCRIPT] → JSONL test set → evaluate_extractive_qa.py

Usage:
    python scripts/eval/build_synthetic_who_test_set.py
    python scripts/eval/build_synthetic_who_test_set.py --target-size 200 --seed 42

Inputs:
    DuckDB store at data/indexes/duckdb_store.db
    Whoosh index at data/indexes/whoosh_v2

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

Last Updated: 2026-05-19
Author: Claude Code (with Marc Jones)

CHANGELOG:
# 2026-05-19: Ported Kuzu Cypher -> DuckDB SQL (Kuzu retired); added the
#             empirical discriminability gate (gold_anchor_50 autopsy).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb
from whoosh import scoring
from whoosh.index import open_dir
from whoosh.qparser import OrGroup, QueryParser

from klareco.parser import parse

# A proper noun governed by one of these prepositions is NOT the agent
# of the clause (companion / topic / location): "kun Manuel", "pri
# Johano", "en Leipzig". Largest incoherence class in the old generator.
_PREP_GOVERNED = {
    'kun', 'pri', 'de', 'per', 'en', 'sub', 'al', 'ĉe', 'post',
    'antaŭ', 'el', 'tra', 'inter', 'kontraŭ', 'apud', 'laŭ', 'pro',
}
_DEMONSTRATIVE = {
    'tia', 'tiu', 'ĉi', 'ĝi', 'ili', 'jen', 'tio', 'ĉio', 'io',
    'la', 'lia', 'ŝia', 'sia', 'nia', 'via', 'mia',
}
_JUNK_MARKERS = ('[', ']', 'REDIRECT', 'ALIDIREKTI', 'ALIDIREKTU',
                 ' ekzemple:', 'Skribu ', '#')


_PARTICIPLE_RE = re.compile(r'(it|at|ot|int|ant|ont|unt)a$', re.I)


def _looks_namelike(name: str) -> bool:
    """Deterministic, gazetteer-free name check. The proper-noun FP class
    (Track B's documented NOUN->propra_nomo) is words the parser only
    flips to propra_nomo in sentence-initial reanalysis: parsed in
    ISOLATION, a genuine name stays propra_nomo while 'Estis'->verbo,
    'Klubo'->substantivo, 'von'->substantivo. Also reject participle/
    adjective morphology ('Diplomita' = -ita). The remaining incoherence
    (nonsensical verb-object, e.g. 'kantis diplomon') is selectional
    preference = the documented learned-model boundary, not deterministic.
    """
    tok = name.split()[0] if name else ''
    if len(tok) < 3 or not tok[:1].isupper():
        return False
    if _PARTICIPLE_RE.search(tok.lower()):
        return False
    ast = parse(tok)
    for key in ('subjekto', 'verbo', 'objekto'):
        n = ast.get(key)
        if not n:
            continue
        v = (n.get('kerno') if isinstance(n, dict)
             and n.get('tipo') == 'vortgrupo' else n)
        if isinstance(v, dict) and v.get('tipo') == 'vorto':
            return v.get('vortspeco') == 'propra_nomo'
    for x in ast.get('aliaj') or []:
        v = (x.get('kerno') if isinstance(x, dict)
             and x.get('tipo') == 'vortgrupo' else x)
        if isinstance(v, dict):
            return v.get('vortspeco') == 'propra_nomo'
    return False


def verify_with_parser(c: dict) -> dict | None:
    """Re-parse the source sentence; accept only when the candidate
    proper noun is genuinely the AST subject (agent) of the templated
    verb. Returns a corrected candidate (full-name answer) or None.
    Kills every defect class from the manual 200-question audit.
    """
    text = c.get('sentence_text') or ''
    if any(m in text for m in _JUNK_MARKERS):
        return None
    try:
        ast = parse(text)
    except Exception:
        return None
    if not isinstance(ast, dict):
        return None
    subj, verb, obj = ast.get('subjekto'), ast.get('verbo'), ast.get('objekto')
    if not (subj and verb and obj):
        return None

    def kerno(n):
        if not isinstance(n, dict):
            return None
        return n.get('kerno', n) if n.get('tipo') == 'vortgrupo' else n

    sk, vk, ok = kerno(subj), kerno(verb), kerno(obj)
    if not (sk and vk and ok):
        return None
    # 1. fresh parse must agree the propra is the subject
    if sk.get('vortspeco') != 'propra_nomo':
        return None
    subj_pv = sk.get('plena_vorto') or ''
    cand = (c.get('subject_pv') or '')
    if not subj_pv or not cand or cand.split()[0] not in subj_pv:
        return None
    # 2. templated verb must be THIS clause's main verb
    if (vk.get('radiko') or '') != c.get('verb_root'):
        return None
    # 3. polarity: reject mal-/ne negated predicates
    if (vk.get('plena_vorto') or '').lower().startswith('mal') \
            or 'mal' in (vk.get('prefiksoj') or []):
        return None
    toks = text.split()
    low = [t.strip('.,;:"()').lower() for t in toks]
    first = subj_pv.split()[0]
    if 'ne' in low and first.lower() in low \
            and low.index('ne') < low.index(first.lower()):
        return None
    # 4. object must be a common noun
    if ok.get('vortspeco') in ('korelativo', 'pronomo', 'propra_nomo'):
        return None
    if (ok.get('radiko') or '').lower() in _DEMONSTRATIVE:
        return None
    obj_pv = ok.get('plena_vorto') or ''
    if len(obj_pv) < 3 or obj_pv[:1].isupper():
        return None
    # 4b. object must be the verb's TRUE accusative patient. In Esperanto
    # the direct object is unambiguously -n (akuzativo); a non-accusative
    # "object" slot is a mis-attachment / nominative complement / oblique
    # — this is the 'Kiu kantis diplomon?' (sang a diploma) nonsense
    # class. Also reject a preposition-governed object (oblique, not the
    # patient), mirroring the subject check.
    if ok.get('kazo') != 'akuzativo':
        return None
    op = obj_pv.split()[0] if obj_pv else ''
    if op and op in toks:
        oi = toks.index(op)
        if oi > 0 and toks[oi - 1].strip('.,;:"()').lower() in _PREP_GOVERNED:
            return None
    # 5. preposition-governed subject -> not the agent
    if first in toks:
        i = toks.index(first)
        if i > 0 and toks[i - 1].strip('.,;:"()').lower() in _PREP_GOVERNED:
            return None
        # 6. full-name answer (kills name-splitting)
        span = [toks[i]]
        j = i + 1
        while j < len(toks) and toks[j][:1].isupper() and toks[j].isalpha():
            span.append(toks[j])
            j += 1
        full_name = ' '.join(span).strip('.,;:"()')
    else:
        full_name = subj_pv
    if len(full_name) < 3 or not full_name[0].isupper():
        return None
    # Reject content words the parser only mis-flips to propra_nomo
    # sentence-initially (Estis, Klubo, Diplomita, von, ...).
    if not _looks_namelike(full_name):
        return None

    out = dict(c)
    out['subject_pv'] = full_name
    out['object_pv'] = obj_pv
    out['object_radiko'] = ok.get('radiko') or c.get('object_radiko')
    return out

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


def _kerno_vorto(node) -> dict:
    """Return the head Vorto dict of a subjekto/objekto AST node."""
    if not isinstance(node, dict):
        return {}
    if node.get('tipo') == 'vortgrupo':
        return node.get('kerno') or {}
    return node


def query_active_who_candidates(conn, verbs, limit=10000):
    """Query WHO candidates from the DuckDB store with a strict
    person-name filter.

    The retired Kuzu field propranoma_kategorio='person' was derived
    from a 600K-entry proper-noun dictionary that is ~78%-polluted by
    its own header (common nouns, phrase fragments, wrong categories).
    The project deliberately keeps that gazetteer OFF the parser path;
    re-introducing it here only as a noisy precision oracle is the wrong
    trade. We drop it entirely and let two DETERMINISTIC gates do the
    quality work instead: verify_with_parser (the propra must be the
    re-parsed sentence's true agent of the templated verb) and the
    empirical discriminability gate. plena_vorto / object_vortspeco are
    read from the stored ast_json blob (shredded cols keep only radikoj).
    """
    placeholders = ','.join('?' * len(verbs))
    sql = f"""
        SELECT sid, text, subj_radiko, verb_radiko, obj_radiko, ast_json
        FROM sentences
        WHERE subj_vortspeco = 'propra_nomo'
          AND verb_radiko IN ({placeholders})
          AND ast_json IS NOT NULL
        LIMIT {int(limit)}
    """
    rows = []
    for sid, text, subj_r, verb_r, obj_r, ast_json in conn.execute(
            sql, list(verbs)).fetchall():
        try:
            ast = json.loads(ast_json)
        except Exception:
            continue
        subj = _kerno_vorto(ast.get('subjekto'))
        subj_pv = subj.get('plena_vorto') or ''
        obj = _kerno_vorto(ast.get('objekto'))
        rows.append({
            'sentence_id':       sid,
            'sentence_text':     text,
            'subject_pv':        subj_pv,
            'subject_radiko':    subj_r,
            'verb_root':         verb_r,
            'object_pv':         obj.get('plena_vorto') or '',
            'object_radiko':     obj_r,
            'object_vortspeco':  obj.get('vortspeco') or '',
        })
    return rows


# --- empirical discriminability gate ----------------------------------
_GATE_STOP = set('kiu kio kie kiam kiom kial kiel kiuj kion estas estis '
                  'estos la de en al el ĉu por kaj aŭ ke ne je da'.split())


def _q_terms(q: str) -> list[str]:
    toks = re.findall(r"[\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+", q.lower())
    return [t for t in toks if t not in _GATE_STOP and len(t) > 2]


def is_discriminating(searcher, qp, question: str,
                       source_sid: int, top_k: int) -> bool:
    """A pair is discriminating iff a RAW BM25 query (no AST filter) on
    the question terms surfaces the source sentence within top_k of the
    full corpus. Generous top_k tests 'findable in principle' (the query
    carries enough signal), leaving headroom for pipeline ranking to be
    what improvement work moves — while excluding the
    information-theoretically-impossible class entirely.

    searcher/qp are created ONCE by the caller and reused across the
    whole gate pass (reopening a searcher on the 5.4M single-segment
    index per candidate is what made this intractable).
    """
    terms = _q_terms(question)
    if not terms:
        return False
    q = qp.parse(' OR '.join(terms))
    for h in searcher.search(q, limit=top_k):
        try:
            if int(h['id']) == int(source_sid):
                return True
        except (KeyError, ValueError):
            continue
    return False


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


def salvage_existing(paths, s, qp, gate_top_k):
    """Run existing WHO pairs through the SAME gates as fresh generation.

    Honors the salvage-don't-replace principle: re-verify coherence
    (parser-AST agent + name filter, correcting the answer to the full
    name) and run the empirical discriminability gate on the ORIGINAL
    question. Keep survivors; report exactly what was dropped and why.
    """
    kept, used_ids = [], set()
    seen = drop_coh = drop_disc = drop_fields = 0
    for p in paths:
        fp = Path(p)
        if not fp.exists():
            print(f"  salvage: {p} not found, skipping")
            continue
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if (e.get('question_type') or 'WHO') != 'WHO':
                continue
            seen += 1
            q = e.get('question') or ''
            sid = e.get('source_sentence_id')
            ans = e.get('expected_answer') or ''
            src = e.get('source_sentence_text')
            vr = e.get('verb_root')
            if sid is None or not q:
                drop_fields += 1
                continue
            corrected = ans
            if src and vr:
                v = verify_with_parser({
                    'sentence_id': sid, 'sentence_text': src,
                    'subject_pv': ans, 'subject_radiko': '',
                    'verb_root': vr, 'object_pv': '',
                    'object_radiko': e.get('object_radiko') or '',
                    'object_vortspeco': '',
                })
                if v is None:
                    drop_coh += 1
                    continue
                corrected = v['subject_pv']
            elif not _looks_namelike(ans):
                drop_coh += 1
                continue
            if corrected.lower() in q.lower():
                drop_coh += 1
                continue
            if not is_discriminating(s, qp, q, sid, gate_top_k):
                drop_disc += 1
                continue
            kept.append({
                'id':                   f'who_salv_{len(kept)+1:03d}',
                'question':             q,
                'expected_answer':      corrected,
                'expected_keywords':    [corrected],
                'source_sentence_id':   sid,
                'source_sentence_text': src or '',
                'question_type':        'WHO',
                'pattern':              'active',
                'verb_root':            vr or '',
                'object_radiko':        e.get('object_radiko') or '',
            })
            used_ids.add(sid)
    print(f"  Salvage: {seen} existing WHO seen -> {len(kept)} kept "
          f"(dropped {drop_coh} incoherent, {drop_disc} non-discriminating, "
          f"{drop_fields} missing-fields)")
    return kept, used_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--duckdb-path', default='data/indexes/duckdb_store.db')
    parser.add_argument('--whoosh-dir',  default='data/indexes/whoosh_v2')
    parser.add_argument('--output',      default='data/test_sets/synthetic_who_active.jsonl')
    parser.add_argument('--target-size', type=int, default=200)
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--query-limit', type=int, default=20000,
                        help='Cap on raw candidates fetched from DuckDB')
    parser.add_argument('--gate-top-k',  type=int, default=200,
                        help='Raw-BM25 rank a pair must hit to be kept '
                             '(discriminability gate; generous on purpose)')
    parser.add_argument('--salvage', nargs='*',
                        default=['data/test_sets/gold_anchor_50.jsonl'],
                        help='Existing WHO sets to salvage through the '
                             'same gates before topping up (hybrid). '
                             'Pass with no value to disable.')
    args = parser.parse_args()

    print(f"Opening DuckDB store: {args.duckdb_path}")
    conn = duckdb.connect(args.duckdb_path, read_only=True)
    print(f"Opening Whoosh index: {args.whoosh_dir}")
    ix = open_dir(args.whoosh_dir)

    print(f"Querying corpus for active-voice WHO candidates "
          f"(verbs={ACTIVE_VERBS}, limit={args.query_limit})...")
    candidates = query_active_who_candidates(conn, ACTIVE_VERBS,
                                              limit=args.query_limit)
    print(f"  Raw candidates: {len(candidates)}")

    surface_ok = [c for c in candidates if is_quality_candidate(c)]
    print(f"  After surface quality filter: {len(surface_ok)}")
    # Parser-AST verification: re-parse each source sentence and keep
    # only those where the propra is the true agent of the templated
    # verb (corrects subject to the full-name span).
    filtered = []
    for c in surface_ok:
        v = verify_with_parser(c)
        if v is not None:
            filtered.append(v)
    print(f"  After parser-AST verification: {len(filtered)} "
          f"({len(filtered)/max(len(surface_ok),1)*100:.0f}% kept)")

    # Empirical discriminability gate: keep only pairs whose question
    # actually carries enough signal to retrieve the source sentence
    # from the full corpus. Generate the question here once and carry it
    # on the candidate so the output stage reuses the exact string.
    # The gate runs one 5.4M-doc Whoosh search per candidate, so bound
    # it: shuffle (seeded) to avoid corpus-order bias, early-stop once we
    # have enough for diversity sampling, and cap total gate evaluations.
    random.seed(args.seed)
    random.shuffle(filtered)
    discriminating = []
    evals = 0
    with ix.searcher(weighting=scoring.BM25F()) as s:
        qp = QueryParser('text', ix.schema, group=OrGroup)
        # 1. Salvage existing WHO pairs through the same gates first.
        salvaged, used_ids = ([], set())
        if args.salvage:
            print(f"Salvaging existing WHO sets: {args.salvage}")
            salvaged, used_ids = salvage_existing(
                args.salvage, s, qp, args.gate_top_k)
        # 2. Top up only the remainder with fresh generation.
        gen_target = max(0, args.target_size - len(salvaged))
        need = gen_target * 4
        max_evals = max(gen_target * 40, 4000) if gen_target else 0
        for c in filtered:
            if len(discriminating) >= need or evals >= max_evals:
                break
            if c['sentence_id'] in used_ids:
                continue  # already salvaged — don't duplicate the source
            q = make_question(c)
            if c['subject_pv'].lower() in q.lower():
                continue  # answer leaked into question
            evals += 1
            if is_discriminating(s, qp, q, c['sentence_id'],
                                 args.gate_top_k):
                c['question'] = q
                discriminating.append(c)
    filtered = discriminating
    print(f"  Generation gate (raw-BM25 top-{args.gate_top_k}, "
          f"{evals} evals): {len(filtered)} fresh kept")

    if not filtered and not salvaged:
        raise SystemExit("ERROR: no candidates passed filter — check verb list / corpus")

    # Diversity: cap per-verb so common verbs don't dominate the
    # generated top-up (salvaged pairs are kept as-is, not capped).
    gen_target = max(0, args.target_size - len(salvaged))
    random.seed(args.seed)
    by_verb = defaultdict(list)
    for c in filtered:
        by_verb[c['verb_root']].append(c)
    n_verbs = max(1, len(by_verb))
    per_verb = max(1, gen_target // n_verbs)

    pool = []
    for verb_root, items in by_verb.items():
        random.shuffle(items)
        pool.extend(items[:per_verb])

    # If under target, top up from leftovers.
    if len(pool) < gen_target:
        leftovers = [c for c in filtered if c not in pool]
        random.shuffle(leftovers)
        pool.extend(leftovers[: gen_target - len(pool)])

    random.shuffle(pool)
    pool = pool[: gen_target]

    # Combined output: salvaged pairs first, then fresh generation.
    output = list(salvaged)
    for i, c in enumerate(pool, 1):
        q = c['question']  # set by the discriminability gate
        output.append({
            'id':                   f'who_gen_{i:03d}',
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
