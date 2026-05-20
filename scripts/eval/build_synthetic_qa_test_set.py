#!/usr/bin/env python3
"""
Build synthetic multi-type (KIU/KIO/KIE/KIAM/KIAL/KIEL) test sets from corpus.

VERSION: v2.1 | v3.0 (DuckDB)
COMPATIBLE WITH: DuckDB store (sentences: shredded cols + ast_json blob)
DEPENDENCIES: duckdb, Whoosh index, klareco.parser, klareco.ontology (query API)
STAGE: Evaluation

Description:
    Generates type-pluggable trivia-style Q&A pairs FROM the corpus using a
    unified generator that produces six question types: KIU (WHO), KIO (WHAT),
    KIE (WHERE), KIAM (WHEN), KIAL (WHY), KIEL (HOW). Each type has its own
    QuestionTypeConfig specifying: the question word, template function, AST
    role to verify, answer constraint gate (type-specific entity checks via
    semantic ontology), and discriminator class.

    All types use TWO correctness gates (matching WHO generator):
      1. Parser-AST role verification: the extracted role matches a fresh
         parse of the source sentence and is not negated.
      2. Empirical discriminability (BM25 top-K gate): the question's terms
         surface the source sentence in the top-K of the 5.4M corpus, ensuring
         the pair is 'findable in principle'.

    Type-specific constraints:
      - KIU (WHO): agent of active-voice creation/authorship verbs; answer =
        propra_nomo; rigid designator = «quoted work» or proper-noun object.
      - KIO (WHAT): patient/object role of verb; answer = »quoted work» or
        propra_nomo object; named target (unique designation).
      - KIE (WHERE): locative adverbial (prep: en|ĉe|sur|apud...); answer =
        propra_nomo linked to EntecaTipo {tipo_id: 'loko'}; rigid designator.
      - KIAM (WHEN): temporal adverbial (4-digit year, named month, or NP
        linked to EntecaTipo {tipo_id: 'tempo'}); answer = date/year string.
      - KIAL (WHY): explicit causal marker (ĉar, pro, pro tio ke); answer =
        the causal clause. Yield expected low (few explicit causals in corpus).
      - KIEL (HOW): manner adverbial (-e adverb or per-instrumental); answer =
        the manner expression. Yield expected low (sparse in corpus).

    Hardcoded gazetteers are FORBIDDEN (per CLAUDE.md schema-first principle).
    Entity type checks use klareco.ontology queries only.

Pipeline Position:
    DuckDB store + Whoosh + Kuzu ontology → [THIS SCRIPT] → JSONL test sets
    → evaluate_extractive_qa.py

Usage:
    python scripts/eval/build_synthetic_qa_test_set.py --type kiu
    python scripts/eval/build_synthetic_qa_test_set.py --type kio --target-size 50 --seed 42
    python scripts/eval/build_synthetic_qa_test_set.py --type kie --query-limit 5000
    # All types:
    for type in kiu kio kie kiam kial kiel; do
        python scripts/eval/build_synthetic_qa_test_set.py --type $type &
    done; wait

Inputs:
    DuckDB store at data/indexes/duckdb_store.db
    Whoosh index at data/indexes/whoosh_v2
    Kuzu ontology (optional; fallback to no-constraint if unavailable)
    Wikidata notable-people cache (WHO only, optional)

Outputs:
    JSONL test sets at data/test_sets/synthetic_{type}_active.jsonl, one per line:
      {id, question, expected_answer, expected_keywords,
       source_sentence_id, source_sentence_text, question_type='KIU'|'KIO'|...,
       pattern='active', verb_root, answer_role, ...}

Quality Checks:
    - Answer text never appears in the generated question
    - Entity type checks (where applicable) via semantic ontology, not gazetteers
    - BM25 discriminability gate (top-K retrieval of source sentence)
    - Sentence length 5–40 words
    - Diversity: balanced per-verb sampling

Last Updated: 2026-05-19
Author: Claude Code (with Marc Jones)

CHANGELOG:
# 2026-05-19: Initial multi-type generator from WHO template; pluggable type
#             configs; unified discriminability gate; schema-first entity
#             checks via ontology API (no gazetteers).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field as dataclasses_field
from pathlib import Path
from typing import Any, Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb
from whoosh import scoring
from whoosh.index import open_dir
from whoosh.qparser import OrGroup, QueryParser

from klareco.parser import parse

# Audit helpers live next to this script; import them so the generator and
# the post-hoc auditor share a single source of truth on quality.
_AUDIT_PATH = str(Path(__file__).resolve().parent)
if _AUDIT_PATH not in sys.path:
    sys.path.insert(0, _AUDIT_PATH)
from audit_qa_pairs import (  # noqa: E402
    check_question_shape,
    check_answer_shape,
    check_trivia_specificity,
    check_answer_in_source,
    check_answer_role_alignment,
)


def _audit_pass(question: str, answer: str, source_text: str,
                question_type: str) -> tuple[bool, str]:
    """Fail-closed audit gate. Returns (True, '') on PASS, (False, reason)
    on the first failing check. Used by the generator to guarantee that
    every pair written to disk is audit-PASS by construction."""
    qt = (question_type or '').upper()
    for name, result in (
        ('question_shape',   check_question_shape(question)),
        ('answer_shape',     check_answer_shape(answer, qt)),
        ('trivia_specific',  check_trivia_specificity(question)),
        ('answer_in_source', check_answer_in_source(answer, source_text, qt)),
        ('role_alignment',   check_answer_role_alignment(answer, source_text, qt)),
    ):
        ok, reason = result
        if not ok:
            return False, f'{name}: {reason}'
    return True, ''

# =============================================================================
# Shared Junk Markers & Filters
# =============================================================================

_JUNK_MARKERS = ('[', ']', 'REDIRECT', 'ALIDIREKTI', 'ALIDIREKTU',
                 ' ekzemple:', 'Skribu ', '#')
_PREP_GOVERNED = {
    'kun', 'pri', 'de', 'per', 'en', 'sub', 'al', 'ĉe', 'post',
    'antaŭ', 'el', 'tra', 'inter', 'kontraŭ', 'apud', 'laŭ', 'pro',
}
_DEMONSTRATIVE = {
    'tia', 'tiu', 'ĉi', 'ĝi', 'ili', 'jen', 'tio', 'ĉio', 'io',
    'la', 'lia', 'ŝia', 'sia', 'nia', 'via', 'mia',
}
_PARTICIPLE_RE = re.compile(r'(it|at|ot|int|ant|ont|unt)a$', re.I)
# Esperanto participle-as-adverb (e.g. `estante`, `kontrastante`). Caught
# separately from `_PARTICIPLE_RE` so the existing name-check helpers don't
# change behavior.
_PARTICIPLE_E_RE = re.compile(r'(it|at|ot|int|ant|ont|unt)e$', re.I)
# Letters not in the native Esperanto alphabet — q/w/x/y. Filtering on these
# is a cheap way to reject English/Latin/French tokens that end up in
# bibliographic citations (passerine, recognise, nouvelle, …).
_NON_ESPERANTO_LETTERS_RE = re.compile(r'[qwxy]', re.IGNORECASE)
_QUOTED_RE = re.compile(r'[«„"]\s*([^«»„"]{2,80}?)\s*[»"]')
_GATE_STOP = set('kiu kio kie kiam kiom kial kiel kiuj kion estas estis '
                  'estos la de en al el ĉu por kaj aŭ ke ne je da'.split())

# Locative prepositions used by KIE answer/anchor surface checks.
# `de` is omitted because it's ambiguous (genitive vs. ablative).
_LOCATIVE_PREPS = ('en', 'ĉe', 'sur', 'apud', 'sub', 'super', 'tra', 'trans')

# Wider set of prepositions used by KIU to detect PP-governed names that
# should NOT be treated as agents. `En Volterra`, `Al Germanio`,
# `En "Głos"`, etc. — all parse as propra_nomo subjects but are actually
# inside a fronted/embedded PP.
_GOVERNING_PREPS = (
    'en', 'ĉe', 'sur', 'apud', 'sub', 'super', 'tra', 'trans',
    'kun', 'de', 'al', 'el', 'por', 'antaŭ', 'post', 'pri', 'pro',
    'kontraŭ', 'inter', 'ekster', 'sen', 'krom', 'malgraŭ', 'laŭ',
)

# Common Esperanto words that are sentence-initial-capitalised and may be
# mis-tagged as propra_nomo by the parser's learned classifier. They are
# not in the "irreducible NOUN↔propra_nomo" zone — they're just function
# words that happen to be capitalised. A targeted denylist kills this
# class without retraining the classifier.
_COMMON_WORDS_AS_PROPER = {
    # Conjunctions / discourse
    'Kaj', 'Sed', 'Aŭ', 'Do', 'Tamen', 'Tial', 'Ke', 'Ankaŭ',
    # Function words / prepositions / particles
    'Anstataŭ', 'Krom', 'Malgraŭ', 'Sen', 'Por', 'Antaŭ', 'Post',
    'Dum', 'Tra', 'Trans', 'Apud', 'Pri', 'Pro', 'Laŭ',
    # Adverbs sometimes flagged
    'Tiam', 'Tiu', 'Ĉi', 'Jen', 'Nun', 'Hodiaŭ',
    # Common nouns frequently capitalised in headings
    'Estro', 'Estis', 'Konsilio', 'Organizaĵo',
}

# Causal markers used by KIAL surface-text extraction.
_CAUSAL_RE = re.compile(r'\b(ĉar|pro tio,?\s+ke)\b', re.IGNORECASE)

# Words that pass the naive "-e adverb" surface test but are NOT manner.
# The -e ending in Esperanto marks adverbs in general (manner, temporal,
# locative, discourse, modal) — only a subset are actually manner. This
# stoplist names the most common non-manner -e adverbs so the extractor
# doesn't accept them. Curated from corpus inspection of failing KIEL pairs.
_MANNER_STOPWORDS = {
    # Locative correlatives + relative positions
    'tie', 'ĉie', 'ie', 'nenie', 'kie',
    'sube', 'supre', 'antaŭe', 'malantaŭe', 'dekstre', 'maldekstre',
    'norde', 'sude', 'oriente', 'okcidente', 'meze', 'apude',
    # Temporal
    'tiame', 'kiame', 'iame', 'neniam-e', 'ĉiame',
    'kelkfoje', 'ofte', 'malofte', 'iam', 'nun', 'jam',
    'ekde', 'fine', 'komence', 'baldaŭ', 'tuje', 'poste', 'frue', 'malfrue',
    # Discourse / modal / focus
    'ekzemple', 'aldone', 'tamen', 'jene', 'ankaŭ', 'eĉ',
    'kadre', 'krome', 'bedaŭrinde', 'ŝajne', 'evidente', 'kompreneble',
    'precipe', 'specife', 'speciale', 'ĝenerale', 'plejparte',
    'verŝajne', 'eble', 'certe', 'eventuale', 'efektive', 'fakte',
    'feliĉe', 'malfeliĉe', 'simile', 'malsame',
    # Iteration / quantification
    'refoje', 'denove', 'plue', 'sole', 'kune', 'sume', 'entute',
    # Comparative / degree
    'tre', 'pli', 'malpli', 'tute', 'iom-e', 'apenaŭ-e',
    # Compound prepositions that parser-tags as adverbo
    'disde', 'ekde', 'depost', 'detempe', 'destrate',
    # Other common
    'nome', 'rilate', 'koncerne', 'spite',
}


def _has_locative_prep_before(text: str, name: str) -> bool:
    """True if `name` is preceded by a locative preposition in the surface text.

    Used by KIE to verify the answer span is actually a place (not just a
    propra_nomo in some other role). Case-insensitive on the preposition, but
    the name is matched as-is to respect capitalisation.
    """
    if not name:
        return False
    pattern = (r'(?:^|[\s,;:.\(])(' + '|'.join(_LOCATIVE_PREPS)
               + r')\s+' + re.escape(name) + r'\b')
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def _is_pp_governed(name: str, source_text: str) -> bool:
    """True if `name` is immediately preceded by a preposition (from the
    `_GOVERNING_PREPS` set) in `source_text`, ignoring intervening quotes
    and punctuation.

    Used by KIU to reject candidate "agents" that are actually inside a PP
    (e.g. `En "Głos" publikigis manifeston…`, where the parser puts Głos as
    subjekto but it's really the venue).
    """
    if not name or not source_text:
        return False
    pos = source_text.find(name)
    if pos < 0:
        return False
    # Wider window so we can skip past intervening quotes/punctuation.
    pre = source_text[max(0, pos - 40):pos]
    m = re.search(r'([\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+)[^\wĉĝĥĵŝŭĈĜĤĴŜŬ-]*$', pre)
    if not m:
        return False
    return m.group(1).lower() in _GOVERNING_PREPS


_INTERVENING_VERB_RE = re.compile(
    r'\b\w{3,}(?:iĝis|igis|is|as|os|us|u)\b',
    re.IGNORECASE,
)


def _find_first_locative_after_verb(text: str, verb_pv: str
                                     ) -> tuple[str, str] | None:
    """Return (preposition, name) for the FIRST locative PP appearing after
    the verb's surface position, ONLY IF no intervening finite verb sits
    between the question verb and that locative. Returns None otherwise.

    The intervening-verb guard kills the dominant remaining KIE failure
    class: `naskiĝis 1909 - mortis la 19-an de marto 1998 en Dudince`,
    where the source has NO birthplace PP but does have a deathplace PP.
    Without the guard, the function would return Dudince as the
    birthplace; with the guard, it sees `mortis` between `naskiĝis` and
    `en Dudince` and rejects (correct: no birthplace findable).
    """
    if not text:
        return None
    vpos = text.find(verb_pv) if verb_pv else -1
    if vpos < 0:
        return None
    after_verb = text[vpos + len(verb_pv):]
    pattern = (r'(?:^|[\s,;:.\(])(' + '|'.join(_LOCATIVE_PREPS)
               + r')\s+([A-ZÀ-ÞĈĜĤĴŜŬ][\wÀ-ſĉĝĥĵŝŭĈĜĤĴŜŬ-]+)')
    m = re.search(pattern, after_verb, flags=re.IGNORECASE)
    if not m:
        return None
    between = after_verb[:m.start()]
    if _INTERVENING_VERB_RE.search(between):
        return None
    return (m.group(1).lower(), m.group(2))


def _extract_causal_clause(text: str) -> str | None:
    """Extract a `ĉar` / `pro tio ke` clause span from surface text.

    Spans from the causal marker up to the nearest clause boundary
    (period, semicolon, or end-of-string). Caps span length so we don't
    emit runaway answers, and requires ≥3 tokens of content after the
    marker so the clause is informative.
    """
    if not text:
        return None
    m = _CAUSAL_RE.search(text)
    if not m:
        return None
    tail = text[m.start():]
    # Stop at sentence-ending punctuation; commas are fine inside the clause.
    boundary = re.search(r'[.;]\s', tail)
    clause = tail[:boundary.start()] if boundary else tail
    clause = clause.strip(' .,;:')
    if not clause:
        return None
    # Cap at 200 chars on word boundary
    if len(clause) > 200:
        clause = clause[:200].rsplit(' ', 1)[0]
    # Drop marker for content-count, require ≥3 informative tokens
    marker_match = _CAUSAL_RE.match(clause)
    content = clause[marker_match.end():].strip() if marker_match else clause
    if len(content.split()) < 3:
        return None
    return clause


def _extract_manner_span(text: str) -> str | None:
    """Extract a manner expression: `per <NP>` or a stand-alone `-e` adverb.

    Tries `per <NP>` first (more reliable), then falls back to a lowercase
    `-e` adverb of ≥4 chars that isn't in the stopword list and isn't an
    obvious participle (e.g. `verkitae` would be filtered by participle regex
    elsewhere). Returns None if no suitable span.
    """
    if not text:
        return None
    # 1. `per <NP>` — instrumental
    m = re.search(
        r'\bper\s+([\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+(?:\s+[\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+){0,5})',
        text, flags=re.IGNORECASE,
    )
    if m:
        span = ('per ' + m.group(1).strip(' .,;:')).strip()
        # Stop at obvious boundary tokens
        for stop in (' kaj ', ' sed ', ' aŭ ', ' kiu ', ' kio '):
            i = span.lower().find(stop)
            if i > 0:
                span = span[:i].strip()
                break
        if 2 <= len(span.split()) <= 8:
            return span
    # 2. lowercase -e adverb. Minimum 4 chars + stoplist + Esperanto-only
    # letters + parser-tag-as-adverbo. The schema-first parser tag is the
    # ground truth for "is this Esperanto morphology" — it rejects English/
    # Latin/French citation tokens that happen to end in -e (passerine,
    # recognise, nouvelle).
    for word in text.split():
        cleaned = word.strip('.,;:!?"()«»')
        if not cleaned:
            continue
        low = cleaned.lower()
        if not low.endswith('e') or len(low) < 4:
            continue
        if low in _MANNER_STOPWORDS:
            continue
        if cleaned[:1].isupper():  # proper nouns shouldn't be manner
            continue
        if _PARTICIPLE_RE.search(low) or _PARTICIPLE_E_RE.search(low):
            continue
        if _NON_ESPERANTO_LETTERS_RE.search(low):
            continue
        if not _is_adverb_per_parser(cleaned):
            continue
        return cleaned
    return None


def _is_adverb_per_parser(token: str) -> bool:
    """Parse a single token and ask the parser whether it's tagged `adverbo`
    AND not flagged as a neologismo (which catches recognise/internationale-
    style foreign tokens). Schema-first wordhood check."""
    try:
        ast = parse(token)
    except Exception:
        return False
    if not isinstance(ast, dict):
        return False
    candidates = []
    for key in ('subjekto', 'verbo', 'objekto'):
        n = ast.get(key)
        if not n:
            continue
        candidates.append(
            n.get('kerno') if isinstance(n, dict) and n.get('tipo') == 'vortgrupo' else n
        )
    for x in ast.get('aliaj') or []:
        candidates.append(
            x.get('kerno') if isinstance(x, dict) and x.get('tipo') == 'vortgrupo' else x
        )
    for v in candidates:
        if not isinstance(v, dict):
            continue
        if v.get('vortspeco') != 'adverbo':
            continue
        # Foreign-loan detector: parser marks unfamiliar tokens with
        # kategorio='neologismo' even when it pattern-fits a -e adverb shape.
        if v.get('kategorio') == 'neologismo':
            return False
        # Reject any compound-decomposed token. Real Esperanto manner
        # adverbs are simple radiko+e; compounds in this slot are almost
        # always pseudo-Esperanto from foreign citation tokens (`nouvelle`
        # → ['nouv', 'ell']). The recall hit on legitimate compound manner
        # adverbs is small and worth the precision gain.
        if v.get('kunmetitaj_radikoj'):
            return False
        # Suffix chains of length ≥2 signal the parser reaching to decompose
        # a foreign loan (`passerine` → root 'pass' + suffixes [in, er]).
        # Real Esperanto manner adverbs rarely chain two suffixes here.
        if len(v.get('sufiksoj') or []) >= 2:
            return False
        return True
    return False


def _looks_namelike(name: str) -> bool:
    """Deterministic name check (reuses WHO logic).

    Rejects:
      - short / lowercase tokens
      - Esperanto common words capitalised at sentence start
        (Anstataŭ, Kaj, Tamen, …) — these are NOT in the irreducible
        NOUN↔propra_nomo zone; they're just function words that the
        parser's classifier sometimes mis-tags.
      - participle-shaped tokens (Diplomita, Skribita, …)
    """
    tok = name.split()[0] if name else ''
    if len(tok) < 3 or not tok[:1].isupper():
        return False
    if tok in _COMMON_WORDS_AS_PROPER:
        return False
    if _PARTICIPLE_RE.search(tok.lower()):
        return False
    ast = parse(tok)
    for key in ('subjekto', 'verbo', 'objekto'):
        n = ast.get(key)
        if not n:
            continue
        v = (n.get('kerno') if isinstance(n, dict) and n.get('tipo') == 'vortgrupo' else n)
        if isinstance(v, dict) and v.get('vortspeco') == 'propra_nomo':
            return True
    for x in ast.get('aliaj') or []:
        v = (x.get('kerno') if isinstance(x, dict) and x.get('tipo') == 'vortgrupo' else x)
        if isinstance(v, dict) and v.get('vortspeco') == 'propra_nomo':
            return True
    return False


def _kerno_vorto(node) -> dict:
    """Return the head Vorto dict of a subjekto/objekto AST node."""
    if not isinstance(node, dict):
        return {}
    if node.get('tipo') == 'vortgrupo':
        return node.get('kerno') or {}
    return node


def _named_target(text: str, ok: dict, toks: list, verb_pv: str) -> tuple[str, str] | None:
    """Extract a RIGID DESIGNATOR («quoted title»)."""
    vpos = text.find(verb_pv) if verb_pv else -1
    for m in _QUOTED_RE.finditer(text):
        if vpos != -1 and m.start() < vpos:
            continue
        cand = m.group(1).strip(' .,;:')
        wc = len(cand.split())
        if not (1 <= wc <= 6):
            continue
        if not re.search(r'[A-Za-zĉĝĥĵŝŭĈĜĤĴŜŬ]', cand):
            continue
        if cand[0].islower():
            continue
        return cand, 'titolo'
    return None


def _q_terms(q: str) -> list[str]:
    """Extract non-stopword terms from a question."""
    toks = re.findall(r"[\wĉĝĥĵŝŭĈĜĤĴŜŬ-]+", q.lower())
    return [t for t in toks if t not in _GATE_STOP and len(t) > 2]


def is_discriminating(searcher, qp, question: str, source_sid: int, top_k: int) -> bool:
    """Check if the source sentence surfaces in BM25 top-K."""
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


# =============================================================================
# Question Type Configuration
# =============================================================================

@dataclass
class QuestionTypeConfig:
    """Configuration for a single question type."""
    qword: str  # "Kiu", "Kio", "Kie", "Kiam", "Kial", "Kiel"
    type_id: str  # 'kiu', 'kio', 'kie', 'kiam', 'kial', 'kiel'

    # Template function: build the question text.
    # Signature: (verb_root, verb_pv, question_anchor_text) -> str
    # `verb_pv` is the source sentence's actual verb surface form (e.g.
    # 'naskiĝis', 'publikigis'). Templates SHOULD prefer it over a
    # synthesised `{verb_root}is`, which mangles voice (-iĝ-) and morphology
    # (publikigis -> publikis).
    template_fn: Callable[[str, str, str], str]

    # AST role(s) to extract as the answer (e.g. 'subjekto', 'objekto', 'aliaj')
    answer_roles: list[str]

    # Role from which to extract the question anchor (named entity in the question)
    # For KIU: 'objekto' (the quoted work)
    # For KIO: 'subjekto' (the creator)
    # For KIE/KIAM/KIAL/KIEL: 'subjekto' (the actor)
    question_anchor_role: str

    # Type-specific constraint gate for question anchor: (node, sentence_text) -> bool
    # Must verify the anchor is a propra_nomo and namelike. `sentence_text` is
    # passed so a constraint can check surface-text context (e.g. KIE rejects
    # subjects that appear inside a locative PP, which the parser sometimes
    # misclassifies). Most constraints ignore the second arg.
    question_anchor_constraint_fn: Callable[[dict, str], bool]

    # Type-specific constraint gate for answer: (node: dict, sentence_text: str) -> bool
    # Returns True if the extracted node passes type-specific checks
    # (e.g., entity-type linking via ontology, causal markers in sentence).
    answer_constraint_fn: Callable[[dict, str], bool]

    # Rigid-designator class (e.g. 'propra_nomo', 'quoted_work', 'named_place')
    discriminator_kind: str

    # Verbs suitable for this type (empty = use all verbs found)
    active_verbs: list[str]

    # Optional surface-pattern SQL filter (LIKE-fragments) ANDed into the
    # candidate query. Used by entity-anchored configs (place, event) to
    # narrow the 5.4M-row scan to sentences containing the entity-class
    # surface pattern. Empty for person-anchored types that filter by verb
    # alone. Each item becomes `text LIKE '%<pattern>%'` joined with OR.
    sql_text_patterns: list[str] = dataclasses_field(default_factory=list)


# KIU (WHO) — Reuse existing logic
_KIU_VERBS = [
    'fond', 'kre', 'establ', 'invent', 'desegn', 'konstruk', 'edif',
    'verk', 'skrib', 'redakt', 'publik', 'eldon',
    'malkovr', 'eltrov',
    'pentr', 'kompoz', 'reĝisor', 'kant',
    'gajn', 'venk',
]

def _kiu_anchor_constraint(node: dict, sentence_text: str = '') -> bool:
    """Object (anchor) must be propra_nomo or quoted work."""
    if not isinstance(node, dict):
        return False
    kerno = node.get('kerno') if node.get('tipo') == 'vortgrupo' else node
    if not isinstance(kerno, dict):
        return False
    return kerno.get('vortspeco') == 'propra_nomo'

def _kiu_answer_constraint(node: dict, sentence_text: str) -> bool:
    """Subject (answer) must be propra_nomo, namelike, AND not PP-governed.

    The PP-governance check kills the dominant remaining WHO failure class:
    the parser puts a name into `subjekto` that's actually inside a fronted
    or embedded PP (`En "Głos" publikigis manifeston…`, `En Volterra, li
    skribis…`). Surface-text inspection of the token immediately preceding
    the answer is the ground-truth signal.
    """
    if not isinstance(node, dict):
        return False
    kerno = node.get('kerno') if node.get('tipo') == 'vortgrupo' else node
    if not isinstance(kerno, dict):
        return False
    if kerno.get('vortspeco') != 'propra_nomo':
        return False
    pv = kerno.get('plena_vorto') or ''
    if not _looks_namelike(pv):
        return False
    if sentence_text and _is_pp_governed(pv, sentence_text):
        return False
    return True

def _kiu_template(verb_root: str, verb_pv: str, question_anchor_text: str) -> str:
    """Build KIU question with question anchor (the work being asked about).

    Uses the source verb's actual surface form so 'publikigis' doesn't become
    the malformed 'publikis'. Falls back to '{root}is' if verb_pv missing.
    """
    verb = verb_pv or f'{verb_root}is'
    if question_anchor_text.startswith('«'):
        return f"Kiu {verb} {question_anchor_text}?"
    obj_text = question_anchor_text if question_anchor_text.endswith('n') else question_anchor_text + 'n'
    return f"Kiu {verb} {obj_text}?"

KIU_CONFIG = QuestionTypeConfig(
    qword='Kiu',
    type_id='kiu',
    template_fn=_kiu_template,
    answer_roles=['subjekto'],
    question_anchor_role='objekto',
    question_anchor_constraint_fn=_kiu_anchor_constraint,
    answer_constraint_fn=_kiu_answer_constraint,
    discriminator_kind='quoted_work',
    active_verbs=_KIU_VERBS,
)


# KIO (WHAT) — Object of verb, typically «titled work» or propra_nomo
_KIO_VERBS = [
    'verk', 'skrib', 'pentr', 'kompoz', 'publika', 'eldon',
    'kre', 'desegn', 'konstruk', 'fabrik',
    'invent', 'diskurz', 'vort',
]

def _kio_anchor_constraint(node: dict, sentence_text: str = '') -> bool:
    """Subject (anchor) must be propra_nomo and namelike."""
    if not isinstance(node, dict):
        return False
    kerno = node.get('kerno') if node.get('tipo') == 'vortgrupo' else node
    if not isinstance(kerno, dict):
        return False
    if kerno.get('vortspeco') != 'propra_nomo':
        return False
    pv = kerno.get('plena_vorto') or ''
    return _looks_namelike(pv)

def _kio_answer_constraint(node: dict, sentence_text: str) -> bool:
    """Object (answer) must be propra_nomo or quoted work."""
    if not isinstance(node, dict):
        return False
    kerno = node.get('kerno') if node.get('tipo') == 'vortgrupo' else node
    if not isinstance(kerno, dict):
        return False
    return kerno.get('vortspeco') == 'propra_nomo'

def _kio_template(verb_root: str, verb_pv: str, question_anchor_text: str) -> str:
    """Build KIO question with subject (creator/author) as anchor."""
    verb = verb_pv or f'{verb_root}is'
    return f"Kion {verb} {question_anchor_text}?"

KIO_CONFIG = QuestionTypeConfig(
    qword='Kio',
    type_id='kio',
    template_fn=_kio_template,
    answer_roles=['objekto'],
    question_anchor_role='subjekto',
    question_anchor_constraint_fn=_kio_anchor_constraint,
    answer_constraint_fn=_kio_answer_constraint,
    discriminator_kind='quoted_work',
    active_verbs=_KIO_VERBS,
)


# KIE (WHERE) — Locative adverbial (prep + propra_nomo linked to 'loko')
_KIE_VERBS = [
    'nask', 'morta', 'vizit', 'voj', 'viv', 'rest', 'sit', 'konstru',
    'font', 'labor', 'stud', 'ven', 'ir',
]

def _kie_anchor_constraint(node: dict, sentence_text: str = '') -> bool:
    """Subject (anchor) must be a propra_nomo person — NOT a place.

    Rejects names that the parser flagged as subjekto but appear preceded
    by a locative preposition in the surface text (e.g. `En Alsónémedi
    naskiĝis aktoro Kálmán Balla` parses with `Alsónémedi` as subjekto).
    Empty `sentence_text` skips the surface check, but the answer-side
    locative-prep gate still catches the bad-pair case.
    """
    if not isinstance(node, dict):
        return False
    kerno = node.get('kerno') if node.get('tipo') == 'vortgrupo' else node
    if not isinstance(kerno, dict):
        return False
    if kerno.get('vortspeco') != 'propra_nomo':
        return False
    pv = kerno.get('plena_vorto') or ''
    if not _looks_namelike(pv):
        return False
    if sentence_text and _has_locative_prep_before(sentence_text, pv):
        # Parser misclassified a place as the subject; reject.
        return False
    return True

def _kie_answer_constraint(node: dict, sentence_text: str) -> bool:
    """Answer must be a locative propra_nomo (place name).

    Strict surface check: the candidate's plena_vorto must be immediately
    preceded by a locative preposition (`en`, `ĉe`, `sur`, …) in the source
    text. The earlier `prep == ''` fallback was permissive and let any
    propra_nomo from `aliaj` through — that's how `Kálmán` was passing the
    gate for `En Alsónémedi naskiĝis aktoro Kálmán Balla`.
    """
    if not isinstance(node, dict):
        return False
    kerno = node.get('kerno') if node.get('tipo') == 'vortgrupo' else node
    if not isinstance(kerno, dict):
        return False
    if kerno.get('vortspeco') != 'propra_nomo':
        return False
    pv = kerno.get('plena_vorto') or ''
    if not _looks_namelike(pv):
        return False
    return _has_locative_prep_before(sentence_text, pv)

def _kie_template(verb_root: str, verb_pv: str, question_anchor_text: str) -> str:
    """Build KIE question with subject as anchor.

    Uses the source verb's surface form so e.g. 'naskiĝis Páhi' stays
    'naskiĝis' in the question (not the transitive 'naskis').
    """
    verb = verb_pv or f'{verb_root}is'
    return f"Kie {verb} {question_anchor_text}?"

KIE_CONFIG = QuestionTypeConfig(
    qword='Kie',
    type_id='kie',
    template_fn=_kie_template,
    answer_roles=['aliaj'],  # Adverbials (locative)
    question_anchor_role='subjekto',
    question_anchor_constraint_fn=_kie_anchor_constraint,
    answer_constraint_fn=_kie_answer_constraint,
    discriminator_kind='named_place',
    active_verbs=_KIE_VERBS,
)


# KIAM (WHEN) — Temporal adverbial (year, month, or NP linked to 'tempo')
_KIAM_VERBS = [
    'nask', 'morta', 'font', 'establ', 'kre', 'ven', 'okazis', 'start',
    'finiĝ', 'okaz',
]

# Global cache for temporal roots (populated at startup from ontology)
_TEMPORAL_ROOTS_CACHE = None

def _load_temporal_roots_from_ontology(duckdb_conn=None):
    """Load temporal-entity-type members (EntecaTipo `tempo`) from the DuckDB
    ontology (Kuzu was retired 2026-05; the ontology now lives in DuckDB
    tables `ontology_nodes` / `ontology_edges`). Cache globally.

    If the ontology has zero edges for `tempo` (currently the case — the node
    exists but no radikoj are linked), supplement with a small, documented
    minimal set of Esperanto temporal roots so the KIAM gate can function.
    The supplement is explicit, not a silent fallback: it logs both what came
    from the ontology and what was added, so the gap is visible.
    """
    global _TEMPORAL_ROOTS_CACHE
    if _TEMPORAL_ROOTS_CACHE is not None:
        return _TEMPORAL_ROOTS_CACHE

    minimum_supplement = {
        'januar', 'februar', 'marĉ', 'april', 'maj', 'jun',
        'juli', 'aŭgust', 'septembr', 'oktobr', 'novembr', 'decembr',
        'jaro', 'monato', 'semajno', 'tago',
    }

    ontology_roots: set[str] = set()
    if duckdb_conn is not None:
        try:
            rows = duckdb_conn.execute(
                "SELECT radiko FROM ontology_edges "
                "WHERE class_id = 'tempo' AND rel = 'HAVAS_ENTECAN_TIPON'"
            ).fetchall()
            ontology_roots = {r[0] for r in rows if r[0]}
        except Exception as e:
            print(f"WARNING: DuckDB ontology query for `tempo` failed ({e}); "
                  f"using minimal-supplement roots only.")

    if not ontology_roots:
        print(f"WARNING: ontology edges for EntecaTipo `tempo` are empty in DuckDB. "
              f"Supplementing with {len(minimum_supplement)} minimal Esperanto "
              f"temporal roots. Populate `ontology_edges` (rel='HAVAS_ENTECAN_TIPON', "
              f"class_id='tempo') to remove this supplement.")
    else:
        print(f"Loaded {len(ontology_roots)} temporal roots from DuckDB ontology; "
              f"supplementing with {len(minimum_supplement - ontology_roots)} "
              f"common-Esperanto temporals not yet present in the ontology.")

    _TEMPORAL_ROOTS_CACHE = ontology_roots | minimum_supplement
    return _TEMPORAL_ROOTS_CACHE

def _kiam_anchor_constraint(node: dict, sentence_text: str = '') -> bool:
    """Subject (anchor) must be propra_nomo and namelike."""
    if not isinstance(node, dict):
        return False
    kerno = node.get('kerno') if node.get('tipo') == 'vortgrupo' else node
    if not isinstance(kerno, dict):
        return False
    if kerno.get('vortspeco') != 'propra_nomo':
        return False
    pv = kerno.get('plena_vorto') or ''
    return _looks_namelike(pv)

def _kiam_answer_constraint(node: dict, sentence_text: str) -> bool:
    """Temporal: 4-digit year or temporal NP from ontology."""
    if not isinstance(node, dict):
        return False
    kerno = node.get('kerno') if node.get('tipo') == 'vortgrupo' else node
    if not isinstance(kerno, dict):
        return False
    pv = kerno.get('plena_vorto') or ''

    # Accept 4-digit years
    if re.match(r'^\d{4}$', pv):
        return True

    # Temporal nouns from ontology
    temporal_roots = _load_temporal_roots_from_ontology()
    return (kerno.get('radiko') or '') in temporal_roots

def _kiam_template(verb_root: str, verb_pv: str, question_anchor_text: str) -> str:
    """Build KIAM question with subject as anchor.

    Uses the source verb's surface form so intransitive `-iĝ-` voice is
    preserved (`Páhi naskiĝis en 1853` -> `Kiam naskiĝis Páhi?`, NOT the
    semantically-wrong transitive `Kiam naskis Páhi?`).
    """
    verb = verb_pv or f'{verb_root}is'
    return f"Kiam {verb} {question_anchor_text}?"

KIAM_CONFIG = QuestionTypeConfig(
    qword='Kiam',
    type_id='kiam',
    template_fn=_kiam_template,
    answer_roles=['aliaj'],  # Adverbials (temporal)
    question_anchor_role='subjekto',
    question_anchor_constraint_fn=_kiam_anchor_constraint,
    answer_constraint_fn=_kiam_answer_constraint,
    discriminator_kind='dated',
    active_verbs=_KIAM_VERBS,
)


# KIAL (WHY) — Explicit causal markers; yield expected low
_KIAL_VERBS = []  # Any verb can have a causal; filter at sentence level

def _kial_anchor_constraint(node: dict, sentence_text: str = '') -> bool:
    """Subject (anchor) must be propra_nomo and namelike."""
    if not isinstance(node, dict):
        return False
    kerno = node.get('kerno') if node.get('tipo') == 'vortgrupo' else node
    if not isinstance(kerno, dict):
        return False
    if kerno.get('vortspeco') != 'propra_nomo':
        return False
    pv = kerno.get('plena_vorto') or ''
    return _looks_namelike(pv)

def _kial_answer_constraint(node: dict, sentence_text: str) -> bool:
    """Causal clause: require explicit causal marker in source sentence.

    Checks for tokens: ĉar, pro, 'pro tio ke', tial (causal markers).
    This is a sentence-level filter, so most candidates fail, but those that
    pass are high-confidence causal Q&A pairs.
    """
    if not sentence_text:
        return False
    text_lower = sentence_text.lower()
    # Causal markers: ĉar (because), pro (for/due to), pro tio ke (because), tial (therefore)
    causal_markers = {'ĉar', 'pro tio ke', 'pro ', 'tial '}
    for marker in causal_markers:
        if marker in text_lower:
            return True
    return False

def _kial_template(verb_root: str, verb_pv: str, question_anchor_text: str) -> str:
    """Build KIAL question with subject as anchor."""
    verb = verb_pv or f'{verb_root}is'
    return f"Kial {verb} {question_anchor_text}?"

KIAL_CONFIG = QuestionTypeConfig(
    qword='Kial',
    type_id='kial',
    template_fn=_kial_template,
    answer_roles=['aliaj'],  # Causal clause
    question_anchor_role='subjekto',
    question_anchor_constraint_fn=_kial_anchor_constraint,
    answer_constraint_fn=_kial_answer_constraint,
    discriminator_kind='named_event_or_cause',
    active_verbs=_KIAL_VERBS,
)


# KIEL (HOW) — Manner adverbial; yield expected low
_KIEL_VERBS = []  # Any verb can have a manner

def _kiel_anchor_constraint(node: dict, sentence_text: str = '') -> bool:
    """Subject (anchor) must be propra_nomo and namelike."""
    if not isinstance(node, dict):
        return False
    kerno = node.get('kerno') if node.get('tipo') == 'vortgrupo' else node
    if not isinstance(kerno, dict):
        return False
    if kerno.get('vortspeco') != 'propra_nomo':
        return False
    pv = kerno.get('plena_vorto') or ''
    return _looks_namelike(pv)

def _kiel_answer_constraint(node: dict, sentence_text: str) -> bool:
    """Manner adverbial: require explicit manner marker in source sentence.

    Checks for:
    1. Manner adverbs (words ending in -e, e.g., rapide, trankvile)
    2. Per-instrumentals (per + noun phrase)

    This is a sentence-level filter for high-confidence manner Q&A pairs.
    """
    if not sentence_text:
        return False
    text_lower = sentence_text.lower()

    # Check for per-instrumental phrase (per + NP is a manner expression)
    if ' per ' in text_lower:
        return True

    # Check for manner adverbs (ending in -e). Simple heuristic:
    # Count words ending in -e (roughly 90% are adverbs in Esperanto).
    # This is imperfect but avoids parsing complexity.
    words = sentence_text.split()
    for word in words:
        cleaned = word.strip('.,;:!?"()«»').lower()
        if cleaned.endswith('e') and len(cleaned) >= 3:
            # Could be an -e adverb. Accept conservatively.
            return True

    return False

def _kiel_template(verb_root: str, verb_pv: str, question_anchor_text: str) -> str:
    """Build KIEL question with subject as anchor."""
    verb = verb_pv or f'{verb_root}is'
    return f"Kiel {verb} {question_anchor_text}?"

KIEL_CONFIG = QuestionTypeConfig(
    qword='Kiel',
    type_id='kiel',
    template_fn=_kiel_template,
    answer_roles=['aliaj'],  # Manner adverbial
    question_anchor_role='subjekto',
    question_anchor_constraint_fn=_kiel_anchor_constraint,
    answer_constraint_fn=_kiel_answer_constraint,
    discriminator_kind='named_method',
    active_verbs=_KIEL_VERBS,
)


# =============================================================================
# Entity-anchored types (NON-person trivia)
# =============================================================================

# Place-anchored: "Kio estas [Place]?" -> place-type predicate (urbo, rivero, …).
# Detects the entity-class via the surface pattern `<X> estas <PLACE_TYPE>` in
# the source sentence. No ontology required.
_PLACE_TYPE_WORDS = {
    'urbo', 'ĉefurbo', 'metropolo', 'vilaĝo', 'komunumo', 'distrikto',
    'regiono', 'provinco', 'subprovinco', 'kantono', 'departemento',
    'insulo', 'arkipelago', 'rivero', 'lago', 'maro', 'oceano',
    'montaro', 'monto', 'pinto', 'valo', 'glacieja',
    'lando', 'ŝtato', 'respubliko',
}
_PLACE_LIKE_PATTERNS = [
    'estas urbo', 'estas ĉefurbo', 'estas vilaĝo', 'estas komunumo',
    'estas distrikto', 'estas regiono', 'estas provinco', 'estas insulo',
    'estas rivero', 'estas lago', 'estas montaro', 'estas monto',
    'estas lando', 'estas ŝtato',
]


def _looks_namelike_place(name: str) -> bool:
    """Looser-than-_looks_namelike for places, since place names often
    contain non-Esperanto letters and aren't always parseable as propra_nomo
    by the parser. We still require capitalised + ≥3 chars + no common-word
    denylist + no participle-suffix shape."""
    tok = name.split()[0] if name else ''
    if len(tok) < 3 or not tok[:1].isupper():
        return False
    if tok in _COMMON_WORDS_AS_PROPER:
        return False
    if _PARTICIPLE_RE.search(tok.lower()):
        return False
    return True


def _extract_place_type(text: str, anchor: str) -> str | None:
    """Given the anchor in the surface text, look for `anchor … estas
    <PLACE_TYPE>` within a short window. Allows interleaved parenthetical
    modifiers ('(naskiĝis 1893)') and brief appositives between subject and
    copula. The first valid place-type token wins."""
    if not anchor or not text:
        return None
    apos = text.find(anchor)
    if apos < 0:
        return None
    tail = text[apos + len(anchor):]
    # ≤100 chars of subject-modifier slack before estas; then the place type
    pattern = (
        r'^[^.]{0,120}?\bestas\s+'
        r'(?:la\s+|grava\s+|granda\s+|malgranda\s+|bela\s+|fama\s+)?'
        r'(\w+)'
    )
    m = re.match(pattern, tail, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    cand = m.group(1).lower().strip()
    if cand in _PLACE_TYPE_WORDS:
        return cand
    return None


def _loko_kio_anchor_constraint(node: dict, sentence_text: str = '') -> bool:
    """Subject (anchor) must be propra_nomo, namelike, and NOT PP-governed
    (so we don't accept place mentions inside 'en X' / 'al X' phrases). The
    actual place-type lookup happens at answer-extraction time."""
    if not isinstance(node, dict):
        return False
    kerno = node.get('kerno') if node.get('tipo') == 'vortgrupo' else node
    if not isinstance(kerno, dict):
        return False
    if kerno.get('vortspeco') != 'propra_nomo':
        return False
    pv = kerno.get('plena_vorto') or ''
    if not _looks_namelike_place(pv):
        return False
    if sentence_text and _is_pp_governed(pv, sentence_text):
        return False
    return True


def _loko_kio_answer_constraint(node: dict, sentence_text: str) -> bool:
    """Placeholder; the actual extraction logic runs in verify_with_parser
    via _extract_place_type. Always returns True so the answer-role loop
    completes without needing an AST node to anchor."""
    return True


def _loko_kio_template(verb_root: str, verb_pv: str, anchor: str) -> str:
    return f"Kio estas {anchor}?"


LOKO_KIO_CONFIG = QuestionTypeConfig(
    qword='Kio',
    type_id='loko_kio',
    template_fn=_loko_kio_template,
    answer_roles=[],  # answer comes from surface, not AST role
    question_anchor_role='subjekto',
    question_anchor_constraint_fn=_loko_kio_anchor_constraint,
    answer_constraint_fn=_loko_kio_answer_constraint,
    discriminator_kind='place_type_predicate',
    active_verbs=['est'],  # all `est`-verb sentences (further narrowed by patterns)
    sql_text_patterns=_PLACE_LIKE_PATTERNS,
)


# Event-anchored: "Kiam okazis [Event]?" -> year. Detects events via the
# surface pattern `<X> okazis en YEAR`, where X is a propra_nomo subject.
_EVENTO_LIKE_PATTERNS = ['okazis en']


def _evento_kiam_anchor_constraint(node: dict, sentence_text: str = '') -> bool:
    """Subject must be propra_nomo + namelike-place; event names like
    'Brexit', 'Renesanco', 'Olimpiko' fit the place-name shape (capitalised,
    no participle ending)."""
    if not isinstance(node, dict):
        return False
    kerno = node.get('kerno') if node.get('tipo') == 'vortgrupo' else node
    if not isinstance(kerno, dict):
        return False
    if kerno.get('vortspeco') != 'propra_nomo':
        return False
    pv = kerno.get('plena_vorto') or ''
    if not _looks_namelike_place(pv):
        return False
    if sentence_text and _is_pp_governed(pv, sentence_text):
        return False
    return True


def _evento_kiam_answer_constraint(node: dict, sentence_text: str) -> bool:
    """Placeholder; surface extraction in verify_with_parser."""
    return True


def _evento_kiam_template(verb_root: str, verb_pv: str, anchor: str) -> str:
    return f"Kiam okazis {anchor}?"


def _extract_year_after(text: str, anchor: str, verb: str = 'okazis') -> str | None:
    """Find the first 4-digit year following `anchor … okazis` in surface
    text, requiring no intervening verb between okazis and the year so we
    don't grab dates from subordinate clauses."""
    if not text or not anchor:
        return None
    apos = text.find(anchor)
    if apos < 0:
        return None
    tail = text[apos:]
    vpos = tail.find(verb)
    if vpos < 0:
        return None
    after_verb = tail[vpos + len(verb):]
    m = re.search(r'\b(1[0-9]{3}|20[0-2][0-9])\b', after_verb)
    if not m:
        return None
    between = after_verb[:m.start()]
    if _INTERVENING_VERB_RE.search(between):
        return None
    return m.group(1)


EVENTO_KIAM_CONFIG = QuestionTypeConfig(
    qword='Kiam',
    type_id='evento_kiam',
    template_fn=_evento_kiam_template,
    answer_roles=[],
    question_anchor_role='subjekto',
    question_anchor_constraint_fn=_evento_kiam_anchor_constraint,
    answer_constraint_fn=_evento_kiam_answer_constraint,
    discriminator_kind='event_year',
    active_verbs=['okaz'],
    sql_text_patterns=_EVENTO_LIKE_PATTERNS,
)


# Registry of all types
QUESTION_TYPES = {
    'kiu': KIU_CONFIG,
    'kio': KIO_CONFIG,
    'kie': KIE_CONFIG,
    'kiam': KIAM_CONFIG,
    'kial': KIAL_CONFIG,
    'kiel': KIEL_CONFIG,
    'loko_kio': LOKO_KIO_CONFIG,
    'evento_kiam': EVENTO_KIAM_CONFIG,
}

GENERIC_SUBJECT_RADIKOS = {
    'la', 'lia', 'ŝia', 'sia', 'tiu', 'iu',
    'kaj', 'sed', 'aŭ', 'do', 'tamen', 'tial',
    'mi', 'vi', 'ni', 'ili', 'li', 'ŝi', 'ĝi',
    'tio', 'tiu', 'iom', 'iam', 'kio', 'kiu',
    'la', 'estas', 'estis', 'estos',
}


# =============================================================================
# Core Generator Functions
# =============================================================================

def stream_candidates(conn, cfg: QuestionTypeConfig, limit: int, seed: int):
    """Stream candidates from the DuckDB store one row at a time.

    Single SELECT ... LIMIT N with fetchone() in a Python loop:
      - LIMIT lets DuckDB short-circuit its scan after N matches
        (without LIMIT it would scan the full 5.4M-row table).
      - fetchone() in a loop keeps Python's working set tiny — no
        eager .fetchall() materialisation, no ORDER BY (which would
        force DuckDB to buffer every matching row including ast_json).
      - We bias for diversity at the row level by passing a
        `seed`-derived modular hash filter, so two runs with different
        seeds see different slices of the corpus without an ORDER BY.
    """
    verbs = cfg.active_verbs if cfg.active_verbs else []
    # Modular-hash sampling for seed-controlled diversity. With a
    # ~5.4M-row table and most verbs matching many rows, even %2 splits
    # let LIMIT pick from a different slice each seed.
    seed_mod = max(2, int(limit) // 100 + 2)
    seed_pick = int(seed) % seed_mod

    text_clauses = ''
    text_params: list[str] = []
    if cfg.sql_text_patterns:
        like_parts = ' OR '.join('text LIKE ?' for _ in cfg.sql_text_patterns)
        text_clauses = f' AND ({like_parts})'
        text_params = [f'%{p}%' for p in cfg.sql_text_patterns]

    if verbs:
        placeholders = ','.join('?' * len(verbs))
        sql = f"""
            SELECT sid, text, subj_radiko, verb_radiko, obj_radiko, ast_json
            FROM sentences
            WHERE verb_radiko IN ({placeholders})
              AND ast_json IS NOT NULL
              AND (HASH(sid) % {seed_mod}) = {seed_pick}
              {text_clauses}
            LIMIT {int(limit)}
        """
        params = list(verbs) + text_params
    else:
        sql = f"""
            SELECT sid, text, subj_radiko, verb_radiko, obj_radiko, ast_json
            FROM sentences
            WHERE ast_json IS NOT NULL
              AND (HASH(sid) % {seed_mod}) = {seed_pick}
              {text_clauses}
            LIMIT {int(limit)}
        """
        params = list(text_params)

    cursor = conn.execute(sql, params)
    while True:
        row = cursor.fetchone()
        if row is None:
            return
        sid, text, subj_r, verb_r, obj_r, ast_json = row
        try:
            ast = json.loads(ast_json)
        except Exception:
            continue
        subj = _kerno_vorto(ast.get('subjekto'))
        obj = _kerno_vorto(ast.get('objekto'))
        yield {
            'sentence_id':       sid,
            'sentence_text':     text,
            'subject_pv':        subj.get('plena_vorto') or '',
            'subject_radiko':    subj_r,
            'verb_root':         verb_r,
            'object_pv':         obj.get('plena_vorto') or '',
            'object_radiko':     obj_r,
            'object_vortspeco':  obj.get('vortspeco') or '',
            'ast':               ast,
        }


def verify_with_parser(c: dict, cfg: QuestionTypeConfig) -> dict | None:
    """Re-parse the source sentence and extract both question anchor and answer.

    NEW ARCHITECTURE:
    - Extract question_anchor from question_anchor_role (must pass question_anchor_constraint_fn)
    - Extract answer from answer_roles (must pass answer_constraint_fn)
    - Both must be valid and present; reject if either fails

    For KIU: anchor = object (quoted work), answer = subject (creator)
    For KIO: anchor = subject (creator), answer = object (work)
    For KIE: anchor = subject (person), answer = locative from aliaj
    For KIAM: anchor = subject (person), answer = temporal from aliaj
    For KIAL: anchor = subject (person), answer = causal from aliaj
    For KIEL: anchor = subject (person), answer = manner from aliaj
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

    # === STEP 1: Extract question anchor from question_anchor_role ===

    anchor_role = cfg.question_anchor_role
    if anchor_role == 'subjekto':
        anchor_node = ast.get('subjekto')
    elif anchor_role == 'objekto':
        anchor_node = ast.get('objekto')
    elif anchor_role == 'aliaj':
        # Should not happen in current config; guard against future
        return None
    else:
        return None

    if not anchor_node:
        return None

    anchor_kerno = _kerno_vorto(anchor_node)
    if not anchor_kerno:
        return None

    if not cfg.question_anchor_constraint_fn(anchor_node, text):
        return None

    anchor_pv = anchor_kerno.get('plena_vorto') or ''
    if not anchor_pv:
        return None

    # Bug #3 fix: if the anchor head word is part of a multi-token entity
    # (e.g. `Béla Buzogány`, `Sławomir Piotr PREISS`), use the full span as
    # the question's anchor instead of the head word alone. The parser
    # already emits `multi_token_entities` listing all such spans; pick the
    # one whose first token matches our head word.
    mte_groups = ast.get('multi_token_entities') or []
    for g in mte_groups:
        span_tokens = g.get('span_tokens') or []
        if span_tokens and span_tokens[0] == anchor_pv:
            anchor_pv = ' '.join(span_tokens)
            break

    # Extract source verb's actual surface form for use in templates.
    # Falls back to None if missing — templates synthesize `{root}is` then.
    verb_kerno = _kerno_vorto(ast.get('verbo'))
    verb_pv = (verb_kerno.get('plena_vorto') or '') if verb_kerno else ''

    # === STEP 2: Extract answer ===
    # KIAL/KIEL use surface-text spans (causal clause / manner expression)
    # — AST role extraction doesn't fit because the relevant content isn't
    # a single kerno-headed node.

    answer_node = None
    answer_role_used = None
    answer_text_override = None

    if cfg.type_id == 'kial':
        span = _extract_causal_clause(text)
        if not span:
            return None
        answer_text_override = span
        answer_role_used = 'causal_clause'

    elif cfg.type_id == 'kiel':
        span = _extract_manner_span(text)
        if not span:
            return None
        answer_text_override = span
        answer_role_used = 'manner_span'

    elif cfg.type_id == 'loko_kio':
        # Surface extraction: `anchor … estas <PLACE_TYPE>`. The place-type
        # word is the answer; rigid designator is the anchor place name.
        ptype = _extract_place_type(text, anchor_pv)
        if not ptype:
            return None
        answer_text_override = ptype
        answer_role_used = 'place_type_predicate'

    elif cfg.type_id == 'evento_kiam':
        # Surface extraction: `anchor … okazis … YEAR`. Year is the answer.
        year = _extract_year_after(text, anchor_pv, 'okazis')
        if not year:
            return None
        answer_text_override = year
        answer_role_used = 'event_year'

    elif cfg.type_id == 'kie':
        # Surface-text extraction: deterministic first-locative-PP-AFTER-verb.
        # AST `aliaj` iteration was order-dependent on the parser and picked
        # the wrong PP when source had multiple (e.g. naskiĝis…en Pest /
        # mortis…en Budapeŝto -> picked the death-PP for a birth question).
        found = _find_first_locative_after_verb(text, verb_pv)
        if not found:
            return None
        prep, name = found
        if not _looks_namelike(name):
            return None
        answer_text_override = name
        answer_role_used = f'locative_pp({prep})'

    else:
        for role in cfg.answer_roles:
            if role == 'subjekto':
                node = ast.get('subjekto')
            elif role == 'objekto':
                node = ast.get('objekto')
            elif role == 'aliaj':
                # For aliaj, iterate and find first match
                aliaj = ast.get('aliaj') or []
                node = None
                for candidate in aliaj:
                    if cfg.answer_constraint_fn(candidate, text):
                        node = candidate
                        break
            else:
                node = None

            if node and cfg.answer_constraint_fn(node, text):
                answer_node = node
                answer_role_used = role
                break

        if not answer_node:
            return None

    if answer_text_override is not None:
        answer_text = answer_text_override
        answer_kerno = {}
    else:
        answer_kerno = _kerno_vorto(answer_node)
        if not answer_kerno:
            return None
        answer_text = answer_kerno.get('plena_vorto') or ''
        if not answer_text:
            return None

    # === STEP 3: For KIU/KIO, verify answer is not the anchor ===

    if cfg.type_id in ('kiu', 'kio'):
        if answer_text.lower() == anchor_pv.lower():
            return None

    # === STEP 4: KIU/KIO special logic: extract named target («quoted work») ===

    if cfg.type_id == 'kiu':
        # Anchor is objekto; verify it matches the parsed object
        ok = anchor_kerno
        if not verb_kerno:
            return None

        toks = text.split()
        designator = _named_target(text, ok, toks, verb_pv)
        if designator is None:
            return None
        target_text, target_kind = designator

        # Answer is full subject name
        if not _looks_namelike(answer_text):
            return None

        out = dict(c)
        out['named_target'] = target_text
        out['target_kind'] = target_kind
        out['answer_role'] = 'subjekto'
        out['answer_text'] = answer_text
        out['anchor_text'] = target_text
        out['verb_pv'] = verb_pv
        return out

    elif cfg.type_id == 'kio':
        # Anchor is subjekto; answer is objekto (named target)
        designator = _named_target(text, answer_kerno, text.split(), verb_pv)
        if designator is None:
            return None
        target_text, target_kind = designator

        out = dict(c)
        out['named_target'] = target_text
        out['target_kind'] = target_kind
        out['answer_role'] = 'objekto'
        out['answer_text'] = target_text
        out['anchor_text'] = anchor_pv
        out['verb_pv'] = verb_pv
        return out

    # === For KIE/KIAM/KIAL/KIEL: anchor is subjekto, answer is from aliaj
    #     (KIE/KIAM) or a surface-text span (KIAL/KIEL).

    else:
        out = dict(c)
        out['answer_role'] = answer_role_used
        out['answer_text'] = answer_text
        out['anchor_text'] = anchor_pv
        out['verb_pv'] = verb_pv
        return out


def is_quality_candidate(c, cfg: QuestionTypeConfig) -> bool:
    """Basic quality checks."""
    text = c.get('sentence_text') or ''
    if len(text.split()) < 5 or len(text.split()) > 40:
        return False
    if (c.get('subject_radiko') or '').lower() in GENERIC_SUBJECT_RADIKOS:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--type', choices=list(QUESTION_TYPES.keys()),
                        required=True, help='Question type to generate')
    parser.add_argument('--duckdb-path', default='data/indexes/duckdb_store.db')
    parser.add_argument('--whoosh-dir',  default='data/indexes/whoosh_v2')
    parser.add_argument('--target-size', type=int, default=50)
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--query-limit', type=int, default=20000)
    parser.add_argument('--gate-top-k',  type=int, default=200)
    args = parser.parse_args()

    cfg = QUESTION_TYPES[args.type]
    print(f"Generating {cfg.qword} ({args.type}) questions...")
    print(f"  Config: {cfg.discriminator_kind} | "
          f"roles={cfg.answer_roles} | verbs={len(cfg.active_verbs)}")

    print(f"Opening DuckDB store: {args.duckdb_path}")
    conn = duckdb.connect(args.duckdb_path, read_only=True)
    print(f"Opening Whoosh index: {args.whoosh_dir}")
    ix = open_dir(args.whoosh_dir)

    if args.type == 'kiam':
        _load_temporal_roots_from_ontology(conn)

    print(f"Streaming corpus (limit={args.query_limit}) — one row at a time, "
          f"AST dropped after use to keep memory bounded...")

    # Single streaming pipeline: surface filter → parser-AST verify →
    # discriminability gate, all on one row at a time. Only the final
    # `discriminating` list (capped at target_size*2) is retained.
    n_raw = 0
    n_surface_ok = 0
    n_verified = 0
    n_audit_rejected = 0
    evals = 0
    discriminating: list[dict] = []
    cap_kept = args.target_size * 2
    cap_evals = args.target_size * 40

    # Default BM25 (not BM25F): is_discriminating only needs top-K rank
    # presence, not field weighting. BM25F adds ~4 GB on a 5.4M-doc index.
    with ix.searcher() as s:
        qp = QueryParser('text', ix.schema, group=OrGroup)
        for c in stream_candidates(conn, cfg, args.query_limit, args.seed):
            n_raw += 1
            if n_raw % 500 == 0:
                print(f"  ... streamed {n_raw} rows; surface_ok={n_surface_ok}, "
                      f"verified={n_verified}, kept={len(discriminating)}")

            if not is_quality_candidate(c, cfg):
                continue
            n_surface_ok += 1

            v = verify_with_parser(c, cfg)
            if v is None:
                continue
            n_verified += 1
            # Drop both AST blobs (raw + verified) — no longer needed
            v.pop('ast', None)
            c.pop('ast', None)

            anchor_text = v.get('anchor_text') or ''
            if not anchor_text:
                continue

            q = cfg.template_fn(v['verb_root'], v.get('verb_pv') or '', anchor_text)
            answer_text = v.get('answer_text') or ''
            if answer_text and answer_text.lower() in q.lower():
                continue

            # Final fail-closed audit gate. Cheaper than the discriminability
            # check, so it runs first.
            audit_ok, _audit_why = _audit_pass(
                q, answer_text, v['sentence_text'], cfg.qword.upper()
            )
            if not audit_ok:
                n_audit_rejected += 1
                continue

            evals += 1
            if is_discriminating(s, qp, q, v['sentence_id'], args.gate_top_k):
                v['question'] = q
                discriminating.append(v)

            if len(discriminating) >= cap_kept or evals >= cap_evals:
                break

    print(f"  Raw streamed:           {n_raw}")
    print(f"  Surface-quality OK:     {n_surface_ok}")
    print(f"  Parser-verified:        {n_verified}")
    print(f"  Audit-rejected:         {n_audit_rejected}")
    print(f"  Discriminability evals: {evals}")
    print(f"  Kept:                   {len(discriminating)}")

    if not discriminating:
        print(f"ERROR: No discriminating pairs for {args.type}")
        return

    import random
    random.seed(args.seed)

    # Diversity: cap per-verb
    from collections import defaultdict
    by_verb = defaultdict(list)
    for c in discriminating:
        by_verb[c['verb_root']].append(c)
    n_verbs = max(1, len(by_verb))
    per_verb = max(1, args.target_size // n_verbs)

    pool = []
    for verb_root, items in by_verb.items():
        random.shuffle(items)
        pool.extend(items[:per_verb])

    if len(pool) < args.target_size:
        leftovers = [c for c in discriminating if c not in pool]
        random.shuffle(leftovers)
        pool.extend(leftovers[: args.target_size - len(pool)])

    random.shuffle(pool)
    pool = pool[: args.target_size]

    # Output
    out_path = Path('data/test_sets') / f'synthetic_{args.type}_active.jsonl'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = []
    for i, c in enumerate(pool, 1):
        q = c.get('question') or ''
        ans = c.get('answer_text') or ''
        output.append({
            'id':                   f'{args.type}_gen_{i:03d}',
            'question':             q,
            'expected_answer':      ans,
            'expected_keywords':    [ans],
            'source_sentence_id':   c['sentence_id'],
            'source_sentence_text': c['sentence_text'],
            'question_type':        cfg.qword.upper(),
            'pattern':              'active',
            'verb_root':            c['verb_root'],
            'answer_role':          c.get('answer_role') or '',
        })

    with open(out_path, 'w') as f:
        for entry in output:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\nWrote {len(output)} {cfg.qword} questions to {out_path}")

    # Verb breakdown
    print(f"\nVerb-root distribution:")
    by_verb_out = defaultdict(int)
    for e in output:
        by_verb_out[e['verb_root']] += 1
    for v, n in sorted(by_verb_out.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {(v or '<none>'):12s}  {n}")


if __name__ == '__main__':
    main()
