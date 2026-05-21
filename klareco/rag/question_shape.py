"""
Question-shape router: deterministic classifier from a question AST to a
*retrieval strategy*.

VERSION: v2.x
COMPATIBLE WITH: klareco.parser (post-bug-fix), klareco.rag.question_classifier
DEPENDENCIES: klareco.parser
STAGE: Retrieval

Description:
    The existing `QuestionClassifier` gives us the question word's type
    (WHO/WHAT/WHERE/WHEN/HOW/WHY) and a coarse expected-answer type.
    For retrieval routing we need a finer-grained shape that maps each
    question to a concrete retrieval strategy:

      Shape                Example                                 Strategy
      -----------------    -------------------------------------   -------------------------
      capital_of           Kio estas la ĉefurbo de Brazilo?        pattern_capital_of lookup
      founded_year_of      En kiu jaro fondiĝis Pakistano?         pattern_founded_year_of
      official_language_of Kio estas la oficiala lingvo de X?      pattern_official_language
      who_did              Kiu kreis Esperanton?                   entity_postings + verb_klaso
      who_did_to_work      Kiu verkis «Faust»?                     entity_postings + quoted-work
      where_born           Kie naskiĝis [Person]?                  entity_postings + locative-PP
      when_born            Kiam naskiĝis [Person]?                 entity_postings + temporal-PP
      what_is              Kio estas Esperanto?                    entity_postings (loose)
      generic_kiu          Kiu (other)                             BM25 fallback
      ...
      unstructured         (no clear shape)                        BM25 fallback

    Each shape carries enough info for the retriever to compose a
    structured query: anchor entity, verb root (if any), expected
    answer type, scope filters (locative / temporal / etc.).

Pipeline Position:
    question_ast → [question_shape.classify] → QuestionShape →
    ASTRetriever (routes by shape) → candidates → reranker → extractor

Last Updated: 2026-05-20
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Shape(Enum):
    """Concrete question shapes the AST retriever can route on."""
    # Definite-description / fact-lookup shapes
    CAPITAL_OF             = 'capital_of'              # Kio estas la ĉefurbo de Y?
    FOUNDED_YEAR_OF        = 'founded_year_of'         # En kiu jaro fondiĝis X?
    OFFICIAL_LANGUAGE_OF   = 'official_language_of'    # Kio estas la oficiala lingvo de Y?
    CURRENCY_OF            = 'currency_of'             # Kio estas la valuto de Y?
    NATIONAL_ANTHEM_OF     = 'national_anthem_of'      # Kio estas la nacia himno de Y?

    # WHO shapes
    WHO_AGENT_OF_WORK      = 'who_agent_of_work'       # Kiu [verb]is «Title»?
    WHO_AGENT              = 'who_agent'               # Kiu [verb]is X? (X is named, not quoted)
    WHO_INVENTED_DISCOVERED = 'who_invented_discovered' # Kiu inventis/malkovris X?

    # KIE shapes
    WHERE_BORN             = 'where_born'              # Kie naskiĝis X?
    WHERE_LOCATED          = 'where_located'           # En kiu lando situas X?
    WHERE_OCCURRED         = 'where_occurred'          # Kie okazis X?

    # KIAM shapes
    WHEN_BORN              = 'when_born'               # Kiam naskiĝis X?
    WHEN_OCCURRED          = 'when_occurred'           # Kiam okazis X?
    WHEN_FOUNDED           = 'when_founded'            # En kiu jaro fondiĝis X?

    # Generic / fallbacks
    WHAT_IS                = 'what_is'                  # Kio estas X?
    GENERIC_KIU            = 'generic_kiu'              # Kiu... (no specific shape)
    GENERIC_KIE            = 'generic_kie'
    GENERIC_KIAM           = 'generic_kiam'
    UNSTRUCTURED           = 'unstructured'             # No clear shape


@dataclass
class QuestionShape:
    """Result of classifying a question."""
    shape:           Shape
    anchor_entity:   Optional[str]   = None    # The named-thing the question is about
    constraint_y:    Optional[str]   = None    # For "X of Y" shapes, Y
    verb_radiko:     Optional[str]   = None    # Question's verb if present
    quoted_work:     Optional[str]   = None    # «...» title if present
    notes:           list[str]       = field(default_factory=list)


# Pre-compiled patterns
_QUOTED_RE = re.compile(r'[«"„]\s*([^«»"]{2,80}?)\s*[»"]')
_PROPER_NOUN = re.compile(
    r'\b[A-ZÀ-ÞĈĜĤĴŜŬ][\wÀ-ſĉĝĥĵŝŭĈĜĤĴŜŬ.-]{2,}'
    r'(?:\s+[A-ZÀ-ÞĈĜĤĴŜŬ][\wÀ-ſĉĝĥĵŝŭĈĜĤĴŜŬ.-]+){0,3}'
)
_DEFINITE_DESCRIPTIONS = {
    'la ĉefurbo de':       Shape.CAPITAL_OF,
    'la oficiala lingvo de': Shape.OFFICIAL_LANGUAGE_OF,
    'la valuto de':        Shape.CURRENCY_OF,
    'la moneda unuo de':   Shape.CURRENCY_OF,
    'la nacia himno de':   Shape.NATIONAL_ANTHEM_OF,
}
_INVENTION_VERBS = {'inventis', 'malkovris', 'eltrovis', 'kreis'}
_BIRTH_VERBS = {'naskiĝis', 'naskis'}


def _kerno(node) -> dict:
    if not isinstance(node, dict):
        return {}
    if node.get('tipo') == 'vortgrupo':
        return node.get('kerno') or {}
    return node


def _extract_anchor_from_text(text: str) -> Optional[str]:
    """Last-resort: pull a proper-noun span from the question."""
    # Strip question words and common verb forms; what remains often contains the anchor
    leading = text
    for q in ('Kio ', 'Kiu ', 'Kion ', 'Kiun ', 'Kie ', 'Kien ', 'Kiam ',
              'Kial ', 'Kiel ', 'Kiom ', 'Kies ', 'Kia '):
        if leading.startswith(q):
            leading = leading[len(q):]
            break
    m = _PROPER_NOUN.search(leading)
    if m:
        return m.group(0)
    return None


def _extract_anchor_from_ast(ast: dict) -> Optional[str]:
    """Pull a named anchor entity from the question AST.

    Priority:
      1. Multi-token entity span
      2. Quoted work («...»)
      3. Propra_nomo in aliaj (most common in question-style fronted PPs)
      4. Propra_nomo in objekto.kerno
      5. Propra_nomo in subjekto.kerno
    """
    mte = ast.get('multi_token_entities') or []
    if mte:
        span = mte[0].get('span_tokens') or []
        if span:
            return ' '.join(span)
    for role in ('aliaj',):
        items = ast.get(role) or []
        for item in items:
            k = _kerno(item)
            if k.get('vortspeco') == 'propra_nomo':
                pv = k.get('plena_vorto')
                if pv:
                    return pv
    for role in ('objekto', 'subjekto'):
        k = _kerno(ast.get(role))
        if k.get('vortspeco') == 'propra_nomo':
            pv = k.get('plena_vorto')
            if pv:
                return pv
    return None


def _extract_constraint_y(text: str, marker_phrase: str) -> Optional[str]:
    """For `<marker_phrase> Y?` patterns, extract Y."""
    idx = text.find(marker_phrase)
    if idx < 0:
        return None
    tail = text[idx + len(marker_phrase):].lstrip()
    m = _PROPER_NOUN.match(tail)
    if m:
        return m.group(0)
    return None


def classify(question_text: str, question_ast: dict) -> QuestionShape:
    """Map (question_text, question_ast) to a QuestionShape.

    Looks at:
      - surface text for definite-description marker phrases
      - quoted spans
      - the question's verb_radiko
      - the question's leading interrogative
      - propra_nomos and multi-token entities in the AST
    """
    q = question_text.strip()
    notes: list[str] = []

    # Definite-description shapes (most specific — check first)
    for marker, shape in _DEFINITE_DESCRIPTIONS.items():
        if marker in q:
            y = _extract_constraint_y(q, marker)
            return QuestionShape(
                shape=shape,
                constraint_y=y,
                notes=[f'matched definite-description marker {marker!r}'],
            )

    # «quoted work» as the anchor for WHO questions
    qm = _QUOTED_RE.search(q)
    quoted_work = qm.group(1).strip() if qm else None

    # Extract verb radiko if available
    verb_radiko = (question_ast.get('verbo') or {}).get('radiko') if question_ast else None
    verb_pv = (question_ast.get('verbo') or {}).get('plena_vorto') if question_ast else None
    anchor = _extract_anchor_from_ast(question_ast) if question_ast else None
    if anchor is None:
        anchor = _extract_anchor_from_text(q)

    first_word = q.split()[0] if q.split() else ''

    # KIU shapes
    if first_word in ('Kiu', 'Kiun'):
        if quoted_work:
            return QuestionShape(
                shape=Shape.WHO_AGENT_OF_WORK,
                anchor_entity=quoted_work,
                verb_radiko=verb_radiko,
                quoted_work=quoted_work,
                notes=['quoted work anchor'],
            )
        if verb_pv in _INVENTION_VERBS or verb_radiko in {'invent', 'malkov', 'eltrov'}:
            return QuestionShape(
                shape=Shape.WHO_INVENTED_DISCOVERED,
                anchor_entity=anchor,
                verb_radiko=verb_radiko,
                notes=['invention/discovery verb'],
            )
        if anchor:
            return QuestionShape(
                shape=Shape.WHO_AGENT,
                anchor_entity=anchor,
                verb_radiko=verb_radiko,
                notes=['named anchor in WHO question'],
            )
        return QuestionShape(shape=Shape.GENERIC_KIU,
                             verb_radiko=verb_radiko,
                             notes=['WHO question without clear anchor'])

    # KIE shapes
    if first_word == 'Kie' or q.startswith('En kiu lando'):
        if verb_pv in _BIRTH_VERBS or verb_radiko == 'nask':
            return QuestionShape(
                shape=Shape.WHERE_BORN,
                anchor_entity=anchor,
                verb_radiko=verb_radiko,
                notes=['birthplace question'],
            )
        if verb_pv in {'okazis', 'okazi'} or verb_radiko == 'okaz':
            return QuestionShape(
                shape=Shape.WHERE_OCCURRED,
                anchor_entity=anchor,
                verb_radiko=verb_radiko,
                notes=['event-location question'],
            )
        if 'situas' in q or 'situa' in q or q.startswith('En kiu lando'):
            return QuestionShape(
                shape=Shape.WHERE_LOCATED,
                anchor_entity=anchor,
                verb_radiko=verb_radiko,
                notes=['where-located question'],
            )
        return QuestionShape(shape=Shape.GENERIC_KIE,
                             anchor_entity=anchor,
                             verb_radiko=verb_radiko,
                             notes=['generic where'])

    # KIAM shapes
    if first_word == 'Kiam' or q.startswith('En kiu jaro'):
        if 'fondiĝis' in q or 'fondita' in q or verb_radiko in {'fond', 'establ', 'kre'}:
            return QuestionShape(
                shape=Shape.WHEN_FOUNDED,
                anchor_entity=anchor,
                verb_radiko=verb_radiko,
                notes=['founding-year question'],
            )
        if verb_pv in _BIRTH_VERBS or verb_radiko == 'nask':
            return QuestionShape(
                shape=Shape.WHEN_BORN,
                anchor_entity=anchor,
                verb_radiko=verb_radiko,
                notes=['birth-year question'],
            )
        if verb_pv in {'okazis', 'okazi'} or verb_radiko == 'okaz':
            return QuestionShape(
                shape=Shape.WHEN_OCCURRED,
                anchor_entity=anchor,
                verb_radiko=verb_radiko,
                notes=['event-when question'],
            )
        return QuestionShape(shape=Shape.GENERIC_KIAM,
                             anchor_entity=anchor,
                             verb_radiko=verb_radiko,
                             notes=['generic when'])

    # KIO shapes
    if first_word in ('Kio', 'Kion'):
        if anchor:
            return QuestionShape(
                shape=Shape.WHAT_IS,
                anchor_entity=anchor,
                verb_radiko=verb_radiko,
                notes=['What is X? with named anchor'],
            )
        return QuestionShape(shape=Shape.UNSTRUCTURED,
                             verb_radiko=verb_radiko,
                             notes=['KIO with no anchor'])

    # Anything else → unstructured
    return QuestionShape(
        shape=Shape.UNSTRUCTURED,
        anchor_entity=anchor,
        verb_radiko=verb_radiko,
        notes=[f'no shape matched (first word: {first_word!r})'],
    )
