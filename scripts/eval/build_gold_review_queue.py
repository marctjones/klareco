#!/usr/bin/env python3
"""
Build a review queue for a corpus-backed Esperanto trivia gold set.

This is intentionally NOT a retrievability-gated generator. It samples
candidate factual questions from DuckDB evidence tables, attaches the exact
DuckDB source sentence as the answer sentence, and marks every row as
`needs_review`. Human review decides whether the item becomes gold.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import duckdb

from klareco.parser import parse


META_RE = re.compile(
    r'\b(?:diskuto|UTC|Vikipedio:|ALIDIREKTI|ALIDIREKTU|REDIRECT|'
    r'ISBN|Dosiero:|Kategorio:)\b',
    re.IGNORECASE,
)
BAD_TEXT_RE = re.compile(r'[\[\]{}<>]|https?://|--')
QUESTION_STARTS = ('Kio ', 'Kiu ', 'Kion ', 'Kiam ', 'Kie ', 'En kiu ')
COMMON_FALSE_ANSWERS = {
    'La', 'Ĝi', 'Li', 'Ŝi', 'Ili', 'Oni', 'Nuntempe', 'Anstataŭe',
    'Teorio', 'Elemente', 'Visite', 'Drame', 'Kompanio', 'Company',
    'Nordo', 'ABD',
}
COUNTRYLIKE_SUFFIXES = (
    'io', 'ujo', 'lando', 'landoj', 'io.', 'ujo.', 'lando.',
    'o',  # Brazilo, Kataro, Meksiko, Japanio already covered by io
)


def clean_text(s: str | None) -> str:
    return ' '.join((s or '').split())


def plausible_source_sentence(text: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not text:
        reasons.append('empty_source')
    if len(text) < 45:
        reasons.append('source_too_short')
    if len(text) > 240:
        reasons.append('source_too_long')
    if META_RE.search(text):
        reasons.append('meta_or_redirect_marker')
    if BAD_TEXT_RE.search(text):
        reasons.append('markup_or_url_marker')
    if '?' in text:
        reasons.append('source_is_question_or_dialog')
    if '))' in text or '( )' in text or '()' in text:
        reasons.append('bad_parentheses')
    if text.count('"') % 2:
        reasons.append('unbalanced_quote')
    return not reasons, reasons


def plausible_answer_fragment(answer: str, source: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    answer = clean_text(answer).strip('.,;:')
    if not answer:
        reasons.append('empty_short_answer')
    if answer not in source:
        reasons.append('short_answer_not_in_source')
    if answer.split()[0] in COMMON_FALSE_ANSWERS:
        reasons.append('common_false_answer')
    if len(answer) < 3:
        reasons.append('short_answer_too_short')
    if len(answer) > 80:
        reasons.append('short_answer_too_long')
    if BAD_TEXT_RE.search(answer):
        reasons.append('bad_answer_marker')
    return not reasons, reasons


def parse_ok(text: str) -> tuple[bool, str]:
    try:
        ast = parse(text)
    except Exception as exc:
        return False, f'parse_exception:{exc}'
    if not isinstance(ast, dict):
        return False, 'parse_returned_non_dict'
    return True, 'ok'


def mk_row(
    *,
    item_id: str,
    question: str,
    short_answer: str,
    source_sid: int,
    source_sentence: str,
    question_type: str,
    topic: str,
    source_pattern: str,
) -> dict | None:
    source_sentence = clean_text(source_sentence)
    short_answer = clean_text(short_answer).strip('.,;:')
    question = clean_text(question)

    source_ok, source_reasons = plausible_source_sentence(source_sentence)
    answer_ok, answer_reasons = plausible_answer_fragment(
        short_answer, source_sentence
    )
    q_parse_ok, q_parse_note = parse_ok(question)
    a_parse_ok, a_parse_note = parse_ok(source_sentence)
    q_shape_ok = question.startswith(QUESTION_STARTS) and question.endswith('?')

    if not source_ok or not answer_ok or not q_shape_ok:
        return None

    quality_flags = []
    if not q_parse_ok:
        quality_flags.append(f'question_{q_parse_note}')
    if not a_parse_ok:
        quality_flags.append(f'answer_sentence_{a_parse_note}')

    return {
        'id': item_id,
        'question': question,
        'expected_answer': source_sentence,
        'short_answer': short_answer,
        'expected_keywords': [short_answer],
        'source_sentence_id': int(source_sid),
        'source_sentence_text': source_sentence,
        'question_type': question_type,
        'topic': topic,
        'source_pattern': source_pattern,
        'review_status': 'needs_review',
        'review_decision': None,
        'reject_reason': None,
        'auto_quality': {
            'source_reasons': source_reasons,
            'answer_reasons': answer_reasons,
            'question_parse': q_parse_note,
            'answer_sentence_parse': a_parse_note,
            'flags': quality_flags,
            'retrieval_gated': False,
        },
    }


def rows_from_patterns(conn, per_pattern: int) -> list[dict]:
    out: list[dict] = []

    pattern_specs = [
        {
            'table': 'pattern_capital_of',
            'cols': 'city, country, sid',
            'question': lambda city, country: f'Kio estas la ĉefurbo de {country}?',
            'answer': lambda city, country: city,
            'qtype': 'capital',
            'topic': 'geografio',
        },
        {
            'table': 'pattern_founded_year_of',
            'cols': 'org, year, sid',
            'question': lambda org, year: f'En kiu jaro fondiĝis {org}?',
            'answer': lambda org, year: year,
            'qtype': 'founded_year',
            'topic': 'historio',
        },
        {
            'table': 'pattern_official_language_of',
            'cols': 'language, country, sid',
            'question': lambda language, country: f'Kiu estas la oficiala lingvo de {country}?',
            'answer': lambda language, country: language,
            'qtype': 'official_language',
            'topic': 'geografio',
        },
        {
            'table': 'pattern_currency_of',
            'cols': 'currency, country, sid',
            'question': lambda currency, country: f"Kiu estas la valuto de {country.strip('.')}?",
            'answer': lambda currency, country: currency,
            'qtype': 'currency',
            'topic': 'geografio',
        },
    ]

    for spec in pattern_specs:
        sql = (
            f"SELECT {spec['cols']}, s.text "
            f"FROM {spec['table']} p "
            f"JOIN sentences s USING (sid) "
            f"ORDER BY sid"
        )
        n = 0
        for a, b, sid, text in conn.execute(sql).fetchall():
            if spec['table'] == 'pattern_capital_of':
                country = clean_text(b).strip('.,;:')
                city = clean_text(a).strip('.,;:')
                if not country.endswith(COUNTRYLIKE_SUFFIXES):
                    continue
                if city in {'Rusio', 'Usono', 'Francio', 'Germanio'}:
                    continue
            question = spec['question'](clean_text(a), clean_text(b))
            short_answer = spec['answer'](clean_text(a), clean_text(b))
            row = mk_row(
                item_id=f"review_{spec['qtype']}_{n + 1:03d}",
                question=question,
                short_answer=short_answer,
                source_sid=sid,
                source_sentence=text,
                question_type=spec['qtype'],
                topic=spec['topic'],
                source_pattern=spec['table'],
            )
            if row is None:
                continue
            out.append(row)
            n += 1
            if n >= per_pattern:
                break
    return out


def rows_from_manual_evidence(conn) -> list[dict]:
    seeds = [
        (
            'Kiu fondis Makita Electric Works?',
            'Musaburo Makita',
            "text ILIKE '%Musaburo Makita%' AND text ILIKE '%fondis%'",
            'who_agent',
            'historio',
        ),
        (
            'Kiu verkis la novelon "La neĝoj de Kilimanĝaro"?',
            'Ernest Hemingway',
            "text ILIKE '%Ernest Hemingway%' AND text ILIKE '%La neĝoj de Kilimanĝaro%'",
            'who_agent',
            'literaturo',
        ),
        (
            'En kiu jaro aperis la unua broŝuro pri Esperanto?',
            '1887',
            "text ILIKE '%unua broŝuro%' AND text ILIKE '%1887%' AND text ILIKE '%Esperanto%'",
            'when_year',
            'Esperanto',
        ),
        (
            'Sub kiu kaŝnomo Zamenhof publikigis sian lingvoprojekton?',
            'd-ro Esperanto',
            "text ILIKE '%pseŭdonimo%' AND text ILIKE '%d-ro Esperanto%' AND text ILIKE '%1887%'",
            'what_name',
            'Esperanto',
        ),
        (
            'Kiu verkis la vortaron "Altdeutsches Wörterbuch"?',
            'Oskar Schade',
            "text ILIKE '%Oskar Schade%' AND text ILIKE '%Altdeutsches Wörterbuch%'",
            'who_agent',
            'literaturo',
        ),
        (
            'Kiu fondis la landon nomitan "Parolando"?',
            'Samuel Longhorn CLEMENS',
            "text ILIKE '%Samuel Longhorn CLEMENS%' AND text ILIKE '%Parolando%'",
            'who_agent',
            'literaturo',
        ),
    ]
    out: list[dict] = []
    for i, (question, short_answer, where, qtype, topic) in enumerate(seeds, 1):
        row = conn.execute(
            f"SELECT sid, text FROM sentences WHERE {where} ORDER BY sid LIMIT 1"
        ).fetchone()
        if not row:
            continue
        sid, text = row
        item = mk_row(
            item_id=f'review_manual_{i:03d}',
            question=question,
            short_answer=short_answer,
            source_sid=sid,
            source_sentence=text,
            question_type=qtype,
            topic=topic,
            source_pattern='manual_seed_query',
        )
        if item:
            out.append(item)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--duckdb-path', default='data/indexes/duckdb_store.db')
    ap.add_argument('--output', default='data/test_sets/gold_trivia_review_queue_v1.jsonl')
    ap.add_argument('--per-pattern', type=int, default=20)
    args = ap.parse_args()

    conn = duckdb.connect(args.duckdb_path, read_only=True)
    rows = rows_from_manual_evidence(conn)
    rows.extend(rows_from_patterns(conn, args.per_pattern))
    conn.close()

    seen_questions: set[str] = set()
    deduped = []
    for row in rows:
        q = row['question']
        if q in seen_questions:
            continue
        seen_questions.add(q)
        row['id'] = f"gold_review_{len(deduped) + 1:03d}"
        deduped.append(row)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for row in deduped:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    by_type: dict[str, int] = {}
    for row in deduped:
        by_type[row['question_type']] = by_type.get(row['question_type'], 0) + 1
    print(f'wrote {len(deduped)} review candidates to {out}')
    for k, v in sorted(by_type.items()):
        print(f'  {k}: {v}')
    print('retrieval_gated: False for all rows')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
