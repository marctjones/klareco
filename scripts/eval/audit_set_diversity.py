#!/usr/bin/env python3
"""
Aggregate diversity audit for Q&A test sets (R15 enforcer).

VERSION: v2.x
COMPATIBLE WITH: test-set JSONL with at least {question, expected_answer,
                 question_type}; uses optional source_sentence_text,
                 anchor_pv, category when present
DEPENDENCIES: klareco.parser (for verb-radiko extraction)
STAGE: Evaluation

Description:
    R1-R14 are per-pair checks. R15 is a SET-LEVEL check: does the
    distribution across question types, anchor roles, topics, and
    template signatures hit the targets? A set of 100 mechanically
    valid questions where 90 are person-naskiĝis-place isn't a
    capability set; it's a stress test for one shape.

    What this script computes per pair:

      - question_type    KIU / KIO / KIE / KIAM / KIAL / KIEL / KIOM / ...
      - anchor_role      verko / persono / loko / evento / organizaĵo /
                         tempo / aliaĵo
      - topic            geography / history / science / arts /
                         esperanto_culture / sports / technology /
                         religion / politics / language / unclassified
      - template_id      question-shape signature (first-word + verb +
                         has-quoted-work flag), as a proxy for the
                         template that produced the pair

    What it reports:

      - distributions for each of the four dimensions
      - per-target-bucket: count, target %, status (under/within/over)
      - anchor-uniqueness check: any anchor used > 5 times → flag
      - template-uniqueness check: any signature > 25% of set → flag
      - unclassified pairs: dumped for human topic labelling

Topic inference (deterministic heuristic):
    The classifier composes signals from multiple sources because we
    don't have Wikipedia categories or Wikidata QIDs offline. Order
    of precedence:

      1. Pair has 'category' field (from OpenTriviaDB / build_trivia_bank
         schema) — use it (mapped to our 10-bucket taxonomy).
      2. Anchor matches Esperanto-cultural-entity list → 'esperanto_culture'
      3. Anchor is a quoted work AND verb is verkis/kreis/kompon/kantis/
         eldonis/publikis → 'arts'
      4. Verb radiko in {invent, eltrov, malkov} → 'science'
      5. Anchor is a year, OR verb is fondiĝ/fond + year answer → 'history'
      6. Anchor is a place-typed entity (place-suffix OR EntecaTipo 'loko')
         + answer is geographic → 'geography'
      7. Verb is naskiĝ/mortis (biographical) + person anchor →
         'biography' (heuristically 'history')
      8. Sports keywords in question (gajnis, ludas, futbal, etc.) →
         'sports'
      9. Tech/business org keywords → 'technology'
     10. Otherwise → 'unclassified' (flagged for human review)

    The classifier is conservative — it prefers 'unclassified' over a
    wrong bucket. We flag false negatives (unclassified pairs that have
    a clear topic) rather than risk false positives (pairs misclassified
    as one topic when they're really another).

Pipeline Position:
    <test_set>.jsonl → [THIS SCRIPT] → diversity report + flagged pairs
                                     → (re-balance / drop / add)

Usage:
    python scripts/eval/audit_set_diversity.py \\
        --in data/test_sets/capability_100.jsonl \\
        --target-size 100

    # Override targets for a smaller dev set
    python scripts/eval/audit_set_diversity.py \\
        --in data/staging/harvest_candidates.jsonl \\
        --target-size 80 --strict

Inputs:
    --in           one or more JSONL test sets
    --target-size  the set's intended final size (drives % targets)
    --strict       exit 1 if any bucket exceeds 2× target or
                   falls below 0.5× target
    --output       write per-pair classifications to JSONL

Outputs:
    Human-readable diversity report to stdout.
    Optional per-pair JSONL with computed fields.

Quality Checks:
    R15 (question-type, topic, anchor-role, template-id distributions).
    See docs/QA_TEST_SET_QUALITY_STANDARD.md.

Last Updated: 2026-05-21
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from klareco.parser import parse  # noqa: E402


# ---------------------------------------------------------------------------
# Taxonomy: targets per R15 (% of set)
# ---------------------------------------------------------------------------

TARGETS_QUESTION_TYPE = {
    'KIO':  0.25, 'KIU':  0.25,
    'KIE':  0.15, 'KIAM': 0.15,
    'KIAL': 0.05, 'KIEL': 0.05, 'KIOM': 0.05,
    'OTHER': 0.05,
}

TARGETS_TOPIC = {
    # Trivial Pursuit-style taxonomy with Esperanto-specific addition.
    # Targets are % of the final set.
    'geography':           0.15,
    'history':             0.15,
    'science_nature':      0.15,
    'arts_literature':     0.15,
    'entertainment':       0.10,   # film, TV, music, video games
    'sports_leisure':      0.10,
    'esperanto_culture':   0.10,
    'technology_business': 0.05,
    'politics_religion':   0.05,   # absorbs religion, politics
    'other':               0.00,   # language, food, philosophy — no target
    'unclassified':        0.00,   # target zero; any > 0 is flagged
}

# Allowed topics — LLM responses outside this set are coerced to 'other'
_ALLOWED_TOPICS = set(TARGETS_TOPIC.keys())

TARGETS_ANCHOR_ROLE = {
    'persono':    0.50,   # ceiling per R10
    'loko':       0.20,
    'verko':      0.15,
    'evento':     0.05,
    'organizaĵo': 0.05,
    'tempo':      0.05,
    'aliaĵo':     0.00,    # catch-all; >0 is OK but not a target
}

ANCHOR_UNIQUENESS_MAX = 5      # no single anchor > 5 pairs in a 100-pair set
TEMPLATE_UNIQUENESS_MAX_PCT = 0.25  # no template > 25% of set


# ---------------------------------------------------------------------------
# Esperanto cultural anchors — small allowlist
# ---------------------------------------------------------------------------

_ESPERANTO_CULTURAL = {
    # Core movement entities
    'Esperanto', 'Esperantujo', 'UEA', 'TEJO', 'SAT', 'ILEI',
    'Vikipedio', 'Pasporta', 'Servo',
    # People
    'Zamenhof', 'Ludoviko', 'Lazaro', 'Lazaro Zamenhof',
    'Hodler', 'Privat',
    # Documents / publications
    'Fundamento', 'Krestomatio', 'Ekzercaro', 'Adresaro',
    'Unua Libro', 'Esperantisto', 'Universala Kongreso',
    'Kongreso', 'Bulonjo',
    # Concepts
    'Esperantismo', 'Interna', 'Ideo',
}


# ---------------------------------------------------------------------------
# Wikipedia anchor lookup — used as grounding context for the LLM
# ---------------------------------------------------------------------------
#
# We DO NOT try to classify topics from Wikipedia category strings via
# substring matching — that was overengineered. Wikipedia categories are
# semantic; LLMs handle semantics. We just hand the LLM the raw_categories
# strings so it can ground its classification in real data instead of
# guessing from the question alone.

_WIKI_CATEGORIES: dict | None = None
_WIKI_CATEGORIES_FOLD: dict | None = None


def _load_wiki_categories(
    path: Path = Path('data/wikipedia_categories.json'),
) -> tuple[dict, dict]:
    global _WIKI_CATEGORIES, _WIKI_CATEGORIES_FOLD
    if _WIKI_CATEGORIES is not None and _WIKI_CATEGORIES_FOLD is not None:
        return _WIKI_CATEGORIES, _WIKI_CATEGORIES_FOLD

    if not path.exists():
        _WIKI_CATEGORIES, _WIKI_CATEGORIES_FOLD = {}, {}
        return _WIKI_CATEGORIES, _WIKI_CATEGORIES_FOLD

    import unicodedata
    def _fold(s: str) -> str:
        return ''.join(
            c for c in unicodedata.normalize('NFKD', s or '')
            if not unicodedata.combining(c)
        ).lower()

    with open(path) as f:
        _WIKI_CATEGORIES = json.load(f)
    _WIKI_CATEGORIES_FOLD = {_fold(k): k for k in _WIKI_CATEGORIES.keys()}
    return _WIKI_CATEGORIES, _WIKI_CATEGORIES_FOLD


def _wikipedia_context(anchor: str | None) -> list[str]:
    """Return raw_categories for the anchor (for LLM grounding). Empty
    list if anchor isn't in Wikipedia."""
    if not anchor:
        return []
    wiki, fold = _load_wiki_categories()
    if not wiki:
        return []

    import unicodedata
    def _fold(s: str) -> str:
        return ''.join(
            c for c in unicodedata.normalize('NFKD', s or '')
            if not unicodedata.combining(c)
        ).lower()

    for c in (anchor, anchor.strip('«»"„'),
              (anchor.split()[-1] if ' ' in anchor else None)):
        if not c:
            continue
        if c in wiki:
            return list(wiki[c].get('raw_categories') or [])
        folded = _fold(c)
        if folded in fold:
            return list(wiki[fold[folded]].get('raw_categories') or [])
    return []


# ---------------------------------------------------------------------------
# Signal patterns
# ---------------------------------------------------------------------------

_QUOTED_WORK_RE = re.compile(r'[«"„]\s*([^«»"]+?)\s*[»"]')
_YEAR_RE = re.compile(r'\b(1[0-9]{3}|20[0-9]{2})\b')
_PROPER_TOKEN_RE = re.compile(
    r'^[A-ZÀ-ÞĈĜĤĴŜŬ][\wÀ-ſĉĝĥĵŝŭĈĜĤĴŜŬ.-]{2,}$'
)
_COMMON_CAPS = {
    'Kaj', 'Sed', 'Aŭ', 'Do', 'Tamen', 'Anstataŭ', 'Krom', 'Pri', 'Pro',
    'Ankaŭ', 'Antaŭ', 'Post', 'Tial', 'Ke', 'Estas', 'Estis',
}

_PLACE_SUFFIXES = ('io', 'lando', 'urbo', 'sko', 'ujo', 'ujo', 'lando')
_PLACE_NAMES = {
    # Common country/region names that won't have suffix pattern
    'Brazilo', 'Hispanio', 'Francio', 'Germanio', 'Italio', 'Polollando',
    'Britio', 'Rusio', 'Japanio', 'Ĉinio', 'Hindio', 'Eŭropo', 'Azio',
    'Afriko', 'Ameriko', 'Tero', 'Luno',
}

_ARTS_VERBS = {'verk', 'kre', 'kompon', 'kant', 'eldon', 'publik', 'pentr',
               'regi', 'reĝi', 'reĝisor', 'aktor', 'sceniz', 'gajn'}
_SCIENCE_VERBS = {'invent', 'eltrov', 'malkov', 'esplor', 'studis'}
_HISTORY_VERBS = {'fond', 'establ', 'mortis', 'mort', 'naskiĝ', 'nask',
                  'okaz', 'reĝ', 'reg', 'venk'}
_SPORTS_KEYWORDS = ('gajn', 'futbol', 'sport', 'olimp', 'medal', 'ludo',
                    'turnir', 'ludis', 'pilkad', 'ĉampion')
_TECH_KEYWORDS = ('disvolv', 'inĝenier', 'komputil', 'softvar', 'reto',
                  'industri', 'kompani')

_CATEGORY_TO_TOPIC = {
    # OpenTriviaDB-ish categories
    'geography':           'geography',
    'history':             'history',
    'science_nature':      'science',
    'science_computers':   'technology',
    'science_mathematics': 'science',
    'science_gadgets':     'technology',
    'entertainment_film':           'arts',
    'entertainment_music':          'arts',
    'entertainment_television':     'arts',
    'entertainment_books':          'arts',
    'entertainment_video_games':    'technology',
    'entertainment_japanese_anime': 'arts',
    'entertainment_board_games':    'sports',
    'entertainment_comics':         'arts',
    'sports':                       'sports',
    'animals':                      'science',
    'mythology':                    'other',
    'celebrities':                  'arts',
    'general_knowledge':            'other',
    'video_games':                  'technology',
    'art':                          'arts',
    'politics':                     'other',
    'vehicles':                     'technology',
}


# ---------------------------------------------------------------------------
# Per-pair classifiers
# ---------------------------------------------------------------------------

def _extract_anchor(question: str) -> tuple[str | None, str]:
    """Return (anchor_text, anchor_role) for the question.

    Roles: verko (quoted work) / persono / loko / tempo / aliaĵo.
    """
    qw_match = _QUOTED_WORK_RE.search(question)
    if qw_match:
        return qw_match.group(0), 'verko'

    toks = question.rstrip('?').split()
    # Skip leading question word and any fronted PP
    skip = {'En', 'De', 'Al', 'Por', 'Pri', 'Pro', 'Kun', 'Sur', 'Sub',
            'Ĉe', 'Tra', 'Antaŭ', 'Post', 'Kontraŭ'}
    start = 1
    while start < len(toks) and toks[start - 1] in skip:
        start += 1
    span = []
    for t in toks[start:]:
        tt = t.strip(',.;:?')
        if _PROPER_TOKEN_RE.match(tt) and tt not in _COMMON_CAPS:
            span.append(tt)
            continue
        if span:
            break
    if not span:
        # Year-only anchor (for some KIAM questions)
        m = _YEAR_RE.search(question)
        if m:
            return m.group(0), 'tempo'
        return None, 'aliaĵo'

    anchor = ' '.join(span)
    # Multi-token = likely person (multi-word name)
    if len(span) >= 2:
        return anchor, 'persono'
    # Single-token: try place-suffix and known-place set
    tok = span[0]
    if tok in _PLACE_NAMES:
        return anchor, 'loko'
    tok_lower = tok.lower()
    if any(tok_lower.endswith(s) for s in _PLACE_SUFFIXES):
        return anchor, 'loko'
    return anchor, 'persono'   # default for single-token proper noun


def _verb_radiko(question: str) -> str | None:
    """Pull the question's verb radiko via parser."""
    try:
        ast = parse(question)
    except Exception:
        return None
    v = ast.get('verbo') if isinstance(ast, dict) else None
    if isinstance(v, dict):
        return v.get('radiko')
    return None


def _question_type(pair: dict) -> str:
    qt = (pair.get('question_type') or '').upper()
    if qt in {'KIU', 'WHO'}:
        return 'KIU'
    if qt in TARGETS_QUESTION_TYPE:
        return qt
    # Fall back to first word
    q = pair.get('question') or ''
    first = q.split()[0] if q.split() else ''
    if first in ('Kio', 'Kion'):
        return 'KIO'
    if first in ('Kiu', 'Kiun'):
        return 'KIU'
    if first in ('Kie', 'Kien'):
        return 'KIE'
    if first == 'Kiam':
        return 'KIAM'
    if first == 'Kial':
        return 'KIAL'
    if first == 'Kiel':
        return 'KIEL'
    if first == 'Kiom':
        return 'KIOM'
    return 'OTHER'


def _template_id(question: str, verb_radiko: str | None,
                 anchor_role: str) -> str:
    """Question-shape signature: <first_word>_<verb_radiko>_<anchor_role>.
    Used to detect template overuse (no single template > 25% of set).
    """
    first = (question.split() or ['?'])[0]
    vr = verb_radiko or 'NOVERB'
    return f'{first}_{vr}_{anchor_role}'


# ---------------------------------------------------------------------------
# Topic classification: cache + handoff to Claude (this conversation)
# ---------------------------------------------------------------------------
#
# The classifier itself isn't a function in this script — it's me,
# Claude, reading a staging file and writing back a cache file. The
# script's job is the file dance:
#
#   1. Walk pairs; for each, compute a question-hash key.
#   2. If the hash is in `topic_classification_cache.jsonl`, use it.
#   3. If not, append a row to `topics_to_classify.jsonl` containing the
#      question + answer + source-sentence excerpt + Wikipedia
#      raw_categories (as grounding context).
#   4. After this script runs, if any pairs are uncached, I read the
#      staging file in this conversation and write the cache. The next
#      script run picks them up.
#
# No anthropic SDK, no API key, no extra dependencies. The "LLM" is the
# conversation we're already in.

_TOPIC_CACHE_PATH = Path('data/staging/topic_classification_cache.jsonl')
_TOPIC_STAGING_PATH = Path('data/staging/topics_to_classify.jsonl')

_TOPIC_TAXONOMY = {
    'geography':           'countries, cities, rivers, mountains, landmarks',
    'history':             'historical events, dynasties, wars, dates, rulers',
    'science_nature':      'science, animals, plants, physics, chemistry, biology, math, medicine',
    'arts_literature':     'novels, poetry, painters, writers, classical music, opera',
    'entertainment':       'film, TV, popular music, video games, celebrities, comics',
    'sports_leisure':      'athletes, sports, games, championships',
    'esperanto_culture':   'Zamenhof, UEA, Fundamento, Esperanto-movement entities',
    'technology_business': 'tech companies, inventions, software, industry, business',
    'politics_religion':   'political figures, government, religions, philosophical movements',
    'other':               'anything else (food, language, miscellaneous)',
}

_TOPIC_CACHE: dict[str, dict] | None = None


def _question_hash(question: str) -> str:
    import hashlib
    return hashlib.sha256(question.encode('utf-8')).hexdigest()[:16]


def _load_topic_cache() -> dict[str, dict]:
    global _TOPIC_CACHE
    if _TOPIC_CACHE is not None:
        return _TOPIC_CACHE
    _TOPIC_CACHE = {}
    if _TOPIC_CACHE_PATH.exists():
        with open(_TOPIC_CACHE_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    _TOPIC_CACHE[entry['hash']] = entry
                except Exception:
                    continue
    return _TOPIC_CACHE


def _write_staging_for_classification(rows: list[dict]) -> None:
    """Append uncached pairs to the staging file. Idempotent — duplicate
    hashes get deduped at read time."""
    if not rows:
        return
    _TOPIC_STAGING_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Dedup against what's already in the staging file (so re-running
    # doesn't pile up duplicates).
    existing_hashes: set[str] = set()
    if _TOPIC_STAGING_PATH.exists():
        with open(_TOPIC_STAGING_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing_hashes.add(json.loads(line).get('hash', ''))
                except Exception:
                    continue
    with open(_TOPIC_STAGING_PATH, 'a') as f:
        for r in rows:
            if r['hash'] not in existing_hashes:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                existing_hashes.add(r['hash'])


def _classify_topic(pair: dict, anchor: str | None, anchor_role: str,
                    verb_radiko: str | None) -> tuple[str, str]:
    """Returns (topic, reason).

    Order:
      1. Explicit `category` field on the pair (OpenTriviaDB schema)
      2. Esperanto-cultural-entity allowlist (small, instant)
      3. Topic-classification cache lookup
      4. Tiny structural fallback (so a never-classified pair gets
         *some* topic) — flagged as low-confidence
      5. Stage for LLM classification (in this conversation)

    The actual semantic classification happens in conversation, not in
    this script. The script's job is to identify what needs labelling
    and stage it.
    """
    question = pair.get('question') or pair.get('eo_question') or ''
    expected_answer = pair.get('expected_answer') or pair.get('eo_answer') or ''
    source_text = pair.get('source_sentence_text') or ''

    # 1. Data-source category
    cat = (pair.get('category') or '').lower()
    if cat in _CATEGORY_TO_TOPIC:
        legacy = _CATEGORY_TO_TOPIC[cat]
        remap = {
            'science':    'science_nature',
            'arts':       'arts_literature',
            'technology': 'technology_business',
            'sports':     'sports_leisure',
        }
        topic = remap.get(legacy, legacy)
        if topic not in _ALLOWED_TOPICS:
            topic = 'other'
        return topic, f'from data-source category {cat!r}'

    # 2. Esperanto cultural allowlist
    if anchor:
        anchor_tokens = anchor.replace('«', '').replace('»', '').split()
        if any(t in _ESPERANTO_CULTURAL for t in anchor_tokens):
            return 'esperanto_culture', f'EO-cultural allowlist ({anchor!r})'

    # 3. Cache lookup
    h = _question_hash(question)
    cache = _load_topic_cache()
    if h in cache:
        entry = cache[h]
        return entry['topic'], f'cached ({entry.get("confidence", "?")} conf)'

    # 4. Stage for LLM classification
    wcats = _wikipedia_context(anchor)
    _write_staging_for_classification([{
        'hash':            h,
        'question':        question,
        'expected_answer': expected_answer,
        'source_text':     source_text[:200],
        'anchor':          anchor,
        'wiki_cats':       wcats[:8],   # cap for prompt size
    }])

    # 5. Cheap structural placeholder so the report still has a topic
    #    for every pair. The placeholder is overwritten when the cache
    #    gets updated and the audit re-runs.
    if anchor_role == 'verko':
        return 'unclassified', 'pending LLM (staged); verko fallback'
    if anchor_role == 'loko':
        return 'unclassified', 'pending LLM (staged); loko fallback'
    if anchor_role == 'tempo':
        return 'unclassified', 'pending LLM (staged); tempo fallback'
    return 'unclassified', 'pending LLM (staged)'


def classify(pair: dict) -> dict:
    """Run all classifiers on one pair."""
    question = pair.get('question') or pair.get('eo_question') or ''
    anchor, anchor_role = _extract_anchor(question)
    verb_radiko = _verb_radiko(question)
    qt = _question_type(pair)
    topic, topic_reason = _classify_topic(
        pair, anchor, anchor_role, verb_radiko
    )
    tid = _template_id(question, verb_radiko, anchor_role)
    return {
        'id':            pair.get('id'),
        'question':      question,
        'expected_answer': pair.get('expected_answer') or pair.get('eo_answer') or '',
        'question_type': qt,
        'anchor':        anchor,
        'anchor_role':   anchor_role,
        'verb_radiko':   verb_radiko,
        'topic':         topic,
        'topic_reason':  topic_reason,
        'template_id':   tid,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _bar(n: int, width: int = 40, scale: int = 1) -> str:
    return '█' * min(width, int(n * scale))


def _report_distribution(name: str, counts: Counter,
                         targets: dict[str, float], total: int) -> int:
    """Print a distribution; return number of violations (over 2× target
    or under 0.5× target)."""
    print(f'\n=== {name} distribution ===')
    print(f'{"bucket":<22s} {"n":>4s} {"%":>6s}  {"target":>7s}  {"status":>10s}')
    print('-' * 70)
    seen = set()
    violations = 0
    sorted_buckets = sorted(
        set(targets) | set(counts),
        key=lambda k: -counts.get(k, 0),
    )
    for bucket in sorted_buckets:
        seen.add(bucket)
        n = counts.get(bucket, 0)
        pct = (n / total * 100) if total else 0
        tgt = targets.get(bucket, 0) * 100
        tgt_str = f'{tgt:.0f}%' if tgt > 0 else '—'
        # Status: only flag if bucket has a non-zero target
        status = 'within'
        if tgt > 0:
            if pct > 2 * tgt:
                status = 'OVER'
                violations += 1
            elif pct < 0.5 * tgt:
                status = 'under'
                violations += 1
        elif bucket == 'unclassified' and n > 0:
            status = 'flagged'
            violations += 1
        bar = _bar(n, width=20)
        print(f'{bucket:<22s} {n:>4d} {pct:>5.1f}%  {tgt_str:>7s}  {status:>10s}  {bar}')
    return violations


def report(classifications: list[dict], target_size: int,
           strict: bool = False) -> int:
    """Print the full diversity report. Returns total violation count."""
    total = len(classifications)
    print(f'\nDiversity audit: {total} pairs (target size {target_size})')

    qt_counts = Counter(c['question_type'] for c in classifications)
    role_counts = Counter(c['anchor_role'] for c in classifications)
    topic_counts = Counter(c['topic'] for c in classifications)

    violations = 0
    violations += _report_distribution(
        'Question type', qt_counts, TARGETS_QUESTION_TYPE, total
    )
    violations += _report_distribution(
        'Anchor role', role_counts, TARGETS_ANCHOR_ROLE, total
    )
    violations += _report_distribution(
        'Topic / knowledge area', topic_counts, TARGETS_TOPIC, total
    )

    # Anchor uniqueness
    anchor_counts = Counter(c['anchor'] for c in classifications if c['anchor'])
    overused_anchors = [(a, n) for a, n in anchor_counts.items()
                        if n > ANCHOR_UNIQUENESS_MAX]
    print(f'\n=== Anchor uniqueness (cap: ≤ {ANCHOR_UNIQUENESS_MAX} pairs per anchor) ===')
    if overused_anchors:
        print(f'  {len(overused_anchors)} overused anchor(s):')
        for a, n in sorted(overused_anchors, key=lambda x: -x[1])[:10]:
            print(f'    {a!r:<40s} {n:>3} pairs')
        violations += len(overused_anchors)
    else:
        print('  OK — no anchor used more than the cap.')

    # Template uniqueness
    tpl_counts = Counter(c['template_id'] for c in classifications)
    over_tpls = [(t, n) for t, n in tpl_counts.items()
                 if total and n / total > TEMPLATE_UNIQUENESS_MAX_PCT]
    print(f'\n=== Template uniqueness (cap: ≤ {TEMPLATE_UNIQUENESS_MAX_PCT*100:.0f}% per template) ===')
    if over_tpls:
        print(f'  {len(over_tpls)} dominant template(s):')
        for t, n in sorted(over_tpls, key=lambda x: -x[1])[:10]:
            pct = n / total * 100
            print(f'    {t:<40s} {n:>3} ({pct:>5.1f}%)')
        violations += len(over_tpls)
    else:
        print('  OK — no template dominates.')

    # Unclassified samples — for human topic-labelling
    unclassified = [c for c in classifications if c['topic'] == 'unclassified']
    if unclassified:
        print(f'\n=== Unclassified pairs ({len(unclassified)}) — need manual topic label ===')
        for c in unclassified[:15]:
            print(f'  - [{c["question_type"]:>5s}] {c["question"][:60]:<60s}')
            print(f'      anchor={c["anchor"]!r:<28s}  verb={c["verb_radiko"]!r}')
        if len(unclassified) > 15:
            print(f'  ... and {len(unclassified) - 15} more')

    # Summary
    print(f'\n=== SUMMARY ===')
    print(f'  Total pairs: {total}')
    print(f'  Violations:  {violations}')
    print(f'  Unclassified: {len(unclassified)} ({100*len(unclassified)/max(1,total):.1f}%)')
    if strict and violations > 0:
        print(f'  STRICT MODE: exiting non-zero ({violations} violations)')
        return violations
    return violations


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--in', dest='inputs', nargs='+', required=True)
    ap.add_argument('--target-size', type=int, default=100,
                    help='Intended final size of the set (drives % targets).')
    ap.add_argument('--strict', action='store_true')
    ap.add_argument('--output', default=None,
                    help='Per-pair classification JSONL path.')
    args = ap.parse_args()

    all_classifications: list[dict] = []
    for ts_path in args.inputs:
        path = Path(ts_path)
        if not path.exists():
            print(f'SKIP: {path} not found', file=sys.stderr)
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pair = json.loads(line)
                c = classify(pair)
                c['source_set'] = path.name
                all_classifications.append(c)

    violations = report(all_classifications, args.target_size,
                        strict=args.strict)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            for c in all_classifications:
                f.write(json.dumps(c, ensure_ascii=False) + '\n')
        print(f'\nPer-pair classifications written to {out}')

    if args.strict and violations > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
