#!/usr/bin/env python3
"""
Extract proper-noun categories from Esperanto Wikipedia article-category links.

VERSION: v2.1
COMPATIBLE WITH: v2.1 parser
DEPENDENCIES: stdlib (xml.etree, bz2, re, json)
STAGE: Data / dictionary build

Description:
    Streams the eo Wikipedia XML dump and extracts (title → categories)
    mappings, then classifies each article into one of:

        person, place, organization, work, other

    by pattern-matching on Esperanto category names. The classification
    powers the `propranoma_kategorio` field used by retrieval to filter
    WHO/WHERE answers.

    For each article we register:
      - the FULL title (e.g. "Albert Einstein") as one entry
      - each capitalized non-function-word token of the title as a
        separate entry (so single-word lookups find it: "Einstein")

    Multi-word titles inherit the article's classification. When two
    title tokens map to different categories from different articles,
    the highest-priority category wins (person > place > organization
    > work > other).

Pipeline Position:
    eo_wikipedia.xml.bz2
        → [THIS SCRIPT]
        → wikipedia_categories.json (title → {category, freq, raw_cats})
        → merge into proper_nouns_dynamic_v3.json

Usage:
    # Quick test on N articles (validate patterns):
    python scripts/extract_wikipedia_categories.py --max-articles 5000

    # Full extract:
    python scripts/extract_wikipedia_categories.py

    # Custom output path:
    python scripts/extract_wikipedia_categories.py --output data/foo.json

Inputs:
    data/raw/eo/wikipedia/eo_wikipedia.xml.bz2

Outputs:
    data/wikipedia_categories.json — JSON map:
        {
          "Albert Einstein": {
            "category":      "person",
            "raw_categories": ["Naskiĝintaj en 1879", "Fizikistoj", ...],
            "is_redirect":   false,
            "is_multi_word": true,
          },
          "Einstein": {
            "category":      "person",
            "from_titles":   ["Albert Einstein"],
            "raw_categories": [...]   # union from all source articles
          },
          ...
        }
    logs/wikipedia_categories_extraction.log

Quality Checks:
    - Skips namespace ≠ 0 (only mainspace articles)
    - Skips #REDIRECT pages (no useful category info)
    - Skips disambiguation pages ([[Kategorio:Apartigaj paĝoj]])
    - Resumable via --max-articles
    - Periodic progress reporting (every 5000 articles)

Last Updated: 2026-05-07
Author: Claude Code (with Marc Jones)
"""
from __future__ import annotations

import argparse
import bz2
import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator
import xml.etree.ElementTree as ET


# ============================================================================
# Esperanto category-name → top-level classification
# ============================================================================
# Priority order: person > place > organization > work > other.
# Each list entry is a substring (lowercase) — case-insensitive match
# against the category name (after stripping the "Kategorio:" prefix).
# Match the FIRST list that has a hit; if no hits anywhere, "other".

#
# Pattern types:
#   - tuple of (pattern,) is a "phrase" — must appear as substring (after
#     simple whitespace normalization). Use for multi-word disambiguation:
#     "mortintaj en" matches death-year categories but NOT
#     "Mortintaj ĝermanaj lingvoj".
#   - bare string is a "token" — must appear as a complete whitespace-
#     separated token. Use for single-word categories. Matching is
#     lowercased on both sides. The token must be EXACTLY equal — this
#     prevents 'reĝoj' (kings) from matching inside 'preĝoj' (prayers).
#
_PERSON_PATTERNS = (
    # Birth / death — REQUIRE year-context phrase to avoid "Mortintaj
    # ĝermanaj lingvoj" (dead Germanic languages) etc.
    ('naskiĝintaj en',), ('mortintaj en',),
    ('naskiĝintaj la',), ('mortintaj la',),
    ('naskiĝintoj en',), ('mortintoj en',),
    ('naskiĝintoj la',), ('mortintoj la',),
    # Esperanto community (suffix-flex; matches esperantistoj/esperantistinoj)
    ('esperantist',),
    # General person markers (token-exact)
    'personoj', 'personecoj',
    'roluloj',
    'viroj', 'virinoj',
    'reĝoj', 'reĝinoj', 'imperiestroj',
    'princoj', 'princinoj',
    'sanktuloj', 'sanktulinoj',
    'martiroj',
    # Profession suffixes — token-exact matching.
    'verkistoj', 'poetoj', 'romanverkistoj', 'dramaturgoj',
    'aktoroj', 'aktorinoj',
    'kantistoj', 'muzikistoj', 'komponistoj',
    'pianistoj', 'gitaristoj', 'violonistoj',
    'sciencistoj', 'fizikistoj', 'kemiistoj', 'biologoj',
    'astronomoj', 'matematikistoj',
    'filozofoj', 'teologoj', 'historiistoj', 'geografistoj',
    'lingvistoj', 'arkeologoj',
    'politikistoj', 'prezidentoj', 'ĉefministroj', 'ministroj',
    'diplomatoj',
    'juristoj', 'advokatoj', 'juĝistoj',
    'kuracistoj', 'inĝenieroj', 'arkitektoj',
    'pentristoj', 'skulptistoj', 'fotografistoj',
    'reĝisoroj', 'filmreĝisoroj',
    'sportistoj', 'futbalistoj', 'basketbalistoj', 'tenisistoj',
    'olimpikuloj', 'olimpianoj',
    'atletoj', 'naĝistoj', 'ciklistoj',
    'soldatoj', 'generaloj', 'oficiroj',
    'jurnalistoj', 'eldonistoj',
    'instruistoj', 'profesoroj',
    'episkopoj', 'pastroj', 'rabenoj', 'imamoj',
    'tradukistoj', 'esploristoj',
    'apostoloj',
    'esperantanoj',
    'profetoj',
)

_PLACE_PATTERNS = (
    'urboj', 'urbetoj', 'urbocentroj',
    'landoj', 'ŝtatoj',
    'insuloj', 'arkipelagoj',
    'riveroj', 'lagoj', 'maroj', 'oceanoj',
    'montoj', 'montaroj', 'valoj',
    'kontinentoj',
    'regionoj', 'provincoj', 'departementoj', 'distriktoj',
    'gubernioj', 'kantonoj', 'komunumoj',
    'vilaĝoj', 'kvartaloj',
    ('lokoj en',),
    ('geografio de',), ('geografio en',),
    'flughavenoj', 'fervojstacioj', 'havenoj',
    'parkoj', 'stratoj', 'placoj',
    'pontoj', 'tuneloj',
    'kateroj', 'desertoj', 'arbaroj',
)

_ORGANIZATION_PATTERNS = (
    'organizaĵoj', 'organizoj',
    'asocioj', 'institutoj',
    'akademioj', 'universitatoj', 'lernejoj', 'kolegioj', 'fakultatoj',
    'entreprenoj', 'kompanioj', 'firmaoj', 'korporacioj',
    'partioj',
    'sindikatoj',
    'fondaĵoj',
    'societoj',
    'kluboj',
    'movadoj',
    'federacioj', 'konfederacioj',
    'religioj',
    'preĝejoj',
    'monaĥejoj',
    'muzeoj', 'bibliotekoj',
    'gazetoj',
    'radiostacioj', 'televidstacioj',
    'eldonejoj',
)

_WORK_PATTERNS = (
    'filmoj', 'romanoj', 'libroj', 'verkoj',
    'kantoj', 'albumoj', 'operoj', 'simfonioj',
    'dokumentaloj',
    'tv-serioj', 'televidserio', 'televidserioj',
    'animeoj', 'mangaoj', 'videoludoj', 'komiksoj',
    'pentraĵoj', 'skulptaĵoj', 'fotografaĵoj',
    'softvaro', 'programoj',
    'poemoj', 'fabeloj',
)

_CLASSIFICATION_GROUPS = (
    ('person',       _PERSON_PATTERNS),
    ('place',        _PLACE_PATTERNS),
    ('organization', _ORGANIZATION_PATTERNS),
    ('work',         _WORK_PATTERNS),
)

# Skip these category names entirely (administrative / disambig)
_SKIP_CATEGORY_PATTERNS = (
    'apartigaj',        # disambiguation pages
    'apartigil',        # disambiguation
    'vikipedio',        # Wikipedia meta
    'ĉefpaĝo',
    'kontrolo',
    'redaktokunlaboro',
    'ŝablon',           # template
    'helpo',
    'projekt',
)


_TOKEN_SPLIT_RE = re.compile(r'[\s\-,/.]+')


def _matches(pattern, cat_lower: str, cat_tokens_lower: list[str]) -> bool:
    """Pattern is either:
       - tuple (phrase,)            → substring match (case-insensitive)
       - bare str (token-exact)     → token must equal pattern OR pattern is a
                                      strict prefix that matches a complete
                                      morpheme boundary inside the token (i.e.
                                      'esperantist' matches 'esperantistoj').
    """
    if isinstance(pattern, tuple):
        phrase = pattern[0]
        return phrase in cat_lower
    # token form
    for tok in cat_tokens_lower:
        if tok == pattern:
            return True
        # Allow Esperanto inflectional/derivational suffix on the pattern:
        # 'esperantist' should match 'esperantistoj'/'esperantistinoj'.
        # Only allow if the pattern is at least 5 chars (avoids 'reg' matching
        # 'regiono' etc.) AND token starts with pattern.
        if len(pattern) >= 5 and tok.startswith(pattern):
            tail = tok[len(pattern):]
            # Tail must be Esperanto inflection (oj/ojn/o/on/ino/inoj/ina/inaj)
            if tail in ('o', 'oj', 'on', 'ojn', 'a', 'aj', 'an', 'ajn',
                        'in', 'ino', 'inoj', 'inon', 'inojn',
                        'ina', 'inaj', 'inan', 'inajn',
                        'ar', 'aro', 'aroj', 'aron', 'aroj',
                        'ec', 'eco', 'ecoj', 'econ', 'ecojn'):
                return True
    return False


def _classify(raw_categories: list[str]) -> str:
    """Return one of {person, place, organization, work, other}.
    First-match-wins by priority order."""
    cats_lower = [c.lower() for c in raw_categories]
    cats_tokens = [_TOKEN_SPLIT_RE.split(c) for c in cats_lower]
    for label, patterns in _CLASSIFICATION_GROUPS:
        for cat_l, cat_toks in zip(cats_lower, cats_tokens):
            for pattern in patterns:
                if _matches(pattern, cat_l, cat_toks):
                    return label
    return 'other'


# Title tokenization: split on whitespace, drop punctuation
_TOKEN_RE = re.compile(r"[A-ZĈĜĤĴŜŬa-zĉĝĥĵŝŭ0-9'\-]+")
# Function words to skip when extracting individual tokens of a title
# (these appear as connectors in multi-word foreign titles)
_FUNCTION_WORDS = {
    'de', 'da', 'en', 'al', 'kun', 'sen', 'por', 'pri', 'pro',
    'ĉe', 'sur', 'sub', 'super', 'inter', 'tra', 'apud',
    'la', "l'", 'aŭ', 'kaj', 'sed',
    'in', 'of', 'the', 'on', 'at', 'to', 'for', 'a', 'an', 'and', 'or',
    'le', 'les', 'des', 'du', 'di', 'del', 'dei', 'della',
    'der', 'die', 'das', 'von', 'van', 'el',
}

# Skip article titles starting with these prefixes (namespace leaks etc.)
_BAD_TITLE_PREFIXES = (
    'Vikipedio:', 'Helpo:', 'Ŝablono:', 'Kategorio:', 'Dosiero:',
    'Modulo:', 'Portalo:', 'Projekto:', 'MediaWiki:', 'Apartigilo:',
    'Listo de', 'Listo da',
)


# Extracts [[Kategorio:Foo]] or [[Kategorio:Foo|sortkey]] from wikitext
_CATEGORY_LINK_RE = re.compile(r'\[\[Kategorio:\s*([^\]\|]+?)\s*(?:\|[^\]]*)?\]\]')


def parse_dump(xml_path: Path, max_articles: int = 0,
                progress_every: int = 5000) -> Iterator[dict]:
    """Yield {title, text} dicts streaming from the bz2 XML dump."""
    t0 = time.time()
    seen = 0
    yielded = 0
    with bz2.open(xml_path, 'rt', encoding='utf-8') as f:
        ctx = ET.iterparse(f, events=('start', 'end'))
        _, root = next(ctx)
        in_page = False
        in_revision = False
        in_redirect = False
        page = {}

        for event, elem in ctx:
            tag = elem.tag.split('}', 1)[1] if '}' in elem.tag else elem.tag

            if event == 'start':
                if tag == 'page':
                    in_page = True
                    page = {}
                    in_redirect = False
                elif tag == 'revision':
                    in_revision = True
                elif tag == 'redirect' and in_page:
                    in_redirect = True

            elif event == 'end' and in_page:
                if tag == 'title':
                    page['title'] = elem.text or ''
                elif tag == 'ns':
                    try:
                        page['ns'] = int(elem.text)
                    except (ValueError, TypeError):
                        page['ns'] = -1
                elif tag == 'text' and in_revision:
                    page['text'] = elem.text or ''
                elif tag == 'revision':
                    in_revision = False
                elif tag == 'page':
                    seen += 1
                    if (page.get('ns') == 0
                            and not in_redirect
                            and 'title' in page
                            and 'text' in page):
                        yield page
                        yielded += 1
                    if progress_every and seen % progress_every == 0:
                        elapsed = time.time() - t0
                        rate = seen / max(elapsed, 0.01)
                        print(f'  …{seen:>7,} articles seen, {yielded:,} kept '
                              f'({rate:.0f}/s, {elapsed:.0f}s)', flush=True)
                    if max_articles and yielded >= max_articles:
                        return
                    page = {}
                    in_page = False
                    elem.clear()
                    root.clear()
                else:
                    elem.clear()


def is_bad_title(title: str) -> bool:
    if not title:
        return True
    for prefix in _BAD_TITLE_PREFIXES:
        if title.startswith(prefix):
            return True
    return False


def extract_categories(text: str) -> list[str]:
    if not text:
        return []
    cats = []
    for m in _CATEGORY_LINK_RE.finditer(text):
        cat = m.group(1).strip()
        if not cat:
            continue
        cl = cat.lower()
        if any(s in cl for s in _SKIP_CATEGORY_PATTERNS):
            continue
        cats.append(cat)
    return cats


def title_tokens(title: str) -> list[str]:
    """Return capitalized non-function-word tokens of a title."""
    out = []
    for tok in _TOKEN_RE.findall(title):
        if tok.lower() in _FUNCTION_WORDS:
            continue
        if not tok or len(tok) < 2:
            continue
        if not tok[0].isupper():
            continue
        out.append(tok)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', default='data/raw/eo/wikipedia/eo_wikipedia.xml.bz2')
    ap.add_argument('--output', default='data/wikipedia_categories.json')
    ap.add_argument('--max-articles', type=int, default=0,
                    help='Cap on kept articles (0 = no cap). Use for testing.')
    ap.add_argument('--log', default='logs/wikipedia_categories_extraction.log')
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)

    if not in_path.exists():
        logger.error(f'Input not found: {in_path}')
        sys.exit(1)
    logger.info(f'Reading {in_path} (max-articles={args.max_articles or "∞"})')

    # Full-title entries: classified by each article's own raw_categories
    # (coherent set per article).
    full_titles: dict[str, dict] = {}
    # Single-token entries: each token aggregated across all articles whose
    # title contains it. We vote per-article rather than unioning raw_cats —
    # otherwise one weird article (e.g., asteroid named "Beethoven") drags
    # the token's classification away from the dominant reading (composer).
    token_label_votes: dict[str, Counter] = defaultdict(Counter)
    # Surname-position evidence: token appears as the LAST capitalized non-
    # function word of a multi-word title. Esperantized personal-name format
    # is "Firstname [Middle] Surname", so surname-position is a strong signal
    # that the token IS the entity (not just a tangential mention). Weighted
    # heavily during aggregation.
    token_surname_votes: dict[str, Counter] = defaultdict(Counter)
    token_source_titles: dict[str, set[str]] = defaultdict(set)
    token_raw_cats: dict[str, list[str]] = defaultdict(list)  # for diagnostics

    label_counter = Counter()
    no_cat_count = 0
    bad_title_count = 0

    t0 = time.time()
    for page in parse_dump(in_path, max_articles=args.max_articles):
        title = page['title'].strip()
        text = page['text']
        if is_bad_title(title):
            bad_title_count += 1
            continue
        raw_cats = extract_categories(text)
        if not raw_cats:
            no_cat_count += 1
            continue

        label = _classify(raw_cats)
        label_counter[label] += 1

        full_titles[title] = {
            'category':       label,
            'raw_categories': raw_cats,
            'is_multi_word':  ' ' in title,
        }

        # Index single-token forms — vote with this article's classification.
        # Identify the LAST capitalized non-function token (surname position).
        # Only person-classified titles contribute surname-position evidence:
        # the surname-position structure is specifically a personal-name pattern
        # ("Ludwig van Beethoven"). Work titles like
        # "13-a simfonio de Ludwig van Beethoven" also place the author's
        # surname at the end, but that's the AUTHOR — the article ENTITY is
        # the composition. Treating those as surname-evidence-for-work would
        # wrongly tag Beethoven as a work.
        toks = title_tokens(title)
        surname_tok = toks[-1] if len(toks) >= 2 else None
        for tok in toks:
            token_label_votes[tok][label] += 1
            if tok == surname_tok and label == 'person':
                token_surname_votes[tok]['person'] += 1
            token_source_titles[tok].add(title)
            if len(token_raw_cats[tok]) < 30:
                token_raw_cats[tok].extend(raw_cats[:5])

    # Classify single-token forms with three signals (highest priority first):
    #
    # 1. CANONICAL OVERRIDE: if a token IS itself a full-title article
    #    (e.g. "Mozart" → exists as Wikipedia article), use that article's
    #    classification.
    #
    # 2. SURNAME-POSITION EVIDENCE: if the token appears as the LAST cap-
    #    italized token of any person-classified multi-word title (e.g.
    #    "Ludwig van Beethoven" → Beethoven is the surname → person
    #    evidence), this is a strong signal that the entity IS a person
    #    even if the token also appears in many work-classified titles
    #    (Beethoven's compositions).
    #    Weight: 1 surname vote == _SURNAME_WEIGHT regular votes.
    #
    # 3. REGULAR VOTE: most common classification across all source titles.
    #    Ties broken by specificity.
    _SPECIFICITY_RANK = {'person': 0, 'place': 1, 'organization': 2,
                          'work': 3, 'other': 9}
    _SURNAME_WEIGHT = 10
    token_entries: dict[str, dict] = {}
    for tok, votes in token_label_votes.items():
        canonical_entry = full_titles.get(tok)
        surname_v = token_surname_votes.get(tok, Counter())
        if canonical_entry is not None:
            label = canonical_entry['category']
            source = 'canonical'
        else:
            # Combine regular + surname votes (surname weighted ×_SURNAME_WEIGHT).
            combined = Counter(votes)
            for cat, n in surname_v.items():
                combined[cat] += n * _SURNAME_WEIGHT
            sorted_votes = sorted(combined.items(),
                                   key=lambda kv: (-kv[1],
                                                    _SPECIFICITY_RANK.get(kv[0], 99)))
            label = sorted_votes[0][0]
            source = 'surname-weighted-vote' if surname_v else 'vote'
        token_entries[tok] = {
            'category':       label,
            'classification_source': source,
            'votes':          dict(votes),
            'surname_votes':  dict(surname_v) if surname_v else {},
            'from_titles':    sorted(token_source_titles[tok])[:30],
            'raw_categories': list(dict.fromkeys(token_raw_cats[tok]))[:30],
        }

    # Combine: full_titles + token_entries. Token entries are added only
    # where they don't conflict with a full-title entry of the same key.
    out: dict[str, dict] = {}
    out.update(full_titles)
    for tok, entry in token_entries.items():
        if tok in out:
            # Full-title entry exists for this exact key — keep it, but
            # augment with token-level info if categories diverge.
            existing = out[tok]
            if existing.get('category') == 'other' and entry['category'] != 'other':
                # Single-token classification is more specific — upgrade.
                existing['category'] = entry['category']
                existing['raw_categories'] = sorted(set(
                    existing.get('raw_categories', []) + entry['raw_categories']
                ))[:80]
        else:
            out[tok] = entry

    elapsed = time.time() - t0
    logger.info(f'Done in {elapsed:.1f}s. Kept {len(full_titles):,} full titles, '
                f'{len(token_entries):,} token entries, {len(out):,} unique keys.')
    logger.info(f'Bad titles: {bad_title_count:,}; articles with no category: {no_cat_count:,}')
    logger.info('Category distribution (full titles):')
    for label, _ in _CLASSIFICATION_GROUPS:
        n = label_counter[label]
        logger.info(f'  {label:14s} {n:>8,}')
    logger.info(f'  {"other":14s} {label_counter["other"]:>8,}')

    logger.info(f'Writing {out_path}')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    logger.info(f'Wrote {len(out):,} entries.')


if __name__ == '__main__':
    main()
