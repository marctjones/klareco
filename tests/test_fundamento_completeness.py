#!/usr/bin/env python3
"""
Fundamento Completeness Tests

Validates that fundamento_roots.json accurately represents the
Fundamento Universala Vortaro (Zamenhof 1905) source text.

This is the authoritative ground-truth test: it cross-checks
the JSON against the raw source, not just against a trained model.

Two properties are tested:
  1. No junk: every root in the JSON maps to a real source line
  2. No gaps: every parseable root in the source is in the JSON
     (minus known exclusions: affixes and OCR artifacts)

Usage:
    pytest tests/test_fundamento_completeness.py -v
"""

import json
import re
import pytest
from pathlib import Path


FUNDAMENTO_JSON = Path('data/vocabularies/fundamento_roots.json')
FUNDAMENTO_SOURCE = Path('data/raw/eo/fundamento/fundamento_de_esperanto.txt')

# Pattern that extracts the root part before the first apostrophe on a line
ROOT_PATTERN = re.compile(r"^([a-zĉĝĥĵŝŭ]{2,})[''7]", re.UNICODE)

# Roots the parser sees in source but are correctly excluded from the JSON
KNOWN_EXCLUSIONS = frozenset({
    # Grammatical affixes (not content roots)
    'ad',       # -ad- (continuous action suffix)
    'aĵ',       # -aĵ- (concrete object suffix)
    'ĉj',       # -ĉj- (masculine diminutive name affix)
    'nj',       # -nj- (feminine diminutive name affix)
    # OCR artifacts / corrupted entries in the source text
    'leĝf',       # should be 'leĝ' (law) — stray 'f'
    'nasklĝ',     # should be 'naskiĝ' — 'i' dropped by OCR
    'patv',       # should be 'patr' (father) — 'r' misread as 'v'
    'yiŝ',        # should be 'viŝ' (wipe) — 'v' misread as 'y'
    'ĉarpentlst', # should be 'ĉarpentist' — OCR garbled
    'ĝr',         # 2-char fragment from compound 'preĝ' — OCR garbled
    'ionnix',     # Cyrillic OCR garbage line
    'kazywac',    # Polish OCR garbage line (backslash raw_line)
})


@pytest.fixture(scope='module')
def json_roots():
    """Load roots from fundamento_roots.json."""
    if not FUNDAMENTO_JSON.exists():
        pytest.skip("fundamento_roots.json not found")
    with open(FUNDAMENTO_JSON) as f:
        return json.load(f)


@pytest.fixture(scope='module')
def source_roots():
    """Parse all roots directly from the Fundamento source text."""
    if not FUNDAMENTO_SOURCE.exists():
        pytest.skip("Fundamento source text not found at data/raw/eo/fundamento/")
    found = set()
    with open(FUNDAMENTO_SOURCE, encoding='utf-8') as f:
        for line in f:
            m = ROOT_PATTERN.match(line.strip())
            if m:
                found.add(m.group(1))
    return found


class TestFundamentoJSON:
    """Validate fundamento_roots.json internal consistency."""

    def test_json_exists(self):
        assert FUNDAMENTO_JSON.exists(), f"Missing: {FUNDAMENTO_JSON}"

    def test_minimum_size(self, json_roots):
        """Should have at least 2400 roots (well above the 2176 pre-expansion baseline)."""
        assert len(json_roots) >= 2400, \
            f"Only {len(json_roots)} roots — expected 2400+"

    def test_no_single_char_roots(self, json_roots):
        """No root should be a single character (those are OCR artifacts)."""
        single = [r for r in json_roots if len(r) < 2]
        assert len(single) == 0, f"Single-char roots found: {single}"

    def test_no_invalid_chars(self, json_roots):
        """No root should contain non-Esperanto characters."""
        invalid_eo = set('wxyq0123456789')
        bad = [r for r in json_roots if any(c in invalid_eo for c in r.lower())]
        assert len(bad) == 0, f"Roots with non-Esperanto chars: {bad[:10]}"

    def test_all_lowercase(self, json_roots):
        """All roots should be lowercase."""
        upper = [r for r in json_roots if r != r.lower()]
        assert len(upper) == 0, f"Uppercase roots: {upper[:10]}"

    def test_all_have_description(self, json_roots):
        """Every root should have a description field."""
        missing = [r for r, v in json_roots.items()
                   if not isinstance(v, dict) or not v.get('description')]
        assert len(missing) == 0, f"Roots without description: {missing[:10]}"

    def test_all_flagged_fundamento(self, json_roots):
        """Every root should have fundamento=True."""
        not_flagged = [r for r, v in json_roots.items()
                       if not isinstance(v, dict) or not v.get('fundamento')]
        assert len(not_flagged) == 0, f"Roots missing fundamento=True: {not_flagged[:10]}"


class TestFundamentoSourceCoverage:
    """Cross-check JSON against the raw Fundamento source text."""

    def test_source_file_exists(self):
        if not FUNDAMENTO_SOURCE.exists():
            pytest.skip("Fundamento source not available")

    def test_no_junk_roots(self, json_roots, source_roots):
        """
        Every root in JSON should either appear in the source OR be a known
        root that the source extraction misses (e.g. bel, glac appear on
        continuation lines not caught by the line-start pattern).

        Fail if we find roots that are clearly OCR garbage (too short, bad chars).
        """
        not_in_source = set(json_roots.keys()) - source_roots - KNOWN_EXCLUSIONS

        # Filter to genuinely suspicious ones (not just missed by line-start regex)
        # A root is suspicious if it's 1 char or contains non-Eo characters
        invalid_eo = set('wxyq0123456789')
        suspicious = [
            r for r in not_in_source
            if len(r) < 2 or any(c in invalid_eo for c in r.lower())
        ]

        assert len(suspicious) == 0, \
            f"Suspicious roots in JSON not found in source: {suspicious}"

    def test_no_gaps(self, json_roots, source_roots):
        """
        Every root parseable from the source (minus known exclusions)
        should be present in the JSON.
        """
        expected = source_roots - KNOWN_EXCLUSIONS
        missing = expected - set(json_roots.keys())

        assert len(missing) == 0, (
            f"{len(missing)} roots found in source but missing from JSON: "
            f"{sorted(missing)}"
        )

    def test_known_exclusions_not_in_json(self, json_roots):
        """Affixes and OCR artifacts should not appear in the JSON."""
        wrongly_included = KNOWN_EXCLUSIONS & set(json_roots.keys())
        assert len(wrongly_included) == 0, \
            f"Excluded roots found in JSON: {wrongly_included}"

    def test_special_char_roots_present(self, json_roots):
        """
        Esperanto special-character roots (ĉ/ĝ/ĥ/ĵ/ŝ/ŭ) should be well
        represented — these were all missing before the 2025-04 expansion.
        """
        special = [r for r in json_roots if any(c in r for c in 'ĉĝĥĵŝŭ')]
        assert len(special) >= 200, \
            f"Only {len(special)} special-char roots — expected 200+"

    def test_core_roots_present(self, json_roots):
        """Spot-check a sample of important Fundamento roots."""
        required = [
            # Core vocabulary
            'hom', 'dom', 'am', 'patr', 'fil', 'frat',
            'lern', 'leg', 'skrib', 'manĝ', 'trink',
            'bon', 'bel', 'grand', 'alt', 'nov',
            'vid', 'aŭd', 'parol',
            # Special-char roots (the ones that were all missing pre-expansion)
            'ĉambr', 'reĝ', 'ŝip', 'feliĉ', 'ĝoj',
            'ĵet', 'ŝton', 'kaŝ', 'sufiĉ', 'kuraĝ',
            # Newly added in 2025-04
            'anĝel', 'ĉef', 'ĉirkaŭ', 'moŝt',
        ]
        missing = [r for r in required if r not in json_roots]
        assert len(missing) == 0, f"Required roots missing from JSON: {missing}"
