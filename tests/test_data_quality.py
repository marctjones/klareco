#!/usr/bin/env python3
"""
Data Quality Tests for Esperanto Training Data.

These tests validate that our training data is clean and contains only
valid Esperanto roots. Run before training to catch data quality issues.

Usage:
    python -m pytest tests/test_data_quality.py -v
"""

import json
import pytest
from pathlib import Path


# Valid Esperanto letters (lowercase only)
VALID_EO_CHARS = set('abcdefghijklmnoprstuvzĉĝĥĵŝŭ')

# Characters that should NEVER appear in Esperanto roots
INVALID_CHARS = set('wxyq0123456789')

# Known affixes (for validation, not training)
KNOWN_PREFIXES = {'bo', 'dis', 'ek', 'eks', 'ge', 'mal', 'mis', 'pra', 're'}
KNOWN_SUFFIXES = {'aĉ', 'ad', 'aĵ', 'an', 'ar', 'ebl', 'ec', 'eg', 'ej', 'em', 
                  'end', 'er', 'estr', 'et', 'id', 'ig', 'iĝ', 'il', 'in', 
                  'ind', 'ing', 'ism', 'ist', 'obl', 'on', 'op', 'uj', 'ul', 'um'}


class TestCleanRootsVocabulary:
    """Tests for the clean_roots.json vocabulary file."""
    
    @pytest.fixture
    def clean_roots(self):
        """Load clean roots vocabulary."""
        path = Path('data/vocabularies/clean_roots.json')
        if not path.exists():
            pytest.skip("clean_roots.json not found - run clean_revo_vocabulary.py first")
        with open(path) as f:
            return json.load(f)
    
    def test_file_exists(self):
        """Verify clean_roots.json exists."""
        path = Path('data/vocabularies/clean_roots.json')
        assert path.exists(), "clean_roots.json not found"
    
    def test_has_metadata(self, clean_roots):
        """Verify metadata is present."""
        assert 'metadata' in clean_roots
        assert 'total_roots' in clean_roots['metadata']
        assert 'roots' in clean_roots
    
    def test_minimum_vocabulary_size(self, clean_roots):
        """Vocabulary should have at least 10,000 roots."""
        assert clean_roots['metadata']['total_roots'] >= 10000
        assert len(clean_roots['roots']) >= 10000
    
    def test_no_invalid_characters(self, clean_roots):
        """No root should contain w, x, y, q, or digits."""
        invalid_roots = []
        for root in clean_roots['roots']:
            if any(c in INVALID_CHARS for c in root.lower()):
                invalid_roots.append(root)
        
        assert len(invalid_roots) == 0, \
            f"Found {len(invalid_roots)} roots with invalid chars: {invalid_roots[:10]}"
    
    def test_only_esperanto_letters(self, clean_roots):
        """All roots should use only valid Esperanto letters."""
        invalid_roots = []
        for root in clean_roots['roots']:
            root_lower = root.lower()
            if not all(c in VALID_EO_CHARS for c in root_lower):
                bad_chars = [c for c in root_lower if c not in VALID_EO_CHARS]
                invalid_roots.append((root, bad_chars))
        
        assert len(invalid_roots) == 0, \
            f"Found {len(invalid_roots)} roots with non-Esperanto chars: {invalid_roots[:10]}"
    
    def test_minimum_length(self, clean_roots):
        """All roots should be at least 2 characters."""
        short_roots = [r for r in clean_roots['roots'] if len(r) < 2]
        assert len(short_roots) == 0, f"Found short roots: {short_roots}"
    
    def test_no_x_notation(self, clean_roots):
        """No root should use x-notation (cx, gx, etc)."""
        x_patterns = ['cx', 'gx', 'hx', 'jx', 'sx', 'ux']
        x_roots = []
        for root in clean_roots['roots']:
            if any(p in root.lower() for p in x_patterns):
                x_roots.append(root)
        
        assert len(x_roots) == 0, f"Found x-notation roots: {x_roots[:10]}"
    
    def test_fundamento_roots_present(self, clean_roots):
        """Core Fundamento roots should be present."""
        essential_roots = [
            'hom', 'dom', 'am', 'patr', 'fil', 'frat', 'hund', 'kat',
            'lern', 'leg', 'skrib', 'manĝ', 'trink', 'bon', 'bel',
            'grand', 'alt', 'nov', 'jun', 'san', 'fort', 'rapid',
            'vid', 'aŭd', 'parol', 'ĉambr', 'tablo', 'seĝ', 'liter',
        ]
        missing = [r for r in essential_roots if r not in clean_roots['roots']]
        assert len(missing) <= 3, f"Missing essential roots: {missing}"
    
    def test_tier_assignment(self, clean_roots):
        """All roots should have tier assignment."""
        missing_tier = []
        for root, info in clean_roots['roots'].items():
            if 'tier' not in info:
                missing_tier.append(root)
        
        assert len(missing_tier) == 0, f"Roots missing tier: {missing_tier[:10]}"
    
    def test_tier_distribution(self, clean_roots):
        """Tier distribution should be reasonable."""
        tiers = clean_roots['metadata'].get('tiers', {})
        
        # Tier 1 (Fundamento) should be 1500-2500
        assert 1500 <= tiers.get('1', tiers.get(1, 0)) <= 2500, \
            f"Tier 1 count unexpected: {tiers.get('1', tiers.get(1, 0))}"


class TestNoJunkInVocabulary:
    """Tests to ensure no junk data leaked into vocabulary."""
    
    @pytest.fixture
    def clean_roots(self):
        """Load clean roots vocabulary."""
        path = Path('data/vocabularies/clean_roots.json')
        if not path.exists():
            pytest.skip("clean_roots.json not found")
        with open(path) as f:
            return json.load(f)
    
    def test_no_ocr_artifacts(self, clean_roots):
        """No OCR-like patterns should be present."""
        ocr_patterns = ['tb', 'bii', 'tbi', 'iib', 'btb', 'lst', 'nfl']
        suspicious = []
        for root in clean_roots['roots']:
            # Allow these only if they're in known affixes context
            if any(p in root for p in ocr_patterns):
                # Some legitimate words might have these patterns
                # Only flag if they look really suspicious
                if len(root) > 10 or root.count('i') > 3:
                    suspicious.append(root)
        
        assert len(suspicious) == 0, f"Possible OCR artifacts: {suspicious[:10]}"
    
    def test_no_foreign_words(self, clean_roots):
        """No obvious foreign words should be present."""
        # Check for patterns common in German, French, English but not Esperanto
        foreign_patterns = [
            'sch',   # German
            'tion',  # English/French (should be -cio in Esperanto)
            'heit',  # German
            'keit',  # German
            'ness',  # English
            'ment',  # French/English (should be -mento in Esperanto)
        ]
        foreign = []
        for root in clean_roots['roots']:
            if any(p in root.lower() for p in foreign_patterns):
                foreign.append(root)
        
        # Allow a few - some loan words exist
        assert len(foreign) <= 50, \
            f"Too many foreign-looking words: {len(foreign)}, sample: {foreign[:20]}"
    
    def test_no_numbers(self, clean_roots):
        """No digits should appear in any root."""
        with_digits = [r for r in clean_roots['roots'] if any(c.isdigit() for c in r)]
        assert len(with_digits) == 0, f"Roots with digits: {with_digits}"
    
    def test_no_uppercase(self, clean_roots):
        """All roots should be lowercase."""
        with_upper = [r for r in clean_roots['roots'] if any(c.isupper() for c in r)]
        assert len(with_upper) == 0, f"Roots with uppercase: {with_upper[:10]}"


class TestTrainingDataConsistency:
    """Tests for consistency between training data files."""
    
    def test_revo_definitions_exist(self):
        """ReVo definitions file should exist."""
        path = Path('data/revo/revo_definitions_with_roots.json')
        assert path.exists(), "ReVo definitions not found"
    
    def test_fundamento_roots_exist(self):
        """Fundamento roots file should exist."""
        path = Path('data/vocabularies/fundamento_roots.json')
        assert path.exists(), "Fundamento roots not found"
    
    def test_clean_roots_matches_metadata(self):
        """Root count should match metadata."""
        path = Path('data/vocabularies/clean_roots.json')
        if not path.exists():
            pytest.skip("clean_roots.json not found")
        
        with open(path) as f:
            data = json.load(f)
        
        assert data['metadata']['total_roots'] == len(data['roots'])


# Run if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
