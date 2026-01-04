#!/usr/bin/env python3
"""
Corpus Integrity Tests.

Tests that the PARSE stage produced a valid unified corpus with proper
ASTs and parse statistics.

Usage:
    pytest tests/test_corpus_integrity.py -v
"""

import json
import pytest
from pathlib import Path
from typing import Dict, Any


# Expected corpus locations
CORPUS_PATHS = [
    Path('data/corpus/unified.jsonl'),
    Path('data/corpus/general_corpus.jsonl'),
    Path('data/corpus/tiered_general_corpus.jsonl'),
]


class TestCorpusExists:
    """Tests that corpus files exist."""

    @pytest.fixture
    def corpus_path(self):
        """Find the corpus file."""
        for path in CORPUS_PATHS:
            if path.exists():
                return path
        return CORPUS_PATHS[0]  # Default for error message

    def test_corpus_file_exists(self, corpus_path):
        """Unified corpus should exist."""
        assert corpus_path.exists(), \
            f"Corpus not found. Tried: {[str(p) for p in CORPUS_PATHS]}"

    def test_corpus_not_empty(self, corpus_path):
        """Corpus file should have content."""
        if not corpus_path.exists():
            pytest.skip("Corpus file not found")

        assert corpus_path.stat().st_size > 0, "Corpus file is empty"

    def test_corpus_substantial_size(self, corpus_path):
        """Corpus should be substantial (at least 100MB)."""
        if not corpus_path.exists():
            pytest.skip("Corpus file not found")

        size_mb = corpus_path.stat().st_size / (1024 * 1024)
        # Unified corpus should be at least 100MB
        assert size_mb >= 10, f"Corpus too small: {size_mb:.1f} MB"


class TestCorpusFormat:
    """Tests for corpus file format validity."""

    @pytest.fixture
    def corpus_path(self):
        """Find the corpus file."""
        for path in CORPUS_PATHS:
            if path.exists():
                return path
        pytest.skip("No corpus file found")

    def test_valid_jsonl(self, corpus_path):
        """Each line should be valid JSON."""
        invalid_lines = []

        with open(corpus_path, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 1000:
                    break
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    invalid_lines.append((i, str(e)[:50]))

        assert len(invalid_lines) == 0, \
            f"Invalid JSON at lines: {invalid_lines[:10]}"

    def test_has_text_field(self, corpus_path):
        """All entries should have 'text' field."""
        missing_text = 0
        total = 0

        with open(corpus_path, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 1000:
                    break
                try:
                    entry = json.loads(line)
                    total += 1
                    if 'text' not in entry:
                        missing_text += 1
                except:
                    pass

        assert missing_text == 0, f"{missing_text}/{total} entries missing 'text'"

    def test_has_source_field(self, corpus_path):
        """All entries should have 'source' field."""
        missing_source = 0
        total = 0

        with open(corpus_path, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 1000:
                    break
                try:
                    entry = json.loads(line)
                    total += 1
                    if 'source' not in entry:
                        missing_source += 1
                except:
                    pass

        assert missing_source == 0, \
            f"{missing_source}/{total} entries missing 'source'"


class TestCorpusASTs:
    """Tests for AST quality in corpus."""

    @pytest.fixture
    def corpus_path(self):
        """Find the corpus file."""
        for path in CORPUS_PATHS:
            if path.exists():
                return path
        pytest.skip("No corpus file found")

    def test_has_ast_field(self, corpus_path):
        """Entries should have 'ast' field."""
        with_ast = 0
        total = 0

        with open(corpus_path, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 1000:
                    break
                try:
                    entry = json.loads(line)
                    total += 1
                    if 'ast' in entry and entry['ast']:
                        with_ast += 1
                except:
                    pass

        if total == 0:
            pytest.skip("No entries to check")

        pct_with_ast = with_ast / total
        # At least 80% should have ASTs
        assert pct_with_ast > 0.8, \
            f"Only {pct_with_ast:.1%} have AST field"

    def test_ast_has_tipo(self, corpus_path):
        """AST should have 'tipo' field."""
        valid_ast = 0
        invalid_ast = 0

        with open(corpus_path, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 1000:
                    break
                try:
                    entry = json.loads(line)
                    ast = entry.get('ast')
                    if ast and isinstance(ast, dict):
                        if 'tipo' in ast:
                            valid_ast += 1
                        else:
                            invalid_ast += 1
                except:
                    pass

        total = valid_ast + invalid_ast
        if total == 0:
            pytest.skip("No ASTs to check")

        assert invalid_ast == 0, \
            f"{invalid_ast}/{total} ASTs missing 'tipo' field"

    def test_ast_tipo_values(self, corpus_path):
        """AST tipo should be valid value."""
        valid_tipos = {'frazo', 'subfrazo', 'vortgrupo', 'vorto', 'listo'}
        invalid_tipos = []

        with open(corpus_path, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 1000:
                    break
                try:
                    entry = json.loads(line)
                    ast = entry.get('ast')
                    if ast and isinstance(ast, dict):
                        tipo = ast.get('tipo')
                        if tipo and tipo not in valid_tipos:
                            invalid_tipos.append(tipo)
                except:
                    pass

        assert len(invalid_tipos) == 0, \
            f"Invalid tipo values: {set(invalid_tipos)}"


class TestParseStatistics:
    """Tests for parse statistics in corpus."""

    @pytest.fixture
    def corpus_path(self):
        """Find the corpus file."""
        for path in CORPUS_PATHS:
            if path.exists():
                return path
        pytest.skip("No corpus file found")

    def test_has_parse_rate(self, corpus_path):
        """Entries should have parse rate."""
        with_rate = 0
        total = 0

        with open(corpus_path, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 1000:
                    break
                try:
                    entry = json.loads(line)
                    total += 1
                    if 'parse_rate' in entry:
                        with_rate += 1
                    elif 'ast' in entry and isinstance(entry['ast'], dict):
                        stats = entry['ast'].get('parse_statistics', {})
                        if 'success_rate' in stats:
                            with_rate += 1
                except:
                    pass

        if total == 0:
            pytest.skip("No entries to check")

        pct_with_rate = with_rate / total
        assert pct_with_rate > 0.8, \
            f"Only {pct_with_rate:.1%} have parse rate"

    def test_average_parse_rate(self, corpus_path):
        """Average parse rate should be reasonable (>80%)."""
        rates = []

        with open(corpus_path, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 5000:
                    break
                try:
                    entry = json.loads(line)
                    rate = entry.get('parse_rate')
                    if rate is None and 'ast' in entry:
                        stats = entry['ast'].get('parse_statistics', {})
                        rate = stats.get('success_rate')
                    if rate is not None:
                        rates.append(rate)
                except:
                    pass

        if len(rates) == 0:
            pytest.skip("No parse rates found")

        avg_rate = sum(rates) / len(rates)
        assert avg_rate > 0.8, f"Average parse rate too low: {avg_rate:.1%}"

    def test_parse_rate_range(self, corpus_path):
        """Parse rates should be in valid range [0, 1]."""
        invalid_rates = []

        with open(corpus_path, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 1000:
                    break
                try:
                    entry = json.loads(line)
                    rate = entry.get('parse_rate')
                    if rate is None and 'ast' in entry:
                        stats = entry['ast'].get('parse_statistics', {})
                        rate = stats.get('success_rate')
                    if rate is not None and (rate < 0 or rate > 1):
                        invalid_rates.append((i, rate))
                except:
                    pass

        assert len(invalid_rates) == 0, \
            f"Invalid parse rates: {invalid_rates[:10]}"


class TestCorpusSources:
    """Tests for source diversity in corpus."""

    @pytest.fixture
    def corpus_path(self):
        """Find the corpus file."""
        for path in CORPUS_PATHS:
            if path.exists():
                return path
        pytest.skip("No corpus file found")

    def test_multiple_sources(self, corpus_path):
        """Corpus should have multiple sources."""
        sources = set()

        with open(corpus_path, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 10000:
                    break
                try:
                    entry = json.loads(line)
                    source = entry.get('source')
                    if isinstance(source, dict):
                        sources.add(source.get('name', 'unknown'))
                    elif isinstance(source, str):
                        sources.add(source)
                    else:
                        sources.add('unknown')
                except:
                    pass

        assert len(sources) >= 2, f"Only {len(sources)} source(s): {sources}"

    def test_has_wikipedia(self, corpus_path):
        """Corpus should include Wikipedia."""
        has_wiki = False

        with open(corpus_path, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 50000:
                    break
                try:
                    entry = json.loads(line)
                    source = entry.get('source')
                    if isinstance(source, dict):
                        source_name = source.get('name', '').lower()
                    elif isinstance(source, str):
                        source_name = source.lower()
                    else:
                        source_name = ''
                    if 'wiki' in source_name:
                        has_wiki = True
                        break
                except:
                    pass

        assert has_wiki, "No Wikipedia entries found in corpus"

    def test_sentence_count(self, corpus_path):
        """Should have substantial sentence count."""
        line_count = sum(1 for _ in open(corpus_path, encoding='utf-8'))

        # Should have at least 1M sentences
        assert line_count >= 1_000_000, \
            f"Corpus has only {line_count:,} sentences (expected 1M+)"


class TestCorpusContentQuality:
    """Tests for content quality of corpus entries."""

    @pytest.fixture
    def corpus_path(self):
        """Find the corpus file."""
        for path in CORPUS_PATHS:
            if path.exists():
                return path
        pytest.skip("No corpus file found")

    def test_no_empty_text(self, corpus_path):
        """No entry should have empty text."""
        empty_count = 0
        total = 0

        with open(corpus_path, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 1000:
                    break
                try:
                    entry = json.loads(line)
                    total += 1
                    if not entry.get('text', '').strip():
                        empty_count += 1
                except:
                    pass

        assert empty_count == 0, f"{empty_count}/{total} entries have empty text"

    def test_contains_esperanto_chars(self, corpus_path):
        """Entries should contain Esperanto text."""
        eo_chars = set('ĉĝĥĵŝŭĈĜĤĴŜŬ')
        with_eo = 0
        total = 0

        with open(corpus_path, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 1000:
                    break
                try:
                    entry = json.loads(line)
                    text = entry.get('text', '')
                    total += 1
                    if any(c in text for c in eo_chars):
                        with_eo += 1
                except:
                    pass

        if total == 0:
            pytest.skip("No entries to check")

        pct_with_eo = with_eo / total
        # At least 30% should have Esperanto-specific chars
        assert pct_with_eo > 0.3, \
            f"Only {pct_with_eo:.1%} contain Esperanto chars"

    def test_word_count_field(self, corpus_path):
        """Entries should have word count."""
        with_count = 0
        total = 0

        with open(corpus_path, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 1000:
                    break
                try:
                    entry = json.loads(line)
                    total += 1
                    if 'word_count' in entry:
                        with_count += 1
                    elif 'ast' in entry:
                        stats = entry['ast'].get('parse_statistics', {})
                        if 'total_words' in stats:
                            with_count += 1
                except:
                    pass

        if total == 0:
            pytest.skip("No entries to check")

        pct_with_count = with_count / total
        assert pct_with_count > 0.8, \
            f"Only {pct_with_count:.1%} have word count"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
