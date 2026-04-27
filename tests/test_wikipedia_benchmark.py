#!/usr/bin/env python3
"""
Wikipedia Benchmark Tests.

Verifies that our extracted Wikipedia data includes content from
the most popular/important articles:
- Esperanto equivalents of top 50 English Wikipedia articles
- 50 popular Esperanto-specific articles

These tests ensure our extraction process captured full article content,
not just snippets or partial data.

Usage:
    pytest tests/test_wikipedia_benchmark.py -v
"""

import json
import pytest
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set


# Paths
BENCHMARK_FILE = Path('data/benchmarks/wikipedia_articles/benchmark_articles.json')
WIKI_EXTRACTED = Path('data/extracted/wikipedia_sentences.jsonl')
CORPUS_FILE = Path('data/corpus/unified.jsonl')

# Cache for expensive data loading (module-level to share across tests)
_EXTRACTED_BY_TITLE_CACHE = None


def _load_extracted_by_title():
    """Load extracted sentences grouped by article title (cached)."""
    global _EXTRACTED_BY_TITLE_CACHE
    if _EXTRACTED_BY_TITLE_CACHE is not None:
        return _EXTRACTED_BY_TITLE_CACHE

    if not WIKI_EXTRACTED.exists():
        return None

    by_title = defaultdict(list)
    with open(WIKI_EXTRACTED, encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                title = entry.get('article_title')
                if title:
                    by_title[title].append(entry.get('text', ''))
            except:
                pass

    _EXTRACTED_BY_TITLE_CACHE = dict(by_title)
    return _EXTRACTED_BY_TITLE_CACHE


class TestBenchmarkDataExists:
    """Tests that benchmark data files exist."""

    def test_benchmark_file_exists(self):
        """Benchmark articles should be downloaded (skip if not yet fetched)."""
        if not BENCHMARK_FILE.exists():
            pytest.skip("Benchmark articles not found. Run: python scripts/fetch_benchmark_articles.py")

    def test_wiki_extracted_exists(self):
        """Wikipedia extraction should exist."""
        assert WIKI_EXTRACTED.exists(), \
            "Wikipedia extraction not found at data/extracted/wikipedia_sentences.jsonl"


class TestBenchmarkArticleCoverage:
    """Tests that benchmark articles appear in our extracted data."""

    @pytest.fixture
    def benchmark_articles(self):
        """Load benchmark articles."""
        if not BENCHMARK_FILE.exists():
            pytest.skip("Benchmark file not found")
        with open(BENCHMARK_FILE, encoding='utf-8') as f:
            data = json.load(f)
        return data['articles']

    @pytest.fixture
    def extracted_titles(self):
        """Get all article titles from extracted Wikipedia data."""
        data = _load_extracted_by_title()
        if data is None:
            pytest.skip("Wikipedia extraction not found")
        return set(data.keys())

    @pytest.fixture
    def extracted_by_title(self):
        """Get extracted sentences grouped by article title."""
        data = _load_extracted_by_title()
        if data is None:
            pytest.skip("Wikipedia extraction not found")
        return data

    def test_benchmark_article_count(self, benchmark_articles):
        """Should have at least 80 benchmark articles."""
        assert len(benchmark_articles) >= 80, \
            f"Only {len(benchmark_articles)} benchmark articles (expected 80+)"

    def test_major_articles_present(self, benchmark_articles, extracted_titles):
        """Major benchmark articles should be in extracted data."""
        # These are critical articles that must be present
        critical_titles = [
            'Esperanto',
            'Usono',  # United States
            'Germanio',
            'Francio',
            'Eŭropo',
        ]

        missing = [t for t in critical_titles if t not in extracted_titles]
        assert len(missing) == 0, \
            f"Critical articles missing from extraction: {missing}"

    def test_benchmark_coverage_rate(self, benchmark_articles, extracted_titles):
        """At least 50% of benchmark articles should be in extracted data."""
        benchmark_titles = {a['title'] for a in benchmark_articles}
        found = benchmark_titles & extracted_titles
        coverage = len(found) / len(benchmark_titles)

        assert coverage >= 0.50, \
            f"Only {coverage:.1%} of benchmark articles found (expected 50%+)"

    def test_article_has_multiple_sentences(self, benchmark_articles, extracted_by_title):
        """Found benchmark articles should have multiple sentences."""
        benchmark_titles = {a['title'] for a in benchmark_articles}
        found_titles = benchmark_titles & set(extracted_by_title.keys())

        single_sentence = []
        for title in found_titles:
            sentences = extracted_by_title.get(title, [])
            if len(sentences) < 3:
                single_sentence.append((title, len(sentences)))

        # Less than 20% should have fewer than 3 sentences
        if found_titles:
            pct_single = len(single_sentence) / len(found_titles)
            assert pct_single < 0.20, \
                f"{pct_single:.1%} of articles have <3 sentences: {single_sentence[:10]}"


class TestBenchmarkContentCompleteness:
    """Tests that benchmark article content is complete, not truncated."""

    @pytest.fixture
    def benchmark_articles(self):
        """Load benchmark articles with their full content."""
        if not BENCHMARK_FILE.exists():
            pytest.skip("Benchmark file not found")
        with open(BENCHMARK_FILE, encoding='utf-8') as f:
            data = json.load(f)
        return {a['title']: a for a in data['articles']}

    @pytest.fixture
    def extracted_by_title(self):
        """Get extracted sentences grouped by article title."""
        data = _load_extracted_by_title()
        if data is None:
            pytest.skip("Wikipedia extraction not found")
        return data

    def test_content_length_reasonable(self, benchmark_articles, extracted_by_title):
        """Extracted content should be substantial portion of original."""
        checked = 0
        low_coverage = []

        for title, article in benchmark_articles.items():
            if title not in extracted_by_title:
                continue

            original_chars = len(article.get('extract', ''))
            if original_chars < 1000:
                continue  # Skip very short articles

            extracted_text = ' '.join(extracted_by_title[title])
            extracted_chars = len(extracted_text)

            # Extracted should be at least 30% of original
            # (some content may be filtered, tables removed, etc.)
            coverage = extracted_chars / original_chars
            checked += 1

            if coverage < 0.30:
                low_coverage.append((title, f"{coverage:.1%}"))

        if checked == 0:
            pytest.skip("No matching articles to check")

        # At least 70% should have reasonable coverage
        pct_low = len(low_coverage) / checked
        assert pct_low < 0.30, \
            f"{pct_low:.1%} of articles have low content coverage: {low_coverage[:10]}"

    def test_key_terms_present(self, benchmark_articles, extracted_by_title):
        """Key terms from original article should appear in extracted text."""
        # Test a few specific articles
        test_cases = [
            ('Esperanto', ['lingvo', 'internacia', 'Zamenhof']),
            ('Germanio', ['lando', 'Eŭropo', 'Berlino']),
            ('Usono', ['lando', 'Ameriko', 'Vaŝingtono']),
        ]

        for title, expected_terms in test_cases:
            if title not in extracted_by_title:
                continue

            extracted_text = ' '.join(extracted_by_title[title]).lower()

            for term in expected_terms:
                assert term.lower() in extracted_text, \
                    f"Term '{term}' not found in extracted '{title}'"

    def test_no_truncated_sentences(self, extracted_by_title):
        """Sentences should be complete, not truncated mid-word."""
        # Check for obvious truncation patterns (mid-word breaks)
        truncated = []

        for title, sentences in list(extracted_by_title.items())[:50]:
            for sent in sentences[:20]:
                # Skip short sentences
                if len(sent) < 20:
                    continue

                # Check for mid-word truncation (ends with hyphen or partial word)
                if sent.rstrip().endswith('-'):
                    truncated.append((title, sent[-30:]))
                # Check for broken HTML entities
                elif '&' in sent[-10:] and ';' not in sent[-10:]:
                    truncated.append((title, sent[-30:]))

        # Allow a few edge cases
        assert len(truncated) < 50, \
            f"Found {len(truncated)} truncated sentences: {truncated[:5]}"


class TestBenchmarkContentQuality:
    """Tests that benchmark article content is clean and properly formatted."""

    @pytest.fixture
    def extracted_by_title(self):
        """Get extracted sentences grouped by article title."""
        data = _load_extracted_by_title()
        if data is None:
            pytest.skip("Wikipedia extraction not found")
        return data

    def test_no_wiki_markup_in_benchmark(self, extracted_by_title):
        """Benchmark article content should have no wiki markup."""
        wiki_patterns = ['[[', ']]', '{{', '}}', '<ref', '</ref>']
        found_markup = []

        # Check all sentences from benchmark articles
        for title, sentences in list(extracted_by_title.items())[:100]:
            for sent in sentences:
                for pattern in wiki_patterns:
                    if pattern in sent:
                        found_markup.append((title, pattern, sent[:50]))
                        break

        assert len(found_markup) == 0, \
            f"Found wiki markup in benchmark articles: {found_markup[:5]}"

    def test_sentences_are_esperanto(self, extracted_by_title):
        """Sentences from benchmark articles should be Esperanto."""
        eo_chars = set('ĉĝĥĵŝŭĈĜĤĴŜŬ')
        articles_checked = 0
        articles_with_eo = 0

        for title, sentences in list(extracted_by_title.items())[:100]:
            if not sentences:
                continue
            articles_checked += 1

            # Check if any sentence has Esperanto characters
            all_text = ' '.join(sentences)
            if any(c in all_text for c in eo_chars):
                articles_with_eo += 1

        if articles_checked == 0:
            pytest.skip("No articles to check")

        pct = articles_with_eo / articles_checked
        assert pct >= 0.80, \
            f"Only {pct:.1%} of articles contain Esperanto characters"


class TestBenchmarkStatistics:
    """Statistical tests on benchmark article extraction."""

    @pytest.fixture
    def benchmark_articles(self):
        """Load benchmark articles."""
        if not BENCHMARK_FILE.exists():
            pytest.skip("Benchmark file not found")
        with open(BENCHMARK_FILE, encoding='utf-8') as f:
            data = json.load(f)
        return data['articles']

    @pytest.fixture
    def extracted_by_title(self):
        """Get extracted sentences grouped by article title."""
        data = _load_extracted_by_title()
        if data is None:
            pytest.skip("Wikipedia extraction not found")
        return data

    def test_average_sentences_per_article(self, benchmark_articles, extracted_by_title):
        """Benchmark articles should average many sentences."""
        benchmark_titles = {a['title'] for a in benchmark_articles}
        found_titles = benchmark_titles & set(extracted_by_title.keys())

        if not found_titles:
            pytest.skip("No benchmark articles found in extraction")

        sentence_counts = [len(extracted_by_title[t]) for t in found_titles]
        avg = sum(sentence_counts) / len(sentence_counts)

        # Average should be at least 10 sentences per article
        assert avg >= 10, \
            f"Average {avg:.1f} sentences/article is too low (expected 10+)"

    def test_total_extracted_content(self, benchmark_articles, extracted_by_title):
        """Total extracted content from benchmarks should be substantial."""
        benchmark_titles = {a['title'] for a in benchmark_articles}

        total_chars = 0
        for title in benchmark_titles:
            if title in extracted_by_title:
                total_chars += sum(len(s) for s in extracted_by_title[title])

        # Should have at least 100KB of content from benchmark articles
        assert total_chars >= 100_000, \
            f"Only {total_chars:,} chars from benchmark articles (expected 100K+)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
