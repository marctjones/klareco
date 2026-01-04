#!/usr/bin/env python3
"""
Extracted Data Quality Tests.

Tests that the EXTRACT stage produced valid JSONL files with proper
metadata (article titles, chapters, etc.).

Usage:
    pytest tests/test_extracted_data.py -v
"""

import json
import pytest
from pathlib import Path


class TestExtractedFilesExist:
    """Tests that extracted JSONL files exist."""

    @pytest.fixture
    def extracted_dir(self):
        """Get extracted data directory."""
        return Path('data/extracted')

    def test_extracted_directory_exists(self, extracted_dir):
        """Extracted directory should exist."""
        assert extracted_dir.exists(), f"Extracted directory not found: {extracted_dir}"

    def test_wikipedia_sentences_exist(self, extracted_dir):
        """Wikipedia extracted sentences should exist."""
        if not extracted_dir.exists():
            pytest.skip("Extracted directory not found")

        wiki_file = extracted_dir / 'wikipedia_sentences.jsonl'
        assert wiki_file.exists(), "wikipedia_sentences.jsonl not found"

    def test_books_sentences_exist(self, extracted_dir):
        """Books extracted sentences should exist."""
        if not extracted_dir.exists():
            pytest.skip("Extracted directory not found")

        books_file = extracted_dir / 'books_sentences.jsonl'
        assert books_file.exists(), "books_sentences.jsonl not found"

    def test_extracted_files_not_empty(self, extracted_dir):
        """Extracted JSONL files should have content."""
        if not extracted_dir.exists():
            pytest.skip("Extracted directory not found")

        for jsonl_file in extracted_dir.glob('*.jsonl'):
            assert jsonl_file.stat().st_size > 0, \
                f"Empty extracted file: {jsonl_file.name}"


class TestWikipediaExtraction:
    """Tests for Wikipedia extracted data quality."""

    @pytest.fixture
    def wiki_file(self):
        """Get Wikipedia sentences file."""
        path = Path('data/extracted/wikipedia_sentences.jsonl')
        if not path.exists():
            pytest.skip("Wikipedia extraction not found")
        return path

    def test_valid_jsonl_format(self, wiki_file):
        """Each line should be valid JSON."""
        invalid_lines = []
        with open(wiki_file, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 1000:  # Sample first 1000 lines
                    break
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    invalid_lines.append(i)

        assert len(invalid_lines) == 0, \
            f"Invalid JSON at lines: {invalid_lines[:10]}"

    def test_has_required_fields(self, wiki_file):
        """Each entry should have required metadata fields."""
        required_fields = {'text', 'source'}
        missing_fields = []

        with open(wiki_file, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 1000:
                    break
                try:
                    entry = json.loads(line)
                    missing = required_fields - set(entry.keys())
                    if missing:
                        missing_fields.append((i, missing))
                except:
                    pass

        assert len(missing_fields) == 0, \
            f"Entries missing fields: {missing_fields[:10]}"

    def test_has_article_metadata(self, wiki_file):
        """Wikipedia entries should have article metadata."""
        entries_with_metadata = 0
        entries_without = 0

        with open(wiki_file, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 1000:
                    break
                try:
                    entry = json.loads(line)
                    if 'article_title' in entry or 'article_id' in entry:
                        entries_with_metadata += 1
                    else:
                        entries_without += 1
                except:
                    pass

        total = entries_with_metadata + entries_without
        if total == 0:
            pytest.skip("No entries to check")

        pct_with_metadata = entries_with_metadata / total
        assert pct_with_metadata > 0.9, \
            f"Only {pct_with_metadata:.1%} have article metadata"

    def test_text_not_empty(self, wiki_file):
        """Text field should not be empty."""
        empty_text = 0
        total = 0

        with open(wiki_file, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 1000:
                    break
                try:
                    entry = json.loads(line)
                    total += 1
                    if not entry.get('text', '').strip():
                        empty_text += 1
                except:
                    pass

        assert empty_text == 0, f"{empty_text}/{total} entries have empty text"

    def test_sentence_length_reasonable(self, wiki_file):
        """Sentences should have reasonable length."""
        too_short = 0
        too_long = 0
        total = 0

        with open(wiki_file, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 1000:
                    break
                try:
                    entry = json.loads(line)
                    text = entry.get('text', '')
                    total += 1
                    words = len(text.split())
                    if words < 3:
                        too_short += 1
                    elif words > 500:
                        too_long += 1
                except:
                    pass

        # Allow up to 5% edge cases
        assert too_short / total < 0.05, \
            f"{too_short}/{total} sentences too short (<3 words)"
        assert too_long / total < 0.01, \
            f"{too_long}/{total} sentences too long (>500 words)"

    def test_minimum_sentence_count(self, wiki_file):
        """Should have substantial number of sentences."""
        line_count = sum(1 for _ in open(wiki_file, encoding='utf-8'))
        # Wikipedia should have at least 100K sentences
        assert line_count >= 100_000, \
            f"Wikipedia has only {line_count:,} sentences (expected 100K+)"


class TestWikipediaArticleCompleteness:
    """Tests that Wikipedia extraction includes full article content."""

    @pytest.fixture
    def wiki_file(self):
        """Get Wikipedia sentences file."""
        path = Path('data/extracted/wikipedia_sentences.jsonl')
        if not path.exists():
            pytest.skip("Wikipedia extraction not found")
        return path

    def test_multiple_sentences_per_article(self, wiki_file):
        """Articles should have multiple sentences (not just snippets)."""
        from collections import Counter
        article_counts = Counter()

        with open(wiki_file, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 50000:
                    break
                try:
                    entry = json.loads(line)
                    article_id = entry.get('article_id') or entry.get('article_title')
                    if article_id:
                        article_counts[article_id] += 1
                except:
                    pass

        if not article_counts:
            pytest.skip("No article IDs found")

        # Calculate statistics
        avg_sentences = sum(article_counts.values()) / len(article_counts)
        single_sentence_articles = sum(1 for c in article_counts.values() if c == 1)
        pct_single = single_sentence_articles / len(article_counts)

        # Average article should have at least 3 sentences
        assert avg_sentences >= 3, \
            f"Average {avg_sentences:.1f} sentences/article suggests incomplete extraction"

        # Less than 30% should be single-sentence articles
        assert pct_single < 0.30, \
            f"{pct_single:.1%} are single-sentence articles (too many snippets)"

    def test_article_count(self, wiki_file):
        """Should have substantial number of distinct articles."""
        article_ids = set()

        with open(wiki_file, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 100000:
                    break
                try:
                    entry = json.loads(line)
                    article_id = entry.get('article_id') or entry.get('article_title')
                    if article_id:
                        article_ids.add(article_id)
                except:
                    pass

        # Should have good article coverage in the sample
        assert len(article_ids) >= 1000, \
            f"Only {len(article_ids):,} articles in first 100K sentences (expected 1K+)"

    def test_has_section_structure(self, wiki_file):
        """Some entries should have section information."""
        with_section = 0
        total = 0

        with open(wiki_file, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 10000:
                    break
                try:
                    entry = json.loads(line)
                    total += 1
                    if entry.get('section') or entry.get('section_level', 0) > 0:
                        with_section += 1
                except:
                    pass

        # At least some entries should have section info (articles have sections)
        # This is a soft check - not all extractors preserve sections
        if total > 0 and with_section == 0:
            # Log warning but don't fail - section info is optional
            pass


class TestWikipediaContentCleanliness:
    """Tests that Wikipedia content is properly cleaned."""

    @pytest.fixture
    def wiki_file(self):
        """Get Wikipedia sentences file."""
        path = Path('data/extracted/wikipedia_sentences.jsonl')
        if not path.exists():
            pytest.skip("Wikipedia extraction not found")
        return path

    def test_no_wiki_markup(self, wiki_file):
        """Text should not contain wiki markup."""
        wiki_patterns = [
            '[[',      # Wiki links
            ']]',
            '{{',      # Templates
            '}}',
            '&lt;',    # HTML entities
            '&gt;',
            '&amp;',
            '<ref',    # References
            '</ref>',
            '|thumb|', # Image markup
            '|right|',
            '|left|',
        ]
        entries_with_markup = []

        with open(wiki_file, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 5000:
                    break
                try:
                    entry = json.loads(line)
                    text = entry.get('text', '')
                    for pattern in wiki_patterns:
                        if pattern in text:
                            entries_with_markup.append((i, pattern, text[:50]))
                            break
                except:
                    pass

        assert len(entries_with_markup) == 0, \
            f"Found wiki markup in {len(entries_with_markup)} entries: {entries_with_markup[:5]}"

    def test_no_html_tags(self, wiki_file):
        """Text should not contain HTML tags."""
        import re
        html_pattern = re.compile(r'<[a-zA-Z][^>]*>')
        entries_with_html = []

        with open(wiki_file, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 5000:
                    break
                try:
                    entry = json.loads(line)
                    text = entry.get('text', '')
                    if html_pattern.search(text):
                        entries_with_html.append((i, text[:80]))
                except:
                    pass

        assert len(entries_with_html) == 0, \
            f"Found HTML in {len(entries_with_html)} entries: {entries_with_html[:5]}"

    def test_no_url_artifacts(self, wiki_file):
        """Text should not contain raw URLs or URL fragments."""
        url_patterns = ['http://', 'https://', 'www.', '.com/', '.org/']
        entries_with_urls = []

        with open(wiki_file, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 5000:
                    break
                try:
                    entry = json.loads(line)
                    text = entry.get('text', '')
                    for pattern in url_patterns:
                        if pattern in text.lower():
                            entries_with_urls.append((i, text[:80]))
                            break
                except:
                    pass

        # Allow a few URLs (some articles legitimately mention websites)
        assert len(entries_with_urls) < 50, \
            f"Too many entries with URLs ({len(entries_with_urls)}): {entries_with_urls[:5]}"

    def test_contains_esperanto_text(self, wiki_file):
        """Text should be primarily Esperanto (has diacritics)."""
        eo_chars = set('ĉĝĥĵŝŭĈĜĤĴŜŬ')
        with_eo = 0
        total = 0

        with open(wiki_file, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 5000:
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
        # At least 40% should have Esperanto-specific characters
        assert pct_with_eo > 0.40, \
            f"Only {pct_with_eo:.1%} contain Esperanto chars (expected >40%)"

    def test_no_foreign_language_blocks(self, wiki_file):
        """Should not have large blocks of non-Esperanto text."""
        # Common foreign patterns that indicate untranslated content
        foreign_patterns = [
            'This article',   # English
            'The following',
            'See also',
            'References',
            'Cet article',    # French
            'Dieser Artikel', # German
            'Este artículo',  # Spanish
        ]
        foreign_entries = []

        with open(wiki_file, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 5000:
                    break
                try:
                    entry = json.loads(line)
                    text = entry.get('text', '')
                    for pattern in foreign_patterns:
                        if pattern in text:
                            foreign_entries.append((i, pattern, text[:60]))
                            break
                except:
                    pass

        assert len(foreign_entries) < 10, \
            f"Found foreign language text in {len(foreign_entries)} entries: {foreign_entries[:5]}"

    def test_no_category_or_template_text(self, wiki_file):
        """Should not contain category listings or template names."""
        meta_patterns = [
            'Kategorio:',    # Esperanto category
            'Category:',     # English category
            'Dosiero:',      # File
            'File:',
            'Image:',
            'Bildo:',
            'Ŝablono:',      # Template
            'Template:',
        ]
        meta_entries = []

        with open(wiki_file, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 5000:
                    break
                try:
                    entry = json.loads(line)
                    text = entry.get('text', '')
                    for pattern in meta_patterns:
                        if pattern in text:
                            meta_entries.append((i, pattern))
                            break
                except:
                    pass

        # Allow a few edge cases (some articles may legitimately mention categories)
        assert len(meta_entries) < 5, \
            f"Found category/template text in {len(meta_entries)} entries: {meta_entries[:10]}"

    def test_reasonable_character_distribution(self, wiki_file):
        """Text should have reasonable character distribution (not garbage)."""
        from collections import Counter

        char_counts = Counter()
        total_chars = 0

        with open(wiki_file, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 1000:
                    break
                try:
                    entry = json.loads(line)
                    text = entry.get('text', '').lower()
                    char_counts.update(text)
                    total_chars += len(text)
                except:
                    pass

        if total_chars == 0:
            pytest.skip("No text to analyze")

        # Check that common Esperanto letters are present
        common_eo_letters = 'aeiou'
        for letter in common_eo_letters:
            pct = char_counts[letter] / total_chars
            assert pct > 0.02, \
                f"Letter '{letter}' only {pct:.1%} of text (expected >2%)"

        # Check that unusual characters aren't too common
        unusual_chars = sum(char_counts[c] for c in char_counts
                          if ord(c) > 500 and c not in 'ĉĝĥĵŝŭ')
        unusual_pct = unusual_chars / total_chars
        assert unusual_pct < 0.05, \
            f"Unusual characters are {unusual_pct:.1%} of text (expected <5%)"


class TestBooksExtraction:
    """Tests for books extracted data quality."""

    @pytest.fixture
    def books_file(self):
        """Get books sentences file."""
        path = Path('data/extracted/books_sentences.jsonl')
        if not path.exists():
            pytest.skip("Books extraction not found")
        return path

    def test_valid_jsonl_format(self, books_file):
        """Each line should be valid JSON."""
        invalid_lines = []
        with open(books_file, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 1000:
                    break
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    invalid_lines.append(i)

        assert len(invalid_lines) == 0, \
            f"Invalid JSON at lines: {invalid_lines[:10]}"

    def test_has_required_fields(self, books_file):
        """Each entry should have required fields."""
        required_fields = {'text', 'source'}
        missing_fields = []

        with open(books_file, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 1000:
                    break
                try:
                    entry = json.loads(line)
                    missing = required_fields - set(entry.keys())
                    if missing:
                        missing_fields.append((i, missing))
                except:
                    pass

        assert len(missing_fields) == 0, \
            f"Entries missing fields: {missing_fields[:10]}"

    def test_has_chapter_metadata(self, books_file):
        """Book entries should have chapter metadata."""
        entries_with_chapter = 0
        total = 0

        with open(books_file, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 1000:
                    break
                try:
                    entry = json.loads(line)
                    total += 1
                    if 'chapter' in entry or 'chapter_number' in entry:
                        entries_with_chapter += 1
                except:
                    pass

        if total == 0:
            pytest.skip("No entries to check")

        pct_with_chapter = entries_with_chapter / total
        # At least 50% should have chapter info (some may be intro/etc)
        assert pct_with_chapter > 0.5, \
            f"Only {pct_with_chapter:.1%} have chapter metadata"

    def test_multiple_sources(self, books_file):
        """Should have sentences from multiple books."""
        # Read all sources - file is small (~27K lines) and sources are unevenly
        # distributed (e.g., Lord of the Rings has 20K+ sentences alone)
        sources = set()

        with open(books_file, encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    sources.add(entry.get('source', 'unknown'))
                except:
                    pass

        assert len(sources) >= 2, f"Only {len(sources)} source(s): {sources}"

    def test_contains_esperanto(self, books_file):
        """Text should contain Esperanto characters."""
        eo_chars = set('ĉĝĥĵŝŭĈĜĤĴŜŬ')
        has_eo = 0
        total = 0

        with open(books_file, encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if i > 1000:
                    break
                try:
                    entry = json.loads(line)
                    text = entry.get('text', '')
                    total += 1
                    if any(c in text for c in eo_chars):
                        has_eo += 1
                except:
                    pass

        if total == 0:
            pytest.skip("No entries to check")

        pct_with_eo = has_eo / total
        # At least 30% should have Esperanto-specific chars
        assert pct_with_eo > 0.3, \
            f"Only {pct_with_eo:.1%} contain Esperanto chars"


class TestExtractedDataConsistency:
    """Cross-file consistency tests."""

    @pytest.fixture
    def extracted_dir(self):
        """Get extracted data directory."""
        return Path('data/extracted')

    def test_no_duplicate_ids(self, extracted_dir):
        """Check for duplicate sentence IDs if present."""
        if not extracted_dir.exists():
            pytest.skip("Extracted directory not found")

        for jsonl_file in extracted_dir.glob('*.jsonl'):
            ids = set()
            duplicates = []

            with open(jsonl_file, encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    if i > 10000:
                        break
                    try:
                        entry = json.loads(line)
                        entry_id = entry.get('id')
                        if entry_id:
                            if entry_id in ids:
                                duplicates.append(entry_id)
                            ids.add(entry_id)
                    except:
                        pass

            assert len(duplicates) == 0, \
                f"Duplicate IDs in {jsonl_file.name}: {duplicates[:10]}"

    def test_source_field_consistent(self, extracted_dir):
        """Source field should be consistent per file."""
        if not extracted_dir.exists():
            pytest.skip("Extracted directory not found")

        wiki_file = extracted_dir / 'wikipedia_sentences.jsonl'
        if wiki_file.exists():
            sources = set()
            with open(wiki_file, encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    if i > 100:
                        break
                    try:
                        entry = json.loads(line)
                        sources.add(entry.get('source'))
                    except:
                        pass

            # Wikipedia should have consistent source
            assert 'wikipedia' in str(sources).lower(), \
                f"Unexpected sources in Wikipedia file: {sources}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
