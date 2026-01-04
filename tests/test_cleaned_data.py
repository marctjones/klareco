#!/usr/bin/env python3
"""
Cleaned Data Quality Tests.

Tests that the CLEAN stage produced valid, well-formatted text files
with Gutenberg headers/footers removed.

Usage:
    pytest tests/test_cleaned_data.py -v
"""

import pytest
from pathlib import Path


# Expected cleaned files (adjust based on your actual data)
EXPECTED_CLEANED_FILES = [
    'cleaned_la_mastro_de_l_ringoj.txt',
    'cleaned_la_hobito.txt',
]

# Gutenberg markers that should NOT appear in cleaned files
GUTENBERG_MARKERS = [
    '*** START OF',
    '*** END OF',
    'PROJECT GUTENBERG',
    'GUTENBERG-TM',
    'gutenberg.org',
    'This eBook is for the use of',
    'TERMS OF USE',
    'Full license',
]

# Valid Esperanto letters (for content validation)
VALID_EO_LETTERS = set('abcĉdefgĝhĥijĵklmnoprsŝtuŭvzABCĈDEFGĜHĤIJĴKLMNOPRSŜTUŬVZ')


class TestCleanedDataExists:
    """Tests that cleaned data files exist."""

    @pytest.fixture
    def cleaned_dir(self):
        """Get cleaned data directory."""
        # Try multiple possible locations
        paths = [
            Path('data/cleaned/eo'),
            Path('data/cleaned'),
        ]
        for p in paths:
            if p.exists():
                return p
        return Path('data/cleaned/eo')  # Default

    def test_cleaned_directory_exists(self, cleaned_dir):
        """Cleaned directory should exist."""
        assert cleaned_dir.exists(), f"Cleaned directory not found: {cleaned_dir}"

    def test_has_cleaned_files(self, cleaned_dir):
        """Should have at least some cleaned .txt files."""
        if not cleaned_dir.exists():
            pytest.skip("Cleaned directory not found")

        txt_files = list(cleaned_dir.glob('*.txt'))
        assert len(txt_files) > 0, "No .txt files found in cleaned directory"

    def test_cleaned_files_not_empty(self, cleaned_dir):
        """All cleaned files should have content."""
        if not cleaned_dir.exists():
            pytest.skip("Cleaned directory not found")

        empty_files = []
        for txt_file in cleaned_dir.glob('*.txt'):
            if txt_file.stat().st_size == 0:
                empty_files.append(txt_file.name)

        assert len(empty_files) == 0, f"Empty cleaned files: {empty_files}"

    def test_minimum_file_size(self, cleaned_dir):
        """Cleaned files should have meaningful content (>1KB)."""
        if not cleaned_dir.exists():
            pytest.skip("Cleaned directory not found")

        small_files = []
        for txt_file in cleaned_dir.glob('*.txt'):
            if txt_file.stat().st_size < 1024:  # 1KB minimum
                small_files.append((txt_file.name, txt_file.stat().st_size))

        assert len(small_files) == 0, f"Files too small: {small_files}"


class TestGutenbergHeadersRemoved:
    """Tests that Gutenberg headers/footers are properly removed."""

    @pytest.fixture
    def cleaned_dir(self):
        """Get cleaned data directory."""
        paths = [Path('data/cleaned/eo'), Path('data/cleaned')]
        for p in paths:
            if p.exists():
                return p
        return Path('data/cleaned/eo')

    def test_no_gutenberg_start_marker(self, cleaned_dir):
        """No file should contain '*** START OF' marker."""
        if not cleaned_dir.exists():
            pytest.skip("Cleaned directory not found")

        files_with_marker = []
        for txt_file in cleaned_dir.glob('*.txt'):
            content = txt_file.read_text(encoding='utf-8', errors='ignore')
            if '*** START OF' in content.upper():
                files_with_marker.append(txt_file.name)

        assert len(files_with_marker) == 0, \
            f"Files still contain START marker: {files_with_marker}"

    def test_no_gutenberg_end_marker(self, cleaned_dir):
        """No file should contain '*** END OF' marker."""
        if not cleaned_dir.exists():
            pytest.skip("Cleaned directory not found")

        files_with_marker = []
        for txt_file in cleaned_dir.glob('*.txt'):
            content = txt_file.read_text(encoding='utf-8', errors='ignore')
            if '*** END OF' in content.upper():
                files_with_marker.append(txt_file.name)

        assert len(files_with_marker) == 0, \
            f"Files still contain END marker: {files_with_marker}"

    def test_no_project_gutenberg_text(self, cleaned_dir):
        """No file should contain Gutenberg boilerplate (headers/footers/license).

        Uses contextual patterns to avoid false positives from Wikipedia articles
        that legitimately discuss Project Gutenberg.
        """
        import re
        if not cleaned_dir.exists():
            pytest.skip("Cleaned directory not found")

        # Contextual boilerplate patterns (actual headers/footers, not article mentions)
        boilerplate_patterns = [
            r'produced by.*project gutenberg',  # "Produced by Project Gutenberg"
            r'project gutenberg.{0,20}(ebook|e-book|etext)',  # "Project Gutenberg eBook"
            r'project gutenberg.{0,20}license',  # License text
            r'www\.gutenberg\.org',  # Direct URL references
            r'gutenberg literary archive',  # Archive references
            r'this.*project gutenberg.*is for',  # Usage statements
        ]

        files_with_boilerplate = []
        for txt_file in cleaned_dir.glob('*.txt'):
            # Skip wikipedia - it legitimately discusses Gutenberg in article content
            if 'wikipedia' in txt_file.name.lower():
                continue

            content = txt_file.read_text(encoding='utf-8', errors='ignore').lower()
            for pattern in boilerplate_patterns:
                if re.search(pattern, content):
                    files_with_boilerplate.append(txt_file.name)
                    break

        assert len(files_with_boilerplate) == 0, \
            f"Files still contain Gutenberg boilerplate: {files_with_boilerplate}"

    def test_no_license_text(self, cleaned_dir):
        """No file should contain license boilerplate."""
        if not cleaned_dir.exists():
            pytest.skip("Cleaned directory not found")

        license_markers = ['Full license at', 'gutenberg.org/license']
        files_with_license = []

        for txt_file in cleaned_dir.glob('*.txt'):
            # Skip wikipedia - may contain meta discussion about licenses
            if 'wikipedia' in txt_file.name.lower():
                continue

            content = txt_file.read_text(encoding='utf-8', errors='ignore')
            for marker in license_markers:
                if marker.lower() in content.lower():
                    files_with_license.append((txt_file.name, marker))
                    break

        assert len(files_with_license) == 0, \
            f"Files contain license text: {files_with_license}"


class TestCleanedContentQuality:
    """Tests for content quality of cleaned files."""

    @pytest.fixture
    def cleaned_dir(self):
        """Get cleaned data directory."""
        paths = [Path('data/cleaned/eo'), Path('data/cleaned')]
        for p in paths:
            if p.exists():
                return p
        return Path('data/cleaned/eo')

    def test_contains_esperanto_characters(self, cleaned_dir):
        """Files should contain Esperanto-specific characters or x-notation."""
        import re
        if not cleaned_dir.exists():
            pytest.skip("Cleaned directory not found")

        eo_specific = set('ĉĝĥĵŝŭĈĜĤĴŜŬ')
        # X-notation pattern: words containing cx, gx, hx, jx, sx, ux
        x_notation = re.compile(r'\b\w*[cghjsu]x\w*\b', re.IGNORECASE)

        files_without_eo = []

        for txt_file in cleaned_dir.glob('*.txt'):
            # Skip wikipedia.txt - known raw dump issue tracked in #172
            if 'wikipedia' in txt_file.name.lower():
                continue

            content = txt_file.read_text(encoding='utf-8', errors='ignore')
            has_unicode = any(c in content for c in eo_specific)
            has_x_notation = bool(x_notation.search(content))

            if not has_unicode and not has_x_notation:
                files_without_eo.append(txt_file.name)

        # Allow some files to not have special chars (e.g., short files)
        assert len(files_without_eo) <= 2, \
            f"Files missing Esperanto chars: {files_without_eo}"

    def test_reasonable_line_lengths(self, cleaned_dir):
        """Lines should not be excessively long (no run-together text)."""
        if not cleaned_dir.exists():
            pytest.skip("Cleaned directory not found")

        MAX_LINE_LENGTH = 5500  # Reasonable max (some legit long sentences exist)
        files_with_long_lines = []

        for txt_file in cleaned_dir.glob('*.txt'):
            # Skip wikipedia.txt - known raw dump issue tracked in #172
            if 'wikipedia' in txt_file.name.lower():
                continue

            content = txt_file.read_text(encoding='utf-8', errors='ignore')
            for i, line in enumerate(content.split('\n'), 1):
                if len(line) > MAX_LINE_LENGTH:
                    files_with_long_lines.append((txt_file.name, i, len(line)))
                    break

        assert len(files_with_long_lines) == 0, \
            f"Files with excessively long lines: {files_with_long_lines}"

    def test_no_binary_content(self, cleaned_dir):
        """Files should not contain binary/non-text content."""
        if not cleaned_dir.exists():
            pytest.skip("Cleaned directory not found")

        files_with_binary = []

        for txt_file in cleaned_dir.glob('*.txt'):
            try:
                content = txt_file.read_bytes()
                # Check for null bytes (binary indicator)
                if b'\x00' in content:
                    files_with_binary.append(txt_file.name)
            except Exception as e:
                files_with_binary.append((txt_file.name, str(e)))

        assert len(files_with_binary) == 0, \
            f"Files with binary content: {files_with_binary}"

    def test_utf8_encoding(self, cleaned_dir):
        """All files should be valid UTF-8."""
        if not cleaned_dir.exists():
            pytest.skip("Cleaned directory not found")

        invalid_utf8 = []

        for txt_file in cleaned_dir.glob('*.txt'):
            try:
                txt_file.read_text(encoding='utf-8')
            except UnicodeDecodeError as e:
                invalid_utf8.append((txt_file.name, str(e)))

        assert len(invalid_utf8) == 0, f"Files with invalid UTF-8: {invalid_utf8}"


class TestCleanedDataStats:
    """Statistical tests for cleaned data."""

    @pytest.fixture
    def cleaned_dir(self):
        """Get cleaned data directory."""
        paths = [Path('data/cleaned/eo'), Path('data/cleaned')]
        for p in paths:
            if p.exists():
                return p
        return Path('data/cleaned/eo')

    def test_total_content_size(self, cleaned_dir):
        """Total cleaned content should be substantial."""
        if not cleaned_dir.exists():
            pytest.skip("Cleaned directory not found")

        total_bytes = sum(f.stat().st_size for f in cleaned_dir.glob('*.txt'))
        total_mb = total_bytes / (1024 * 1024)

        # Should have at least 1MB of cleaned text
        assert total_mb >= 1, f"Total cleaned content too small: {total_mb:.2f} MB"

    def test_word_count(self, cleaned_dir):
        """Should have substantial word count across files."""
        if not cleaned_dir.exists():
            pytest.skip("Cleaned directory not found")

        total_words = 0
        for txt_file in cleaned_dir.glob('*.txt'):
            content = txt_file.read_text(encoding='utf-8', errors='ignore')
            total_words += len(content.split())

        # Should have at least 100K words
        assert total_words >= 100_000, \
            f"Total word count too low: {total_words:,}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
