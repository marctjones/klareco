#!/usr/bin/env python3
"""
Clean corpus texts for Esperanto parser training.

This script processes raw Esperanto texts and produces cleaned versions:
- Removes Project Gutenberg headers/footers
- Removes markdown formatting
- Removes tables, illustrations, and other non-prose content
- Removes page numbers and running headers
- Normalizes whitespace
- Outputs one sentence per line (preserving paragraph breaks as empty lines)

Usage:
    python scripts/clean_corpus_texts.py --input data/clean_corpus/esperanto \
                                          --output data/clean_corpus/cleaned
"""

import argparse
import re
from pathlib import Path
from typing import List, Optional


# Files to SKIP (copyrighted, not public domain)
SKIP_FILES = {
    'la_hobito.txt',           # The Hobbit - copyrighted
    'la_mastro_de_l_ringoj.txt',  # Lord of the Rings - copyrighted
    'MANIFEST.md',              # Not a text
}


def remove_gutenberg_header_footer(text: str) -> str:
    """Remove Project Gutenberg header and footer."""
    # Find start marker - look for the most definitive one first
    start_markers = [
        (r'\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*', True),
        (r'\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG', True),
        (r'Produced by[^\n]+\n', False),  # Less definitive
    ]

    # Find end marker
    end_markers = [
        r'\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG',
        r'End of (the )?Project Gutenberg',
    ]

    # Try to find start
    start_idx = 0
    for marker, skip_after_blank in start_markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            if skip_after_blank:
                # Find the next blank line after the marker, skip any preamble
                after_marker = text[match.end():]
                # Look for the first real content (paragraph after blank lines)
                blank_match = re.search(r'\n\s*\n', after_marker)
                if blank_match:
                    start_idx = match.end() + blank_match.end()
                else:
                    start_idx = match.end()
            else:
                start_idx = match.end()
            break

    # Try to find end
    end_idx = len(text)
    for marker in end_markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            end_idx = match.start()
            break

    return text[start_idx:end_idx]


def remove_markdown_formatting(text: str) -> str:
    """Remove markdown formatting."""
    # Remove markdown headers (# ## ### etc.)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)

    # Remove markdown emphasis (*text* **text** _text_ __text__)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)

    # Remove any remaining standalone asterisks and underscores (unbalanced markdown)
    # But preserve apostrophes and legitimate punctuation
    text = re.sub(r'\*+', '', text)  # Remove all asterisks
    text = re.sub(r'(?<![a-zĉĝĥĵŝŭ])_+(?![a-zĉĝĥĵŝŭ])', '', text, flags=re.IGNORECASE)  # Remove underscores not in words

    # Remove markdown links [text](url)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Remove bare URLs
    text = re.sub(r'https?://\S+', '', text)

    # Remove markdown code blocks
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
    text = re.sub(r'`[^`]+`', '', text)

    return text


def remove_formatting_artifacts(text: str) -> str:
    """Remove various formatting artifacts."""
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        original = line

        # Skip illustration markers
        if re.match(r'\s*\[Ilustrajxo:', line, re.IGNORECASE):
            continue
        if re.match(r'\s*\[Illustration', line, re.IGNORECASE):
            continue

        # Skip page numbers (standalone numbers)
        if re.match(r'^\s*\d+\s*$', line):
            continue

        # Skip separator lines (asterisks, dashes, etc.)
        if re.match(r'^\s*[\*\-_=·]+\s*$', line):
            continue
        if re.match(r'^\s*\*\s+\*\s+\*', line):
            continue

        # Skip running headers (all caps, short lines)
        stripped = line.strip()
        if stripped.isupper() and len(stripped) < 60 and len(stripped) > 2:
            # Could be a chapter heading, check if it looks like a running header
            if re.match(r'^[A-ZĈĜĤĴŜŬ\s\-]+$', stripped):
                # If it's a repeated title, skip it
                if stripped in ['DOKTORO JEKYLL KAJ SINJORO HYDE', 'LA TEMPO-MASXINO', 'LA TEMPO-MAŝINO']:
                    continue

        # Skip omnibus.se and similar markers
        if re.match(r'^\s*@omnibus', line, re.IGNORECASE):
            continue
        if re.match(r'^\s*(www\.|http)', line, re.IGNORECASE):
            continue

        # Skip Document Outline lines (table of contents markers)
        if re.match(r'^\s*Document Outline\s*$', line, re.IGNORECASE):
            continue
        if re.match(r'^\s*\+\s+(Enhavo|La\s|Serc|Doktoro|La\s+rakonto|La\s+lasta|La\s+okazaj|La\s+plena)', line):
            continue

        # Skip single punctuation or special characters
        if re.match(r'^\s*[·•◦▪▫–—]\s*$', line):
            continue

        # Remove BOM
        line = line.replace('\ufeff', '')

        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def fix_hyphenation(text: str) -> str:
    """Fix words broken across lines by hyphenation."""
    # Pattern: word- at end of line followed by continuation
    # e.g., "kla-\nrigadis" -> "klarigadis"
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace and join sentences that span multiple lines."""
    # Replace multiple spaces with single space
    text = re.sub(r'[ \t]+', ' ', text)

    # Normalize line endings
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\r', '\n', text)

    # Replace 3+ newlines with 2 (paragraph break marker)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Smart line joining: join lines that don't end with sentence-ending punctuation
    # This handles books where lines are wrapped at fixed width (even with blank lines between)
    lines = text.split('\n')
    joined_lines = []
    current_paragraph = []
    blank_count = 0

    for line in lines:
        stripped = line.strip()

        # Track consecutive blank lines
        if not stripped:
            blank_count += 1
            # Only treat 2+ consecutive blanks as definite paragraph break
            if blank_count >= 2 and current_paragraph:
                joined_lines.append(' '.join(current_paragraph))
                current_paragraph = []
                joined_lines.append('')  # Paragraph break marker
            continue

        blank_count = 0
        current_paragraph.append(stripped)

        # Check if this line ends a sentence
        # Sentence-ending: . ! ? : ; (possibly followed by quote/paren/dash)
        ends_sentence = bool(re.search(r'[.!?;:][\s"\')\]\-—]*$', stripped))

        # Is this a chapter/section heading?
        # Short lines that look like titles (capitalized, no trailing comma)
        # Also check for common chapter patterns
        is_heading = False
        if len(stripped) < 80 and not stripped[-1] == ',':
            # Chapter markers: "ĈAPITRO I", "La rakonto pri la pordo", numbers like "I" "II" "III"
            if re.match(r'^(ĈAPITRO|Ĉapitro|PARTO|Parto|LIBRO|Libro)\b', stripped):
                is_heading = True
            elif re.match(r'^[IVXLC]+\.?$', stripped):  # Roman numerals alone
                is_heading = True
            elif re.match(r'^\d+\.?\s*$', stripped):  # Just numbers
                is_heading = True
            elif (len(stripped) < 60 and
                  stripped[0].isupper() and
                  not any(c in stripped for c in '.!?;:') and
                  len(stripped.split()) <= 8):  # Short title-like phrases
                is_heading = True

        # Flush paragraph if we hit sentence end or heading
        if ends_sentence or is_heading:
            joined_lines.append(' '.join(current_paragraph))
            current_paragraph = []

    # Don't forget last paragraph
    if current_paragraph:
        joined_lines.append(' '.join(current_paragraph))

    text = '\n'.join(joined_lines)

    # Clean up multiple blank lines
    text = re.sub(r'\n{2,}', '\n\n', text)

    # Clean up any double spaces created by joining
    text = re.sub(r'  +', ' ', text)

    # Strip leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text


def remove_non_esperanto_content(text: str) -> str:
    """Remove content that's clearly not Esperanto prose."""
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Skip empty lines (but keep them for paragraph structure)
        if not stripped:
            cleaned_lines.append('')
            continue

        # Skip lines that are mostly numbers/dates/addresses
        if re.match(r'^[\d\s\.\-/,]+$', stripped):
            continue

        # Skip lines that look like metadata
        if re.match(r'^(Title|Author|Translator|Release Date|Language|ISBN|ISSN):', stripped, re.IGNORECASE):
            continue

        # Skip lines that are just special characters and whitespace
        if re.match(r'^[\s\*\-_=\.\,\:\;\!\?\(\)\[\]\{\}\'\"·]+$', stripped):
            continue

        # Skip very short lines that are just initials or abbreviations
        if len(stripped) <= 3 and not re.search(r'[aeiouĉĝĥĵŝŭ]', stripped, re.IGNORECASE):
            continue

        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def remove_front_matter(text: str) -> str:
    """Remove title pages, publication info, and other front matter."""
    lines = text.split('\n')

    # Look for the start of actual content
    # Usually after "ANTAŬPAROLO" or first chapter marker or substantial paragraph
    content_start = 0
    in_front_matter = True
    consecutive_short = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Check for explicit content start markers
        if re.match(r'^(ANTAŬPAROLO|ĈAPITRO|PARTO|LIBRO)\b', stripped, re.IGNORECASE):
            content_start = i
            break

        # Count consecutive short lines (title pages have many)
        if len(stripped) < 50:
            consecutive_short += 1
        else:
            # If we hit a long paragraph and we've passed the front matter pattern
            if consecutive_short > 5 and len(stripped) > 100:
                content_start = i
                break
            consecutive_short = 0

        # If we've scanned 200 lines, just start from there
        if i > 200:
            content_start = 0  # Keep everything
            break

    return '\n'.join(lines[content_start:])


def handle_wikipedia(text: str) -> str:
    """Special handling for Wikipedia dump (one giant line with articles).

    Splits into articles and outputs JSONL format with article boundaries preserved.
    Each line is: {"source": "wikipedia", "article": "Title", "text": "content..."}
    """
    import json as json_module

    # First, check if this looks like a Wikipedia dump (one very long line)
    lines = text.split('\n')
    if len(lines) < 10 and len(text) > 100000:
        # This is a single-line dump, split into articles
        # Wikipedia articles end with Kategorio: tags
        # Use findall to find all Kategorio: positions, then split between them

        # Find all Kategorio: tag positions
        kategorio_pattern = re.compile(r'Kategorio:[A-ZĈĜĤĴŜŬa-zĉĝĥĵŝŭ\-]+')
        matches = list(kategorio_pattern.finditer(text))

        if not matches:
            # No Kategorio tags found, treat as single article
            return text

        # Split text into articles based on Kategorio positions
        # Articles typically have multiple Kategorio tags at the end
        # Look for gaps between Kategorio tags (new article starts)
        raw_articles = []
        prev_end = 0
        last_kategorio_end = 0

        for i, match in enumerate(matches):
            # If there's a significant gap since last Kategorio,
            # this might be a new article's Kategorio
            if match.start() - last_kategorio_end > 200 and last_kategorio_end > 0:
                # Extract the article from prev_end to just before this match
                article_text = text[prev_end:match.start()].strip()
                if len(article_text) > 50:
                    raw_articles.append(article_text)
                prev_end = match.start()
            last_kategorio_end = match.end()

        # Don't forget the last article
        if prev_end < len(text):
            article_text = text[prev_end:].strip()
            if len(article_text) > 50:
                raw_articles.append(article_text)

        result_lines = []

        for article_text in raw_articles:
            if len(article_text) < 50:
                continue

            # Try to extract article title
            # Wikipedia articles typically start with the subject name, then description
            # Pattern: "SUBJECT (definition)" or "SUBJECT estas..." or "SUBJECT, ..."
            title = "Unknown"

            # First, clean the start of text from any leftover Kategorio tags
            clean_start = re.sub(r'^Kategorio:[^\s]+\s*', '', article_text).strip()

            # Try various patterns to extract title
            # Pattern 1: Name followed by parenthetical (e.g., "AIM (angla mallongigo...")
            title_match = re.match(r'^([A-ZĈĜĤĴŜŬ][A-ZĈĜĤĴŜŬa-zĉĝĥĵŝŭ0-9\s\-\']+?)(?:\s*\(|\s+estas\s|\s*,)', clean_start)
            if title_match:
                title = title_match.group(1).strip()

            # Pattern 2: Just capitalized words at start if no other match
            if title == "Unknown" or len(title) > 60:
                # Try to get first 1-4 capitalized words
                words_match = re.match(r'^([A-ZĈĜĤĴŜŬ][A-ZĈĜĤĴŜŬa-zĉĝĥĵŝŭ0-9\-\']*(?:\s+[A-ZĈĜĤĴŜŬ][A-ZĈĜĤĴŜŬa-zĉĝĥĵŝŭ0-9\-\']*){0,3})', clean_start)
                if words_match:
                    title = words_match.group(1).strip()

            # Validate title: skip if contains wiki markup or is too long
            if '==' in title or '{{' in title or '[[' in title or len(title) > 60:
                title = "Unknown"

            # Use first word as fallback if still Unknown
            if title == "Unknown":
                first_word = clean_start.split()[0] if clean_start.split() else "Unknown"
                if len(first_word) > 2 and first_word[0].isupper():
                    title = first_word

            # Clean the article text
            clean_text = article_text

            # Remove wiki markup - be thorough!
            clean_text = re.sub(r'\[\[[^\]]*\]\]', '', clean_text)  # [[links]]
            clean_text = re.sub(r'\[+', '', clean_text)              # leftover [[
            clean_text = re.sub(r'\]+', '', clean_text)              # leftover ]]
            clean_text = re.sub(r'\{\{[^\}]*\}\}', '', clean_text)  # {{templates}}
            clean_text = re.sub(r'\{+', '', clean_text)              # leftover {{
            clean_text = re.sub(r'\}+', '', clean_text)              # leftover }}
            clean_text = re.sub(r'==+[^=]+==+', '', clean_text)     # == headers ==
            clean_text = re.sub(r'=+', '', clean_text)               # leftover ==
            clean_text = re.sub(r'\*\s*\[', '', clean_text)         # * [ list items
            clean_text = re.sub(r'^\s*\*\s*', '', clean_text, flags=re.MULTILINE)  # bullet points
            clean_text = re.sub(r'https?://\S+', '', clean_text)    # URLs
            clean_text = re.sub(r'www\.\S+', '', clean_text)        # www URLs
            clean_text = re.sub(r'\S+\.com\b', '', clean_text)      # .com domains
            clean_text = re.sub(r'\S+\.org\b', '', clean_text)      # .org domains
            clean_text = re.sub(r'Kategorio:[^\s]+', '', clean_text)  # Category tags
            clean_text = re.sub(r'#\S+', '', clean_text)            # hashtags/anchors
            clean_text = re.sub(r'__[A-Z]+__', '', clean_text)      # __TOC__, __NOTOC__
            clean_text = re.sub(r'&[a-z]+;', '', clean_text)        # HTML entities &nbsp;
            clean_text = re.sub(r'<[^>]+>', '', clean_text)         # HTML tags
            clean_text = re.sub(r'File:[^\s]+', '', clean_text)     # File: references
            clean_text = re.sub(r'Image:[^\s]+', '', clean_text)    # Image: references
            clean_text = re.sub(r'\|[^\s|]+', '', clean_text)       # pipe parameters

            # Remove markdown artifacts (asterisks and underscores)
            clean_text = re.sub(r'\*+', '', clean_text)             # Remove all asterisks
            clean_text = re.sub(r'(?<![a-zĉĝĥĵŝŭ])_+(?![a-zĉĝĥĵŝŭ])', '', clean_text, flags=re.IGNORECASE)  # Remove standalone underscores

            # Normalize whitespace
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()

            # Skip if too short after cleaning
            if len(clean_text) < 50:
                continue

            # Output as JSONL
            result_lines.append(json_module.dumps({
                "source": "wikipedia",
                "article": title[:100],  # Truncate long titles
                "text": clean_text
            }, ensure_ascii=False))

        return '\n'.join(result_lines)

    return text


def clean_text(text: str, filename: str) -> str:
    """Apply all cleaning steps to a text."""
    original_len = len(text)
    is_wikipedia = 'wikipedia' in filename.lower()

    # Step 0: Special handling for Wikipedia - returns JSONL, skip other cleaning
    if is_wikipedia:
        text = handle_wikipedia(text)
        # Wikipedia handler returns JSONL format - just clean and return
        return text.strip() + '\n'

    # Step 1: Remove Gutenberg header/footer
    text = remove_gutenberg_header_footer(text)

    # Step 2: Remove markdown formatting
    text = remove_markdown_formatting(text)

    # Step 3: Remove formatting artifacts
    text = remove_formatting_artifacts(text)

    # Step 4: Fix hyphenation
    text = fix_hyphenation(text)

    # Step 5: Remove non-Esperanto content
    text = remove_non_esperanto_content(text)

    # Step 6: Normalize whitespace
    text = normalize_whitespace(text)

    # Step 7: Remove front matter (title pages, etc.)
    text = remove_front_matter(text)

    # Final cleanup: strip and add final newline
    text = text.strip() + '\n'

    return text


def process_file(input_path: Path, output_dir: Path, verbose: bool = False) -> Optional[dict]:
    """Process a single file and write cleaned version."""
    filename = input_path.name

    # Skip files that shouldn't be processed
    if filename in SKIP_FILES:
        if verbose:
            print(f"  Skipping {filename} (in skip list)")
        return None

    # Read input file
    try:
        text = input_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        try:
            text = input_path.read_text(encoding='latin-1')
        except Exception as e:
            print(f"  Error reading {filename}: {e}")
            return None

    original_len = len(text)
    original_lines = len(text.split('\n'))

    # Clean the text
    cleaned = clean_text(text, filename)

    cleaned_len = len(cleaned)
    cleaned_lines = len(cleaned.split('\n'))

    # Write output
    # Change extension to .txt if it's .md
    output_name = filename.replace('.md', '.txt')
    output_path = output_dir / output_name
    output_path.write_text(cleaned, encoding='utf-8')

    stats = {
        'filename': filename,
        'original_chars': original_len,
        'cleaned_chars': cleaned_len,
        'original_lines': original_lines,
        'cleaned_lines': cleaned_lines,
        'reduction_pct': round((1 - cleaned_len / original_len) * 100, 1) if original_len > 0 else 0,
    }

    if verbose:
        print(f"  {filename}: {original_lines} -> {cleaned_lines} lines ({stats['reduction_pct']}% reduction)")

    return stats


def main():
    parser = argparse.ArgumentParser(description='Clean corpus texts for parser training')
    parser.add_argument('--input', '-i', type=Path, required=True,
                        help='Input directory with raw texts')
    parser.add_argument('--output', '-o', type=Path, required=True,
                        help='Output directory for cleaned texts')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Find all text files
    input_files = list(args.input.glob('*.txt')) + list(args.input.glob('*.md'))
    input_files = sorted(input_files)

    print(f"Processing {len(input_files)} files from {args.input}")
    print(f"Output directory: {args.output}")
    print()

    all_stats = []
    skipped = 0

    for input_file in input_files:
        stats = process_file(input_file, args.output, args.verbose)
        if stats:
            all_stats.append(stats)
        else:
            skipped += 1

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files processed: {len(all_stats)}")
    print(f"Files skipped: {skipped}")

    if all_stats:
        total_original = sum(s['original_chars'] for s in all_stats)
        total_cleaned = sum(s['cleaned_chars'] for s in all_stats)
        total_original_lines = sum(s['original_lines'] for s in all_stats)
        total_cleaned_lines = sum(s['cleaned_lines'] for s in all_stats)

        print(f"Total original chars: {total_original:,}")
        print(f"Total cleaned chars: {total_cleaned:,}")
        print(f"Total reduction: {(1 - total_cleaned / total_original) * 100:.1f}%")
        print(f"Total original lines: {total_original_lines:,}")
        print(f"Total cleaned lines: {total_cleaned_lines:,}")

    print()
    print(f"Cleaned files written to: {args.output}")


if __name__ == '__main__':
    main()
