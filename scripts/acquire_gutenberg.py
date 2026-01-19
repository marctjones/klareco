#!/usr/bin/env python3
"""
Acquire Esperanto literature from Project Gutenberg - Tier 1 literary texts.

Downloads selected literary works (not grammars) from Project Gutenberg.

Quality: Born-digital, manually proofread by PGDP team
License: Public domain
Source: https://www.gutenberg.org/
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project Gutenberg Esperanto literary works (excluding grammars/textbooks)
# Source: gutenberg.org/ebooks/bookshelf/98 (Esperanto shelf)
LITERARY_WORKS = [
    {
        'id': 17482,
        'title': 'La Aventuroj de Alicio en Mirlando',
        'author': 'Lewis Carroll',
        'translator': 'E.L. Kearney',
        'url': 'https://www.gutenberg.org/files/17482/17482-0.txt',
        'tier': 1,
    },
    {
        'id': 27915,
        'title': 'Fabeloj de Andersen',
        'author': 'Hans Christian Andersen',
        'translator': 'F. Skeel-Giörling',
        'url': 'https://www.gutenberg.org/files/27915/27915-0.txt',
        'tier': 1,
    },
    # Add more as needed
]


def download_text(url: str, output_path: Path) -> bool:
    """Download text file from Project Gutenberg."""
    logger.info(f"Downloading from {url}")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        response.encoding = 'utf-8'

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save original text
        output_path.write_text(response.text, encoding='utf-8')

        logger.info(f"Downloaded: {output_path} ({len(response.text)} characters)")
        return True

    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def clean_gutenberg_text(text: str, title: str) -> str:
    """
    Clean Project Gutenberg text by removing headers and footers.

    Gutenberg texts have standard headers/footers that should be removed.
    """
    logger.info(f"Cleaning Gutenberg text: {title}")

    # Find start of actual content (after Gutenberg header)
    start_markers = [
        '*** START OF THIS PROJECT GUTENBERG',
        '*** START OF THE PROJECT GUTENBERG',
        '*END*THE SMALL PRINT',
    ]

    start_pos = 0
    for marker in start_markers:
        pos = text.find(marker)
        if pos != -1:
            # Find end of line after marker
            start_pos = text.find('\n', pos) + 1
            break

    # Find end of actual content (before Gutenberg footer)
    end_markers = [
        '*** END OF THIS PROJECT GUTENBERG',
        '*** END OF THE PROJECT GUTENBERG',
        'End of Project Gutenberg',
    ]

    end_pos = len(text)
    for marker in end_markers:
        pos = text.find(marker)
        if pos != -1:
            end_pos = pos
            break

    # Extract content
    content = text[start_pos:end_pos].strip()

    # Remove excessive blank lines
    content = re.sub(r'\n{4,}', '\n\n\n', content)

    logger.info(f"Cleaned: {len(content)} characters (removed {len(text) - len(content)} header/footer chars)")

    return content


def validate_text_quality(text: str) -> dict:
    """Validate extraction quality."""
    # Check for Esperanto diacritics
    diacritics = {'ĉ', 'ĝ', 'ĥ', 'ĵ', 'ŝ', 'ŭ', 'Ĉ', 'Ĝ', 'Ĥ', 'Ĵ', 'Ŝ', 'Ŭ'}
    found_diacritics = set()
    for char in text:
        if char in diacritics:
            found_diacritics.add(char)

    # Count lines and words (rough estimate)
    lines = text.count('\n')
    words = len(text.split())

    quality = {
        'total_chars': len(text),
        'lines': lines,
        'words': words,
        'diacritics_found': list(found_diacritics),
        'diacritics_count': len(found_diacritics),
        'has_esperanto_chars': len(found_diacritics) >= 4,
    }

    return quality


def acquire_gutenberg_works(output_dir: Path, works_list: list = None):
    """Main acquisition workflow for multiple works."""
    logger.info("=" * 70)
    logger.info("ACQUIRE ESPERANTO LITERATURE FROM PROJECT GUTENBERG")
    logger.info("=" * 70)

    if works_list is None:
        works_list = LITERARY_WORKS

    logger.info(f"Acquiring {len(works_list)} literary works")
    logger.info("")

    successful = []
    failed = []

    for work in works_list:
        logger.info(f"Processing: {work['title']} by {work['author']}")

        # Create filename
        safe_title = re.sub(r'[^a-zA-Z0-9]+', '_', work['title'].lower())
        filename = f"gutenberg_{work['id']}_{safe_title}"

        raw_path = output_dir / f"{filename}_raw.txt"
        cleaned_path = output_dir / f"{filename}.txt"
        metadata_path = output_dir / f"{filename}.metadata.json"

        # Download
        success = download_text(work['url'], raw_path)
        if not success:
            failed.append(work)
            continue

        # Read and clean
        raw_text = raw_path.read_text(encoding='utf-8')
        cleaned_text = clean_gutenberg_text(raw_text, work['title'])

        # Validate
        quality = validate_text_quality(cleaned_text)

        # Save cleaned text
        cleaned_path.write_text(cleaned_text, encoding='utf-8')
        logger.info(f"Saved: {cleaned_path}")

        # Save metadata
        import json
        metadata = {
            'id': work['id'],
            'title': work['title'],
            'author': work['author'],
            'translator': work['translator'],
            'source': work['url'],
            'tier': work['tier'],
            'acquisition_date': time.strftime('%Y-%m-%d'),
            'quality': quality,
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Quality: {quality['words']:,} words, {quality['diacritics_count']}/12 diacritics")
        logger.info("")

        successful.append(work)

        # Be polite to Gutenberg servers
        time.sleep(2)

    # Report
    logger.info("=" * 70)
    logger.info("ACQUISITION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Successful: {len(successful)}/{len(works_list)}")
    if successful:
        logger.info("Acquired works:")
        for work in successful:
            logger.info(f"  ✓ {work['title']} ({work['id']})")

    if failed:
        logger.info("")
        logger.info(f"Failed: {len(failed)}")
        for work in failed:
            logger.info(f"  ✗ {work['title']} ({work['id']})")

    logger.info("")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Review cleaned texts for quality")
    logger.info("2. Proceed to extract_gutenberg.py for sentence extraction")

    return len(successful) > 0


def main():
    parser = argparse.ArgumentParser(description='Acquire Esperanto literature from Project Gutenberg')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('data/raw/eo/gutenberg'),
                        help='Output directory')
    parser.add_argument('--works', type=str, nargs='+',
                        help='Specific work IDs to download (default: all)')

    args = parser.parse_args()

    # Filter works if specific IDs requested
    works_list = LITERARY_WORKS
    if args.works:
        requested_ids = [int(w) for w in args.works]
        works_list = [w for w in LITERARY_WORKS if w['id'] in requested_ids]
        if not works_list:
            logger.error(f"No matching works found for IDs: {requested_ids}")
            sys.exit(1)

    try:
        success = acquire_gutenberg_works(args.output_dir, works_list)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Acquisition failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
