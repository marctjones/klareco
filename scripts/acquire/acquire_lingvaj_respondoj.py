#!/usr/bin/env python3
"""
Acquire Lingvaj Respondoj from Tekstaro.com - Tier 0 authoritative grammar Q&A.

Scrapes the HTML from Tekstaro.com and extracts clean text.

Author: L.L. Zamenhof
Edition: 7th edition (1990), edited by Gaston Waringhien
Source: https://tekstaro.com/t?nomo=lingvaj-respondoj
Quality: Born-digital, zero OCR errors
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
TEKSTARO_BASE_URL = "https://tekstaro.com/t"
LINGVAJ_RESPONDOJ_URL = f"{TEKSTARO_BASE_URL}?nomo=lingvaj-respondoj"

def fetch_page(url: str, params: dict = None) -> str:
    """Fetch page HTML from Tekstaro.com."""
    logger.info(f"Fetching: {url}")

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        response.encoding = 'utf-8'

        logger.info(f"Fetched {len(response.text)} characters")
        return response.text

    except Exception as e:
        logger.error(f"Failed to fetch page: {e}")
        return ""


def parse_lingvaj_respondoj(html: str) -> dict:
    """Parse Lingvaj Respondoj HTML and extract structured content."""
    logger.info("Parsing HTML content")

    soup = BeautifulSoup(html, 'html.parser')

    # Find the main content area
    # Tekstaro typically uses a specific structure - we'll extract all text
    main_content = soup.find('div', class_='teksto')
    if not main_content:
        # Fallback: get body content
        main_content = soup.find('body')

    if not main_content:
        logger.error("Could not find main content area")
        return {}

    # Extract text, preserving structure
    text_parts = []

    # Get all paragraphs, headings, and list items
    for element in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li', 'div']):
        text = element.get_text(separator=' ', strip=True)
        if text and len(text) > 10:  # Skip very short fragments
            text_parts.append(text)

    full_text = '\n\n'.join(text_parts)

    logger.info(f"Extracted {len(full_text)} characters from {len(text_parts)} elements")

    return {
        'title': 'Lingvaj Respondoj',
        'author': 'L.L. Zamenhof',
        'editor': 'Gaston Waringhien',
        'edition': '7th (1990)',
        'source': LINGVAJ_RESPONDOJ_URL,
        'text': full_text,
        'elements_count': len(text_parts),
    }


def clean_text(text: str) -> str:
    """Clean extracted text."""
    logger.info("Cleaning text")

    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    # Remove navigation elements (common patterns)
    text = re.sub(r'^\s*›.*$', '', text, flags=re.MULTILINE)  # Navigation arrows
    text = re.sub(r'^\s*\[.*?\]\s*$', '', text, flags=re.MULTILINE)  # [Links]

    # Normalize line breaks
    text = text.strip()

    logger.info(f"Cleaned text: {len(text)} characters")
    return text


def validate_text_quality(text: str) -> dict:
    """Validate extraction quality."""
    logger.info("Validating text quality")

    # Check for Esperanto diacritics
    diacritics = {'ĉ', 'ĝ', 'ĥ', 'ĵ', 'ŝ', 'ŭ', 'Ĉ', 'Ĝ', 'Ĥ', 'Ĵ', 'Ŝ', 'Ŭ'}
    found_diacritics = set()
    for char in text:
        if char in diacritics:
            found_diacritics.add(char)

    # Count common Esperanto words to verify it's actual content
    esperanto_markers = ['estas', 'kaj', 'tiu', 'kiu', 'estas', 'oni', 'nur', 'aŭ']
    marker_count = sum(1 for marker in esperanto_markers if marker in text.lower())

    # Sample first 1000 characters
    sample = text[:1000]

    quality = {
        'total_chars': len(text),
        'diacritics_found': list(found_diacritics),
        'diacritics_count': len(found_diacritics),
        'esperanto_markers': marker_count,
        'sample': sample,
        'has_esperanto_chars': len(found_diacritics) >= 4,
        'has_esperanto_content': marker_count >= 5,
    }

    logger.info(f"Quality check: {quality['diacritics_count']}/12 diacritics found")
    logger.info(f"Esperanto markers: {marker_count}/{len(esperanto_markers)}")
    logger.info(f"Quality: {'✓ GOOD' if quality['has_esperanto_chars'] and quality['has_esperanto_content'] else '⚠ NEEDS REVIEW'}")

    return quality


def save_metadata(data: dict, output_path: Path):
    """Save metadata as JSON."""
    import json

    metadata = {
        'title': data['title'],
        'author': data['author'],
        'editor': data['editor'],
        'edition': data['edition'],
        'source': data['source'],
        'elements_count': data['elements_count'],
        'acquisition_date': time.strftime('%Y-%m-%d'),
    }

    metadata_path = output_path.with_suffix('.metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved metadata: {metadata_path}")


def acquire_lingvaj_respondoj(output_dir: Path):
    """Main acquisition workflow."""
    logger.info("=" * 70)
    logger.info("ACQUIRE LINGVAJ RESPONDOJ FROM TEKSTARO.COM")
    logger.info("=" * 70)

    # Paths
    html_path = output_dir / 'lingvaj_respondoj.html'
    raw_text_path = output_dir / 'lingvaj_respondoj_raw.txt'
    cleaned_text_path = output_dir / 'lingvaj_respondoj.txt'

    # Step 1: Fetch HTML
    html = fetch_page(LINGVAJ_RESPONDOJ_URL)
    if not html:
        logger.error("Failed to fetch page, exiting")
        return False

    # Save HTML for reference
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding='utf-8')
    logger.info(f"Saved HTML: {html_path}")

    # Step 2: Parse content
    data = parse_lingvaj_respondoj(html)
    if not data or not data.get('text'):
        logger.error("Failed to parse content, exiting")
        return False

    # Save raw extracted text
    raw_text_path.write_text(data['text'], encoding='utf-8')
    logger.info(f"Saved raw text: {raw_text_path}")

    # Step 3: Clean text
    cleaned_text = clean_text(data['text'])

    # Step 4: Validate quality
    quality = validate_text_quality(cleaned_text)

    # Step 5: Save cleaned text
    cleaned_text_path.write_text(cleaned_text, encoding='utf-8')
    logger.info(f"Saved cleaned text: {cleaned_text_path}")

    # Step 6: Save metadata
    save_metadata(data, cleaned_text_path)

    # Report
    logger.info("")
    logger.info("=" * 70)
    logger.info("ACQUISITION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Title: {data['title']}")
    logger.info(f"Author: {data['author']}")
    logger.info(f"Editor: {data['editor']}")
    logger.info(f"Edition: {data['edition']}")
    logger.info(f"Source: {data['source']}")
    logger.info("")
    logger.info(f"HTML: {html_path}")
    logger.info(f"Raw text: {raw_text_path}")
    logger.info(f"Cleaned text: {cleaned_text_path}")
    logger.info(f"Total characters: {quality['total_chars']:,}")
    logger.info(f"Diacritics found: {', '.join(quality['diacritics_found'])}")
    logger.info(f"Quality: {'✓ EXCELLENT' if quality['has_esperanto_chars'] and quality['has_esperanto_content'] else '⚠ NEEDS REVIEW'}")
    logger.info("")
    logger.info("Sample (first 1000 chars):")
    logger.info(quality['sample'])
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Review lingvaj_respondoj.txt for quality")
    logger.info("2. Check that structure is preserved (sections, numbering)")
    logger.info("3. Proceed to extract_lingvaj_respondoj.py for sentence extraction")

    return True


def main():
    parser = argparse.ArgumentParser(description='Acquire Lingvaj Respondoj from Tekstaro.com')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('data/raw/eo/lingvaj_respondoj'),
                        help='Output directory')

    args = parser.parse_args()

    try:
        success = acquire_lingvaj_respondoj(args.output_dir)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Acquisition failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
