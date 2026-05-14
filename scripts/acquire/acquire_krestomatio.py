#!/usr/bin/env python3
"""
Acquire Fundamenta Krestomatio from Project Gutenberg.

Source: Project Gutenberg #8224
Author: L.L. Zamenhof
Year: 1903 (first edition), various reprints
Format: Plain text UTF-8 (born-digital, PGDP proofread)
Size: 855 KB
Quality: Excellent (same source as Alice in Wonderland)
"""

import argparse
import logging
from pathlib import Path

import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
KRESTOMATIO_URL = "https://www.gutenberg.org/files/8224/8224-0.txt"
KRESTOMATIO_TITLE = "Fundamenta Krestomatio"


def download_krestomatio(url: str, output_path: Path) -> bool:
    """Download Fundamenta Krestomatio from Project Gutenberg."""
    logger.info(f"Downloading {KRESTOMATIO_TITLE} from {url}")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Ensure UTF-8 encoding
        response.encoding = 'utf-8'
        text = response.text

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding='utf-8')

        logger.info(f"Downloaded: {output_path} ({len(text):,} chars)")
        return True

    except Exception as e:
        logger.error(f"Failed to download: {e}")
        return False


def validate_quality(text: str) -> dict:
    """Validate text quality (diacritics, content)."""
    logger.info("Validating text quality")

    # Check for Esperanto diacritics
    diacritics = {'ĉ', 'ĝ', 'ĥ', 'ĵ', 'ŝ', 'ŭ', 'Ĉ', 'Ĝ', 'Ĥ', 'Ĵ', 'Ŝ', 'Ŭ'}
    found_diacritics = set()
    for char in text:
        if char in diacritics:
            found_diacritics.add(char)

    # Count words (rough estimate)
    word_count = len(text.split())

    # Sample for manual inspection
    sample = text[10000:11000]  # Middle sample

    quality = {
        'total_chars': len(text),
        'word_count': word_count,
        'diacritics_found': sorted(found_diacritics),
        'diacritics_count': len(found_diacritics),
        'sample': sample,
        'has_esperanto_chars': len(found_diacritics) >= 6,
    }

    logger.info(f"Quality check: {quality['diacritics_count']}/12 diacritics found")
    logger.info(f"Word count: ~{word_count:,}")
    logger.info(f"Quality: {'✓ GOOD' if quality['has_esperanto_chars'] else '⚠ NEEDS REVIEW'}")

    return quality


def acquire_krestomatio(output_dir: Path):
    """Main acquisition workflow."""
    logger.info("=" * 70)
    logger.info("ACQUIRE FUNDAMENTA KRESTOMATIO")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Source: Project Gutenberg #8224")
    logger.info("Author: L.L. Zamenhof")
    logger.info("Quality: Born-digital (PGDP proofread)")
    logger.info("")

    # Paths
    raw_path = output_dir / 'krestomatio_raw.txt'
    output_path = output_dir / 'krestomatio.txt'

    # Step 1: Download
    success = download_krestomatio(KRESTOMATIO_URL, raw_path)
    if not success:
        logger.error("Download failed, exiting")
        return False

    # Step 2: Read and validate
    raw_text = raw_path.read_text(encoding='utf-8')
    quality = validate_quality(raw_text)

    # Step 3: Copy to output (will be cleaned by clean_gutenberg.py later)
    output_path.write_text(raw_text, encoding='utf-8')
    logger.info(f"Saved: {output_path}")

    # Report
    logger.info("")
    logger.info("=" * 70)
    logger.info("ACQUISITION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Raw text: {raw_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Total characters: {quality['total_chars']:,}")
    logger.info(f"Word count: ~{quality['word_count']:,}")
    logger.info(f"Diacritics: {', '.join(quality['diacritics_found'])}")
    logger.info(f"Quality: {'✓ EXCELLENT' if quality['has_esperanto_chars'] else '⚠ NEEDS REVIEW'}")
    logger.info("")
    logger.info("Sample (middle section):")
    logger.info(quality['sample'])
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Clean with: python scripts/clean/clean_gutenberg.py")
    logger.info("  2. Extract sentences: python scripts/extract/extract_tier0_literary.py")
    logger.info("  3. Integrate into corpus")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Acquire Fundamenta Krestomatio from Project Gutenberg'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/raw/eo/gutenberg'),
        help='Output directory'
    )

    args = parser.parse_args()

    try:
        success = acquire_krestomatio(args.output_dir)
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Acquisition failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
