#!/usr/bin/env python3
"""
Test quality of Proverbaro Esperanta scanned PDF before full acquisition.

Downloads sample pages and checks for OCR errors.

Source: Wikimedia Commons (1910 scan)
"""

import argparse
import logging
import random
import re
import subprocess
import sys
from pathlib import Path

import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
# Alternative URL found - different filename format
PROVERBARO_PDF_URL = "https://upload.wikimedia.org/wikipedia/commons/c/cd/Zamenhof_L._L._-_Proverbaro_Esperanta,_1910.pdf"

def download_pdf(url: str, output_path: Path) -> bool:
    """Download Proverbaro PDF from Wikimedia."""
    logger.info(f"Downloading Proverbaro PDF from {url}")

    try:
        # Add User-Agent header to avoid 403 Forbidden from Wikimedia
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(url, stream=True, timeout=30, headers=headers)
        response.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Downloaded PDF: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return True

    except Exception as e:
        logger.error(f"Failed to download PDF: {e}")
        return False


def check_pandoc_available() -> bool:
    """Check if pandoc is installed."""
    try:
        result = subprocess.run(['pandoc', '--version'],
                                capture_output=True,
                                text=True,
                                timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def extract_sample_pages(pdf_path: Path, output_text_path: Path, start_page: int = 10, num_pages: int = 20) -> bool:
    """Extract sample pages from PDF using PyPDF2."""
    logger.info(f"Extracting sample pages {start_page}-{start_page + num_pages} from {pdf_path}")

    try:
        from PyPDF2 import PdfReader

        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            text = ''

            total_pages = len(reader.pages)
            logger.info(f"PDF has {total_pages} pages")

            # Extract sample pages
            end_page = min(start_page + num_pages, total_pages)
            for i in range(start_page, end_page):
                page_text = reader.pages[i].extract_text()
                text += page_text + '\n\n'

            logger.info(f"Extracted {len(text)} characters from pages {start_page}-{end_page}")

            # Save extracted text
            output_text_path.write_text(text, encoding='utf-8')
            logger.info(f"Saved sample: {output_text_path}")

            return True

    except Exception as e:
        logger.error(f"Failed to extract sample: {e}")
        return False


def extract_proverbs(text: str) -> list:
    """Extract individual proverbs from text."""
    # Proverbs are typically numbered or on separate lines
    # Try to split on common patterns

    # Split on double newlines
    potential_proverbs = [p.strip() for p in text.split('\n\n') if p.strip()]

    # Filter out very short lines (headers, page numbers)
    proverbs = [p for p in potential_proverbs if len(p) > 20]

    return proverbs


def check_ocr_errors(text: str) -> dict:
    """Check for common OCR errors."""
    errors = {
        'missing_diacritics': 0,
        'common_substitutions': [],
        'suspicious_patterns': [],
    }

    # Check for Esperanto diacritics
    diacritics = {'ĉ', 'ĝ', 'ĥ', 'ĵ', 'ŝ', 'ŭ', 'Ĉ', 'Ĝ', 'Ĥ', 'Ĵ', 'Ŝ', 'Ŭ'}
    found_diacritics = set()
    for char in text:
        if char in diacritics:
            found_diacritics.add(char)

    errors['diacritics_found'] = list(found_diacritics)
    errors['has_esperanto_chars'] = len(found_diacritics) >= 4

    # Common OCR substitutions (letters that look similar)
    # l→I, O→0, c→e, etc.
    suspicious_words = re.findall(r'\b[A-Z][I0]+[A-Z]*\b', text)  # Words with I or 0 that might be l or O
    if suspicious_words:
        errors['suspicious_patterns'].extend(suspicious_words[:10])  # Sample

    # Check for gibberish (long sequences without vowels)
    gibberish = re.findall(r'\b[bcdfghjklmnpqrstvwxz]{6,}\b', text, re.IGNORECASE)
    if gibberish:
        errors['suspicious_patterns'].extend(gibberish[:5])

    return errors


def calculate_quality_score(errors: dict, sample_text: str) -> dict:
    """Calculate overall quality score."""
    score = 100  # Start at perfect

    # Penalize missing diacritics heavily (critical for Esperanto)
    if not errors['has_esperanto_chars']:
        score -= 50
        quality = "POOR - Missing Esperanto diacritics"
    elif len(errors['diacritics_found']) < 6:
        score -= 20
        quality = "FAIR - Some diacritics missing"
    else:
        quality = "GOOD - Diacritics present"

    # Penalize suspicious patterns
    if len(errors['suspicious_patterns']) > 10:
        score -= 30
        quality = "POOR - Many OCR errors detected"
    elif len(errors['suspicious_patterns']) > 5:
        score -= 15

    # Final verdict
    if score >= 80:
        verdict = "✓ ACCEPTABLE - Proceed with acquisition"
    elif score >= 60:
        verdict = "⚠ MARGINAL - Manual correction required"
    else:
        verdict = "✗ UNACCEPTABLE - Find alternative source"

    return {
        'score': score,
        'quality': quality,
        'verdict': verdict,
    }


def test_proverbaro_quality(output_dir: Path):
    """Main quality testing workflow."""
    logger.info("=" * 70)
    logger.info("TEST PROVERBARO QUALITY")
    logger.info("=" * 70)

    # Paths
    pdf_path = output_dir / 'proverbaro_test.pdf'
    sample_text_path = output_dir / 'proverbaro_sample.txt'

    # Step 1: Download PDF
    success = download_pdf(PROVERBARO_PDF_URL, pdf_path)
    if not success:
        logger.error("Download failed, exiting")
        return False

    # Step 2: Extract sample using PyPDF2
    success = extract_sample_pages(pdf_path, sample_text_path)
    if not success:
        logger.error("Sample extraction failed, exiting")
        return False

    # Step 3: Read sample
    sample_text = sample_text_path.read_text(encoding='utf-8')

    # Step 4: Extract proverbs
    proverbs = extract_proverbs(sample_text)
    logger.info(f"Extracted {len(proverbs)} potential proverbs")

    # Step 5: Check for OCR errors
    errors = check_ocr_errors(sample_text)

    # Step 6: Calculate quality score
    quality_result = calculate_quality_score(errors, sample_text)

    # Step 7: Display sample proverbs for manual inspection
    logger.info("")
    logger.info("=" * 70)
    logger.info("MANUAL INSPECTION - Sample Proverbs (first 20)")
    logger.info("=" * 70)

    sample_proverbs = random.sample(proverbs, min(20, len(proverbs)))
    for i, proverb in enumerate(sample_proverbs, 1):
        logger.info(f"{i}. {proverb[:100]}{'...' if len(proverb) > 100 else ''}")

    # Report
    logger.info("")
    logger.info("=" * 70)
    logger.info("QUALITY TEST RESULTS")
    logger.info("=" * 70)
    logger.info(f"Sample size: {len(sample_text):,} characters")
    logger.info(f"Proverbs found: {len(proverbs)}")
    logger.info(f"Diacritics found: {', '.join(errors['diacritics_found'])}")
    logger.info(f"Suspicious patterns: {len(errors['suspicious_patterns'])}")
    if errors['suspicious_patterns']:
        logger.info(f"  Examples: {', '.join(errors['suspicious_patterns'][:5])}")
    logger.info("")
    logger.info(f"Quality: {quality_result['quality']}")
    logger.info(f"Score: {quality_result['score']}/100")
    logger.info(f"Verdict: {quality_result['verdict']}")
    logger.info("")
    logger.info("Next steps:")
    if quality_result['score'] >= 80:
        logger.info("✓ Proceed with full acquisition (scripts/acquire_proverbaro.py)")
    elif quality_result['score'] >= 60:
        logger.info("⚠ Proceed with caution - plan for manual correction")
        logger.info("  - Extract all proverbs")
        logger.info("  - Create correction script for common OCR errors")
        logger.info("  - Manual review of 10% sample")
    else:
        logger.info("✗ Do NOT proceed with this source")
        logger.info("  - Check Tekstaro.com for digital version")
        logger.info("  - Check Gutenberg for transcribed version")
        logger.info("  - Consider re-typing or finding alternative")

    return True


def main():
    parser = argparse.ArgumentParser(description='Test Proverbaro Esperanta quality')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('data/raw/eo/proverbaro'),
                        help='Output directory')

    args = parser.parse_args()

    try:
        success = test_proverbaro_quality(args.output_dir)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Quality test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
