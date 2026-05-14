#!/usr/bin/env python3
"""
Acquire PMEG (Plena Manlibro de Esperanta Gramatiko) - Tier 0 authoritative grammar.

Downloads PDF and extracts structured markdown using pandoc.

Author: Bertilo Wennergren
Edition: v15.5 (2024)
License: CC BY-SA 4.0
Source: https://bertilow.com/pmeg/elshutebla/pmeg15.5.pdf
"""

import argparse
import logging
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
PMEG_PDF_URL = "https://h.bertilow.com/pmeg/elshutebla/pmeg_15.5.pdf"

def download_pdf(url: str, output_path: Path) -> bool:
    """Download PMEG PDF from URL."""
    logger.info(f"Downloading PMEG PDF from {url}")

    try:
        response = requests.get(url, stream=True, timeout=30)
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


def extract_text_from_pdf(pdf_path: Path, output_text_path: Path) -> bool:
    """Extract text from PMEG PDF using PyPDF2."""
    logger.info(f"Extracting text from {pdf_path} using PyPDF2")

    try:
        from PyPDF2 import PdfReader

        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            text = ''

            logger.info(f"PDF has {len(reader.pages)} pages")

            for i, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                text += page_text + '\n\n'

                if i % 50 == 0:
                    logger.info(f"Processed {i}/{len(reader.pages)} pages")

            logger.info(f"Extracted {len(text)} characters")

            # Save extracted text
            output_text_path.write_text(text, encoding='utf-8')
            logger.info(f"Saved text: {output_text_path}")

            return True

    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        return False


def clean_pmeg_text(raw_text: str) -> str:
    """Clean extracted text."""
    logger.info("Cleaning extracted text")

    text = raw_text

    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    # Remove page numbers (common pattern: just a number on a line)
    text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)

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

    # Sample first 1000 characters (PMEG starts with title page)
    sample = text[:1000]

    quality = {
        'total_chars': len(text),
        'diacritics_found': list(found_diacritics),
        'diacritics_count': len(found_diacritics),
        'sample': sample,
        'has_esperanto_chars': len(found_diacritics) >= 4,
    }

    logger.info(f"Quality check: {quality['diacritics_count']}/12 diacritics found")
    logger.info(f"Esperanto text detected: {quality['has_esperanto_chars']}")

    if not quality['has_esperanto_chars']:
        logger.warning("WARNING: Few or no Esperanto diacritics found")

    return quality


def acquire_pmeg(output_dir: Path, skip_download: bool = False):
    """Main acquisition workflow."""
    logger.info("=" * 70)
    logger.info("ACQUIRE PMEG (Plena Manlibro de Esperanta Gramatiko)")
    logger.info("=" * 70)

    # Paths
    pdf_path = output_dir / 'pmeg.pdf'
    raw_text_path = output_dir / 'pmeg_raw.txt'
    cleaned_text_path = output_dir / 'pmeg.txt'

    # Step 1: Download PDF
    if not skip_download or not pdf_path.exists():
        success = download_pdf(PMEG_PDF_URL, pdf_path)
        if not success:
            logger.error("Download failed, exiting")
            return False
    else:
        logger.info(f"Using existing PDF: {pdf_path}")

    # Step 2: Extract text using PyPDF2
    success = extract_text_from_pdf(pdf_path, raw_text_path)
    if not success:
        logger.error("Text extraction failed, exiting")
        return False

    # Read extracted text
    raw_text = raw_text_path.read_text(encoding='utf-8')

    # Step 3: Clean text
    cleaned_text = clean_pmeg_text(raw_text)

    # Step 4: Validate quality
    quality = validate_text_quality(cleaned_text)

    # Step 5: Save cleaned text
    cleaned_text_path.write_text(cleaned_text, encoding='utf-8')
    logger.info(f"Saved cleaned text: {cleaned_text_path}")

    # Report
    logger.info("")
    logger.info("=" * 70)
    logger.info("ACQUISITION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"PDF: {pdf_path}")
    logger.info(f"Raw text: {raw_text_path}")
    logger.info(f"Cleaned text: {cleaned_text_path}")
    logger.info(f"Total characters: {quality['total_chars']:,}")
    logger.info(f"Diacritics found: {', '.join(quality['diacritics_found'])}")
    logger.info(f"Quality: {'✓ GOOD' if quality['has_esperanto_chars'] else '⚠ NEEDS REVIEW'}")
    logger.info("")
    logger.info("Sample (first 1000 chars):")
    logger.info(quality['sample'])
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Review pmeg.txt for quality (check for Esperanto diacritics)")
    logger.info("2. If quality is good, proceed to sentence extraction")

    return True


def main():
    parser = argparse.ArgumentParser(description='Acquire PMEG (Plena Manlibro de Esperanta Gramatiko)')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('data/raw/eo/pmeg'),
                        help='Output directory')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip download if PDF already exists')

    args = parser.parse_args()

    try:
        success = acquire_pmeg(args.output_dir, args.skip_download)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Acquisition failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
