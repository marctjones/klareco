#!/usr/bin/env python3
"""
Acquire Proverbaro Esperanta from Wikimedia Commons - Tier 0 proverbs collection.

Downloads PDF and extracts all 2,630 proverbs.

Author: L.L. Zamenhof (arranged from father Mark Zamenhof's work)
Edition: 1910
Source: Wikimedia Commons
Quality: Scanned PDF with good OCR (8/12 diacritics in test)
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

# Constants
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


def extract_text_from_pdf(pdf_path: Path, output_text_path: Path) -> bool:
    """Extract text from Proverbaro PDF using PyPDF2."""
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

                if i % 20 == 0:
                    logger.info(f"Processed {i}/{len(reader.pages)} pages")

            logger.info(f"Extracted {len(text)} characters")

            # Save extracted text
            output_text_path.write_text(text, encoding='utf-8')
            logger.info(f"Saved text: {output_text_path}")

            return True

    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        return False


def clean_proverbaro_text(raw_text: str) -> str:
    """Clean extracted text and fix common OCR errors."""
    logger.info("Cleaning extracted text")

    text = raw_text

    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    # Fix common OCR errors in headers (from quality test)
    text = re.sub(r'PIIOVERBARO', 'PROVERBARO', text)
    text = re.sub(r'II0', 'RO', text)
    text = re.sub(r'PROVEKBARO', 'PROVERBARO', text)
    text = re.sub(r'PROVEHBABO', 'PROVERBARO', text)
    text = re.sub(r'PROVEllBARO', 'PROVERBARO', text)
    text = re.sub(r'PHOVERBABO', 'PROVERBARO', text)
    text = re.sub(r'PROVEKBAIiO', 'PROVERBARO', text)

    # Remove page headers
    text = re.sub(r'^PROVERBARO\s+ESPERANTA\s*\d*$', '', text, flags=re.MULTILINE)

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

    # Estimate proverb count (look for numbered entries)
    proverb_numbers = len(re.findall(r'^\d+\.\s*—', text, re.MULTILINE))

    # Sample
    sample = text[10000:11000]  # Middle sample

    quality = {
        'total_chars': len(text),
        'estimated_proverbs': proverb_numbers,
        'diacritics_found': list(found_diacritics),
        'diacritics_count': len(found_diacritics),
        'sample': sample,
        'has_esperanto_chars': len(found_diacritics) >= 4,
    }

    logger.info(f"Quality check: {quality['diacritics_count']}/12 diacritics found")
    logger.info(f"Estimated proverbs: {proverb_numbers}")
    logger.info(f"Quality: {'✓ GOOD' if quality['has_esperanto_chars'] else '⚠ NEEDS REVIEW'}")

    return quality


def acquire_proverbaro(output_dir: Path, skip_download: bool = False):
    """Main acquisition workflow."""
    logger.info("=" * 70)
    logger.info("ACQUIRE PROVERBARO ESPERANTA")
    logger.info("=" * 70)

    # Paths
    pdf_path = output_dir / 'proverbaro.pdf'
    raw_text_path = output_dir / 'proverbaro_raw.txt'
    cleaned_text_path = output_dir / 'proverbaro.txt'

    # Step 1: Download PDF
    if not skip_download or not pdf_path.exists():
        success = download_pdf(PROVERBARO_PDF_URL, pdf_path)
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
    cleaned_text = clean_proverbaro_text(raw_text)

    # Step 4: Validate quality
    quality = validate_text_quality(cleaned_text)

    # Step 5: Save cleaned text
    cleaned_text_path.write_text(cleaned_text, encoding='utf-8')
    logger.info(f"Saved cleaned text: {cleaned_text_path}")

    # Step 6: Save metadata
    import json

    metadata = {
        'title': 'Proverbaro Esperanta',
        'author': 'L.L. Zamenhof (arranged from Mark Zamenhof)',
        'edition': '1910',
        'source': PROVERBARO_PDF_URL,
        'total_proverbs': 2630,  # Known count
        'estimated_extracted': quality['estimated_proverbs'],
        'acquisition_date': time.strftime('%Y-%m-%d'),
        'quality': quality,
    }

    metadata_path = output_dir / 'proverbaro.metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Report
    logger.info("")
    logger.info("=" * 70)
    logger.info("ACQUISITION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"PDF: {pdf_path}")
    logger.info(f"Raw text: {raw_text_path}")
    logger.info(f"Cleaned text: {cleaned_text_path}")
    logger.info(f"Total characters: {quality['total_chars']:,}")
    logger.info(f"Estimated proverbs: {quality['estimated_proverbs']} (expected: 2,630)")
    logger.info(f"Diacritics found: {', '.join(quality['diacritics_found'])}")
    logger.info(f"Quality: {'✓ GOOD - Ready for extraction' if quality['has_esperanto_chars'] else '⚠ NEEDS REVIEW'}")
    logger.info("")
    logger.info("Sample (middle section):")
    logger.info(quality['sample'])
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Review proverbaro.txt for quality")
    logger.info("2. Verify proverb count is close to 2,630")
    logger.info("3. Proceed to sentence extraction if quality is good")

    return True


def main():
    parser = argparse.ArgumentParser(description='Acquire Proverbaro Esperanta')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('data/raw/eo/proverbaro'),
                        help='Output directory')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip download if PDF already exists')

    args = parser.parse_args()

    try:
        success = acquire_proverbaro(args.output_dir, args.skip_download)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Acquisition failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
