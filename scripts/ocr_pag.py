#!/usr/bin/env python3
"""
Apply Tesseract OCR to PAG PDF to recover Esperanto diacritics.

The PyPDF2 extraction resulted in 0 diacritics due to special font encoding
in the scanned PDF. Tesseract OCR should recover the proper Unicode characters.

Requires: tesseract-ocr with Esperanto language pack
Install: sudo apt install tesseract-ocr tesseract-ocr-epo
"""

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_tesseract_available() -> bool:
    """Check if Tesseract is installed."""
    try:
        result = subprocess.run(['tesseract', '--version'],
                                capture_output=True,
                                text=True,
                                timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def check_esperanto_support() -> bool:
    """Check if Esperanto language pack is installed."""
    try:
        result = subprocess.run(['tesseract', '--list-langs'],
                                capture_output=True,
                                text=True,
                                timeout=5)
        return 'epo' in result.stdout
    except Exception:
        return False


def ocr_pdf(pdf_path: Path, output_text_path: Path, language: str = 'epo') -> bool:
    """
    OCR a PDF using Tesseract.

    Args:
        pdf_path: Path to input PDF
        output_text_path: Path to output text file
        language: Tesseract language code (epo = Esperanto)
    """
    logger.info(f"OCR-ing {pdf_path} with Tesseract (language: {language})")

    if not check_tesseract_available():
        logger.error("Tesseract not found - please install: sudo apt install tesseract-ocr")
        return False

    if not check_esperanto_support():
        logger.error("Esperanto language pack not found - please install: sudo apt install tesseract-ocr-epo")
        logger.error("Falling back to English (may result in poor diacritic recognition)")
        language = 'eng'

    # Create temporary directory for intermediate files
    temp_dir = output_text_path.parent / 'ocr_temp'
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Tesseract requires output path without extension (it adds .txt automatically)
    output_base = temp_dir / output_text_path.stem

    try:
        # Run Tesseract OCR
        # -l: language
        # pdf: input format (Tesseract can read PDFs directly)
        result = subprocess.run(
            ['tesseract', str(pdf_path), str(output_base), '-l', language, 'pdf'],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout for large PDF
        )

        if result.returncode != 0:
            logger.error(f"Tesseract OCR failed: {result.stderr}")
            return False

        # Read OCR output
        ocr_output = (temp_dir / f"{output_text_path.stem}.txt")
        if not ocr_output.exists():
            logger.error(f"OCR output not found: {ocr_output}")
            return False

        text = ocr_output.read_text(encoding='utf-8')
        logger.info(f"OCR extracted {len(text)} characters")

        # Save to final location
        output_text_path.write_text(text, encoding='utf-8')
        logger.info(f"Saved OCR text: {output_text_path}")

        return True

    except subprocess.TimeoutExpired:
        logger.error("OCR timed out (>1 hour)")
        return False
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return False
    finally:
        # Cleanup temp files
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def validate_ocr_quality(text: str) -> dict:
    """Validate OCR quality."""
    logger.info("Validating OCR quality")

    # Check for Esperanto diacritics
    diacritics = {'ĉ', 'ĝ', 'ĥ', 'ĵ', 'ŝ', 'ŭ', 'Ĉ', 'Ĝ', 'Ĥ', 'Ĵ', 'Ŝ', 'Ŭ'}
    found_diacritics = set()
    for char in text:
        if char in diacritics:
            found_diacritics.add(char)

    # Sample
    sample = text[:1000]

    quality = {
        'total_chars': len(text),
        'diacritics_found': list(found_diacritics),
        'diacritics_count': len(found_diacritics),
        'sample': sample,
        'has_esperanto_chars': len(found_diacritics) >= 4,
    }

    logger.info(f"OCR Quality: {quality['diacritics_count']}/12 diacritics found")
    logger.info(f"Quality: {'✓ GOOD' if quality['has_esperanto_chars'] else '⚠ POOR'}")

    return quality


def ocr_pag(pdf_path: Path, output_dir: Path):
    """Main OCR workflow for PAG."""
    logger.info("=" * 70)
    logger.info("OCR PAG (Plena Analiza Gramatiko)")
    logger.info("=" * 70)

    # Check if PDF exists
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        logger.error("Run ./scripts/acquire_pag.sh first to download the PDF")
        return False

    # Output paths
    ocr_text_path = output_dir / 'pag_ocr.txt'
    cleaned_text_path = output_dir / 'pag_ocr_cleaned.txt'

    # Step 1: OCR the PDF
    success = ocr_pdf(pdf_path, ocr_text_path)
    if not success:
        logger.error("OCR failed, exiting")
        return False

    # Step 2: Read OCR text
    ocr_text = ocr_text_path.read_text(encoding='utf-8')

    # Step 3: Basic cleaning
    # Remove excessive whitespace
    cleaned_text = re.sub(r'\n{3,}', '\n\n', ocr_text)
    cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)
    cleaned_text = cleaned_text.strip()

    # Save cleaned text
    cleaned_text_path.write_text(cleaned_text, encoding='utf-8')
    logger.info(f"Saved cleaned text: {cleaned_text_path}")

    # Step 4: Validate quality
    quality = validate_ocr_quality(cleaned_text)

    # Report
    logger.info("")
    logger.info("=" * 70)
    logger.info("OCR COMPLETE")
    logger.info("=" * 70)
    logger.info(f"PDF: {pdf_path}")
    logger.info(f"OCR text: {ocr_text_path}")
    logger.info(f"Cleaned text: {cleaned_text_path}")
    logger.info(f"Total characters: {quality['total_chars']:,}")
    logger.info(f"Diacritics found: {', '.join(quality['diacritics_found'])}")
    logger.info(f"Quality: {'✓ GOOD - Ready for extraction' if quality['has_esperanto_chars'] else '⚠ POOR - May need manual correction'}")
    logger.info("")
    logger.info("Sample (first 1000 chars):")
    logger.info(quality['sample'])
    logger.info("")
    logger.info("Next steps:")
    if quality['has_esperanto_chars']:
        logger.info("1. Review pag_ocr_cleaned.txt for quality")
        logger.info("2. Compare with pag.txt (PyPDF2 extraction) for accuracy")
        logger.info("3. Proceed to sentence extraction if quality is good")
    else:
        logger.info("1. Check that Esperanto language pack is installed: tesseract --list-langs")
        logger.info("2. If 'epo' not listed, install: sudo apt install tesseract-ocr-epo")
        logger.info("3. Re-run this script")

    return quality['has_esperanto_chars']


def main():
    parser = argparse.ArgumentParser(description='OCR PAG PDF to recover Esperanto diacritics')
    parser.add_argument('--pdf-path', type=Path,
                        default=Path('data/raw/eo/pag/pag.pdf'),
                        help='Path to PAG PDF')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('data/raw/eo/pag'),
                        help='Output directory')

    args = parser.parse_args()

    try:
        success = ocr_pag(args.pdf_path, args.output_dir)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"OCR failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
