#!/usr/bin/env python3
"""
Extract Ekzercaro and Fundamenta Krestomatio from existing Fundamento file.

These sections are already in our corpus at:
data/raw/eo/fundamento/fundamento_de_esperanto.txt

This script extracts them as separate files for easier processing.

Author: L.L. Zamenhof
Source: Fundamento de Esperanto
"""

import argparse
import logging
import re
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_section(text: str, start_marker: str, end_marker: str = None) -> tuple[str, int, int]:
    """
    Extract a section from Fundamento text.

    Returns: (section_text, start_pos, end_pos)
    """
    # Find start
    start_pos = text.find(start_marker)
    if start_pos == -1:
        logger.error(f"Start marker not found: {start_marker}")
        return "", -1, -1

    # Find end
    if end_marker:
        end_pos = text.find(end_marker, start_pos + len(start_marker))
        if end_pos == -1:
            logger.error(f"End marker not found: {end_marker}")
            return "", -1, -1
    else:
        # No end marker = go to end of file
        end_pos = len(text)

    section = text[start_pos:end_pos].strip()

    return section, start_pos, end_pos


def extract_ekzercaro(fundamento_text: str) -> str:
    """
    Extract Ekzercaro (exercises) section from Fundamento.

    Ekzercaro is between "EKZERCARO" and "Fundamenta Krestomatio".
    """
    logger.info("Extracting Ekzercaro")

    # Try different variations of the marker
    start_markers = ['EKZERCARO', 'Ekzercaro', 'EKZERCARO.']
    end_markers = ['Fundamenta Krestomatio', 'FUNDAMENTA KRESTOMATIO', 'KRESTOMATIO']

    section = ""
    for start in start_markers:
        for end in end_markers:
            section, start_pos, end_pos = extract_section(fundamento_text, start, end)
            if section:
                logger.info(f"Found Ekzercaro: {start_pos}-{end_pos} ({len(section)} chars)")
                return section

    logger.error("Could not extract Ekzercaro")
    return ""


def extract_krestomatio(fundamento_text: str) -> str:
    """
    Extract Fundamenta Krestomatio (readings) section from Fundamento.

    Krestomatio typically goes from "Fundamenta Krestomatio" to "UNIVERSALA VORTARO" or end.
    """
    logger.info("Extracting Fundamenta Krestomatio")

    # Try different variations
    start_markers = ['Fundamenta Krestomatio', 'FUNDAMENTA KRESTOMATIO', 'KRESTOMATIO']
    end_markers = ['UNIVERSALA VORTARO', 'Universala Vortaro', 'VORTARO', None]  # None = to end of file

    section = ""
    for start in start_markers:
        for end in end_markers:
            section, start_pos, end_pos = extract_section(fundamento_text, start, end)
            if section:
                logger.info(f"Found Krestomatio: {start_pos}-{end_pos} ({len(section)} chars)")
                return section

    logger.error("Could not extract Krestomatio")
    return ""


def clean_section(text: str, section_name: str) -> str:
    """Clean extracted section."""
    logger.info(f"Cleaning {section_name}")

    # Remove excessive whitespace
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    # Remove page markers (common pattern: [page number])
    text = re.sub(r'\[?\d+\]?', '', text)

    text = text.strip()

    logger.info(f"Cleaned: {len(text)} characters")
    return text


def validate_section(text: str, section_name: str) -> dict:
    """Validate extracted section."""
    logger.info(f"Validating {section_name}")

    # Check for Esperanto diacritics
    diacritics = {'ĉ', 'ĝ', 'ĥ', 'ĵ', 'ŝ', 'ŭ', 'Ĉ', 'Ĝ', 'Ĥ', 'Ĵ', 'Ŝ', 'Ŭ'}
    found_diacritics = set()
    for char in text:
        if char in diacritics:
            found_diacritics.add(char)

    # Rough estimates
    lines = text.count('\n')
    words = len(text.split())

    # Section-specific checks
    if section_name == 'Ekzercaro':
        # Ekzercaro should have numbered exercises
        exercise_numbers = len(re.findall(r'^\s*\d+\.', text, re.MULTILINE))
        has_expected_content = exercise_numbers > 10
    else:  # Krestomatio
        # Krestomatio should have readings/stories
        has_expected_content = words > 1000  # Substantial text

    quality = {
        'section': section_name,
        'total_chars': len(text),
        'lines': lines,
        'words': words,
        'diacritics_found': list(found_diacritics),
        'diacritics_count': len(found_diacritics),
        'has_esperanto_chars': len(found_diacritics) >= 4,
        'has_expected_content': has_expected_content,
    }

    logger.info(f"Quality: {words:,} words, {quality['diacritics_count']}/12 diacritics")
    logger.info(f"Valid: {'✓ YES' if quality['has_esperanto_chars'] and quality['has_expected_content'] else '⚠ CHECK'}")

    return quality


def extract_ekzercaro_krestomatio(fundamento_path: Path, output_dir: Path):
    """Main extraction workflow."""
    logger.info("=" * 70)
    logger.info("EXTRACT EKZERCARO & KRESTOMATIO FROM FUNDAMENTO")
    logger.info("=" * 70)

    # Read Fundamento
    logger.info(f"Reading Fundamento from: {fundamento_path}")
    fundamento_text = fundamento_path.read_text(encoding='utf-8')
    logger.info(f"Fundamento: {len(fundamento_text):,} characters")

    # Extract Ekzercaro
    ekzercaro_text = extract_ekzercaro(fundamento_text)
    if not ekzercaro_text:
        logger.error("Failed to extract Ekzercaro")
        return False

    # Extract Krestomatio
    krestomatio_text = extract_krestomatio(fundamento_text)
    if not krestomatio_text:
        logger.error("Failed to extract Krestomatio")
        return False

    # Clean both sections
    ekzercaro_clean = clean_section(ekzercaro_text, 'Ekzercaro')
    krestomatio_clean = clean_section(krestomatio_text, 'Krestomatio')

    # Validate
    ekz_quality = validate_section(ekzercaro_clean, 'Ekzercaro')
    krest_quality = validate_section(krestomatio_clean, 'Krestomatio')

    # Save extracted sections
    output_dir.mkdir(parents=True, exist_ok=True)

    ekzercaro_path = output_dir / 'ekzercaro.txt'
    krestomatio_path = output_dir / 'krestomatio.txt'

    ekzercaro_path.write_text(ekzercaro_clean, encoding='utf-8')
    krestomatio_path.write_text(krestomatio_clean, encoding='utf-8')

    logger.info(f"Saved Ekzercaro: {ekzercaro_path}")
    logger.info(f"Saved Krestomatio: {krestomatio_path}")

    # Save metadata
    import json
    import time

    metadata = {
        'source': str(fundamento_path),
        'extraction_date': time.strftime('%Y-%m-%d'),
        'ekzercaro': ekz_quality,
        'krestomatio': krest_quality,
    }

    metadata_path = output_dir / 'extraction_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Report
    logger.info("")
    logger.info("=" * 70)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Ekzercaro:")
    logger.info(f"  Words: {ekz_quality['words']:,}")
    logger.info(f"  File: {ekzercaro_path}")
    logger.info("")
    logger.info("Krestomatio:")
    logger.info(f"  Words: {krest_quality['words']:,}")
    logger.info(f"  File: {krestomatio_path}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Review extracted files for quality")
    logger.info("2. Verify section boundaries are correct")
    logger.info("3. Proceed to sentence extraction")

    return True


def main():
    parser = argparse.ArgumentParser(description='Extract Ekzercaro & Krestomatio from Fundamento')
    parser.add_argument('--fundamento-path', type=Path,
                        default=Path('data/raw/eo/fundamento/fundamento_de_esperanto.txt'),
                        help='Path to Fundamento file')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('data/raw/eo/fundamento'),
                        help='Output directory')

    args = parser.parse_args()

    if not args.fundamento_path.exists():
        logger.error(f"Fundamento file not found: {args.fundamento_path}")
        sys.exit(1)

    try:
        success = extract_ekzercaro_krestomatio(args.fundamento_path, args.output_dir)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
