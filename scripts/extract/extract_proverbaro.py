#!/usr/bin/env python3
"""
Extract proverbs from Proverbaro Esperanta (Zamenhof, 1910).

Format: Numbered proverbs like "1. — La hundo bojas, la karavano iras."

Expected: 2,630 proverbs from 98-page scanned book.

Output: JSONL with proverb + metadata (source, author, tier, quality, proverb_number)
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_numbered_proverbs(text: str) -> List[Tuple[int, str]]:
    """
    Extract numbered proverbs from Proverbaro text.

    Format: "123. — Proverb text here."

    Returns list of (number, proverb_text) tuples.
    """
    proverbs = []

    # Pattern: number, period, optional whitespace, em-dash, proverb text
    # Matches: "1. — Text" or "1. - Text" or "1.— Text"
    pattern = r'^(\d+)\.\s*[—\-]\s*(.+?)(?=^\d+\.\s*[—\-]|\Z)'

    matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)

    for match in matches:
        number = int(match.group(1))
        proverb = match.group(2).strip()

        # Clean up the proverb text
        # Remove excessive whitespace
        proverb = re.sub(r'\s+', ' ', proverb)

        # Remove page numbers (isolated numbers at line breaks)
        proverb = re.sub(r'\s+\d+\s+', ' ', proverb)

        # Remove header fragments
        proverb = re.sub(r'PROVERBARO\s+ESPERANTA', '', proverb, flags=re.IGNORECASE)

        proverb = proverb.strip()

        # Only keep proverbs with actual content
        if len(proverb) > 5 and any(c.isalpha() for c in proverb):
            proverbs.append((number, proverb))

    return proverbs


def validate_proverb_sequence(proverbs: List[Tuple[int, str]]) -> dict:
    """
    Validate that proverbs are numbered sequentially.

    Returns dict with validation results.
    """
    if not proverbs:
        return {
            'is_valid': False,
            'total_found': 0,
            'expected': 2630,
            'gaps': [],
            'duplicates': [],
        }

    numbers = [num for num, _ in proverbs]

    # Check for duplicates
    duplicates = [num for num in set(numbers) if numbers.count(num) > 1]

    # Check for gaps
    expected_max = max(numbers)
    gaps = [i for i in range(1, expected_max + 1) if i not in numbers]

    return {
        'is_valid': len(duplicates) == 0 and len(gaps) == 0,
        'total_found': len(proverbs),
        'expected': 2630,
        'min_number': min(numbers),
        'max_number': max(numbers),
        'gaps': gaps[:20],  # Sample of gaps
        'duplicates': duplicates,
        'coverage': len(proverbs) / 2630 * 100,
    }


def extract_proverbaro(
    text_path: Path,
    output_path: Path
) -> int:
    """Extract proverbs from Proverbaro Esperanta."""

    logger.info(f"Extracting proverbs from: {text_path}")

    # Read text
    try:
        text = text_path.read_text(encoding='utf-8')
    except Exception as e:
        logger.error(f"Failed to read {text_path}: {e}")
        return 0

    logger.info(f"Text size: {len(text):,} characters")

    # Extract numbered proverbs
    proverbs = extract_numbered_proverbs(text)
    logger.info(f"Found {len(proverbs)} proverbs")

    # Validate sequence
    validation = validate_proverb_sequence(proverbs)

    logger.info("")
    logger.info("Validation results:")
    logger.info(f"  Total found: {validation['total_found']}")
    logger.info(f"  Expected: {validation['expected']}")
    logger.info(f"  Coverage: {validation['coverage']:.1f}%")
    logger.info(f"  Number range: {validation['min_number']}-{validation['max_number']}")
    logger.info(f"  Gaps: {len(validation['gaps'])}")
    logger.info(f"  Duplicates: {len(validation['duplicates'])}")

    if validation['gaps']:
        logger.warning(f"  First gaps: {validation['gaps'][:10]}")
    if validation['duplicates']:
        logger.warning(f"  Duplicate numbers: {validation['duplicates']}")

    # Write JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for number, proverb in proverbs:
            entry = {
                'sentence': proverb,
                'source': 'proverbaro',
                'source_title': 'Proverbaro Esperanta',
                'author': 'L.L. Zamenhof',
                'translator': None,
                'year': 1910,
                'tier': 0,
                'quality': 'authoritative',
                'source_type': 'proverbs',
                'proverb_number': number,
            }

            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            extracted_count += 1

    logger.info(f"\n✓ Extracted {extracted_count} proverbs to {output_path}")

    # Save validation report
    report_path = output_path.with_suffix('.validation.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(validation, f, indent=2)

    logger.info(f"✓ Validation report saved to {report_path}")

    return extracted_count


def main():
    parser = argparse.ArgumentParser(
        description='Extract proverbs from Proverbaro Esperanta'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/raw/eo/proverbaro/proverbaro.txt'),
        help='Input cleaned text file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/extracted/eo/tier0/proverbaro_sentences.jsonl'),
        help='Output JSONL file'
    )

    args = parser.parse_args()

    try:
        logger.info("=" * 70)
        logger.info("EXTRACT PROVERBARO ESPERANTA")
        logger.info("=" * 70)
        logger.info("")

        count = extract_proverbaro(args.input, args.output)

        if count == 0:
            logger.error("No proverbs extracted")
            return 1

        logger.info("")
        logger.info("=" * 70)
        logger.info("EXTRACTION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total proverbs: {count:,}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Review validation report for gaps/duplicates")
        logger.info("  2. Check sample proverbs for quality")
        logger.info("  3. Integrate into unified corpus")

        return 0

    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
