#!/usr/bin/env python3
"""
Extract Universala Vortaro (UV) from Fundamento de Esperanto.

Phase 0.1 of Fundamento-Centered Training (Issue #66)

Input: data/raw/fundamento/fundamento_de_esperanto.txt
Output: data/vocabularies/fundamento_roots.json

Format of output:
{
    "am": {
        "translations": {"fr": "aimer", "en": "love", "de": "lieben", "ru": "любить", "pl": "kochać"},
        "line_number": 5336,
        "category": "verb_root"
    },
    ...
}
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def setup_file_logging(log_path: Path):
    """Add file handler for logging."""
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)


def parse_uv_entry(line: str, line_number: int) -> Optional[dict]:
    """
    Parse a single UV entry line.

    Format examples:
    - aks' axe | axle | Achse | ocb | oŝ.
    - am' aimer | love | lieben | .noŭuib | kochaĉ, lubiĉ.
    - akv 7 eau | water | Wasser | вода | woda  (OCR artifact)
    """
    # Clean up common OCR artifacts
    line = line.strip()
    if not line:
        return None

    # Skip section headers and non-entry lines
    if line.startswith('UNIVERSALA') or line.isnumeric() or len(line) < 5:
        return None

    # Pattern: root' or root (with possible OCR artifacts like '7' for ')
    # Followed by translations separated by |

    # Try to extract root (ends with ' or has OCR artifact like 7, ' etc.)
    # Match: word followed by ' or digit (OCR for '), then translations
    match = re.match(r"^([a-zĉĝĥĵŝŭ]+)[\'\`\'\'\s\d]+(.+)$", line, re.IGNORECASE)

    if not match:
        return None

    root = match.group(1).lower()
    rest = match.group(2).strip()

    # Split by | to get translations (FR | EN | DE | RU | PL)
    parts = [p.strip() for p in rest.split('|')]

    if len(parts) < 5:
        # May have continuation lines or parsing issues
        return None

    translations = {
        'fr': parts[0].rstrip('.'),
        'en': parts[1].rstrip('.'),
        'de': parts[2].rstrip('.'),
        'ru': parts[3].rstrip('.'),
        'pl': parts[4].rstrip('.')
    }

    # Categorize root based on translation patterns
    category = categorize_root(root, translations)

    return {
        'root': root,
        'translations': translations,
        'line_number': line_number,
        'category': category
    }


def categorize_root(root: str, translations: dict) -> str:
    """
    Categorize root based on translation patterns.

    This is a heuristic - translations give hints about the word class.
    """
    en = translations.get('en', '').lower()

    # Verb indicators in English translations
    verb_indicators = ['to ', 'be ', '-ing', '-ed']
    if any(vi in en for vi in verb_indicators):
        return 'verb_root'

    # Adjective indicators
    adj_indicators = ['-ful', '-ous', '-ive', '-able', '-ible']
    if any(ai in en for ai in adj_indicators):
        return 'adjective_root'

    # Common noun patterns
    if en in ['man', 'woman', 'child', 'house', 'tree', 'water', 'sun', 'moon']:
        return 'noun_root'

    # Default
    return 'root'


def extract_universala_vortaro(fundamento_path: Path, uv_start_line: int = 5296) -> dict:
    """
    Extract all UV entries from the Fundamento text.

    Args:
        fundamento_path: Path to fundamento_de_esperanto.txt
        uv_start_line: Line number where UV section starts (1-indexed)

    Returns:
        Dictionary mapping roots to their data
    """
    logger.info(f"Reading Fundamento from {fundamento_path}")

    with open(fundamento_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    logger.info(f"Total lines: {len(lines)}")
    logger.info(f"Starting UV extraction from line {uv_start_line}")

    roots = {}
    parsed_count = 0
    failed_count = 0

    # Process lines from UV start to end
    for i, line in enumerate(lines[uv_start_line - 1:], start=uv_start_line):
        entry = parse_uv_entry(line, i)
        if entry:
            root = entry.pop('root')
            if root not in roots:
                roots[root] = entry
                parsed_count += 1
            else:
                # Duplicate root - might be continuation
                logger.debug(f"Duplicate root '{root}' at line {i}")
        elif line.strip() and '|' in line:
            # Line had | but couldn't parse - log for debugging
            failed_count += 1
            if failed_count <= 10:
                logger.debug(f"Failed to parse line {i}: {line[:80]}")

    logger.info(f"Extracted {parsed_count} unique roots")
    logger.info(f"Failed to parse {failed_count} lines with separators")

    return roots


def save_checkpoint(roots: dict, checkpoint_path: Path):
    """Save intermediate results for restartability."""
    temp_path = checkpoint_path.with_suffix('.tmp')
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'count': len(roots),
            'roots': roots
        }, f, ensure_ascii=False, indent=2)
    temp_path.rename(checkpoint_path)
    logger.info(f"Checkpoint saved: {len(roots)} roots")


def main():
    parser = argparse.ArgumentParser(description='Extract Universala Vortaro from Fundamento')
    parser.add_argument('--input', type=Path,
                        default=Path('data/raw/fundamento/fundamento_de_esperanto.txt'),
                        help='Path to Fundamento text file')
    parser.add_argument('--output', type=Path,
                        default=Path('data/vocabularies/fundamento_roots.json'),
                        help='Output JSON file')
    parser.add_argument('--log-dir', type=Path,
                        default=Path('logs/training'),
                        help='Directory for log files')
    parser.add_argument('--uv-start', type=int, default=5296,
                        help='Line number where UV section starts')
    parser.add_argument('--dry-run', action='store_true',
                        help='Parse and report but do not write output')

    args = parser.parse_args()

    # Setup logging
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f'extract_fundamento_uv_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    setup_file_logging(log_path)

    logger.info("=" * 60)
    logger.info("Phase 0.1: Fundamento UV Extraction")
    logger.info("=" * 60)

    # Validate input
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Extract roots
    roots = extract_universala_vortaro(args.input, args.uv_start)

    # Report statistics
    categories = {}
    for root, data in roots.items():
        cat = data.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1

    logger.info("\nCategory breakdown:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        logger.info(f"  {cat}: {count}")

    # Sample output
    logger.info("\nSample entries:")
    for root in list(roots.keys())[:5]:
        logger.info(f"  {root}: {roots[root]}")

    if args.dry_run:
        logger.info("\nDry run - not writing output")
        return

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = args.output.with_suffix('.tmp')

    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'source': 'Fundamento de Esperanto - Universala Vortaro',
                    'extracted': datetime.now().isoformat(),
                    'input_file': str(args.input),
                    'uv_start_line': args.uv_start,
                    'total_roots': len(roots)
                },
                'roots': roots
            }, f, ensure_ascii=False, indent=2)
        temp_path.rename(args.output)
        logger.info(f"\nOutput written to {args.output}")
    except Exception as e:
        logger.error(f"Failed to write output: {e}")
        if temp_path.exists():
            temp_path.unlink()
        sys.exit(1)

    logger.info(f"Log saved to {log_path}")
    logger.info("Phase 0.1 complete!")


if __name__ == '__main__':
    main()
