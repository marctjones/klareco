#!/usr/bin/env python3
"""
Filter metadata from Tier 0 extracted sentences.

Removes:
- Publisher metadata (Paris, London, Warszawa, etc.)
- Table of contents entries
- Short fragments (<20 chars)
- Non-Esperanto content
- Historical formatting artifacts
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def is_valid_esperanto_sentence(sentence: dict) -> bool:
    """
    Filter out metadata and low-quality sentences.

    Returns True if sentence should be kept, False if filtered.
    """
    text = sentence['sentence']

    # Minimum length
    if len(text) < 20:
        return False

    # Must have alphabetic content
    if not any(c.isalpha() for c in text):
        return False

    # Common Esperanto words (at least one should appear)
    esperanto_words = [
        'la', 'kaj', 'de', 'en', 'estas', 'mi', 'vi', 'li', 'ŝi',
        'ni', 'ili', 'tiu', 'ĉi', 'ne', 'al', 'el', 'sur', 'sub',
        'per', 'kun', 'sen', 'por', 'pro', 'dum', 'post', 'antaŭ'
    ]

    # Non-Esperanto language markers
    non_esperanto_markers = {
        'german': ['ist', 'und', 'der', 'die', 'das', 'ein', 'eine', 'mit', 'von', 'zu', 'im', 'auf'],
        'english': ['the', 'and', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'will', 'shall'],
        'french': ['le', 'les', 'un', 'une', 'est', 'sont', 'avec', 'pour', 'dans', 'sur'],
        'polish': ['się', 'jest', 'są', 'tego', 'dla', 'przez', 'który', 'która'],
    }

    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)

    # Check if any Esperanto words present
    esperanto_found = any(word in esperanto_words for word in words)
    if not esperanto_found:
        return False

    # Check for non-Esperanto language markers (if multiple found, likely not Esperanto)
    for lang, markers in non_esperanto_markers.items():
        marker_count = sum(1 for word in words if word in markers)
        if marker_count >= 3:  # 3+ markers from same language = probably that language
            return False

    # Publisher keywords (multilingual metadata)
    publisher_keywords = [
        'Paris', 'London', 'New-York', 'Kjöbenhavn', 'Warszawa', 'Barcelona',
        'HACHETTE', 'PARIS', 'LONDON', 'Oficiala Gazeto', 'Esperantista',
        'korespondantoj', 'Administracio', 'HOST & SON', 'HOEST & SOEN',
        'ESPASA', 'FRANCUJO', 'Cⁱᵉ', "C'%"
    ]

    if any(keyword in text for keyword in publisher_keywords):
        return False

    # Table of contents markers
    toc_markers = ['ĈAPITRO PAĜO', 'paĝo', 'Enhavo.']
    if any(marker in text for marker in toc_markers):
        # Check if it's a TOC entry (short with "paĝo" and numbers)
        if 'paĝo' in text_lower and len(text) < 100:
            return False

    # Historical publisher formatting
    if re.search(r'—.*et.*,.*PARIS', text):
        return False

    # Dictionary/vocabulary fragments (multilingual)
    # Pattern: word | translation | übersetzung \ перевод | tłumaczenie
    # Also catch single-pipe entries (common in Ekzercaro)
    if '|' in text:
        # If has pipe and is short, likely dictionary
        if len(text) < 150:
            return False
        # If has multiple pipes, definitely dictionary
        if text.count('|') >= 2:
            return False

    # Chapter/section headers only (very short, ends with period or roman numeral)
    if len(text) < 30 and re.match(r'^[IVXLCDM]+\.?$', text.strip()):
        return False

    # Book title pages (common patterns)
    title_patterns = [
        r'^(FUNDAMENTA|EKZERCARO|UNIVERSALA VORTARO)',
        r'^Bertilo Wennergren',
        r'^PMEG$',
        r'^Plena Manlibro',
        r'^Tekstaro de Esperanto',
        r'^ZAMENHOF FUNDAMENTA'
    ]

    for pattern in title_patterns:
        if re.match(pattern, text.strip()):
            return False

    # Copyright/metadata lines
    if text.startswith('©') or 'Copyright' in text:
        return False

    return True


def filter_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    """
    Filter metadata from a single JSONL file.

    Returns (total_sentences, kept_sentences)
    """
    logger.info(f"Filtering {input_path.name}")

    sentences = []
    filtered_count = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)

            if is_valid_esperanto_sentence(entry):
                sentences.append(entry)
            else:
                filtered_count += 1

    total = len(sentences) + filtered_count

    # Write filtered output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in sentences:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    logger.info(f"  Total: {total:,} sentences")
    logger.info(f"  Kept: {len(sentences):,} ({len(sentences)/total*100:.1f}%)")
    logger.info(f"  Filtered: {filtered_count:,} ({filtered_count/total*100:.1f}%)")

    return total, len(sentences)


def filter_all_tier0(input_dir: Path, output_dir: Path):
    """Filter all Tier 0 extracted files."""

    logger.info("=" * 70)
    logger.info("FILTER TIER 0 METADATA")
    logger.info("=" * 70)
    logger.info("")

    # Process literary works
    literary_input = input_dir / 'literary'
    literary_output = output_dir / 'literary'

    logger.info("Literary Works:")
    literary_total = 0
    literary_kept = 0

    for file in sorted(literary_input.glob('*_sentences.jsonl')):
        output_file = literary_output / file.name
        total, kept = filter_file(file, output_file)
        literary_total += total
        literary_kept += kept
        logger.info("")

    # Process grammar works
    grammar_input = input_dir / 'grammar'
    grammar_output = output_dir / 'grammar'

    logger.info("Grammar Works:")
    grammar_total = 0
    grammar_kept = 0

    for file in sorted(grammar_input.glob('*_sentences.jsonl')):
        output_file = grammar_output / file.name
        total, kept = filter_file(file, output_file)
        grammar_total += total
        grammar_kept += kept
        logger.info("")

    # Summary
    overall_total = literary_total + grammar_total
    overall_kept = literary_kept + grammar_kept
    overall_filtered = overall_total - overall_kept

    logger.info("=" * 70)
    logger.info("FILTERING COMPLETE")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"Total sentences: {overall_total:,}")
    logger.info(f"Kept: {overall_kept:,} ({overall_kept/overall_total*100:.1f}%)")
    logger.info(f"Filtered: {overall_filtered:,} ({overall_filtered/overall_total*100:.1f}%)")
    logger.info("")
    logger.info("Breakdown:")
    logger.info(f"  Literary works: {literary_kept:,}/{literary_total:,} kept ({literary_total-literary_kept:,} filtered)")
    logger.info(f"  Grammar works: {grammar_kept:,}/{grammar_total:,} kept ({grammar_total-grammar_kept:,} filtered)")
    logger.info("")
    logger.info(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Filter metadata from Tier 0 extracted sentences'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('data/extracted/eo/tier0'),
        help='Input directory with extracted JSONL files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/extracted/eo/tier0_filtered'),
        help='Output directory for filtered files'
    )

    args = parser.parse_args()

    try:
        filter_all_tier0(args.input_dir, args.output_dir)
        return 0
    except Exception as e:
        logger.error(f"Filtering failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
