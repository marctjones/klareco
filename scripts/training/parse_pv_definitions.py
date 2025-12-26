#!/usr/bin/env python3
"""
Parse Plena Vortaro definitions for definition-based semantic similarity.

Phase 1 of Fundamento-Centered Training (Issue #73)

Input: data/grammar/plena_vortaro.txt
Output: data/vocabularies/pv_definitions.json

The Plena Vortaro contains Esperanto→Esperanto definitions. We extract:
1. Headwords (entry words)
2. Definition text
3. Roots used in definitions (for similarity training)

Training insight: Words with overlapping definition roots should have similar embeddings.
E.g., "hundo" and "kato" both defined as "dommastrumata karnomanĝanto" share roots.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Set
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def setup_file_logging(log_path: Path):
    """Add file handler for logging."""
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)


# Known Esperanto roots for validation (subset - we'll expand from Fundamento)
KNOWN_PREFIXES = {'mal', 'ge', 'ek', 'dis', 're', 'mis', 'bo', 'pra', 'eks'}
KNOWN_SUFFIXES = {'ad', 'aĵ', 'an', 'ar', 'ebl', 'ec', 'eg', 'ej', 'em', 'end',
                  'er', 'estr', 'et', 'id', 'ig', 'iĝ', 'il', 'in', 'ind', 'ing',
                  'ism', 'ist', 'nj', 'obl', 'on', 'op', 'uj', 'ul', 'um', 'ĉj'}


def extract_root_from_word(word: str) -> Optional[str]:
    """
    Extract the root from an Esperanto word.

    Simple heuristic approach - strips common endings and affixes.
    """
    word = word.lower().strip()

    if len(word) < 2:
        return None

    # Skip function words
    if word in {'la', 'kaj', 'en', 'de', 'al', 'kun', 'por', 'pri', 'ĉe', 'el',
                'sur', 'sub', 'tra', 'ĝis', 'per', 'pro', 'sen', 'post', 'anstataŭ',
                'krom', 'dum', 'apud', 'inter', 'kontraŭ', 'malgraŭ', 'preter',
                'ĉirkaŭ', 'ekster', 'super', 'trans', 'ke', 'ĉu', 'se', 'ĉar',
                'kiam', 'kie', 'kio', 'kiu', 'kiel', 'kiom', 'tiam', 'tie', 'tio',
                'tiu', 'tiel', 'tiom', 'iam', 'ie', 'io', 'iu', 'iel', 'iom',
                'neniam', 'nenie', 'nenio', 'neniu', 'neniel', 'neniom',
                'ĉiam', 'ĉie', 'ĉio', 'ĉiu', 'ĉiel', 'ĉiom',
                'mi', 'vi', 'li', 'ŝi', 'ĝi', 'ni', 'ili', 'oni', 'si',
                'mia', 'via', 'lia', 'ŝia', 'ĝia', 'nia', 'ilia', 'onia', 'sia',
                'tiu', 'kiu', 'ĉiu', 'neniu', 'iu',
                'unu', 'du', 'tri', 'kvar', 'kvin', 'ses', 'sep', 'ok', 'naŭ', 'dek',
                'cent', 'mil', 'ne', 'jes', 'ankaŭ', 'nur', 'eĉ', 'ja', 'tamen',
                'do', 'ĉu', 'aŭ'}:
        return None

    # Strip grammatical endings
    root = word
    for ending in ['ojn', 'ajn', 'oj', 'aj', 'on', 'an', 'in', 'en',
                   'as', 'is', 'os', 'us', 'o', 'a', 'e', 'i', 'u', 'n']:
        if root.endswith(ending) and len(root) - len(ending) >= 2:
            root = root[:-len(ending)]
            break

    # Strip common suffixes (from outside in)
    for suffix in KNOWN_SUFFIXES:
        if root.endswith(suffix) and len(root) - len(suffix) >= 2:
            root = root[:-len(suffix)]
            break

    # Strip common prefixes
    for prefix in KNOWN_PREFIXES:
        if root.startswith(prefix) and len(root) - len(prefix) >= 2:
            root = root[len(prefix):]
            break

    if len(root) >= 2:
        return root

    return None


def extract_roots_from_text(text: str) -> List[str]:
    """Extract all probable roots from Esperanto text."""
    # Clean text
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = text.split()

    roots = []
    for word in words:
        root = extract_root_from_word(word)
        if root:
            roots.append(root)

    return list(set(roots))


def is_headword_line(line: str) -> bool:
    """Check if line starts a new dictionary entry."""
    # PV entries typically start with a word followed by definition
    # Often marked with * or special formatting
    line = line.strip()

    if not line:
        return False

    # Check for entry patterns
    # *Word. or Word. pattern at start
    if re.match(r'^\*?[A-ZĈĜĤĴŜŬ][a-zĉĝĥĵŝŭ]+\.', line):
        return True

    # All-caps word at start (common in PV)
    if re.match(r'^[A-ZĈĜĤĴŜŬ]{2,}\.?\s', line):
        return True

    return False


def parse_entry(lines: List[str], start_idx: int) -> tuple:
    """
    Parse a single dictionary entry.

    Returns: (headword, definition_text, end_idx)
    """
    first_line = lines[start_idx].strip()

    # Extract headword
    match = re.match(r'^\*?([A-ZĈĜĤĴŜŬa-zĉĝĥĵŝŭ]+)\.?\s*(.*)$', first_line)
    if not match:
        # Try all-caps pattern
        match = re.match(r'^([A-ZĈĜĤĴŜŬ]{2,})\.?\s*(.*)$', first_line)

    if not match:
        return None, None, start_idx + 1

    headword = match.group(1).lower()
    definition_parts = [match.group(2)] if match.group(2) else []

    # Collect continuation lines
    idx = start_idx + 1
    while idx < len(lines):
        line = lines[idx].strip()

        # Stop at next entry
        if is_headword_line(line):
            break

        if line:
            definition_parts.append(line)

        idx += 1

    definition = ' '.join(definition_parts)

    return headword, definition, idx


def parse_plena_vortaro(pv_path: Path, skip_front_matter: int = 800) -> Dict[str, dict]:
    """
    Parse the Plena Vortaro file.

    Args:
        pv_path: Path to plena_vortaro.txt
        skip_front_matter: Lines to skip (prefaces, abbreviations, etc.)

    Returns:
        Dictionary mapping headwords to their data
    """
    logger.info(f"Reading Plena Vortaro from {pv_path}")

    with open(pv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    logger.info(f"Total lines: {len(lines)}")
    logger.info(f"Skipping first {skip_front_matter} lines (front matter)")

    entries = {}
    idx = skip_front_matter

    while idx < len(lines):
        line = lines[idx].strip()

        if is_headword_line(line):
            headword, definition, next_idx = parse_entry(lines, idx)

            if headword and definition and len(definition) > 10:
                # Extract roots from definition
                definition_roots = extract_roots_from_text(definition)

                # Don't include the headword's own root in definition_roots
                headword_root = extract_root_from_word(headword)
                if headword_root in definition_roots:
                    definition_roots.remove(headword_root)

                if definition_roots:  # Only include entries with extractable roots
                    entries[headword] = {
                        'definition': definition[:500],  # Truncate long definitions
                        'definition_roots': definition_roots[:20],  # Limit roots
                        'line_number': idx + 1
                    }

            idx = next_idx
        else:
            idx += 1

        # Progress logging
        if idx % 10000 == 0:
            logger.info(f"Processed {idx}/{len(lines)} lines, {len(entries)} entries")

    logger.info(f"Extracted {len(entries)} entries with definitions")

    return entries


def build_similarity_graph(entries: Dict[str, dict]) -> Dict[str, List[str]]:
    """
    Build a graph of related entries based on shared definition roots.

    Entries with overlapping definition roots should be semantically similar.
    """
    # Index: definition_root -> list of headwords using it
    root_to_entries = defaultdict(list)

    for headword, data in entries.items():
        for root in data.get('definition_roots', []):
            root_to_entries[root].append(headword)

    # Find related entries for each headword
    related = {}
    for headword, data in entries.items():
        related_entries = set()
        for root in data.get('definition_roots', []):
            for other in root_to_entries[root]:
                if other != headword:
                    related_entries.add(other)

        if related_entries:
            related[headword] = list(related_entries)[:10]  # Limit relations

    return related


def main():
    parser = argparse.ArgumentParser(description='Parse Plena Vortaro definitions')
    parser.add_argument('--input', type=Path,
                        default=Path('data/grammar/plena_vortaro.txt'),
                        help='Path to Plena Vortaro text file')
    parser.add_argument('--output', type=Path,
                        default=Path('data/vocabularies/pv_definitions.json'),
                        help='Output JSON file')
    parser.add_argument('--log-dir', type=Path,
                        default=Path('logs/training'),
                        help='Directory for log files')
    parser.add_argument('--skip-lines', type=int, default=800,
                        help='Lines to skip at start (front matter)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Parse and report but do not write output')

    args = parser.parse_args()

    # Setup logging
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f'parse_pv_definitions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    setup_file_logging(log_path)

    logger.info("=" * 60)
    logger.info("Phase 1: Plena Vortaro Definition Parsing")
    logger.info("=" * 60)

    # Validate input
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Parse entries
    entries = parse_plena_vortaro(args.input, args.skip_lines)

    # Build similarity graph
    logger.info("Building definition-based similarity graph...")
    related = build_similarity_graph(entries)

    # Add related entries to data
    for headword in entries:
        if headword in related:
            entries[headword]['related_entries'] = related[headword]

    # Statistics
    total_roots = set()
    for data in entries.values():
        total_roots.update(data.get('definition_roots', []))

    logger.info(f"\nStatistics:")
    logger.info(f"  Total entries: {len(entries)}")
    logger.info(f"  Entries with relations: {len(related)}")
    logger.info(f"  Unique definition roots: {len(total_roots)}")

    # Sample output
    logger.info("\nSample entries:")
    for headword in list(entries.keys())[:5]:
        data = entries[headword]
        logger.info(f"  {headword}:")
        logger.info(f"    Def: {data['definition'][:60]}...")
        logger.info(f"    Roots: {data['definition_roots'][:5]}")
        if 'related_entries' in data:
            logger.info(f"    Related: {data['related_entries'][:3]}")

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
                    'source': 'Plena Vortaro de Esperanto (1980)',
                    'extracted': datetime.now().isoformat(),
                    'input_file': str(args.input),
                    'total_entries': len(entries),
                    'entries_with_relations': len(related),
                    'unique_definition_roots': len(total_roots)
                },
                'entries': entries
            }, f, ensure_ascii=False, indent=2)
        temp_path.rename(args.output)
        logger.info(f"\nOutput written to {args.output}")
    except Exception as e:
        logger.error(f"Failed to write output: {e}")
        if temp_path.exists():
            temp_path.unlink()
        sys.exit(1)

    logger.info(f"Log saved to {log_path}")
    logger.info("Phase 1 complete!")


if __name__ == '__main__':
    main()
