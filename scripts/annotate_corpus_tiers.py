#!/usr/bin/env python3
"""
Annotate existing corpus with source tiers and weights.

This script reads an existing corpus (corpus_with_sources_v2.jsonl) and adds
proper tier, weight, and citation annotations based on the source field.

Usage:
    python scripts/annotate_corpus_tiers.py \
        --input data/corpus_with_sources_v2.jsonl \
        --output data/corpus/tiered_corpus.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Source to tier mapping
SOURCE_TIERS = {
    # Tier 5: Gutenberg translations (quality literature)
    'la_mastro_de_l_ringoj': {
        'tier': 5,
        'weight': 1.5,
        'full_name': 'La Mastro de l\' Ringoj (Lord of the Rings)',
        'author': 'J.R.R. Tolkien (translated)',
        'type': 'translation'
    },
    'la_hobito': {
        'tier': 5,
        'weight': 1.5,
        'full_name': 'La Hobito (The Hobbit)',
        'author': 'J.R.R. Tolkien (translated)',
        'type': 'translation'
    },
    'alicio': {
        'tier': 5,
        'weight': 1.5,
        'full_name': 'Alicio en Mirlando (Alice in Wonderland)',
        'author': 'Lewis Carroll (translated)',
        'type': 'translation'
    },
    'frankenstejno': {
        'tier': 5,
        'weight': 1.5,
        'full_name': 'Frankenstein',
        'author': 'Mary Shelley (translated)',
        'type': 'translation'
    },
    'milito_de_la_mondoj': {
        'tier': 5,
        'weight': 1.5,
        'full_name': 'La Milito de la Mondoj (War of the Worlds)',
        'author': 'H.G. Wells (translated)',
        'type': 'translation'
    },
    'sorcxisto_de_oz': {
        'tier': 5,
        'weight': 1.5,
        'full_name': 'La Sorĉisto de Oz (Wizard of Oz)',
        'author': 'L. Frank Baum (translated)',
        'type': 'translation'
    },
    'jekyll_hyde': {
        'tier': 5,
        'weight': 1.5,
        'full_name': 'Dr Jekyll kaj Mr Hyde',
        'author': 'R.L. Stevenson (translated)',
        'type': 'translation'
    },
    # Poe stories
    'kadavrejo_strato': {
        'tier': 5,
        'weight': 1.5,
        'full_name': 'Kadavrejo Strato (Murders in the Rue Morgue)',
        'author': 'Edgar Allan Poe (translated)',
        'type': 'translation'
    },
    'la_korvo': {
        'tier': 5,
        'weight': 1.5,
        'full_name': 'La Korvo (The Raven)',
        'author': 'Edgar Allan Poe (translated)',
        'type': 'translation'
    },
    'puto_kaj_pendolo': {
        'tier': 5,
        'weight': 1.5,
        'full_name': 'La Puto kaj la Pendolo (The Pit and the Pendulum)',
        'author': 'Edgar Allan Poe (translated)',
        'type': 'translation'
    },
    'usxero_domo': {
        'tier': 5,
        'weight': 1.5,
        'full_name': 'La Falo de Uŝero-Domo (Fall of the House of Usher)',
        'author': 'Edgar Allan Poe (translated)',
        'type': 'translation'
    },
    'ses_noveloj': {
        'tier': 5,
        'weight': 1.5,
        'full_name': 'Ses Noveloj',
        'author': 'Various (translated)',
        'type': 'translation'
    },

    # Tier 6: Wikipedia (community-written)
    'wikipedia': {
        'tier': 6,
        'weight': 1.0,
        'full_name': 'Vikipedio (Esperanto Wikipedia)',
        'author': 'Community',
        'type': 'encyclopedia'
    },
    'vikipedio': {
        'tier': 6,
        'weight': 1.0,
        'full_name': 'Vikipedio (Esperanto Wikipedia)',
        'author': 'Community',
        'type': 'encyclopedia'
    },

    # Original Esperanto works (Tier 5, slightly higher weight)
    'vivo_de_zamenhof': {
        'tier': 5,
        'weight': 1.8,
        'full_name': 'Vivo de Zamenhof',
        'author': 'Edmond Privat',
        'type': 'original'
    },
    'esperanta_sintakso': {
        'tier': 5,
        'weight': 1.8,
        'full_name': 'Esperanta Sintakso',
        'author': 'Various',
        'type': 'original'
    },
    'dokumentoj_de_esperanto': {
        'tier': 5,
        'weight': 1.8,
        'full_name': 'Dokumentoj de Esperanto',
        'author': 'Various',
        'type': 'original'
    },
}

# Default for unknown sources
DEFAULT_TIER = {
    'tier': 7,
    'weight': 0.5,
    'full_name': 'Unknown Source',
    'author': 'Unknown',
    'type': 'unknown'
}


def normalize_source_name(source: str) -> str:
    """Normalize source name for matching."""
    # Remove common prefixes/suffixes
    source = source.lower()
    source = source.replace('cleaned_', '')
    source = source.replace('.txt', '')
    source = source.replace(' ', '_')
    return source


def annotate_entry(entry: dict, line_number: int) -> dict:
    """Add tier and weight annotations to a corpus entry."""
    source = entry.get('source', 'unknown')
    source_normalized = normalize_source_name(source)

    # Look up tier info
    tier_info = SOURCE_TIERS.get(source_normalized, DEFAULT_TIER)

    # Build new source structure
    new_source = {
        'tier': tier_info['tier'],
        'name': source_normalized,
        'full_name': tier_info['full_name'],
        'citation': f"{source_normalized}:L{line_number}",
        'author': tier_info['author'],
        'type': tier_info['type'],
        'weight': tier_info['weight'],
        'verified': tier_info['tier'] <= 5,  # Gutenberg and above are verified
        # Preserve original fields
        'original_source': source,
        'original_source_name': entry.get('source_name', '')
    }

    # Add paragraph info if available
    if 'paragraph' in entry:
        new_source['paragraph'] = entry['paragraph']
        new_source['citation'] = f"{source_normalized}:p{entry['paragraph']}"

    # Build annotated entry
    annotated = {
        'text': entry.get('text', ''),
        'source': new_source,
        'ast': entry.get('ast', {}),
        'parse_statistics': {
            'parse_rate': entry.get('parse_rate', 0.0),
            'word_count': entry.get('word_count', 0)
        }
    }

    # Preserve AST parse statistics if present
    if 'ast' in entry and 'parse_statistics' in entry['ast']:
        annotated['parse_statistics'].update(entry['ast']['parse_statistics'])

    return annotated


def main():
    parser = argparse.ArgumentParser(description='Annotate corpus with source tiers')
    parser.add_argument('--input', type=Path, required=True,
                        help='Input corpus JSONL file')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output annotated corpus JSONL file')
    parser.add_argument('--sample', type=int, default=0,
                        help='Only process first N entries (0 = all)')

    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Annotating Corpus with Source Tiers")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")

    # Statistics
    tier_counts = {i: 0 for i in range(1, 8)}
    source_counts = {}
    total = 0

    with open(args.input, 'r', encoding='utf-8') as infile, \
         open(args.output, 'w', encoding='utf-8') as outfile:

        for line_number, line in enumerate(infile, 1):
            if args.sample and line_number > args.sample:
                break

            try:
                entry = json.loads(line.strip())
                annotated = annotate_entry(entry, line_number)

                # Track statistics
                tier = annotated['source']['tier']
                tier_counts[tier] += 1

                source_name = annotated['source']['name']
                source_counts[source_name] = source_counts.get(source_name, 0) + 1

                outfile.write(json.dumps(annotated, ensure_ascii=False) + '\n')
                total += 1

                if line_number % 100000 == 0:
                    logger.info(f"Processed {line_number:,} entries...")

            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_number}: JSON error: {e}")
            except Exception as e:
                logger.warning(f"Line {line_number}: Error: {e}")

    # Report
    logger.info("=" * 60)
    logger.info("ANNOTATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total entries: {total:,}")
    logger.info("")
    logger.info("By Tier:")
    for tier in range(1, 8):
        if tier_counts[tier] > 0:
            pct = tier_counts[tier] / total * 100
            logger.info(f"  Tier {tier}: {tier_counts[tier]:,} ({pct:.1f}%)")

    logger.info("")
    logger.info("By Source (top 10):")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1])[:10]:
        pct = count / total * 100
        logger.info(f"  {source}: {count:,} ({pct:.1f}%)")


if __name__ == '__main__':
    main()
