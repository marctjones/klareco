#!/usr/bin/env python3
"""
Create a quality-filtered, tier-weighted training corpus.

This script:
1. Filters sentences by parse quality (90%+ success rate)
2. Applies aggressive tier weights to prioritize authoritative sources
3. Separates authoritative and general corpora for training flexibility
4. Outputs training-ready JSONL files with computed weights

Usage:
    python scripts/create_training_corpus.py \
        --input data/corpus/unified_corpus.jsonl \
        --output-dir data/training \
        --min-parse-rate 0.9
"""

import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Iterator, Dict, Any, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Tier weights - heavily favor authoritative sources
TIER_WEIGHTS = {
    1: 100.0,  # Fundamento Ekzercaro - THE gold standard
    2: 50.0,   # Fundamenta Krestomatio - Zamenhof's selections
    3: 20.0,   # Gerda Malaperis - pedagogical, canonical
    4: 5.0,    # Reta Vortaro (ReVo) - used in root embedding training
    5: 2.0,    # High-quality Gutenberg literature
    6: 1.0,    # Wikipedia - general usage
    7: 0.5,    # Low-quality or unverified
}

# Source categories
AUTHORITATIVE_TIERS = {1, 2, 3}
LITERATURE_TIERS = {5}
GENERAL_TIERS = {6, 7}


def compute_parse_rate(ast: Dict[str, Any]) -> float:
    """Compute parse success rate from AST."""
    total_words = 0
    success_words = 0

    def count_words(node):
        nonlocal total_words, success_words
        if isinstance(node, dict):
            if node.get('tipo') == 'vorto':
                total_words += 1
                if node.get('parse_status') == 'success':
                    success_words += 1
            if 'kerno' in node:
                count_words(node['kerno'])
            for p in node.get('priskriboj', []) or []:
                count_words(p)
        elif isinstance(node, list):
            for item in node:
                count_words(item)

    count_words(ast.get('aliaj', []) or [])
    count_words(ast.get('subjekto', {}) or {})
    count_words(ast.get('verbo', {}) or {})
    count_words(ast.get('objekto', {}) or {})

    return success_words / total_words if total_words > 0 else 0.0


def stream_corpus(input_path: Path) -> Iterator[Dict[str, Any]]:
    """Stream entries from corpus file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def get_source_info(entry: Dict[str, Any]) -> Tuple[str, int, float]:
    """Extract source name, tier, and weight from entry."""
    source_data = entry.get('source', {})
    if isinstance(source_data, dict):
        name = source_data.get('name', 'unknown')
        tier = source_data.get('tier', 6)
        # Use our new weights, not the stored ones
        weight = TIER_WEIGHTS.get(tier, 1.0)
    else:
        name = str(source_data)
        tier = 6
        weight = 1.0
    return name, tier, weight


def create_training_corpus(
    input_path: Path,
    output_dir: Path,
    min_parse_rate: float = 0.9,
    max_wikipedia_samples: int = None
) -> Dict[str, Any]:
    """Create filtered training corpus with tier weights."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Output files
    authoritative_path = output_dir / 'authoritative_training.jsonl'
    literature_path = output_dir / 'literature_training.jsonl'
    general_path = output_dir / 'general_training.jsonl'
    combined_path = output_dir / 'combined_training.jsonl'

    # Statistics
    stats = {
        'input_total': 0,
        'filtered_low_quality': 0,
        'by_tier': defaultdict(int),
        'by_source': defaultdict(int),
        'authoritative_count': 0,
        'literature_count': 0,
        'general_count': 0,
        'wikipedia_sampled': 0,
    }

    # Track Wikipedia for potential sampling
    wikipedia_entries = []

    logger.info(f"Processing corpus: {input_path}")
    logger.info(f"Minimum parse rate: {min_parse_rate}")

    with open(authoritative_path, 'w') as f_auth, \
         open(literature_path, 'w') as f_lit, \
         open(general_path, 'w') as f_gen:

        for entry in stream_corpus(input_path):
            stats['input_total'] += 1

            # Compute parse rate
            ast = entry.get('ast', {})
            parse_rate = compute_parse_rate(ast)

            # Filter by quality
            if parse_rate < min_parse_rate:
                stats['filtered_low_quality'] += 1
                continue

            # Get source info and apply new weights
            name, tier, weight = get_source_info(entry)

            # Update source with new weight
            if isinstance(entry.get('source'), dict):
                entry['source']['weight'] = weight
                entry['source']['training_weight'] = weight

            # Add parse rate to entry
            entry['parse_rate'] = parse_rate

            stats['by_tier'][tier] += 1
            stats['by_source'][name] += 1

            # Route to appropriate file
            output_line = json.dumps(entry, ensure_ascii=False) + '\n'

            if tier in AUTHORITATIVE_TIERS:
                f_auth.write(output_line)
                stats['authoritative_count'] += 1
            elif tier in LITERATURE_TIERS:
                f_lit.write(output_line)
                stats['literature_count'] += 1
            else:
                # Collect Wikipedia for sampling if needed
                if name == 'wikipedia' and max_wikipedia_samples:
                    wikipedia_entries.append(entry)
                else:
                    f_gen.write(output_line)
                    stats['general_count'] += 1

            if stats['input_total'] % 500000 == 0:
                logger.info(f"  Processed {stats['input_total']:,} entries...")

    # Handle Wikipedia sampling if requested
    if max_wikipedia_samples and wikipedia_entries:
        import random
        random.seed(42)  # Reproducible

        if len(wikipedia_entries) > max_wikipedia_samples:
            sampled = random.sample(wikipedia_entries, max_wikipedia_samples)
            logger.info(f"Sampled {max_wikipedia_samples:,} from {len(wikipedia_entries):,} Wikipedia entries")
        else:
            sampled = wikipedia_entries

        with open(general_path, 'a') as f_gen:
            for entry in sampled:
                f_gen.write(json.dumps(entry, ensure_ascii=False) + '\n')
                stats['general_count'] += 1

        stats['wikipedia_sampled'] = len(sampled)
        stats['wikipedia_total'] = len(wikipedia_entries)

    # Create combined file with all training data
    logger.info("Creating combined training file...")
    with open(combined_path, 'w') as f_combined:
        for source_path in [authoritative_path, literature_path, general_path]:
            with open(source_path, 'r') as f_source:
                for line in f_source:
                    f_combined.write(line)

    # Compute effective training signal
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING CORPUS CREATED")
    logger.info("=" * 60)

    effective_signal = defaultdict(float)
    for tier, count in stats['by_tier'].items():
        weight = TIER_WEIGHTS.get(tier, 1.0)
        effective_signal[tier] = count * weight

    total_effective = sum(effective_signal.values())

    logger.info(f"\nInput: {stats['input_total']:,} entries")
    logger.info(f"Filtered (low quality): {stats['filtered_low_quality']:,}")
    logger.info(f"\nBy Category:")
    logger.info(f"  Authoritative: {stats['authoritative_count']:,}")
    logger.info(f"  Literature: {stats['literature_count']:,}")
    logger.info(f"  General: {stats['general_count']:,}")

    if stats.get('wikipedia_sampled'):
        logger.info(f"  (Wikipedia sampled: {stats['wikipedia_sampled']:,} from {stats['wikipedia_total']:,})")

    logger.info(f"\nBy Tier (with weights):")
    for tier in sorted(stats['by_tier'].keys()):
        count = stats['by_tier'][tier]
        weight = TIER_WEIGHTS.get(tier, 1.0)
        effective = effective_signal[tier]
        pct = 100 * effective / total_effective if total_effective > 0 else 0
        logger.info(f"  Tier {tier}: {count:,} × {weight}x = {effective:,.0f} ({pct:.1f}% of signal)")

    logger.info(f"\nTop Sources:")
    for name, count in sorted(stats['by_source'].items(), key=lambda x: -x[1])[:10]:
        logger.info(f"  {name}: {count:,}")

    logger.info(f"\nOutput files:")
    logger.info(f"  {authoritative_path}")
    logger.info(f"  {literature_path}")
    logger.info(f"  {general_path}")
    logger.info(f"  {combined_path}")

    # Save metadata
    metadata = {
        'min_parse_rate': min_parse_rate,
        'tier_weights': TIER_WEIGHTS,
        'stats': {k: dict(v) if isinstance(v, defaultdict) else v for k, v in stats.items()},
        'effective_signal': dict(effective_signal),
        'total_effective_signal': total_effective,
        'files': {
            'authoritative': str(authoritative_path),
            'literature': str(literature_path),
            'general': str(general_path),
            'combined': str(combined_path),
        }
    }

    metadata_path = output_dir / 'training_corpus_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"  {metadata_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description='Create quality-filtered training corpus')
    parser.add_argument('--input', type=Path, default=Path('data/corpus/unified_corpus.jsonl'),
                        help='Input unified corpus')
    parser.add_argument('--output-dir', type=Path, default=Path('data/training'),
                        help='Output directory for training files')
    parser.add_argument('--min-parse-rate', type=float, default=0.9,
                        help='Minimum parse success rate (0.0-1.0)')
    parser.add_argument('--max-wikipedia', type=int, default=None,
                        help='Maximum Wikipedia samples (for capping)')

    args = parser.parse_args()

    create_training_corpus(
        input_path=args.input,
        output_dir=args.output_dir,
        min_parse_rate=args.min_parse_rate,
        max_wikipedia_samples=args.max_wikipedia
    )


if __name__ == '__main__':
    main()
