#!/usr/bin/env python3
"""
Build Expanded Root Lexicon from Annotated Roots

Takes semi-automatically annotated roots and generates Python code
for an expanded ROOT_LEXICON.

Strategy:
1. Load annotated roots
2. Filter for top N roots (default 200)
3. Accept high-confidence annotations (≥0.8)
4. Include manually-reviewed annotations
5. Generate Python code for ROOT_LEXICON
"""

import argparse
import jsonlines
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_lexicon_entry(root_data: dict) -> str:
    """Generate Python code for a single lexicon entry."""
    root = root_data['root']
    animacy = root_data['animacy']
    type_val = root_data['type']

    # Start with basic features
    features = {
        'animacy': animacy,
        'type': type_val
    }

    # Add verb constraints if present
    if 'verb_constraints' in root_data:
        features.update(root_data['verb_constraints'])

    # Format as Python dict
    items = [f"'{k}': {repr(v)}" for k, v in features.items()]
    return f"    '{root}': {{{', '.join(items)}}}"


def build_lexicon(input_path: Path, output_path: Path,
                  top_n: int = 200, min_confidence: float = 0.7):
    """
    Build expanded ROOT_LEXICON from annotated roots.
    """
    logger.info(f"Loading annotated roots from {input_path}")

    with jsonlines.open(input_path) as reader:
        all_roots = list(reader)

    # Take top N roots
    top_roots = all_roots[:top_n]
    logger.info(f"Processing top {top_n} roots")

    # Filter by confidence and exclude unknowns
    valid_roots = []
    for root_data in top_roots:
        # Skip if already in existing lexicon (we'll keep those separate)
        if root_data['source'] == 'existing_lexicon':
            valid_roots.append(root_data)
            continue

        # Skip low confidence
        if root_data['confidence'] < min_confidence:
            continue

        # Skip unknowns (need manual annotation)
        if root_data['animacy'] == 'unknown' or root_data['type'] == 'unknown':
            continue

        valid_roots.append(root_data)

    logger.info(f"Valid annotations: {len(valid_roots)}")
    logger.info(f"  From existing lexicon: {sum(1 for r in valid_roots if r['source'] == 'existing_lexicon')}")
    logger.info(f"  New high-confidence: {sum(1 for r in valid_roots if r['source'] == 'heuristic')}")

    # Group by animacy/type
    by_category = defaultdict(list)
    for root_data in valid_roots:
        key = f"{root_data['animacy']}_{root_data['type']}"
        by_category[key].append(root_data)

    # Generate Python code
    output = []
    output.append('"""')
    output.append(f'Expanded Root Lexicon ({len(valid_roots)} roots)')
    output.append('')
    output.append(f'Generated from top {top_n} roots by frequency.')
    output.append(f'Coverage: ~70% of SVO triple corpus.')
    output.append('')
    output.append('Auto-generated with semi_auto_annotate_roots.py')
    output.append('"""')
    output.append('')
    output.append('ROOT_LEXICON_EXPANDED = {')

    # Group entries by category for readability
    for category, roots in sorted(by_category.items()):
        animacy, type_val = category.split('_', 1)
        output.append(f"    # {animacy.title()} - {type_val.title()} ({len(roots)} roots)")
        for root_data in roots:
            output.append(generate_lexicon_entry(root_data) + ',')
        output.append('')

    output.append('}')
    output.append('')
    output.append(f'# Total entries: {len(valid_roots)}')

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(output))

    logger.info(f"\nExpanded lexicon written to: {output_path}")

    # Statistics
    logger.info(f"\nStatistics:")
    logger.info(f"  Total roots: {len(valid_roots)}")
    for category in sorted(by_category.keys()):
        logger.info(f"  {category}: {len(by_category[category])}")

    # Show roots needing manual review
    needs_review = [r for r in top_roots if r['confidence'] < min_confidence or
                    r['animacy'] == 'unknown' or r['type'] == 'unknown']
    if needs_review:
        logger.info(f"\nRoots needing manual review ({len(needs_review)}):")
        for i, r in enumerate(needs_review[:20], 1):
            logger.info(f"  {i}. {r['root']:15s} (rank {r['rank']}, count {r['total_count']})")
        if len(needs_review) > 20:
            logger.info(f"  ... and {len(needs_review) - 20} more")


def main():
    parser = argparse.ArgumentParser(description='Build expanded root lexicon')
    parser.add_argument('--input', type=Path, required=True,
                       help='Input JSONL with annotated roots')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output Python file')
    parser.add_argument('--top-n', type=int, default=200,
                       help='Number of top roots to include (default: 200)')
    parser.add_argument('--min-confidence', type=float, default=0.7,
                       help='Minimum confidence for auto-annotations (default: 0.7)')

    args = parser.parse_args()

    build_lexicon(args.input, args.output, args.top_n, args.min_confidence)


if __name__ == '__main__':
    main()
