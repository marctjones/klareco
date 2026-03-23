#!/usr/bin/env python3
"""
Add Simple Word Decomposition to Existing SVO Triples

Simple heuristic approach: extract affixes by comparing root to full word.
Good enough for testing word-level training data generation.
"""

import argparse
import jsonlines
from pathlib import Path
from typing import Dict, List, Optional
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


# Common Esperanto suffixes (in priority order - longer first)
SUFFIXES = [
    'ista', 'isto', 'ina', 'ino', 'ejo', 'ilo', 'aĵo', 'aro',
    'eta', 'eto', 'ega', 'ego', 'ema', 'emo', 'ida', 'ido',
    'inda', 'ebl', 'iĝ', 'ig', 'ad', 'ant', 'int', 'ont',
    'ata', 'ita', 'ota', 'ec', 'aĉ', 'ul', 'um', 'er', 'il',
    'uj', 'an', 'op', 'obl', 'on', 'ist', 'in', 'et', 'eg',
    'em', 'id', 'aĵ', 'ej', 'ar', 'il'
]

# Common prefixes
PREFIXES = ['mal', 'dis', 'eks', 'ge', 'pra', 're', 'mis', 'fi']

# Word endings (grammatical)
ENDINGS = {
    'o': 'substantivo',
    'a': 'adjektivo',
    'e': 'adverbo',
    'i': 'infinitivo',
    'as': 'verbo',
    'is': 'verbo',
    'os': 'verbo',
    'us': 'verbo',
    'u': 'verbo',
    'on': 'substantivo',
    'an': 'adjektivo',
    'en': 'adverbo',
    'in': 'infinitivo',
    'aj': 'adjektivo',
    'ojn': 'substantivo',
    'ajn': 'adjektivo'
}


def extract_affixes_simple(root: str, full_word: str) -> Dict:
    """
    Extract affixes by comparing root to full word.

    Simple heuristic approach:
    1. Check for prefix
    2. Find root position
    3. Extract middle part as suffixes
    4. Extract ending
    """
    if not root or not full_word:
        return {'root': root, 'affixes': [], 'prefix': None, 'ending': None, 'pos': 'unknown'}

    # Normalize
    root = root.lower()
    full_word = full_word.lower()

    # Special case: root == full word (no affixes)
    if root == full_word:
        return {'root': root, 'affixes': [], 'prefix': None, 'ending': None, 'pos': 'unknown'}

    # Check for prefix
    prefix = None
    word_without_prefix = full_word
    for pfx in PREFIXES:
        if full_word.startswith(pfx) and full_word[len(pfx):].startswith(root):
            prefix = pfx
            word_without_prefix = full_word[len(pfx):]
            break

    # Find root position
    if root not in word_without_prefix:
        # Root doesn't match - maybe truncated (verbo → verb)
        # Try with root variations
        for i in range(len(root), 0, -1):
            if root[:i] in word_without_prefix:
                root = root[:i]
                break
        else:
            # Can't find root, return simple decomposition
            return {
                'root': root,
                'affixes': [],
                'prefix': prefix,
                'ending': None,
                'pos': 'unknown'
            }

    root_idx = word_without_prefix.index(root)
    after_root = word_without_prefix[root_idx + len(root):]

    if not after_root:
        return {'root': root, 'affixes': [], 'prefix': prefix, 'ending': None, 'pos': 'unknown'}

    # Extract ending
    ending = None
    pos = 'unknown'
    middle_part = after_root

    for end, pos_val in sorted(ENDINGS.items(), key=lambda x: -len(x[0])):
        if after_root.endswith(end):
            ending = end
            pos = pos_val
            middle_part = after_root[:-len(end)]
            break

    # Extract suffixes from middle part
    affixes = []
    remaining = middle_part

    while remaining:
        found = False
        for suffix in SUFFIXES:
            if remaining.startswith(suffix):
                affixes.append(suffix)
                remaining = remaining[len(suffix):]
                found = True
                break

        if not found:
            # Can't parse further, treat as single suffix
            if remaining:
                affixes.append(remaining)
            break

    return {
        'root': root,
        'affixes': affixes,
        'prefix': prefix,
        'ending': ending,
        'pos': pos
    }


def convert_triple(triple: Dict) -> Optional[Dict]:
    """Convert root-level triple to word-level format."""
    subject_word = triple.get('subject_full', '')
    verb_word = triple.get('verb_full', '')
    object_word = triple.get('object_full', '')

    subject_root = triple.get('subject_root', '')
    verb_root = triple.get('verb_root', '')
    object_root = triple.get('object_root', '')

    # Decompose each word
    subject_decomp = extract_affixes_simple(subject_root, subject_word)
    verb_decomp = extract_affixes_simple(verb_root, verb_word)
    object_decomp = extract_affixes_simple(object_root, object_word)

    # Add text and status fields
    subject_decomp['text'] = subject_word
    subject_decomp['status'] = 'sukceso'
    verb_decomp['text'] = verb_word
    verb_decomp['status'] = 'sukceso'
    object_decomp['text'] = object_word
    object_decomp['status'] = 'sukceso'

    # Create word-level triple
    word_level = {
        # Backward compatibility (root-level)
        'subject_root': subject_root,
        'verb_root': verb_root,
        'object_root': object_root,
        'subject_full': subject_word,
        'verb_full': verb_word,
        'object_full': object_word,

        # Word-level decomposition
        'subject': subject_decomp,
        'verb': verb_decomp,
        'object': object_decomp,

        # Metadata
        'relation_type': triple.get('relation_type', 'SVO'),
        'source': triple.get('source', 'unknown'),
        'sentence': triple.get('sentence', ''),
        'sentence_id': triple.get('sentence_id'),
        'confidence': triple.get('confidence', 1.0)
    }

    return word_level


def main():
    parser = argparse.ArgumentParser(description='Add simple word decomposition to triples')
    parser.add_argument('--input', type=Path, required=True, help='Input triples (root-level)')
    parser.add_argument('--output', type=Path, required=True, help='Output triples (word-level)')
    parser.add_argument('--limit', type=int, help='Limit number of triples to convert')

    args = parser.parse_args()

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Converting {args.input} to word-level format...")

    converted = 0

    with jsonlines.open(args.input) as reader, \
         jsonlines.open(args.output, mode='w') as writer:

        for i, triple in enumerate(reader):
            if args.limit and i >= args.limit:
                break

            word_level = convert_triple(triple)
            writer.write(word_level)
            converted += 1

            if (i + 1) % 100 == 0:
                logging.info(f"Processed {i+1} triples")

    logging.info(f"\nDone! Converted {converted} triples")
    logging.info(f"Output: {args.output}")


if __name__ == '__main__':
    main()
