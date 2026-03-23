#!/usr/bin/env python3
"""
Convert Root-Level SVO Triples to Word-Level Format

Quick utility to add word decomposition to existing SVO triples for testing.
Uses the parser to decompose full words into root + affixes.
"""

import argparse
import jsonlines
from pathlib import Path
from typing import Dict, Optional
import logging

from klareco.parser import parse

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


def decompose_word(word: str) -> Optional[Dict]:
    """
    Parse a word to get its decomposition.

    Returns:
        {'text': word, 'root': ..., 'affixes': [...], 'pos': ..., ...}
    """
    if not word:
        return None

    # Parse the word as a single-word sentence
    try:
        ast = parse(word)

        # Extract the word node from AST
        if not ast or ast.get('analizstato') == 'malsukceso':
            # Try as part of simple phrase
            ast = parse(f"La {word}")
            if not ast or ast.get('analizstato') == 'malsukceso':
                return None

        # Find first successful word node
        word_node = None

        def find_word_node(node):
            nonlocal word_node
            if not node:
                return

            if isinstance(node, dict):
                if node.get('tipo') == 'vorto' and node.get('analizstato') in ['sukceso', 'neplu_analiz']:
                    if word.lower() in node.get('originala_teksto', '').lower():
                        word_node = node
                        return

                # Recurse
                for key, value in node.items():
                    if isinstance(value, (dict, list)):
                        find_word_node(value)
            elif isinstance(node, list):
                for item in node:
                    find_word_node(item)

        find_word_node(ast)

        if not word_node:
            return None

        return {
            'text': word,
            'root': word_node.get('radiko', ''),
            'affixes': word_node.get('sufiksoj', []),
            'prefix': word_node.get('prefikso'),
            'pos': word_node.get('vortspeco', 'unknown'),
            'ending': word_node.get('vortspeco_finaĵo'),
            'status': word_node.get('analizstato')
        }

    except Exception as e:
        logging.debug(f"Failed to decompose '{word}': {e}")
        return None


def convert_triple(triple: Dict) -> Optional[Dict]:
    """Convert root-level triple to word-level format."""
    subject_word = triple.get('subject_full', '')
    verb_word = triple.get('verb_full', '')
    object_word = triple.get('object_full', '')

    # Decompose each word
    subject_decomp = decompose_word(subject_word)
    verb_decomp = decompose_word(verb_word)
    object_decomp = decompose_word(object_word)

    if not all([subject_decomp, verb_decomp, object_decomp]):
        return None

    # Create word-level triple
    word_level = {
        # Backward compatibility (root-level)
        'subject_root': triple.get('subject_root'),
        'verb_root': triple.get('verb_root'),
        'object_root': triple.get('object_root'),
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
    parser = argparse.ArgumentParser(description='Convert root-level triples to word-level')
    parser.add_argument('--input', type=Path, required=True, help='Input triples (root-level)')
    parser.add_argument('--output', type=Path, required=True, help='Output triples (word-level)')
    parser.add_argument('--limit', type=int, help='Limit number of triples to convert')

    args = parser.parse_args()

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Converting {args.input} to word-level format...")

    converted = 0
    failed = 0

    with jsonlines.open(args.input) as reader, \
         jsonlines.open(args.output, mode='w') as writer:

        for i, triple in enumerate(reader):
            if args.limit and converted >= args.limit:
                break

            word_level = convert_triple(triple)

            if word_level:
                writer.write(word_level)
                converted += 1
            else:
                failed += 1

            if (i + 1) % 100 == 0:
                logging.info(f"Processed {i+1} triples: {converted} converted, {failed} failed")

    logging.info(f"\nDone!")
    logging.info(f"  Converted: {converted}")
    logging.info(f"  Failed: {failed}")
    logging.info(f"  Success rate: {converted/(converted+failed)*100:.1f}%")
    logging.info(f"  Output: {args.output}")


if __name__ == '__main__':
    main()
