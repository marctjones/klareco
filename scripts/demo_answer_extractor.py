#!/usr/bin/env python3
"""
Demo: AST-Based Answer Extraction

Shows how the deterministic answer extractor works on different question types.

Usage:
    python scripts/demo_answer_extractor.py
    python scripts/demo_answer_extractor.py --query "Kiu fondis Esperanton?" --doc "Zamenhof fondis Esperanton."
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.rag.answer_extractor import ASTAnswerExtractor


def demo_extraction(query: str, document: str):
    """
    Demo answer extraction on a query-document pair.

    Args:
        query: Question text
        document: Document text containing answer
    """
    print("=" * 70)
    print(f"Query: {query}")
    print(f"Document: {document}")
    print("=" * 70)

    # Parse
    query_ast = parse(query)
    doc_ast = parse(document)

    # Extract answer
    extractor = ASTAnswerExtractor()
    answer = extractor.extract_answer(query_ast, doc_ast, document)

    if answer:
        print(f"\n✓ Answer extracted:")
        print(f"  Text: {answer['text']}")
        print(f"  Confidence: {answer['confidence']:.2f}")
        print(f"  Method: {answer['method']}")
        print(f"  Explanation: {answer['explanation']}")
    else:
        print("\n✗ No answer found")

    print()


def main():
    parser = argparse.ArgumentParser(description="Demo AST-based answer extraction")
    parser.add_argument('--query', type=str, help='Question to answer')
    parser.add_argument('--doc', type=str, help='Document containing answer')
    args = parser.parse_args()

    if args.query and args.doc:
        # Single demo
        demo_extraction(args.query, args.doc)
    else:
        # Multiple demos showing different question types
        print("\n" + "=" * 70)
        print("AST-Based Answer Extraction Demo")
        print("=" * 70)
        print("\nDemonstrates deterministic answer extraction for different question types.\n")

        examples = [
            # WHO questions
            {
                'query': 'Kiu fondis Esperanton?',
                'doc': 'Zamenhof fondis Esperanton en 1887.',
            },
            {
                'query': 'Kiu skribis la libron?',
                'doc': 'La verkulo skribis la libron.',
            },

            # WHAT questions
            {
                'query': 'Kio estas Esperanto?',
                'doc': 'Esperanto estas internacia planlingvo.',
            },
            {
                'query': 'Kion Zamenhof kreis?',
                'doc': 'Zamenhof kreis Esperanton.',
            },

            # WHERE questions
            {
                'query': 'Kie naskiĝis Zamenhof?',
                'doc': 'Zamenhof naskiĝis en Bjalistoko.',
            },
            {
                'query': 'Kie li loĝas?',
                'doc': 'Li loĝas en Varsovio.',
            },

            # WHEN questions
            {
                'query': 'Kiam estis fondita Esperanto?',
                'doc': 'Esperanto estis fondita en 1887.',
            },
            {
                'query': 'Kiam li venos?',
                'doc': 'Li venos morgaŭ.',
            },

            # HOW MANY questions
            {
                'query': 'Kiom da homoj parolas Esperanton?',
                'doc': 'Du milionoj da homoj parolas Esperanton.',
            },

            # No answer (verb mismatch)
            {
                'query': 'Kiu fondis Esperanton?',
                'doc': 'Zamenhof parolis pri lingvoj.',
            },
        ]

        for example in examples:
            demo_extraction(example['query'], example['doc'])

        print("\n" + "=" * 70)
        print("Demo complete!")
        print("=" * 70)
        print("\nNote: This is the FIRST tier in the cascading answer extraction system.")
        print("Fallback methods (heuristic, learned) will be added in future issues.")


if __name__ == '__main__':
    main()
