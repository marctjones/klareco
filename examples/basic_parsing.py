#!/usr/bin/env python3
"""
Basic Esperanto Parsing Examples

This script demonstrates how to parse Esperanto sentences and explore the AST.
"""
import sys
import json
from pathlib import Path

# Add parent directory to path to import klareco
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse, parse_word


def example_1_simple_word():
    """Parse a single word and examine its morphemes."""
    print("=" * 60)
    print("Example 1: Parsing a Simple Word")
    print("=" * 60)

    word = "hundon"
    print(f"\nInput: '{word}'")

    ast = parse_word(word)

    print("\nMorpheme Analysis:")
    print(f"  Root (radiko):        {ast['radiko']}")
    print(f"  Part of Speech:       {ast['vortspeco']}")
    print(f"  Number (nombro):      {ast['nombro']}")
    print(f"  Case (kazo):          {ast['kazo']}")

    print("\nExplanation:")
    print("  'hundon' = hund (root) + o (noun) + n (accusative)")
    print("  This is 'dog' in the accusative case (direct object)")


def example_2_complex_word():
    """Parse a word with prefixes and suffixes."""
    print("\n" + "=" * 60)
    print("Example 2: Word with Affix")
    print("=" * 60)

    word = "malsanuloj"
    print(f"\nInput: '{word}'")

    ast = parse_word(word)

    print("\nMorpheme Analysis:")
    print(f"  Prefix:               {ast['prefikso']}")
    print(f"  Root:                 {ast['radiko']}")
    print(f"  Suffixes:             {ast['sufiksoj']}")
    print(f"  Part of Speech:       {ast['vortspeco']}")
    print(f"  Number:               {ast['nombro']}")

    print("\nExplanation:")
    print("  'malsanuloj' = mal (opposite) + san (health) + ul (person) + o (noun) + j (plural)")
    print("  This means 'sick people' (literally: 'opposite-health-persons')")


def example_3_simple_sentence():
    """Parse a complete sentence."""
    print("\n" + "=" * 60)
    print("Example 3: Simple Sentence")
    print("=" * 60)

    sentence = "La hundo vidas la katon."
    print(f"\nInput: '{sentence}'")

    ast = parse(sentence)

    print("\nSentence Structure:")
    print(f"  Type: {ast['tipo']}")
    print(f"\n  Subject (subjekto):")
    print(f"    Core: {ast['subjekto']['kerno']['radiko']}")

    print(f"\n  Verb (verbo):")
    print(f"    Root: {ast['verbo']['radiko']}")
    print(f"    Tense: {ast['verbo']['tempo']}")

    print(f"\n  Object (objekto):")
    print(f"    Core: {ast['objekto']['kerno']['radiko']}")
    print(f"    Case: {ast['objekto']['kerno']['kazo']}")

    print("\nExplanation:")
    print("  Subject: 'La hundo' (the dog) - nominative case")
    print("  Verb: 'vidas' (sees) - present tense")
    print("  Object: 'la katon' (the cat) - accusative case (-n)")


def example_4_complex_sentence():
    """Parse a sentence with adjectives."""
    print("\n" + "=" * 60)
    print("Example 4: Sentence with Adjectives")
    print("=" * 60)

    sentence = "Mi amas la grandan hundon."
    print(f"\nInput: '{sentence}'")

    ast = parse(sentence)

    print("\nSentence Structure:")
    print(f"  Subject: {ast['subjekto']['kerno']['radiko']}")
    print(f"  Verb: {ast['verbo']['radiko']} ({ast['verbo']['tempo']})")
    print(f"  Object: {ast['objekto']['kerno']['radiko']}")

    # Show the adjective modifying the object
    if ast['objekto']['priskriboj']:
        print(f"\n  Adjectives modifying object:")
        for adj in ast['objekto']['priskriboj']:
            print(f"    - {adj['radiko']} (case: {adj['kazo']})")

    print("\nExplanation:")
    print("  'grandan' has accusative -n because it modifies 'hundon' (accusative)")
    print("  Adjectives agree with nouns in case and number (Rule 4)")


def example_5_json_output():
    """Show the full AST as JSON."""
    print("\n" + "=" * 60)
    print("Example 5: Full AST as JSON")
    print("=" * 60)

    sentence = "La kato manĝas."
    print(f"\nInput: '{sentence}'")

    ast = parse(sentence)

    print("\nComplete AST (JSON format):")
    print(json.dumps(ast, indent=2, ensure_ascii=False))

    print("\nNote: This AST can be:")
    print("  - Stored in a database")
    print("  - Queried programmatically")
    print("  - Converted back to text (see round_trip.py)")
    print("  - Used for semantic analysis (future: GNN encoder)")


def main():
    """Run all examples."""
    print("\n")
    print("*" * 60)
    print("  KLARECO: Basic Esperanto Parsing Examples")
    print("*" * 60)

    example_1_simple_word()
    example_2_complex_word()
    example_3_simple_sentence()
    example_4_complex_sentence()
    example_5_json_output()

    print("\n" + "=" * 60)
    print("Examples Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Try examples/full_pipeline.py for end-to-end processing")
    print("  - See examples/round_trip.py for AST ↔ text conversion")
    print("  - Read examples/morpheme_analysis.py for detailed parsing")
    print("\n")


if __name__ == "__main__":
    main()
