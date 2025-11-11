#!/usr/bin/env python3
"""
Morpheme Analysis Deep Dive

Explores the detailed morphological analysis that Klareco performs.
Shows how every word is decomposed into its constituent morphemes.
"""
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse_word


def analyze_word(word: str, explanation: str = ""):
    """Analyze a word's morphology in detail."""
    print(f"\nWord: '{word}'")
    if explanation:
        print(f"Meaning: {explanation}")

    try:
        ast = parse_word(word)

        print("\nMorphological Breakdown:")

        # Build the morpheme chain
        morphemes = []

        if ast['prefikso']:
            morphemes.append(f"PREFIX: {ast['prefikso']}")

        morphemes.append(f"ROOT: {ast['radiko']}")

        if ast['sufiksoj']:
            for suf in ast['sufiksoj']:
                morphemes.append(f"SUFFIX: {suf}")

        # Part of speech ending
        pos_endings = {
            'substantivo': 'o',
            'adjektivo': 'a',
            'adverbo': 'e',
            'verbo': ast.get('tempo', ast.get('modo', 'unknown'))
        }
        if ast['vortspeco'] in pos_endings:
            ending = pos_endings[ast['vortspeco']]
            morphemes.append(f"POS: {ending} ({ast['vortspeco']})")

        # Grammatical markers
        if ast['nombro'] == 'pluralo':
            morphemes.append("PLURAL: j")

        if ast['kazo'] == 'akuzativo':
            morphemes.append("ACCUSATIVE: n")

        # Display morpheme chain
        print("  " + " + ".join(morphemes))

        # Show grammatical properties
        print("\nGrammatical Properties:")
        print(f"  Part of Speech: {ast['vortspeco']}")
        print(f"  Number:         {ast['nombro']}")
        print(f"  Case:           {ast['kazo']}")

        if 'tempo' in ast:
            print(f"  Tense:          {ast['tempo']}")
        if 'modo' in ast:
            print(f"  Mood:           {ast['modo']}")

        print("\nComplete AST:")
        print(f"  {json.dumps(ast, indent=4, ensure_ascii=False)}")

    except ValueError as e:
        print(f"\nError: {e}")


def main():
    """Run morpheme analysis examples."""
    print("\n")
    print("*" * 60)
    print("  KLARECO: Morpheme Analysis Deep Dive")
    print("*" * 60)
    print("\nEsperanto's transparent morphology enables deterministic parsing.")
    print("Every word can be decomposed into its constituent morphemes.\n")

    print("=" * 60)
    print("Example 1: Simple Noun")
    print("=" * 60)
    analyze_word("hundo", "dog")

    print("\n" + "=" * 60)
    print("Example 2: Noun with Grammatical Markers")
    print("=" * 60)
    analyze_word("hundojn", "dogs (accusative)")

    print("\n" + "=" * 60)
    print("Example 3: Word with Prefix")
    print("=" * 60)
    analyze_word("malsano", "sickness (opposite of health)")

    print("\n" + "=" * 60)
    print("Example 4: Word with Suffix")
    print("=" * 60)
    analyze_word("sanulo", "healthy person")

    print("\n" + "=" * 60)
    print("Example 5: Complex Word (Prefix + Suffix)")
    print("=" * 60)
    analyze_word("malsanulo", "sick person")

    print("\n" + "=" * 60)
    print("Example 6: Verb Forms")
    print("=" * 60)

    print("\nPresent tense:")
    analyze_word("vidas", "sees (present)")

    print("\n" + "-" * 60)
    print("Past tense:")
    analyze_word("vidis", "saw (past)")

    print("\n" + "-" * 60)
    print("Future tense:")
    analyze_word("vidos", "will see (future)")

    print("\n" + "-" * 60)
    print("Conditional mood:")
    analyze_word("vidus", "would see (conditional)")

    print("\n" + "-" * 60)
    print("Volitive mood:")
    analyze_word("vidu", "may/let see (volitive/imperative)")

    print("\n" + "-" * 60)
    print("Infinitive:")
    analyze_word("vidi", "to see (infinitive)")

    print("\n" + "=" * 60)
    print("Example 7: Adjective Agreement")
    print("=" * 60)

    print("\nNominative singular:")
    analyze_word("granda", "big")

    print("\n" + "-" * 60)
    print("Nominative plural:")
    analyze_word("grandaj", "big (plural)")

    print("\n" + "-" * 60)
    print("Accusative singular:")
    analyze_word("grandan", "big (accusative)")

    print("\n" + "-" * 60)
    print("Accusative plural:")
    analyze_word("grandajn", "big (accusative plural)")

    print("\n" + "=" * 60)
    print("The 16 Rules in Action")
    print("=" * 60)
    print("""
Every morphological analysis follows Zamenhof's 16 Rules:

Rule 1:  No definite article (only 'la')
Rule 2:  Nouns end in -o, plural -j, accusative -n
Rule 3:  Adjectives end in -a, agree with nouns
Rule 4:  Cardinal numbers don't change form
Rule 5:  Pronouns (mi, vi, li, ŝi, ĝi, ni, vi, ili)
Rule 6:  Verbs never change for person/number
Rule 7:  Adverbs end in -e
Rule 8:  All prepositions govern nominative
Rule 9:  Every word as written is pronounced
Rule 10: Stress always on penultimate syllable
Rule 11: Compound words formed by simple juxtaposition
Rule 12: Only one negative per clause
Rule 13: To show direction: accusative -n
Rule 14: Every preposition has definite meaning
Rule 15: Foreign words take Esperanto form
Rule 16: Final -o and article can be omitted (poetry)

This regularity means:
- NO irregular verbs (unlike English: go/went, be/was)
- NO noun declension tables (unlike German, Latin)
- NO gender agreement (unlike Spanish, French)
- NO exceptions to memorize

Every grammatical feature is EXPLICIT in the morphology.
This is what enables symbolic, deterministic parsing.
    """)

    print("\n" + "=" * 60)
    print("Examples Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Read 16RULES.MD for complete grammar specification")
    print("  - See examples/round_trip.py to verify AST completeness")
    print("  - Try examples/basic_parsing.py for sentence parsing")
    print("\n")


if __name__ == "__main__":
    main()
