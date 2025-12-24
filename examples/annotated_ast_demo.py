#!/usr/bin/env python3
"""
Annotated AST Visualization Demo

Shows complex Esperanto sentences with richly annotated ASTs,
demonstrating how deterministic parsing extracts all grammatical structure.
"""
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_word_annotation(word, indent=2):
    """Print detailed annotation for a single word."""
    prefix = " " * indent

    print(f"{prefix}Word: '{word.get('plena_vorto', 'N/A')}'")
    print(f"{prefix}├─ Root (radiko): {word.get('radiko', 'N/A')}")

    if word.get('prefikso'):
        print(f"{prefix}├─ Prefix: {word['prefikso']} (modifies meaning)")

    if word.get('sufiksoj'):
        print(f"{prefix}├─ Suffixes: {', '.join(word['sufiksoj'])} (derivational)")

    vortspeco = word.get('vortspeco', 'N/A')
    print(f"{prefix}├─ Part of Speech: {vortspeco}")

    # Case and number (nouns/adjectives)
    if word.get('nombro'):
        print(f"{prefix}├─ Number: {word['nombro']} ({'plural' if word['nombro'] == 'pluralo' else 'singular'})")

    if word.get('kazo'):
        case = word['kazo']
        case_explanation = {
            'nominativo': 'subject/predicate',
            'akuzativo': 'direct object/direction'
        }.get(case, case)
        print(f"{prefix}├─ Case: {case} ({case_explanation})")

    # Tense/mood (verbs)
    if word.get('tempo'):
        tense = word['tempo']
        tense_explanation = {
            'prezenco': 'present -as',
            'pasinteco': 'past -is',
            'futuro': 'future -os'
        }.get(tense, tense)
        print(f"{prefix}├─ Tense: {tense} ({tense_explanation})")

    if word.get('modo'):
        mood = word['modo']
        mood_explanation = {
            'infinitivo': 'infinitive -i',
            'imperativo': 'imperative -u',
            'kondiĉa': 'conditional -us'
        }.get(mood, mood)
        print(f"{prefix}├─ Mood: {mood} ({mood_explanation})")

    print(f"{prefix}└─ Parse Status: {word.get('parse_status', 'unknown')}")


def print_vortgrupo_annotation(vortgrupo, role, indent=0):
    """Print detailed annotation for a noun phrase (vortgrupo)."""
    prefix = " " * indent

    print(f"\n{prefix}[{role.upper()}] Noun Phrase (vortgrupo):")

    if vortgrupo.get('artikolo'):
        print(f"{prefix}  Article: '{vortgrupo['artikolo']}' (definite)")

    if vortgrupo.get('priskriboj'):
        print(f"{prefix}  Adjectives ({len(vortgrupo['priskriboj'])}):")
        for i, adj in enumerate(vortgrupo['priskriboj'], 1):
            print(f"{prefix}    Adjective {i}:")
            print_word_annotation(adj, indent + 6)

    if vortgrupo.get('kerno'):
        print(f"{prefix}  Core Noun:")
        print_word_annotation(vortgrupo['kerno'], indent + 4)


def visualize_complex_ast(sentence, description=""):
    """Visualize a complex sentence with full AST annotation."""
    print_separator("=")
    if description:
        print(f"SENTENCE: {description}")
        print_separator("-")

    print(f"\nOriginal Text:")
    print(f"  '{sentence}'")
    print()

    # Parse
    ast = parse(sentence)

    # Sentence type
    print(f"Sentence Type: {ast.get('tipo', 'unknown')}")
    print()

    # Subject
    if ast.get('subjekto'):
        if ast['subjekto'].get('tipo') == 'vortgrupo':
            print_vortgrupo_annotation(ast['subjekto'], 'subject', indent=0)
        else:
            print("\n[SUBJECT] Word:")
            print_word_annotation(ast['subjekto'], indent=2)

    # Verb
    if ast.get('verbo'):
        print("\n[VERB]:")
        print_word_annotation(ast['verbo'], indent=2)

    # Object
    if ast.get('objekto'):
        if ast['objekto'].get('tipo') == 'vortgrupo':
            print_vortgrupo_annotation(ast['objekto'], 'object', indent=0)
        else:
            print("\n[OBJECT] Word:")
            print_word_annotation(ast['objekto'], indent=2)

    # Additional words
    if ast.get('aliaj') and len(ast['aliaj']) > 0:
        print(f"\n[OTHER WORDS] ({len(ast['aliaj'])}):")
        for i, word in enumerate(ast['aliaj'], 1):
            print(f"  Word {i}:")
            print_word_annotation(word, indent=4)

    # Parse statistics
    if ast.get('parse_statistics'):
        stats = ast['parse_statistics']
        print(f"\nParse Statistics:")
        print(f"  Total words: {stats.get('total_words', 0)}")
        print(f"  Successfully parsed: {stats.get('esperanto_words', 0)}")
        print(f"  Success rate: {stats.get('success_rate', 0):.1%}")

    print()


def example_1_simple_with_adjective():
    """Example 1: Simple sentence with adjective agreement."""
    visualize_complex_ast(
        "La granda hundo vidas la malgrandan katon.",
        "Adjective Agreement (adjectives match noun case)"
    )

    print("\nKEY INSIGHT:")
    print("  'granda' has nominative (no -n) because 'hundo' is nominative (subject)")
    print("  'malgrandan' has accusative (-n) because 'katon' is accusative (object)")
    print("  ➜ Adjectives MUST agree with their nouns in case and number")
    print()


def example_2_morphological_complexity():
    """Example 2: Complex morphology with prefixes and suffixes."""
    visualize_complex_ast(
        "La maljunaj gepatroj revenis al la malnova domo.",
        "Complex Morphology (prefixes, suffixes, multiple adjectives)"
    )

    print("\nKEY INSIGHTS:")
    print("  'maljunaj' = mal (opposite) + jun (young) + a (adj) + j (plural)")
    print("            = 'old' (literally: opposite-of-young)")
    print("  'gepatroj' = ge (both genders) + patr (parent) + o (noun) + j (plural)")
    print("            = 'parents' (both mother and father)")
    print("  'revenis' = re (again) + ven (come) + is (past tense)")
    print("            = 'returned' (came again)")
    print("  ➜ Every morpheme is deterministically identified!")
    print()


def example_3_all_grammatical_features():
    """Example 3: Sentence showcasing many grammatical features."""
    visualize_complex_ast(
        "La belaj birdoj flugas rapide super la verdajn arbojn.",
        "Multiple Grammatical Features (plural, accusative, adverb, preposition)"
    )

    print("\nKEY INSIGHTS:")
    print("  'belaj' (plural, nominative) agrees with 'birdoj' (plural, nominative)")
    print("  'verdajn' (plural, accusative -jn) agrees with 'arbojn' (plural, accusative -jn)")
    print("  'rapide' (adverb -e) modifies the verb 'flugas'")
    print("  'super' (preposition) governs 'arbojn' (accusative shows direction)")
    print("  ➜ All relationships are explicit and deterministic!")
    print()


def example_4_complex_nested():
    """Example 4: Complex sentence with nested structures."""
    visualize_complex_ast(
        "Mi amas la grandan belan hundon kiu ludas en la ĝardeno.",
        "Complex Structure (relative clause with 'kiu')"
    )

    print("\nKEY INSIGHTS:")
    print("  'grandan belan' = TWO adjectives (both accusative -n) modifying 'hundon'")
    print("  'kiu' = relative pronoun (who/which)")
    print("  Adjective agreement: grandan + belan + hundon (all accusative)")
    print("  ➜ Multiple adjectives, all must agree with noun!")
    print()


def example_5_tense_variations():
    """Example 5: Different tenses and moods."""
    print_separator("=")
    print("EXAMPLE 5: Verb Tenses and Moods")
    print_separator("=")
    print()

    sentences = [
        ("La hundo manĝas.", "Present tense (-as)"),
        ("La hundo manĝis.", "Past tense (-is)"),
        ("La hundo manĝos.", "Future tense (-os)"),
        ("La hundo manĝus.", "Conditional mood (-us)"),
        ("Manĝu la hundon!", "Imperative mood (-u)"),
    ]

    for sentence, description in sentences:
        print(f"  {description}:")
        print(f"    '{sentence}'")
        ast = parse(sentence)
        verb = ast.get('verbo')
        if verb and isinstance(verb, dict):
            if verb.get('tempo'):
                print(f"    Verb: {verb['radiko']} + {verb['tempo']}")
            elif verb.get('modo'):
                print(f"    Verb: {verb['radiko']} + {verb['modo']}")
        else:
            # Imperative might be in different location
            print(f"    (Verb structure varies for imperative)")
        print()

    print("KEY INSIGHT:")
    print("  Esperanto has 6 verb forms (3 tenses + 3 moods)")
    print("  ZERO irregular verbs - every verb follows the same pattern!")
    print("  ➜ 100% deterministic tense/mood detection from ending")
    print()


def example_6_show_full_json():
    """Example 6: Show complete JSON AST for inspection."""
    print_separator("=")
    print("EXAMPLE 6: Complete JSON AST")
    print_separator("=")
    print()

    sentence = "La belaj birdoj flugas."
    print(f"Sentence: '{sentence}'")
    print()

    ast = parse(sentence)
    print("Full AST as JSON (this is what gets passed to neural models):")
    print()
    print(json.dumps(ast, indent=2, ensure_ascii=False))
    print()

    print("KEY INSIGHT:")
    print("  This structured AST is the INPUT to our neural models!")
    print("  Traditional LLMs get: ['The', 'beautiful', 'birds', 'fly', '.']")
    print("  Klareco gets: Structured tree with explicit grammar relations")
    print("  ➜ Neural models learn on STRUCTURE, not raw tokens!")
    print()


def main():
    """Run all annotated AST examples."""
    print()
    print("*" * 80)
    print("  KLARECO: Annotated AST Demonstration")
    print("  Showing how deterministic parsing extracts rich grammatical structure")
    print("*" * 80)
    print()

    example_1_simple_with_adjective()

    input("Press Enter to continue to Example 2...")
    example_2_morphological_complexity()

    input("Press Enter to continue to Example 3...")
    example_3_all_grammatical_features()

    input("Press Enter to continue to Example 4...")
    example_4_complex_nested()

    input("Press Enter to continue to Example 5...")
    example_5_tense_variations()

    input("Press Enter to see complete JSON AST...")
    example_6_show_full_json()

    print_separator("=")
    print("DEMONSTRATION COMPLETE")
    print_separator("=")
    print()
    print("What you just saw:")
    print("  ✅ Deterministic morphological analysis (prefix+root+suffix+ending)")
    print("  ✅ Explicit grammatical features (case, number, tense, mood)")
    print("  ✅ Structural relationships (subject, verb, object)")
    print("  ✅ Adjective agreement detection")
    print("  ✅ Complex nested structures")
    print("  ✅ Full JSON AST for neural processing")
    print()
    print("This AST is the foundation for AST-native neural models:")
    print("  → Root Embeddings learn semantic similarity")
    print("  → Graph Transformers learn reasoning on AST structure")
    print("  → All grammar is FREE (0 learned parameters)")
    print()
    print("Next steps:")
    print("  - Train models that operate on these ASTs (not raw text)")
    print("  - Use pure Esperanto corpus (26K sentences)")
    print("  - Focus learned capacity on reasoning, not grammar")
    print()


if __name__ == "__main__":
    main()
