#!/usr/bin/env python3
"""
Round-Trip Conversion: Text → AST → Text

Demonstrates parsing Esperanto to AST and reconstructing the text.
This shows that the AST preserves all linguistic information.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.deparser import deparse


def round_trip(sentence: str):
    """Parse a sentence and reconstruct it."""
    print(f"\n  Original:      '{sentence}'")

    # Parse: text → AST
    ast = parse(sentence)

    # Deparse: AST → text
    reconstructed = deparse(ast)

    print(f"  Reconstructed: '{reconstructed}'")

    # Check if they match (after normalization)
    original_normalized = sentence.lower().strip('.')
    reconstructed_normalized = reconstructed.lower().strip('.')

    match = original_normalized == reconstructed_normalized
    status = "✓" if match else "✗"
    print(f"  Match: {status}")

    return match


def main():
    """Run round-trip tests on various sentences."""
    print("\n")
    print("*" * 60)
    print("  KLARECO: Round-Trip Conversion Examples")
    print("*" * 60)
    print("\nDemonstrating: Text → AST → Text conversion")
    print("This proves the AST preserves all linguistic information.\n")

    print("=" * 60)
    print("Example 1: Simple Sentences")
    print("=" * 60)

    sentences = [
        "La hundo vidas la katon.",
        "Mi amas katon.",
        "La kato manĝas.",
    ]

    results = []
    for sentence in sentences:
        results.append(round_trip(sentence))

    print("\n" + "=" * 60)
    print("Example 2: Sentences with Adjectives")
    print("=" * 60)

    sentences = [
        "Mi amas la grandan hundon.",
        "La bona kato dormas.",
        "Grandaj hundoj kuras.",
    ]

    for sentence in sentences:
        results.append(round_trip(sentence))

    print("\n" + "=" * 60)
    print("Example 3: Complex Morphology")
    print("=" * 60)

    sentences = [
        "Malgrandaj katoj dormas.",
        "La programisto programas.",
        "Sanaj hundoj kuras rapide.",
    ]

    for sentence in sentences:
        results.append(round_trip(sentence))

    print("\n" + "=" * 60)
    print("Example 4: Different Tenses")
    print("=" * 60)

    sentences = [
        "La hundo vidis la katon.",  # Past
        "La hundo vidas la katon.",  # Present
        "La hundo vidos la katon.",  # Future
    ]

    for sentence in sentences:
        results.append(round_trip(sentence))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total = len(results)
    passed = sum(results)

    print(f"\nTotal sentences tested: {total}")
    print(f"Successful round-trips: {passed}")
    print(f"Success rate: {passed/total*100:.1f}%")

    print("\n" + "=" * 60)
    print("Why This Matters")
    print("=" * 60)
    print("""
The AST (Abstract Syntax Tree) is not just a parse tree - it's a
COMPLETE representation of the linguistic structure:

1. **Lossless**: Every morpheme is preserved
   - Roots, prefixes, suffixes
   - Case, number, tense, mood
   - Agreement relationships

2. **Queryable**: The AST can be queried symbolically
   - "Find all accusative nouns" → Simple filter
   - "Find all future tense verbs" → No LLM needed
   - "Check agreement" → Deterministic validation

3. **Composable**: ASTs can be combined and transformed
   - Negate a sentence: Add 'ne' before verb
   - Make plural: Change 'nombro' field
   - Change tense: Swap verb ending

4. **Efficient**: Stored as structured data, not text
   - Database-friendly (JSON, SQL)
   - Graph-friendly (for GNN encoder)
   - No re-parsing needed

This is the foundation for Klareco's neuro-symbolic approach:
symbolic operations on structure, neural operations on semantics.
    """)

    print("\n" + "=" * 60)
    print("Examples Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - See examples/morpheme_analysis.py for AST internals")
    print("  - Try examples/full_pipeline.py for end-to-end processing")
    print("  - Read 16RULES.MD to understand the grammar")
    print("\n")


if __name__ == "__main__":
    main()
