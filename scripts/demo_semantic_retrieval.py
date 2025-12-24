#!/usr/bin/env python3
"""
Demo: Semantic Role-Based Retrieval

Shows how semantic signatures improve retrieval by filtering
based on agent/action/patient roles rather than just keywords.

Example:
    Query: "Kiu vidas la katon?" (Who sees the cat?)

    OLD (keyword-based): Returns sentences with "vid" and "kat"
        - "La kato vidas la hundon." (cat sees dog) ❌ Cat is AGENT here!
        - "Mi vidas la katon." (I see the cat) ✅

    NEW (semantic role-based): Returns sentences where cat is PATIENT
        - "Mi vidas la katon." (I see the cat) ✅
        - "La hundo vidas la katon." (dog sees cat) ✅
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.semantic_signatures import extract_signature, signature_to_string
from klareco.semantic_search import SemanticIndex


def main():
    print("=" * 70)
    print("Semantic Role-Based Retrieval Demo")
    print("=" * 70)
    print()

    # Load semantic index
    index_dir = Path("data/semantic_index")
    if not index_dir.exists():
        print(f"ERROR: Semantic index not found at {index_dir}")
        print("Run: ./scripts/run_semantic_index_builder.sh --clean")
        sys.exit(1)

    print(f"Loading semantic index from {index_dir}...")
    index = SemanticIndex(index_dir)
    stats = index.stats()
    print(f"  Unique signatures: {stats['unique_signatures']:,}")
    print(f"  Total sentences: {stats['total_sentences']:,}")
    print()

    # Test queries demonstrating role disambiguation
    test_queries = [
        # Query, English translation, expected role
        ("Kiu vidas la katon?", "Who sees the cat?", "cat should be PATIENT"),
        ("Kion vidas la hundo?", "What does the dog see?", "dog should be AGENT"),
        ("Kiu amas Frodon?", "Who loves Frodo?", "Frodo should be PATIENT"),
        ("Kion faras Gandalf?", "What does Gandalf do?", "Gandalf should be AGENT"),
    ]

    for eo_query, en_query, explanation in test_queries:
        print("-" * 70)
        print(f"Query: {eo_query}")
        print(f"       ({en_query})")
        print(f"       {explanation}")
        print()

        # Parse and extract signature
        ast = parse(eo_query)
        sig = extract_signature(ast)
        sig_str = signature_to_string(sig)

        print(f"Extracted signature: {sig_str}")
        print(f"  Agent:   {sig[0] or '*'}")
        print(f"  Action:  {sig[1] or '*'}")
        print(f"  Patient: {sig[2] or '*'}")
        print()

        # Search
        results = index.search(sig, k=5)

        if results:
            print("Top matches:")
            for i, r in enumerate(results, 1):
                print(f"  {i}. [{r['score']:.2f}] {r['text'][:70]}...")
                print(f"       Signature: {r['signature']}")
        else:
            print("  No matches found")

        print()

    # Show signature statistics
    print("=" * 70)
    print("Signature Statistics")
    print("=" * 70)

    # Most common signatures
    print("\nMost common semantic patterns:")
    sig_counts = {}
    for sig_str, ids in index.signatures.items():
        sig_counts[sig_str] = len(ids)

    top_sigs = sorted(sig_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    for sig_str, count in top_sigs:
        print(f"  {sig_str:30} -> {count:,} sentences")

    print()
    print("Demo complete!")


if __name__ == '__main__':
    main()
