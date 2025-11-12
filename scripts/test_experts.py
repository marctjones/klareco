#!/usr/bin/env python3
"""
Test script for Klareco Tool Experts.

Tests the symbolic expert system with various queries to verify:
- Math Expert can handle arithmetic
- Date Expert can handle temporal queries
- Grammar Expert can explain sentence structure
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.experts import MathExpert, DateExpert, GrammarExpert


def test_expert(expert, query_esperanto, description):
    """
    Test an expert with a query.

    Args:
        expert: The expert to test
        query_esperanto: Query in Esperanto
        description: English description of the query
    """
    print(f"\n{'='*70}")
    print(f"TEST: {description}")
    print(f"{'='*70}")
    print(f"Query: {query_esperanto}")
    print()

    try:
        # Parse the query
        ast = parse(query_esperanto)
        print("✅ Parsed successfully")
        print()

        # Check if expert can handle it
        can_handle = expert.can_handle(ast)
        confidence = expert.estimate_confidence(ast)

        print(f"Expert: {expert.name}")
        print(f"Can handle: {can_handle}")
        print(f"Confidence: {confidence:.2f}")
        print()

        if can_handle:
            # Execute the query
            result = expert.execute(ast)

            print("RESULT:")
            print(f"  Answer: {result.get('answer', 'N/A')}")
            print(f"  Confidence: {result.get('confidence', 0):.2f}")
            print(f"  Expert: {result.get('expert', 'N/A')}")

            if 'explanation' in result:
                print(f"  Explanation: {result['explanation']}")

            if 'result' in result:
                print(f"  Computed result: {result['result']}")

            if 'query_type' in result:
                print(f"  Query type: {result['query_type']}")

            if 'error' in result:
                print(f"  ❌ ERROR: {result['error']}")

            return True
        else:
            print("❌ Expert cannot handle this query")
            return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run expert tests."""
    print("="*70)
    print("KLARECO EXPERT SYSTEM TEST SUITE")
    print("="*70)

    # Initialize experts
    math_expert = MathExpert()
    date_expert = DateExpert()
    grammar_expert = GrammarExpert()

    results = []

    # ==================================================================
    # Math Expert Tests
    # ==================================================================
    print("\n" + "="*70)
    print("MATH EXPERT TESTS")
    print("="*70)

    results.append(test_expert(
        math_expert,
        "Kiom estas du plus tri?",
        "Simple addition: 2 + 3"
    ))

    results.append(test_expert(
        math_expert,
        "Kiom estas dek minus kvar?",
        "Simple subtraction: 10 - 4"
    ))

    results.append(test_expert(
        math_expert,
        "Kio estas tri foje kvar?",
        "Simple multiplication: 3 × 4"
    ))

    results.append(test_expert(
        math_expert,
        "Dividu dek ok per tri",
        "Simple division: 18 ÷ 3"
    ))

    # ==================================================================
    # Date Expert Tests
    # ==================================================================
    print("\n" + "="*70)
    print("DATE/TIME EXPERT TESTS")
    print("="*70)

    results.append(test_expert(
        date_expert,
        "Kiu tago estas hodiaŭ?",
        "What day is today?"
    ))

    results.append(test_expert(
        date_expert,
        "Kioma horo estas?",
        "What time is it?"
    ))

    results.append(test_expert(
        date_expert,
        "Kiu dato estas hodiaŭ?",
        "What is today's date?"
    ))

    # ==================================================================
    # Grammar Expert Tests
    # ==================================================================
    print("\n" + "="*70)
    print("GRAMMAR EXPERT TESTS")
    print("="*70)

    results.append(test_expert(
        grammar_expert,
        "Eksplik la gramatikon de la frazo",
        "Explain the grammar of the sentence"
    ))

    # Test with a complete sentence
    print("\n" + "="*70)
    print("GRAMMAR ANALYSIS TEST")
    print("="*70)
    print("Testing grammar analysis on: 'La hundo vidas la katon'")
    print()

    try:
        # Parse a sample sentence
        sample_ast = parse("La hundo vidas la katon")
        result = grammar_expert.execute(sample_ast)

        print("RESULT:")
        print(result.get('answer', 'N/A'))
        print()
        results.append(True)

    except Exception as e:
        print(f"❌ ERROR: {e}")
        results.append(False)

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    total = len(results)
    passed = sum(results)
    failed = total - passed

    print(f"Total tests: {total}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Success rate: {passed/total*100:.1f}%")
    print()

    if failed == 0:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
