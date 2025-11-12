#!/usr/bin/env python3
"""
Test script for Klareco Orchestrator integration.

Tests the complete expert system with:
- Intent classification (Gating Network)
- Expert routing (Orchestrator)
- All three symbolic experts (Math, Date, Grammar)
- Fallback routing mechanism
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.orchestrator import create_orchestrator_with_experts


def test_orchestrator_query(orchestrator, query_esperanto, description):
    """
    Test the orchestrator with a query.

    Args:
        orchestrator: The orchestrator to test
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

        # Route through orchestrator
        response = orchestrator.route(ast)

        # Display result
        print("ORCHESTRATOR RESPONSE:")
        print(f"  Intent: {response.get('intent', 'N/A')}")
        print(f"  Intent Confidence: {response.get('intent_confidence', 0):.2f}")
        print(f"  Expert: {response.get('expert', 'N/A')}")
        print(f"  Answer: {response.get('answer', 'N/A')}")
        print(f"  Confidence: {response.get('confidence', 0):.2f}")

        if 'explanation' in response:
            print(f"  Explanation: {response['explanation']}")

        if 'result' in response:
            print(f"  Computed result: {response['result']}")

        if 'error' in response:
            print(f"  ❌ ERROR: {response['error']}")
            return False

        return True

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run orchestrator tests."""
    print("="*70)
    print("KLARECO ORCHESTRATOR INTEGRATION TEST SUITE")
    print("="*70)

    # Create orchestrator with all experts
    print("\nInitializing orchestrator...")
    orchestrator = create_orchestrator_with_experts()

    print(f"\n{orchestrator}")
    print(f"Registered experts: {', '.join(orchestrator.list_experts())}")
    print(f"Registered intents: {', '.join(orchestrator.list_intents())}")

    results = []

    # ==================================================================
    # Math Expert Tests (via Orchestrator)
    # ==================================================================
    print("\n" + "="*70)
    print("MATH EXPERT TESTS (via Orchestrator)")
    print("="*70)

    results.append(test_orchestrator_query(
        orchestrator,
        "Kiom estas du plus tri?",
        "Simple addition: 2 + 3"
    ))

    results.append(test_orchestrator_query(
        orchestrator,
        "Kiom estas dek minus kvar?",
        "Simple subtraction: 10 - 4"
    ))

    results.append(test_orchestrator_query(
        orchestrator,
        "Kio estas tri foje kvar?",
        "Simple multiplication: 3 × 4"
    ))

    # ==================================================================
    # Date Expert Tests (via Orchestrator)
    # ==================================================================
    print("\n" + "="*70)
    print("DATE/TIME EXPERT TESTS (via Orchestrator)")
    print("="*70)

    results.append(test_orchestrator_query(
        orchestrator,
        "Kiu tago estas hodiaŭ?",
        "What day is today?"
    ))

    results.append(test_orchestrator_query(
        orchestrator,
        "Kioma horo estas?",
        "What time is it?"
    ))

    results.append(test_orchestrator_query(
        orchestrator,
        "Kiu dato estas hodiaŭ?",
        "What is today's date?"
    ))

    # ==================================================================
    # Grammar Expert Tests (via Orchestrator)
    # ==================================================================
    print("\n" + "="*70)
    print("GRAMMAR EXPERT TESTS (via Orchestrator)")
    print("="*70)

    results.append(test_orchestrator_query(
        orchestrator,
        "Eksplik la gramatikon de la frazo",
        "Explain the grammar of the sentence"
    ))

    # ==================================================================
    # Intent Classification Tests
    # ==================================================================
    print("\n" + "="*70)
    print("INTENT CLASSIFICATION TESTS")
    print("="*70)

    test_cases = [
        ("Kiom estas du plus tri?", "calculation_request"),
        ("Kiu tago estas hodiaŭ?", "temporal_query"),
        ("Eksplik la gramatikon", "grammar_query"),
        ("Kio estas Esperanto?", "factoid_question"),
    ]

    for query, expected_intent in test_cases:
        print(f"\nQuery: {query}")
        ast = parse(query)
        classification = orchestrator.gating_network.classify(ast)
        actual_intent = classification['intent']
        confidence = classification['confidence']

        if actual_intent == expected_intent:
            print(f"✅ Correct: {actual_intent} (confidence: {confidence:.2f})")
            results.append(True)
        else:
            print(f"❌ Wrong: expected '{expected_intent}', got '{actual_intent}'")
            results.append(False)

    # ==================================================================
    # Fallback Routing Test
    # ==================================================================
    print("\n" + "="*70)
    print("FALLBACK ROUTING TEST")
    print("="*70)

    # This query should be classified as 'factoid_question' but
    # should be routed to DateExpert via fallback
    results.append(test_orchestrator_query(
        orchestrator,
        "Kiam estas la hodiaŭa tago?",
        "Temporal query with question word (tests fallback)"
    ))

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
