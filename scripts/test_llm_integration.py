#!/usr/bin/env python3
"""
Test LLM Integration - Demonstrates auto-detected LLM provider

This script tests:
1. LLM provider auto-detection (should detect Claude Code)
2. Summarize Expert with LLM backend
3. Factoid QA Expert with RAG + LLM
4. Full orchestrator integration

Usage:
    python scripts/test_llm_integration.py
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from klareco.llm_provider import get_llm_provider, LLMProviderType
from klareco.parser import parse_esperanto
from klareco.orchestrator import create_orchestrator_with_experts

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_llm_provider_detection():
    """Test that LLM provider auto-detection works"""
    print("\n" + "="*80)
    print("TEST 1: LLM Provider Auto-Detection")
    print("="*80)

    provider = get_llm_provider()
    print(f"✓ Detected provider: {provider.provider_type.value}")

    if provider.provider_type == LLMProviderType.CLAUDE_CODE:
        print("✓ Running in Claude Code - will use Claude itself for LLM tasks!")
    elif provider.provider_type == LLMProviderType.ANTHROPIC_API:
        print("✓ Will use Anthropic API for LLM tasks")
    else:
        print(f"✓ Will use {provider.provider_type.value} for LLM tasks")

    return provider


def test_summarize_expert():
    """Test Summarize Expert with LLM"""
    print("\n" + "="*80)
    print("TEST 2: Summarize Expert")
    print("="*80)

    from klareco.experts.summarize_expert import create_summarize_expert

    expert = create_summarize_expert()
    print(f"✓ Summarize Expert initialized")
    print(f"  Provider: {expert.llm_provider.provider_type.value}")
    print(f"  Capabilities: {expert.capabilities}")

    # Test can_handle
    query = "Resumu la tekston."  # "Summarize the text"
    ast = parse_esperanto(query)

    can_handle = expert.can_handle(ast)
    print(f"✓ Can handle '{query}': {can_handle}")

    if can_handle:
        print(f"\nNOTE: To test summarization, call expert.handle(ast, context)")
        print(f"      where context contains 'text_to_summarize'")


def test_factoid_qa_expert():
    """Test Factoid QA Expert with RAG + LLM"""
    print("\n" + "="*80)
    print("TEST 3: Factoid QA Expert")
    print("="*80)

    from klareco.experts.factoid_qa_expert import create_factoid_qa_expert

    expert = create_factoid_qa_expert()
    print(f"✓ Factoid QA Expert initialized")
    print(f"  Provider: {expert.llm_provider.provider_type.value}")
    print(f"  RAG System: {'Available' if expert.rag_system else 'Not available'}")
    print(f"  Capabilities: {expert.capabilities}")

    # Test can_handle
    queries = [
        "Kio estas Esperanto?",  # What is Esperanto?
        "Kiu kreis Esperanton?",  # Who created Esperanto?
    ]

    for query in queries:
        ast = parse_esperanto(query)
        can_handle = expert.can_handle(ast)
        print(f"✓ Can handle '{query}': {can_handle}")


def test_orchestrator_integration():
    """Test full orchestrator with all experts including LLM-powered ones"""
    print("\n" + "="*80)
    print("TEST 4: Orchestrator Integration")
    print("="*80)

    orchestrator = create_orchestrator_with_experts()

    print(f"✓ Orchestrator created with {len(orchestrator.experts)} experts:")
    for expert_name in orchestrator.list_experts():
        print(f"  - {expert_name}")

    # Test intent classification
    print("\n" + "-"*80)
    print("Testing Intent Classification:")
    print("-"*80)

    test_queries = [
        ("Kiom estas du plus tri?", "calculation_request"),
        ("Kio estas hodiaŭ?", "temporal_query"),
        ("Eksplikaci la strukturon.", "grammar_query"),
        ("Kio estas Esperanto?", "factoid_question"),
        ("Resumu la tekston.", "summarization_request"),
    ]

    for query, expected_intent in test_queries:
        try:
            ast = parse_esperanto(query)
            classification = orchestrator.gating_network.classify(ast)
            intent = classification['intent']
            confidence = classification['confidence']

            status = "✓" if intent == expected_intent else "✗"
            print(f"{status} '{query}'")
            print(f"    Expected: {expected_intent}, Got: {intent} (conf: {confidence:.2f})")

        except Exception as e:
            print(f"✗ '{query}' - Error: {e}")


def main():
    print("\n" + "="*80)
    print("KLARECO LLM INTEGRATION TEST")
    print("="*80)
    print("\nThis test demonstrates:")
    print("1. Auto-detection of Claude Code environment")
    print("2. LLM-powered experts (Summarize, Factoid QA)")
    print("3. Integration with existing symbolic experts")
    print("4. Automatic fallback to APIs when Claude Code unavailable")
    print("="*80)

    try:
        # Run tests
        test_llm_provider_detection()
        test_summarize_expert()
        test_factoid_qa_expert()
        test_orchestrator_integration()

        print("\n" + "="*80)
        print("ALL TESTS COMPLETED!")
        print("="*80)
        print("\nNOTE: Actual LLM generation requires either:")
        print("1. Running in Claude Code (auto-detected)")
        print("2. Setting ANTHROPIC_API_KEY environment variable")
        print("3. Setting OPENAI_API_KEY environment variable")
        print("\nFor Claude Code: LLM requests will be shown and you can respond via:")
        print("  python scripts/claude_llm_respond.py 'Your response here'")
        print("="*80 + "\n")

    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
