#!/usr/bin/env python3
"""
Test Claude Code as LLM Backend

This script demonstrates using Claude Code (the AI running your code) as the
LLM backend for Klareco's Factoid QA Expert, without requiring any API keys.

Usage:
    python scripts/test_claude_llm.py

    # With mock responses for testing
    python scripts/test_claude_llm.py --mock

When running in Claude Code, the script will print LLM requests and Claude
can respond directly in the conversation.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.claude_code_llm import create_claude_code_provider, create_mock_provider
from klareco.experts.factoid_qa_expert import create_factoid_qa_expert
from klareco.orchestrator import create_orchestrator_with_experts
from klareco.pipeline import KlarecoPipeline


def test_factoid_qa_with_claude(use_mock=False):
    """
    Test Factoid QA Expert using Claude Code as LLM backend.

    Args:
        use_mock: If True, use mock responses for testing
    """
    print("="*80)
    print("TESTING FACTOID QA WITH CLAUDE CODE LLM")
    print("="*80)
    print()

    # Create LLM provider
    if use_mock:
        print("Using MOCK LLM provider for testing")
        mock_responses = {
            "Frodo": (
                "Frodo Sakvil-Benso estas la ĉefa protagonisto de "
                "'La Mastro de l' Ringoj'. Li estas hobito, kiu "
                "heredis la Unu Ringon de sia onklo Bilbo."
            ),
            "Gandalfo": (
                "Gandalfo estas saĝulo (wizardo) kaj unu el la ĉefaj "
                "karakteroj en la verkoj de Tolkien. Li helpas Frodon "
                "en lia kvesto detrui la Unu Ringon."
            ),
            "Esperanto": (
                "Esperanto estas internacia planlingvo, kreita de "
                "D-ro L. L. Zamenhof en 1887. Ĝi estas desegnita por "
                "esti facile lernebla kaj kulture neŭtrala."
            ),
            "hobito": (
                "Hobitoj estas fikcia raso en la verkoj de J.R.R. Tolkien. "
                "Ili estas malgrandaj homecaj estaĵoj, kutime 60-120 cm altaj, "
                "kun haraj piedoj kaj ŝato de paco kaj komforto."
            )
        }
        llm_provider = create_mock_provider(mock_responses)
    else:
        print("Using CLAUDE CODE as LLM provider (interactive)")
        print("When LLM requests appear, Claude will respond in conversation")
        llm_provider = create_claude_code_provider()

    print(f"Provider type: {llm_provider.provider_type.value}")
    print()

    # Create Factoid QA Expert with the provider
    print("Creating Factoid QA Expert...")
    expert = create_factoid_qa_expert(llm_provider=llm_provider)
    print(f"✓ Expert created: {expert.name}")
    print(f"✓ RAG system available: {expert.rag_system is not None}")
    print()

    # Test questions
    test_questions = [
        "Kiu estas Frodo?",  # Who is Frodo?
        "Kiu estas Gandalfo?",  # Who is Gandalf?
        "Kio estas hobito?",  # What is a hobbit?
    ]

    for i, question in enumerate(test_questions, 1):
        print("="*80)
        print(f"TEST {i}/{len(test_questions)}: {question}")
        print("="*80)
        print()

        # Parse question
        print(f"1. Parsing question...")
        ast = parse(question)
        print(f"   ✓ Parsed successfully")
        print()

        # Check if expert can handle
        can_handle = expert.can_handle(ast)
        confidence = expert.estimate_confidence(ast)
        print(f"2. Expert evaluation:")
        print(f"   Can handle: {can_handle}")
        print(f"   Confidence: {confidence:.2f}")
        print()

        if not can_handle:
            print("   ⚠️  Expert cannot handle this question")
            print()
            continue

        # Execute expert
        print(f"3. Executing expert (retrieving + generating)...")
        context = {'original_text': question}
        result = expert.execute(ast, context)

        print(f"   ✓ Execution complete")
        print()

        # Display result
        print("4. RESULT:")
        print(f"   Expert: {result.get('expert')}")
        print(f"   Confidence: {result.get('confidence', 0):.2f}")

        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
        else:
            print(f"   Sources: {result.get('num_sources', 0)} documents retrieved")
            if 'stage1_stats' in result:
                stats = result['stage1_stats']
                print(f"   Stage 1: {stats.get('total_candidates', 0)} keyword matches")

            print()
            print("   ANSWER:")
            print("   " + "-"*76)
            answer = result.get('answer', '(No answer)')
            for line in answer.split('\n'):
                print(f"   {line}")
            print("   " + "-"*76)

        print()


def test_full_pipeline_with_claude(use_mock=False):
    """
    Test full Klareco pipeline with Claude Code LLM.

    Args:
        use_mock: If True, use mock responses
    """
    print("="*80)
    print("TESTING FULL PIPELINE WITH CLAUDE CODE LLM")
    print("="*80)
    print()

    # Create LLM provider
    if use_mock:
        mock_responses = {
            "Frodo": "Frodo estas la ĉefa karaktero de Tolkien...",
            "Gandalfo": "Gandalfo estas saĝulo...",
        }
        llm_provider = create_mock_provider(mock_responses)
    else:
        llm_provider = create_claude_code_provider()

    # Create orchestrator with all experts
    print("Creating orchestrator with all experts...")
    orchestrator = create_orchestrator_with_experts(llm_provider=llm_provider)
    print(f"✓ Orchestrator created with {len(orchestrator.experts)} experts:")
    for expert in orchestrator.experts:
        print(f"  - {expert.name}")
    print()

    # Create pipeline
    print("Creating pipeline...")
    pipeline = KlarecoPipeline(orchestrator=orchestrator)
    print("✓ Pipeline ready")
    print()

    # Test queries
    test_queries = [
        "Kiu estas Frodo?",  # Factoid QA
        "Kiom estas 15 plus 27?",  # Math
        "Kio estas la dato hodiaŭ?",  # Date
    ]

    for query in test_queries:
        print("="*80)
        print(f"QUERY: {query}")
        print("="*80)

        try:
            trace = pipeline.run(query)

            print(f"\n✓ Pipeline completed")
            print(f"  Steps: {len(trace.steps)}")
            print(f"  Success: {trace.success}")

            # Show result
            if trace.success and trace.result:
                print(f"\n  RESULT:")
                print(f"  {trace.result}")

        except Exception as e:
            print(f"\n❌ Error: {e}")

        print()


def main():
    parser = argparse.ArgumentParser(description="Test Claude Code as LLM backend")
    parser.add_argument('--mock', action='store_true',
                        help='Use mock LLM responses for testing')
    parser.add_argument('--full-pipeline', action='store_true',
                        help='Test full pipeline instead of just Factoid QA')

    args = parser.parse_args()

    if args.full_pipeline:
        test_full_pipeline_with_claude(use_mock=args.mock)
    else:
        test_factoid_qa_with_claude(use_mock=args.mock)


if __name__ == "__main__":
    main()
