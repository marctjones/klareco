#!/usr/bin/env python3
"""
Klareco End-to-End Demo

Demonstrates the complete neuro-symbolic AI pipeline:
1. Multi-language input → Translation to Esperanto
2. Parsing to symbolic AST
3. Intent classification via Gating Network
4. Expert routing via Orchestrator
5. Symbolic/neural processing by specialized experts
6. Natural language response

This shows the power of Klareco's architecture where:
- Symbolic experts handle deterministic tasks (math, time, grammar)
- Neural components handle semantic tasks (translation, future RAG)
- AST provides structured representation enabling both approaches
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.pipeline import KlarecoPipeline


def print_separator(title=""):
    """Print a fancy separator."""
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}\n")
    else:
        print(f"{'='*70}\n")


def demo_query(pipeline, query, description):
    """
    Run a demo query through the pipeline.

    Args:
        pipeline: KlarecoPipeline instance
        query: Input query in any language
        description: Human-readable description
    """
    print_separator(description)
    print(f"📝 Input: \"{query}\"")
    print()

    # Run pipeline
    trace = pipeline.run(query)

    if trace.error:
        print(f"❌ ERROR: {trace.error}")
        return

    # Extract key information from trace
    steps = trace.steps

    # Language detection
    front_door_step = next((s for s in steps if s['name'] == 'FrontDoor'), None)
    if front_door_step:
        lang = front_door_step['outputs'].get('original_lang', 'unknown')
        esperanto_text = front_door_step['outputs'].get('processed_text', '')
        print(f"🌍 Detected language: {lang}")
        if lang != 'eo':
            print(f"🔄 Translated to: \"{esperanto_text}\"")
        print()

    # Parser
    parser_step = next((s for s in steps if s['name'] == 'Parser'), None)
    if parser_step:
        print("🌲 Parsed to symbolic AST")
        print()

    # Orchestrator
    orchestrator_step = next((s for s in steps if s['name'] == 'Orchestrator'), None)
    if orchestrator_step:
        outputs = orchestrator_step['outputs']
        intent = outputs.get('intent', 'unknown')
        expert = outputs.get('expert', 'none')
        confidence = outputs.get('confidence', 0)

        print(f"🎯 Intent: {intent}")
        print(f"🤖 Expert: {expert}")
        print(f"📊 Confidence: {confidence:.2%}")
        print()

    # Final response
    response = trace.final_response
    print(f"💬 Response: \"{response}\"")
    print()


def main():
    """Run the Klareco demo."""
    print_separator("KLARECO - Neuro-Symbolic AI Demo")

    print("Klareco combines:")
    print("  • Symbolic processing (AST-based, deterministic, traceable)")
    print("  • Neural components (translation, future semantic understanding)")
    print("  • Expert system routing (specialized handlers for different tasks)")
    print()
    print("Initializing pipeline...")
    print()

    # Initialize pipeline with orchestrator
    pipeline = KlarecoPipeline(use_orchestrator=True)

    print("✅ Pipeline ready with Expert System Orchestrator")
    print()

    # ==================================================================
    # Math Queries
    # ==================================================================
    print_separator("DEMO 1: Mathematical Computation (MathExpert)")

    demo_query(
        pipeline,
        "Kiom estas du plus tri?",
        "Math Query (Esperanto)"
    )

    demo_query(
        pipeline,
        "What is ten minus four?",
        "Math Query (English → Translation → Math)"
    )

    # ==================================================================
    # Date/Time Queries
    # ==================================================================
    print_separator("DEMO 2: Temporal Queries (DateExpert)")

    demo_query(
        pipeline,
        "Kiu tago estas hodiaŭ?",
        "Date Query (Esperanto)"
    )

    demo_query(
        pipeline,
        "What time is it?",
        "Time Query (English → Translation → Date)"
    )

    # ==================================================================
    # Grammar Queries
    # ==================================================================
    print_separator("DEMO 3: Grammar Analysis (GrammarExpert)")

    demo_query(
        pipeline,
        "Eksplik la gramatikon de la frazo",
        "Grammar Query (Esperanto)"
    )

    # ==================================================================
    # General Queries (showing fallback)
    # ==================================================================
    print_separator("DEMO 4: General Query (Fallback Routing)")

    demo_query(
        pipeline,
        "La hundo vidas la katon",
        "Simple Declarative Sentence"
    )

    # ==================================================================
    # Summary
    # ==================================================================
    print_separator("DEMO COMPLETE")

    print("What you just saw:")
    print()
    print("✅ Multi-language input (English, Esperanto)")
    print("✅ Automatic translation to Esperanto")
    print("✅ Symbolic parsing to AST (100% deterministic)")
    print("✅ Intent classification via Gating Network")
    print("✅ Smart routing to specialized experts")
    print("✅ Expert-specific processing:")
    print("   • MathExpert: Symbolic computation")
    print("   • DateExpert: Temporal reasoning")
    print("   • GrammarExpert: AST analysis")
    print("✅ Natural language response generation")
    print()
    print("Key Advantages:")
    print("  🎯 Traceable: Every step logged, inspectable")
    print("  ⚡ Fast: Symbolic processing where possible")
    print("  🎨 Extensible: Add new experts easily")
    print("  🔒 Safe: Input validation, AST complexity checks")
    print("  🌍 Multi-lingual: Translation layer handles any language")
    print()
    print("Next Steps:")
    print("  → Add neural RAG expert for factual questions")
    print("  → Add Dictionary expert for word lookup")
    print("  → Add multi-step execution loop")
    print("  → Add memory system (STM/LTM)")
    print()
    print_separator()


if __name__ == '__main__':
    main()
