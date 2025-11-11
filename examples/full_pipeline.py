#!/usr/bin/env python3
"""
Full Pipeline Example

Demonstrates the complete Klareco pipeline:
1. Language identification
2. Translation to Esperanto
3. Parsing to AST
4. Intent classification
5. Response generation
"""
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.pipeline import KlarecoPipeline


def example_1_english_input():
    """Process an English query through the full pipeline."""
    print("=" * 60)
    print("Example 1: English Input")
    print("=" * 60)

    pipeline = KlarecoPipeline()
    query = "The dog sees the cat"

    print(f"\nInput: '{query}'")
    print("\nProcessing through pipeline...")

    trace = pipeline.run(query)

    print("\nPipeline Steps:")
    for i, step in enumerate(trace.steps, 1):
        print(f"  {i}. {step['step']}: {step.get('description', 'N/A')}")

    print(f"\nFinal Result:")
    print(f"  {trace.final_result}")

    print("\nDetailed Trace Available:")
    print(f"  Trace ID: {trace.trace_id}")
    print(f"  Total steps: {len(trace.steps)}")


def example_2_spanish_input():
    """Process a Spanish query."""
    print("\n" + "=" * 60)
    print("Example 2: Spanish Input")
    print("=" * 60)

    pipeline = KlarecoPipeline()
    query = "El perro ve al gato"

    print(f"\nInput: '{query}'")
    print("\nProcessing through pipeline...")

    trace = pipeline.run(query)

    print(f"\nLanguage Detected:")
    front_door_step = next(s for s in trace.steps if s['step'] == 'FrontDoor')
    print(f"  {front_door_step['outputs']['detected_lang']}")

    print(f"\nTranslated to Esperanto:")
    print(f"  {front_door_step['outputs']['esperanto_text']}")

    print(f"\nFinal Result:")
    print(f"  {trace.final_result}")


def example_3_direct_esperanto():
    """Process Esperanto directly (no translation needed)."""
    print("\n" + "=" * 60)
    print("Example 3: Direct Esperanto Input")
    print("=" * 60)

    pipeline = KlarecoPipeline()
    query = "Mi volas manĝi."

    print(f"\nInput: '{query}'")
    print("\nProcessing through pipeline...")

    trace = pipeline.run(query)

    # Get the AST from the Parser step
    parser_step = next(s for s in trace.steps if s['step'] == 'Parser')
    ast = parser_step['outputs']['ast']

    print(f"\nParsed AST:")
    print(f"  Subject: {ast['subjekto']['kerno']['radiko']}")
    print(f"  Verb: {ast['verbo']['radiko']} ({ast['verbo']['modo']})")

    # Get intent
    intent_step = next(s for s in trace.steps if s['step'] == 'IntentClassifier')
    intent = intent_step['outputs']['intent']

    print(f"\nDetected Intent:")
    print(f"  Category: {intent['category']}")
    print(f"  Confidence: {intent['confidence']}")

    print(f"\nFinal Result:")
    print(f"  {trace.final_result}")


def example_4_stop_after_parser():
    """Run the pipeline but stop after parsing (for debugging)."""
    print("\n" + "=" * 60)
    print("Example 4: Stop After Parser (Debugging)")
    print("=" * 60)

    pipeline = KlarecoPipeline()
    query = "La programisto programas."

    print(f"\nInput: '{query}'")
    print("Stopping pipeline after Parser step...")

    trace = pipeline.run(query, stop_after="Parser")

    print(f"\nSteps Executed: {len(trace.steps)}")
    for step in trace.steps:
        print(f"  - {step['step']}")

    print("\nThis is useful for:")
    print("  - Debugging parser issues")
    print("  - Testing individual pipeline stages")
    print("  - Examining intermediate representations")


def example_5_export_full_trace():
    """Export the complete execution trace as JSON."""
    print("\n" + "=" * 60)
    print("Example 5: Export Full Execution Trace")
    print("=" * 60)

    pipeline = KlarecoPipeline()
    query = "The programmer programs"

    print(f"\nInput: '{query}'")

    trace = pipeline.run(query)

    # Export to JSON
    trace_json = trace.to_json()
    trace_path = Path("example_trace.json")
    trace_path.write_text(trace_json)

    print(f"\nTrace exported to: {trace_path}")
    print(f"File size: {len(trace_json)} bytes")

    print("\nTrace contents (first 500 chars):")
    print(trace_json[:500] + "...")

    print("\nThe trace contains:")
    print("  - Every pipeline step with inputs/outputs")
    print("  - Timestamps for each operation")
    print("  - Complete AST representation")
    print("  - All intermediate transformations")

    print(f"\nThis enables:")
    print("  - Complete traceability (audit trail)")
    print("  - Debugging and error analysis")
    print("  - Future learning loop (Phase 9)")


def main():
    """Run all examples."""
    print("\n")
    print("*" * 60)
    print("  KLARECO: Full Pipeline Examples")
    print("*" * 60)
    print("\nNote: First run may be slow due to model downloads (~1GB)")
    print("Subsequent runs will be much faster.\n")

    try:
        example_1_english_input()
        example_2_spanish_input()
        example_3_direct_esperanto()
        example_4_stop_after_parser()
        example_5_export_full_trace()

        print("\n" + "=" * 60)
        print("Examples Complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("  - See examples/basic_parsing.py for parser details")
        print("  - Try examples/morpheme_analysis.py for deep dives")
        print("  - Read DESIGN.md for the full system architecture")
        print("\n")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: Make sure you have installed all dependencies:")
        print("  pip install -r requirements.txt")
        print("\nAnd that translation models have been downloaded")
        print("(first run downloads ~1GB of models)")


if __name__ == "__main__":
    main()
