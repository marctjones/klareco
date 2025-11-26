#!/usr/bin/env python3
"""
Quick Query Script - Test Klareco pipeline with concise output

Focus on pure Esperanto processing. Optionally translate output for readability.

Usage:
    python scripts/quick_query.py "Kiu estas Frodo?"       # Pure Esperanto (default)
    python scripts/quick_query.py "Kiu estas Frodo?" --translate  # Show English translations
    python scripts/quick_query.py --debug                  # Show debug logging
    python scripts/quick_query.py --show-stage1            # Show keyword filter details
"""

import sys
import argparse
import logging
from pathlib import Path

# Suppress logging early if not in debug mode
# Check args before importing anything
if '--debug' not in sys.argv:
    logging.basicConfig(level=logging.CRITICAL)
    logging.disable(logging.WARNING)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.pipeline import KlarecoPipeline
from klareco.translator import TranslationService


def setup_logging(debug: bool):
    """Configure logging based on debug flag."""
    if debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(levelname)s [%(name)s]: %(message)s'
        )
    else:
        # Suppress all logging - set root logger and disable propagation
        logging.basicConfig(level=logging.CRITICAL)
        # Disable all loggers
        logging.getLogger().setLevel(logging.CRITICAL)
        logging.disable(logging.INFO)


def format_line(eo_text: str, show_translation: bool, translator=None, indent: str = "") -> str:
    """
    Format a line with optional English translation.

    If show_translation=True: Show ONLY English (translated)
    If show_translation=False: Show ONLY Esperanto (original)
    """
    if not show_translation:
        # Pure Esperanto mode
        return f"{indent}{eo_text}"

    if translator:
        try:
            # Translate mode - show ONLY English
            # Split multi-line text and translate each line separately
            lines = eo_text.split('\n')
            result_lines = []

            for line in lines:
                line = line.strip()
                if not line:
                    result_lines.append("")  # Preserve blank lines
                    continue

                # Translate this line and show ONLY the English
                en_text = translator.translate(line, 'eo', 'en')
                result_lines.append(f"{indent}{en_text}")

            return '\n'.join(result_lines)
        except:
            # If translation fails, fall back to original
            return f"{indent}{eo_text}"

    # No translator available, show original
    return f"{indent}{eo_text}"


def main():
    parser = argparse.ArgumentParser(
        description='Quick query test for Klareco - Pure Esperanto processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s "Kiu estas Frodo?"              # Pure Esperanto output
  %(prog)s "Kiu estas Frodo?" --translate  # Show English translations
  %(prog)s --show-stage1                   # Show keyword filtering details
        '''
    )
    parser.add_argument(
        'query',
        nargs='?',
        default='Kiu estas Gandalfo?',
        help='Query to process (default: "Kiu estas Gandalfo?")'
    )
    parser.add_argument(
        '--translate',
        action='store_true',
        help='Translate OUTPUT to English (for readability - does NOT affect processing)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Show debug logging'
    )
    parser.add_argument(
        '--show-stage1',
        action='store_true',
        help='Show Stage 1 keyword filter results before Stage 2 reranking'
    )

    args = parser.parse_args()

    # Setup logging (already configured at module level if not debug)
    if args.debug:
        setup_logging(True)

    show_translation = args.translate  # Only translate if explicitly requested
    translator = TranslationService() if show_translation else None

    print("=" * 70)
    print("KLARECO QUICK QUERY - Pure Esperanto Processing")
    print("=" * 70)
    print()

    # Input
    print(f"📝 Query: {args.query}")
    if show_translation:
        print("   [English translations enabled for OUTPUT]")
    else:
        print("   [Pure Esperanto mode - use --translate to show English]")
    print()

    # Initialize pipeline
    if not args.debug:
        print("⚙️  Loading pipeline...")
    pipeline = KlarecoPipeline(use_orchestrator=True)
    if not args.debug:
        print("✓ Pipeline ready")
    print()

    # Process
    try:
        trace = pipeline.run(args.query)

        # Extract key information from trace
        steps = trace.steps

        # Show each stage
        print("🔄 PIPELINE STAGES")
        print("-" * 70)

        for i, step in enumerate(steps, 1):
            step_name = step.get('name', step.get('step', 'unknown'))

            if step_name == "FrontDoor":
                outputs = step.get('outputs', {})
                detected_lang = outputs.get('detected_language', 'unknown')
                esperanto_text = outputs.get('esperanto_text', '')

                if detected_lang != 'eo':
                    print(f"{i}. 🌍 Language Detection: {detected_lang} → eo")
                    if esperanto_text:
                        print(f"   Esperanto: {esperanto_text}")
                else:
                    print(f"{i}. 🌍 Language: Esperanto ✓")
                print()

            elif step_name == "Parser":
                print(f"{i}. 🌲 Parsing → AST created")
                if args.debug:
                    outputs = step.get('outputs', {})
                    print(f"   AST nodes: {outputs.get('node_count', 'N/A')}")
                print()

            elif step_name == "SafetyMonitor":
                # Skip safety checks in output unless debug
                if args.debug:
                    print(f"{i}. 🛡️  Safety check passed")
                    print()

            elif step_name == "Orchestrator":
                outputs = step.get('outputs', {})
                intent = outputs.get('intent', 'unknown')
                expert = outputs.get('expert_used', 'N/A')
                confidence = outputs.get('confidence', 0)

                print(f"{i}. 🎯 Intent: {intent}")
                if expert and expert != 'N/A':
                    print(f"   Expert: {expert}")
                    if confidence > 0:
                        print(f"   Confidence: {confidence:.0%}")

                # Show retrieval info if available
                full_response = outputs.get('full_response', {})

                # Show stage1 stats (always, if available)
                if 'stage1_stats' in full_response:
                    stage1 = full_response['stage1_stats']
                    print(f"   Stage 1: {stage1.get('total_candidates', 0)} keyword matches")
                    print(f"   Stage 2: Reranked top {stage1.get('candidates_reranked', 0)}")

                if 'sources' in full_response:
                    num_sources = len(full_response['sources'])
                    print(f"   Final: {num_sources} results")
                print()

                # Show full stage1 results if requested
                if args.show_stage1 and 'stage1_stats' in full_response:
                    stage1 = full_response['stage1_stats']
                    print()
                    print("   " + "-" * 64)
                    print("   STAGE 1 KEYWORD MATCHES (first 10)")
                    print("   " + "-" * 64)
                    keywords = stage1.get('keywords', [])
                    print(f"   Keywords: {', '.join(keywords)}")
                    print()

                    candidates = stage1.get('candidates_shown', [])
                    for idx, cand in enumerate(candidates[:10], 1):
                        text = cand.get('text', '')
                        source = cand.get('source_name', 'Unknown')
                        print(f"   {idx:2d}. {source}")
                        print(f"       {text[:60]}{'...' if len(text) > 60 else ''}")

                    if stage1.get('total_candidates', 0) > 10:
                        remaining = stage1['total_candidates'] - 10
                        print(f"       ... and {remaining} more")
                    print()
                    print("   " + "-" * 64)
                    print()

        # Final response
        print("=" * 70)
        print("💬 RESPONSE")
        print("=" * 70)
        print()

        response = trace.final_response
        print(format_line(response, show_translation, translator))
        print()

        # Show sources if available
        orchestrator_step = next((s for s in steps if s.get('name') == 'Orchestrator' or s.get('step') == 'Orchestrator'), None)
        if orchestrator_step:
            outputs = orchestrator_step.get('outputs', {})
            full_response = outputs.get('full_response', {})
            sources = full_response.get('sources', [])

            if sources:
                print()
                print("=" * 70)
                print("📚 SOURCES")
                print("=" * 70)
                print()

                for i, source in enumerate(sources[:3], 1):
                    score = source.get('score', 0.0)
                    text = source.get('text', '')
                    source_name = source.get('source_name', 'Unknown')
                    line_num = source.get('line', '?')

                    print(f"{i}. [{score:.3f}] {source_name}:{line_num}")
                    print(format_line(text[:200] + ('...' if len(text) > 200 else ''),
                                    show_translation, translator, "   "))
                    print()

                if len(sources) > 3:
                    print(f"   ... and {len(sources) - 3} more")
                    print()

    except Exception as e:
        print()
        print("❌ ERROR")
        print("=" * 70)
        print(f"{type(e).__name__}: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    print("=" * 70)
    print("✓ Done")
    print("=" * 70)


if __name__ == "__main__":
    main()
