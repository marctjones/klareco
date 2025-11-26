#!/usr/bin/env python3
"""
⚠️ EXTERNAL LLM - STOPGAP MEASURE ONLY

This uses EXTERNAL Claude Code LLM (interactive) to generate answers.
This is a TEMPORARY FALLBACK. Use query_with_local_model.py instead.

PREFER: scripts/query_with_local_model.py (local QA Decoder, fully automatic)

Usage:
    python scripts/query_with_llm.py "Kiu estas Frodo?"
    python scripts/query_with_llm.py "Kiu estas Frodo?" --translate

WARNING: Requires Claude Code and interactive response.
"""

import sys
import argparse
import logging
from pathlib import Path

# Suppress logging early
if '--debug' not in sys.argv:
    logging.basicConfig(level=logging.CRITICAL)
    logging.disable(logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.rag.retriever import create_retriever
from klareco.claude_code_llm import create_claude_code_provider
from klareco.translator import TranslationService


def generate_answer_with_llm(query: str, retriever, llm_provider, k: int = 5):
    """
    Generate answer using RAG + LLM.

    Args:
        query: Query text (Esperanto)
        retriever: RAG retriever
        llm_provider: LLM provider (Claude Code)
        k: Number of context documents to retrieve

    Returns:
        Dict with answer and sources
    """
    # Parse query
    try:
        ast = parse(query)
    except Exception as e:
        return {
            'error': f'Failed to parse query: {e}',
            'answer': None,
            'sources': []
        }

    # Retrieve context
    try:
        results = retriever.retrieve_hybrid(
            ast,
            k=k,
            return_scores=True
        )
    except Exception as e:
        return {
            'error': f'Failed to retrieve: {e}',
            'answer': None,
            'sources': []
        }

    if not results:
        return {
            'error': 'No results found',
            'answer': 'Mi ne trovis rilatan informon en la korpuso.',
            'sources': []
        }

    # Build context from top results
    context_texts = []
    for i, result in enumerate(results[:k], 1):
        text = result.get('text', '')
        score = result.get('score', 0.0)
        source = result.get('source_name', 'Unknown')
        context_texts.append(f"[{i}] {text} (Source: {source}, Score: {score:.3f})")

    context = "\n".join(context_texts)

    # Generate answer with LLM
    prompt = f"""Based on the following context from an Esperanto corpus about Lord of the Rings, answer the question in Esperanto.

QUESTION (Esperanto):
{query}

CONTEXT (Retrieved sentences):
{context}

INSTRUCTIONS:
1. Answer in Esperanto (the same language as the question)
2. Be concise (2-3 sentences maximum)
3. Base your answer ONLY on the provided context
4. If the context doesn't contain the answer, say "Mi ne trovis sufiĉan informon."
5. Do not include translations or explanations

ANSWER (Esperanto):"""

    system_prompt = "You are a helpful assistant that answers questions about Lord of the Rings in Esperanto based on provided context. Always respond in Esperanto."

    try:
        answer = llm_provider.generate(
            prompt=prompt,
            system=system_prompt,
            max_tokens=200,
            temperature=0.7
        )
    except Exception as e:
        return {
            'error': f'LLM generation failed: {e}',
            'answer': None,
            'sources': results
        }

    return {
        'answer': answer,
        'sources': results,
        'context_used': len(results)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Query with LLM answer generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s "Kiu estas Frodo?"              # Generate answer in Esperanto
  %(prog)s "Kiu estas Frodo?" --translate  # Translate answer to English
  %(prog)s "Kiu estas Frodo?" -k 10        # Use 10 context documents
        '''
    )
    parser.add_argument(
        'query',
        nargs='?',
        default='Kiu estas Frodo?',
        help='Query in Esperanto'
    )
    parser.add_argument(
        '-k',
        type=int,
        default=30,
        help='Number of context documents to retrieve (default: 30)'
    )
    parser.add_argument(
        '--translate',
        action='store_true',
        help='Translate answer to English'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Show debug logging'
    )

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    translator = TranslationService() if args.translate else None

    print("=" * 70)
    print("⚠️ KLARECO - EXTERNAL LLM (STOPGAP)")
    print("=" * 70)
    print()
    print("⚠️ WARNING: This uses EXTERNAL Claude Code LLM (interactive)")
    print("⚠️ PREFER: ./ask.sh (local QA Decoder, fully automatic)")
    print()
    print("=" * 70)
    print()
    print(f"📝 Query: {args.query}")
    if args.translate:
        print("   [English translation enabled for OUTPUT]")
    else:
        print("   [Pure Esperanto mode - use --translate to show English]")
    print()
    print(f"⚙️  Loading RAG system (k={args.k})...")

    # Initialize retriever
    try:
        retriever = create_retriever()
        print(f"✓ Loaded {retriever.index.ntotal:,} sentences")
    except Exception as e:
        print(f"✗ Failed to load retriever: {e}")
        return 1

    # Initialize LLM
    print("⚙️  Initializing Claude Code LLM...")
    llm_provider = create_claude_code_provider()
    print("✓ LLM ready")
    print()

    # Generate answer
    print("🔍 Retrieving context...")
    print()

    result = generate_answer_with_llm(args.query, retriever, llm_provider, k=args.k)

    if result.get('error'):
        print(f"✗ ERROR: {result['error']}")
        return 1

    # Show context
    print("=" * 70)
    print(f"📚 RETRIEVED CONTEXT ({result.get('context_used', 0)} documents)")
    print("=" * 70)
    print()

    for i, source in enumerate(result.get('sources', [])[:3], 1):
        score = source.get('score', 0.0)
        text = source.get('text', '')
        source_name = source.get('source_name', 'Unknown')

        print(f"{i}. [{score:.3f}] {source_name}")

        # Apply same translation logic as answer
        if args.translate and translator:
            try:
                en_text = translator.translate(text[:150], 'eo', 'en')
                print(f"   {en_text}...")
            except:
                print(f"   {text[:150]}...")
        else:
            print(f"   {text[:150]}...")
        print()

    if len(result.get('sources', [])) > 3:
        print(f"   ... and {len(result['sources']) - 3} more")
        print()

    # Show answer
    print("=" * 70)
    print("💬 GENERATED ANSWER")
    print("=" * 70)
    print()

    answer = result.get('answer', '')

    if args.translate and translator and answer:
        # Translate mode: Show ONLY English (consistent with query.sh)
        try:
            en_answer = translator.translate(answer, 'eo', 'en')
            print(en_answer)
        except Exception as e:
            # If translation fails, fall back to Esperanto
            print(f"[Translation failed: {e}]")
            print(answer)
    else:
        # Default: Show ONLY Esperanto
        print(answer)

    print()
    print("=" * 70)
    print("✓ Done")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
