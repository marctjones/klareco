#!/usr/bin/env python3
"""
Test LLM answer generation step.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.llm_provider import get_llm_provider

def test_llm():
    """Test LLM provider."""

    print("=" * 70)
    print("LLM GENERATION TEST")
    print("=" * 70)
    print()

    # Step 1: Get LLM provider
    print("Step 1: Initializing LLM provider...")
    start = time.time()

    try:
        llm = get_llm_provider()
        elapsed = time.time() - start
        print(f"✓ LLM provider initialized in {elapsed:.3f}s")
        print(f"  Provider type: {llm.provider_type.value}")
        print()
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ FAILED after {elapsed:.3f}s")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 2: Generate answer
    print("Step 2: Generating answer...")
    start = time.time()

    question = "Kiu estas Gandalf?"
    context = """
    uloj, la kunulon de Mitrandiro

    Mitrandiro estas saĝa sorĉisto.
    """

    system_prompt = "Vi estas helpema asistanto. Respondu en Esperanto."
    user_prompt = f"""Bazita sur la sekvanta kunteksto, respondu la demandon:

Kunteksto:
{context}

Demando: {question}

Respondu koncize en Esperanto."""

    try:
        answer = llm.generate(
            prompt=user_prompt,
            system=system_prompt,
            max_tokens=100,
            temperature=0.1
        )
        elapsed = time.time() - start
        print(f"✓ Answer generated in {elapsed:.3f}s")
        print()
        print("Answer:")
        print("-" * 70)
        print(answer)
        print("-" * 70)
        print()
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ FAILED after {elapsed:.3f}s")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return

    print("=" * 70)
    print("✓ LLM GENERATION TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_llm()
