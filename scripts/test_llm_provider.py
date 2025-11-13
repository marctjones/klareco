#!/usr/bin/env python3
"""
Simple LLM Provider Test - No parser dependencies

Tests the LLM provider auto-detection without requiring full Klareco setup.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from klareco.llm_provider import get_llm_provider, LLMProviderType


def main():
    print("="*80)
    print("LLM PROVIDER AUTO-DETECTION TEST")
    print("="*80)

    # Test auto-detection
    provider = get_llm_provider()

    print(f"\n✓ Provider Type: {provider.provider_type.value}")

    if provider.provider_type == LLMProviderType.CLAUDE_CODE:
        print("\n✓ DETECTED: Claude Code Environment")
        print("  → LLM requests will use Claude Code as the backend")
        print("  → When Klareco needs LLM generation (summarization, QA),")
        print("    it will output a request and wait for response")
        print("\n  To respond to LLM requests:")
        print("    python scripts/claude_llm_respond.py 'Your response here'")

    elif provider.provider_type == LLMProviderType.ANTHROPIC_API:
        print("\n✓ DETECTED: Anthropic API")
        print("  → Will use Anthropic API for LLM requests")
        print("  → API key found in environment")

    elif provider.provider_type == LLMProviderType.OPENAI_API:
        print("\n✓ DETECTED: OpenAI API")
        print("  → Will use OpenAI API for LLM requests")
        print("  → API key found in environment")

    else:
        print(f"\n✓ DETECTED: {provider.provider_type.value}")

    print("\n" + "="*80)
    print("EXAMPLE USAGE IN CODE:")
    print("="*80)
    print("""
from klareco.llm_provider import get_llm_provider

# Auto-detect and get provider
provider = get_llm_provider()

# Generate text
response = provider.generate(
    prompt="What is Esperanto?",
    system="You are a helpful assistant.",
    max_tokens=300,
    temperature=0.7
)

print(response)
""")

    print("="*80)
    print("INTEGRATION WITH KLARECO EXPERTS:")
    print("="*80)
    print("""
# Summarize Expert automatically uses detected provider
from klareco.experts.summarize_expert import create_summarize_expert
expert = create_summarize_expert()  # Auto-detects LLM provider

# Factoid QA Expert uses RAG + detected LLM provider
from klareco.experts.factoid_qa_expert import create_factoid_qa_expert
qa_expert = create_factoid_qa_expert()  # Auto-detects LLM provider
""")

    print("="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
