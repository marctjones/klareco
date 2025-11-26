"""
⚠️ EXTERNAL LLM - STOPGAP MEASURE ONLY

Claude Code LLM Adapter - Use Claude Code as TEMPORARY fallback LLM backend

WARNING: This is a STOPGAP measure. Klareco prefers LOCAL models (QA Decoder, etc.)
This module should only be used when local features are not yet implemented.

Usage:
    from klareco.claude_code_llm import create_claude_code_provider

    provider = create_claude_code_provider()
    response = provider.generate("Summarize this text...")

The provider will print requests clearly in the output, and you can respond
directly in the conversation with Claude Code.

PREFER: Local models whenever possible (see scripts/query_with_local_model.py)
"""

import logging
from typing import Optional
from .llm_provider import LLMProvider, LLMProviderType

logger = logging.getLogger(__name__)


def claude_code_conversation_callback(
    prompt: str,
    system: Optional[str],
    max_tokens: int,
    temperature: float
) -> str:
    """
    Callback that outputs LLM requests for Claude Code to see and respond to.

    When running code in Claude Code, the AI will see this output and can
    respond directly with the generated text. This avoids needing API keys.

    Args:
        prompt: User prompt/query
        system: System prompt (optional)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Placeholder string indicating awaiting response
    """
    print("\n" + "="*80)
    print("⚠️ EXTERNAL LLM REQUEST (STOPGAP - Local model preferred)")
    print("="*80)
    print()
    print("⚠️ WARNING: Using EXTERNAL Claude Code LLM (interactive)")
    print("⚠️ PREFER: Local QA Decoder (see ./ask.sh)")
    print()
    print("="*80)
    print("🤖 LLM REQUEST FOR CLAUDE CODE")
    print("="*80)

    if system:
        print(f"\n📋 SYSTEM:")
        print(f"   {system}")

    print(f"\n💬 PROMPT:")
    print(f"   {prompt}")

    print(f"\n⚙️  PARAMS:")
    print(f"   Max tokens: {max_tokens}")
    print(f"   Temperature: {temperature}")

    print("\n" + "="*80)
    print("Claude, please respond with the generated text above.")
    print("The response will be used by the system.")
    print("="*80 + "\n")

    # In Claude Code, you would respond in conversation and the code would
    # use that response. For now, return a placeholder.
    return "[AWAITING CLAUDE CODE RESPONSE - respond in conversation]"


def create_claude_code_provider(
    callback = None,
    wait_timeout: int = 30  # Shorter timeout for interactive use
) -> LLMProvider:
    """
    Create an LLM provider that uses Claude Code as the backend.

    This allows Klareco to use Claude Code (the AI running your code) as the
    LLM without requiring any API keys. Perfect for development and testing.

    Args:
        callback: Optional custom callback function
            If None, uses default conversation callback
        wait_timeout: Timeout in seconds (default: 30s for interactive use)

    Returns:
        LLM provider configured to use Claude Code

    Example:
        >>> provider = create_claude_code_provider()
        >>> response = provider.generate("What is Esperanto?")
        # Claude Code will see the request and can respond in conversation
    """
    if callback is None:
        callback = claude_code_conversation_callback

    provider = LLMProvider(
        force_provider=LLMProviderType.CLAUDE_CODE,
        claude_code_callback=callback,
        wait_timeout=wait_timeout
    )

    logger.info("Created Claude Code LLM provider (no API keys required)")
    return provider


def create_mock_provider(mock_responses: dict) -> LLMProvider:
    """
    Create a mock LLM provider for testing (returns predefined responses).

    Args:
        mock_responses: Dict mapping prompt substrings to responses
            Example: {"Kio estas": "Esperanto estas internacia lingvo..."}

    Returns:
        LLM provider that returns mock responses

    Example:
        >>> responses = {
        ...     "Frodo": "Frodo estas la ĉefa karaktero...",
        ...     "Gandalfo": "Gandalfo estas saĝulo..."
        ... }
        >>> provider = create_mock_provider(responses)
    """
    def mock_callback(prompt, system, max_tokens, temperature):
        """Return mock response based on prompt"""
        for key, response in mock_responses.items():
            if key.lower() in prompt.lower():
                logger.debug(f"Mock LLM: matched '{key}' → returning response")
                return response

        # Default response if no match
        logger.debug("Mock LLM: no match, returning default")
        return f"[Mock response for: {prompt[:50]}...]"

    provider = LLMProvider(
        force_provider=LLMProviderType.CLAUDE_CODE,
        claude_code_callback=mock_callback,
        wait_timeout=1  # Instant for mocks
    )

    logger.info("Created mock LLM provider for testing")
    return provider


# Convenience function for getting Claude Code provider
def get_claude_provider():
    """Get Claude Code LLM provider (shorthand for create_claude_code_provider)"""
    return create_claude_code_provider()


if __name__ == "__main__":
    # Demo/test
    print("Testing Claude Code LLM Adapter\n")

    # Test 1: Conversation callback
    print("Test 1: Conversation callback")
    provider = create_claude_code_provider()
    print(f"Provider type: {provider.provider_type.value}\n")

    # This will print a request for Claude Code to see
    response = provider.generate(
        prompt="Kio estas Esperanto?",
        system="You are a helpful assistant that answers questions about Esperanto.",
        max_tokens=200,
        temperature=0.7
    )
    print(f"Response: {response}\n")

    # Test 2: Mock responses
    print("\nTest 2: Mock provider")
    mock_responses = {
        "Esperanto": "Esperanto estas internacia planlingvo kreita de D-ro Zamenhof en 1887.",
        "Frodo": "Frodo Sakvil-Benso estas la ĉefa protagonisto de 'La Mastro de l' Ringoj'."
    }

    mock_provider = create_mock_provider(mock_responses)

    for question in ["Kio estas Esperanto?", "Kiu estas Frodo?"]:
        response = mock_provider.generate(question)
        print(f"Q: {question}")
        print(f"A: {response}\n")
