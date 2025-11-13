"""
LLM Provider Abstraction with Auto-Detection

This module automatically detects the execution environment and routes LLM
requests appropriately:
- Claude Code Web/CLI: Uses Claude itself via interactive protocol
- Standalone: Falls back to Anthropic/OpenAI APIs

Part of Klareco's hybrid architecture where LLMs are used only for
genuinely creative tasks (summarization, QA) while structure is symbolic.
"""

import os
import sys
import json
import tempfile
from typing import Optional, Dict, Any, List
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class LLMProviderType(Enum):
    """Supported LLM provider types"""
    CLAUDE_CODE = "claude_code"  # Both web and CLI
    ANTHROPIC_API = "anthropic_api"
    OPENAI_API = "openai_api"
    INTERACTIVE = "interactive"  # Manual input for testing


class LLMProvider:
    """
    Auto-detecting LLM provider that uses Claude Code when available.

    Detection Logic:
    1. Check for CLAUDE_CODE environment variable (set by Claude Code)
    2. Check for Claude Code-specific paths/markers
    3. Fall back to API keys if available
    4. Default to interactive mode for testing

    Usage:
        provider = LLMProvider()  # Auto-detects environment
        response = provider.generate("Summarize this text: ...")
    """

    def __init__(
        self,
        force_provider: Optional[LLMProviderType] = None,
        claude_code_callback = None,
        wait_timeout: int = 300  # 5 minutes default
    ):
        """
        Initialize provider with auto-detection.

        Args:
            force_provider: Override auto-detection for testing
            claude_code_callback: Custom callback for Claude Code LLM requests
            wait_timeout: Timeout in seconds for Claude Code responses
        """
        if force_provider:
            self.provider_type = force_provider
        else:
            self.provider_type = self._detect_environment()

        logger.info(f"LLM Provider initialized: {self.provider_type.value}")

        # Initialize provider-specific clients
        self._anthropic_client = None
        self._openai_client = None
        self._claude_code_callback = claude_code_callback
        self._wait_timeout = wait_timeout

    def _detect_environment(self) -> LLMProviderType:
        """
        Detect which LLM provider to use based on environment.

        Returns:
            Detected provider type
        """
        # Check for Claude Code environment variables
        # Claude Code sets these when running code
        if os.environ.get('CLAUDE_CODE_SESSION_ID'):
            logger.debug("Detected Claude Code via SESSION_ID")
            return LLMProviderType.CLAUDE_CODE

        # Check for anthropic package in user site-packages (indicates Claude Code)
        # This is a heuristic - Claude Code often has anthropic installed
        try:
            import anthropic
            # If we can import but no API key, likely in Claude Code
            if not os.environ.get('ANTHROPIC_API_KEY'):
                logger.debug("Detected Claude Code via anthropic package without API key")
                return LLMProviderType.CLAUDE_CODE
        except ImportError:
            pass

        # Check for API keys
        if os.environ.get('ANTHROPIC_API_KEY'):
            logger.debug("Detected Anthropic API via API key")
            return LLMProviderType.ANTHROPIC_API

        if os.environ.get('OPENAI_API_KEY'):
            logger.debug("Detected OpenAI API via API key")
            return LLMProviderType.OPENAI_API

        # Default to Claude Code if we're uncertain
        # (better to assume we're in Claude Code than fail)
        logger.debug("No specific environment detected, defaulting to Claude Code")
        return LLMProviderType.CLAUDE_CODE

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text using the detected provider.

        Args:
            prompt: User prompt/query
            system: System prompt (optional)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Provider-specific options

        Returns:
            Generated text
        """
        logger.debug(f"Generating with {self.provider_type.value}: {prompt[:100]}...")

        if self.provider_type == LLMProviderType.CLAUDE_CODE:
            return self._generate_claude_code(prompt, system, max_tokens, temperature)
        elif self.provider_type == LLMProviderType.ANTHROPIC_API:
            return self._generate_anthropic(prompt, system, max_tokens, temperature)
        elif self.provider_type == LLMProviderType.OPENAI_API:
            return self._generate_openai(prompt, system, max_tokens, temperature)
        else:  # INTERACTIVE
            return self._generate_interactive(prompt, system)

    def _generate_claude_code(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> str:
        """
        Generate using Claude Code as the LLM backend.

        Strategies (in priority order):
        1. Use callback function if provided
        2. Wait for file-based response (polling with timeout)
        3. Return placeholder for autonomous operation

        File-based protocol:
        1. Write prompt to .klareco_llm_request.json
        2. Claude Code detects it and writes response to .klareco_llm_response.json
        3. Poll for response with timeout
        """
        import time

        logger.info("Using Claude Code as LLM provider")

        # Strategy 1: Use callback if provided
        if self._claude_code_callback:
            logger.debug("Using callback function for Claude Code LLM")
            try:
                return self._claude_code_callback(prompt, system, max_tokens, temperature)
            except Exception as e:
                logger.error(f"Callback failed: {e}")
                # Fall through to file-based protocol

        # Create request in a standard format
        request = {
            "type": "llm_request",
            "prompt": prompt,
            "system": system,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timestamp": time.time()
        }

        # Write to a temporary file that Claude Code can read
        request_file = Path(tempfile.gettempdir()) / ".klareco_llm_request.json"
        response_file = Path(tempfile.gettempdir()) / ".klareco_llm_response.json"

        # Clean up old response
        if response_file.exists():
            response_file.unlink()

        # Write request
        with open(request_file, 'w') as f:
            json.dump(request, f, indent=2)

        # Output marker for Claude Code to detect
        print("\n" + "="*80)
        print("🤖 KLARECO LLM REQUEST")
        print("="*80)
        if system:
            print(f"System: {system}")
        print(f"Prompt: {prompt}")
        print("="*80)
        print(f"📁 Request written to: {request_file}")
        print(f"📁 Respond by writing to: {response_file}")
        print(f"⏰ Waiting up to {self._wait_timeout} seconds for response...")
        print("\nTo respond manually, use:")
        print(f"  python scripts/claude_llm_respond.py 'Your response here'")
        print("="*80 + "\n")

        # Strategy 2: Wait for file-based response (polling)
        start_time = time.time()
        poll_interval = 0.5  # Check every 0.5 seconds
        last_log_time = start_time

        while (time.time() - start_time) < self._wait_timeout:
            if response_file.exists():
                try:
                    with open(response_file, 'r') as f:
                        response_data = json.load(f)

                    if response_data.get('status') == 'success':
                        logger.info("Received Claude Code LLM response")
                        return response_data.get('response', '')
                    else:
                        logger.warning(f"Invalid response status: {response_data.get('status')}")

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in response file: {e}")
                except Exception as e:
                    logger.error(f"Error reading response: {e}")

            # Log every 10 seconds to show we're still waiting
            current_time = time.time()
            if current_time - last_log_time >= 10:
                elapsed = int(current_time - start_time)
                remaining = self._wait_timeout - elapsed
                logger.info(f"Still waiting for response... ({elapsed}s elapsed, {remaining}s remaining)")
                last_log_time = current_time

            time.sleep(poll_interval)

        # Timeout reached
        logger.warning(f"Timeout waiting for Claude Code response after {self._wait_timeout}s")

        # Strategy 3: Return placeholder for autonomous operation
        # This allows the code to continue without blocking indefinitely
        return f"[CLAUDE_CODE_LLM_PENDING: Please provide response for: {prompt[:100]}...]"

    def _generate_anthropic(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate using Anthropic API"""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required for Anthropic API. "
                "Install with: pip install anthropic"
            )

        if not self._anthropic_client:
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            self._anthropic_client = anthropic.Anthropic(api_key=api_key)

        messages = [{"role": "user", "content": prompt}]

        response = self._anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=max_tokens,
            temperature=temperature,
            system=system if system else anthropic.NOT_GIVEN,
            messages=messages
        )

        return response.content[0].text

    def _generate_openai(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate using OpenAI API"""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required for OpenAI API. "
                "Install with: pip install openai"
            )

        if not self._openai_client:
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self._openai_client = openai.OpenAI(api_key=api_key)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        return response.choices[0].message.content

    def _generate_interactive(self, prompt: str, system: Optional[str]) -> str:
        """Generate using interactive stdin (for testing)"""
        print("\n" + "="*80)
        print("INTERACTIVE LLM REQUEST")
        print("="*80)
        if system:
            print(f"System: {system}")
        print(f"Prompt: {prompt}")
        print("="*80)
        print("Enter response (end with empty line):")

        lines = []
        while True:
            try:
                line = input()
                if not line:
                    break
                lines.append(line)
            except EOFError:
                break

        return "\n".join(lines)


# Global singleton instance
_global_provider: Optional[LLMProvider] = None


def get_llm_provider(
    force_provider: Optional[LLMProviderType] = None,
    claude_code_callback = None,
    wait_timeout: int = 300
) -> LLMProvider:
    """
    Get the global LLM provider instance (singleton pattern).

    Args:
        force_provider: Override auto-detection for testing
        claude_code_callback: Custom callback for Claude Code LLM requests
        wait_timeout: Timeout in seconds for Claude Code responses

    Returns:
        Global LLM provider
    """
    global _global_provider

    if _global_provider is None or force_provider is not None:
        _global_provider = LLMProvider(force_provider, claude_code_callback, wait_timeout)

    return _global_provider


def generate_text(prompt: str, **kwargs) -> str:
    """
    Convenience function for generating text with default provider.

    Args:
        prompt: User prompt
        **kwargs: Additional arguments for generate()

    Returns:
        Generated text
    """
    provider = get_llm_provider()
    return provider.generate(prompt, **kwargs)


if __name__ == "__main__":
    # Test detection
    provider = LLMProvider()
    print(f"Detected provider: {provider.provider_type.value}")

    # Test generation
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        response = provider.generate(prompt, system="You are a helpful assistant.")
        print(f"\nResponse: {response}")
