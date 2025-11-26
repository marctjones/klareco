"""
Environment Detection for Klareco

Detects execution environment and provides warnings for inappropriate operations.

Environments:
- Claude Code Web: Browser-based, limited compute, NO TRAINING
- Claude Code CLI: Local execution, full compute, can train models
- Standalone: Regular Python execution
"""

import os
import sys
import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ExecutionEnvironment(Enum):
    """Detected execution environment"""
    CLAUDE_CODE_WEB = "claude_code_web"  # Browser-based Claude Code
    CLAUDE_CODE_CLI = "claude_code_cli"  # CLI Claude Code
    STANDALONE = "standalone"            # Regular Python


def detect_environment() -> ExecutionEnvironment:
    """
    Detect the current execution environment.

    Returns:
        Detected environment type
    """
    # Check for Claude Code session
    if os.environ.get('CLAUDE_CODE_SESSION_ID'):
        if _is_web_environment():
            return ExecutionEnvironment.CLAUDE_CODE_WEB
        else:
            return ExecutionEnvironment.CLAUDE_CODE_CLI

    # Check for anthropic package (Claude Code indicator)
    try:
        import anthropic
        if not os.environ.get('ANTHROPIC_API_KEY'):
            if _is_web_environment():
                return ExecutionEnvironment.CLAUDE_CODE_WEB
            else:
                return ExecutionEnvironment.CLAUDE_CODE_CLI
    except ImportError:
        pass

    return ExecutionEnvironment.STANDALONE


def _is_web_environment() -> bool:
    """
    Detect if we're in a web-based environment vs local CLI.

    Returns:
        True if web environment, False otherwise
    """
    # CLI indicators
    cli_indicators = ['SSH_CONNECTION', 'DISPLAY', 'TMUX', 'STY']
    if any(os.environ.get(ind) for ind in cli_indicators):
        return False

    # Terminal type check
    term = os.environ.get('TERM', '')
    if term in ['xterm', 'xterm-256color', 'screen', 'tmux-256color']:
        return False

    # GPU check
    try:
        import torch
        if torch.cuda.is_available():
            return False
    except ImportError:
        pass

    return True


def is_web_environment() -> bool:
    """Check if currently in web environment"""
    return detect_environment() == ExecutionEnvironment.CLAUDE_CODE_WEB


def is_cli_environment() -> bool:
    """Check if currently in CLI environment"""
    return detect_environment() == ExecutionEnvironment.CLAUDE_CODE_CLI


def is_claude_code() -> bool:
    """Check if in any Claude Code environment (web or CLI)"""
    env = detect_environment()
    return env in [ExecutionEnvironment.CLAUDE_CODE_WEB,
                   ExecutionEnvironment.CLAUDE_CODE_CLI]


def warn_if_web_training():
    """
    Warn if attempting to train models in web environment.

    Should be called at the start of training scripts.
    """
    if is_web_environment():
        print("\n" + "="*80)
        print("⚠️  WARNING: TRAINING IN CLAUDE CODE WEB")
        print("="*80)
        print()
        print("You are running in Claude Code Web (browser-based).")
        print("Model training is NOT recommended in this environment:")
        print()
        print("  ❌ Limited compute resources")
        print("  ❌ May timeout or crash browser")
        print("  ❌ No GPU acceleration")
        print("  ❌ Session may disconnect")
        print()
        print("Recommendations:")
        print("  ✅ Use Claude Code CLI for training")
        print("  ✅ Use pre-trained models in web environment")
        print("  ✅ Test with small datasets only")
        print()
        print("="*80)

        response = input("Continue anyway? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Training aborted.")
            sys.exit(0)


def require_cli(operation_name: str = "This operation"):
    """
    Require CLI environment for operation.

    Args:
        operation_name: Name of the operation requiring CLI

    Raises:
        RuntimeError: If not in CLI environment
    """
    if is_web_environment():
        raise RuntimeError(
            f"{operation_name} requires Claude Code CLI or standalone environment.\n"
            f"You are currently in Claude Code Web (browser-based).\n"
            f"Please use Claude Code CLI for heavy compute operations."
        )


def get_environment_info() -> dict:
    """
    Get detailed environment information.

    Returns:
        Dictionary with environment details
    """
    env = detect_environment()

    return {
        'environment': env.value,
        'is_web': env == ExecutionEnvironment.CLAUDE_CODE_WEB,
        'is_cli': env == ExecutionEnvironment.CLAUDE_CODE_CLI,
        'is_claude_code': env in [ExecutionEnvironment.CLAUDE_CODE_WEB,
                                   ExecutionEnvironment.CLAUDE_CODE_CLI],
        'can_train': env != ExecutionEnvironment.CLAUDE_CODE_WEB,
        'should_use_claude_llm': env in [ExecutionEnvironment.CLAUDE_CODE_WEB,
                                         ExecutionEnvironment.CLAUDE_CODE_CLI],
        'python_version': sys.version,
        'platform': sys.platform,
    }


if __name__ == "__main__":
    # Test detection
    info = get_environment_info()

    print("Environment Detection")
    print("="*80)
    for key, value in info.items():
        print(f"{key:25s}: {value}")
    print("="*80)

    if info['is_web']:
        print("\n⚠️  Running in Claude Code Web - no training recommended")
    elif info['is_cli']:
        print("\n✅ Running in Claude Code CLI - can train models")
    else:
        print("\n✅ Running standalone - full capabilities")
