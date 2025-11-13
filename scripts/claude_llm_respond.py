#!/usr/bin/env python3
"""
Claude Code LLM Response Helper

This script allows Claude Code to respond to LLM requests from Klareco.
When Klareco makes an LLM request in Claude Code mode, use this script
to provide the response.

Usage:
    python scripts/claude_llm_respond.py "Your response text here"

    # Or provide response via stdin:
    echo "Your response" | python scripts/claude_llm_respond.py -

    # Or use an editor:
    python scripts/claude_llm_respond.py --edit
"""

import sys
import json
import tempfile
from pathlib import Path
import subprocess


def read_latest_request():
    """Read the latest LLM request"""
    request_file = Path(tempfile.gettempdir()) / ".klareco_llm_request.json"

    if not request_file.exists():
        print("No LLM request found.")
        return None

    with open(request_file, 'r') as f:
        request = json.load(f)

    return request


def write_response(response_text: str):
    """Write LLM response"""
    response_file = Path(tempfile.gettempdir()) / ".klareco_llm_response.json"

    response = {
        "response": response_text,
        "status": "success"
    }

    with open(response_file, 'w') as f:
        json.dump(response, f, indent=2)

    print(f"Response written to: {response_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/claude_llm_respond.py <response>")
        print("   or: python scripts/claude_llm_respond.py - (read from stdin)")
        print("   or: python scripts/claude_llm_respond.py --edit (use editor)")
        sys.exit(1)

    # Show the request
    request = read_latest_request()
    if request:
        print("\n" + "="*80)
        print("ORIGINAL REQUEST:")
        print("="*80)
        if request.get('system'):
            print(f"System: {request['system']}")
        print(f"Prompt: {request['prompt']}")
        print("="*80 + "\n")

    # Get response
    if sys.argv[1] == '--edit':
        # Use editor
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("# Enter your LLM response below:\n\n")
            temp_path = f.name

        editor = os.environ.get('EDITOR', 'nano')
        subprocess.call([editor, temp_path])

        with open(temp_path, 'r') as f:
            lines = [line for line in f.readlines() if not line.startswith('#')]
            response_text = ''.join(lines).strip()

        os.unlink(temp_path)

    elif sys.argv[1] == '-':
        # Read from stdin
        response_text = sys.stdin.read().strip()

    else:
        # Read from arguments
        response_text = ' '.join(sys.argv[1:])

    if not response_text:
        print("Error: Empty response")
        sys.exit(1)

    # Write response
    write_response(response_text)
    print("\n✅ Response sent!")


if __name__ == '__main__':
    main()
