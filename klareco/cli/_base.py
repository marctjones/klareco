"""
Shared CLI conventions (see docs/CLI_ARCHITECTURE.md).

Design goals: adding a command = add one module under commands/ and one entry in
the _GROUPS list. Every handler returns an int EXIT CODE. Output goes through
emit() so every command can speak --json for third-party scripting.
"""
from __future__ import annotations

import json as _json
import sys
from typing import Any, Optional

# Stable exit codes — part of the CLI contract (scriptable by third parties).
EXIT_OK = 0
EXIT_ERROR = 1        # the command ran and failed
EXIT_USAGE = 2        # bad invocation (argparse also uses 2)
EXIT_PLANNED = 3      # a 🎯 target command that is not implemented yet
EXIT_DEGRADED = 4     # ran, but against a degraded environment

DOCS = "docs/CLI_ARCHITECTURE.md"


def add_common(parser) -> None:
    """Flags every command shares."""
    parser.add_argument('--json', action='store_true',
                        help='Emit machine-readable JSON on stdout')


def emit(args, *, text: Optional[str] = None, data: Any = None) -> None:
    """Print human text, or JSON when --json is set. Pass both: text for
    humans, data for machines."""
    if getattr(args, 'json', False):
        print(_json.dumps(data if data is not None else {'message': text},
                          indent=2, ensure_ascii=False))
    elif text is not None:
        print(text)


def err(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return EXIT_ERROR


def planned(args, what: str, tracking: str = "") -> int:
    """A tidy placeholder for a 🎯 target command. Never fakes work."""
    ref = f" (tracked: {tracking})" if tracking else ""
    msg = f"🎯 planned — `{what}` is a target interface, not yet implemented{ref}. See {DOCS}."
    emit(args, text=msg, data={'status': 'planned', 'command': what,
                               'tracking': tracking, 'docs': DOCS})
    return EXIT_PLANNED


def read_text_input(args) -> str:
    """Resolve text from a positional arg, --file, or stdin — the shared
    input convention for parse/translate."""
    if getattr(args, 'text', None):
        return args.text
    if getattr(args, 'file', None):
        with open(args.file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    print("Enter Esperanto text:", file=sys.stderr)
    return input().strip()
