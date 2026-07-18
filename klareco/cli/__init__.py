"""
Klareco CLI — a registry-based dispatcher.

The command surface is defined in docs/CLI_ARCHITECTURE.md. Each command GROUP is
a module under commands/ exposing `register(subparsers)`; handlers return an int
exit code. Adding a command = add a module + one line in _GROUPS. That is the
whole extension story — keep it that tidy.

Run: `python -m klareco <command>`  (or `klareco <command>` once installed).
"""
from __future__ import annotations

import argparse
import importlib
import sys

from klareco.cli._base import EXIT_USAGE

# The command groups, in help-display order. One module each; each must expose
# register(subparsers). See docs/CLI_ARCHITECTURE.md for the full target.
_GROUPS = (
    'run',        # query, explain            — the orchestration engine
    'lang',       # parse, translate          — language primitives
    'data',       # data <stage>              — setup / data lifecycle
    'inspect',    # doctor, info, inspect     — environment & store
    'corpus',     # corpus <sub>              — corpus registry
    'evaluate',   # eval                      — the merge-gate ledger
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='klareco',
        description=('Klareco — Esperanto AI. Deterministic-first: the '
                    'orchestrator threads an AST-thought through mandatory '
                    'stages; optional modules plug into a stable core.'),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Examples:\n"
                "  klareco explain \"Kiu fondis Esperanton?\"   # decode the thought\n"
                "  klareco query   \"Kio estas Esperanto?\"\n"
                "  klareco doctor                              # is this machine ready?\n"
                "  klareco data parse                          # preview a setup step\n"
                "  klareco inspect store\n\n"
                "Full command surface & release target: docs/CLI_ARCHITECTURE.md"))
    sub = parser.add_subparsers(dest='command', metavar='<command>')
    for name in _GROUPS:
        mod = importlib.import_module(f'klareco.cli.commands.{name}')
        mod.register(sub)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, 'func', None)
    if func is None:
        parser.print_help()
        return EXIT_USAGE
    return func(args) or 0


if __name__ == '__main__':
    sys.exit(main())
