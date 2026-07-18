"""
Klareco CLI entry point.

Allows running Klareco as a module:
    python -m klareco <command>
"""
import sys

from klareco.cli import main

if __name__ == '__main__':
    sys.exit(main())
