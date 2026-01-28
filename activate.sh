#!/bin/bash
# Convenience script to activate the Klareco Python development environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "Error: Virtual environment not found at $SCRIPT_DIR/.venv"
    echo "Run: python3 -m venv .venv && pip install -r requirements.txt"
    exit 1
fi

echo "Activating Klareco Python environment..."
source "$SCRIPT_DIR/.venv/bin/activate"

echo "✓ Environment activated"
echo "Python: $(python --version)"
echo "Location: $(which python)"
echo ""
echo "To deactivate, run: deactivate"
