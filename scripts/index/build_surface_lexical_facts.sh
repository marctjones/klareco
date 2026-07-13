#!/bin/bash
# Derive protected_roots + capitalization_ratio from RAW SURFACE TEXT.
#
# This must run BEFORE the rebuild (#807) — the rebuild must not bake in a
# lexicon laundered from the parser's own output (#806).
#
# Usage:
#   ./scripts/index/build_surface_lexical_facts.sh            # full corpus
#   ./scripts/index/build_surface_lexical_facts.sh --limit 300000   # sample
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "No venv found"; exit 1
fi

mkdir -p logs
LOG_FILE="logs/build_surface_lexical_facts_$(date +%Y%m%d_%H%M%S).log"

echo "Scanning raw text (no parser-derived column is read). Logging to $LOG_FILE"
python scripts/index/build_surface_lexical_facts.py "$@" 2>&1 | tee "$LOG_FILE"
