#!/bin/bash
# Import predicate triples into Kuzu database.
#
# Part of Issue #254 - Phase 1.2 of Enhanced AST-Aware Retriever
#
# This imports predicates.jsonl (from extract_predicates.py) into Kuzu.
# Expected runtime: ~10-20 minutes for full corpus
#
# Prerequisites:
#   1. Run extract_predicates.py first (creates predicates.jsonl)
#   2. Run build_kuzu_index.py first (creates kuzu.db)
#
# Usage:
#   ./scripts/import_predicates_kuzu.sh           # Full import
#   ./scripts/import_predicates_kuzu.sh --limit 10000  # Test run
#
# Output: Updates data/indexes/kuzu_index/kuzu.db
# Log:    logs/import_predicates_kuzu_YYYYMMDD_HHMMSS.log
#
# Note: This script is transactional - it clears existing predicates before
#       import. No checkpoint needed as Kuzu COPY is atomic.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "No virtual environment found"
    exit 1
fi

echo "=============================================="
echo "Import Predicates into Kuzu (Issue #254)"
echo "=============================================="
echo ""
echo "Started: $(date)"
echo ""

# Show version and pass through any arguments (like --limit)
python scripts/import_predicates_kuzu.py --version
python scripts/import_predicates_kuzu.py "$@"

echo ""
echo "=============================================="
echo "Complete!"
echo "=============================================="
