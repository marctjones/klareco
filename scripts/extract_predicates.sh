#!/bin/bash
# Extract predicate triples from unified corpus.
#
# Part of Issue #253 - Phase 1.1 of Enhanced AST-Aware Retriever
#
# This extracts (verb, subj, obj) triples for predicate-based retrieval.
# Expected runtime: ~30-60 minutes for 4.3M docs
#
# Usage:
#   ./scripts/extract_predicates.sh           # Full extraction (auto-resumes)
#   ./scripts/extract_predicates.sh --fresh   # Start fresh, ignore checkpoint
#   ./scripts/extract_predicates.sh --limit 10000  # Test run
#
# Output: data/indexes/kuzu_index/predicates.jsonl
# Log:    logs/extract_predicates_YYYYMMDD_HHMMSS.log

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
echo "Extract Predicate Triples (Issue #253)"
echo "=============================================="
echo ""
echo "Started: $(date)"
echo ""

# Show version and pass through any arguments (like --limit, --fresh, --resume)
python scripts/extract_predicates.py --version
python scripts/extract_predicates.py "$@"

echo ""
echo "=============================================="
echo "Complete!"
echo "=============================================="
