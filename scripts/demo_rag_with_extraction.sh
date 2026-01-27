#!/bin/bash
#
# Demo: Full RAG Pipeline with Answer Extraction
#
# Shows end-to-end question answering:
#   1. Parse query
#   2. Retrieve documents from corpus
#   3. Extract answer using AST patterns
#   4. Validate against expected answer
#
# Usage:
#   ./scripts/demo_rag_with_extraction.sh                    # Run 5 test queries
#   ./scripts/demo_rag_with_extraction.sh -i                  # Interactive mode
#   ./scripts/demo_rag_with_extraction.sh "Kiu fondis Esperanton?"  # Single query

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
    echo "❌ No virtual environment found (.venv or venv)"
    exit 1
fi

# Check if index exists
INDEX="data/indexes/kuzu_index"
if [ ! -d "$INDEX" ] || [ ! -f "$INDEX/kuzu.db" ]; then
    echo "❌ Kuzu index not found: $INDEX/kuzu.db"
    echo "Build index: python scripts/index_kuzu.py"
    exit 1
fi

echo "=============================================================================="
echo "RAG Pipeline with Answer Extraction Demo"
echo "=============================================================================="
echo "Pipeline: Query → Retrieval → Answer Extraction → Validation"
echo ""
echo "Index: $INDEX"
echo "=============================================================================="
echo ""

# Parse arguments
if [ "$1" == "-i" ] || [ "$1" == "--interactive" ]; then
    # Interactive mode
    PYTHONPATH=. python scripts/demo_rag_with_extraction.py --interactive
elif [ -n "$1" ]; then
    # Single query (all args passed as query)
    PYTHONPATH=. python scripts/demo_rag_with_extraction.py --query "$*"
else
    # Run test queries
    PYTHONPATH=. python scripts/demo_rag_with_extraction.py
fi
