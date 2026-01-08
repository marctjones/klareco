#!/bin/bash
# Build Kuzu Graph Index from Corpus
#
# This script builds a Kuzu graph database containing:
# - Phase 1: Inverted index (Root nodes, Sentence nodes, HAS_ROOT edges)
# - Phase 2: Semantic relations (IS_SYNONYM, IS_HYPERNYM, IS_ANTONYM edges)
# - Phase 3: Sentence adjacency (NEXT_SENTENCE edges)
#
# Usage:
#   ./scripts/build_kuzu_index.sh              # Build full index
#   ./scripts/build_kuzu_index.sh --fresh      # Start fresh, remove existing
#   ./scripts/build_kuzu_index.sh --phase 1    # Only run phase 1
#
# Output:
#   data/indexes/kuzu_index/kuzu.db            # Kuzu graph database
#   data/indexes/kuzu_index/documents.jsonl    # Document storage
#   data/indexes/kuzu_index/doc_offsets.npy    # O(1) document access

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Parse arguments
FRESH_FLAG=""
PHASE_FLAG=""
CORPUS_FLAG=""
RELATIONS_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --fresh)
            FRESH_FLAG="--fresh"
            shift
            ;;
        --phase)
            PHASE_FLAG="--phase $2"
            shift 2
            ;;
        --corpus)
            CORPUS_FLAG="--corpus $2"
            shift 2
            ;;
        --relations)
            RELATIONS_FLAG="--relations $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: No virtual environment found (.venv or venv)"
    exit 1
fi

echo "============================================================"
echo "Building Kuzu Graph Index"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo "Python: $(which python)"
echo ""

# Check for required files
CORPUS="${CORPUS_FLAG:-data/corpus/unified_corpus.jsonl}"
RELATIONS="data/raw/eo/dictionaries/revo/revo_semantic_relations.json"

if [ ! -f "data/corpus/unified_corpus.jsonl" ] && [ -z "$CORPUS_FLAG" ]; then
    echo "Warning: Default corpus not found: data/corpus/unified_corpus.jsonl"
    echo "Use --corpus to specify an alternative corpus file."
fi

if [ ! -f "$RELATIONS" ]; then
    echo "Warning: Semantic relations file not found: $RELATIONS"
    echo "Phase 2 (semantic relations) will be skipped."
fi

# Create output directory
mkdir -p data/indexes/kuzu_index
mkdir -p logs

# Build index
LOG_FILE="logs/build_kuzu_index_$(date +%Y%m%d_%H%M%S).log"
echo "Log file: $LOG_FILE"
echo ""

python scripts/build_kuzu_index.py \
    $FRESH_FLAG \
    $PHASE_FLAG \
    $CORPUS_FLAG \
    $RELATIONS_FLAG \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "Build Complete!"
echo "============================================================"
echo ""
echo "Output:"
echo "  Database: data/indexes/kuzu_index/kuzu.db"
echo "  Documents: data/indexes/kuzu_index/documents.jsonl"
echo "  Log: $LOG_FILE"
echo ""
echo "To use the index:"
echo "  from klareco.rag import ASTAwareRetriever, IndexBackend"
echo "  retriever = ASTAwareRetriever(backend=IndexBackend.KUZU)"
