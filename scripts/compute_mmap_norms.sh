#!/bin/bash
# Compute pre-computed norms for mmap slot embeddings
# This provides a 20% speedup by avoiding repeated norm computations during retrieval

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
    echo "Error: No virtual environment found"
    exit 1
fi

# Default values
INDEX_PATH="data/indexes/slot_full"
FORCE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --index)
            INDEX_PATH="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --index PATH    Path to slot index (default: data/indexes/slot_full)"
            echo "  --force         Recompute norms even if they already exist"
            echo "  -h, --help      Show this help"
            echo ""
            echo "Example:"
            echo "  $0                                    # Compute norms for slot_full"
            echo "  $0 --index data/indexes/slot_test    # Compute for test index"
            echo "  $0 --force                           # Force recomputation"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

FORCE_FLAG=""
if [ "$FORCE" = true ]; then
    FORCE_FLAG="--force"
fi

python scripts/compute_mmap_norms.py --index "$INDEX_PATH" $FORCE_FLAG
