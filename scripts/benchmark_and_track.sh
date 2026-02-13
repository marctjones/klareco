#!/bin/bash
# Benchmark RAG system and optionally save/commit snapshot
#
# Usage:
#   ./scripts/benchmark_and_track.sh                    # Just evaluate and show
#   ./scripts/benchmark_and_track.sh after_my_fix       # Evaluate, save, and commit
#   ./scripts/benchmark_and_track.sh --quick            # Show current only (no eval)
#   ./scripts/benchmark_and_track.sh --compare          # Just compare snapshots

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
SNAPSHOT_NAME=""
QUICK_MODE=false
COMPARE_ONLY=false

if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
elif [[ "$1" == "--compare" ]]; then
    COMPARE_ONLY=true
elif [[ -n "$1" ]]; then
    SNAPSHOT_NAME="$1"
fi

# If compare only, just show comparison and exit
if $COMPARE_ONLY; then
    echo -e "${BLUE}=== Snapshot Comparison ===${NC}"
    python scripts/track_evaluation_progress.py --compare
    exit 0
fi

# If quick mode, just show current metrics
if $QUICK_MODE; then
    echo -e "${BLUE}=== Current Metrics (from last evaluation) ===${NC}"
    python scripts/track_evaluation_progress.py
    exit 0
fi

# Full workflow: evaluate, show, optionally save
echo -e "${BLUE}=== Step 1: Running Evaluation ===${NC}"
echo "This will take ~30-60 seconds..."
echo ""
python scripts/evaluate_rag_test_set.py

echo ""
echo -e "${BLUE}=== Step 2: Current Metrics ===${NC}"
python scripts/track_evaluation_progress.py

echo ""
echo -e "${BLUE}=== Step 3: Comparison with Previous ===${NC}"
python scripts/track_evaluation_progress.py --compare

# If snapshot name provided, save and commit
if [[ -n "$SNAPSHOT_NAME" ]]; then
    echo ""
    echo -e "${YELLOW}=== Step 4: Saving Snapshot ===${NC}"
    python scripts/track_evaluation_progress.py --save --name "$SNAPSHOT_NAME"

    # Get the snapshot file (most recent)
    SNAPSHOT_FILE=$(ls -t data/evaluation/snapshots/*.json | head -1)

    echo ""
    echo -e "${YELLOW}=== Step 5: Git Commit ===${NC}"

    # Extract metrics from snapshot for commit message
    GRANULAR=$(python3 -c "import json; print(json.load(open('$SNAPSHOT_FILE'))['granular']['overall_score'])")
    BINARY=$(python3 -c "import json; print(f\"{json.load(open('$SNAPSHOT_FILE'))['binary']['accuracy']*100:.1f}\")")

    # Get previous snapshot for delta calculation
    PREV_SNAPSHOT=$(ls -t data/evaluation/snapshots/*.json | head -2 | tail -1)
    if [[ -f "$PREV_SNAPSHOT" ]]; then
        PREV_GRANULAR=$(python3 -c "import json; print(json.load(open('$PREV_SNAPSHOT'))['granular']['overall_score'])")
        DELTA=$(python3 -c "print(f'{$GRANULAR - $PREV_GRANULAR:+.3f}')")
    else
        DELTA="+0.000"
    fi

    # Create commit message
    COMMIT_MSG="Benchmark: $SNAPSHOT_NAME - granular ${GRANULAR} (${DELTA})

After: $SNAPSHOT_NAME

Metrics:
- Granular Score: ${GRANULAR} / 1.000 (${DELTA} from previous)
- Binary Accuracy: ${BINARY}%

Snapshot: $(basename $SNAPSHOT_FILE)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

    # Stage and commit
    git add -f "$SNAPSHOT_FILE"
    git commit -m "$COMMIT_MSG"

    echo -e "${GREEN}✓ Snapshot committed!${NC}"
    echo ""
    echo "Snapshot: $(basename $SNAPSHOT_FILE)"
    echo "Granular: $GRANULAR ($DELTA)"
    echo "Binary:   ${BINARY}%"

else
    # No snapshot name - just show instructions
    echo ""
    echo -e "${YELLOW}=== Next Steps ===${NC}"
    echo "If you want to save this as a snapshot:"
    echo "  ./scripts/benchmark_and_track.sh <snapshot_name>"
    echo ""
    echo "Examples:"
    echo "  ./scripts/benchmark_and_track.sh after_extraction_fix"
    echo "  ./scripts/benchmark_and_track.sh improved_definitions"
    echo "  ./scripts/benchmark_and_track.sh new_reranker_v2"
fi
