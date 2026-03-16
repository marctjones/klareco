#!/bin/bash
#
# Phase 0 Orchestrator: Validate Deterministic Baseline
#
# This script runs the complete Phase 0 workflow:
# 1. Extend Kuzu schema with semantic properties
# 2. Load semantic annotations (10 roots initially)
# 3. Test deterministic baseline on 10 queries
# 4. Evaluate results
#
# Usage:
#   ./scripts/run_phase_0.sh                    # Run full workflow
#   ./scripts/run_phase_0.sh --dry-run          # Show what would happen
#   ./scripts/run_phase_0.sh --skip-schema      # Skip schema extension
#
# Requirements:
#   - Kuzu database at data/indexes/v2.1_kuzu_index_full
#   - Python 3.10+ with kuzu installed
#
# See: docs/GETTING_STARTED_IMPLEMENTATION.md

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATABASE="data/indexes/v2.1_kuzu_index_full"
ANNOTATIONS="data/annotations/phase_0_template.jsonl"
QUERIES="data/test_queries/phase_0.jsonl"
LOG_DIR="logs/phase_0"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse arguments
DRY_RUN=false
SKIP_SCHEMA=false
SKIP_ANNOTATIONS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-schema)
            SKIP_SCHEMA=true
            shift
            ;;
        --skip-annotations)
            SKIP_ANNOTATIONS=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Setup
cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_${TIMESTAMP}.log"

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Phase 0: Deterministic Baseline Test${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Database: $DATABASE"
echo "Annotations: $ANNOTATIONS"
echo "Test queries: $QUERIES"
echo "Log file: $LOG_FILE"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}🔍 DRY RUN MODE - No changes will be made${NC}"
    echo ""
fi

# Check prerequisites
echo -e "${BLUE}📋 Checking prerequisites...${NC}"

if [ ! -e "$DATABASE" ]; then
    echo -e "${RED}❌ Database not found: $DATABASE${NC}"
    echo "   Run corpus loading scripts first."
    exit 1
fi
echo -e "${GREEN}✅ Database found ($(du -h $DATABASE | cut -f1))${NC}"

if [ ! -f "$ANNOTATIONS" ]; then
    echo -e "${RED}❌ Annotations file not found: $ANNOTATIONS${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Annotations file found ($(wc -l < $ANNOTATIONS) roots)${NC}"

if [ ! -f "$QUERIES" ]; then
    echo -e "${RED}❌ Test queries not found: $QUERIES${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Test queries found ($(wc -l < $QUERIES) queries)${NC}"

# Check Python dependencies
if ! python3 -c "import kuzu" 2>/dev/null; then
    echo -e "${RED}❌ kuzu not installed${NC}"
    echo "   Run: pip install kuzu"
    exit 1
fi
echo -e "${GREEN}✅ Python dependencies OK${NC}"
echo ""

# Step 1: Extend Schema
if [ "$SKIP_SCHEMA" = true ]; then
    echo -e "${YELLOW}⏭️  Skipping schema extension (--skip-schema)${NC}"
    echo "" | tee -a "$LOG_FILE"
else
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Step 1/4: Extend Kuzu Schema${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    SCHEMA_CMD="python scripts/extend_kuzu_schema.py --database $DATABASE"
    if [ "$DRY_RUN" = true ]; then
        SCHEMA_CMD="$SCHEMA_CMD --dry-run"
    fi

    echo "Running: $SCHEMA_CMD" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    if $SCHEMA_CMD 2>&1 | tee -a "$LOG_FILE"; then
        echo -e "${GREEN}✅ Schema extension complete${NC}" | tee -a "$LOG_FILE"
    else
        echo -e "${RED}❌ Schema extension failed${NC}" | tee -a "$LOG_FILE"
        echo "See log: $LOG_FILE"
        exit 1
    fi
    echo "" | tee -a "$LOG_FILE"
fi

# Step 2: Load Annotations
if [ "$SKIP_ANNOTATIONS" = true ]; then
    echo -e "${YELLOW}⏭️  Skipping annotation loading (--skip-annotations)${NC}"
    echo "" | tee -a "$LOG_FILE"
else
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Step 2/4: Load Semantic Annotations${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    LOAD_CMD="python scripts/load_semantic_annotations.py --annotations $ANNOTATIONS --database $DATABASE"
    if [ "$DRY_RUN" = true ]; then
        LOAD_CMD="$LOAD_CMD --dry-run"
    fi

    echo "Running: $LOAD_CMD" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    if $LOAD_CMD 2>&1 | tee -a "$LOG_FILE"; then
        echo -e "${GREEN}✅ Annotation loading complete${NC}" | tee -a "$LOG_FILE"
    else
        echo -e "${RED}❌ Annotation loading failed${NC}" | tee -a "$LOG_FILE"
        echo "See log: $LOG_FILE"
        exit 1
    fi
    echo "" | tee -a "$LOG_FILE"
fi

# Step 3: Test Deterministic Baseline
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 3/4: Test Deterministic Baseline${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${YELLOW}⚠️  Baseline testing not yet implemented${NC}"
echo ""
echo "TODO: Create scripts/test_deterministic_baseline.py with:"
echo "  1. Retrieve relevant sentences from Kuzu"
echo "  2. Parse to ASTs"
echo "  3. Extract facts"
echo "  4. Classify into schema slots"
echo "  5. Score importance"
echo "  6. Select and synthesize"
echo ""
echo "For now, schema is ready for manual testing." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Step 4: Evaluate Results
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Step 4/4: Evaluate Results${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${YELLOW}⚠️  Evaluation not yet implemented${NC}"
echo ""
echo "TODO: Create scripts/evaluate_phase_0.py with:"
echo "  1. Load generated summaries"
echo "  2. Human evaluation interface (1-5 scale)"
echo "  3. Calculate quality metrics"
echo "  4. Generate report"
echo "" | tee -a "$LOG_FILE"

# Summary
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Phase 0 Summary${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}🔍 Dry run complete - no changes made${NC}"
    echo ""
    echo "Run without --dry-run to apply changes."
elif [ "$SKIP_SCHEMA" = false ] && [ "$SKIP_ANNOTATIONS" = false ]; then
    echo -e "${GREEN}✅ Schema extended with semantic properties${NC}"
    echo -e "${GREEN}✅ 10 root annotations loaded into database${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Implement deterministic baseline (scripts/test_deterministic_baseline.py)"
    echo "  2. Test on 10 queries"
    echo "  3. Evaluate quality (target: ≥75%)"
    echo "  4. If successful, proceed to Phase 1 (200 roots)"
else
    echo -e "${YELLOW}⚠️  Partial run (some steps skipped)${NC}"
fi

echo ""
echo "Log saved to: $LOG_FILE"
echo ""
echo -e "${GREEN}Phase 0 preparation complete! 🎉${NC}"
echo ""
echo "To query the database with annotations:"
echo "  python -c \"import kuzu; db=kuzu.Database('$DATABASE'); \\"
echo "             conn=kuzu.Connection(db); \\"
echo "             print(conn.execute('MATCH (r:Radiko) RETURN r.radiko, r.verba_klaso, r.graveco_biografia LIMIT 5').get_as_pl())\""
