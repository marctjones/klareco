#!/bin/bash
# Import external semantic categories and regenerate M1 training data
#
# This script:
# 1. Runs feasibility study on ConceptNet and Wikidata
# 2. Imports categories from both external sources
# 3. Merges with manual categories
# 4. Regenerates M1 training data with expanded coverage
#
# Usage:
#   ./scripts/m1_import_external_categories.sh           # Interactive mode
#   ./scripts/m1_import_external_categories.sh --yes     # Skip confirmations
#   ./scripts/m1_import_external_categories.sh --test    # Test mode (10 nouns only)

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
    echo "❌ No venv found. Please create virtual environment first."
    exit 1
fi

# Parse arguments
AUTO_YES=false
TEST_MODE=false
SAMPLE_SIZE=50
MAX_NOUNS=""

for arg in "$@"; do
    case $arg in
        --yes|-y)
            AUTO_YES=true
            shift
            ;;
        --test)
            TEST_MODE=true
            MAX_NOUNS="--max-nouns 10"
            shift
            ;;
        *)
            ;;
    esac
done

echo "========================================================================"
echo "IMPORT EXTERNAL SEMANTIC CATEGORIES"
echo "========================================================================"
echo ""

# Phase 1: Feasibility Study
echo "------------------------------------------------------------------------"
echo "Phase 1: Feasibility Study"
echo "------------------------------------------------------------------------"
echo ""
echo "This will query ConceptNet and Wikidata to assess coverage on a sample"
echo "of $SAMPLE_SIZE uncategorized nouns."
echo ""

if [ "$AUTO_YES" = false ]; then
    read -p "Run feasibility study? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

echo ""
echo "Querying ConceptNet (sample size: $SAMPLE_SIZE)..."
python scripts/query_conceptnet.py --sample-size $SAMPLE_SIZE

echo ""
echo "Querying Wikidata (sample size: $SAMPLE_SIZE)..."
python scripts/query_wikidata.py --sample-size $SAMPLE_SIZE

echo ""
echo "Analyzing coverage..."
python scripts/analyze_external_coverage.py

# Check if analysis recommends proceeding
COVERAGE_FILE="data/vocabularies/external/coverage_analysis.json"
if [ ! -f "$COVERAGE_FILE" ]; then
    echo "❌ Coverage analysis failed - file not found: $COVERAGE_FILE"
    exit 1
fi

PROCEED=$(python -c "import json; print(json.load(open('$COVERAGE_FILE')).get('proceed_with_import', False))")

if [ "$PROCEED" != "True" ]; then
    echo ""
    echo "⚠️  Coverage analysis recommends NOT proceeding with import."
    echo "Check $COVERAGE_FILE for details."
    echo ""
    if [ "$AUTO_YES" = false ]; then
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 0
        fi
    else
        echo "Auto-yes mode: exiting due to low coverage."
        exit 1
    fi
fi

# Phase 2: Import from External Resources
echo ""
echo "------------------------------------------------------------------------"
echo "Phase 2: Import from External Resources"
echo "------------------------------------------------------------------------"
echo ""

if [ "$TEST_MODE" = true ]; then
    echo "⚠️  TEST MODE: Processing only 10 nouns"
    echo ""
fi

if [ "$AUTO_YES" = false ] && [ "$TEST_MODE" = false ]; then
    read -p "Proceed with full import? This may take 30-60 minutes. (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

echo ""
echo "Importing from ConceptNet..."
python scripts/import_from_conceptnet.py $MAX_NOUNS

echo ""
echo "Importing from Wikidata..."
python scripts/import_from_wikidata.py $MAX_NOUNS

# Phase 3: Merge Categories
echo ""
echo "------------------------------------------------------------------------"
echo "Phase 3: Merge Categories"
echo "------------------------------------------------------------------------"
echo ""
echo "Merging manual + ConceptNet + Wikidata categories..."
python scripts/merge_semantic_categories.py

# Check if merge was successful and meets target
MERGED_METADATA="data/vocabularies/semantic_categories_merged.metadata.json"
if [ ! -f "$MERGED_METADATA" ]; then
    echo "❌ Merge failed - metadata file not found: $MERGED_METADATA"
    exit 1
fi

COVERAGE=$(python -c "import json; print(json.load(open('$MERGED_METADATA'))['coverage']['percentage'])")
TARGET=$(python -c "import json; print(json.load(open('$MERGED_METADATA'))['target_coverage'] * 100)")

echo ""
echo "Merge complete!"
echo "Coverage: $COVERAGE% (target: $TARGET%)"

# Phase 4: Regenerate M1 Training Data
echo ""
echo "------------------------------------------------------------------------"
echo "Phase 4: Regenerate M1 Training Data"
echo "------------------------------------------------------------------------"
echo ""

if [ "$TEST_MODE" = true ]; then
    echo "⚠️  TEST MODE: Skipping M1 data regeneration"
    echo "To regenerate manually, run: ./scripts/m1_generate_semantic_data.sh --fresh"
else
    if [ "$AUTO_YES" = false ]; then
        read -p "Regenerate M1 training data with expanded categories? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Skipping M1 regeneration."
            echo "To regenerate manually, run: ./scripts/m1_generate_semantic_data.sh --fresh"
        else
            echo ""
            echo "Regenerating M1 training data..."
            if [ -f "scripts/m1_generate_semantic_data.sh" ]; then
                ./scripts/m1_generate_semantic_data.sh --fresh
            else
                echo "⚠️  m1_generate_semantic_data.sh not found, running Python script directly..."
                python scripts/generate_m1_semantic_data.py --fresh
            fi
        fi
    else
        echo "Auto-yes mode: Regenerating M1 training data..."
        if [ -f "scripts/m1_generate_semantic_data.sh" ]; then
            ./scripts/m1_generate_semantic_data.sh --fresh
        else
            python scripts/generate_m1_semantic_data.py --fresh
        fi
    fi
fi

echo ""
echo "========================================================================"
echo "✓ EXTERNAL CATEGORY IMPORT COMPLETE!"
echo "========================================================================"
echo ""
echo "Summary:"
echo "  - Merged categories: data/vocabularies/semantic_categories_merged.json"
echo "  - Metadata: data/vocabularies/semantic_categories_merged.metadata.json"
echo "  - Coverage: $COVERAGE% (target: $TARGET%)"
echo ""
echo "Next steps:"
echo "  1. Review merged categories: cat data/vocabularies/semantic_categories_merged.metadata.json"
echo "  2. Train M1 model: ./scripts/m1_train_selectional.sh"
echo "  3. Validate results: python -m pytest tests/test_m1_selectional.py"
echo ""
