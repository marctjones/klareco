#!/bin/bash
# Clean Esperanto corpus and optionally retrain models with cleaned data
#
# Usage:
#   ./clean_and_retrain.sh preview       # Preview cleaning only
#   ./clean_and_retrain.sh clean         # Clean corpus only
#   ./clean_and_retrain.sh full          # Clean + reindex + retrain everything

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

MODE=${1:-preview}

print_header() {
    echo -e "${BOLD}========================================================================"
    echo "$1"
    echo -e "========================================================================${NC}"
}

print_section() {
    echo ""
    echo -e "${BLUE}${BOLD}>>> $1${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

case "$MODE" in
    preview)
        print_header "CORPUS CLEANING PREVIEW"
        echo ""
        echo "This will show a preview of what would be cleaned without making changes."
        echo ""

        python scripts/clean_corpus_language.py \
            --input data/corpus_sentences.jsonl \
            --dry-run

        echo ""
        print_warning "This was a PREVIEW only. No files were modified."
        echo ""
        echo "To actually clean the corpus, run:"
        echo "  ./clean_and_retrain.sh clean"
        echo ""
        ;;

    clean)
        print_header "CORPUS CLEANING"
        echo ""

        # Backup original
        if [ -f "data/corpus_sentences.jsonl" ]; then
            BACKUP="data/corpus_sentences.jsonl.original_$(date +%Y%m%d_%H%M%S)"
            print_warning "Creating backup: $BACKUP"
            cp data/corpus_sentences.jsonl "$BACKUP"
            print_success "Backup created"
            echo ""
        fi

        # Clean corpus
        python scripts/clean_corpus_language.py \
            --input data/corpus_sentences.jsonl \
            --output data/corpus_sentences_cleaned.jsonl \
            --report data/corpus_cleaning_report.txt \
            --min-confidence 0.6 \
            --min-length 10

        echo ""
        print_success "Corpus cleaned!"
        echo ""
        echo "Files created:"
        echo "  data/corpus_sentences_cleaned.jsonl  - Cleaned corpus"
        echo "  data/corpus_cleaning_report.txt      - Detailed report"
        echo ""
        echo "Review the report:"
        echo "  less data/corpus_cleaning_report.txt"
        echo ""
        echo "To use the cleaned corpus for training:"
        echo "  mv data/corpus_sentences.jsonl data/corpus_sentences.jsonl.contaminated"
        echo "  mv data/corpus_sentences_cleaned.jsonl data/corpus_sentences.jsonl"
        echo "  ./clean_and_retrain.sh full"
        echo ""
        ;;

    full)
        print_header "FULL CLEANING AND RETRAINING PIPELINE"
        echo ""

        # Step 1: Clean corpus (if not already done)
        if [ ! -f "data/corpus_sentences_cleaned.jsonl" ]; then
            print_section "Step 1: Cleaning corpus"

            if [ -f "data/corpus_sentences.jsonl" ]; then
                BACKUP="data/corpus_sentences.jsonl.original_$(date +%Y%m%d_%H%M%S)"
                cp data/corpus_sentences.jsonl "$BACKUP"
                print_success "Backup: $BACKUP"
            fi

            python scripts/clean_corpus_language.py \
                --input data/corpus_sentences.jsonl \
                --output data/corpus_sentences_cleaned.jsonl \
                --report data/corpus_cleaning_report.txt

            print_success "Corpus cleaned"
        else
            print_warning "Using existing cleaned corpus: data/corpus_sentences_cleaned.jsonl"
        fi

        # Step 2: Replace corpus with cleaned version
        print_section "Step 2: Activating cleaned corpus"

        if [ -f "data/corpus_sentences.jsonl" ] && [ ! -f "data/corpus_sentences.jsonl.contaminated" ]; then
            mv data/corpus_sentences.jsonl data/corpus_sentences.jsonl.contaminated
            print_success "Original moved to: data/corpus_sentences.jsonl.contaminated"
        fi

        cp data/corpus_sentences_cleaned.jsonl data/corpus_sentences.jsonl
        print_success "Cleaned corpus activated"

        # Step 3: Reindex corpus with GNN
        print_section "Step 3: Reindexing corpus with GNN"

        if [ -f "models/tree_lstm/checkpoint_epoch_20.pt" ]; then
            ./reindex_with_new_model.sh
            print_success "Corpus reindexed"
        else
            print_warning "GNN model not found - skipping reindexing"
            print_warning "Run ./retrain_gnn.sh first if you need GNN retraining"
        fi

        # Step 4: Retrain with more context (optional)
        print_section "Step 4: Retraining with extended context (optional)"

        echo ""
        echo "The corpus is now cleaned and reindexed."
        echo ""
        echo "If you want to also upgrade to 50+ context documents, run:"
        echo "  ./retrain_with_more_context.sh --context 50"
        echo ""
        print_success "Pipeline complete!"
        echo ""
        echo "Summary:"
        echo "  ✓ Corpus cleaned (English removed)"
        echo "  ✓ Corpus reindexed with GNN"
        echo "  • Ready for queries with clean data"
        echo ""
        ;;

    *)
        echo "Usage: $0 {preview|clean|full}"
        echo ""
        echo "Modes:"
        echo "  preview  - Preview what would be cleaned (no changes)"
        echo "  clean    - Clean corpus only (creates cleaned file)"
        echo "  full     - Clean + reindex + ready for retraining"
        echo ""
        echo "Examples:"
        echo "  $0 preview    # See what would be removed"
        echo "  $0 clean      # Clean the corpus"
        echo "  $0 full       # Clean and reindex everything"
        echo ""
        exit 1
        ;;
esac
