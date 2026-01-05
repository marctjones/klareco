#!/bin/bash
#
# Run All Klareco Data Quality Tests
#
# This script runs quality checks organized by dataset:
# 1. Vocabulary (clean_roots.json, fundamento_roots.json)
# 2. Cleaned Files (data/cleaned/*.txt)
# 3. Extracted Data (wikipedia_sentences.jsonl, books_sentences.jsonl)
# 4. Corpus (unified.jsonl, general_corpus.jsonl)
# 5. Index (embeddings, FAISS, metadata)
# 6. Models (semantic embeddings) - optional
#
# Usage:
#   ./scripts/test_all.sh              # Run all tests
#   ./scripts/test_all.sh --quick      # Skip slow/model tests
#   ./scripts/test_all.sh --verbose    # Show detailed output
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Parse arguments
QUICK_MODE=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./scripts/test_all.sh [--quick] [--verbose]"
            exit 1
            ;;
    esac
done

# Activate venv
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
elif [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
fi

echo -e "${BLUE}${BOLD}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}${BOLD}  Klareco Data Quality Test Suite${NC}"
echo -e "${BLUE}${BOLD}════════════════════════════════════════════════════════════${NC}"
echo ""

# Track results
TOTAL_PASSED=0
TOTAL_FAILED=0
TOTAL_SKIPPED=0

# Function to run a single test class and parse results
run_test_class() {
    local test_file=$1
    local class_name=$2
    local class_desc=$3

    if [[ ! -f "$test_file" ]]; then
        echo -e "  ${YELLOW}⊘${NC} $class_desc: ${YELLOW}test file not found${NC}"
        return
    fi

    # Run pytest and capture output
    local output
    output=$(python -m pytest "$test_file::$class_name" --no-header -q 2>&1) || true

    # Parse counts from the summary line (e.g., "10 passed in 0.5s" or "2 passed, 1 failed in 0.3s")
    local passed=0
    local failed=0
    local skipped=0

    # Extract numbers
    if echo "$output" | grep -qE '[0-9]+ passed'; then
        passed=$(echo "$output" | grep -oE '[0-9]+ passed' | head -1 | grep -oE '[0-9]+')
    fi
    if echo "$output" | grep -qE '[0-9]+ failed'; then
        failed=$(echo "$output" | grep -oE '[0-9]+ failed' | head -1 | grep -oE '[0-9]+')
    fi
    if echo "$output" | grep -qE '[0-9]+ skipped'; then
        skipped=$(echo "$output" | grep -oE '[0-9]+ skipped' | head -1 | grep -oE '[0-9]+')
    fi

    # Ensure numbers
    passed=${passed:-0}
    failed=${failed:-0}
    skipped=${skipped:-0}

    # Update totals
    TOTAL_PASSED=$((TOTAL_PASSED + passed))
    TOTAL_FAILED=$((TOTAL_FAILED + failed))
    TOTAL_SKIPPED=$((TOTAL_SKIPPED + skipped))

    # Show result based on verbosity
    if [[ "$VERBOSE" == true ]]; then
        # Verbose: run with -v --tb=short and show each individual test + failure details
        local verbose_output
        verbose_output=$(python -m pytest "$test_file::$class_name" --no-header -v --tb=short 2>&1) || true

        echo -e "  ${BOLD}$class_desc${NC}"

        # Parse individual test results - format: "path::Class::test_name PASSED [xx%]"
        echo "$verbose_output" | grep -E '::test_.*\s+(PASSED|FAILED|SKIPPED)' | while IFS= read -r line; do
            # Extract test name (after last ::test_)
            local test_name
            test_name=$(echo "$line" | sed 's/.*::test_\([^ ]*\).*/\1/' | tr '_' ' ')

            if echo "$line" | grep -q "PASSED"; then
                echo -e "    ${GREEN}✓${NC} $test_name"
            elif echo "$line" | grep -q "FAILED"; then
                echo -e "    ${RED}✗${NC} $test_name"
            elif echo "$line" | grep -q "SKIPPED"; then
                echo -e "    ${YELLOW}⊘${NC} $test_name (skipped)"
            fi
        done

        # Show failure details if any tests failed
        if [[ "$failed" -gt 0 ]]; then
            # Extract assertion errors from the output
            echo "$verbose_output" | grep -E "^E\s+" | head -10 | while IFS= read -r line; do
                # Remove the "E   " prefix and indent
                local msg
                msg=$(echo "$line" | sed 's/^E\s*//')
                echo -e "      ${RED}→${NC} $msg"
            done
        fi
    else
        # Non-verbose: show summary per class
        if [[ "$failed" -gt 0 ]]; then
            echo -e "  ${RED}✗${NC} $class_desc: ${GREEN}$passed passed${NC}, ${RED}$failed failed${NC}"
        elif [[ "$skipped" -gt 0 && "$passed" -eq 0 ]]; then
            echo -e "  ${YELLOW}⊘${NC} $class_desc: ${YELLOW}$skipped skipped${NC}"
        elif [[ "$passed" -eq 0 && "$failed" -eq 0 && "$skipped" -eq 0 ]]; then
            echo -e "  ${YELLOW}⊘${NC} $class_desc: ${YELLOW}no tests found${NC}"
        else
            local extra=""
            [[ "$skipped" -gt 0 ]] && extra=", ${YELLOW}$skipped skipped${NC}"
            echo -e "  ${GREEN}✓${NC} $class_desc: ${GREEN}$passed passed${NC}$extra"
        fi
    fi
}

# Function to run tests for a dataset
run_dataset_tests() {
    local test_file=$1
    local dataset_name=$2
    local description=$3
    shift 3
    local test_classes=("$@")

    echo -e "${CYAN}${BOLD}[$dataset_name]${NC} $description"
    echo -e "${CYAN}────────────────────────────────────────${NC}"

    if [[ ! -f "$test_file" ]]; then
        echo -e "  ${YELLOW}⊘ Test file not found: $test_file${NC}"
        echo ""
        return
    fi

    # Run each test class
    for test_class in "${test_classes[@]}"; do
        local class_name="${test_class%%:*}"
        local class_desc="${test_class#*:}"
        run_test_class "$test_file" "$class_name" "$class_desc"
    done
    echo ""
}

# ═══════════════════════════════════════════════════════════════════
# STAGE 0: VOCABULARY DATA
# ═══════════════════════════════════════════════════════════════════
echo -e "${BLUE}${BOLD}STAGE 0: VOCABULARY${NC}"
echo ""

run_dataset_tests "tests/test_data_quality.py" \
    "clean_roots.json" \
    "Core vocabulary (data/vocabularies/core/clean_roots.json)" \
    "TestCleanRootsVocabulary:Structure & Metadata" \
    "TestNoJunkInVocabulary:Content Quality"

run_dataset_tests "tests/test_data_quality.py" \
    "fundamento_roots.json" \
    "Fundamento roots (data/vocabularies/core/fundamento_roots.json)" \
    "TestTrainingDataConsistency:Cross-file Consistency"

# ═══════════════════════════════════════════════════════════════════
# STAGE 1: CLEANED TEXT FILES
# ═══════════════════════════════════════════════════════════════════
echo -e "${BLUE}${BOLD}STAGE 1: CLEANED DATA${NC}"
echo ""

run_dataset_tests "tests/test_cleaned_data.py" \
    "data/cleaned/*.txt" \
    "Cleaned text files (Gutenberg, Wikipedia)" \
    "TestCleanedDataExists:File Existence" \
    "TestGutenbergHeadersRemoved:Boilerplate Removal" \
    "TestCleanedContentQuality:Content Quality" \
    "TestCleanedDataStats:Statistics"

# ═══════════════════════════════════════════════════════════════════
# STAGE 2: EXTRACTED SENTENCES
# ═══════════════════════════════════════════════════════════════════
echo -e "${BLUE}${BOLD}STAGE 2: EXTRACTED DATA${NC}"
echo ""

run_dataset_tests "tests/test_extracted_data.py" \
    "wikipedia_sentences.jsonl" \
    "Wikipedia extraction (data/extracted/wikipedia_sentences.jsonl)" \
    "TestExtractedFilesExist:File Existence" \
    "TestWikipediaExtraction:Format & Metadata" \
    "TestWikipediaArticleCompleteness:Article Completeness" \
    "TestWikipediaContentCleanliness:Content Cleanliness"

run_dataset_tests "tests/test_extracted_data.py" \
    "books_sentences.jsonl" \
    "Books extraction (data/extracted/books_sentences.jsonl)" \
    "TestBooksExtraction:Format & Metadata"

run_dataset_tests "tests/test_extracted_data.py" \
    "Consistency" \
    "Cross-file consistency checks" \
    "TestExtractedDataConsistency:ID & Source Consistency"

run_dataset_tests "tests/test_wikipedia_benchmark.py" \
    "Benchmark Articles" \
    "86 popular articles (data/benchmarks/wikipedia_articles/)" \
    "TestBenchmarkDataExists:Benchmark Files" \
    "TestBenchmarkArticleCoverage:Article Coverage" \
    "TestBenchmarkContentCompleteness:Content Completeness" \
    "TestBenchmarkContentQuality:Content Quality" \
    "TestBenchmarkStatistics:Statistics"

# ═══════════════════════════════════════════════════════════════════
# STAGE 3: PARSED CORPUS
# ═══════════════════════════════════════════════════════════════════
echo -e "${BLUE}${BOLD}STAGE 3: CORPUS${NC}"
echo ""

run_dataset_tests "tests/test_corpus_integrity.py" \
    "unified.jsonl" \
    "Unified corpus (data/corpus/unified.jsonl)" \
    "TestCorpusExists:File Existence" \
    "TestCorpusFormat:JSONL Format" \
    "TestCorpusASTs:AST Quality" \
    "TestParseStatistics:Parse Rates" \
    "TestCorpusSources:Source Diversity" \
    "TestCorpusContentQuality:Content Quality"

# ═══════════════════════════════════════════════════════════════════
# STAGE 4: INDEX
# ═══════════════════════════════════════════════════════════════════
echo -e "${BLUE}${BOLD}STAGE 4: INDEX${NC}"
echo ""

run_dataset_tests "tests/test_index_integrity.py" \
    "embeddings.npy" \
    "Embeddings (data/indexes/*/embeddings.npy)" \
    "TestIndexExists:File Existence" \
    "TestEmbeddingsQuality:Embedding Quality"

run_dataset_tests "tests/test_index_integrity.py" \
    "FAISS Index" \
    "FAISS search index (data/indexes/*/*.bin)" \
    "TestFAISSIndex:Index Validity"

run_dataset_tests "tests/test_index_integrity.py" \
    "Metadata" \
    "Index metadata (data/indexes/*/metadata.jsonl)" \
    "TestMetadataConsistency:Metadata Consistency" \
    "TestIndexPerformance:Search Performance"

# ═══════════════════════════════════════════════════════════════════
# STAGE 5: MODELS (optional)
# ═══════════════════════════════════════════════════════════════════
if [[ "$QUICK_MODE" != true ]]; then
    echo -e "${BLUE}${BOLD}STAGE 5: MODELS${NC}"
    echo ""

    if [[ -f "tests/test_semantic_evaluation.py" ]]; then
        run_dataset_tests "tests/test_semantic_evaluation.py" \
            "Semantic Model" \
            "Root embeddings (models/root_embeddings/)" \
            "TestSemanticModel:Model Quality"
    else
        echo -e "  ${YELLOW}⊘ Model tests not found${NC}"
    fi
    echo ""
fi

# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
echo -e "${BLUE}${BOLD}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}${BOLD}  Summary${NC}"
echo -e "${BLUE}${BOLD}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${GREEN}Passed:${NC}  $TOTAL_PASSED"
if [[ $TOTAL_FAILED -gt 0 ]]; then
    echo -e "  ${RED}Failed:${NC}  $TOTAL_FAILED"
else
    echo -e "  ${GREEN}Failed:${NC}  $TOTAL_FAILED"
fi
if [[ $TOTAL_SKIPPED -gt 0 ]]; then
    echo -e "  ${YELLOW}Skipped:${NC} $TOTAL_SKIPPED"
fi
echo ""

if [[ $TOTAL_FAILED -gt 0 ]]; then
    echo -e "${RED}Some quality checks failed!${NC}"
    echo -e "Run with ${BOLD}--verbose${NC} for details, or:"
    echo -e "  python -m pytest tests/test_*.py -v --tb=short"
    exit 1
else
    echo -e "${GREEN}All quality checks passed!${NC}"
    exit 0
fi
