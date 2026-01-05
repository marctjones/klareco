#!/bin/bash
#
# Build Unified Corpus with Wikipedia Data (Task #25)
#
# This script builds the complete Klareco corpus including:
# - Gutenberg books extraction
# - Wikipedia articles extraction
# - AST parsing and quality filtering
# - Tiered source weighting
#
# Features:
# - Progress tracking with live updates
# - Verbose mode for detailed logging
# - Automatic checkpointing (restartable)
# - Validation of results
# - Color-coded output
#
# Usage:
#   ./scripts/build_corpus_with_wikipedia.sh          # Normal mode
#   ./scripts/build_corpus_with_wikipedia.sh --verbose # Verbose mode
#   ./scripts/build_corpus_with_wikipedia.sh --fresh   # Start from scratch
#   ./scripts/build_corpus_with_wikipedia.sh --help    # Show help
#

set -e  # Exit on error
set -o pipefail  # Exit on pipe failure

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Paths
LOG_DIR="logs"
DATA_DIR="data"
CORPUS_DIR="$DATA_DIR/corpus"
EXTRACTED_DIR="$DATA_DIR/extracted"
OUTPUT_DIR="$CORPUS_DIR/build_$(date +%Y%m%d_%H%M%S)"
FINAL_OUTPUT="$CORPUS_DIR/unified_corpus.jsonl"
CHECKPOINT_FILE="$LOG_DIR/corpus_build_checkpoint.json"
LOG_FILE="$LOG_DIR/corpus_build_$(date +%Y%m%d_%H%M%S).log"

# Input files
BOOKS_INPUT="$EXTRACTED_DIR/books_sentences.jsonl"
WIKI_INPUT="$EXTRACTED_DIR/wikipedia_sentences.jsonl"

# Options
VERBOSE=false
FRESH=false
MIN_PARSE_RATE=0.5

# Colors
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    MAGENTA='\033[0;35m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    MAGENTA=''
    CYAN=''
    BOLD=''
    NC=''
fi

# ============================================================================
# Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_step() {
    echo -e "\n${CYAN}${BOLD}=== $1 ===${NC}\n" | tee -a "$LOG_FILE"
}

show_help() {
    cat << EOF
${BOLD}Build Unified Corpus with Wikipedia Data (Task #25)${NC}

${BOLD}Usage:${NC}
  $0 [OPTIONS]

${BOLD}Options:${NC}
  --verbose, -v     Enable verbose output
  --fresh, -f       Start from scratch (ignore checkpoints)
  --help, -h        Show this help message

${BOLD}What this script does:${NC}
  1. Validates input files (books + Wikipedia extractions)
  2. Builds corpus from Gutenberg books
  3. Builds corpus from Wikipedia articles
  4. Merges into unified corpus
  5. Validates Wikipedia data is included
  6. Shows summary statistics

${BOLD}Features:${NC}
  ✓ Automatic checkpointing (resume from interruptions)
  ✓ Progress tracking (sentences/sec, ETA)
  ✓ Quality filtering (parse rate > ${MIN_PARSE_RATE})
  ✓ Tiered source weighting
  ✓ Wikipedia metadata preserved

${BOLD}Estimated Time:${NC}
  1-2 hours for full corpus (~4.5M sentences)

${BOLD}Output:${NC}
  ${FINAL_OUTPUT}

${BOLD}Logs:${NC}
  ${LOG_FILE}

${BOLD}Examples:${NC}
  # Normal run
  $0

  # Verbose mode (detailed progress)
  $0 --verbose

  # Start fresh (ignore previous checkpoint)
  $0 --fresh

  # Resume after interruption
  $0    # Automatically resumes from checkpoint

EOF
}

check_prerequisites() {
    log_step "Checking Prerequisites"

    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "python3 not found. Please install Python 3.8+"
        exit 1
    fi
    log_info "Python: $(python3 --version)"

    # Check venv
    if [ -d ".venv" ]; then
        log_info "Virtual environment: .venv"
    elif [ -d "venv" ]; then
        log_info "Virtual environment: venv"
    else
        log_warning "No virtual environment found (.venv or venv)"
    fi

    # Create log directory
    mkdir -p "$LOG_DIR"
    log_success "Log directory ready: $LOG_DIR"

    # Check input files
    if [ ! -f "$BOOKS_INPUT" ]; then
        log_error "Books extraction not found: $BOOKS_INPUT"
        log_error "Run: ./scripts/extract_gutenberg.sh first"
        exit 1
    fi
    log_success "Books extraction found: $(ls -lh "$BOOKS_INPUT" | awk '{print $5}')"

    if [ ! -f "$WIKI_INPUT" ]; then
        log_error "Wikipedia extraction not found: $WIKI_INPUT"
        log_error "Run: ./scripts/extract_wikipedia.sh first"
        exit 1
    fi
    log_success "Wikipedia extraction found: $(ls -lh "$WIKI_INPUT" | awk '{print $5}')"

    # Count input sentences
    local books_count=$(wc -l < "$BOOKS_INPUT")
    local wiki_count=$(wc -l < "$WIKI_INPUT")
    local total_count=$((books_count + wiki_count))
    log_info "Input sentences: $books_count books + $wiki_count Wikipedia = $total_count total"
}

activate_venv() {
    if [ -d ".venv" ]; then
        log_info "Activating virtual environment: .venv"
        source .venv/bin/activate
    elif [ -d "venv" ]; then
        log_info "Activating virtual environment: venv"
        source venv/bin/activate
    else
        log_warning "No virtual environment found, using system Python"
    fi
}

backup_existing_corpus() {
    if [ -f "$FINAL_OUTPUT" ]; then
        local backup_path="${FINAL_OUTPUT}.backup_$(date +%Y%m%d_%H%M%S)"
        log_step "Backing Up Existing Corpus"
        log_info "Backing up existing corpus to: $(basename "$backup_path")"
        mv "$FINAL_OUTPUT" "$backup_path"
        log_success "Backup created: $(ls -lh "$backup_path" | awk '{print $5}')"
    fi
}

clean_checkpoint() {
    if [ "$FRESH" = true ] && [ -f "$CHECKPOINT_FILE" ]; then
        log_info "Removing checkpoint (--fresh mode)"
        rm -f "$CHECKPOINT_FILE"
    fi
}

build_corpus() {
    log_step "Building Corpus with Wikipedia Data"

    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    log_info "Output directory: $OUTPUT_DIR"

    # Determine fresh flag
    local fresh_flag=""
    if [ "$FRESH" = true ]; then
        fresh_flag="--fresh"
    fi

    # Build command
    local cmd="python3 scripts/build_enhanced_corpus.py"
    cmd="$cmd --stage all"
    cmd="$cmd --output-dir \"$OUTPUT_DIR\""
    cmd="$cmd --min-parse-rate $MIN_PARSE_RATE"
    cmd="$cmd $fresh_flag"

    log_info "Command: $cmd"
    echo ""

    # Run build with progress monitoring
    if eval $cmd 2>&1 | tee -a "$LOG_FILE"; then
        log_success "Corpus build completed"
    else
        log_error "Corpus build failed (see log: $LOG_FILE)"
        exit 1
    fi
}

merge_corpus_files() {
    log_step "Finalizing Corpus"

    local merged_corpus="$OUTPUT_DIR/corpus_with_metadata.jsonl"

    # Check merged file exists (created by Python script)
    if [ ! -f "$merged_corpus" ]; then
        log_error "Merged corpus not found: $merged_corpus"
        log_error "Python script may have failed"
        exit 1
    fi

    # Copy to final location
    cp "$merged_corpus" "$FINAL_OUTPUT"

    # Verify
    local final_count=$(wc -l < "$FINAL_OUTPUT")
    log_success "Corpus finalized: $final_count sentences"
    log_success "Output: $FINAL_OUTPUT ($(ls -lh "$FINAL_OUTPUT" | awk '{print $5}'))"
}

validate_wikipedia_inclusion() {
    log_step "Validating Wikipedia Inclusion"

    log_info "Checking for Wikipedia sentences in corpus..."

    # Check for Wikipedia metadata
    local wiki_count=$(python3 << 'EOF'
import json
import sys

count = 0
with open(sys.argv[1], 'r') as f:
    for line in f:
        doc = json.loads(line)
        source = doc.get('source', {})
        if source.get('name') == 'wikipedia' or source.get('tier') == 6:
            count += 1

print(count)
EOF
"$FINAL_OUTPUT")

    if [ "$wiki_count" -gt 0 ]; then
        log_success "Wikipedia sentences found: $wiki_count"

        # Check for key articles
        log_info "Checking for key articles..."
        python3 << 'EOF' "$FINAL_OUTPUT"
import json
import sys
from collections import Counter

key_articles = ['L. L. Zamenhof', 'Esperanto', 'La Espero']
found = Counter()

with open(sys.argv[1], 'r') as f:
    for line in f:
        doc = json.loads(line)
        source = doc.get('source', {})
        article = source.get('article_title', '')
        if article in key_articles:
            found[article] += 1

print("\nKey articles found:")
for article in key_articles:
    count = found.get(article, 0)
    if count > 0:
        print(f"  ✓ {article}: {count} sentences")
    else:
        print(f"  ✗ {article}: NOT FOUND")

if sum(found.values()) > 0:
    print(f"\n✓ SUCCESS: Found {sum(found.values())} sentences from key articles")
else:
    print("\n⚠ WARNING: No key articles found in corpus")
    sys.exit(1)
EOF
        if [ $? -eq 0 ]; then
            log_success "Key articles validated"
        else
            log_warning "Key articles check had warnings"
        fi
    else
        log_error "No Wikipedia sentences found in corpus!"
        log_error "Expected >0 Wikipedia sentences, found 0"
        exit 1
    fi
}

show_statistics() {
    log_step "Corpus Statistics"

    python3 << 'EOF' "$FINAL_OUTPUT"
import json
import sys
from collections import Counter

total = 0
by_tier = Counter()
by_source = Counter()
total_words = 0

with open(sys.argv[1], 'r') as f:
    for line in f:
        doc = json.loads(line)
        total += 1

        source = doc.get('source', {})
        tier = source.get('tier', 5)
        name = source.get('name', 'unknown')

        by_tier[tier] += 1
        by_source[name] += 1
        total_words += doc.get('word_count', len(doc['text'].split()))

print(f"Total sentences: {total:,}")
print(f"Total words: {total_words:,}")
print(f"Average words/sentence: {total_words/total:.1f}")
print()

print("By Tier:")
for tier in sorted(by_tier.keys()):
    count = by_tier[tier]
    pct = count / total * 100
    print(f"  Tier {tier}: {count:,} ({pct:.1f}%)")

print()
print("Top Sources:")
for source, count in by_source.most_common(10):
    pct = count / total * 100
    print(f"  {source}: {count:,} ({pct:.1f}%)")
EOF
}

cleanup() {
    log_step "Cleanup"

    # Remove checkpoint on success
    if [ -f "$CHECKPOINT_FILE" ]; then
        log_info "Removing checkpoint file"
        rm -f "$CHECKPOINT_FILE"
    fi

    # Keep output directory for reference
    log_info "Build artifacts preserved in: $OUTPUT_DIR"
}

show_summary() {
    log_step "Summary"

    echo -e "${GREEN}${BOLD}✓ Task #25 Complete!${NC}\n"

    echo -e "${BOLD}Output:${NC}"
    echo -e "  File: $FINAL_OUTPUT"
    echo -e "  Size: $(ls -lh "$FINAL_OUTPUT" | awk '{print $5}')"
    echo -e "  Lines: $(wc -l < "$FINAL_OUTPUT" | awk '{printf "%\047d\n", $1}')"
    echo ""

    echo -e "${BOLD}Logs:${NC}"
    echo -e "  $LOG_FILE"
    echo ""

    echo -e "${BOLD}Next Steps:${NC}"
    echo -e "  1. Rebuild index:"
    echo -e "     ${CYAN}./scripts/index_slot.sh --fresh${NC}"
    echo ""
    echo -e "  2. Test Q&A queries:"
    echo -e "     ${CYAN}python scripts/demo_slot_retrieval.py --query \"Kiu kreis Esperanton?\"${NC}"
    echo ""
    echo -e "  3. Run benchmarks:"
    echo -e "     ${CYAN}./scripts/benchmark_slot_retrievers.py${NC}"
    echo ""

    echo -e "${GREEN}Wikipedia data is now included in the corpus!${NC}"
}

# ============================================================================
# Main
# ============================================================================

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -f|--fresh)
                FRESH=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Show header
    echo -e "${CYAN}${BOLD}"
    echo "================================================================================"
    echo "  Build Unified Corpus with Wikipedia Data (Task #25)"
    echo "================================================================================"
    echo -e "${NC}"
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Log file: $LOG_FILE"
    echo ""

    # Execute steps
    check_prerequisites
    activate_venv
    backup_existing_corpus
    clean_checkpoint
    build_corpus
    merge_corpus_files
    validate_wikipedia_inclusion
    show_statistics
    cleanup
    show_summary

    echo ""
    echo -e "${GREEN}${BOLD}Completed: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
}

# Run main
main "$@"
