#!/bin/bash
#
# Klareco Corpus Builder - Robust Runner
#
# Features:
# - Automatic virtual environment setup
# - Checkpointing and resumption
# - Comprehensive logging with timestamps
# - Error handling and cleanup
# - Progress monitoring
# - Safe interruption (Ctrl+C)
#
# Usage:
#   ./scripts/run_corpus_builder.sh [options]
#
# Options:
#   --clean     Remove checkpoint and start fresh
#   --fast      Use faster settings (may freeze on slow systems)
#   --gentle    Use gentler settings (slower but safer)
#   --no-ast    Skip AST generation (faster but no quality filtering)
#

set -e  # Exit on error (but we'll handle interrupts specially)

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/.venv"

# Data paths
DATA_DIR="$PROJECT_ROOT/data"
CLEANED_DIR="$DATA_DIR/cleaned"
OUTPUT_FILE="$DATA_DIR/corpus_with_sources_v2.jsonl"
CHECKPOINT_FILE="$DATA_DIR/build_corpus_v2_checkpoint.json"

# Log paths
LOG_DIR="$PROJECT_ROOT/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/corpus_builder_$TIMESTAMP.log"
LATEST_LOG="$LOG_DIR/corpus_builder_latest.log"

# Default settings (balanced)
BATCH_SIZE=20
THROTTLE=0.1
MIN_PARSE_RATE=0.0
PARSE_TIMEOUT=30

# Parse command-line arguments
CLEAN_START=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_START=true
            shift
            ;;
        --fast)
            BATCH_SIZE=50
            THROTTLE=0.0
            PARSE_TIMEOUT=15
            echo "⚡ Fast mode enabled (15s parse timeout)"
            shift
            ;;
        --gentle)
            BATCH_SIZE=10
            THROTTLE=0.3
            PARSE_TIMEOUT=60
            echo "🐌 Gentle mode enabled (60s parse timeout)"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--clean] [--fast] [--gentle]"
            exit 1
            ;;
    esac
done

# ============================================================================
# Utility Functions
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

log_section() {
    echo "" | tee -a "$LOG_FILE"
    echo "============================================================================" | tee -a "$LOG_FILE"
    echo "$1" | tee -a "$LOG_FILE"
    echo "============================================================================" | tee -a "$LOG_FILE"
}

# ============================================================================
# Setup Functions
# ============================================================================

setup_directories() {
    log_section "Setting Up Directories"

    # Create necessary directories
    mkdir -p "$LOG_DIR"
    mkdir -p "$DATA_DIR"

    if [ ! -d "$CLEANED_DIR" ]; then
        log_error "Cleaned directory not found: $CLEANED_DIR"
        log_error "Please run text cleaning first!"
        exit 1
    fi

    log_success "Directories ready"
}

setup_virtualenv() {
    log_section "Setting Up Python Environment"

    # Find system python3 (avoid conda)
    SYSTEM_PYTHON=$(which python3 2>/dev/null)
    if [ -z "$SYSTEM_PYTHON" ]; then
        log_error "python3 not found in PATH"
        log_error "Please install Python 3.8+ using your system package manager"
        exit 1
    fi

    # Verify Python version (need 3.8+)
    PYTHON_VERSION=$($SYSTEM_PYTHON --version 2>&1 | awk '{print $2}')
    log_info "Found Python: $SYSTEM_PYTHON (version $PYTHON_VERSION)"

    # Check if virtual environment exists
    if [ ! -d "$VENV_DIR" ]; then
        log_info "Creating virtual environment at $VENV_DIR"
        $SYSTEM_PYTHON -m venv "$VENV_DIR"

        if [ $? -ne 0 ]; then
            log_error "Failed to create virtual environment"
            log_error "Make sure python3-venv is installed:"
            log_error "  Ubuntu/Debian: sudo apt-get install python3-venv"
            log_error "  Fedora: sudo dnf install python3-venv"
            exit 1
        fi

        log_success "Virtual environment created"
    else
        log_info "Virtual environment already exists"
    fi

    # Activate virtual environment
    log_info "Activating virtual environment"
    source "$VENV_DIR/bin/activate"

    # Verify we're using venv python, not conda
    ACTIVE_PYTHON=$(which python3)
    if [[ "$ACTIVE_PYTHON" != "$VENV_DIR"* ]]; then
        log_warning "Virtual environment may not be activated correctly"
        log_warning "Expected: $VENV_DIR/bin/python3"
        log_warning "Got: $ACTIVE_PYTHON"
    fi

    # Upgrade pip (using venv's pip)
    log_info "Upgrading pip"
    python3 -m pip install --upgrade pip >> "$LOG_FILE" 2>&1

    # Install requirements
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        log_info "Installing requirements (this may take a few minutes)"
        python3 -m pip install -r "$PROJECT_ROOT/requirements.txt" >> "$LOG_FILE" 2>&1

        if [ $? -ne 0 ]; then
            log_error "Failed to install requirements"
            log_error "See $LOG_FILE for details"
            exit 1
        fi

        log_success "Requirements installed"
    else
        log_warning "No requirements.txt found, skipping installation"
    fi

    # Verify key dependencies
    log_info "Verifying dependencies"
    python3 -c "import sys; sys.path.insert(0, '$PROJECT_ROOT'); from klareco.parser import parse; print('Parser import OK')" >> "$LOG_FILE" 2>&1
    if [ $? -eq 0 ]; then
        log_success "Dependencies verified"
    else
        log_error "Failed to import klareco modules"
        log_error "See $LOG_FILE for details"
        exit 1
    fi
}

check_checkpoint() {
    log_section "Checking for Previous Run"

    if [ -f "$CHECKPOINT_FILE" ]; then
        if [ "$CLEAN_START" = true ]; then
            log_warning "Clean start requested - removing checkpoint"
            rm "$CHECKPOINT_FILE"
            if [ -f "$OUTPUT_FILE" ]; then
                log_warning "Removing existing output file"
                rm "$OUTPUT_FILE"
            fi
            log_success "Starting fresh"
        else
            log_info "Found checkpoint file"
            CHECKPOINT_INFO=$(cat "$CHECKPOINT_FILE")
            log_info "Previous progress: $CHECKPOINT_INFO"
            log_warning "Will resume from checkpoint"
            log_info "Use --clean to start fresh"
        fi
    else
        log_info "No checkpoint found - starting from beginning"
    fi
}

show_system_info() {
    log_section "System Information"

    log_info "Hostname: $(hostname)"
    log_info "User: $(whoami)"
    log_info "Working directory: $PROJECT_ROOT"

    # Show system Python (before venv activation)
    SYSTEM_PY=$(which python3 2>/dev/null || echo "not found")
    SYSTEM_PY_VER=$($SYSTEM_PY --version 2>&1 || echo "N/A")
    log_info "System Python: $SYSTEM_PY ($SYSTEM_PY_VER)"

    # Show memory info
    if command -v free &> /dev/null; then
        TOTAL_MEM=$(free -h | awk '/^Mem:/{print $2}')
        AVAIL_MEM=$(free -h | awk '/^Mem:/{print $7}')
        log_info "Memory: $AVAIL_MEM available of $TOTAL_MEM total"
    fi

    # Show disk space
    DISK_SPACE=$(df -h "$DATA_DIR" | awk 'NR==2{print $4}')
    log_info "Disk space available: $DISK_SPACE"

    # Check if corpus builder script exists
    BUILDER_SCRIPT="$PROJECT_ROOT/scripts/build_corpus_v2.py"
    if [ ! -f "$BUILDER_SCRIPT" ]; then
        log_error "Corpus builder script not found: $BUILDER_SCRIPT"
        exit 1
    fi
    log_info "Builder script: $BUILDER_SCRIPT"
}

# ============================================================================
# Cleanup and Signal Handling
# ============================================================================

cleanup() {
    EXIT_CODE=$?

    echo ""  # New line after potential Ctrl+C
    log_section "Cleanup"

    if [ $EXIT_CODE -eq 0 ]; then
        log_success "Corpus builder completed successfully!"

        # Show final statistics
        if [ -f "$OUTPUT_FILE" ]; then
            SENTENCE_COUNT=$(wc -l < "$OUTPUT_FILE")
            FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
            log_success "Total sentences: $SENTENCE_COUNT"
            log_success "Output file size: $FILE_SIZE"
            log_success "Output location: $OUTPUT_FILE"
        fi

        # Remove checkpoint on success
        if [ -f "$CHECKPOINT_FILE" ]; then
            rm "$CHECKPOINT_FILE"
            log_info "Checkpoint removed (build complete)"
        fi
    else
        log_warning "Corpus builder interrupted or failed (exit code: $EXIT_CODE)"

        if [ -f "$CHECKPOINT_FILE" ]; then
            CHECKPOINT_INFO=$(cat "$CHECKPOINT_FILE")
            log_info "Progress saved in checkpoint: $CHECKPOINT_INFO"
            log_info "Run this script again to resume from checkpoint"
        fi
    fi

    log_info "Full log saved to: $LOG_FILE"
    log_info "Latest log link: $LATEST_LOG"

    # Deactivate virtual environment
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate 2>/dev/null || true
    fi
}

# Set up signal handlers
trap cleanup EXIT
trap 'log_warning "Received interrupt signal (Ctrl+C)"; exit 130' INT TERM

# ============================================================================
# Main Execution
# ============================================================================

main() {
    # Create log file and symlink
    touch "$LOG_FILE"
    ln -sf "$LOG_FILE" "$LATEST_LOG"

    log_section "Klareco Corpus Builder - Robust Runner"
    log_info "Started at: $(date)"
    log_info "Log file: $LOG_FILE"

    # Setup
    setup_directories
    show_system_info
    setup_virtualenv
    check_checkpoint

    # Show configuration
    log_section "Build Configuration"
    log_info "Batch size: $BATCH_SIZE (checkpoint frequency)"
    log_info "Throttle: ${THROTTLE}s (delay between batches)"
    log_info "Min parse rate: $MIN_PARSE_RATE (0.0 = no filtering)"
    log_info "Parse timeout: ${PARSE_TIMEOUT}s (problem sentences logged to data/problem_sentences.jsonl)"

    # Build command
    log_section "Running Corpus Builder"

    BUILDER_CMD="python3 $PROJECT_ROOT/scripts/build_corpus_v2.py \
        --cleaned-dir $CLEANED_DIR \
        --output $OUTPUT_FILE \
        --checkpoint $CHECKPOINT_FILE \
        --batch-size $BATCH_SIZE \
        --throttle $THROTTLE \
        --min-parse-rate $MIN_PARSE_RATE \
        --parse-timeout $PARSE_TIMEOUT"

    log_info "Command: $BUILDER_CMD"
    echo "" | tee -a "$LOG_FILE"

    # Run with full output to both terminal and log
    $BUILDER_CMD 2>&1 | tee -a "$LOG_FILE"

    # Exit code will be caught by trap
}

# Run main function
main
