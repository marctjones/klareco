#!/bin/bash
# =============================================================================
# Cleanup Disk Space - Safely Remove Temporary and Backup Files
# =============================================================================
# Removes:
# - Backup files (corpus/wikipedia backups from old sessions)
# - Python cache files (__pycache__, .pyc)
# - Old log files
# - CSV exports (intermediate files used to load Kuzu database)
#
# Before deleting CSV exports, verifies the Kuzu database is working.
#
# Usage:
#   ./scripts/cleanup_disk_space.sh           # Interactive mode (asks before deleting)
#   ./scripts/cleanup_disk_space.sh --dry-run # Show what would be deleted
#   ./scripts/cleanup_disk_space.sh --yes     # Delete without confirmation
# =============================================================================

set -e

# Project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
DRY_RUN=false
AUTO_YES=false
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --yes)
            AUTO_YES=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--dry-run] [--yes]"
            echo ""
            echo "Options:"
            echo "  --dry-run    Show what would be deleted without deleting"
            echo "  --yes        Delete without confirmation"
            echo "  --help       Show this help message"
            exit 0
            ;;
    esac
done

# Log file
LOG_FILE="logs/cleanup_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo -e "${BLUE}=== Klareco Disk Space Cleanup ===${NC}"
echo ""
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}DRY RUN MODE - Nothing will be deleted${NC}"
    echo ""
fi

# Helper function to calculate size
get_size() {
    du -sh "$1" 2>/dev/null | cut -f1 || echo "0"
}

# Helper function to ask yes/no
ask_yes_no() {
    if [ "$AUTO_YES" = true ]; then
        return 0
    fi
    while true; do
        read -p "$1 (y/n): " yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}

# Track total space freed
TOTAL_FREED=0

# =============================================================================
# Category 1: Backup Files (ALWAYS SAFE)
# =============================================================================
echo -e "${YELLOW}=== Category 1: Backup Files ===${NC}"
echo "These are old backup copies of data files that can be safely deleted."
echo ""

BACKUP_FILES=(
    "data/enhanced_corpus/corpus_with_metadata.jsonl.backup_20260224_172318"
    "data/extracted/wikipedia_sentences.jsonl.backup"
)

BACKUP_SIZE=0
BACKUP_COUNT=0
for file in "${BACKUP_FILES[@]}"; do
    if [ -f "$file" ]; then
        size=$(get_size "$file")
        echo "  - $file ($size)"
        BACKUP_COUNT=$((BACKUP_COUNT + 1))
        # Convert size to bytes for total (rough estimate)
        if [[ $size == *G ]]; then
            gb=${size%G}
            BACKUP_SIZE=$((BACKUP_SIZE + ${gb%.*}))
        fi
    fi
done

if [ $BACKUP_COUNT -gt 0 ]; then
    echo ""
    echo -e "${GREEN}Total backup files: $BACKUP_COUNT (~${BACKUP_SIZE}GB)${NC}"

    if ask_yes_no "Delete backup files?"; then
        for file in "${BACKUP_FILES[@]}"; do
            if [ -f "$file" ]; then
                if [ "$DRY_RUN" = false ]; then
                    echo "Deleting: $file" | tee -a "$LOG_FILE"
                    rm "$file"
                else
                    echo "[DRY RUN] Would delete: $file"
                fi
            fi
        done
        TOTAL_FREED=$((TOTAL_FREED + BACKUP_SIZE))
        echo -e "${GREEN}✓ Backup files deleted${NC}"
    else
        echo "Skipping backup files"
    fi
else
    echo "No backup files found"
fi
echo ""

# =============================================================================
# Category 2: Python Cache Files (ALWAYS SAFE)
# =============================================================================
echo -e "${YELLOW}=== Category 2: Python Cache Files ===${NC}"
echo "Python automatically regenerates these. Always safe to delete."
echo ""

PYCACHE_COUNT=$(find . -type d -name "__pycache__" 2>/dev/null | wc -l)
PYC_COUNT=$(find . -type f -name "*.pyc" 2>/dev/null | wc -l)

echo "  - __pycache__ directories: $PYCACHE_COUNT"
echo "  - .pyc files: $PYC_COUNT"
echo ""
echo -e "${GREEN}Total: ~50-100MB${NC}"

if ask_yes_no "Delete Python cache files?"; then
    if [ "$DRY_RUN" = false ]; then
        echo "Deleting __pycache__ directories..." | tee -a "$LOG_FILE"
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        echo "Deleting .pyc files..." | tee -a "$LOG_FILE"
        find . -type f -name "*.pyc" -delete 2>/dev/null || true
    else
        echo "[DRY RUN] Would delete __pycache__ and .pyc files"
    fi
    echo -e "${GREEN}✓ Python cache files deleted${NC}"
else
    echo "Skipping Python cache files"
fi
echo ""

# =============================================================================
# Category 3: Old Log Files (SAFE)
# =============================================================================
echo -e "${YELLOW}=== Category 3: Old Log Files ===${NC}"
echo "Old logs from previous corpus building sessions."
echo ""

OLD_LOGS=(
    "logs/corpus_builder_20251127_223123.log"
    "logs/corpus_builder_20251128_115716.log"
    "logs/corpus_rebuild_20251226_183333.log"
    "logs/corpus_rebuild_20251229_001434.log"
)

LOG_COUNT=0
LOG_SIZE=0
for log in "${OLD_LOGS[@]}"; do
    if [ -f "$log" ]; then
        size=$(get_size "$log")
        echo "  - $log ($size)"
        LOG_COUNT=$((LOG_COUNT + 1))
    fi
done

if [ $LOG_COUNT -gt 0 ]; then
    echo ""
    echo -e "${GREEN}Total old logs: $LOG_COUNT (~10MB)${NC}"

    if ask_yes_no "Delete old log files?"; then
        for log in "${OLD_LOGS[@]}"; do
            if [ -f "$log" ]; then
                if [ "$DRY_RUN" = false ]; then
                    echo "Deleting: $log" | tee -a "$LOG_FILE"
                    rm "$log"
                else
                    echo "[DRY RUN] Would delete: $log"
                fi
            fi
        done
        echo -e "${GREEN}✓ Old logs deleted${NC}"
    else
        echo "Skipping old logs"
    fi
else
    echo "No old log files found"
fi
echo ""

# =============================================================================
# Category 4: CSV Export (VERIFY DATABASE FIRST)
# =============================================================================
echo -e "${YELLOW}=== Category 4: CSV Export (Intermediate Files) ===${NC}"
echo "These are CSV files used to bulk-load data into the Kuzu database."
echo "They are no longer needed once the database is loaded and working."
echo ""
echo "Workflow: Corpus (JSONL) → CSV Export → Kuzu Database"
echo "                           ↑            ↑"
echo "                           Temporary    Final format"
echo ""

CSV_DIR="data/csv_export_v2.1_full"
DB_PATH="data/indexes/v2.1_kuzu_index_full"

if [ -d "$CSV_DIR" ]; then
    CSV_SIZE=$(get_size "$CSV_DIR")
    echo "  - $CSV_DIR ($CSV_SIZE)"
    echo ""

    # Verify database exists and works
    echo "Verifying Kuzu database is working..."
    if [ -d "$DB_PATH" ]; then
        # Activate venv for Python check
        if [ -d ".venv" ]; then
            source .venv/bin/activate
        elif [ -d "venv" ]; then
            source venv/bin/activate
        fi

        # Test database
        DB_CHECK=$(python -c "
import kuzu
try:
    db = kuzu.Database('$DB_PATH', read_only=True)
    conn = kuzu.Connection(db)
    result = conn.execute('MATCH (r:Radiko) RETURN count(r) as cnt')
    count = result.get_next()[0]
    print(f'OK:{count}')
except Exception as e:
    print(f'ERROR:{e}')
" 2>&1)

        if [[ $DB_CHECK == OK:* ]]; then
            radiko_count=${DB_CHECK#OK:}
            echo -e "${GREEN}✓ Database OK ($radiko_count Radiko nodes)${NC}"
            echo ""
            echo "Since the database is working, CSV export can be safely deleted."
            echo "You can regenerate it from the corpus if you ever need to rebuild the database."
            echo ""

            if ask_yes_no "Delete CSV export? (saves ~${CSV_SIZE})"; then
                if [ "$DRY_RUN" = false ]; then
                    echo "Deleting: $CSV_DIR" | tee -a "$LOG_FILE"
                    rm -rf "$CSV_DIR"
                    if [[ $CSV_SIZE == *G ]]; then
                        gb=${CSV_SIZE%G}
                        TOTAL_FREED=$((TOTAL_FREED + ${gb%.*}))
                    fi
                else
                    echo "[DRY RUN] Would delete: $CSV_DIR"
                fi
                echo -e "${GREEN}✓ CSV export deleted${NC}"
            else
                echo "Keeping CSV export"
            fi
        else
            echo -e "${RED}✗ Database check failed: $DB_CHECK${NC}"
            echo -e "${YELLOW}Keeping CSV export for safety${NC}"
        fi
    else
        echo -e "${YELLOW}Database not found at $DB_PATH${NC}"
        echo -e "${YELLOW}Keeping CSV export for safety${NC}"
    fi
else
    echo "CSV export directory not found (already deleted?)"
fi
echo ""

# =============================================================================
# Category 5: Old Model Checkpoints (OPTIONAL)
# =============================================================================
echo -e "${YELLOW}=== Category 5: Old Model Checkpoints (Optional) ===${NC}"
echo "Old model versions that are about to be replaced by the tier-filtered retraining (Epic #616)."
echo ""
echo -e "${YELLOW}NOTE: Only delete if you don't need to compare old vs new models.${NC}"
echo ""

OLD_MODELS=(
    "models/m1_semantic_full_backup"
    "models/m1_tier0_only"
    "models/m1_selectional_tier0"
    "models/m1_selectional_v2"
    "models/root_embeddings_tier0"
)

MODEL_COUNT=0
MODEL_SIZE=0
for model_dir in "${OLD_MODELS[@]}"; do
    if [ -d "$model_dir" ]; then
        size=$(get_size "$model_dir")
        echo "  - $model_dir ($size)"
        MODEL_COUNT=$((MODEL_COUNT + 1))
    fi
done

if [ $MODEL_COUNT -gt 0 ]; then
    echo ""
    echo -e "${GREEN}Total old models: $MODEL_COUNT (~60MB)${NC}"
    echo ""
    echo -e "${YELLOW}These models will be superseded by Epic #616 retraining.${NC}"
    echo "Keep them if you want to compare old vs new model quality."

    if ask_yes_no "Delete old model checkpoints?"; then
        for model_dir in "${OLD_MODELS[@]}"; do
            if [ -d "$model_dir" ]; then
                if [ "$DRY_RUN" = false ]; then
                    echo "Deleting: $model_dir" | tee -a "$LOG_FILE"
                    rm -rf "$model_dir"
                else
                    echo "[DRY RUN] Would delete: $model_dir"
                fi
            fi
        done
        echo -e "${GREEN}✓ Old models deleted${NC}"
    else
        echo "Keeping old models for comparison"
    fi
else
    echo "No old model checkpoints found"
fi
echo ""

# =============================================================================
# Summary
# =============================================================================
echo -e "${BLUE}=== Cleanup Summary ===${NC}"
if [ "$DRY_RUN" = false ]; then
    echo -e "${GREEN}Estimated space freed: ~${TOTAL_FREED}GB${NC}"
    echo ""
    echo "Log saved to: $LOG_FILE"
else
    echo -e "${YELLOW}DRY RUN - No files were deleted${NC}"
    echo ""
    echo "Run without --dry-run to actually delete files."
fi
echo ""
echo "Current disk usage:"
du -sh data/ models/ logs/ 2>/dev/null | column -t

echo ""
echo -e "${GREEN}Done!${NC}"
