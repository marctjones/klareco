#!/bin/bash
# Master Re-Index Script for Kuzu v2.1 Database
# Runs all 4 steps: Parse → Load → Semantic Ontology → ReVo
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log directory
LOG_DIR="logs/reindex"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/reindex_$(date +%Y%m%d_%H%M%S).log"

# Function to log with timestamp
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$MASTER_LOG"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓ $1${NC}" | tee -a "$MASTER_LOG"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗ $1${NC}" | tee -a "$MASTER_LOG"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠ $1${NC}" | tee -a "$MASTER_LOG"
}

# Function to check exit code
check_exit() {
    if [ $1 -ne 0 ]; then
        log_error "Step failed with exit code $1"
        log_error "Check logs for details"
        exit $1
    fi
}

# Start timer
START_TIME=$(date +%s)

echo ""
echo "================================================================================"
echo "               KUZU v2.1 DATABASE RE-INDEX"
echo "================================================================================"
echo ""
log "Master log: $MASTER_LOG"
echo ""

# Confirmation prompt
log_warning "This will DELETE the existing database at data/indexes/v2.1_kuzu_index_full/"
log_warning "All data will be regenerated from corpus and external sources."
echo ""
read -p "Are you sure you want to proceed? (yes/no): " CONFIRM
if [[ "$CONFIRM" != "yes" ]]; then
    log "Re-index cancelled by user"
    exit 0
fi

echo ""
log "Starting re-index process..."
echo ""

# ============================================================================
# STEP 1: Parse Corpus to CSV
# ============================================================================
log "STEP 1/4: Parse corpus to CSV (~4-6 hours)"
log "  Script: ./scripts/corpus_to_csv_v2.1.sh"
log "  Input: data/corpus/unified_corpus.jsonl (1.6GB, 5.39M sentences)"
log "  Output: data/csv_export_v2.1_full/"
echo ""

STEP1_START=$(date +%s)
./scripts/corpus_to_csv_v2.1.sh 2>&1 | tee -a "$MASTER_LOG"
check_exit ${PIPESTATUS[0]}
STEP1_END=$(date +%s)
STEP1_DURATION=$((STEP1_END - STEP1_START))

log_success "Step 1 complete in $((STEP1_DURATION / 60)) minutes"
echo ""

# ============================================================================
# STEP 2: Load CSV to Kuzu
# ============================================================================
log "STEP 2/4: Load CSV to Kuzu (~2 hours)"
log "  Script: ./scripts/load_csv_to_kuzu_v2.1.sh --fresh"
log "  Input: data/csv_export_v2.1_full/"
log "  Output: data/indexes/v2.1_kuzu_index_full/"
echo ""

STEP2_START=$(date +%s)
./scripts/load_csv_to_kuzu_v2.1.sh --fresh 2>&1 | tee -a "$MASTER_LOG"
check_exit ${PIPESTATUS[0]}
STEP2_END=$(date +%s)
STEP2_DURATION=$((STEP2_END - STEP2_START))

log_success "Step 2 complete in $((STEP2_DURATION / 60)) minutes"
echo ""

# ============================================================================
# STEP 3: Load Semantic Ontology
# ============================================================================
log "STEP 3/4: Load semantic ontology (~5-10 minutes)"
log "  Script: scripts/extend_kuzu_schema_semantic_ontology.py"
log "  Creates: VerbaKlaso, EntecaTipo, TemaRolo, AspektaKlaso, EnhavaSkemo, SkemaSloto"
log "  Links: 86 verb classes, 115 entity types"
echo ""

STEP3_START=$(date +%s)

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    log_error "No Python virtual environment found (.venv or venv)"
    exit 1
fi

python scripts/extend_kuzu_schema_semantic_ontology.py 2>&1 | tee -a "$MASTER_LOG"
check_exit ${PIPESTATUS[0]}
STEP3_END=$(date +%s)
STEP3_DURATION=$((STEP3_END - STEP3_START))

log_success "Step 3 complete in $STEP3_DURATION seconds"
echo ""

# ============================================================================
# STEP 4: Load ReVo Dictionary Relationships
# ============================================================================
log "STEP 4/4: Load ReVo dictionary relationships (~5-10 minutes)"
log "  Script: ./scripts/load_revo_v2.1.sh"
log "  Input: data/raw/eo/dictionaries/revo/revo_semantic_relations.json"
log "  Creates: 3,453 semantic relationships (synonyms, antonyms, hypernyms, etc.)"
echo ""

STEP4_START=$(date +%s)
./scripts/load_revo_v2.1.sh 2>&1 | tee -a "$MASTER_LOG"
check_exit ${PIPESTATUS[0]}
STEP4_END=$(date +%s)
STEP4_DURATION=$((STEP4_END - STEP4_START))

log_success "Step 4 complete in $STEP4_DURATION seconds"
echo ""

# ============================================================================
# VERIFICATION
# ============================================================================
log "Running verification checks..."
echo ""

python3 << 'VERIFY_EOF' 2>&1 | tee -a "$MASTER_LOG"
import kuzu

db = kuzu.Database("data/indexes/v2.1_kuzu_index_full")
conn = kuzu.Connection(db)

print("=== Database Verification ===\n")

# 1. Check sentence count
result = conn.execute("MATCH (ft:Frazoteksto) RETURN count(ft)")
ft_count = result.get_next()[0]
print(f"✓ Frazoteksto nodes: {ft_count:,}")
assert ft_count > 5_000_000, "Expected >5M sentences"

# 2. Check word count
result = conn.execute("MATCH (v:Vorto) RETURN count(v)")
v_count = result.get_next()[0]
print(f"✓ Vorto nodes: {v_count:,}")
assert v_count > 70_000_000, "Expected >70M words"

# 3. Check proper noun fix - "Kaj" should be konjunkcio, not propra_nomo
result = conn.execute("""
    MATCH (v:Vorto {plena_vorto: 'Kaj', vortspeco: 'propra_nomo'})
    RETURN count(v)
""")
kaj_propra = result.get_next()[0]
print(f"✓ 'Kaj' as propra_nomo: {kaj_propra:,} (should be 0)")
if kaj_propra > 0:
    print("  ⚠ WARNING: Found 'Kaj' marked as proper noun!")

result = conn.execute("""
    MATCH (v:Vorto {plena_vorto: 'Kaj', vortspeco: 'konjunkcio'})
    RETURN count(v)
""")
kaj_konj = result.get_next()[0]
print(f"✓ 'Kaj' as konjunkcio: {kaj_konj:,} (should be >0)")
assert kaj_konj > 0, "Expected 'Kaj' to be konjunkcio"

# 4. Check semantic ontology loaded
result = conn.execute("MATCH (vk:VerbaKlaso) RETURN count(vk)")
vk_count = result.get_next()[0]
print(f"✓ VerbaKlaso nodes: {vk_count}")
assert vk_count == 8, "Expected 8 verb classes"

result = conn.execute("MATCH (et:EntecaTipo) RETURN count(et)")
et_count = result.get_next()[0]
print(f"✓ EntecaTipo nodes: {et_count}")
assert et_count == 6, "Expected 6 entity types"

# 5. Check ReVo relationships loaded
result = conn.execute("MATCH ()-[r:REVO_SINONIMO]->() RETURN count(r)")
syn_count = result.get_next()[0]
print(f"✓ REVO_SINONIMO edges: {syn_count:,}")
assert syn_count > 1000, "Expected >1000 synonym relationships"

result = conn.execute("MATCH ()-[r:REVO_HIPERNIMO]->() RETURN count(r)")
hyper_count = result.get_next()[0]
print(f"✓ REVO_HIPERNIMO edges: {hyper_count:,}")
assert hyper_count > 1500, "Expected >1500 hypernym relationships"

print("\n✓ All verification checks passed!")

VERIFY_EOF

check_exit $?

# ============================================================================
# SUMMARY
# ============================================================================
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINS=$(((TOTAL_DURATION % 3600) / 60))

echo ""
echo "================================================================================"
log_success "RE-INDEX COMPLETE!"
echo "================================================================================"
echo ""
log "Time breakdown:"
log "  Step 1 (Parse corpus):      $((STEP1_DURATION / 60)) min"
log "  Step 2 (Load Kuzu):          $((STEP2_DURATION / 60)) min"
log "  Step 3 (Semantic ontology):  $STEP3_DURATION sec"
log "  Step 4 (ReVo relationships): $STEP4_DURATION sec"
log "  Total:                       ${TOTAL_HOURS}h ${TOTAL_MINS}m"
echo ""
log "Database location: data/indexes/v2.1_kuzu_index_full/"
log "Master log: $MASTER_LOG"
echo ""
log_success "Database is ready with fixed proper noun detection!"
echo ""

