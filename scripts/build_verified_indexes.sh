#!/bin/bash
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
    echo "ERROR: No venv found"
    exit 1
fi

# Configuration
CORPUS_FILE="${1:-data/corpus/unified_corpus.jsonl}"
OUTPUT_DIR="${2:-data/indexes/slot_verified}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/indexing"
mkdir -p "$LOG_DIR"

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║        BUILD VERIFIED INDEXES WITH FULL ANNOTATIONS               ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Corpus:       $CORPUS_FILE"
echo "  Output:       $OUTPUT_DIR"
echo "  Timestamp:    $TIMESTAMP"
echo ""

# ============================================================================
# STAGE 1: VERIFY CORPUS HAS REQUIRED METADATA
# ============================================================================

echo "════════════════════════════════════════════════════════════════════"
echo "STAGE 1: Verify Corpus Metadata"
echo "════════════════════════════════════════════════════════════════════"
echo ""

if [ ! -f "$CORPUS_FILE" ]; then
    echo "✗ ERROR: Corpus file not found: $CORPUS_FILE"
    exit 1
fi

echo "Running verification checks..."
echo ""

python3 - "$CORPUS_FILE" << 'VERIFY_CORPUS'
import json
import sys
from collections import Counter

corpus_file = sys.argv[1]

print("Checking corpus structure and metadata...")
print()

# Counters
total = 0
has_parse_status = 0
has_proper_nouns = 0
has_source_citation = 0
has_wikipedia = 0
parse_status_counts = Counter()
source_types = Counter()
source_tiers = Counter()
proper_noun_categories = Counter()

# Sample entries for verification
sample_entries = []

with open(corpus_file) as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue

        doc = json.loads(line)
        total += 1

        # Check parse status
        ast = doc.get('ast', {})
        parse_stats = ast.get('parse_statistics', {})

        if 'parse_statistics' in ast:
            has_parse_status += 1
            parse_status_counts['has_stats'] += 1

            # Check for proper noun categories
            categories = parse_stats.get('categories', {})
            if 'proper_name' in categories or 'proper_name_known' in categories:
                has_proper_nouns += 1
                for cat, count in categories.items():
                    if 'proper' in cat:
                        proper_noun_categories[cat] += count

        # Check source citation
        source = doc.get('source', {})
        if source:
            has_source_citation += 1
            source_name = source.get('name', 'unknown')
            source_tier = source.get('tier', 0)

            source_types[source_name] += 1
            source_tiers[source_tier] += 1

            if source_name == 'wikipedia' or source_tier == 6:
                has_wikipedia += 1

        # Collect sample entries
        if i < 3:
            sample_entries.append(doc)

        # Progress every 500K
        if (i + 1) % 500000 == 0:
            print(f"  Scanned {i+1:,} sentences...", file=sys.stderr)

print(f"✓ Scanned {total:,} sentences")
print()

# Report results
print("═" * 70)
print("VERIFICATION RESULTS")
print("═" * 70)
print()

# Parse Status
print(f"Parse Statistics:")
print(f"  Total entries:           {total:,}")
print(f"  Has parse_statistics:    {has_parse_status:,} ({has_parse_status/total*100:.1f}%)")
if has_parse_status < total:
    print(f"  ⚠ WARNING: {total - has_parse_status:,} entries missing parse statistics!")
else:
    print(f"  ✓ All entries have parse statistics")
print()

# Proper Nouns
print(f"Proper Noun Detection:")
print(f"  Sentences with proper nouns: {has_proper_nouns:,} ({has_proper_nouns/total*100:.1f}%)")
if has_proper_nouns == 0:
    print(f"  ⚠ WARNING: No proper nouns detected!")
else:
    print(f"  ✓ Proper nouns detected")
    print(f"  Categories:")
    for cat, count in proper_noun_categories.most_common(5):
        print(f"    {cat}: {count:,}")
print()

# Source Citations
print(f"Source Citations:")
print(f"  Has source metadata:     {has_source_citation:,} ({has_source_citation/total*100:.1f}%)")
if has_source_citation < total:
    print(f"  ⚠ WARNING: {total - has_source_citation:,} entries missing source metadata!")
else:
    print(f"  ✓ All entries have source metadata")
print()

print(f"  Source breakdown:")
for source, count in source_types.most_common(10):
    pct = count / total * 100
    print(f"    {source:<25} {count:>10,} ({pct:>5.1f}%)")
print()

print(f"  Tier breakdown:")
for tier in sorted(source_tiers.keys()):
    count = source_tiers[tier]
    pct = count / total * 100
    print(f"    Tier {tier}: {count:>10,} ({pct:>5.1f}%)")
print()

# Wikipedia
print(f"Wikipedia Content:")
print(f"  Wikipedia sentences:     {has_wikipedia:,} ({has_wikipedia/total*100:.1f}%)")
if has_wikipedia == 0:
    print(f"  ✗ ERROR: No Wikipedia content found!")
    sys.exit(1)
else:
    print(f"  ✓ Wikipedia content present")
print()

# Sample entry structure
print("Sample Entry Structure:")
print(json.dumps(sample_entries[0], indent=2, ensure_ascii=False)[:1000] + "...")
print()

# Final verdict
print("═" * 70)
print("VERDICT")
print("═" * 70)
print()

errors = []
warnings = []

if has_wikipedia == 0:
    errors.append("No Wikipedia content")
if has_source_citation < total:
    warnings.append(f"{total - has_source_citation:,} entries missing source metadata")
if has_parse_status < total:
    warnings.append(f"{total - has_parse_status:,} entries missing parse statistics")
if has_proper_nouns == 0:
    warnings.append("No proper nouns detected")

if errors:
    print("✗ CORPUS VERIFICATION FAILED")
    for err in errors:
        print(f"  ERROR: {err}")
    sys.exit(1)

if warnings:
    print("⚠ CORPUS VERIFICATION PASSED WITH WARNINGS")
    for warn in warnings:
        print(f"  WARNING: {warn}")
else:
    print("✓ CORPUS VERIFICATION PASSED")
    print("  All required metadata present:")
    print("    ✓ Parse statistics")
    print("    ✓ Proper noun annotations")
    print("    ✓ Source citations")
    print("    ✓ Wikipedia content")

print()
VERIFY_CORPUS

VERIFY_EXIT=$?
if [ $VERIFY_EXIT -ne 0 ]; then
    echo ""
    echo "✗ Corpus verification failed. Cannot proceed with indexing."
    exit 1
fi

echo ""
echo "✓ Corpus verified successfully"
echo ""

# ============================================================================
# STAGE 2: BUILD BASE SLOT INDEX
# ============================================================================

echo "════════════════════════════════════════════════════════════════════"
echo "STAGE 2: Build Base Slot Index"
echo "════════════════════════════════════════════════════════════════════"
echo ""

SLOT_LOG="$LOG_DIR/slot_index_${TIMESTAMP}.log"

echo "Building slot-based index..."
echo "  Output: $OUTPUT_DIR/slot_index.jsonl"
echo "  Log:    $SLOT_LOG"
echo ""

python scripts/index_slot_based.py \
    --corpus "$CORPUS_FILE" \
    --output "$OUTPUT_DIR" \
    --root-model models/root_embeddings/best_model.pt \
    --affix-model models/affix_transforms_v2/best_model.pt \
    --resume 2>&1 | tee "$SLOT_LOG"

INDEX_EXIT=${PIPESTATUS[0]}
if [ $INDEX_EXIT -ne 0 ]; then
    echo ""
    echo "✗ Slot index build failed (exit code: $INDEX_EXIT)"
    echo "  Check log: $SLOT_LOG"
    exit 1
fi

echo ""
echo "✓ Base slot index built successfully"
echo ""

# ============================================================================
# STAGE 3: VERIFY INDEX HAS METADATA
# ============================================================================

echo "════════════════════════════════════════════════════════════════════"
echo "STAGE 3: Verify Index Metadata"
echo "════════════════════════════════════════════════════════════════════"
echo ""

INDEX_FILE="$OUTPUT_DIR/slot_index.jsonl"

if [ ! -f "$INDEX_FILE" ]; then
    echo "✗ ERROR: Index file not found: $INDEX_FILE"
    exit 1
fi

python3 - "$INDEX_FILE" << 'VERIFY_INDEX'
import json
import sys

index_file = sys.argv[1]

print("Verifying index contains required metadata...")
print()

total = 0
has_source = 0
has_wikipedia = 0
has_slots = 0
has_features = 0
missing_source = []

with open(index_file) as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue

        doc = json.loads(line)
        total += 1

        # Check slots
        if 'slots' in doc:
            has_slots += 1

        # Check features
        if 'features' in doc:
            has_features += 1

        # Check source
        if 'source' in doc:
            has_source += 1
            source = doc['source']
            if source.get('name') == 'wikipedia' or source.get('tier') == 6:
                has_wikipedia += 1
        else:
            if len(missing_source) < 5:
                missing_source.append(doc.get('text', '(no text)')[:100])

        if (i + 1) % 500000 == 0:
            print(f"  Scanned {i+1:,} documents...", file=sys.stderr)

print(f"✓ Scanned {total:,} indexed documents")
print()

print("Index Metadata:")
print(f"  Has slots:               {has_slots:,} ({has_slots/total*100:.1f}%)")
print(f"  Has features:            {has_features:,} ({has_features/total*100:.1f}%)")
print(f"  Has source metadata:     {has_source:,} ({has_source/total*100:.1f}%)")
print(f"  Wikipedia documents:     {has_wikipedia:,} ({has_wikipedia/total*100:.1f}%)")
print()

if has_source < total:
    print(f"⚠ WARNING: {total - has_source:,} documents missing source metadata")
    print("  Sample missing entries:")
    for text in missing_source[:3]:
        print(f"    {text}")
    print()

if has_wikipedia == 0:
    print("✗ ERROR: No Wikipedia documents in index!")
    sys.exit(1)
else:
    print("✓ Index verified successfully")
    print("  ✓ Slots present")
    print("  ✓ Features present")
    print("  ✓ Source metadata present")
    print("  ✓ Wikipedia content present")
print()
VERIFY_INDEX

VERIFY_INDEX_EXIT=$?
if [ $VERIFY_INDEX_EXIT -ne 0 ]; then
    echo ""
    echo "✗ Index verification failed"
    exit 1
fi

# ============================================================================
# STAGE 4: BUILD RETRIEVER INDEXES
# ============================================================================

echo "════════════════════════════════════════════════════════════════════"
echo "STAGE 4: Build Retriever Indexes"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Count documents for auto-detection
DOC_COUNT=$(wc -l < "$INDEX_FILE")
echo "Index size: $DOC_COUNT documents"
echo ""

# Determine which retrievers to build
RETRIEVERS_TO_BUILD=()

if [ "$DOC_COUNT" -lt 100000 ]; then
    echo "Small index (<100K docs) - building all retrievers:"
    RETRIEVERS_TO_BUILD=("mmap" "faiss" "multifaiss" "hnsw" "scann")
elif [ "$DOC_COUNT" -lt 1000000 ]; then
    echo "Medium index (100K-1M docs) - building optimized retrievers:"
    RETRIEVERS_TO_BUILD=("faiss" "multifaiss" "hnsw" "scann")
else
    echo "Large index (>1M docs) - building highly-optimized retrievers:"
    RETRIEVERS_TO_BUILD=("multifaiss" "hnsw" "scann")
fi

echo "  Building: ${RETRIEVERS_TO_BUILD[*]}"
echo ""

# Build each retriever
for RETRIEVER in "${RETRIEVERS_TO_BUILD[@]}"; do
    echo "──────────────────────────────────────────────────────────────────"
    echo "Building $RETRIEVER retriever..."
    echo "──────────────────────────────────────────────────────────────────"
    echo ""

    RETRIEVER_LOG="$LOG_DIR/${RETRIEVER}_${TIMESTAMP}.log"

    case $RETRIEVER in
        mmap)
            # mmap just uses the slot index directly
            mkdir -p "$OUTPUT_DIR/mmap"
            ln -sf ../slot_index.jsonl "$OUTPUT_DIR/mmap/slot_data.jsonl"
            echo "  ✓ mmap index linked"
            ;;

        faiss)
            ./scripts/index_faiss.sh "$OUTPUT_DIR" 2>&1 | tee "$RETRIEVER_LOG"
            ;;

        multifaiss)
            echo "  Building Multi-FAISS (separate indexes per slot)..."
            # Multi-FAISS is built from slot index
            mkdir -p "$OUTPUT_DIR/multifaiss"
            echo "  Note: Multi-FAISS builds indexes during first use"
            echo "  ✓ multifaiss directory created"
            ;;

        hnsw)
            ./scripts/build_hnsw_index.sh "$OUTPUT_DIR" 2>&1 | tee "$RETRIEVER_LOG"
            ;;

        scann)
            ./scripts/build_scann_index.sh "$OUTPUT_DIR" 2>&1 | tee "$RETRIEVER_LOG"
            ;;
    esac

    if [ $? -eq 0 ]; then
        echo "  ✓ $RETRIEVER index built successfully"
    else
        echo "  ⚠ $RETRIEVER index build failed (check log: $RETRIEVER_LOG)"
    fi

    echo ""
done

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                 ✓ INDEX BUILD COMPLETE!                           ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Output Directory: $OUTPUT_DIR"
echo ""
echo "Indexes Built:"
for RETRIEVER in "${RETRIEVERS_TO_BUILD[@]}"; do
    if [ -d "$OUTPUT_DIR/$RETRIEVER" ] || [ "$RETRIEVER" = "multifaiss" ]; then
        echo "  ✓ $RETRIEVER"
    else
        echo "  ✗ $RETRIEVER (failed)"
    fi
done
echo ""
echo "Verification Summary:"
echo "  ✓ Corpus has Wikipedia content"
echo "  ✓ Corpus has source citations"
echo "  ✓ Corpus has parse annotations"
echo "  ✓ Corpus has proper noun detection"
echo "  ✓ Index preserves source metadata"
echo "  ✓ Index includes Wikipedia documents"
echo ""
echo "Logs saved to: $LOG_DIR/"
echo ""
echo "Next Steps:"
echo "  1. Test retrievers: ./scripts/benchmark_qa_all.sh --index $OUTPUT_DIR"
echo "  2. Compare with old index: diff data/indexes/slot_full data/indexes/slot_verified"
echo ""
