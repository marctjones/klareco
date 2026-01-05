#!/bin/bash
#
# Validate Wikipedia Inclusion in Unified Corpus (Task #25)
#
# This standalone script verifies that Wikipedia data is properly
# included in the unified corpus and that key articles are present.
#
# Usage:
#   ./scripts/validate_corpus_wikipedia.sh
#   ./scripts/validate_corpus_wikipedia.sh /path/to/corpus.jsonl
#

set -e  # Exit on error
set -o pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Default corpus path
CORPUS_FILE="${1:-data/corpus/unified_corpus.jsonl}"

# Colors
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    BOLD=''
    NC=''
fi

# ============================================================================
# Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${CYAN}${BOLD}=== $1 ===${NC}\n"
}

# ============================================================================
# Main Validation
# ============================================================================

log_step "Wikipedia Corpus Validation (Task #25)"

# Check corpus file exists
if [ ! -f "$CORPUS_FILE" ]; then
    log_error "Corpus file not found: $CORPUS_FILE"
    exit 1
fi

log_success "Corpus file found: $CORPUS_FILE"
log_info "Size: $(ls -lh "$CORPUS_FILE" | awk '{print $5}')"
echo ""

# Count Wikipedia sentences
log_step "Stage 1: Count Wikipedia Sentences"

wiki_count=$(python3 - "$CORPUS_FILE" << 'EOF'
import json
import sys

count = 0
total = 0
with open(sys.argv[1], 'r') as f:
    for line in f:
        total += 1
        doc = json.loads(line)
        source = doc.get('source', {})
        if source.get('name') == 'wikipedia' or source.get('tier') == 6:
            count += 1

        # Progress indicator every 500K lines
        if total % 500000 == 0:
            print(f"  Scanned {total:,} sentences... (found {count:,} Wikipedia)", file=sys.stderr)

print(count)
EOF
)

if [ -n "$wiki_count" ] && [ "$wiki_count" -gt 0 ]; then
    log_success "Wikipedia sentences found: $wiki_count"
else
    log_error "No Wikipedia sentences found in corpus!"
    log_error "Expected >0 Wikipedia sentences, found ${wiki_count:-0}"
    exit 1
fi

# Check key articles
log_step "Stage 2: Verify Key Articles"

python3 - "$CORPUS_FILE" << 'EOF'
import json
import sys
from collections import Counter

key_articles = ['L. L. Zamenhof', 'Esperanto', 'La Espero']
found = Counter()
total = 0

with open(sys.argv[1], 'r') as f:
    for line in f:
        total += 1
        doc = json.loads(line)
        source = doc.get('source', {})
        article = source.get('article_title', '')
        if article in key_articles:
            found[article] += 1

        # Progress indicator every 500K lines
        if total % 500000 == 0:
            print(f"  Scanned {total:,} sentences...", file=sys.stderr)

print("Key articles found:")
for article in key_articles:
    count = found.get(article, 0)
    if count > 0:
        print(f"  ✓ {article}: {count} sentences")
    else:
        print(f"  ✗ {article}: NOT FOUND")

print()
if sum(found.values()) > 0:
    print(f"✓ SUCCESS: Found {sum(found.values())} sentences from key articles")
    sys.exit(0)
else:
    print("⚠ WARNING: No key articles found in corpus")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    log_success "Key articles validated"
else
    echo ""
    log_warning "Key articles check had warnings"
fi

# Corpus statistics
log_step "Stage 3: Corpus Statistics"

python3 - "$CORPUS_FILE" << 'EOF'
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
        source_type = source.get('name', 'unknown')

        by_tier[tier] += 1
        by_source[source_type] += 1
        total_words += doc.get('word_count', len(doc['text'].split()))

        # Progress indicator every 500K lines
        if total % 500000 == 0:
            print(f"  Computing statistics... {total:,} sentences", file=sys.stderr)

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
for source_type, count in by_source.most_common(10):
    pct = count / total * 100
    print(f"  {source_type}: {count:,} ({pct:.1f}%)")
EOF

# Summary
log_step "Summary"

echo -e "${GREEN}${BOLD}✓ Task #25 Wikipedia Validation Complete!${NC}\n"

echo -e "${BOLD}Corpus:${NC}"
echo -e "  File: $CORPUS_FILE"
echo -e "  Wikipedia sentences: $wiki_count"
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

echo -e "${GREEN}Wikipedia data successfully validated in corpus!${NC}"
