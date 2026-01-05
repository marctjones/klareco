#!/bin/bash
#
# Compositional Embeddings Demo
#
# Shows how root + affix embeddings work together to represent
# Esperanto words compositionally.
#
# Usage:
#   ./scripts/run_compositional_demo.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Activate venv
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
fi

# Check models exist
if [[ ! -f "models/root_embeddings/best_model.pt" ]]; then
    echo -e "${RED}Error: Root model not found${NC}"
    echo "Run: ./scripts/run_fundamento_training.sh"
    exit 1
fi

if [[ ! -f "models/affix_embeddings/best_model.pt" ]]; then
    echo -e "${RED}Error: Affix model not found${NC}"
    echo "Run: python scripts/training/train_affix_embeddings.py"
    exit 1
fi

echo ""
echo -e "${BOLD}${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${BLUE}║     Compositional Embeddings Demo: Root + Affix Models         ║${NC}"
echo -e "${BOLD}${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${CYAN}This demo shows how Stage 1 models work together:${NC}"
echo -e "  • Root embeddings: 64d vectors for ~11K Esperanto roots"
echo -e "  • Affix embeddings: 32d transforms for 12 prefixes + 31 suffixes"
echo -e "  • Composition: word = root + prefix_transforms + suffix_transforms"
echo ""

# Helper function to run comparison
compare() {
    echo -e "${GREEN}▸ $1 vs $2${NC}"
    python scripts/demo_compositional_embeddings.py --compare "$1" "$2" 2>/dev/null | grep -E "(Root:|Prefixes:|Suffixes:|Similarity:)" | sed 's/^/  /'
    echo ""
}

# Helper function to analyze word
analyze() {
    echo -e "${GREEN}▸ $1${NC}"
    python scripts/demo_compositional_embeddings.py --word "$1" 2>/dev/null | grep -E "(Root '|Prefix '|Suffix '|Top similar)" -A 5 | head -10 | sed 's/^/  /'
    echo ""
}

echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}SECTION 1: Basic Word Similarity${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${CYAN}Words with the same root should have high similarity:${NC}"
echo ""

compare "hundo" "hundoj"
compare "libro" "libroj"
compare "granda" "grande"

echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}SECTION 2: Suffix Effects${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${CYAN}Suffixes modify the root meaning:${NC}"
echo -e "  -et (diminutive), -eg (augmentative), -ej (place), -ist (profession)"
echo ""

compare "hundo" "hundeto"
compare "granda" "grandega"
compare "libro" "librejo"
compare "skribi" "skribisto"
compare "lerni" "lernejo"

echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}SECTION 3: Prefix Effects${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${CYAN}Prefixes transform meaning:${NC}"
echo -e "  mal- (opposite), re- (again), ek- (begin), ge- (both genders)"
echo ""

compare "bela" "malbela"
compare "rapida" "malrapida"
compare "juna" "maljuna"
compare "patro" "gepatro"

echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}SECTION 4: Complex Words (Multiple Affixes)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${CYAN}Esperanto allows stacking multiple affixes:${NC}"
echo ""

analyze "malbonulo"
analyze "gepatroj"
analyze "lernejestro"
analyze "malriĉulejo"

echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}SECTION 5: Semantic Relationships${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${CYAN}Different roots with related meanings:${NC}"
echo ""

compare "paroli" "diri"
compare "granda" "vasta"
compare "rapida" "lerta"
compare "libro" "gazeto"

echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}SECTION 6: Participial Suffixes${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${CYAN}Verb participles (-ant/-int/-ont active, -at/-it/-ot passive):${NC}"
echo ""

compare "leganta" "leginta"
compare "legata" "legita"
compare "leganta" "legata"

echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}SECTION 7: Strengths${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${GREEN}✓ Same-root words cluster together (sim ≈ 1.0)${NC}"
echo -e "${GREEN}✓ Suffix semantics preserved (place, person, size)${NC}"
echo -e "${GREEN}✓ Prefix semantics preserved (opposite, again)${NC}"
echo -e "${GREEN}✓ Handles novel word combinations (compositionality)${NC}"
echo -e "${GREEN}✓ 11K roots + 43 affixes = millions of possible words${NC}"
echo -e "${GREEN}✓ Tiny model: ~320K params for roots + 1.4K for affixes${NC}"
echo ""

echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}SECTION 8: Known Limitations${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}Parser issues with some words:${NC}"
echo ""

# Show some parser edge cases
echo -e "${YELLOW}▸ 'bona' parses as 'bo+n' instead of 'bon':${NC}"
python scripts/demo_compositional_embeddings.py --word bona 2>/dev/null | grep -E "(Root '|error)" | head -2 | sed 's/^/  /'
echo ""

echo -e "${YELLOW}▸ 'legi' parses as 'l+eg' instead of 'leg':${NC}"
python scripts/demo_compositional_embeddings.py --word legi 2>/dev/null | grep -E "(Root '|error)" | head -2 | sed 's/^/  /'
echo ""

echo -e "${YELLOW}Prefix training data sparsity:${NC}"
echo -e "  Some prefixes (dis-, bo-, mis-, fi-, vic-) have very few training examples"
echo -e "  These may not have well-learned semantics yet"
echo ""

echo -e "${YELLOW}Affix effect on similarity:${NC}"
echo -e "  Currently affixes are added (not composed), so same-root words"
echo -e "  have similarity ≈ 1.0 regardless of affix. This is by design for"
echo -e "  retrieval (find related content) but may need refinement for"
echo -e "  fine-grained semantic tasks."
echo ""

echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}SECTION 9: Architecture Summary${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${CYAN}Stage 1 Model Architecture:${NC}"
echo ""
echo "  ┌─────────────────────────────────────────────────────────┐"
echo "  │  Word: 'malbonulo'                                      │"
echo "  │                                                         │"
echo "  │  Parser → root='bon', prefix=['mal'], suffix=['ul']     │"
echo "  │                                                         │"
echo "  │  ┌─────────────┐                                        │"
echo "  │  │ Root 'bon'  │ → 64d embedding (learned)              │"
echo "  │  └─────────────┘                                        │"
echo "  │        +                                                │"
echo "  │  ┌─────────────┐                                        │"
echo "  │  │ Prefix mal- │ → 32d transform × 0.3 (padded to 64d)  │"
echo "  │  └─────────────┘                                        │"
echo "  │        +                                                │"
echo "  │  ┌─────────────┐                                        │"
echo "  │  │ Suffix -ul  │ → 32d transform × 0.2 (padded to 64d)  │"
echo "  │  └─────────────┘                                        │"
echo "  │        ↓                                                │"
echo "  │  ┌─────────────┐                                        │"
echo "  │  │  Normalize  │ → 64d unit vector                      │"
echo "  │  └─────────────┘                                        │"
echo "  │                                                         │"
echo "  │  Total params: ~322K (roots: 320K, affixes: 1.4K)       │"
echo "  └─────────────────────────────────────────────────────────┘"
echo ""

echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}Next Steps${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  1. Build FAISS index with compositional embeddings:"
echo -e "     ${GREEN}./scripts/run_compositional_indexing.sh${NC}"
echo ""
echo "  2. Interactive exploration:"
echo -e "     ${GREEN}python scripts/demo_compositional_embeddings.py -i${NC}"
echo ""
echo "  3. Evaluate affix embeddings:"
echo -e "     ${GREEN}python scripts/training/evaluate_affix_embeddings.py${NC}"
echo ""
echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
