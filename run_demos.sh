#!/bin/bash
#
# Klareco Demo Runner
# Runs all working demos to showcase the deterministic parser and AST generation
#

set -e  # Exit on error

echo "=========================================="
echo "  KLARECO PARSER DEMOS"
echo "=========================================="
echo ""
echo "These demos show how Esperanto's regular grammar"
echo "enables deterministic parsing to rich ASTs."
echo ""
echo "All demos work WITHOUT any ML dependencies!"
echo "  - No FAISS"
echo "  - No PyTorch"
echo "  - Just pure deterministic parsing"
echo ""
echo "=========================================="
echo ""

# Function to run a demo with a header
run_demo() {
    local script=$1
    local description=$2

    echo ""
    echo "=========================================="
    echo "DEMO: $description"
    echo "=========================================="
    echo "Running: python $script"
    echo ""

    python "$script"

    echo ""
    echo ">>> Demo complete! <<<"
    echo ""
    read -p "Press Enter to continue to next demo..."
    echo ""
}

# Demo 1: Basic Parsing
run_demo "examples/basic_parsing.py" "Basic Parsing - Words and Sentences"

# Demo 2: Round-Trip
run_demo "examples/round_trip.py" "Round-Trip Conversion (Parse → Deparse)"

# Demo 3: Morpheme Analysis
run_demo "examples/morpheme_analysis.py" "Morpheme Analysis Deep Dive"

# Demo 4: Annotated AST (interactive)
echo ""
echo "=========================================="
echo "DEMO: Annotated AST Visualization"
echo "=========================================="
echo "Running: python examples/annotated_ast_demo.py"
echo ""
echo "This demo is INTERACTIVE - press Enter to advance"
echo "through complex sentence examples."
echo ""
read -p "Press Enter to start..."
echo ""

python examples/annotated_ast_demo.py

echo ""
echo "=========================================="
echo "  ALL DEMOS COMPLETE!"
echo "=========================================="
echo ""
echo "What you just saw:"
echo "  ✅ Deterministic morphological analysis"
echo "  ✅ Explicit grammatical feature extraction"
echo "  ✅ Subject/Verb/Object detection (0 params!)"
echo "  ✅ Adjective agreement checking"
echo "  ✅ Complex nested structure parsing"
echo "  ✅ Perfect round-trip consistency"
echo ""
echo "These ASTs are the INPUT to our neural models:"
echo "  → Traditional LLMs: ['The', 'dog', 'sees', 'the', 'cat']"
echo "  → Klareco: Structured AST with explicit grammar"
echo ""
echo "Next steps:"
echo "  - Train neural models ON these ASTs (not raw tokens)"
echo "  - Use pure Esperanto corpus (26K sentences)"
echo "  - Focus learned capacity on REASONING, not grammar"
echo ""
echo "For more info:"
echo "  - README.md - Project overview"
echo "  - POC_STATUS.md - What's implemented, what's next"
echo "  - CLAUDE.md - Development guide"
echo ""
