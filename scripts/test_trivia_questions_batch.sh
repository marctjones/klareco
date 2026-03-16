#!/bin/bash
#
# Test Trivia Questions (Batch Mode) - Run All 8 Questions Non-Interactively
#
# Usage:
#   ./scripts/test_trivia_questions_batch.sh                    # Run all 8
#   ./scripts/test_trivia_questions_batch.sh > results.txt      # Save to file
#   ./scripts/test_trivia_questions_batch.sh --expand           # Use expansion

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Parse arguments
EXPAND_FLAG=""
if [ "$1" = "--expand" ]; then
    EXPAND_FLAG="--expand"
fi

# Define the 8 trivia questions
declare -a QUESTIONS=(
    "Kiu kreis Esperanton?"
    "Kiam estis publikigita la unua libro pri Esperanto?"
    "Kie okazis la unua Tutmonda Esperanto-Kongreso?"
    "Kio estas la Fundamento de Esperanto?"
    "Kiu estis Benjamin Franklin?"
    "Kiam naskiĝis Benjamin Franklin?"
    "Kie naskiĝis Benjamin Franklin?"
    "Kion inventis Benjamin Franklin?"
)

declare -a TRANSLATIONS=(
    "Who created Esperanto?"
    "When was the first book about Esperanto published?"
    "Where was the first World Esperanto Congress held?"
    "What is the Fundamento of Esperanto?"
    "Who was Benjamin Franklin?"
    "When was Benjamin Franklin born?"
    "Where was Benjamin Franklin born?"
    "What did Benjamin Franklin invent?"
)

declare -a TYPES=(
    "WHO"
    "WHEN"
    "WHERE"
    "WHAT"
    "WHO"
    "WHEN"
    "WHERE"
    "WHAT"
)

echo "######################################################################"
echo "# TRIVIA QUESTIONS TEST SUITE (BATCH MODE)"
echo "######################################################################"
echo ""
echo "Testing 8 questions:"
echo "  - 4 Esperanto (WHAT, WHO, WHERE, WHEN)"
echo "  - 4 Benjamin Franklin (WHO, WHEN, WHERE, WHAT)"
echo ""
echo "Settings: ${EXPAND_FLAG:-no expansion}, max-facts=3"
echo ""
echo "######################################################################"
echo ""

# Run all questions
for i in "${!QUESTIONS[@]}"; do
    num=$((i + 1))

    echo ""
    echo "======================================================================"
    echo "QUESTION ${num}/8 [${TYPES[$i]}]"
    echo "======================================================================"
    echo "Esperanto: ${QUESTIONS[$i]}"
    echo "English:   ${TRANSLATIONS[$i]}"
    echo "----------------------------------------------------------------------"
    echo ""

    # Run extractive QA (suppress stderr to reduce noise)
    python scripts/demo_extractive_qa.py "${QUESTIONS[$i]}" ${EXPAND_FLAG} --max-facts 3 2>/dev/null || {
        echo "ERROR: Query failed"
        echo ""
    }

    echo ""
    echo "======================================================================"
    echo ""
done

echo ""
echo "######################################################################"
echo "# TEST COMPLETE - 8/8 QUESTIONS"
echo "######################################################################"
echo ""
