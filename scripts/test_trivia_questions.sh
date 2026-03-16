#!/bin/bash
#
# Test Trivia Questions - Run 8 Test Queries Through Extractive QA
#
# Tests both Esperanto and Benjamin Franklin questions to verify:
# - Different question types (WHAT, WHO, WHERE, WHEN)
# - Different entity types (language vs. person)
# - Different relation types (IS-A, CREATED-BY, BORN, PUBLISHED)
#
# Usage:
#   ./scripts/test_trivia_questions.sh                    # Run all 8 questions
#   ./scripts/test_trivia_questions.sh --save results.txt # Save output to file
#   ./scripts/test_trivia_questions.sh --expand           # Use embedding expansion

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Parse arguments
SAVE_FILE=""
EXPAND_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --save)
            SAVE_FILE="$2"
            shift 2
            ;;
        --expand)
            EXPAND_FLAG="--expand"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--save FILE] [--expand]"
            exit 1
            ;;
    esac
done

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

# Function to run a single question
run_question() {
    local num=$1
    local question=$2
    local translation=$3
    local qtype=$4

    echo ""
    echo "======================================================================"
    echo "QUESTION ${num}/8 [${qtype}]"
    echo "======================================================================"
    echo "Esperanto: ${question}"
    echo "English:   ${translation}"
    echo "----------------------------------------------------------------------"
    echo ""

    # Run the extractive QA system
    python scripts/demo_extractive_qa.py "${question}" ${EXPAND_FLAG} --max-facts 3

    echo ""
    echo "======================================================================"
    echo ""
}

# Main execution
main() {
    echo "######################################################################"
    echo "# TRIVIA QUESTIONS TEST SUITE"
    echo "######################################################################"
    echo ""
    echo "Testing extractive QA system with 8 trivia questions:"
    echo "  - 4 Esperanto questions (WHAT, WHO, WHERE, WHEN)"
    echo "  - 4 Benjamin Franklin questions (WHO, WHEN, WHERE, WHAT)"
    echo ""
    echo "Settings:"
    echo "  - Expansion: ${EXPAND_FLAG:-disabled}"
    echo "  - Max facts: 3"
    echo "  - Database: data/indexes/v2.1_kuzu_index_full"
    echo ""

    if [ -n "$SAVE_FILE" ]; then
        echo "Output will be saved to: $SAVE_FILE"
        echo ""
    fi

    echo "Press Enter to start..."
    read

    # Run all questions
    for i in "${!QUESTIONS[@]}"; do
        num=$((i + 1))
        run_question "$num" "${QUESTIONS[$i]}" "${TRANSLATIONS[$i]}" "${TYPES[$i]}"

        # Pause between questions (except after last)
        if [ $num -lt 8 ]; then
            echo "Press Enter for next question (or Ctrl+C to stop)..."
            read
        fi
    done

    echo ""
    echo "######################################################################"
    echo "# TEST COMPLETE"
    echo "######################################################################"
    echo ""
    echo "Summary:"
    echo "  - Questions tested: 8"
    echo "  - Esperanto topics: 4"
    echo "  - Franklin topics: 4"
    echo ""
    echo "Analysis suggestions:"
    echo "  - Which question types worked best? (WHAT, WHO, WHERE, WHEN)"
    echo "  - Did answers directly address the questions?"
    echo "  - Were facts ranked correctly?"
    echo "  - What limitations were observed?"
    echo ""
}

# Run main function, optionally saving to file
if [ -n "$SAVE_FILE" ]; then
    main 2>&1 | tee "$SAVE_FILE"
else
    main
fi
