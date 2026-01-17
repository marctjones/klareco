#!/bin/bash
# Validate M1 selectional preference model on practical examples
# Tests the specific bug we're fixing: object discrimination

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
    echo "❌ No venv found"
    exit 1
fi

# Check if model exists
MODEL_DIR="models/m1_selectional_v2"
if [ ! -f "$MODEL_DIR/best_model.pt" ]; then
    echo "❌ Model not found: $MODEL_DIR/best_model.pt"
    echo ""
    echo "Train model first:"
    echo "  ./scripts/m1_train_selectional.sh"
    exit 1
fi

echo "========================================================================"
echo "M1 SELECTIONAL PREFERENCE VALIDATION"
echo "========================================================================"
echo ""
echo "Testing the specific bug we're fixing:"
echo "  OLD: (hund, manĝ, viand) = 0.964, (hund, manĝ, ideo) = 0.937 (too close!)"
echo "  NEW: Should have >0.4 difference"
echo ""

# Create a simple validation script inline
python << 'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from klareco.models.m1_inference import M1Inference

print("Loading model...")
m1 = M1Inference(
    model_path=Path('models/m1_selectional_v2/best_model.pt'),
    stage1_path=Path('models/root_embeddings/best_model.pt')
)
print("✓ Model loaded\n")

# Test cases from issue #475
test_cases = [
    # Should be HIGH (plausible)
    ('hund', 'manĝ', 'viand', 'dog eats meat', True),
    ('homo', 'leg', 'libro', 'person reads book', True),

    # Should be LOW (implausible - selectional violations)
    ('hund', 'manĝ', 'ideo', 'dog eats idea', False),
    ('homo', 'leg', 'tablo', 'person reads table', False),
    ('tablo', 'pens', 'problemo', 'table thinks problem', False),
    ('koloro', 'aŭd', 'sono', 'color hears sound', False),
]

print("=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)
print()

plausible_scores = []
implausible_scores = []
all_pass = True

for subj, verb, obj, desc, should_be_plausible in test_cases:
    score = m1.score_triple(subj, verb, obj)

    if should_be_plausible:
        plausible_scores.append(score)
        passed = score >= 0.7
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}  {score:.3f}  ({subj}, {verb}, {obj})")
        print(f"       {desc} - should be ≥0.7")
    else:
        implausible_scores.append(score)
        passed = score <= 0.3
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}  {score:.3f}  ({subj}, {verb}, {obj})")
        print(f"       {desc} - should be ≤0.3")

    if not passed:
        all_pass = False
    print()

# Check separation
if plausible_scores and implausible_scores:
    min_plausible = min(plausible_scores)
    max_implausible = max(implausible_scores)
    separation = min_plausible - max_implausible

    print("=" * 70)
    print("SCORE SEPARATION")
    print("=" * 70)
    print(f"  Plausible (min):    {min_plausible:.3f}")
    print(f"  Implausible (max):  {max_implausible:.3f}")
    print(f"  Separation:         {separation:.3f} (should be >0.4)")
    print()

    if separation >= 0.4:
        print("✓ PASS: Adequate score separation")
    else:
        print("✗ FAIL: Insufficient score separation")
        all_pass = False

print()
print("=" * 70)
if all_pass:
    print("✓ ALL VALIDATION CHECKS PASSED")
    print("=" * 70)
    print()
    print("Model successfully discriminates selectional preferences!")
    print("The bug is FIXED.")
    sys.exit(0)
else:
    print("✗ SOME VALIDATION CHECKS FAILED")
    print("=" * 70)
    print()
    print("Model still has issues with selectional preference.")
    print("Consider:")
    print("  1. More training epochs")
    print("  2. More selectional constraints")
    print("  3. Adjusting loss weights")
    sys.exit(1)
EOF

EXIT_CODE=$?
echo ""
exit $EXIT_CODE
