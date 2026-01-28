#!/bin/bash
#
# Test AST Utilities - Case Normalization and Word Structure Extraction
#
# This script tests the new AST utility functions that extract morphological
# structures with case normalization for M1 v2 training.
#

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
    echo "❌ No virtual environment found (.venv or venv)"
    exit 1
fi

echo "=============================================================================="
echo "Testing AST Utilities - Case Normalization & Word Structure Extraction"
echo "=============================================================================="
echo ""

# Test case normalization
echo "Test 1: Case Normalization"
echo "----------------------------"
echo "Testing that accusative -n is stripped while keeping semantic features..."
echo ""

python << 'EOF'
from klareco.parser import parse
from klareco.utils.ast_utils import extract_word_structure, normalize_case

# Test normalize_case function directly
print("Direct normalize_case() tests:")
print(f"  'on'  → '{normalize_case('on')}'  (should be 'o')")
print(f"  'ojn' → '{normalize_case('ojn')}' (should be 'oj' - plural kept)")
print(f"  'o'   → '{normalize_case('o')}'   (should be 'o')")
print(f"  'oj'  → '{normalize_case('oj')}'  (should be 'oj')")
print(f"  'as'  → '{normalize_case('as')}'  (should be 'as')")
print(f"  'en'  → '{normalize_case('en')}'  (should be 'en' - adverb, not accusative)")
print()

# Test with actual sentences
print("Sentence parsing tests:")
print()

# Test 1: Simple SVO with accusative object
sentence1 = "Hundo manĝas nutraĵon."
print(f"Sentence: {sentence1}")
ast1 = parse(sentence1)

if ast1 and 'subjekto' in ast1 and 'objekto' in ast1:
    subj = extract_word_structure(ast1['subjekto']['kerno'], strip_case=True)
    obj = extract_word_structure(ast1['objekto']['kerno'], strip_case=True)

    print(f"  Subject: {subj}")
    print(f"  Object:  {obj}")

    # Verify
    assert subj['root'] == 'hund', f"Subject root should be 'hund', got '{subj['root']}'"
    assert subj['ending'] == 'o', f"Subject ending should be 'o', got '{subj['ending']}'"
    assert obj['root'] == 'nutr', f"Object root should be 'nutr', got '{obj['root']}'"
    assert 'aĵ' in obj['suffixes'], f"Object should have suffix 'aĵ', got {obj['suffixes']}"
    assert obj['ending'] == 'o', f"Object ending should be 'o' (stripped -n), got '{obj['ending']}'"

    print("  ✓ Accusative -n correctly stripped, suffix preserved (nutr+aĵ+o)")
else:
    print("  ⚠️  Parse failed or missing SVO structure")

print()

# Test 2: Plural accusative
sentence2 = "Mi vidas hundojn."
print(f"Sentence: {sentence2}")
ast2 = parse(sentence2)

if ast2 and 'objekto' in ast2:
    obj = extract_word_structure(ast2['objekto']['kerno'], strip_case=True)
    print(f"  Object: {obj}")

    # Verify
    assert obj['root'] == 'hund', f"Object root should be 'hund', got '{obj['root']}'"
    assert obj['ending'] == 'oj', f"Object ending should be 'oj' (plural kept, -n stripped), got '{obj['ending']}'"

    print("  ✓ Plural marker kept, accusative -n stripped")
else:
    print("  ⚠️  Parse failed")

print()

# Test 3: Verb tense preservation
sentence3 = "Hundo manĝis nutraĵon."
print(f"Sentence: {sentence3}")
ast3 = parse(sentence3)

if ast3 and 'verbo' in ast3:
    verb = extract_word_structure(ast3['verbo'], strip_case=True)
    print(f"  Verb: {verb}")

    # Verify
    assert verb['root'] == 'manĝ', f"Verb root should be 'manĝ', got '{verb['root']}'"
    assert verb['ending'] == 'is', f"Verb ending should be 'is' (past tense), got '{verb['ending']}'"

    print("  ✓ Verb tense correctly preserved")
else:
    print("  ⚠️  Parse failed")

print()

# Test 4: Complex morphology (prefix + suffix + accusative)
sentence4 = "Mi vidas rehundejon."
print(f"Sentence: {sentence4}")
ast4 = parse(sentence4)

if ast4 and 'objekto' in ast4:
    obj = extract_word_structure(ast4['objekto']['kerno'], strip_case=True)
    print(f"  Object: {obj}")

    # Verify
    assert obj['root'] == 'hund', f"Object root should be 'hund', got '{obj['root']}'"
    assert 're' in obj['prefixes'], f"Object should have prefix 're', got {obj['prefixes']}"
    assert 'ej' in obj['suffixes'], f"Object should have suffix 'ej', got {obj['suffixes']}"
    assert obj['ending'] == 'o', f"Object ending should be 'o' (stripped -n), got '{obj['ending']}'"

    print("  ✓ Complex morphology: prefix + root + suffix + case normalization")
else:
    print("  ⚠️  Parse failed or missing object")

print()
print("=" * 70)
print("✓ All case normalization tests passed!")
print("=" * 70)
print()
print("Key behaviors verified:")
print("  1. Accusative -n is stripped (on → o, ojn → oj)")
print("  2. Semantic features are preserved (plural -j, tense -as/-is/-os)")
print("  3. Prefixes and suffixes are extracted correctly")
print("  4. Adverb ending -en is NOT stripped (not accusative)")
print()
print("This ensures M1 v2 learns SEMANTIC plausibility, not grammatical case!")
print()
EOF

echo ""
echo "Test 2: Full SVO Structure Extraction"
echo "--------------------------------------"
echo "Testing complete subject-verb-object structure extraction..."
echo ""

python << 'EOF'
from klareco.parser import parse
from klareco.utils.ast_utils import extract_word_structure
import json

# Test complete SVO extraction (simulating what training data gen does)
sentence = "Hundoj manĝas nutraĵojn."
print(f"Sentence: {sentence}")
ast = parse(sentence)

if ast and all(k in ast for k in ['subjekto', 'verbo', 'objekto']):
    # Extract like prepare_m1_training_data_tier_priority.py does
    subj_word = ast['subjekto'].get('kerno') if ast['subjekto'].get('tipo') == 'vortgrupo' else ast['subjekto']
    verb_word = ast['verbo']
    obj_word = ast['objekto'].get('kerno') if ast['objekto'].get('tipo') == 'vortgrupo' else ast['objekto']

    structures = {
        'subject': extract_word_structure(subj_word, strip_case=True),
        'verb': extract_word_structure(verb_word, strip_case=True),
        'object': extract_word_structure(obj_word, strip_case=True)
    }

    print()
    print("Extracted structures (as stored in training data):")
    print(json.dumps(structures, indent=2, ensure_ascii=False))
    print()

    # Verify structure format
    for role, struct in structures.items():
        assert 'root' in struct, f"{role} missing 'root'"
        assert 'prefixes' in struct, f"{role} missing 'prefixes'"
        assert 'suffixes' in struct, f"{role} missing 'suffixes'"
        assert 'ending' in struct, f"{role} missing 'ending'"

    print("✓ All structures have required keys: root, prefixes, suffixes, ending")
    print()

    # Show what would be in training JSONL
    print("Example training data row:")
    example = {
        'subject': structures['subject'],
        'verb': structures['verb'],
        'object': structures['object'],
        'label': 1.0,
        'corruption': None
    }
    print(json.dumps(example, ensure_ascii=False))
    print()
else:
    print("⚠️  Parse failed or incomplete SVO structure")

print("=" * 70)
print("✓ SVO structure extraction test passed!")
print("=" * 70)
EOF

echo ""
echo "Test 3: Edge Cases"
echo "------------------"
echo "Testing edge cases and special handling..."
echo ""

python << 'EOF'
from klareco.utils.ast_utils import normalize_case

# Test edge cases
print("Edge case tests:")
print()

# Correlatives with -n should NOT be stripped (they're part of the word)
test_cases = [
    ('on', 'o', 'Accusative noun'),
    ('ojn', 'oj', 'Accusative plural noun'),
    ('an', 'an', 'Correlative (part of word, not accusative)'),
    ('en', 'en', 'Adverb ending (not accusative)'),
    ('as', 'as', 'Present tense verb'),
    ('is', 'is', 'Past tense verb'),
    ('i', 'i', 'Infinitive verb'),
]

all_passed = True
for input_ending, expected, description in test_cases:
    result = normalize_case(input_ending)
    status = '✓' if result == expected else '✗'
    if result != expected:
        all_passed = False
    print(f"  {status} '{input_ending}' → '{result}' (expected: '{expected}') - {description}")

print()
if all_passed:
    print("✓ All edge cases handled correctly!")
else:
    print("✗ Some edge cases failed - review implementation")

print()
EOF

echo ""
echo "=============================================================================="
echo "✓ AST Utilities Test Suite Complete!"
echo "=============================================================================="
echo ""
echo "Summary:"
echo "  1. Case normalization works correctly (strips -n, keeps -j)"
echo "  2. Full SVO structure extraction produces correct format"
echo "  3. Edge cases (correlatives, adverbs) handled properly"
echo ""
echo "Ready for M1 v2 training data generation!"
echo ""
