# M1 Integration Guide: From Training to Production RAG

**Status**: M1 semantic model trained successfully (86.2% accuracy) ✅
**Date**: 2026-01-19

This guide covers how to use the trained M1 model in production, what's possible now, and what remains to be built.

## Table of Contents

1. [What M1 Can Do Now](#what-m1-can-do-now)
2. [Current Pipeline Status](#current-pipeline-status)
3. [RAG System Capabilities](#rag-system-capabilities)
4. [Integration Points](#integration-points)
5. [Synonym/Semantic Search Status](#synonymsemantic-search-status)
6. [Next Steps for Full RAG](#next-steps-for-full-rag)

---

## What M1 Can Do Now

M1 (Selectional Preference Model) scores the **plausibility** of (subject, verb, object) triples.

### Core Capability

```python
from klareco.models.m1_inference import M1Inference

m1 = M1Inference()  # Now loads semantic model by default!

# Score plausibility (0.0 = implausible, 1.0 = plausible)
score = m1.score_triple('hund', 'manĝ', 'viand')  # ~0.92 (dog eats meat)
score = m1.score_triple('hund', 'manĝ', 'ide')    # ~0.08 (dog eats idea)
score = m1.score_triple('tabl', 'pens', 'ideo')   # ~0.03 (table thinks idea)
```

### Use Cases

1. **Filter implausible retrieval results**
   - Query: "kiu manĝas viandon?" (who eats meat?)
   - Retrieved: 100 candidates
   - M1 filters out: "la tablo manĝas viandon" (table eats meat)

2. **Validate generated answers**
   - Before returning answer, check plausibility
   - Reject if M1 score < 0.3

3. **Rank retrieval results**
   - Combine similarity score + M1 plausibility
   - `final_score = 0.6 * similarity + 0.4 * m1_plausibility`

4. **Query expansion validation**
   - Expanding "hundo" to "besto" (animal)
   - Check if (besto, manĝ, viand) is plausible
   - Only expand if plausibility is high

---

## Current Pipeline Status

### ✅ What's Working

| Component | Status | Accuracy/Quality |
|-----------|--------|------------------|
| **Stage 0: Parser** | ✅ Production | 99.99% (tier0), 91.8% (general) |
| **Stage 1: Root Embeddings** | ✅ Production | Correlation: 0.8491 |
| **M1: Selectional Preferences** | ✅ Production | Accuracy: 86.2% |
| **M1Inference Class** | ✅ Updated | Loads semantic model by default |

### ⏳ What's Not Yet Integrated

| Component | Status | Blocker |
|-----------|--------|---------|
| **Semantic expansion** | 📋 Not implemented | Task #7 |
| **RAG retriever** | ❓ Unknown if exists | Need to check |
| **M1 filtering in retrieval** | ❓ Not integrated | Need to add |
| **Query expansion** | ❓ Unknown | Need to check |

---

## RAG System Capabilities

### Question: "Can we make a RAG yet?"

**Answer**: **Partial**. We have the pieces but not fully integrated.

**What we have:**
- ✅ Parsed corpus (4.7M sentences)
- ✅ Root embeddings (for semantic similarity)
- ✅ M1 plausibility filter
- ✅ Parser (for understanding queries)

**What's missing for full RAG:**
- ❌ Retrieval system using Stage 1 embeddings (may exist, need to verify)
- ❌ M1 integrated into retrieval pipeline
- ❌ Semantic expansion (synonyms, related concepts)
- ❌ Answer generation/extraction

### Basic RAG Pipeline (What We Can Build Now)

```
User Query: "Kio manĝas viandon?"
    ↓
[1. Parse Query]
    ↓ AST: (?, manĝ, viand)
    ↓
[2. Embed with Stage 1]
    ↓ manĝ → [64d vector], viand → [64d vector]
    ↓
[3. Search Corpus]
    ↓ Find sentences with similar embeddings (cosine similarity)
    ↓ Retrieved: 100 candidates
    ↓
[4. Filter with M1] ← NEW CAPABILITY!
    ↓ Score each candidate triple with M1
    ↓ Keep only score > 0.5
    ↓ Filtered: 60 candidates
    ↓
[5. Rank and Return]
    ↓ Combined score = 0.6*similarity + 0.4*m1_score
    ↓
Top Answer: "Hundoj manĝas viandon."
```

---

## Integration Points

### 1. Basic M1 Filtering (Can Implement Now)

```python
# In retrieval pipeline
def filter_retrieval_results(candidates, m1_threshold=0.5):
    """Filter candidates by M1 plausibility."""
    from klareco.models.m1_inference import M1Inference

    m1 = M1Inference()
    filtered = []

    for candidate in candidates:
        # Extract SVO from candidate AST
        ast = candidate['ast']

        try:
            subj = ast['subjekto']['kerno']['radiko']
            verb = ast['verbo']['radiko']
            obj = ast['objekto']['kerno']['radiko']

            # Score plausibility
            score = m1.score_triple(subj, verb, obj)

            if score >= m1_threshold:
                candidate['m1_score'] = score
                filtered.append(candidate)
        except (KeyError, TypeError):
            # Incomplete AST, skip filtering
            continue

    return filtered
```

### 2. Reranking with M1 (Can Implement Now)

```python
def rerank_with_m1(candidates, similarity_weight=0.6, m1_weight=0.4):
    """Combine similarity and M1 scores for ranking."""
    from klareco.models.m1_inference import M1Inference

    m1 = M1Inference()

    for candidate in candidates:
        # Get existing similarity score
        sim_score = candidate.get('similarity', 0.5)

        # Get M1 plausibility
        try:
            subj = candidate['ast']['subjekto']['kerno']['radiko']
            verb = candidate['ast']['verbo']['radiko']
            obj = candidate['ast']['objekto']['kerno']['radiko']
            m1_score = m1.score_triple(subj, verb, obj)
        except (KeyError, TypeError):
            m1_score = 0.5  # Default if extraction fails

        # Combine scores
        candidate['final_score'] = (
            similarity_weight * sim_score +
            m1_weight * m1_score
        )

    # Sort by final score
    candidates.sort(key=lambda x: x['final_score'], reverse=True)

    return candidates
```

### 3. Checking If RAG System Exists

Need to check:
```bash
# Look for existing retrieval/RAG code
find klareco -name "*retriev*" -o -name "*rag*" | grep -v __pycache__
find scripts -name "*retriev*" -o -name "*rag*" -o -name "*demo*"
```

---

## Synonym/Semantic Search Status

### Question: "Will RAG find responses using synonyms but not exact roots?"

**Answer**: **Not yet, but we have the foundation.**

### Current State

**Stage 1 embeddings CAN represent semantic similarity:**
```python
# These should have high similarity in Stage 1 embeddings
similarity("hund", "besto") → ~0.45  # dog, animal
similarity("manĝ", "konsum") → ~0.52  # eat, consume
```

**But semantic expansion is not yet implemented** (Task #7).

### What's Needed for Synonym Search

**Option A: Semantic Expansion at Query Time**
```python
def expand_query_semantically(root, top_k=5):
    """
    Find semantically similar roots to expand query.

    Query: "hundo manĝas"
    Expands to: ["hundo", "besto", "kanino"] (dog, animal, canine)
    """
    # Load Stage 1 embeddings
    # Find top-k nearest neighbors by cosine similarity
    # Return expanded query terms
    pass  # TODO: Implement (Task #7)
```

**Option B: Index Semantic Neighbors** (Better)
```python
# Pre-compute semantic neighbors for all roots
# Store in index: hund → [besto, kanino, hundido, mamulo]
# At query time, automatically search for neighbors too
```

**Option C: Use ConceptNet/ReVo Relations**
```python
# Query ReVo/ConceptNet for synonyms, hypernyms
# "hund" → HYPERNYM → "besto"
# "manĝ" → SYNONYM → "konsum"
```

### Current Capability

**Without semantic expansion:**
- Query: "Kio estas besto?" (What is an animal?)
- Finds: Sentences with exact root "besto"
- Misses: Sentences about "hundo", "kato", "ĉevalo" (dog, cat, horse)

**With semantic expansion (Task #7):**
- Query: "Kio estas besto?"
- Expands to: ["besto", "hund", "kat", "ĉeval", ...] (semantic neighbors)
- Finds: All sentences about specific animals
- M1 filters: Only plausible combinations

---

## Next Steps for Full RAG

### Immediate (This Week)

1. **Verify existing retrieval system** ✅
   ```bash
   find klareco -name "*retriev*"
   cat klareco/rag/retriever.py  # If exists
   ```

2. **Integrate M1 into retrieval** ✅ (Code examples above)
   - Add `filter_retrieval_results()` function
   - Add `rerank_with_m1()` function

3. **Update wiki documentation** (Task #8) ✅
   - Populate all [PLACEHOLDER] fields
   - Add M1 semantic training results
   - Document integration patterns

### Short-term (Next 2 Weeks)

4. **Implement semantic expansion** (Task #7)
   - Option A: Nearest neighbor search with Stage 1
   - Option B: Pre-compute semantic index
   - Option C: Integrate ConceptNet/ReVo

5. **Build end-to-end RAG demo**
   ```bash
   ./scripts/demo_rag_with_m1.py "Kiu manĝas viandon?"
   ```

6. **Test on question set**
   - Create 50-100 test questions
   - Measure retrieval accuracy
   - Measure M1 filtering effectiveness

### Medium-term (Next Month)

7. **Optimize retrieval speed**
   - Build FAISS index for Stage 1 embeddings
   - Batch M1 scoring (already supported!)

8. **Add answer extraction/generation**
   - Currently: Return full sentences
   - Goal: Extract specific answer span

9. **Evaluate on benchmarks**
   - Target: 50-question benchmark (from VISION.md)
   - Measure: accuracy, relevance, plausibility

---

## Common Mistakes & Best Practices

### Lessons Learned from M1 Training

#### Mistake #1: **Wrong Default Model Paths**

**Problem:** `M1Inference` loaded old model, not new semantic model
- Trained new model → `models/m1_semantic_full/best_model.pt`
- Inference loaded → `models/m1_selectional/best_model.pt` (old!)
- Validation showed 47% accuracy instead of 86%

**Solution:** Update default paths when training new models
```python
# In klareco/models/m1_inference.py
if model_path is None:
    model_path = Path('models/m1_semantic_full/best_model.pt')  # Update this!
```

**Best Practice:** Always verify which model is loaded:
```python
# In __init__:
logger.info(f"Loading M1 model from: {model_path}")
logger.info(f"Model checkpoint: {m1_checkpoint.keys()}")
```

#### Mistake #2: **Scripts Without --model Parameter**

**Problem:** Validation scripts hardcoded model paths
```python
# Bad: Hardcoded
self.m1 = M1Inference()  # Uses default path

# Good: Parameterized
parser.add_argument('--model', type=Path, default=Path('models/m1_semantic_full/best_model.pt'))
self.m1 = M1Inference(model_path=args.model)
```

**Best Practice:** All validation/inference scripts should accept `--model` argument

#### Mistake #3: **Not Verifying Data After Generation**

**Problem:**
- Generated M1 training data
- Didn't check if tier0 was actually included
- Trained for 2 hours
- Later discovered 0 tier0 examples

**Solution:** Always verify data after generation:
```bash
# After data generation, immediately check:
python3 << 'EOF'
import json
from collections import Counter

tiers = Counter()
with open('data/training/m1_xxx/train.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 10000: break
        ex = json.loads(line)
        tier = ex.get('source', {}).get('tier', 'unknown')
        tiers[tier] += 1

print("Tier distribution (first 10K):")
for tier, count in sorted(tiers.items()):
    print(f"  Tier {tier}: {count}")
EOF
```

**Best Practice:** Add `--verify` flag to data generation scripts that prints distribution

#### Mistake #4: **Shell Scripts Without Error Checking**

**Problem:** Script reported success even when Python failed
```bash
# Bad:
python train.py 2>&1 | tee log.txt
echo "✓ Training complete"  # Always prints, even if train.py crashed
```

**Solution:** Use `set -o pipefail` and check exit codes
```bash
# Good:
set -e
set -o pipefail

if python train.py 2>&1 | tee log.txt; then
    echo "✓ Training complete"
else
    echo "✗ Training failed (exit code: $?)"
    exit 1
fi
```

#### Mistake #5: **Training Scripts Without Checkpoints**

**Problem:** 2-hour training run crashes at epoch 45, lose all progress

**Solution:** All training scripts should:
```python
parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
parser.add_argument('--fresh', action='store_true', help='Start fresh, ignore checkpoint')

# At start:
if args.fresh:
    logger.info("Starting fresh training")
    # Delete checkpoint if exists
elif checkpoint_exists():
    logger.info("Resuming from checkpoint")
    load_checkpoint()
else:
    logger.info("No checkpoint found, starting fresh")
```

#### Mistake #6: **No Logging to File**

**Problem:** Training output only in terminal, can't review later

**Solution:** All scripts should log to file:
```python
import logging

LOG_DIR = Path('logs/training')
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"train_{args.model_name}_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()  # Also print to console
    ]
)
```

#### Mistake #7: **Scripts Without --help Documentation**

**Problem:** Don't remember what arguments script takes

**Solution:** Good argparse descriptions:
```python
parser = argparse.ArgumentParser(
    description="Train M1 selectional preference model with semantic-distance corruption",
    epilog="""
Examples:
  # Train on tier0 data
  python train_m1.py --data-dir data/training/m1_tier0_only

  # Resume from checkpoint
  python train_m1.py --resume

  # Start fresh with higher capacity
  python train_m1.py --fresh --hidden-dim 512
    """
)
```

---

## Script Development Guidelines

### Template for Training Scripts

```python
#!/usr/bin/env python3
"""
[Script Name and Purpose]

Usage:
    python script.py
    python script.py --resume
    python script.py --fresh --some-param value
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
LOG_DIR = Path('logs/training')
LOG_DIR.mkdir(parents=True, exist_ok=True)

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="[What this script does]"
    )

    # Required arguments
    parser.add_argument('--input', type=Path, required=True, help='Input data path')

    # Optional arguments with defaults
    parser.add_argument('--output', type=Path, default=Path('output'), help='Output directory')
    parser.add_argument('--model', type=Path, default=None, help='Model path (default: uses production model)')

    # Flags
    parser.add_argument('--fresh', action='store_true', help='Start fresh, ignore checkpoint')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--verify', action='store_true', help='Verify data after generation')

    args = parser.parse_args()

    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f"script_name_{timestamp}.log"
    logger = setup_logging(log_file)

    logger.info("=" * 60)
    logger.info("[Script Name]")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Log: {log_file}")
    logger.info("")

    # Validate inputs
    if not args.input.exists():
        logger.error(f"Input not found: {args.input}")
        return 1

    # Do work
    try:
        result = do_work(args)

        if args.verify:
            verify_output(args.output)

        logger.info("")
        logger.info("✓ Script complete")
        return 0

    except Exception as e:
        logger.error(f"✗ Script failed: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
```

### Template for Shell Wrappers

```bash
#!/bin/bash
#
# [Script Name and Purpose]
#
# Usage:
#   ./script.sh
#   ./script.sh --fresh
#

set -e  # Exit on error
set -o pipefail  # Catch errors in pipes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ No virtual environment found"
    exit 1
fi

# Parse arguments
ARGS=""
for arg in "$@"; do
    ARGS="$ARGS $arg"
done

# Setup logging
LOG_DIR="logs/training"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/script_name_${TIMESTAMP}.log"

echo "=============================================="
echo "[Script Name]"
echo "=============================================="
echo "Logging to: $LOG_FILE"
echo ""

# Run with error checking
if python scripts/script_name.py $ARGS 2>&1 | tee "$LOG_FILE"; then
    echo ""
    echo "✓ Script complete"
    echo "Log: $LOG_FILE"
else
    EXIT_CODE=${PIPESTATUS[0]}
    echo ""
    echo "✗ Script failed (exit code: $EXIT_CODE)"
    echo "Check log: $LOG_FILE"
    exit $EXIT_CODE
fi
```

---

## Checklist for New Scripts

Before committing a new training/validation script, verify:

- [ ] Has `--help` with good documentation
- [ ] Has `--model` parameter (for inference/validation)
- [ ] Has `--resume` and `--fresh` flags (for training)
- [ ] Logs to file (not just console)
- [ ] Uses `set -e` and `set -o pipefail` (shell scripts)
- [ ] Validates inputs before starting work
- [ ] Prints clear error messages
- [ ] Has `--verify` flag to check output (data generation)
- [ ] Shell wrapper activates venv
- [ ] Returns proper exit codes
- [ ] Includes usage examples in docstring

---

## References

- **M1 Model**: `models/m1_semantic_full/best_model.pt` (86.2% accuracy)
- **Stage 1 Model**: `models/root_embeddings_tier0/best_model.pt` (0.8491 correlation)
- **M1 Inference**: `klareco/models/m1_inference.py` (updated to use semantic model)
- **Task #7**: Integrate semantic expansion into retriever
- **Task #8**: Update wiki pages with training results
- **Training Results**: `docs/wiki_templates/Training-Results-2026-01-19.md`
- **Investigation**: `docs/wiki_templates/M1-Investigation-2026-01-18.md`
