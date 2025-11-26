# Phase 3 Evaluation Guide

**Date:** 2025-11-11
**Status:** Training in Progress
**Next Step:** Evaluate Tree-LSTM vs Baseline RAG

---

## Overview

This document describes how to evaluate the trained Tree-LSTM encoder against the baseline sentence-transformers RAG system.

---

## Current Training Status

**Model:** Tree-LSTM (Child-Sum architecture)
**Training Data:** 5,495 pairs (495 positive + 5,000 negative)
**Current Progress:** Epoch 1/10, Batch 177/344 (51%)
**Current Metrics:**
- Loss: ~0.05-0.10
- Accuracy: ~89.7%

**Estimated Completion:** ~25-30 minutes from training start

---

## After Training Completes

Once training finishes, you'll find the trained model at:

```
models/tree_lstm/
├── best_model.pt          # Best model checkpoint (lowest loss)
├── final_model.pt         # Final model after all epochs
├── epoch_*.pt             # Per-epoch checkpoints
└── training_history.json  # Training metrics history
```

---

## Running Evaluation

### Quick Start

```bash
python scripts/evaluate_embeddings.py \
    --tree-lstm models/tree_lstm/best_model.pt \
    --baseline data/faiss_baseline \
    --corpus data/ast_corpus \
    --output evaluation_results \
    --num-queries 100
```

This will:
1. Load the trained Tree-LSTM model
2. Load the baseline RAG system (sentence-transformers)
3. Generate 100 test queries from the corpus
4. Encode queries with both systems
5. Measure retrieval performance
6. Generate comparison report

---

## Evaluation Metrics

### Precision@K

**Definition:** Of the top K retrieved documents, what fraction are relevant?

**Formula:** `Precision@K = (# relevant in top K) / K`

**Interpretation:**
- Higher is better
- Precision@5 = 0.80 means 4 out of 5 retrieved docs are relevant

### Recall@K

**Definition:** Of all relevant documents, what fraction are in the top K?

**Formula:** `Recall@K = (# relevant in top K) / (total relevant)`

**Interpretation:**
- Higher is better
- Recall@5 = 0.60 means top 5 captures 60% of all relevant docs

### Mean Reciprocal Rank (MRR)

**Definition:** Average of reciprocal ranks of first relevant document.

**Formula:** `MRR = average(1 / rank_of_first_relevant)`

**Interpretation:**
- Range: [0, 1]
- MRR = 1.0 means first result is always relevant
- MRR = 0.5 means first result is on average rank 2

---

## Relevance Definition

For this evaluation, a document is considered **relevant** to a query if:

**Condition:** `|doc_index - query_index| <= threshold`

**Rationale:** Documents that appear close together in the corpus likely share similar context, topics, or narrative structure.

**Default Threshold:** 10 documents

**Example:**
- Query: Document #50
- Relevant documents: #40 through #60 (21 docs total)

---

## Expected Results

### Baseline (Sentence-Transformers)

**Strengths:**
- Captures semantic similarity well
- Strong at lexical overlap
- Fast encoding (~1ms per sentence)

**Weaknesses:**
- Ignores syntactic structure
- AST information discarded during deparsing
- Purely text-based embeddings

### Tree-LSTM (Structural)

**Strengths:**
- Preserves AST structure during encoding
- Captures morphological relationships
- Bottom-up compositional semantics

**Weaknesses:**
- Slower encoding (~10-50ms per sentence)
- More complex model (1.7M parameters)
- Requires training data

---

## Success Criteria

### Minimum Viable (MVP)

- ✅ Training completes without errors
- ✅ Evaluation runs successfully
- ✅ GNN shows improvement on **at least one metric**

### Strong Success

- 🎯 Tree-LSTM beats baseline on Precision@5 by **5%+**
- 🎯 Tree-LSTM beats baseline on MRR by **10%+**
- 🎯 Structural similarity correlates with semantic similarity

### Outstanding Success

- 🌟 Tree-LSTM beats baseline on **all metrics**
- 🌟 Improvement of **15%+ on Precision@5**
- 🌟 Encoding latency < 100ms per sentence

---

## Interpreting Results

### Scenario 1: Tree-LSTM Wins on All Metrics

**Interpretation:** Structure matters! AST-based encoding captures meaningful semantic relationships.

**Next Steps:**
- Scale up training data (50K+ pairs)
- Integrate Tree-LSTM into RAG pipeline
- Test on real-world queries

### Scenario 2: Tree-LSTM Wins on Some Metrics

**Interpretation:** Structure helps in certain cases (e.g., complex syntax).

**Next Steps:**
- Analyze which types of queries benefit from structure
- Consider hybrid approach (combine both embeddings)
- Investigate complementary strengths

### Scenario 3: Baseline Wins on All Metrics

**Interpretation:** For this corpus, text-based embeddings suffice.

**Next Steps:**
- Ship baseline RAG (simpler is better)
- Investigate why structure didn't help:
  - Training data quality?
  - Model architecture?
  - Corpus characteristics?
- Consider alternative GNN architectures (GAT, GCN)

---

## Customizing Evaluation

### More Test Queries

```bash
python scripts/evaluate_embeddings.py \
    --num-queries 500 \
    --max-corpus-asts 50000 \
    ...
```

### Different K Values

```bash
python scripts/evaluate_embeddings.py \
    --k-values 1 3 5 10 20 \
    ...
```

### Adjust Relevance Threshold

```bash
python scripts/evaluate_embeddings.py \
    --threshold 20 \
    ...
```

More lenient relevance (documents within ±20 positions are relevant).

### GPU Acceleration

```bash
python scripts/evaluate_embeddings.py \
    --device cuda \
    ...
```

Speeds up Tree-LSTM encoding significantly.

---

## Output Files

After evaluation completes, you'll find:

```
evaluation_results/
├── evaluation_report.txt     # Human-readable comparison
├── evaluation_results.json   # Raw metrics (JSON)
└── training_curves.png       # (future) Visualization
```

### Sample Report

```
================================================================================
TREE-LSTM VS BASELINE RAG EVALUATION
================================================================================

## Precision@K

Precision@5:
  Baseline:   0.6400
  Tree-LSTM:  0.7200
  Improvement: +12.50%

## Recall@K

Recall@5:
  Baseline:   0.3048
  Tree-LSTM:  0.3429
  Improvement: +12.50%

## Mean Reciprocal Rank (MRR)

MRR:
  Baseline:   0.7245
  Tree-LSTM:  0.8021
  Improvement: +10.71%

================================================================================
```

---

## Troubleshooting

### Issue: "Model file not found"

**Cause:** Training hasn't completed or failed.

**Fix:** Check `tree_lstm_training.log` for errors. Wait for training to complete.

### Issue: "FAISS index dimension mismatch"

**Cause:** Baseline index has different embedding dimension than Tree-LSTM output.

**Fix:** Ensure Tree-LSTM `output_dim=512` matches baseline embedding size.

### Issue: "Out of memory during encoding"

**Cause:** Too many ASTs being encoded at once.

**Fix:** Reduce `--max-corpus-asts` or use GPU with `--device cuda`.

### Issue: "Evaluation is too slow"

**Cause:** Tree-LSTM encoding on CPU is slow.

**Fix:**
- Use smaller `--num-queries` (e.g., 50)
- Reduce `--max-corpus-asts` (e.g., 5000)
- Use GPU with `--device cuda`

---

## Next Steps After Evaluation

### If Tree-LSTM Wins

1. **Scale Training:**
   ```bash
   python scripts/prepare_training_data.py \
       --corpus data/ast_corpus \
       --output data/training_pairs_large \
       --num-pairs 25000 \
       --max-asts 200000
   ```

2. **Retrain on Larger Dataset:**
   ```bash
   python scripts/train_tree_lstm.py \
       --training-data data/training_pairs_large \
       --output models/tree_lstm_v2 \
       --epochs 20 \
       --batch-size 32
   ```

3. **Build Full RAG System:**
   - Index all 1.27M sentences with Tree-LSTM
   - Integrate with query interface
   - Test on real-world use cases

### If Baseline Wins

1. **Ship Baseline RAG:**
   ```bash
   python scripts/build_baseline_rag.py \
       --corpus data/ast_corpus \
       --output data/faiss_baseline_full \
       --max-sentences 0  # No limit, index all
   ```

2. **Document Findings:**
   - Why didn't structure help?
   - What types of queries were tested?
   - Corpus characteristics analysis

3. **Consider Alternatives:**
   - Hybrid approach (baseline + GNN for complex queries)
   - Different GNN architecture (GAT, GCN)
   - Alternative training strategy (triplet loss, etc.)

---

## Key Takeaways

1. **Dual-Track Strategy Works:** We built both systems, so we ship what works.
2. **Empirical Validation:** No guessing - let metrics decide.
3. **Structure Hypothesis:** This evaluation tests if AST structure improves retrieval.
4. **Incremental Progress:** Start with PoC (100 queries), scale if promising.

---

## References

- **Tree-LSTM Paper:** Tai et al. (2015) - "Improved Semantic Representations From Tree-Structured LSTM Networks"
- **Contrastive Learning:** Chen et al. (2020) - "A Simple Framework for Contrastive Learning of Visual Representations"
- **FAISS:** Johnson et al. (2019) - "Billion-scale similarity search with GPUs"

---

**Last Updated:** 2025-11-11
**Training Status:** In Progress (Epoch 1/10)
**Next Action:** Wait for training completion, then evaluate
