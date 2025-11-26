# Klareco Context Size Upgrade Guide

## The Problem

Currently, Klareco's RAG system is limited to retrieving only **3-5 context documents**. This is a significant bottleneck for queries about topics with extensive information (e.g., "Who is Frodo?" could have hundreds of relevant sentences from Lord of the Rings or Wikipedia).

### Current Limitations

1. **`scripts/generate_qa_dataset.py:151`** - Limits to **3** context sentences when generating training data
2. **`scripts/train_qa_decoder.py:209`** - Limits to **5** context ASTs during training
3. **`klareco/experts/rag_expert.py:46`** - Default retrieval k=**5**

For rich topics like characters from books, historical events, or scientific concepts, this severely limits answer quality.

## The Solution

This upgrade regenerates your models to handle **50 context documents** (or any number you choose), dramatically improving retrieval for information-rich queries.

## Quick Start

### Option 1: Fully Automated (Recommended)

Run the complete retraining pipeline:

```bash
# Default: 50 context documents
./retrain_with_more_context.sh

# Custom: 100 context documents
./retrain_with_more_context.sh --context 100

# Resume if interrupted
./retrain_with_more_context.sh --resume --context 50
```

This script will:
1. ✅ Backup your existing models automatically
2. ✅ Regenerate QA dataset with larger context
3. ✅ Retrain QA decoder
4. ✅ Validate the new models
5. ✅ Provide rollback instructions if needed

**Time estimate:** 30-90 minutes depending on hardware (mostly unattended)

### Option 2: Manual Steps

If you prefer control over each step:

#### Step 1: Regenerate QA Dataset

```bash
# Backup original
cp data/qa_dataset.jsonl data/qa_dataset.jsonl.backup

# Edit scripts/generate_qa_dataset.py line 151
# Change: 'context': context[:3]
# To:     'context': context[:50]

# Regenerate
python scripts/generate_qa_dataset.py \
    --corpus data/corpus_sentences.jsonl \
    --output data/qa_dataset.jsonl \
    --method both
```

#### Step 2: Retrain QA Decoder

```bash
# Backup existing model
cp -r models/qa_decoder models/qa_decoder_backup_$(date +%Y%m%d)

# Edit scripts/train_qa_decoder.py line 209
# Change: context_asts = item.get('context_asts', [])[:5]
# To:     context_asts = item.get('context_asts', [])[:50]

# Retrain
python scripts/train_qa_decoder.py \
    --dataset data/qa_dataset.jsonl \
    --gnn-checkpoint models/tree_lstm/checkpoint_epoch_20.pt \
    --output models/qa_decoder \
    --epochs 10 \
    --batch-size 16
```

#### Step 3: Update RAG Expert Configuration

```bash
# Automatically update default k value
./update_rag_context_size.sh 50

# Or manually edit klareco/experts/rag_expert.py line 46
# Change: k: int = 5,
# To:     k: int = 50,
```

## Testing the Improvement

### Before Upgrade (3-5 context docs)

```bash
python scripts/quick_query.py "Who is Gandalf?"
# Returns: 5 context sentences (limited)
```

### After Upgrade (50 context docs)

```bash
python scripts/quick_query.py "Who is Gandalf?"
# Returns: 50 context sentences (comprehensive!)
```

### Compare Results

```bash
# Test with complex queries that benefit from more context
python scripts/quick_query.py "Kiu estas Frodo Baggins?"
python scripts/quick_query.py "Kio estas la Unu Ringo?"
python scripts/quick_query.py "Kie estas Mordor?"
```

## Script Reference

### retrain_with_more_context.sh

Main retraining script with automatic backups and resumption.

```bash
./retrain_with_more_context.sh [OPTIONS]

Options:
  --context N        Number of context documents (default: 50)
  --resume           Resume from last checkpoint
  --epochs N         Training epochs (default: 10)
  --batch-size N     Batch size (default: 16)
  --device DEVICE    cpu or cuda (default: cpu)
  --help             Show help

Examples:
  ./retrain_with_more_context.sh                    # Standard: 50 docs
  ./retrain_with_more_context.sh --context 100      # High capacity: 100 docs
  ./retrain_with_more_context.sh --context 20       # Conservative: 20 docs
  ./retrain_with_more_context.sh --resume           # Resume interrupted training
```

### update_rag_context_size.sh

Updates RAG Expert default k value to match your trained models.

```bash
./update_rag_context_size.sh 50    # Set k=50
./update_rag_context_size.sh 100   # Set k=100
```

## Checkpoint Resumption

The training script supports resuming from checkpoints:

```bash
# If training is interrupted, just run with --resume
./retrain_with_more_context.sh --resume --context 50
```

Checkpoints are saved after each epoch in `models/qa_decoder/checkpoint_epoch_N.pt`.

## Rollback

If you need to revert to the original models:

```bash
# Find your backup (timestamped)
ls -lt data/*.backup_*
ls -lt models/qa_decoder_backup_*

# Restore dataset
cp data/qa_dataset.jsonl.backup_TIMESTAMP data/qa_dataset.jsonl

# Restore models
rm -rf models/qa_decoder
cp -r models/qa_decoder_backup_TIMESTAMP models/qa_decoder

# Restore RAG expert
cp klareco/experts/rag_expert.py.backup_TIMESTAMP klareco/experts/rag_expert.py
```

## Performance Considerations

### Context Size Recommendations

| Use Case | Recommended k | Rationale |
|----------|---------------|-----------|
| General queries | 20-30 | Good balance of coverage and speed |
| Rich topics (books, history) | 50-100 | Comprehensive context for complex answers |
| Real-time applications | 10-20 | Faster retrieval, lower latency |
| Research/analysis | 100-200 | Maximum context for deep questions |

### Training Time

| Hardware | Epochs | Estimated Time |
|----------|--------|----------------|
| CPU (4 cores) | 10 | 60-90 minutes |
| CPU (8 cores) | 10 | 40-60 minutes |
| GPU (CUDA) | 10 | 10-20 minutes |

### Memory Usage

- **3 context docs:** ~2GB RAM during training
- **50 context docs:** ~4GB RAM during training
- **100 context docs:** ~6GB RAM during training

## Architecture Impact

### What Changes

1. **QA Dataset** - More context documents per training example
2. **QA Decoder** - Trained to handle variable-length context (up to your max)
3. **RAG Expert** - Retrieves more documents at query time

### What Stays the Same

1. **GNN Encoder** - No retraining needed (it encodes individual ASTs)
2. **Tree-LSTM** - Already trained, works with any number of context docs
3. **FAISS Index** - No changes needed
4. **Parser/Deparser** - Unchanged

## Why This Works

The QA Decoder uses **attention mechanisms** that can handle variable-length context:

```python
# Before (limited)
context_asts = item.get('context_asts', [])[:5]  # Fixed: max 5

# After (flexible)
context_asts = item.get('context_asts', [])[:50]  # Flexible: up to 50
```

The model learns to:
- Attend to relevant context documents
- Ignore irrelevant ones
- Scale gracefully from 1 to 50+ documents

## Troubleshooting

### "ERROR: GNN model not found"

You need to train the GNN first:

```bash
./retrain_gnn.sh
```

### "ERROR: Corpus not found"

Ensure you have the corpus:

```bash
ls -lh data/corpus_sentences.jsonl
```

### Training is very slow

Try reducing batch size or using GPU:

```bash
./retrain_with_more_context.sh --batch-size 8 --device cuda
```

### Out of memory during training

Reduce batch size or context size:

```bash
./retrain_with_more_context.sh --context 30 --batch-size 8
```

## Advanced Usage

### Custom Context Size Per Query

You can override k at query time:

```python
from klareco.experts.rag_expert import RAGExpert
from klareco.rag.retriever import create_retriever

retriever = create_retriever()
expert = RAGExpert(retriever, k=100)  # Override default

# Or use retriever directly
results = retriever.retrieve_hybrid(ast, k=200)  # Up to 200 docs!
```

### Hybrid Retrieval (Keyword + Semantic)

For maximum recall on rich topics:

```python
# Stage 1: Find 200 keyword candidates
# Stage 2: Rerank top 50 semantically
results = retriever.retrieve_hybrid(
    ast,
    k=50,                    # Final results
    keyword_candidates=200   # Initial candidates
)
```

## FAQ

**Q: Do I need to retrain the GNN?**
A: No! The GNN encodes individual ASTs and is context-size independent.

**Q: Will this break existing models?**
A: No. The script creates backups automatically. You can rollback anytime.

**Q: Can I use different k values for different queries?**
A: Yes! The model is trained on *up to* k documents, so it works with any k ≤ trained_max.

**Q: How does this compare to the original?**
A: Original (k=3-5): Good for simple questions. Upgraded (k=50+): Handles complex, information-rich queries.

**Q: Is 50 the optimal value?**
A: 50 is a good default. For book-length context (e.g., LOTR), try 100-200.

## Next Steps

After upgrading:

1. **Test thoroughly** with diverse queries
2. **Benchmark performance** (retrieval time, answer quality)
3. **Tune k value** based on your use case
4. **Consider GPU** for faster training if doing frequent retrains

## Support

If you encounter issues:

1. Check logs in the script output
2. Review backups: `ls -lt *backup*`
3. Try with `--debug` flag for verbose output
4. File an issue with the error message and configuration

---

**Ready to upgrade?**

```bash
./retrain_with_more_context.sh
```

This will take 30-90 minutes but will dramatically improve your RAG system for complex queries!
