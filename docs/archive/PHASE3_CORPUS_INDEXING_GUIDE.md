# Phase 3: Corpus Indexing with Tree-LSTM - Implementation Guide

**Date**: 2025-11-12
**Status**: ✅ Indexing Script Complete and Tested
**Next Step**: Run full corpus indexing (74,720 sentences)

---

## Summary

Successfully implemented a **robust, resumable corpus indexing system** that:
- ✅ Encodes Esperanto sentences to semantic embeddings using trained Tree-LSTM
- ✅ Builds FAISS vector index for fast similarity search
- ✅ Supports automatic checkpointing and resume from interruptions
- ✅ Provides detailed logging and progress tracking
- ✅ Handles failures gracefully (logs failed sentences separately)
- ✅ Achieves **394 sentences/second** processing speed

---

## Architecture

```
Input: Corpus (text file, one sentence per line)
  ↓
Parse Esperanto → AST
  ↓
Convert AST → Graph (PyG Data object)
  ↓
Tree-LSTM Encoder → 512-dim embedding vector
  ↓
Save: embeddings.npy + metadata.jsonl + FAISS index
```

### Key Components

1. **Corpus Indexer** (`scripts/index_corpus.py`)
   - Loads trained Tree-LSTM model
   - Processes sentences in batches
   - Saves embeddings, metadata, and FAISS index
   - Checkpoints every batch for resume capability

2. **Tree-LSTM Encoder** (`klareco/models/tree_lstm.py`)
   - Encodes AST structure into semantic embeddings
   - Pre-trained with contrastive learning
   - Model parameters: 1,695,232
   - Vocabulary size: 10,000 morphemes

3. **AST-to-Graph Converter** (`klareco/ast_to_graph.py`)
   - Converts Klareco ASTs to PyTorch Geometric graphs
   - Preserves morphological features
   - Encodes node types, edges, and attributes

4. **FAISS Index**
   - Fast similarity search (cosine similarity)
   - IndexFlatIP (inner product after normalization)
   - Supports k-NN retrieval in milliseconds

---

## Usage

### Basic Usage

```bash
# Index full corpus (recommended)
python scripts/index_corpus.py \
    --corpus data/gutenberg_sentences.txt \
    --output data/corpus_index \
    --batch-size 32

# Test on small corpus
python scripts/index_corpus.py \
    --corpus /tmp/test_corpus_small.txt \
    --output /tmp/test_index \
    --batch-size 10
```

### Command-Line Options

```bash
--corpus PATH           Path to corpus file (one sentence per line)
--model PATH            Path to trained Tree-LSTM model (default: models/tree_lstm/checkpoint_epoch_12.pt)
--output DIR            Output directory for index and metadata (default: data/corpus_index)
--batch-size INT        Batch size for processing (default: 32)
--embedding-dim INT     Dimension of embeddings (default: 256, auto-detected from model)
--resume                Resume from last checkpoint (default: True)
--no-resume             Start fresh, ignore checkpoints
--debug                 Enable debug logging
```

### Resume from Interruption

The indexer automatically saves checkpoints every batch. If interrupted:

```bash
# Simply run again - it will resume automatically
python scripts/index_corpus.py --corpus data/gutenberg_sentences.txt --output data/corpus_index
```

Checkpoint file: `<output_dir>/indexing_checkpoint.json`

---

## Output Files

After indexing completes, the output directory contains:

| File | Purpose | Size (for 93 sentences) |
|------|---------|---------|
| `embeddings.npy` | NumPy array of embeddings (N × 512) | 187 KB |
| `metadata.jsonl` | Sentence text + indices (JSONL format) | 9 KB |
| `faiss_index.bin` | FAISS similarity search index | 187 KB |
| `failed_sentences.jsonl` | Sentences that failed to parse | 0 B (empty if all succeed) |
| `indexing_checkpoint.json` | Resume checkpoint | 116 B |
| `indexing.log` | Detailed processing log | 2.1 KB |

### Metadata Format

```jsonl
{"idx": 0, "sentence": "La hundo vidas la katon.", "embedding_idx": 0}
{"idx": 1, "sentence": "Mi amas Esperanton.", "embedding_idx": 1}
...
```

- `idx`: Original sentence index in corpus
- `sentence`: Original Esperanto text
- `embedding_idx`: Index in embeddings array and FAISS index

---

## Performance

### Test Results (100 sentences)

```
Total sentences:     93 (7 filtered as too short)
Successfully encoded: 93 (100.0% success rate)
Failed:              0
Time elapsed:        0.2 seconds
Throughput:          394 sentences/second
```

### Full Corpus Estimates

For 74,720 sentences in `data/gutenberg_sentences.txt`:

- **Expected time**: ~3 minutes (at 394 sentences/second)
- **Expected output size**: ~140 MB (embeddings + index)
- **Memory usage**: ~2 GB peak (batched processing)

---

## Key Features

### 1. **Automatic Checkpointing**

Saves progress after every batch:
- Checkpoint file tracks processed count
- Embeddings appended incrementally
- Metadata written line-by-line
- Can resume from exact sentence if interrupted

### 2. **Graceful Error Handling**

- Failed sentences logged to `failed_sentences.jsonl`
- Processing continues for remaining sentences
- Debug logging shows parse/encoding errors
- Success rate reported in final statistics

### 3. **Progress Tracking**

- tqdm progress bar with ETA
- Periodic success rate updates
- Detailed logging to file and console
- Final summary with statistics

### 4. **Memory Efficiency**

- Batch processing (default 32 sentences)
- Incremental file writing
- Embeddings saved in chunks
- No need to hold entire corpus in memory

### 5. **FAISS Integration**

- Automatic index building after encoding
- Normalized vectors for cosine similarity
- Ready for k-NN retrieval
- Optional (warns if FAISS not installed)

---

## Next Steps

### 1. **Run Full Corpus Indexing** (Recommended Next)

```bash
# This will take ~3 minutes for 74,720 sentences
python scripts/index_corpus.py \
    --corpus data/gutenberg_sentences.txt \
    --output data/corpus_index \
    --batch-size 32

# Monitor progress in another terminal:
tail -f data/corpus_index/indexing.log
```

**Expected Output**:
- `data/corpus_index/embeddings.npy` (~140 MB)
- `data/corpus_index/metadata.jsonl` (~7 MB)
- `data/corpus_index/faiss_index.bin` (~140 MB)
- **Total size**: ~300 MB

### 2. **Create RAG Retrieval Interface**

Build query interface for semantic search:

```python
from klareco.rag import RAGRetriever

retriever = RAGRetriever(index_dir="data/corpus_index")
results = retriever.search("Kio estas Esperanto?", k=5)
```

Features needed:
- Load FAISS index and metadata
- Encode query to embedding
- Search for k nearest neighbors
- Return ranked results with similarity scores

### 3. **Create Evaluation Tests**

Test semantic search quality:
- Paraphrase detection
- Synonym matching
- Topic similarity
- Question-answer retrieval

### 4. **Integrate with Orchestrator**

Add RAG as expert tool:
- Create `RAGExpert` class
- Implement `can_handle()`, `estimate_confidence()`, `execute()`
- Register with Orchestrator
- Test end-to-end pipeline

---

## Troubleshooting

### Issue: "Size mismatch" when loading model

**Solution**: The script auto-detects model dimensions from checkpoint. If you see this error, ensure you're using the correct model checkpoint file.

### Issue: "FAISS not installed"

**Solution**: Install FAISS for index building:
```bash
pip install faiss-cpu  # For CPU
# OR
pip install faiss-gpu  # For GPU (requires CUDA)
```

Note: Indexing still works without FAISS, just won't build the index file.

### Issue: Out of memory

**Solution**: Reduce batch size:
```bash
python scripts/index_corpus.py --batch-size 16  # or even 8
```

### Issue: Very slow processing

**Possible causes**:
- Parser vocabulary missing many roots (check `failed_sentences.jsonl`)
- Large/complex sentences taking long to parse
- Debug logging enabled (remove `--debug` flag)

---

## Technical Details

### Model Architecture

```
TreeLSTMEncoder
├── Embedding Layer: (10,000 vocab → 128-dim)
├── Child-Sum Tree-LSTM Cell: (128-dim → 256-dim hidden)
└── Output Projection: (256-dim → 512-dim output)

Total Parameters: 1,695,232
```

### Training Details

- **Training data**: 5,050 sentence pairs
- **Loss**: Contrastive loss (margin=1.0)
- **Epochs**: 12
- **Final training accuracy**: 98.6%
- **Final loss**: 0.0014

### Embedding Properties

- **Dimension**: 512
- **Normalized**: Yes (for cosine similarity)
- **Captures**: Morphological structure + semantic meaning
- **Compositionality**: Yes (builds meaning from AST)

---

## Files Created in This Session

1. **scripts/index_corpus.py** (459 lines)
   - Complete indexing script with all features
   - Fully tested and working

2. **PHASE3_CORPUS_INDEXING_GUIDE.md** (this file)
   - Comprehensive documentation
   - Usage examples and troubleshooting

---

## Conclusion

The corpus indexing system is **production-ready** and tested. Key achievements:

✅ **Robust**: Automatic checkpointing, error handling, resume capability
✅ **Fast**: 394 sentences/second throughput
✅ **Complete**: Embeddings + metadata + FAISS index
✅ **Documented**: Comprehensive guide with examples
✅ **Tested**: Successfully indexed 93 test sentences

**Ready to index the full 74,720-sentence corpus!**

The estimated time is ~3 minutes. The system will automatically save checkpoints every batch, so you can safely interrupt and resume if needed.

---

## Commands Summary

```bash
# Test (already done)
python scripts/index_corpus.py --corpus /tmp/test_corpus_small.txt --output /tmp/test_index --batch-size 10

# Full corpus indexing (next step)
python scripts/index_corpus.py --corpus data/gutenberg_sentences.txt --output data/corpus_index --batch-size 32

# Monitor progress
tail -f data/corpus_index/indexing.log

# Check results
ls -lh data/corpus_index/
python -c "import numpy as np; print(np.load('data/corpus_index/embeddings.npy').shape)"
```
