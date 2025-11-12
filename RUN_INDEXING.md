# How to Run Corpus Indexing

## Quick Start

### In Your Terminal:

```bash
cd /home/marc/klareco
./scripts/run_corpus_indexing.sh
```

That's it! The script will:
- ✅ Check for corpus and model files
- ✅ Show you the estimated time (~3 minutes)
- ✅ Start indexing with progress bar
- ✅ Automatically resume if interrupted
- ✅ Display results when complete

---

## Two-Terminal Setup (Recommended)

### Terminal 1: Run Indexing
```bash
cd /home/marc/klareco
./scripts/run_corpus_indexing.sh
```

### Terminal 2: Watch Logs
```bash
tail -f data/corpus_index/indexing.log
```

This lets you see detailed progress in real-time!

---

## What You'll See

### Terminal 1 Output:
```
========================================
  Klareco Corpus Indexing
========================================

Corpus: data/gutenberg_sentences.txt
Sentences: 74720
Output: data/corpus_index
Batch size: 32

Estimated time: ~187 seconds (~3 minutes)

To monitor progress in another terminal:
  tail -f data/corpus_index/indexing.log

Starting indexing in 3 seconds...
2...
1...

Starting corpus indexing...

Indexing corpus:  15%|███▍       | 11234/74720 [00:25<02:23, 442.3it/s]
...
```

### Terminal 2 Output (Log Monitoring):
```
2025-11-12 09:30:15,123 - INFO - Loading model from models/tree_lstm/checkpoint_epoch_12.pt
2025-11-12 09:30:15,145 - INFO - Model loaded successfully
2025-11-12 09:30:15,146 - INFO - Loading corpus from data/gutenberg_sentences.txt
2025-11-12 09:30:15,234 - INFO -   Loaded 74720 sentences
2025-11-12 09:30:15,234 - INFO - Processing sentences...
```

---

## If Interrupted

The indexing is **fully resumable**. Just run the same command again:

```bash
./scripts/run_corpus_indexing.sh
```

It will automatically detect the checkpoint and resume from where it left off:

```
Found existing checkpoint: 35000/74720 sentences processed
Will resume from checkpoint
```

---

## Output Files

After completion, you'll find these files in `data/corpus_index/`:

| File | Description | Size |
|------|-------------|------|
| `embeddings.npy` | Semantic embeddings (74K × 512) | ~140 MB |
| `metadata.jsonl` | Sentence text + indices | ~7 MB |
| `faiss_index.bin` | Fast similarity search index | ~140 MB |
| `failed_sentences.jsonl` | Sentences that failed to parse | <1 KB |
| `indexing_checkpoint.json` | Resume checkpoint | <1 KB |
| `indexing.log` | Detailed processing log | <5 MB |

**Total**: ~300 MB

---

## Advanced Usage

### Run with Different Options

If you want to customize, you can run the Python script directly:

```bash
python scripts/index_corpus.py \
    --corpus data/gutenberg_sentences.txt \
    --output data/corpus_index \
    --batch-size 32 \
    --resume
```

### Options:
- `--batch-size 16` - Use smaller batches (if running out of memory)
- `--no-resume` - Start fresh (ignore existing checkpoint)
- `--debug` - Enable debug logging

### Check Progress Anytime

```bash
# View checkpoint
cat data/corpus_index/indexing_checkpoint.json

# Count processed sentences
wc -l data/corpus_index/metadata.jsonl

# Check for failures
cat data/corpus_index/failed_sentences.jsonl
```

---

## Troubleshooting

### Script won't run
```bash
chmod +x scripts/run_corpus_indexing.sh
```

### "Corpus file not found"
Make sure you're in the `/home/marc/klareco` directory

### Out of memory
Reduce batch size:
```bash
python scripts/index_corpus.py --batch-size 16
```

### Want to start over
```bash
rm -rf data/corpus_index
./scripts/run_corpus_indexing.sh
```

---

## What Happens Next

After indexing completes, you can:

1. **Query the index** - Search for similar sentences
2. **Build RAG retrieval** - Semantic search for question answering
3. **Integrate with Orchestrator** - Add as expert tool
4. **Evaluate quality** - Test semantic similarity

---

## Performance

- **Speed**: ~400-500 sentences/second
- **Time**: ~3 minutes for 74K sentences
- **Success rate**: ~99.5% (parsing success)
- **Checkpoints**: Every 32 sentences (every ~0.1 seconds)

---

## Questions?

- View the full guide: `PHASE3_CORPUS_INDEXING_GUIDE.md`
- Check the script source: `scripts/index_corpus.py`
- Monitor logs: `tail -f data/corpus_index/indexing.log`
