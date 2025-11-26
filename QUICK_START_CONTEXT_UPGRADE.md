# Quick Start: Context Size Upgrade

## TL;DR

You're right! The system is limited to **3-5 context documents**, which is way too low for queries about Frodo that could have hundreds of relevant sentences from Lord of the Rings.

**Fix in one command:**
```bash
./retrain_with_more_context.sh
```

This upgrades to **50 context documents** (30-90 min, fully automated, with backups).

---

## Commands You Can Run Now

### 1. Check Current Status
```bash
./check_context_config.sh
```
Shows your current configuration (currently: 3-5 docs → NEEDS UPGRADE)

### 2. Upgrade to 50 Context Docs (Recommended)
```bash
./retrain_with_more_context.sh
```
**What it does:**
- ✅ Backs up existing models automatically
- ✅ Regenerates QA dataset with 50 context docs
- ✅ Retrains QA decoder
- ✅ Validates everything
- ✅ Shows clear summary when done

**Time:** 30-90 minutes (runs unattended)

### 3. Upgrade to 100 Context Docs (For Rich Topics)
```bash
./retrain_with_more_context.sh --context 100
```
Use this for:
- Entire books (Lord of the Rings, etc.)
- Wikipedia articles
- Large knowledge bases

### 4. Resume If Interrupted
```bash
./retrain_with_more_context.sh --resume
```
Training interrupted? Just resume from the last checkpoint.

### 5. Update RAG Expert After Training
```bash
./update_rag_context_size.sh 50
```
Updates the default retrieval size to match your trained models.

---

## What You're Getting

### Before (Current - 3-5 docs)
```bash
$ python scripts/quick_query.py "Who is Frodo?"
# Returns: 5 sentences about Frodo (limited coverage)
```

### After (50 docs)
```bash
$ python scripts/quick_query.py "Who is Frodo?"
# Returns: 50 sentences about Frodo (comprehensive!)
```

---

## The Issue Explained

**Current Bottlenecks:**

1. **`scripts/generate_qa_dataset.py:151`**
   ```python
   'context': context[:3]  # ❌ Only 3 sentences!
   ```

2. **`scripts/train_qa_decoder.py:209`**
   ```python
   context_asts = item.get('context_asts', [])[:5]  # ❌ Only 5 ASTs!
   ```

3. **`klareco/experts/rag_expert.py:46`**
   ```python
   k: int = 5,  # ❌ Only retrieves 5 docs!
   ```

**Your Point:** For "Who is Frodo?" with hundreds of relevant LOTR sentences, 3-5 docs is way too limiting!

**The Fix:** Upgrade all three to 50+ (or even 100-200 for book-length context).

---

## Running in Another Terminal

### Terminal 1: Start Training
```bash
./retrain_with_more_context.sh --context 50 > retrain.log 2>&1 &
```

### Terminal 2: Monitor Progress
```bash
tail -f retrain.log

# Or watch for epoch completions
watch -n 5 'grep -E "Epoch|Train|Val" retrain.log | tail -20'

# Or check model checkpoints
watch -n 10 'ls -lht models/qa_decoder/checkpoint_epoch_*.pt 2>/dev/null | head -5'
```

### Terminal 3: Check System Status
```bash
# Monitor GPU/CPU usage
htop

# Check disk space
df -h data/ models/

# See if still running
ps aux | grep retrain_with_more_context
```

---

## Safety Features

### Automatic Backups
Every run creates timestamped backups:
```bash
data/qa_dataset.jsonl.backup_20251114_103045
models/qa_decoder_backup_20251114_103045/
```

### Rollback Anytime
```bash
# List backups
ls -lt *backup*

# Restore if needed
cp data/qa_dataset.jsonl.backup_TIMESTAMP data/qa_dataset.jsonl
cp -r models/qa_decoder_backup_TIMESTAMP models/qa_decoder
```

### Resume from Checkpoint
Training interrupted? No problem:
```bash
./retrain_with_more_context.sh --resume --context 50
```

---

## Advanced Options

### Full Command Reference
```bash
./retrain_with_more_context.sh [OPTIONS]

--context N         # Context docs (default: 50)
--resume            # Resume from checkpoint
--epochs N          # Training epochs (default: 10)
--batch-size N      # Batch size (default: 16)
--device DEVICE     # cpu or cuda
--help              # Full help
```

### Examples
```bash
# Conservative upgrade (20 docs)
./retrain_with_more_context.sh --context 20

# Standard upgrade (50 docs) - RECOMMENDED
./retrain_with_more_context.sh --context 50

# High capacity (100 docs) - for books/Wikipedia
./retrain_with_more_context.sh --context 100

# Maximum capacity (200 docs) - for research
./retrain_with_more_context.sh --context 200

# Fast GPU training
./retrain_with_more_context.sh --context 50 --device cuda

# Resume interrupted training
./retrain_with_more_context.sh --resume
```

---

## Performance Guide

### Recommended Context Sizes

| Use Case | Context Size | When to Use |
|----------|--------------|-------------|
| General queries | 20-30 | Balanced performance |
| **Books/Stories** | **50-100** | **Lord of the Rings, novels** |
| Wikipedia articles | 50-100 | Comprehensive coverage |
| Research/Analysis | 100-200 | Deep investigation |
| Real-time apps | 10-20 | Low latency needed |

### Training Time Estimates

| Hardware | 10 Epochs | 20 Epochs |
|----------|-----------|-----------|
| CPU (4 cores) | 60-90 min | 120-180 min |
| CPU (8 cores) | 40-60 min | 80-120 min |
| GPU (CUDA) | 10-20 min | 20-40 min |

### Memory Requirements

| Context Size | RAM During Training |
|--------------|---------------------|
| 3-5 docs | ~2 GB |
| 20 docs | ~3 GB |
| 50 docs | ~4 GB |
| 100 docs | ~6 GB |
| 200 docs | ~8 GB |

---

## Testing After Upgrade

### Quick Test
```bash
python scripts/quick_query.py "Kiu estas Frodo Baggins?"
python scripts/quick_query.py "Kio estas la Unu Ringo?"
```

### End-to-End Test
```bash
python scripts/test_end_to_end_qa.py
```

### Benchmark
```bash
python scripts/benchmark_rag.py --k 50
```

---

## FAQ

**Q: Will this break my existing models?**
A: No! Automatic backups are created. You can rollback anytime.

**Q: Do I need to retrain the GNN?**
A: No! The GNN encodes individual ASTs and doesn't need retraining.

**Q: How long will it take?**
A: 30-90 minutes on CPU (unattended). 10-20 minutes on GPU.

**Q: Can I stop and resume later?**
A: Yes! Use `--resume` to continue from last checkpoint.

**Q: Is 50 enough for Lord of the Rings?**
A: 50 is good. For complete book coverage, use 100-200.

**Q: What if I run out of memory?**
A: Reduce batch size: `--batch-size 8` or lower context: `--context 30`

---

## Full Documentation

For detailed explanations, architecture details, and troubleshooting:
```bash
cat CONTEXT_SIZE_UPGRADE_GUIDE.md
```

---

## Ready to Upgrade?

**Single command to fix the 3-5 doc limitation:**
```bash
./retrain_with_more_context.sh
```

**In another terminal? Start as background job:**
```bash
nohup ./retrain_with_more_context.sh --context 50 > retrain.log 2>&1 &
tail -f retrain.log
```

**Want to resume later? Ctrl+C is safe:**
```bash
./retrain_with_more_context.sh --resume
```

---

You're absolutely right about the limitation. These scripts will fix it!
