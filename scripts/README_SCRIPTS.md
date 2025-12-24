# Klareco Scripts Guide

## Corpus Builder Scripts

### Main Runner: `run_corpus_builder.sh`

**Robust corpus builder with automatic environment setup, checkpointing, and logging.**

#### Basic Usage

```bash
# Standard run (recommended)
./scripts/run_corpus_builder.sh

# Clean start (remove checkpoint and start fresh)
./scripts/run_corpus_builder.sh --clean

# Fast mode (max speed, may freeze on slow systems)
./scripts/run_corpus_builder.sh --fast

# Gentle mode (slower but safer, won't freeze)
./scripts/run_corpus_builder.sh --gentle

# Skip AST generation (faster but no quality info)
./scripts/run_corpus_builder.sh --no-ast
```

#### What It Does

1. **Environment Setup**
   - Creates/activates Python virtual environment
   - Installs requirements automatically
   - Verifies dependencies

2. **Checkpointing**
   - Saves progress every 20 sentences (default)
   - Automatically resumes if interrupted
   - Use `--clean` to start fresh

3. **Logging**
   - Creates timestamped log file in `logs/`
   - Symlinks to `logs/corpus_builder_latest.log`
   - Shows output in terminal AND saves to file

4. **Safe Interruption**
   - Press Ctrl+C to stop gracefully
   - Progress saved in checkpoint
   - Run again to resume

#### Default Settings

- **Batch size**: 20 sentences (checkpoint frequency)
- **Throttle**: 0.1s delay between batches
- **Min parse rate**: 0.0 (no filtering, keep all Esperanto)
- **AST generation**: Enabled

#### Modes

**Standard (Balanced)**
```bash
./scripts/run_corpus_builder.sh
```
- Good balance of speed and stability
- Won't freeze most systems

**Fast Mode**
```bash
./scripts/run_corpus_builder.sh --fast
```
- Batch size: 50
- Throttle: 0.0s (no delay)
- Use if you have a fast system and want maximum speed

**Gentle Mode**
```bash
./scripts/run_corpus_builder.sh --gentle
```
- Batch size: 10
- Throttle: 0.3s (more delay)
- Use if system is freezing or low on resources

**No AST Mode**
```bash
./scripts/run_corpus_builder.sh --no-ast
```
- Skips AST generation
- Much faster but no parse quality info
- Use for quick corpus building without quality filtering

#### Logs

All logs saved to `logs/corpus_builder_TIMESTAMP.log`

Access latest log:
```bash
# View latest log
cat logs/corpus_builder_latest.log

# Follow log in real-time
tail -f logs/corpus_builder_latest.log

# View with colors
less -R logs/corpus_builder_latest.log
```

---

### Monitor Progress: `monitor_corpus_builder.sh`

**Check corpus builder status and progress.**

#### Usage

```bash
# View current status
./scripts/monitor_corpus_builder.sh

# Monitor every 5 seconds
watch -n 5 ./scripts/monitor_corpus_builder.sh
```

#### What It Shows

1. **Running Status**
   - Is builder currently running?
   - Process ID

2. **Current Progress**
   - Checkpoint info (which file, sentences processed)
   - Output file size
   - Sentence count

3. **System Resources**
   - Memory usage
   - Disk space
   - CPU usage (if running)

4. **Recent Log Output**
   - Last 20 lines of log

---

## Running from Another Terminal

### Start Builder in Background

```bash
# Option 1: Run in background with nohup
nohup ./scripts/run_corpus_builder.sh > /dev/null 2>&1 &

# Option 2: Use tmux (recommended)
tmux new -s corpus
./scripts/run_corpus_builder.sh
# Press Ctrl+B, then D to detach

# Option 3: Use screen
screen -S corpus
./scripts/run_corpus_builder.sh
# Press Ctrl+A, then D to detach
```

### Monitor from Another Terminal

```bash
# Terminal 1: Running builder
./scripts/run_corpus_builder.sh

# Terminal 2: Monitor progress
watch -n 5 ./scripts/monitor_corpus_builder.sh

# Terminal 3: Follow logs
tail -f logs/corpus_builder_latest.log
```

### Reattach to Running Session

```bash
# If using tmux
tmux attach -t corpus

# If using screen
screen -r corpus
```

---

## Troubleshooting

### Builder Stops with Error

**Check the log:**
```bash
cat logs/corpus_builder_latest.log
```

**Common issues:**
- Missing files: Check `data/cleaned/` directory exists
- Import errors: Run `pip install -r requirements.txt`
- Permission errors: Check file permissions

**Resume after fixing:**
```bash
./scripts/run_corpus_builder.sh
# Automatically resumes from checkpoint
```

### System Freezing

**Use gentler settings:**
```bash
./scripts/run_corpus_builder.sh --gentle
```

**Or adjust throttle manually:**
```bash
# Edit the script or use Python directly
python scripts/build_corpus_v2.py --throttle 0.5 --batch-size 5
```

### Out of Memory

**Try:**
1. Close other applications
2. Use gentler settings (smaller batch size)
3. Skip AST generation temporarily:
   ```bash
   ./scripts/run_corpus_builder.sh --no-ast --gentle
   ```

### Checkpoint Corruption

**Start fresh:**
```bash
./scripts/run_corpus_builder.sh --clean
```

### Want to Access Logs Later

All logs are saved in `logs/` directory with timestamps:
```bash
# List all logs
ls -lht logs/corpus_builder_*.log

# View specific log
cat logs/corpus_builder_20251127_143022.log

# Search for errors
grep -i error logs/corpus_builder_*.log
```

---

## File Locations

### Input
- `data/cleaned/` - Cleaned text files

### Output
- `data/corpus_with_sources_v2.jsonl` - Built corpus

### Temporary
- `data/build_corpus_v2_checkpoint.json` - Checkpoint (auto-deleted on success)

### Logs
- `logs/corpus_builder_TIMESTAMP.log` - Timestamped logs
- `logs/corpus_builder_latest.log` - Symlink to latest log

---

## Examples

### Standard Workflow

```bash
# 1. Start builder
./scripts/run_corpus_builder.sh

# 2. Monitor from another terminal
watch -n 5 ./scripts/monitor_corpus_builder.sh

# 3. Follow logs (optional)
tail -f logs/corpus_builder_latest.log
```

### Interrupted and Resume

```bash
# First run - Ctrl+C after processing some files
./scripts/run_corpus_builder.sh
# ... working ... ^C

# Check what was saved
cat data/build_corpus_v2_checkpoint.json

# Resume where left off
./scripts/run_corpus_builder.sh
# Automatically continues!
```

### Run Overnight

```bash
# Option 1: Use tmux
tmux new -s corpus
./scripts/run_corpus_builder.sh
# Detach: Ctrl+B, then D

# Option 2: Use nohup
nohup ./scripts/run_corpus_builder.sh &

# Next morning - check status
./scripts/monitor_corpus_builder.sh
```

### Debug Issues

```bash
# Run with verbose Python output
./scripts/run_corpus_builder.sh 2>&1 | tee debug.log

# Check what went wrong
grep -i error logs/corpus_builder_latest.log

# Try with minimal settings
./scripts/run_corpus_builder.sh --no-ast --gentle
```

---

## Script Features Summary

### `run_corpus_builder.sh`
✅ Automatic environment setup
✅ Checkpointing every N sentences
✅ Automatic resumption on restart
✅ Comprehensive logging
✅ Safe Ctrl+C handling
✅ Multiple speed modes
✅ System info display
✅ Error handling

### `monitor_corpus_builder.sh`
✅ Running status check
✅ Progress display
✅ Resource monitoring
✅ Recent log output
✅ Helpful commands

---

## Quick Reference

```bash
# Start building
./scripts/run_corpus_builder.sh

# Monitor progress
./scripts/monitor_corpus_builder.sh

# Follow logs
tail -f logs/corpus_builder_latest.log

# Stop gracefully
pkill -INT -f build_corpus_v2.py

# Start fresh
./scripts/run_corpus_builder.sh --clean

# Gentle mode (safe)
./scripts/run_corpus_builder.sh --gentle

# Fast mode (risky)
./scripts/run_corpus_builder.sh --fast
```
