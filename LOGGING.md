# Real-Time Log Monitoring with watch.sh

## Quick Start

**Terminal 1 - Run your tests:**
```bash
./run.sh
```

**Terminal 2 - Watch logs in real-time:**
```bash
./watch.sh
```

That's it! You'll see structured Python logging output in real-time as tests run.

## How It Works

### Log Files

1. **`klareco.log`** - Structured Python logging
   - Timestamped entries with log levels (INFO, WARNING, ERROR)
   - Appends with "NEW RUN STARTED" separators
   - Watched by `watch.sh` in real-time
   - Format: `2025-11-11 10:19:59,221 - root - INFO - KlarecoPipeline initialized.`

2. **`run_output.txt`** - Complete stdout/stderr capture
   - Includes tqdm progress bars
   - Created by `run.sh` redirection
   - For final review after tests complete
   - Not watched in real-time (tqdm doesn't work well with pipes)

### The Scripts

**`watch.sh`:**
- Displays last 50 lines of `klareco.log`
- Follows file in real-time with `tail -f`
- Press Ctrl+C to exit
- Safe to run before tests start (creates empty file if needed)

**`run.sh`:**
- Activates conda environment
- Runs integration tests with coverage
- Redirects output to `run_output.txt`
- Python logging still goes to `klareco.log` (dual output)

## Why This Design?

**Problem:** tqdm progress bars don't work well when piped through `tail -f` or `tee`

**Solution:** Separate concerns:
- **Structured logs** → `klareco.log` → watchable in real-time
- **Console output + progress bars** → `run_output.txt` → review after completion

This gives you:
- ✅ Real-time visibility into what's happening
- ✅ Progress bars that actually work
- ✅ Complete output for final review
- ✅ No pipe/buffer issues

## Examples

### Watch logs while running a long test:
```bash
# Terminal 1
./run.sh

# Terminal 2
./watch.sh
```

### Watch logs during corpus processing:
```bash
# Terminal 1
python scripts/build_morpheme_vocab.py

# Terminal 2
./watch.sh
```

### View recent logs without watching:
```bash
tail -50 klareco.log
```

### Clear old logs (optional):
```bash
rm klareco.log
# Will be recreated on next run
```

## Log Levels

The system logs at these levels:
- **INFO**: Normal operation (pipeline steps, progress)
- **WARNING**: Potential issues (missing data, fallbacks)
- **ERROR**: Failures (parser errors, test failures)

## Troubleshooting

**Q: watch.sh shows "No such file or directory"**
```bash
# Create the log file first
touch klareco.log
./watch.sh
```

**Q: Logs not updating in real-time**
- Make sure you're in the project root directory
- Check that tests are actually running in Terminal 1
- Try restarting watch.sh

**Q: Want to see more history**
```bash
# Edit watch.sh and change -n 50 to -n 100
tail -n 100 -f klareco.log
```

**Q: Want to see less history (just new logs)**
```bash
# Edit watch.sh and change -n 50 to -n 0
tail -n 0 -f klareco.log
```
