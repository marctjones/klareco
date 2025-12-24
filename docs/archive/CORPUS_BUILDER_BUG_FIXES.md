# Corpus Builder Bug Fix Plan

## Overview

This document details all bugs found in the corpus building scripts and the plan to fix them.

**Files affected:**
- `scripts/build_corpus_v2.py` - main corpus builder
- `scripts/extract_sentences.py` - sentence extraction with timeout
- `scripts/run_corpus_builder.sh` - shell wrapper

---

## CRITICAL-1: No Protection for Existing Output File

### Problem
The script decides write mode based solely on checkpoint existence:
```python
mode = 'a' if checkpoint else 'w'
```

If checkpoint is missing/corrupt but output file has millions of sentences, everything gets overwritten.

### Root Cause
- Line 112 in `build_corpus_v2.py`
- No validation that output file should be preserved

### Fix Plan

1. **Add output file existence check before choosing mode:**
```python
# Check if output file exists and has content
output_exists = output_file.exists() and output_file.stat().st_size > 0

if checkpoint:
    # Checkpoint exists - append mode
    mode = 'a'
elif output_exists:
    # No checkpoint but output file has data - REFUSE to overwrite
    print(f"❌ ERROR: Output file {output_file} exists with data but no checkpoint found!")
    print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"   Use --clean to start fresh (will delete existing data)")
    print(f"   Or restore checkpoint file to resume")
    sys.exit(1)
else:
    # No checkpoint, no output file - fresh start
    mode = 'w'
```

2. **Add `--clean` flag behavior to delete output file:**
```python
if args.clean:
    if output_file.exists():
        print(f"🗑️  Removing existing output file: {output_file}")
        output_file.unlink()
    if checkpoint_path.exists():
        print(f"🗑️  Removing checkpoint: {checkpoint_path}")
        checkpoint_path.unlink()
```

3. **Location:** Lines 100-115 in `build_corpus_v2.py`

---

## BUG-1: Checkpoint Points to Completed File Instead of Next File

### Problem
After completing a file, checkpoint saves:
```json
{"file": "cleaned_la_hobito.txt", "total": 23828, "sentence_offset": 0}
```

On resume, it re-processes `cleaned_la_hobito.txt` from the beginning, creating duplicates.

### Root Cause
- Line 249-251 in `build_corpus_v2.py`:
```python
if checkpoint_path:
    save_checkpoint(checkpoint_path, filename, stats['total_sentences'], 0)
```

### Fix Plan

1. **Create helper to get next file in list:**
```python
def get_next_file(texts: List[Tuple[str, str]], current_file: str) -> Optional[str]:
    """Get the next file after current_file in the texts list."""
    for i, (filename, _) in enumerate(texts):
        if filename == current_file:
            if i + 1 < len(texts):
                return texts[i + 1][0]
            return None  # current_file is the last one
    return None
```

2. **Update checkpoint save after file completion:**
```python
# After file completes, checkpoint should point to NEXT file
if checkpoint_path:
    next_file = get_next_file(texts, filename)
    if next_file:
        # More files to process - point to next one
        save_checkpoint(checkpoint_path, next_file, stats['total_sentences'], 0, 0, 0)
    else:
        # All files done - remove checkpoint
        checkpoint_path.unlink()
```

3. **Location:** Lines 249-251 in `build_corpus_v2.py`

---

## BUG-2: Byte Position Not Tracked for Small Files

### Problem
Files <100MB use `_process_text_streaming()` directly without byte tracking.
Checkpoint shows `byte_position: 0, file_size: 0` for these files.

### Root Cause
- Lines 294-302 in `extract_sentences.py`
- Small file path doesn't add `_byte_position` or `_file_size` to entries

### Fix Plan

1. **Add byte tracking to small file path:**
```python
def extract_sentences_streaming(...) -> Iterator[Dict]:
    source_file = str(file_path.name)
    file_size = file_path.stat().st_size  # Always get file size

    if file_size_mb > 100:
        yield from _extract_sentences_chunked(...)
    else:
        # For smaller files, load entire file (faster)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Add byte tracking to each entry
        for entry in _process_text_streaming(
            text, min_words, max_words, with_ast, batch_size,
            start_para=1, source_file=source_file, parse_timeout=parse_timeout
        ):
            entry['_byte_position'] = file_size  # Already read whole file
            entry['_file_size'] = file_size
            yield entry
```

2. **Location:** Lines 284-302 in `extract_sentences.py`

---

## BUG-3: Re-parses All Skipped Sentences on Resume

### Problem
On resume, the script iterates through ALL sentences, parsing each one (with 30s timeout), just to skip them:
```python
for sent_data in sentence_iterator:
    if sent_idx < start_idx:
        sent_idx += 1
        continue  # Throws away parsed result
```

If resuming at sentence 100,000, it re-parses 100,000 sentences (potentially hours of work).

### Root Cause
- Lines 161-166 in `build_corpus_v2.py`
- No way to seek to a byte position in the file

### Fix Plan

1. **Add byte-position based resumption to `extract_sentences_streaming`:**
```python
def extract_sentences_streaming(
    file_path: Path,
    ...
    start_byte: int = 0,  # NEW: byte position to start from
) -> Iterator[Dict]:
```

2. **Update `_extract_sentences_chunked` to seek to start position:**
```python
def _extract_sentences_chunked(..., start_byte: int = 0) -> Iterator[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        if start_byte > 0:
            f.seek(start_byte)
            # Read to next paragraph boundary to avoid partial paragraph
            partial = f.readline()  # Discard partial line
            # Find next double-newline
            while True:
                line = f.readline()
                if not line or line.strip() == '':
                    break

        bytes_read = f.tell()
        # ... rest of chunked processing
```

3. **Update checkpoint to store byte position and use it on resume:**
```python
# In build_corpus_v2.py
byte_offset = checkpoint.get('byte_position', 0) if checkpoint else 0

sentence_iterator = extract_sentences_streaming(
    file_path,
    ...
    start_byte=byte_offset,  # NEW: start from byte position
)

# No longer need to skip sentences in Python - file seek handles it
```

4. **Remove the inefficient skip loop:**
```python
# DELETE this code:
sent_idx = 0
for sent_data in sentence_iterator:
    if sent_idx < start_idx:
        sent_idx += 1
        continue
```

5. **Locations:**
   - `extract_sentences.py`: lines 257-302, 392-470
   - `build_corpus_v2.py`: lines 145-166

---

## MINOR-1: Checkpoint Delete Can Crash on Success

### Problem
```python
if not args.no_checkpoint and args.checkpoint.exists():
    args.checkpoint.unlink()
```
If file is locked, this crashes after successful completion.

### Fix Plan
```python
if not args.no_checkpoint and args.checkpoint.exists():
    try:
        args.checkpoint.unlink()
        print("🧹 Checkpoint file removed (build complete)")
    except OSError as e:
        print(f"⚠️  Could not remove checkpoint file: {e}")
```

**Location:** Lines 366-368 in `build_corpus_v2.py`

---

## MINOR-2: Documentation Says 5s Timeout But Code Uses 30s

### Problem
Docstring says 5s but actual default is 30s.

### Fix Plan
Update docstring at line 88 in `build_corpus_v2.py`:
```python
parse_timeout: Max seconds to wait for parsing a sentence (default: 30)
```

**Location:** Line 88 in `build_corpus_v2.py`

---

## MINOR-3: Problem Sentences Log Directory May Not Exist

### Problem
```python
PROBLEM_SENTENCES_LOG = Path(__file__).parent.parent / "data" / "problem_sentences.jsonl"
# Later:
with open(PROBLEM_SENTENCES_LOG, 'a', ...) as f:  # Crashes if data/ doesn't exist
```

### Fix Plan
```python
def log_problem_sentence(sentence: str, source_file: str, para_num: int, error: str):
    """Log a problem sentence for later analysis."""
    # Ensure directory exists
    PROBLEM_SENTENCES_LOG.parent.mkdir(parents=True, exist_ok=True)

    entry = {...}
    with open(PROBLEM_SENTENCES_LOG, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
```

**Location:** Lines 68-79 in `extract_sentences.py`

---

## MINOR-4: SIGALRM Not Available on Windows

### Problem
```python
signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(timeout_seconds)
```
This crashes on Windows.

### Fix Plan
Use a cross-platform timeout approach with threading:
```python
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

def parse_with_timeout(text: str, timeout_seconds: int = 30):
    """Parse text with a timeout (cross-platform)."""
    from klareco.parser import parse

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(parse, text)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            raise ParseTimeout(f"Parse timed out after {timeout_seconds}s")
```

**Note:** This is lower priority since the target system is Linux. Can keep SIGALRM with a platform check:
```python
import platform

if platform.system() != 'Windows':
    # Use SIGALRM (more efficient on Unix)
    def parse_with_timeout(...):
        # ... SIGALRM implementation
else:
    # Use threading (Windows fallback)
    def parse_with_timeout(...):
        # ... ThreadPoolExecutor implementation
```

**Location:** Lines 39-65 in `extract_sentences.py`

---

## MINOR-5: Confusing Flag Reset Logic

### Problem
```python
start_idx = sentence_offset if resuming_file else 0
if resuming_file:
    print(f"   ⏩ Skipping first {start_idx} sentences (already processed)")
    resuming_file = False
    sentence_offset = 0
```

The flag is reset but `start_idx` is already set. Works but confusing.

### Fix Plan
This will be eliminated by BUG-3 fix (byte-position based resumption). No separate fix needed.

---

## Implementation Order

1. **CRITICAL-1** - Output file protection (prevents data loss)
2. **BUG-1** - Checkpoint points to next file (prevents duplicates)
3. **MINOR-3** - Create log directory (prevents crash)
4. **MINOR-1** - Safe checkpoint delete (prevents crash)
5. **MINOR-2** - Fix documentation
6. **BUG-2** - Byte tracking for small files (enables progress display)
7. **BUG-3** - Byte-position based resumption (performance improvement)
8. **MINOR-4** - Windows compatibility (optional, low priority)

---

## Testing Plan

After implementing fixes:

1. **Test CRITICAL-1:**
   - Create corpus with some data
   - Delete checkpoint file
   - Run script without --clean
   - Verify it refuses to overwrite
   - Run with --clean
   - Verify it starts fresh

2. **Test BUG-1:**
   - Run until first file completes
   - Kill script
   - Check checkpoint points to second file
   - Resume and verify no duplicates

3. **Test BUG-2:**
   - Run on small file (<100MB)
   - Check checkpoint has byte_position > 0

4. **Test BUG-3:**
   - Run partway through large file
   - Kill and resume
   - Verify fast resumption (no re-parsing)

5. **Test MINOR-3:**
   - Delete data/ directory
   - Run with a problem sentence
   - Verify directory created and log written
