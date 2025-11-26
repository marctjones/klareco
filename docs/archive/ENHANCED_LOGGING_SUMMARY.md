# Enhanced Logging System - Your Questions Answered

## Your Question 1: "Will I be able to see which tests and the results as they happen?"

### ✅ YES - Here's what you'll see in watch.sh:

**Progress Updates** (every 10%):
```
2025-11-11 10:25:29,031 - INFO - Testing sentences: 1/5 (20%) - Sentence 1: En la maniero... [ETA: 3s]
2025-11-11 10:25:35,031 - INFO - Testing sentences: 2/5 (40%) - Sentence 2: Forfalintis... [ETA: 9s]
2025-11-11 10:25:40,031 - INFO - Testing sentences: 3/5 (60%) - Sentence 3: Estis tiu... [ETA: 6s]
```

**Test Results** (immediate):
```
2025-11-11 10:25:35,031 - INFO - PASSED sentence 2
2025-11-11 10:25:40,031 - ERROR - FAILED sentence 3: Parser error...
```

**Pipeline Steps** (for every sentence):
```
2025-11-11 10:25:29,032 - INFO - Step 1: SafetyMonitor - Checking input length
2025-11-11 10:25:29,032 - INFO - Step 2: FrontDoor - Processing input text
2025-11-11 10:25:29,373 - INFO - Step 3: Parser - Parsing Esperanto text to AST
2025-11-11 10:25:29,500 - INFO - Step 4: IntentClassifier - Classifying intent
2025-11-11 10:25:29,600 - INFO - Step 5: Responder - Generating response
```

**Final Summary**:
```
2025-11-11 10:25:45,031 - INFO - ================================================================================
2025-11-11 10:25:45,031 - INFO - Integration test COMPLETE: 42 passed, 3 failed out of 45
2025-11-11 10:25:45,031 - INFO - ================================================================================
```

## Your Question 2: "Will this give you enough debugging information?"

### ✅ YES - Two modes available:

### Normal Mode (Default):
- Which test is running
- Which test failed/passed
- Error message (truncated to 100 chars)
- Pipeline step where failure occurred

**Good for:** Monitoring progress, seeing overall health

### Debug Mode (--debug flag):
```bash
python scripts/run_integration_test.py --num-sentences 10 --debug
```

**Gives you:**
- ✅ **Full input text** that caused the error
- ✅ **Complete stack traces** with file:line numbers
- ✅ **Full error messages** (not truncated)
- ✅ **Complete execution trace** (JSON dumps of AST state)
- ✅ **File and line number** for every log entry

**Example Debug Output:**
```
2025-11-11 10:25:29,031 - root - DEBUG - [run_integration_test.py:81] - Full input: En la maniero de mia amiko tujege mirigis min ioma nekohereco,
ioma nekonstanteco; kaj baldaŭ mi konsciis ke tio fontas el
sinsekvo da malfortaj kaj malutilaj baraktoj supervenki kutiman
trepidadon—troabundan nervozan maltrankvilon

2025-11-11 10:25:29,374 - root - ERROR - [pipeline.py:114] - Pipeline failed with error: Ne povis...
Traceback (most recent call last):
  File "/home/marc/klareco/klareco/pipeline.py", line 65, in run
    ast = parse_esperanto(processed_text)
  File "/home/marc/klareco/klareco/parser.py", line 137, in parse
    word_asts = [parse_word(w) for w in words]
  File "/home/marc/klareco/klareco/parser.py", line 118, in parse_word
    raise ValueError(f"Ne povis trovi validan radikon en '{original_word}'. Restaĵo: '{stem}'")
ValueError: Ne povis trovi validan radikon en 'En'. Restaĵo: ''

2025-11-11 10:25:29,374 - root - DEBUG - [run_integration_test.py:90] - Trace: {
  "trace_id": "1f29a099-2820-4894-9c08-3816a3447d84",
  "start_time": "2025-11-11T15:25:29.032112Z",
  "end_time": "2025-11-11T15:25:29.374342Z",
  "initial_query": "En la maniero de mia amiko...",
  "steps": [...]
}
```

**You can see:**
- Exact word that failed ("En")
- Exact code location (parser.py:118)
- What the parser tried to do (parse_word)
- What remained after stripping ("Restaĵo: ''")
- Complete call stack

**This eliminates guessing!**

## Your Question 3: "Will I get progress indicators on long-running processes?"

### ✅ YES - Progress logging built-in:

**For Integration Tests:**
```
Testing sentences: 5/45 (11%) - Sentence 5: Estis tiu... [ETA: 82s]
Testing sentences: 10/45 (22%) - Sentence 10: La hundo... [ETA: 71s]
Testing sentences: 15/45 (33%) - Sentence 15: Mi amas... [ETA: 60s]
...
Testing sentences: 45/45 (100%)
```

**Updates:**
- Every 10% progress
- Shows current item being processed
- Shows estimated time remaining (ETA)
- Shows X/Y (Z%) format

**For Future Model Training** (Phase 3+):

The `ProgressLogger` class is designed for this:

```python
from klareco.logging_config import ProgressLogger

# Training example:
progress = ProgressLogger(total=100, desc="Training epochs")
for epoch in range(100):
    loss = train_epoch(model, data)
    progress.update(1, item_desc=f"Epoch {epoch}, Loss: {loss:.4f}")
progress.close()
```

**Output in watch.sh:**
```
Training epochs: 10/100 (10%) - Epoch 10, Loss: 0.2341 [ETA: 450s]
Training epochs: 20/100 (20%) - Epoch 20, Loss: 0.1823 [ETA: 400s]
Training epochs: 30/100 (30%) - Epoch 30, Loss: 0.1456 [ETA: 350s]
...
```

**For Corpus Processing** (Phase 3):
```python
progress = ProgressLogger(total=len(corpus), desc="Parsing corpus")
for i, doc in enumerate(corpus):
    ast = parse(doc)
    progress.update(1, item_desc=f"Document {i}: {doc[:30]}...")
progress.close()
```

**Output:**
```
Parsing corpus: 1000/10000 (10%) - Document 1000: En la komenco estis... [ETA: 270s]
Parsing corpus: 2000/10000 (20%) - Document 2000: La rapida bruna... [ETA: 240s]
```

## How to Use:

**Terminal 1 (Run tests):**
```bash
# Normal mode
./run.sh

# Or custom with progress visible in logs
python scripts/run_integration_test.py --num-sentences 20

# Or debug mode with full context
python scripts/run_integration_test.py --num-sentences 20 --debug
```

**Terminal 2 (Watch logs):**
```bash
./watch.sh
```

## Key Features:

### 1. Dual Progress Tracking:
- **tqdm progress bar** → console (for visual feedback)
- **ProgressLogger** → klareco.log (for watch.sh visibility)
- No conflicts, both work simultaneously

### 2. Test-by-Test Results:
- See PASSED/FAILED immediately
- Not batched - real-time results

### 3. Context on Demand:
- Normal mode: Clean, summary info
- Debug mode: Full context, stack traces, AST dumps

### 4. Long-Running Process Support:
- Progress percentage
- ETA calculations
- Item descriptions
- Updates every 10% to avoid log spam

### 5. No Guessing for Debugging:
- **Exact inputs** that caused failures
- **Complete stack traces** with line numbers
- **Intermediate state** (AST, trace objects)
- **File locations** for every log entry

## What's Still Missing (Future):

- [ ] pytest integration (unit tests don't log to klareco.log yet)
  - Workaround: Run pytest with `-v` and redirect: `pytest -v > test_output.txt`

- [ ] Model training progress (Phase 3+)
  - But ProgressLogger is ready to use when you get there

## Summary:

**Your questions answered:**

1. ✅ **Test visibility**: Yes - see which tests run and results in real-time
2. ✅ **Debugging info**: Yes - debug mode gives full context (inputs, stack traces, AST state)
3. ✅ **Progress indicators**: Yes - X/Y (Z%) [ETA: Ns] for all long-running processes

**The system is production-ready for your workflow!**
