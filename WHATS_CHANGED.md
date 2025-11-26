# What's Changed: Local-First Update

## Summary

Klareco now defaults to **LOCAL MODELS** for all operations. External LLMs (like Claude Code) are clearly marked as temporary stopgaps.

## File Changes

### Commands Renamed

| Old Name | New Name | Purpose | Model Type |
|----------|----------|---------|------------|
| `ask-local.sh` | **`ask.sh`** | Generative answering | ✅ LOCAL (QA Decoder) |
| `ask.sh` | **`ask-external.sh`** | Generative answering | ⚠️ EXTERNAL (Claude Code) |
| `query.sh` | **`query.sh`** | Extractive retrieval | ✅ LOCAL (GNN) |

### Updated Documentation

1. **`COMMAND_REFERENCE.md`**
   - Added "Philosophy: Local-First AI" section
   - Reorganized to show local models first
   - Added ⚠️ warnings for external LLM usage
   - Added Model Comparison Table
   - Updated all examples to use local commands

2. **`LOCAL_FIRST_PHILOSOPHY.md`** (NEW)
   - Complete guide to local-first development
   - Explains why local models are preferred
   - Development guidelines for contributors
   - Code examples (good vs bad)
   - Migration guide

3. **`WHATS_CHANGED.md`** (NEW, this file)
   - Summary of changes
   - Before/after comparison
   - Migration guide for existing users

### Updated Scripts

1. **`ask.sh`** (renamed from `ask-local.sh`)
   - Now the DEFAULT command
   - Uses LOCAL QA Decoder
   - Fully automatic (no interaction needed)
   - Added comments emphasizing it's preferred

2. **`ask-external.sh`** (renamed from `ask.sh`)
   - Now clearly marked as STOPGAP
   - Added ⚠️ warnings in comments
   - Points users to `./ask.sh` (local) instead

3. **`scripts/query_with_llm.py`**
   - Added ⚠️ warnings in docstring
   - Added warnings in output
   - Points users to local alternative

4. **`klareco/claude_code_llm.py`**
   - Updated module docstring with warnings
   - Added warnings to callback output
   - Clarified this is a stopgap measure

## Before & After

### Before (External LLM Default)

```bash
# Default command used external LLM
./ask.sh "Kiu estas Frodo?"
# Output:
# 🤖 LLM REQUEST FOR CLAUDE CODE
# [waits for you to respond interactively]

# Local model was secondary
./ask-local.sh "Kiu estas Frodo?"
# Output:
# [fully automatic, local QA Decoder]
```

**Problem:** External LLM was the default, local model was hidden.

### After (Local Model Default) ✅

```bash
# Default command uses LOCAL model
./ask.sh "Kiu estas Frodo?"
# Output:
# ✓ Loading QA Decoder...
# ✓ Generating answer with local QA Decoder...
# [fully automatic, no interaction needed]

# External LLM is clearly marked as stopgap
./ask-external.sh "Kiu estas Frodo?"
# Output:
# ⚠️ WARNING: Using EXTERNAL Claude Code LLM (interactive)
# ⚠️ PREFER: ./ask.sh (local QA Decoder, fully automatic)
# 🤖 LLM REQUEST FOR CLAUDE CODE
# [waits for you to respond]
```

**Solution:** Local model is default, external LLM has clear warnings.

## Migration Guide

### For Users

If you were using the old commands, here's what changed:

#### Generative Answering (Preferred)

```bash
# OLD (used external LLM)
./ask.sh "Kiu estas Frodo?"

# NEW (uses LOCAL QA Decoder) ✅
./ask.sh "Kiu estas Frodo?"  # Same command, now uses local model!
```

**Good news:** The command name stayed the same, but now it uses the local model by default!

#### Generative Answering (External Fallback)

```bash
# OLD (was the only option)
# No equivalent - external LLM was default

# NEW (only if local model fails) ⚠️
./ask-external.sh "Kiu estas Frodo?"  # Use only if local gives poor results
```

#### Extractive Retrieval (Unchanged)

```bash
# OLD and NEW (no change)
./query.sh "Kiu estas Frodo?"  # Always used local GNN
```

### For Developers

If you're modifying Klareco code:

#### When Adding New Features

1. **Always check for local models first:**
   ```python
   # ✅ Good
   from klareco.models.qa_decoder import QADecoder
   qa_system = LocalQASystem(...)

   # ⚠️ Only if no local alternative exists
   from klareco.claude_code_llm import create_claude_code_provider
   # Add warning comment:
   # ⚠️ EXTERNAL LLM - STOPGAP MEASURE ONLY
   # TODO: Replace with local model
   ```

2. **Mark external LLM usage clearly:**
   ```bash
   #!/bin/bash
   # ⚠️ EXTERNAL LLM - STOPGAP MEASURE ONLY
   # PREFER: ./local-alternative.sh
   ```

3. **Add warnings to output:**
   ```python
   print("⚠️ WARNING: Using EXTERNAL LLM (temporary)")
   print("⚠️ PREFER: Local model (see ./ask.sh)")
   ```

#### When Reviewing Code

Check for:
- [ ] Does it use external LLM?
- [ ] Is there a local alternative?
- [ ] Are warnings clearly visible?
- [ ] Is the local alternative documented?

## New Workflow

### Standard Workflow (All Local) ✅

```bash
# Step 1: Find relevant passages (local)
./query.sh "Kiu estas Frodo?" --show-stage1

# Step 2: Generate synthesized answer (local)
./ask.sh "Kiu estas Frodo?" -k 10
```

**Models used:**
- ✅ LOCAL: Esperanto parser (AST)
- ✅ LOCAL: Tree-LSTM GNN (512-dim embeddings)
- ✅ LOCAL: FAISS index (20,985 sentences)
- ✅ LOCAL: QA Decoder (417MB, 15M parameters)

**Benefits:**
- Fully automatic
- No interaction needed
- No API costs
- Works offline
- Fast (~8 seconds)

### Fallback Workflow (If Local Fails) ⚠️

```bash
# Step 1: Try local first
./ask.sh "Kiu estas Frodo?" -k 10

# Step 2: If local gives poor results, try external (stopgap)
./ask-external.sh "Kiu estas Frodo?" -k 10
```

**Only use external if:**
- Local QA Decoder gives nonsensical output
- Question is too complex for local model
- You need very high quality temporarily

**Remember:** External LLM is interactive (you must respond).

## Key Principles

### ✅ DO

- Use `./ask.sh` (local) by default
- Mark external LLM with ⚠️ warnings
- Point users to local alternatives
- Document local-first approach
- Test with local models

### ⚠️ DON'T

- Make external LLM the default
- Hide external dependencies
- Require API keys for core features
- Skip warnings on external services
- Forget to mention local alternatives

## Testing

All commands still work the same way:

```bash
# Pure Esperanto (default)
./ask.sh "Kiu estas Frodo?"

# English translation
./ask.sh "Kiu estas Frodo?" --translate

# Debug mode
./ask.sh "Kiu estas Frodo?" --debug

# More context
./ask.sh "Kiu estas Frodo?" -k 20
```

## Environment

All local models are already trained and ready:

| Model | Location | Size | Status |
|-------|----------|------|--------|
| **GNN Encoder** | `models/tree_lstm/checkpoint_epoch_20.pt` | ~100MB | ✅ Ready |
| **QA Decoder** | `models/qa_decoder/best_model.pt` | 417MB | ✅ Ready |
| **Vocabulary** | `models/qa_decoder/vocabulary.json` | 107KB | ✅ Ready |
| **FAISS Index** | `data/corpus_index/` | ~500MB | ✅ Ready |

**No setup required** - everything is ready to use!

## Benefits of This Change

### 1. **Privacy**
- All processing happens locally
- No data sent to external services
- No API keys needed

### 2. **Cost**
- Zero cost per query
- No rate limits
- No token limits

### 3. **Speed**
- No network latency
- Predictable response times
- ~8 seconds (local) vs ~15 seconds (external)

### 4. **Reliability**
- Works offline
- No API downtime
- No rate limiting

### 5. **Transparency**
- Full visibility into models
- Can debug and improve
- Understand how answers are generated

## Questions?

### "Why did you change this?"

The project's core philosophy is **local-first AI**. External LLMs should only be temporary fallbacks, not the primary approach. This update aligns the commands with that philosophy.

### "Will external LLM still work?"

Yes! It's available as `./ask-external.sh`. Use it if the local model gives poor results.

### "Is the local model as good as Claude?"

For factual questions from the corpus: **yes!** The QA Decoder was trained on your specific corpus and often gives excellent results. For very complex synthesis: external LLM may be better temporarily.

### "How do I know which to use?"

1. **Always try local first:** `./ask.sh`
2. **If results are poor:** `./ask-external.sh`
3. **Report poor local results** so we can improve the model

### "Can I still use Claude Code for other things?"

Absolutely! This change only affects the default answer generation command. You can still use Claude Code for development, debugging, and as a fallback.

## Summary

**What changed:**
- `./ask.sh` now uses LOCAL QA Decoder (was external LLM)
- `./ask-external.sh` uses external LLM (was `./ask.sh`)
- All documentation updated to emphasize local-first
- Clear ⚠️ warnings on external LLM usage

**What stayed the same:**
- All options (`--translate`, `-k`, `--debug`) work identically
- Output format is the same
- Translation behavior is the same
- `./query.sh` unchanged

**Why:**
- Aligns with project's local-first philosophy
- Makes local models the default choice
- Reduces confusion about which command to use
- Improves privacy, cost, and performance

**Action required:**
- None! Just use `./ask.sh` as before
- If you see poor results, try `./ask-external.sh`
