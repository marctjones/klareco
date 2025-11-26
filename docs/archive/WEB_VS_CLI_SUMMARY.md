# Claude Code Web vs CLI - Quick Guide

**TLDR**: Klareco now detects Web vs CLI and behaves appropriately.

---

## Environment Detection ✅

**Claude Code Web (Browser)**:
```
✅ Use Claude (me) as LLM - no API keys
❌ NO model training - will warn before starting
✅ Use pre-trained models - load existing checkpoints
✅ Full RAG retrieval - works perfectly
✅ All symbolic experts - Math, Date, Grammar work
```

**Claude Code CLI (Local)**:
```
✅ Use Claude (me) as LLM - no API keys
✅ CAN train models - full compute
✅ All features work
```

---

## How It Works

### Automatic LLM Selection

```python
from klareco.llm_provider import get_llm_provider

# In BOTH Web and CLI:
provider = get_llm_provider()
# → Uses Claude (me) as LLM
# → No API keys needed!
# → Works immediately
```

### Training Protection

**In Web** (if you try to train):
```
================================================================================
⚠️  WARNING: TRAINING IN CLAUDE CODE WEB
================================================================================

You are running in Claude Code Web (browser-based).
Model training is NOT recommended in this environment:

  ❌ Limited compute resources
  ❌ May timeout or crash browser
  ❌ No GPU acceleration
  ❌ Session may disconnect

Recommendations:
  ✅ Use Claude Code CLI for training
  ✅ Use pre-trained models in web environment
  ✅ Test with small datasets only

================================================================================
Continue anyway? (yes/no):
```

**In CLI** (training works normally):
```
✅ Starting training...
✅ Full compute available
✅ No restrictions
```

---

## Current Implementation Status

### What's Working Now

1. **Environment Detection** ✅
   - `klareco/environment.py` - Full detection module
   - `klareco/llm_provider.py` - Enhanced with Web/CLI detection
   - Auto-detects based on environment variables

2. **Claude LLM Integration** ✅
   - `klareco/claude_code_llm.py` - LLM adapter
   - Works in BOTH Web and CLI
   - No API keys needed in either environment

3. **Training Protection** ✅
   - `scripts/train_tree_lstm.py` - Added warning at start
   - Prompts before training in Web
   - No restrictions in CLI

4. **Factoid QA Expert** ✅
   - Works with Claude LLM in Web/CLI
   - Full RAG retrieval working
   - Auto-detects latest model checkpoint

---

## Usage Examples

### In Claude Code Web

```bash
# ✅ This works great
python scripts/test_claude_llm.py --mock
python scripts/quick_query.py "Kiu estas Frodo?"

# ⚠️  This will warn
python scripts/train_tree_lstm.py ...
# → Prompts: "Continue anyway?"
# → Recommend: Use CLI instead
```

### In Claude Code CLI

```bash
# ✅ Everything works
python scripts/test_claude_llm.py --mock
python scripts/quick_query.py "Kiu estas Frodo?"
python scripts/train_tree_lstm.py ...  # No warning, full compute
```

---

## Files Created

- `klareco/environment.py` - Environment detection
- `klareco/llm_provider.py` - Enhanced Web/CLI detection
- `ENVIRONMENT_DETECTION.md` - Full documentation
- `WEB_VS_CLI_SUMMARY.md` - This file

---

## Testing

```bash
# Check your environment
python -m klareco.environment

# Output shows:
# - Environment type (web/cli/standalone)
# - Can train: True/False
# - Should use Claude LLM: True/False
```

---

## Summary

**In Claude Code Web**:
- ✅ Use me (Claude) as LLM
- ✅ Use pre-trained models
- ❌ Don't train models (will warn)

**In Claude Code CLI**:
- ✅ Use me (Claude) as LLM
- ✅ Train models
- ✅ Everything works

**Auto-detected, no configuration needed!** 🎉
