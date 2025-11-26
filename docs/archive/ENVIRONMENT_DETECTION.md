# Environment Detection - Claude Code Web vs CLI

**Date**: 2025-11-13
**Status**: ✅ Implemented

---

## Overview

Klareco now automatically detects whether it's running in:
1. **Claude Code Web** (browser-based) - use Claude as LLM, NO training
2. **Claude Code CLI** (local) - use Claude as LLM, CAN train models
3. **Standalone** - regular Python, use APIs if configured

---

## Why This Matters

### Claude Code Web (Browser)
- ❌ **NO model training** - limited compute, may crash browser
- ✅ **Use Claude (me) as LLM** - no API keys needed
- ✅ **Use pre-trained models** - load existing checkpoints
- ✅ **Testing and development** - perfect for quick experiments

### Claude Code CLI (Local)
- ✅ **Can train models** - full compute resources
- ✅ **Use Claude (me) as LLM** - no API keys needed
- ✅ **Full capabilities** - everything works

### Standalone
- ✅ **Full capabilities** - train, use APIs, etc.
- ⏳ **Requires API keys** - for LLM features (if not using Claude Code)

---

## Automatic Detection

The system automatically detects your environment:

```python
from klareco.environment import get_environment_info

info = get_environment_info()
print(f"Environment: {info['environment']}")
print(f"Can train: {info['can_train']}")
print(f"Should use Claude LLM: {info['should_use_claude_llm']}")
```

### Detection Logic

**Web Environment Indicators**:
- No SSH_CONNECTION, DISPLAY, TMUX, STY environment variables
- Limited TERM value (not xterm, screen, etc.)
- No GPU/CUDA available
- → Detected as Claude Code Web

**CLI Environment Indicators**:
- Has SSH_CONNECTION, DISPLAY, TMUX, or similar
- Full terminal (xterm, xterm-256color, etc.)
- GPU available (CUDA)
- → Detected as Claude Code CLI

---

## LLM Provider Auto-Selection

### In Claude Code Web or CLI

```python
from klareco.llm_provider import get_llm_provider

# Automatically detects environment and uses Claude Code as LLM
provider = get_llm_provider()
# Provider type: claude_code_web or claude_code_cli
# No API keys needed!
```

### With API Keys (Any Environment)

```bash
# Set API key
export ANTHROPIC_API_KEY="your-key"

# Or
export OPENAI_API_KEY="your-key"
```

```python
# Will automatically use the API instead of Claude Code
provider = get_llm_provider()
# Provider type: anthropic_api or openai_api
```

---

## Training Protection

### Automatic Warning in Web

GNN training scripts now check environment:

```python
from klareco.environment import warn_if_web_training

# At start of training script
warn_if_web_training()  # Warns and prompts if in web
```

**Example output in Web**:
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

### Require CLI for Operations

```python
from klareco.environment import require_cli

# At start of compute-intensive operation
require_cli("GNN model training")
# Raises RuntimeError if in web environment
```

---

## Usage Examples

### Check Environment

```python
from klareco.environment import (
    is_web_environment,
    is_cli_environment,
    is_claude_code
)

if is_web_environment():
    print("Running in Claude Code Web - use pre-trained models")

if is_cli_environment():
    print("Running in Claude Code CLI - can train models")

if is_claude_code():
    print("Running in Claude Code - use Claude as LLM")
```

### Conditional Training

```python
from klareco.environment import get_environment_info

env_info = get_environment_info()

if env_info['can_train']:
    # Train model
    train_gnn_model()
else:
    # Use pre-trained
    print("⚠️  Cannot train in this environment")
    print("Loading pre-trained model instead...")
    load_pretrained_model()
```

### Conditional LLM

```python
from klareco.environment import get_environment_info
from klareco.llm_provider import get_llm_provider

env_info = get_environment_info()

if env_info['should_use_claude_llm']:
    print("✅ Using Claude Code as LLM (no API keys needed)")
    provider = get_llm_provider()
else:
    print("ℹ️  Using API-based LLM")
    provider = get_llm_provider()
```

---

## Files

### New Files
- `klareco/environment.py` - Environment detection module
  - `detect_environment()` - Detect Web/CLI/Standalone
  - `is_web_environment()` - Check if Web
  - `is_cli_environment()` - Check if CLI
  - `warn_if_web_training()` - Warn before training in Web
  - `require_cli()` - Require CLI environment
  - `get_environment_info()` - Get detailed info

### Modified Files
- `klareco/llm_provider.py` - Enhanced detection
  - Now distinguishes Web vs CLI
  - Auto-selects Claude LLM in both environments
  - Better heuristics for detection

- `scripts/train_tree_lstm.py` - Added warning
  - Warns if training in Web environment
  - Prompts user before continuing

---

## Decision Matrix

| Operation | Web | CLI | Standalone |
|-----------|-----|-----|------------|
| Use Claude as LLM | ✅ | ✅ | ❌ (need API) |
| Train GNN models | ⚠️ Not recommended | ✅ | ✅ |
| Load pre-trained models | ✅ | ✅ | ✅ |
| RAG retrieval | ✅ | ✅ | ✅ |
| Symbolic experts | ✅ | ✅ | ✅ |
| Full pipeline | ✅ | ✅ | ✅ (with API) |
| Heavy compute | ❌ | ✅ | ✅ |

---

## Current Training Status

**GNN Training (Started in CLI)**:
- Environment: Claude Code CLI (detected properly)
- Training: Epoch 5/20 (25% complete)
- Status: Running in background
- Will complete in ~3-4 hours

**If this was Web**:
- Would have been warned
- Could have used pre-trained checkpoint
- Or continued with small dataset test

---

## Testing

```bash
# Test environment detection
python -m klareco.environment

# Output will show:
# - Environment type (web/cli/standalone)
# - Capabilities (can_train, should_use_claude_llm)
# - Platform info
```

---

## Benefits

1. **Safety**: Prevents accidental heavy compute in browser
2. **Clarity**: Clear warnings about environment limitations
3. **Flexibility**: Works in Web and CLI with appropriate behavior
4. **No Configuration**: Auto-detects everything
5. **User-Friendly**: Clear messages and prompts

---

## Summary

- ✅ **Web**: Use Claude LLM, pre-trained models, NO training
- ✅ **CLI**: Use Claude LLM, can train models, full capabilities
- ✅ **Auto-detection**: No manual configuration needed
- ✅ **Warnings**: Clear messages before heavy operations
- ✅ **Implemented**: All detection and warnings working

**Klareco now safely adapts to your environment!** 🎉
