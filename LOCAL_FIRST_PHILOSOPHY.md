# Klareco: Local-First AI Philosophy

## Core Principle

**Klareco uses LOCAL MODELS as the primary approach.**

External LLMs (like Claude Code, GPT-4, etc.) are **STOPGAP measures** only, used temporarily when local features are not yet implemented. These are temporary fallbacks and should be clearly marked with ⚠️ warnings.

## Why Local-First?

### 1. **Privacy & Control**
- Your data stays on your machine
- No API keys required
- No external dependencies in production
- Complete control over model behavior

### 2. **Cost**
- No per-query fees
- No token limits
- No rate limiting
- Run unlimited queries for free

### 3. **Performance**
- No network latency
- Predictable response times
- Works offline
- Scales with your hardware

### 4. **Transparency**
- You know exactly how models work
- Full visibility into training data
- Ability to debug and improve
- No black-box external services

### 5. **Alignment with Klareco's Mission**
Klareco is a **neuro-symbolic AI system** that fundamentally reimagines language processing:
- **Symbolic processing** handles structure (AST parsing, grammar validation)
- **Neural models** handle semantics (GNN encoding, QA decoding)
- **LLMs are only for truly generative tasks** (creative synthesis, summarization)

Using external LLMs for everything defeats the purpose of the neuro-symbolic architecture.

## Current Status

### ✅ Fully Local Components

| Component | Model | Size | Status |
|-----------|-------|------|--------|
| **Parser** | Rule-based AST | N/A | ✅ 95.7% accuracy |
| **GNN Encoder** | Tree-LSTM | 512-dim | ✅ 98.7% accuracy |
| **Retriever** | FAISS + Hybrid | 20,985 sentences | ✅ Production ready |
| **QA Decoder** | 8-layer Transformer | 417MB, 15M params | ✅ Trained (10 epochs) |
| **Experts** | Math, Date, Grammar | Symbolic | ✅ Production ready |

### ⚠️ Temporary External Dependencies (Stopgaps)

| Component | External Service | Reason | Replacement Plan |
|-----------|-----------------|--------|------------------|
| **Answer Generation** | Claude Code LLM | Convenience | ✅ **REPLACED** with local QA Decoder |
| **Summarization** | None yet | Not implemented | Train local summarization model |
| **Translation (optional)** | Opus-MT | Convenience | Already local, just for display |

## Command Reference

### ✅ Local Commands (PREFERRED)

```bash
# Extractive retrieval (local GNN)
./query.sh "Kiu estas Frodo?"

# Generative answering (local QA Decoder) ⭐ PREFERRED
./ask.sh "Kiu estas Frodo?"
```

**Models used:**
- ✅ LOCAL: Esperanto AST parser
- ✅ LOCAL: Tree-LSTM GNN encoder
- ✅ LOCAL: FAISS index (20,985 sentences)
- ✅ LOCAL: QA Decoder (417MB, 15M parameters)

### ⚠️ External Commands (STOPGAP)

```bash
# Generative answering (external Claude Code LLM)
./ask-external.sh "Kiu estas Frodo?"  # ⚠️ Only use if local fails
```

**Models used:**
- ✅ LOCAL: Retrieval (GNN + FAISS)
- ⚠️ EXTERNAL: Claude Code LLM (interactive, temporary)

## Development Guidelines

### When Writing New Code

1. **Always prefer local models first**
   - Check if a local model exists before using external LLM
   - If local model exists, use it as default

2. **Mark external LLM usage clearly**
   - Add ⚠️ warnings in comments
   - Add "STOPGAP" or "TEMPORARY" labels
   - Document the plan to replace with local model

3. **Make local the default**
   - `./ask.sh` should use local model
   - `./ask-external.sh` should use external LLM
   - Never make external LLM the default command

4. **Document local alternatives**
   - Always mention the local alternative in warnings
   - Provide clear migration path

### Code Markers for External LLM Usage

```python
# ⚠️ EXTERNAL LLM - STOPGAP MEASURE ONLY
# This uses Claude Code LLM interactively.
# PREFER: Local QA Decoder (see scripts/query_with_local_model.py)

from klareco.claude_code_llm import create_claude_code_provider

provider = create_claude_code_provider()  # ⚠️ EXTERNAL LLM
```

```bash
#!/bin/bash
# ⚠️ EXTERNAL LLM - STOPGAP MEASURE ONLY
# PREFER: ./ask.sh (local QA Decoder)

python scripts/query_with_llm.py "$@"  # ⚠️ Uses external LLM
```

## User-Facing Guidelines

### What Users Should See

1. **Clear default behavior**
   ```bash
   ./ask.sh "Kiu estas Frodo?"  # Uses LOCAL model (no warnings)
   ```

2. **Explicit warnings for external LLM**
   ```bash
   ./ask-external.sh "Kiu estas Frodo?"
   ⚠️ WARNING: Using EXTERNAL Claude Code LLM (interactive)
   ⚠️ PREFER: ./ask.sh (local QA Decoder, fully automatic)
   ```

3. **Consistent messaging**
   - Always use ⚠️ emoji for warnings
   - Always say "STOPGAP" or "TEMPORARY"
   - Always point to local alternative

### Documentation Standards

All documentation should:
1. List local models first
2. Mark external LLMs with ⚠️
3. Explain the local-first philosophy
4. Provide migration guidance

## Future Vision

### Short-term (Next 3 months)
- ✅ QA Decoder for factual questions (DONE)
- 🔄 Summarization Expert (local model)
- 🔄 Better answer formatting with deparser

### Medium-term (6 months)
- Remove all external LLM dependencies
- Fully local multi-language support
- Self-improving models (learning loop)

### Long-term (1 year)
- Fully self-contained AI system
- No external dependencies
- Community-trained models
- Zero-cost operation

## Examples

### ✅ Good: Local-First Code

```python
#!/usr/bin/env python3
"""
Query with Local QA Decoder - Fully automatic answer generation

Uses your trained QA Decoder model (no external LLM needed, fully local).
"""

from klareco.models.qa_decoder import QADecoder
from klareco.rag.retriever import create_retriever

# All local models
qa_system = LocalQASystem(
    qa_model_path=Path('models/qa_decoder/best_model.pt'),  # ✅ LOCAL
    gnn_path=Path('models/tree_lstm/checkpoint_epoch_20.pt'),  # ✅ LOCAL
)

result = qa_system.answer_question(query)  # ✅ Fully automatic
```

### ⚠️ Bad: External LLM Without Warning

```python
#!/usr/bin/env python3
"""
Query with LLM - Generate answers
"""

from klareco.claude_code_llm import create_claude_code_provider

# ❌ No warning about external LLM
# ❌ No mention of local alternative
provider = create_claude_code_provider()
result = provider.generate(query)
```

### ✅ Acceptable: External LLM With Clear Warning

```python
#!/usr/bin/env python3
"""
⚠️ EXTERNAL LLM - STOPGAP MEASURE ONLY

PREFER: scripts/query_with_local_model.py (local QA Decoder)
"""

from klareco.claude_code_llm import create_claude_code_provider

# ⚠️ EXTERNAL LLM (temporary fallback)
# Use ./ask.sh (local) instead
provider = create_claude_code_provider()
result = provider.generate(query)
```

## Testing Philosophy

When testing:
1. **Test local models extensively**
   - Unit tests for all local components
   - Integration tests for end-to-end local pipeline
   - Performance benchmarks

2. **Mock external LLMs in tests**
   ```python
   from klareco.claude_code_llm import create_mock_provider

   mock_responses = {"Frodo": "Frodo estas hobito..."}
   provider = create_mock_provider(mock_responses)  # ✅ Local mock
   ```

3. **Never require external services in CI/CD**
   - All tests must pass offline
   - No API keys in test suite
   - Local models only

## Migration Guide

If you encounter code using external LLMs:

### Step 1: Identify Usage
```bash
grep -r "claude_code_llm" .
grep -r "ask-external" .
```

### Step 2: Check for Local Alternative
```bash
ls models/qa_decoder/  # Is there a trained model?
cat COMMAND_REFERENCE.md  # What's the local command?
```

### Step 3: Switch to Local
```bash
# Before (external)
./ask-external.sh "Kiu estas Frodo?"

# After (local)
./ask.sh "Kiu estas Frodo?"
```

### Step 4: Update Code
```python
# Before (external)
from klareco.claude_code_llm import create_claude_code_provider
provider = create_claude_code_provider()

# After (local)
from klareco.models.qa_decoder import QADecoder
qa_system = LocalQASystem(...)
```

## Summary

**Remember:**
- ✅ **LOCAL MODELS** are the primary approach
- ⚠️ **EXTERNAL LLMs** are stopgap measures only
- Always prefer `./ask.sh` (local) over `./ask-external.sh` (external)
- Mark all external LLM usage with ⚠️ warnings
- Point users to local alternatives

This philosophy ensures Klareco remains:
- **Private** - Your data stays local
- **Free** - No API costs
- **Fast** - No network latency
- **Transparent** - Full model visibility
- **Aligned** - True to neuro-symbolic architecture
