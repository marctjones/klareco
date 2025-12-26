# Codebase Cleanup Analysis

**Date**: 2025-12-23
**Context**: Aligning codebase with new POC plan (Month 1-2: Zero learned reasoning, Month 3-4: Add 20M reasoning core)

---

## Summary

The current codebase has **significant over-engineering** that doesn't align with the new POC plan. We built for a complex "Phase 5-7" agentic system when the POC requires only:
1. **Deterministic components**: Parser, deparser, retrieval
2. **Minimal learning**: Root embeddings (320K), optional reasoning core (20M)

**Recommendation**: Delete ~50% of current code, keep the proven deterministic core, defer all "agentic" components.

---

## What to KEEP (Essential POC Components)

### ✅ Deterministic Core (0 parameters)

| File | LOC | Status | Purpose |
|------|-----|--------|---------|
| `parser.py` | 1,051 | **KEEP** | Core deterministic parser (16 Esperanto rules) |
| `deparser.py` | 125 | **KEEP** | AST → text reconstruction |
| `canonicalizer.py` | 240 | **KEEP** | SUBJ/VERB/OBJ extraction for structural search |
| `structural_index.py` | ~80 | **KEEP** | Zero-parameter structural filtering |
| `proper_nouns.py` | ~200 | **KEEP** | Proper noun dictionary (Phase 5.1) |
| `semantic_signatures.py` | 211 | **KEEP** | Canonical slot signatures for retrieval |

**Total: ~1,900 LOC of proven deterministic code**

### ✅ Embeddings (320K parameters)

| File | LOC | Status | Purpose |
|------|-----|--------|---------|
| `embeddings/compositional.py` | 650+ | **KEEP** | Root + prefix + suffix embeddings |
| `embeddings/unknown_tracker.py` | 250+ | **KEEP** | Track unknown roots during parsing |
| `ast_to_graph.py` | 522 | **KEEP** | AST → PyG graph for GNN encoding |

**Total: ~1,400 LOC for compositional embeddings**

### ✅ Retrieval (15M parameters for neural reranker)

| File | LOC | Status | Purpose |
|------|-----|--------|---------|
| `rag/retriever.py` | 650+ | **KEEP** | Two-stage hybrid retrieval (structural + neural) |
| `models/tree_lstm.py` | 350+ | **KEEP** | Tree-LSTM encoder for AST embeddings |
| `semantic_search.py` | 226 | **KEEP** | Semantic similarity search utilities |

**Total: ~1,200 LOC for retrieval system**

### ✅ Support Infrastructure

| File | LOC | Status | Purpose |
|------|-----|--------|---------|
| `corpus_manager.py` | 444 | **KEEP** | Corpus building and validation |
| `dataloader.py` | 216 | **KEEP** | Training data loading |
| `lang_id.py` | ~80 | **KEEP** | Language detection (front door) |
| `translator.py` | ~100 | **KEEP** | EN → EO translation (front door) |
| `logging_config.py` | ~150 | **KEEP** | Logging setup |
| `trace.py` | ~150 | **KEEP** | Execution tracing for debugging |

**Total: ~1,100 LOC for support**

---

## What to DELETE/DEPRECATE (Over-engineered)

### ❌ Agentic System (Not Needed for POC)

These files implement a complex "Phase 5-7" agentic architecture that's **premature** for the POC:

| File | LOC | Reason to Delete |
|------|-----|------------------|
| `blueprint.py` | 354 | Multi-step planning system - not needed for POC |
| `gating_network.py` | 495 | Neural intent classifier - POC uses simple retrieval |
| `orchestrator.py` | ~160 | Expert routing system - over-engineered |
| `execution_loop.py` | 235 | Task execution loop - not needed |
| `learning_loop.py` | 339 | Self-improvement loop - future feature |
| `memory.py` | 515 | Persistent memory system - not POC requirement |
| `goals.py` | 398 | Goal tracking - agentic feature |
| `values.py` | 399 | Value alignment - agentic feature |
| `emergent_intent_analyzer.py` | 471 | Emergent intent detection - over-engineered |
| `trace_analyzer.py` | 423 | Trace analysis for learning - future feature |
| `pr_generator.py` | 470 | PR generation from traces - specific tool |
| `environment.py` | ~200 | Agentic environment - not needed |

**Total: ~4,500 LOC to deprecate**

**Justification**: The POC goal is "answer 50 questions with zero learned reasoning." We don't need:
- Multi-step planning (blueprint)
- Neural intent classification (gating network)
- Expert orchestration (orchestrator)
- Self-improvement loops (learning_loop)
- Goal/value systems (goals, values)

These are **excellent features for Phase 5-7**, but they're blocking POC clarity.

### ❌ Over-Engineered Experts

| File | LOC | Reason to Delete |
|------|-----|------------------|
| `experts/*` (if exists) | ? | POC needs only simple extractive answering |

### ❌ Redundant or Unused Tools

| File | LOC | Reason to Delete |
|------|-----|------------------|
| `llm_provider.py` | 469 | Only needed if using LLM for summarization (not POC core) |
| `front_door.py` | ~100 | Minimal - could merge into main CLI |
| `responder.py` | ~100 | Generic responder - superseded by extractive expert |
| `safety.py` | ~150 | Safety checks - add later if needed |
| `intent_classifier.py` | ~100 | Superseded by gating_network (both to delete) |
| `cli.py` | 323 | Overly complex CLI - simplify |
| `tools/*` (if directory) | ? | Tool abstractions not needed for POC |

**Total: ~1,200 LOC to deprecate**

### ❌ Old Model Architectures

| File | LOC | Reason to Delete |
|------|-----|------------------|
| `models/generator.py` | ~200 | AST-to-text generation - deparser does this |
| `models/graph2seq.py` | ~200 | Graph-to-sequence - not used |

**Total: ~400 LOC to deprecate**

---

## What to SIMPLIFY (Keep but Reduce Complexity)

### 🔧 Simplify CLI

**Current**: `cli.py` (323 LOC) + `__main__.py` + `cli/corpus.py`
**Target**: Single `cli.py` (~150 LOC) with just:
- `parse` - Parse Esperanto sentence
- `query` - Query corpus with retrieval
- `corpus validate/add/list` - Corpus management

**Delete**: Complex orchestration, tracing, interactive modes

### 🔧 Simplify Retriever

**Current**: `rag/retriever.py` (650+ LOC) with Tree-LSTM + baseline modes
**Target**: Keep Tree-LSTM mode, remove baseline mode, simplify API

### 🔧 Merge Small Files

- Merge `front_door.py` → `cli.py` (language detection is 1 function)
- Merge `responder.py` → remove (superseded)
- Merge `intent_classifier.py` → remove (not using neural classification in POC)

---

## Reorganized File Structure

### After Cleanup:

```
klareco/
├── parser.py              (1,051 LOC) - Deterministic parser
├── deparser.py            (125 LOC)   - AST → text
├── canonicalizer.py       (240 LOC)   - Slot extraction
├── structural_index.py    (80 LOC)    - Structural filtering
├── proper_nouns.py        (200 LOC)   - Proper noun dict
├── semantic_signatures.py (211 LOC)   - Signature matching
├── corpus_manager.py      (444 LOC)   - Corpus building
├── dataloader.py          (216 LOC)   - Training data
├── lang_id.py             (80 LOC)    - Language detection
├── translator.py          (100 LOC)   - EN → EO translation
├── logging_config.py      (150 LOC)   - Logging
├── trace.py               (150 LOC)   - Execution tracing
├── cli.py                 (150 LOC)   - Simplified CLI
├── embeddings/
│   ├── compositional.py   (650 LOC)   - Compositional embeddings
│   ├── unknown_tracker.py (250 LOC)   - Unknown root tracking
│   └── __init__.py
├── models/
│   └── tree_lstm.py       (350 LOC)   - Tree-LSTM encoder
├── rag/
│   └── retriever.py       (400 LOC)   - Simplified retriever
└── ast_to_graph.py        (522 LOC)   - AST → graph conversion

TOTAL: ~5,500 LOC (down from ~9,300 LOC = 40% reduction)
```

### Deprecated (move to `deprecated/`):

```
deprecated/
├── agentic/               # Phase 5-7 features
│   ├── blueprint.py
│   ├── gating_network.py
│   ├── orchestrator.py
│   ├── execution_loop.py
│   ├── learning_loop.py
│   ├── memory.py
│   ├── goals.py
│   ├── values.py
│   ├── emergent_intent_analyzer.py
│   ├── trace_analyzer.py
│   └── pr_generator.py
├── models/
│   ├── generator.py
│   └── graph2seq.py
├── llm_provider.py        # Only if not using LLM for summarization
├── front_door.py          # Merged into cli.py
├── responder.py           # Superseded
├── safety.py              # Add back later
└── intent_classifier.py   # Superseded
```

---

## Migration Plan

### Phase 1: Create Deprecated Directory (Safe)

```bash
mkdir deprecated
mkdir deprecated/agentic
mkdir deprecated/models

# Move agentic components
git mv klareco/blueprint.py deprecated/agentic/
git mv klareco/gating_network.py deprecated/agentic/
git mv klareco/orchestrator.py deprecated/agentic/
git mv klareco/execution_loop.py deprecated/agentic/
git mv klareco/learning_loop.py deprecated/agentic/
git mv klareco/memory.py deprecated/agentic/
git mv klareco/goals.py deprecated/agentic/
git mv klareco/values.py deprecated/agentic/
git mv klareco/emergent_intent_analyzer.py deprecated/agentic/
git mv klareco/trace_analyzer.py deprecated/agentic/
git mv klareco/pr_generator.py deprecated/agentic/
git mv klareco/environment.py deprecated/agentic/

# Move old models
git mv klareco/models/generator.py deprecated/models/
git mv klareco/models/graph2seq.py deprecated/models/

# Move redundant files
git mv klareco/llm_provider.py deprecated/  # if not using LLM
git mv klareco/front_door.py deprecated/
git mv klareco/responder.py deprecated/
git mv klareco/safety.py deprecated/
git mv klareco/intent_classifier.py deprecated/
```

### Phase 2: Simplify Remaining Files

1. **Simplify `cli.py`**: Remove orchestration, keep only parse/query/corpus
2. **Simplify `rag/retriever.py`**: Remove baseline mode, keep Tree-LSTM only
3. **Update imports**: Remove references to deprecated modules

### Phase 3: Update Tests

```bash
# Move deprecated tests
mkdir tests/deprecated
git mv tests/test_blueprint.py tests/deprecated/
git mv tests/test_gating_network.py tests/deprecated/
git mv tests/test_learning_loop.py tests/deprecated/
git mv tests/test_memory.py tests/deprecated/
git mv tests/test_goals_values.py tests/deprecated/
```

### Phase 4: Update Documentation

- Update README: Remove references to agentic features
- Update CLAUDE.md: Focus on deterministic + retrieval POC
- Create `deprecated/README.md`: Explain what's there and why

---

## Benefits of Cleanup

### 1. **Clarity**

- **Before**: 9,300 LOC, unclear which parts are POC-critical
- **After**: 5,500 LOC, every file serves POC goals

### 2. **Maintainability**

- **Before**: 20+ interdependent modules (blueprint → orchestrator → gating → memory → ...)
- **After**: 10 core modules with clear dependencies

### 3. **Alignment with POC**

**POC Goal**: "Answer 50 questions using ONLY deterministic + retrieval (zero learned reasoning)"

**Before**: Codebase includes multi-step planning, neural intent classification, self-improvement loops → confusing
**After**: Codebase has parser → retriever → answer extraction → clear

### 4. **Faster Development**

- **Before**: Changes require understanding complex orchestration
- **After**: Changes are localized (parser, retriever, or embeddings)

### 5. **Easier Testing**

- **Before**: Tests require complex setup (memory, blueprints, orchestrator)
- **After**: Tests are simple (parse → retrieve → extract)

---

## Recommended Next Steps

### Option A: Aggressive Cleanup (Recommended)

1. Move all agentic files to `deprecated/` (1 hour)
2. Simplify `cli.py` to basic parse/query/corpus (1 hour)
3. Simplify `rag/retriever.py` to remove baseline mode (30 min)
4. Update imports and tests (1 hour)
5. Commit: "refactor: Deprecate agentic components, focus on POC core"

**Result**: Clean codebase aligned with POC, all features preserved in `deprecated/` for future use

### Option B: Gradual Deprecation

1. Create `deprecated/` directory but don't move files yet
2. Add `# DEPRECATED: Not needed for POC` comments to files
3. Create `CODEBASE_ROADMAP.md` explaining what's POC vs future
4. Clean up over multiple sessions

**Result**: Slower but safer, good for understanding dependencies first

### Option C: Keep Everything (Not Recommended)

Keep all files as-is, just document what's POC-critical

**Risk**: Confusion about what's actually needed, harder to onboard contributors

---

## Files Summary

### KEEP (POC-Critical): 15 files, ~5,500 LOC
- parser.py, deparser.py, canonicalizer.py, structural_index.py
- proper_nouns.py, semantic_signatures.py, semantic_search.py
- embeddings/compositional.py, embeddings/unknown_tracker.py
- ast_to_graph.py, models/tree_lstm.py, rag/retriever.py
- corpus_manager.py, dataloader.py, cli.py (simplified)
- logging_config.py, trace.py, lang_id.py, translator.py

### DEPRECATE (Phase 5-7): 17 files, ~6,200 LOC
- All agentic components (blueprint, gating, orchestrator, loops, memory, goals, values)
- Old models (generator, graph2seq)
- Redundant utils (llm_provider, front_door, responder, safety, intent_classifier)

### DELETE (Truly Unused): TBD after import analysis
- `tools/` directory (if exists and unused)
- `experts/` directory (if superseded by simple extraction)

---

## Conclusion

**The current codebase is ~40-50% over-engineered for the POC.**

We have excellent **deterministic core components** (parser, deparser, canonicalizer) and **proven retrieval** (two-stage hybrid, compositional embeddings), but we're carrying a lot of **premature agentic architecture** (blueprints, orchestration, self-improvement).

**Recommendation**:
1. Move agentic components to `deprecated/` (preserves work for Phase 5-7)
2. Simplify remaining code to focus on POC goals
3. Document clearly what's POC vs future in README/CLAUDE

This will make the POC **much clearer** while preserving all the good work for future phases.
