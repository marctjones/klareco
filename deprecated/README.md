# Deprecated Components

**Last Updated**: 2025-12-23

This directory contains components that are **not needed for the POC** but represent valuable work for future phases (Phase 5-7).

---

## Why These Were Deprecated

The POC plan focuses on:
1. **Month 1-2**: Answer 50 questions using ONLY deterministic processing + retrieval (zero learned reasoning)
2. **Month 3-4**: Add minimal 20M param reasoning core, measure improvement

The components in this directory implement a more complex "agentic" architecture (multi-step planning, orchestration, self-improvement) that's **premature for proving the core thesis**.

**Core thesis**: Traditional LLMs waste 100M+ parameters learning grammar. By making grammar deterministic through Esperanto, a small reasoning core can match larger models.

---

## Deprecated Components

### Agentic System (`agentic/`)

**Phase**: 5-7 (Future)
**Purpose**: Multi-agent orchestration, planning, and self-improvement

| File | LOC | Purpose | When to Restore |
|------|-----|---------|-----------------|
| `blueprint.py` | 354 | Multi-step planning system | Phase 5: Complex query decomposition |
| `gating_network.py` | 495 | Neural intent classification | Phase 5: When symbolic routing insufficient |
| `orchestrator.py` | 160 | Expert routing and coordination | Phase 5: Multi-expert system |
| `execution_loop.py` | 235 | Task execution loop | Phase 5: Autonomous task completion |
| `learning_loop.py` | 339 | Self-improvement from traces | Phase 6: Online learning |
| `memory.py` | 515 | Persistent memory system | Phase 5: Context across sessions |
| `goals.py` | 398 | Goal tracking | Phase 7: Goal-directed behavior |
| `values.py` | 399 | Value alignment | Phase 7: Ethical constraints |
| `emergent_intent_analyzer.py` | 471 | Emergent intent detection | Phase 6: Advanced intent understanding |
| `trace_analyzer.py` | 423 | Trace analysis for learning | Phase 6: Learning from execution |
| `pr_generator.py` | 470 | PR generation tool | Utility (restore if needed) |
| `environment.py` | ~200 | Agentic environment | Phase 7: Agent interactions |

**Total**: ~4,500 LOC

**Why valuable**: These implement a sophisticated agentic architecture. Excellent for Phase 5-7 when we need:
- Complex query decomposition
- Multi-step reasoning
- Self-improvement from execution traces
- Context persistence across sessions

**Why not POC**: The POC goal is simpler: prove that deterministic grammar + small reasoning core beats traditional LLMs. We don't need planning, orchestration, or self-improvement to prove this.

### Old Model Architectures (`models/`)

| File | LOC | Purpose | Status |
|------|-----|---------|--------|
| `generator.py` | ~200 | AST-to-text generation | Superseded by `deparser.py` |
| `graph2seq.py` | ~200 | Graph-to-sequence model | Not currently used |

**When to restore**: If we need learned AST-to-text generation (beyond deterministic deparser)

### Redundant Utilities

| File | LOC | Purpose | Replacement |
|------|-----|---------|-------------|
| `llm_provider.py` | 469 | LLM abstraction (Claude/OpenAI) | Only if adding summarization/creativity |
| `front_door.py` | ~100 | Language detection wrapper | Merged into `cli.py` |
| `responder.py` | ~100 | Generic responder | Superseded by extractive expert |
| `safety.py` | ~150 | Safety checks | Add back when needed |
| `intent_classifier.py` | ~100 | Old intent classifier | Superseded by gating_network |

**Total**: ~1,000 LOC

### Deprecated Tests (`tests/`)

| File | Tests |
|------|-------|
| `test_blueprint.py` | Blueprint system tests |
| `test_gating_network.py` | Gating network tests |
| `test_learning_loop.py` | Learning loop tests |
| `test_memory.py` | Memory system tests |
| `test_goals_values.py` | Goals and values tests |
| `test_front_door.py` | Front door tests |
| `test_safety.py` | Safety tests |

---

## When to Restore

### Phase 5: AST Enrichment & Multi-Step Reasoning

**Restore**:
- `blueprint.py` - Multi-step planning
- `orchestrator.py` - Expert coordination
- `memory.py` - Context persistence
- `gating_network.py` - Advanced intent classification

**Goal**: Handle complex queries like "Compare Frodo's journey to Bilbo's, focusing on character development"

### Phase 6: Learning & Improvement

**Restore**:
- `learning_loop.py` - Self-improvement
- `trace_analyzer.py` - Execution analysis
- `emergent_intent_analyzer.py` - Pattern discovery

**Goal**: System improves from usage, discovers new reasoning patterns

### Phase 7: Goal-Directed Agents

**Restore**:
- `goals.py` - Goal tracking
- `values.py` - Value alignment
- `environment.py` - Agent interactions
- `execution_loop.py` - Autonomous task completion

**Goal**: Autonomous agents with goals, values, and multi-turn interactions

---

## How to Restore

1. **Move file back**: `git mv deprecated/agentic/blueprint.py klareco/`
2. **Restore tests**: `git mv deprecated/tests/test_blueprint.py tests/`
3. **Update imports**: Ensure no broken imports
4. **Update documentation**: Add to README features list

---

## Architecture Vision (Phase 5-7)

The deprecated components implement this architecture:

```
┌───────────────────────────────────────────────────────────────────┐
│                    KLARECO PIPELINE v2                            │
│                  (AST-as-Consciousness)                           │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Input → Front Door → Parser → Enricher → Reasoner → Deparser   │
│             ↓           ↓         ↓          ↓           ↓        │
│          Language    Grammar   Semantics  Inference  Esperanto   │
│           AST         AST       AST         AST        Text      │
│             ↓                                ↓                    │
│         Translate ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ Output               │
│                                                                   │
│  Every module enriches the AST with metadata (all in Esperanto)  │
│                                                                   │
│  Components:                                                      │
│  - Blueprint: Multi-step planning (complex queries)              │
│  - Orchestrator: Routes to expert modules                        │
│  - Memory: Persists context across sessions                      │
│  - Learning Loop: Improves from execution traces                 │
│  - Goals/Values: Ethical constraints and objectives              │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

This is **excellent architecture**, just not needed for POC.

---

## Current POC Architecture (Simplified)

```
┌─────────────────────────────────────────────────────────────────┐
│                    KLARECO POC                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Query → Parser → Structural Filter → Neural Reranker → Answer │
│            ↓             ↓                    ↓                  │
│          AST      (0 params)            (15M params)            │
│                                                                 │
│  Components:                                                    │
│  - Parser: Deterministic (0 params)                            │
│  - Structural Filter: Deterministic slot matching (0 params)   │
│  - Neural Reranker: Tree-LSTM embeddings (15M params)         │
│  - Answer Extraction: Template-based (0 params)                │
│                                                                 │
│  GOAL: Answer 50 questions with ONLY this (Month 1-2)         │
│  NEXT: Add 20M param reasoning core (Month 3-4)               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

Simple, focused, proves the thesis.

---

## Metrics

### Code Reduction

- **Before cleanup**: ~9,300 LOC
- **After cleanup**: ~5,500 LOC
- **Reduction**: 40% (4,500 LOC deprecated)

### Component Breakdown

- **Deterministic core** (KEEP): ~1,900 LOC (parser, deparser, canonicalizer)
- **Embeddings** (KEEP): ~1,400 LOC (compositional embeddings, AST-to-graph)
- **Retrieval** (KEEP): ~1,200 LOC (two-stage hybrid, Tree-LSTM)
- **Support** (KEEP): ~1,000 LOC (corpus manager, logging, translation)
- **Agentic** (DEPRECATED): ~4,500 LOC
- **Old models** (DEPRECATED): ~400 LOC
- **Redundant utils** (DEPRECATED): ~1,000 LOC

---

## Lessons Learned

**What worked**:
- Deterministic parser (91.8% parse rate on real corpus)
- Two-stage hybrid retrieval (30-40% faster, 37% better relevance)
- Compositional embeddings (75% parameter reduction)

**What was premature**:
- Multi-step planning before proving single-step works
- Neural intent classification before testing symbolic routing
- Self-improvement loops before establishing baseline
- Goal/value systems before defining core functionality

**Going forward**:
- Prove the core thesis first (Month 1-4 POC)
- Add complexity incrementally (Phase 5-7)
- Keep deprecated components for reference and future restoration

---

## See Also

- `CODEBASE_CLEANUP_ANALYSIS.md` - Full cleanup analysis
- `ESPERANTO_FIRST_IMPLEMENTATION_PLAN.md` - Phase-by-phase plan
- `README.md` - POC goals and current state
- `CLAUDE.md` - Development guide
