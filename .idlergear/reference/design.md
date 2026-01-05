---
id: 1
title: Design
created: '2026-01-01T06:50:15.797505Z'
updated: '2026-01-01T06:50:15.797554Z'
---
# Klareco Design & Architecture

## Core Principle

The **AST is the contract** between all components. Parsing, retrieval, routing, and generation all operate on structured, role-annotated trees.

## Why Esperanto Shrinks the Model

| Problem | Solution |
|---------|----------|
| Attention learns roles | Deterministic: endings encode case/tense/mood |
| Large embedding tables | Compositional: morpheme-level embeddings shared |
| Tokenizer drift | Grammar-driven: prefix + root + suffix + ending |
| Large rerankers | Structural filtering: slot signatures reduce candidates |

## Pipeline Architecture

```
Front Door (lang ID, translation)
    ↓
Parser (16 rules) → AST with roles/case/tense
    ↓
Two-Stage Retrieval
  - Stage 1: Structural filter (roots/roles) - SQLite
  - Stage 2: Neural rerank (Tree-LSTM) - FAISS
    ↓
Generation
  - Default: Extractive/template from AST
  - Optional: Small AST-aware seq2seq
    ↓
Deparser (rules) → Grammatically correct text
```

## What We Shrink or Avoid

| LLM Component | Klareco Approach |
|---------------|------------------|
| Tokenizer/embeddings | Grammar tokens, 128d |
| Positional encoding | Slot signatures shorten sequences |
| Attention layers | Role/case explicit; shallow or none |
| Output projection | Extractive avoids large vocab softmax |
| Rerankers | Structural filtering → tiny or skipped |

## Key Modules

| Module | Purpose |
|--------|---------|
| `parser.py` | 16 rules → AST with roles |
| `deparser.py` | AST → text reconstruction |
| `ast_to_graph.py` | AST → PyG graph for neural |
| `rag/retriever.py` | Two-stage search |
| `embeddings/compositional.py` | Root embeddings (320K params) |
| `orchestrator.py` | Intent routing, pipeline |

## Development Standards

- TDD: >90% coverage for core paths
- Long-running scripts: checkpoint resume required
- Environment: Python 3.13, venv, pip (no Conda)
