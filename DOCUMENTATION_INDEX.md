# Klareco Documentation Index

**Last Updated**: 2025-12-23

This document provides a complete index of all Klareco documentation with clear guidance on where to find specific information.

---

## Quick Navigation

- **New to Klareco?** → Start with [README.md](README.md)
- **AI Assistant working on code?** → Read [CLAUDE.md](CLAUDE.md)
- **Want to understand the vision?** → Read [VISION.md](VISION.md)
- **Need technical details?** → Read [DESIGN.md](DESIGN.md)
- **Looking for implementation plan?** → Read [ESPERANTO_FIRST_IMPLEMENTATION_PLAN.md](ESPERANTO_FIRST_IMPLEMENTATION_PLAN.md)

---

## Core Documentation (Repository Root)

### Essential Reading

| File | Purpose | Audience |
|------|---------|----------|
| **[README.md](README.md)** | Project overview, setup, usage examples | Everyone |
| **[CLAUDE.md](CLAUDE.md)** | Development guide for AI assistants | Claude Code/AI assistants |
| **[VISION.md](VISION.md)** | Long-term architectural vision and thesis | Architects, researchers |
| **[DESIGN.md](DESIGN.md)** | Technical design decisions | Developers |
| **[ESPERANTO_FIRST_IMPLEMENTATION_PLAN.md](ESPERANTO_FIRST_IMPLEMENTATION_PLAN.md)** | Detailed AST enrichment phases | Developers, project managers |

### Supporting Documentation

| File | Purpose | Status |
|------|---------|--------|
| **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** | This file - complete doc index | Active |
| **[AGENTS.md](AGENTS.md)** | Agent-specific documentation | Active |
| **[MIGRATION_PLAN.md](MIGRATION_PLAN.md)** | Documentation migration plan | Completed ✅ |

---

## Technical Documentation (`docs/`)

### Current Technical Docs

| File | Purpose | Audience |
|------|---------|----------|
| **[docs/CORPUS_INVENTORY.md](docs/CORPUS_INVENTORY.md)** | Complete catalog of Esperanto corpus | Everyone working with data |
| **[docs/CORPUS_MANAGEMENT.md](docs/CORPUS_MANAGEMENT.md)** | How to manage and build corpus | Developers |
| **[docs/RAG_SYSTEM.md](docs/RAG_SYSTEM.md)** | RAG system architecture | Developers |
| **[docs/RETRIEVAL_GUIDE.md](docs/RETRIEVAL_GUIDE.md)** | Retrieval implementation guide | Developers |
| **[docs/TWO_STAGE_RETRIEVAL.md](docs/TWO_STAGE_RETRIEVAL.md)** | Two-stage retrieval design | Developers, ML engineers |
| **[docs/SESSION_SUMMARY.md](docs/SESSION_SUMMARY.md)** | Development session notes | Developers |

---

## GitHub Wiki Pages

**URL**: https://github.com/marctjones/klareco/wiki

### Published Pages ✅

| Wiki Page | Description | Source |
|-----------|-------------|--------|
| **[Home](https://github.com/marctjones/klareco/wiki/Home)** | Wiki navigation and overview | - |
| **[Esperanto Grammar Reference](https://github.com/marctjones/klareco/wiki/Esperanto-Grammar-Reference)** | Complete 16 rules, morphology, why it enables deterministic parsing | docs/wiki-drafts/ |
| **[LLM Architecture Fundamentals](https://github.com/marctjones/klareco/wiki/LLM-Architecture-Fundamentals)** | Traditional LLMs vs Klareco, tensor types, parameter savings | docs/wiki-drafts/ |
| **[Compositional Embeddings](https://github.com/marctjones/klareco/wiki/Compositional-Embeddings)** | Root + prefix + suffix embedding strategy | COMPOSITIONAL_EMBEDDINGS.md |

### Planned Wiki Pages (To Be Created)

1. **AST Structure and Theory** (P0)
   - AST definition and benefits
   - Klareco AST schema
   - Traversal strategies
   - Canonicalization

2. **Compositional Embeddings** (P0)
   - Theory and motivation
   - Implementation details
   - Parameter savings analysis
   - Source: `COMPOSITIONAL_EMBEDDINGS.md`

3. **Graph-to-Graph Transformers** (P1)
   - Architecture design
   - Attention mechanisms
   - Training strategies
   - Source: Epic 2 planning

4. **Deterministic vs Learned Boundaries** (P1)
   - Design philosophy
   - Decision criteria
   - Examples and trade-offs
   - Source: `VISION.md`

5. **Semantic Similarity Training** (P1)
   - Tatoeba oracle method
   - Training process
   - Evaluation results
   - Source: Training scripts and logs

6. **Retrieval Strategies** (P2)
   - Two-stage hybrid retrieval
   - Structural filtering
   - Neural reranking
   - Source: `docs/TWO_STAGE_RETRIEVAL.md`

7. **Proper Noun Handling** (P2)
   - Dictionary system
   - Static vs dynamic lists
   - Integration with parser
   - Source: `ESPERANTO_FIRST_IMPLEMENTATION_PLAN.md` Phase 5.1

---

## Archived Documentation (`docs/archive/`)

**Purpose**: Historical docs, superseded plans, session notes

| File | Reason for Archive | Superseded By |
|------|-------------------|---------------|
| **SESSION_SUMMARY.md** | Historical session notes | Current work tracked in GitHub issues |
| **TODO.md** | Old todo list | GitHub issues and milestones |
| **QUICK_WINS_ANALYSIS.md** | Historical analysis | Current implementation plan |
| **CORPUS_BUILDER_BUG_FIXES.md** | Historical bug notes | Fixed in current code |
| **CORPUS_BUILD_IMPROVEMENTS.md** | Historical improvements | Implemented in V2 |
| **CORPUS_IMPROVEMENT_PLAN.md** | Old planning doc | ESPERANTO_FIRST_IMPLEMENTATION_PLAN.md |
| **IMPLEMENTATION_ROADMAP_V2.md** | Old roadmap | ESPERANTO_FIRST_IMPLEMENTATION_PLAN.md + GitHub issues |
| **QUICK_START.md** | To be consolidated | Will merge into README or docs/SETUP.md |
| **PYTHON_SETUP.md** | To be consolidated | Will merge into docs/SETUP.md |

---

## Data Documentation (`data/`)

| File | Purpose |
|------|---------|
| **[data/GUTENBERG_CORPUS_SUMMARY.md](data/GUTENBERG_CORPUS_SUMMARY.md)** | Project Gutenberg corpus details |

---

## Script Documentation (`scripts/`)

| File | Purpose |
|------|---------|
| **[scripts/README_SCRIPTS.md](scripts/README_SCRIPTS.md)** | Script usage guide |
| **[scripts/QUICK_QUERY_README.md](scripts/QUICK_QUERY_README.md)** | Quick query tool guide |

---

## Examples (`examples/`)

| File | Purpose |
|------|---------|
| **[examples/README.md](examples/README.md)** | Example usage and demos |

---

## GitHub Documentation

### Issues
- **Epics**: #4 (Foundation), #5 (Learning), #6 (Data), #7 (Evaluation), #8 (Retrieval), #9 (Creativity)
- **Milestones**: M1 (Symbolic Baseline), M2 (Neural Core), M3 (Scaled System), M4 (Production)
- **Labels**: priority (P0-P3), type (deterministic, learned, data, evaluation), epic (foundation-creativity)

### Wiki (To Be Created)
- Esperanto Grammar Reference
- LLM Architecture Fundamentals
- AST Structure and Theory
- Compositional Embeddings
- Graph-to-Graph Transformers
- Deterministic vs Learned Boundaries
- Semantic Similarity Training
- Retrieval Strategies

### Discussions (To Be Created)
- **Research Findings**: Semantic similarity training results, corpus quality analysis
- **Ideas**: Creativity module design, future directions
- **Questions**: Model architecture decisions, implementation choices

---

## Documentation by Use Case

### "I want to understand the project"
1. Read [README.md](README.md) - Overview and setup
2. Read [VISION.md](VISION.md) - Core thesis and long-term goals
3. Read [docs/wiki-drafts/Esperanto-Grammar-Reference.md](docs/wiki-drafts/Esperanto-Grammar-Reference.md) - Why Esperanto
4. Read [DESIGN.md](DESIGN.md) - Technical implementation

### "I want to work on the code"
1. Read [CLAUDE.md](CLAUDE.md) - Development guide
2. Read [ESPERANTO_FIRST_IMPLEMENTATION_PLAN.md](ESPERANTO_FIRST_IMPLEMENTATION_PLAN.md) - Current plan
3. Check GitHub issues for specific tasks
4. Read relevant docs/ files for technical details

### "I want to understand the corpus"
1. Read [docs/CORPUS_INVENTORY.md](docs/CORPUS_INVENTORY.md) - Complete catalog
2. Read [data/GUTENBERG_CORPUS_SUMMARY.md](data/GUTENBERG_CORPUS_SUMMARY.md) - Gutenberg details
3. Read [CORPUS_V2_RESULTS.md](CORPUS_V2_RESULTS.md) - V2 build results
4. Read [docs/CORPUS_MANAGEMENT.md](docs/CORPUS_MANAGEMENT.md) - Management guide

### "I want to understand the architecture"
1. Read [docs/wiki-drafts/LLM-Architecture-Fundamentals.md](docs/wiki-drafts/LLM-Architecture-Fundamentals.md) - Traditional LLMs
2. Read [VISION.md](VISION.md) - Why AST-first
3. Read [DESIGN.md](DESIGN.md) - Implementation details
4. Read [COMPOSITIONAL_EMBEDDINGS.md](COMPOSITIONAL_EMBEDDINGS.md) - Embedding strategy

### "I want to understand retrieval"
1. Read [RAG_STATUS.md](RAG_STATUS.md) - Current status
2. Read [docs/TWO_STAGE_RETRIEVAL.md](docs/TWO_STAGE_RETRIEVAL.md) - Design
3. Read [docs/RETRIEVAL_GUIDE.md](docs/RETRIEVAL_GUIDE.md) - Implementation
4. Read [docs/RAG_SYSTEM.md](docs/RAG_SYSTEM.md) - System architecture

---

## Maintenance Notes

### When to Archive a Document
- File is superseded by newer version
- Content is historical (session notes, old plans)
- Content has been migrated to wiki or issues
- File is no longer referenced in code or other docs

### When to Create a Wiki Page
- Content is timeless educational material
- Explains concepts, not implementation
- Useful for broad audience (not just developers)
- Referenced frequently

### When to Create a GitHub Discussion
- Open-ended research question
- Experimental results to share
- Ideas for future directions
- Lab notebook entries

### When to Create a GitHub Issue
- Specific, actionable task
- Clear completion criteria
- Bug to fix or feature to implement
- Assigned to milestone

---

## Recent Updates

**2025-12-23**: Major documentation cleanup
- Created `docs/CORPUS_INVENTORY.md` - Complete corpus catalog
- Created `docs/wiki-drafts/Esperanto-Grammar-Reference.md` - Comprehensive grammar reference
- Created `docs/wiki-drafts/LLM-Architecture-Fundamentals.md` - LLM architecture explanation
- Archived 9 outdated markdown files to `docs/archive/`
- Updated README and CLAUDE with POC plan
- Created GitHub structure (labels, milestones, epic issues)

**2025-11-27**: Corpus V2 completion
- Built 26,725 sentence corpus with complete sentences
- Achieved 88-94% parse rates
- Created Index V3 with AST metadata

---

## TODO: Documentation Tasks

### High Priority
- [ ] Create `docs/SETUP.md` - Consolidate QUICK_START.md + PYTHON_SETUP.md
- [ ] Migrate wiki drafts to GitHub Wiki
- [ ] Create AST Structure wiki page
- [ ] Create Compositional Embeddings wiki page (from existing MD)

### Medium Priority
- [ ] Create GitHub Discussions for research findings
- [ ] Update DESIGN.md with latest architecture changes
- [ ] Create Graph-to-Graph Transformers wiki page
- [ ] Document training data generation process

### Low Priority
- [ ] Create video walkthrough of AST parsing
- [ ] Create interactive parser demo
- [ ] Write blog post about deterministic vs learned boundaries
- [ ] Create diagrams for all wiki pages

---

## GitHub Discussions

**URL**: https://github.com/marctjones/klareco/discussions

### Published Discussions ✅

| Discussion | Category | Description |
|------------|----------|-------------|
| **[Corpus V2 Build Results](https://github.com/marctjones/klareco/discussions/11)** | Show and tell | 26,725 complete sentences, 91.8% parse rate, +37% retrieval improvement |
| **[Corpus Quality Analysis](https://github.com/marctjones/klareco/discussions/12)** | Show and tell | Parse failure analysis, proper noun handling, recommendations |
| **[Two-Stage Retrieval Performance](https://github.com/marctjones/klareco/discussions/13)** | Show and tell | Structural + neural retrieval, 30-40% faster |

---

## Migrated Files ✅

**Date**: 2025-12-23

| Original File | Migrated To | Status |
|---------------|-------------|--------|
| `docs/wiki-drafts/*.md` | [GitHub Wiki](https://github.com/marctjones/klareco/wiki) | ✅ Published |
| `COMPOSITIONAL_EMBEDDINGS.md` | [Wiki: Compositional Embeddings](https://github.com/marctjones/klareco/wiki/Compositional-Embeddings) | ✅ Published |
| `CORPUS_V2_RESULTS.md` | [Discussion #11](https://github.com/marctjones/klareco/discussions/11) | ✅ Published |
| `CORPUS_AND_AST_AUDIT.md` | [Discussion #12](https://github.com/marctjones/klareco/discussions/12) | ✅ Published |
| `RAG_STATUS.md` | [Discussion #13](https://github.com/marctjones/klareco/discussions/13) | ✅ Published |

**All migrated files moved to `docs/archive/` for historical reference.**
