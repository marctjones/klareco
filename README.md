# Klareco - Pure Esperanto AI

**A general-purpose conversational AI that maximizes deterministic processing and minimizes learned parameters.**

Klareco leverages Esperanto's regular grammar to replace most traditional LLM components with programmatic structure:
- **100% deterministic**: Parser, deparser, morphology, grammar checker, symbolic reasoner
- **Minimal learned**: Root embeddings (320K params) + Reasoning Core (20-100M params)
- **The thesis**: By making grammar explicit through ASTs, a small reasoning core can match larger models while being fully explainable and grammatically perfect.

## Vision & Purpose

**Core Thesis**: Traditional LLMs waste capacity learning grammar. By factoring out linguistic structure programmatically, we can focus all learned parameters on *reasoning*.

**Architectural Approach**: Multi-model semantic system (M0/Stage1/M1/M2/M3)
- Each model solves ONE semantic problem (selectional preference, taxonomy, discourse)
- Models compose together on top of deterministic AST foundation
- Explainable through decomposable contributions (what came from rules vs learned models)

**Why Esperanto Enables This**:
- Fully regular morphology → 100% programmatic parsing (no learned POS/NER needed)
- Fixed endings for case/tense → deterministic role detection (no attention needed)
- Compositional lexicon → root embeddings only (prefix/suffix as transformation vectors)
- 16 explicit grammar rules → symbolic reasoning over AST structures

**Key Architectural Lessons** (learned through development):
- **Function words must be excluded**: Including grammatical words in embeddings causes collapse
- **Compositional embeddings generalize**: Root + affix composition handles unseen words perfectly
- **Small, specialized models work**: 10M param M1 model achieves 80%+ accuracy on its specific task
- **Don't learn what you know**: Grammar is deterministic - focus learned parameters on semantics only

## Current State (January 2026)

**✅ Working RAG System**: Full retrieval pipeline with AST-aware search + neural reranking operational on 5.3M sentence corpus

**Architecture**: Multi-model semantic system (M0/Stage1/M1/M2/M3)
- 📋 **[GitHub Project Board](https://github.com/users/marctjones/projects/16)** - Track current work
- 📚 **[Wiki: Current-Architecture](https://github.com/marctjones/klareco/wiki/Current-Architecture)** - Architecture details
- 🎯 **[Epic #453](https://github.com/marctjones/klareco/issues/453)** - Overall progress tracking

### ✅ RAG System: Question Answering (WORKING)
- **Corpus**: 5.3M Esperanto sentences from Wikipedia + books
- **Pipeline**: AST-aware retrieval → Neural reranking → Answer extraction
- **Reranker**: 180K param model with frozen compositional embeddings
- **Entity-aware**: Question type detection, entity recognition, relevance boosting
- **Demo**: `./scripts/demo_full_rag.sh` - Try "Kiu fondis Esperanton?" and see it work!
- **Files**: `klareco/rag/ast_aware_retriever.py`, `klareco/rag/kuzu_inverted_index.py`

### ✅ M0: Deterministic Parser (COMPLETE)
- **Parser/Deparser**: 16 Esperanto grammar rules, 91.8% parse rate on 4.2M sentences
- **AST generation**: Explicit roles (subjekto, verbo, objekto, aliaj)
- **Morpheme decomposition**: 100% deterministic
- **Files**: `klareco/parser.py`, `klareco/deparser.py`

### 🚧 Stage 1: Root Embeddings (NEEDS RETRAIN)
- **Architecture**: 64D embeddings for content words only (~320K params)
- **Status**: Trained but vocabulary corruption found (Issue #479 - CRITICAL)
- **Target**: 18,928 roots from Tier 2-5 vocabulary
- **Function words**: Excluded (handled deterministically by M0)
- **Files**: `klareco/embeddings/compositional.py`, `models/root_embeddings/`

### 🚧 M1: Selectional Preference (IN PROGRESS)
- **Architecture**: Subject-verb-object compatibility scoring (~10M params)
- **Status**: Model trained, object selectional preference issues (Issue #475)
- **Accuracy**: 80.2% overall, 83% plausible detection
- **Files**: `scripts/train_m1_selectional.py`, `tests/test_m1_model_quality.py`

### ✅ Semantic Enrichment: Three-Tier Entity Taxonomy (NEW)
- **Architecture**: Deterministic + learned semantic annotation (~5M params)
- **Three-tier hierarchy**: Aristotelian (6) → NER-compatible (18) → Fine-grained (286)
- **Tier 1 (100% deterministic)**: From vortspeco alone (entity, attribute, quantity, relation, spacetime, action)
- **Tier 2 (70% deterministic)**: From correlatives + affixes (person, organization, location, etc.)
- **Tier 3 (30% deterministic)**: GNN-based classifier for fine-grained types (besto:mamulo, ŝtato:eŭropa)
- **Function word exclusion**: Only content words get learned embeddings (prevents embedding collapse)
- **Files**: `klareco/semantic_enrichment/`, `klareco/models/entity_classifier.py`

### ❌ M2: Taxonomic + Discourse (TODO)
- **M2.1 Taxonomic**: IS-A relationships (~10M params) - Issue #443
- **M2.2 Discourse**: Passage coherence (~30-50M params) - Issue #444
- **Status**: Not started

### ❌ M3: Orchestration (TODO)
- **Components**: Multi-model coordination, Kuzu graph database (5.2GB active)
- **Status**: Research phase - Issue #449
- **Files**: `klareco/rag/kuzu_inverted_index.py`

### Development Stage

**Milestone Achieved**: Working RAG system answering Esperanto questions with 500K learned parameters!

After 2 years of exploration (documented in [Development History](https://github.com/marctjones/klareco/wiki/Klareco-Development-History)), we've validated the core thesis:
- ✅ **Parser works**: 91.8% parse rate on 4.2M sentences proves deterministic grammar is viable
- ✅ **Compositional embeddings work**: 320K params covers 18,928 roots with perfect generalization
- ✅ **RAG system works**: 500K total params answering real questions on 5.3M sentence corpus
- ✅ **AST-aware retrieval works**: Entity detection, question classification, relevance ranking operational
- 🎯 **Now**: Improving answer extraction and multi-document reasoning
- 🔮 **Next**: Expanding to conversational Q&A with context management

### Current Priorities
1. **WORKING**: RAG system operational - try `./scripts/demo_full_rag.sh`!
2. **NEXT**: Improve answer extraction quality and multi-document reasoning
3. **FUTURE**: Add conversational context and multi-turn Q&A
4. **RESEARCH**: Expand semantic models (M1/M2) for enhanced understanding

## Architecture

```
Text → M0 (Parser) → AST → Compositional Embeddings → Retrieval → Reranker → Answer
       └─ 0 params            └─ 320K params            └─ deterministic  └─ 180K params
       └─ deterministic       └─ learned                                  └─ learned

RAG Pipeline (WORKING):
  Query → AST Parse → Entity Detection → Kuzu Graph Search (5.3M docs)
       → Entity Boost → Quality Filter → Neural Reranking → Top Results
```

**Current learned parameters**:
- Compositional embeddings: 320K params (root + affix embeddings)
- Reranker: 180K params (relevance scoring)
- Entity classifier: ~5M params (Tier 3 fine-grained semantic types)
- **Total: ~5.5M params** serving real Q&A queries on 5.3M sentence corpus with semantic enrichment

See the [Wiki](https://github.com/marctjones/klareco/wiki/Current-Architecture) for detailed architecture, `VISION.md` for the thesis, and `DESIGN.md` for technical details.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional for neural components:
pip install torch-geometric faiss-cpu
```

## Usage

### Parse Esperanto
```bash
python -m klareco parse "Mi amas la hundon."
python -m klareco translate "The dog sees the cat." --to eo
```

### Demos

**⭐ Try the RAG System:**
```bash
# Full pipeline: Retrieval → Reranking (recommended)
./scripts/demo_full_rag.sh

# Single question
./scripts/demo_full_rag.sh "Kiu fondis Esperanton?"

# With M1 filtering (optional, slower)
./scripts/demo_full_rag.sh --use-m1
```

**Other demos:**
```bash
# Root embeddings demo
python scripts/demo_root_embeddings.py

# M1 selectional preference demo
python scripts/demo_m1_selectional.py

# Basic AST retrieval (no reranking)
python scripts/demo_ast_retriever.py -i

# Semantic enrichment demo
python -c "from klareco.semantic_enrichment import ASTSemanticEnricher; \
from klareco.parser import parse; \
enricher = ASTSemanticEnricher(); \
ast = parse('La hundo kurtas.'); \
print(enricher.enrich(ast))"
```

### Train Models
```bash
# Train Stage 1 root embeddings (in separate terminal)
./scripts/train_roots.sh

# Train M1 selectional model
./scripts/m1_train_selectional.sh

# Validate M1 model
./scripts/m1_validate_selectional.sh

# Train entity classifier (Tier 3 semantic enrichment)
./scripts/train_entity_classifier.sh

# Generate training data for entity classifier
./scripts/generate_entity_training_data.sh
```

See the [GitHub Project Board](https://github.com/users/marctjones/projects/16) for current work and the [Wiki](https://github.com/marctjones/klareco/wiki) for architecture details.

## Documentation

| Document | Purpose |
|----------|---------|
| **[GitHub Project #16](https://github.com/users/marctjones/projects/16)** | Current work tracking (visual kanban board) |
| **[Epic #453](https://github.com/marctjones/klareco/issues/453)** | Multi-model architecture progress tracking |
| **[Wiki: Current-Architecture](https://github.com/marctjones/klareco/wiki/Current-Architecture)** | Active architecture (M0/Stage1/M1/M2/M3) |
| **[Wiki: Development-History](https://github.com/marctjones/klareco/wiki/Klareco-Development-History)** | Complete history: 5 phases, lessons learned, architectural evolution |
| `VISION.md` | Core thesis: decomposable contributions, explainability |
| `DESIGN.md` | Technical architecture details |
| `CLAUDE.md` | Development guide for Claude Code |
| `AGENTS.md` | IdlerGear agent instructions |
| `16RULES.MD` | Esperanto grammar specification |

## Tests

```bash
python -m pytest                           # All tests
python -m pytest tests/test_parser.py -v   # Parser tests
python -m pytest --cov=klareco             # With coverage
```

## Project Status

| Component | Status | Details |
|-----------|--------|---------|
| **RAG System** | ✅ **WORKING** | Q&A on 5.3M sentences, AST-aware + neural reranking |
| **M0: Parser** | ✅ Complete | 91.8% parse rate on 4.2M sentences |
| **Compositional Embeddings** | ✅ Complete | 320K params, frozen for reranking |
| **Reranker** | ✅ Complete | 180K params, learned relevance scoring |
| **Kuzu Graph Database** | ✅ Active | 5.2GB AST-first retrieval infrastructure |
| **Entity Detection** | ✅ Working | Question classification, entity recognition, boosting |
| **Answer Extraction** | 🚧 In progress | AST-based extraction with multi-document support |
| **M1: Selectional Preference** | 🔲 Future | Optional enhancement for query expansion |
| **Test Suite** | 🚧 In progress | Integration tests for RAG pipeline |

## License

Data and logs stay local and untracked. Add your own texts under `data/raw/` and build indexes locally.
