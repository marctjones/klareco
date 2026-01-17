# Klareco - Pure Esperanto AI

**A general-purpose conversational AI that maximizes deterministic processing and minimizes learned parameters.**

Klareco leverages Esperanto's regular grammar to replace most traditional LLM components with programmatic structure:
- **100% deterministic**: Parser, deparser, morphology, grammar checker, symbolic reasoner
- **Minimal learned**: Root embeddings (320K params) + Reasoning Core (20-100M params)
- **The thesis**: By making grammar explicit through ASTs, a small reasoning core can match larger models while being fully explainable and grammatically perfect.

## Vision & Purpose

**Core Thesis**: Traditional LLMs waste capacity learning grammar. By factoring out linguistic structure programmatically, we can focus all learned parameters on *reasoning*.

**Proof of Concept Plan**:
1. Month 1-2: Symbolic reasoner + deterministic features → answer 50 questions with ZERO learned reasoning
2. Month 3-4: Add 20M param reasoning core → measure improvement
3. Success: 80%+ accuracy on Esperanto Q&A, fully explainable, grammatically perfect

**Why Esperanto Enables This**:
- Fully regular morphology → 100% programmatic parsing (no learned POS/NER needed)
- Fixed endings for case/tense → deterministic role detection (no attention needed)
- Compositional lexicon → root embeddings only (prefix/suffix as transformation vectors)
- 16 explicit grammar rules → symbolic reasoning over AST structures

## Current State (January 2026)

**Architecture**: Multi-model semantic system (M0/Stage1/M1/M2/M3) - see `CURRENT_PLAN.md`

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

### ❌ M2: Taxonomic + Discourse (TODO)
- **M2.1 Taxonomic**: IS-A relationships (~10M params) - Issue #443
- **M2.2 Discourse**: Passage coherence (~30-50M params) - Issue #444
- **Status**: Not started

### ❌ M3: Orchestration (TODO)
- **Components**: Multi-model coordination, Kuzu graph database (5.2GB active)
- **Status**: Research phase - Issue #449
- **Files**: `klareco/rag/kuzu_inverted_index.py`

### Current Priorities
1. **CRITICAL**: Fix Stage 1 vocabulary corruption (#479)
2. **HIGH**: Improve M1 object selectional preference (#475)
3. **NEXT**: Build M2 models (#443, #444)
4. **FUTURE**: Research M3 orchestration (#449)

## Architecture

```
Text → M0 (Parser) → AST → Stage 1 (Roots) → M1 (Selectional) → M2 (Taxonomic+Discourse) → M3 (Orchestration) → Text
       └─ 0 params            └─ 320K params   └─ 10M params     └─ 40-50M params            └─ 0 params
       └─ deterministic                        └─ learned models                               └─ deterministic
```

**Total learned parameters**: ~60-70M (vs 1B+ for typical LLMs)

See `CURRENT_PLAN.md` for detailed architecture, `VISION.md` for the thesis, and `DESIGN.md` for technical details.

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
```bash
# Root embeddings demo
python scripts/demo_root_embeddings.py

# M1 selectional preference demo
python scripts/demo_m1_selectional.py

# AST-aware retrieval demo
python scripts/demo_ast_retriever.py

# Semantic retrieval demo
python scripts/demo_semantic_retrieval.py
```

### Train Models
```bash
# Train Stage 1 root embeddings (in separate terminal)
./scripts/train_roots.sh

# Train M1 selectional model
./scripts/m1_train_selectional.sh

# Validate M1 model
./scripts/m1_validate_selectional.sh
```

See `CURRENT_PLAN.md` for the full development roadmap.

## Documentation

| Document | Purpose |
|----------|---------|
| `CURRENT_PLAN.md` | Multi-model architecture roadmap (M0/Stage1/M1/M2/M3) |
| `VISION.md` | Core thesis and long-term vision |
| `DESIGN.md` | Technical architecture details |
| `CLAUDE.md` | Development guide for Claude Code |
| `AGENTS.md` | IdlerGear agent instructions |
| `16RULES.MD` | Esperanto grammar specification |
| **Wiki** | Detailed component docs (Current-Architecture, M0, Stage1, M1, M2, M3) |

## Tests

```bash
python -m pytest                           # All tests
python -m pytest tests/test_parser.py -v   # Parser tests
python -m pytest --cov=klareco             # With coverage
```

## Project Status

| Component | Status | Details |
|-----------|--------|---------|
| **M0: Parser** | ✅ Complete | 91.8% parse rate on 4.2M sentences |
| **Stage 1: Root Embeddings** | 🚧 Needs retrain | Issue #479 - vocabulary corruption (CRITICAL) |
| **M1: Selectional Preference** | 🚧 In progress | 80.2% accuracy, object preference issues (Issue #475) |
| **M2.1: Taxonomic Model** | 🔲 TODO | Issue #443 - Pure IS-A relationships |
| **M2.2: Discourse Coherence** | 🔲 TODO | Issue #444 - Passage ranking |
| **M3: Orchestration** | 🔲 Research | Issue #449 - Multi-model coordination |
| **Kuzu Graph Database** | ✅ Active | 5.2GB AST-first retrieval infrastructure |
| **Test Suite** | 🚧 In progress | Issue #470 - Data quality + integration tests |

## License

Data and logs stay local and untracked. Add your own texts under `data/raw/` and build indexes locally.
