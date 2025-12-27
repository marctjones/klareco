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

## Current State (December 2025)

### Production Ready
- **Deterministic parser/deparser** (`parser.py`, `deparser.py`) - 16 Esperanto grammar rules, 91.8% parse rate
- **Two-stage hybrid retrieval** - Structural filtering (0 params) + neural reranking
- **Canonical slot signatures** (`canonicalizer.py`) - SUBJ/VERB/OBJ extraction
- **Extractive responders** (`experts/extractive.py`, `experts/summarizer.py`)
- **Production corpus index** (`data/corpus_index_v3`)

### Training Data Ready
- **Unified corpus**: 4.2M parsed sentences with ASTs
- **Training corpus**: 2M high-quality sentences (90%+ parse rate)
  - Authoritative (Tier 1-3): 16,557 sentences (Fundamento, Krestomatio, Gerda)
  - Literature (Tier 5): 19,796 sentences
  - General (Tier 6): 1.9M sentences (Wikipedia)
- **ReVo dictionary**: 10,766 entries for semantic training
- **Fundamento roots**: Extracted from Universala Vortaro

### Ready to Train
- Root embedding training script ready (`scripts/training/train_root_embeddings.py`)
- Affix embedding training ready (`scripts/training/train_affix_embeddings.py`)
- See `TRAINING_QUICKSTART.md` for instructions

## Architecture

```
Text → Parser (16 rules) → AST → Compositional Embeddings → Retrieval/Reasoning → Linearizer → Text
       └─ deterministic        └─ learned (~333K params)                          └─ deterministic
```

See `VISION.md` for the full architecture and `DESIGN.md` for technical details.

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

### RAG Query
```bash
python scripts/demo_rag.py --interactive
python scripts/demo_rag.py "Kio estas Esperanto?"
```

### Train Models
```bash
# Run training pipeline (in separate terminal)
./scripts/run_fundamento_training.sh

# Monitor progress
tail -f logs/training/fundamento_training_*.log
```

See `TRAINING_QUICKSTART.md` for the complete training guide.

## Training Pipeline (TRAINING_PLAN_V3)

The training follows a staged approach where each stage is frozen before the next begins:

```
STAGE 0: PARSER/DETERMINISTIC (complete)
├── 16 grammar rules
├── Morpheme decomposition
├── Role detection (S/V/O)
└── Negation/question type marking

STAGE 1: SEMANTIC MODEL (~333K params) ← CURRENT
├── Phase 1: Root embeddings (5K roots × 64d)
├── Phase 2: Affix embeddings (50 affixes × 64d)
└── Phase 3: Corpus integration

STAGE 2: GRAMMATICAL MODEL (~52K params)
├── Negation transform
├── Tense/mood transforms
└── Sentence type transforms

STAGE 3: DISCOURSE MODEL (~100K params)
├── Coreference resolution
└── Discourse relations

STAGE 4: REASONING CORE (20-100M params) - FUTURE
└── AST-to-AST reasoning
```

## Key Design Principles

1. **Function Word Exclusion**: Function words (la, kaj, de, en, mi...) are handled by the AST layer, not learned. Including them causes embedding collapse.

2. **Fundamento-Centered Training**: Zamenhof's original works have 100x weight vs Wikipedia. Authoritative sources define correct Esperanto.

3. **Compositional Morphology**: Words are decomposed into root + affixes. Embeddings compose: `malgrandega = mal- + grand + -eg-`

4. **Staged Training**: Each stage frozen before the next. No catastrophic forgetting, clear checkpoints.

## Documentation

| Document | Purpose |
|----------|---------|
| `TRAINING_PLAN_V3.md` | Definitive training pipeline design |
| `TRAINING_QUICKSTART.md` | Quick start guide for training |
| `VISION.md` | Long-term architecture vision |
| `DESIGN.md` | Technical architecture details |
| `CLAUDE.md` | Development guide for Claude Code |
| `DATA_INVENTORY.md` | Data sources and status |

## Tests

```bash
python -m pytest                           # All tests
python -m pytest tests/test_parser.py -v   # Parser tests
python -m pytest --cov=klareco             # With coverage
```

## Project Status

| Component | Status |
|-----------|--------|
| Parser (16 rules) | Production |
| Training corpus | Ready (2M sentences) |
| Root embeddings | Ready to train |
| Affix embeddings | Ready to train |
| Grammatical model | Designed |
| Discourse model | Designed |
| Reasoning core | Future |

## License

Data and logs stay local and untracked. Add your own texts under `data/raw/` and build indexes locally.
