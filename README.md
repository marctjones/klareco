# Klareco - Esperanto-First AST RAG

Klareco uses Esperanto’s regular grammar and limited vocabulary to replace most probabilistic LLM steps with deterministic structure:
- Parse every sentence into an AST with explicit roles (subject/object/verb), case, number, tense, and morphemes.
- Tokenize by grammar (prefix + root + suffix + ending) so tokens align with meaning and stay stable.
- Retrieve by structural signatures first, then use a small model only for semantic smoothing.
- Generate extractively from AST contexts, with a lightweight AST-aware seq2seq as an optional layer.

**Why Esperanto helps**
- Fully regular morphology → programmatic tokenization and tagging; minimal need for learned POS/NER.
- Fixed endings for roles/case/tense → deterministic subject/object detection; less reliance on attention to infer roles.
- Small, compositional lexicon → smaller embedding tables and shallower models; reuse morpheme embeddings across the corpus.

## Current state (Updated Nov 2025)
✅ **Production Ready** - Two-stage hybrid retrieval with complete sentence corpus

- ✅ **Deterministic parser/deparser** (`parser.py`, `deparser.py`) with AST-to-graph converter
- ✅ **Two-stage hybrid retrieval** - Structural filtering (0 params, ~2ms) + neural reranking (15M params, ~15ms)
- ✅ **Canonical slot signatures** (`canonicalizer.py`) - SUBJ/VERB/OBJ extraction for structural search
- ✅ **Extractive responders** (`experts/extractive.py`, `experts/summarizer.py`) - Template-based answers from AST contexts
- ✅ **AST-first orchestrator** (`orchestrator.py`) - Intent routing with structural retrieval integration
- ✅ **High-quality corpus** - 26,725 complete sentences (Corpus V2) with 88-94% parse quality
- ✅ **Production index** (`data/corpus_index_v3`) - Complete sentences with structural metadata
- ✅ **Compositional embeddings** (`klareco/embeddings/compositional.py`) - Morpheme-aware embeddings (320K vs 1.28M params)
- ✅ **Semantic similarity training** - Using Tatoeba EN-EO parallel corpus (271K pairs) as similarity oracle
- Language ID + translation front door with graceful fallback (`lang_id.py`, `front_door.py`)
- Tracing/logging and symbolic intent gating (`trace.py`, `logging_config.py`, `gating_network.py`)
- Comprehensive test coverage (11 new tests for structural components, all passing)

## Key Achievements
- **7x smaller models** - 15M params vs 110M+ for traditional LLMs
- **30-40% faster retrieval** - Two-stage hybrid vs neural-only
- **Zero-parameter Stage 1** - Deterministic structural filtering
- **Production corpus** - Complete sentences vs hard-wrapped fragments (+37% better relevance scores)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Optional for RAG/graphs:
pip install torch-geometric faiss-cpu
```
Place required local assets (not in git):
- `data/corpus_index/{faiss_index.bin,metadata.jsonl,embeddings.npy}`
- `models/tree_lstm/checkpoint_epoch_12.pt` (or your own checkpoint)

## Usage (current)
### Parse / translate
```bash
python -m klareco parse "Mi amas la hundon."
python -m klareco translate "The dog sees the cat." --to eo  # downloads MarianMT on first run
```

### Corpus management (V2 - Complete Sentences)
```bash
# Validate or register texts
python -m klareco corpus validate data/raw/book.txt
python -m klareco corpus add data/raw/book.txt --title "My Book" --type literature
python -m klareco corpus list

# Build Corpus V2 with proper sentence extraction
python scripts/build_corpus_v2.py \
  --cleaned-dir data/cleaned \
  --output data/corpus_with_sources_v2.jsonl \
  --min-parse-rate 0.5  # Filter low-quality sentences

# Build Index V3 from Corpus V2
python scripts/index_corpus.py \
  --corpus data/corpus_with_sources_v2.jsonl \
  --output data/corpus_index_v3 \
  --batch-size 32
```

### RAG Query Demo
```bash
# Run interactive demo with Index V3
python scripts/demo_rag.py --interactive

# Or run demo queries
python scripts/demo_rag.py

# Single query
python scripts/demo_rag.py "Kio estas la Unu Ringo?"
```

### Retrieval (Programmatic)
```python
from klareco.rag.retriever import create_retriever
from klareco.experts.extractive import create_extractive_responder
from klareco.parser import parse

# Create retriever with Index V3
retriever = create_retriever(
    index_dir="data/corpus_index_v3",
    model_path="models/tree_lstm/best_model.pt"
)

# Create extractive responder
responder = create_extractive_responder(retriever, top_k=3)

# Query
query = "Kio estas Esperanto?"
query_ast = parse(query)
result = responder.execute(query_ast, query)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.3f}")
for src in result['sources']:
    print(f"  - {src['text'][:100]}...")
```

### Pipeline Status
✅ **Production Ready**
- Two-stage hybrid retrieval operational
- Extractive responders integrated
- AST-first orchestrator with intent routing
- See `RAG_STATUS.md` and `CORPUS_V2_RESULTS.md` for details

## Tests
- Fast checks: `python -m pytest tests/test_parser.py -k basic`, `python -m pytest tests/test_gating_network.py -k classify`.
- RAG/generator tests are skipped or fail if `torch-geometric`, `faiss`, and local indexes/models are missing.

## Roadmap
### ✅ Completed (Nov 2025)
- ✅ Grammar-driven tokenizer + canonical slot signatures
- ✅ Two-stage retrieval with structural filtering first
- ✅ Extractive answerer from AST contexts
- ✅ AST-first orchestrator with intent gating
- ✅ Resumable scripts with checkpoints and real-time logs
- ✅ Comprehensive tests for structural components

### 🔜 Next Steps
- ⏳ Train semantic similarity model on Tatoeba-derived pairs (`scripts/run_semantic_training.sh`)
- Evaluate semantic similarity model and integrate into retrieval
- Train AST-aware seq2seq model (scripts ready: `scripts/train_graph2seq.py`)
- SQLite/JSONL cache for structural filtering (currently in-memory)
- Expand corpus with more Esperanto texts
- Multi-field structural filtering (tense, case, mood)

See `DESIGN.md`, `RAG_STATUS.md`, and `CORPUS_V2_RESULTS.md` for details.

## Data & Licensing
`data/` and `logs/` stay local and untracked; do not commit corpora, checkpoints, or generated logs. Add your own texts under `data/raw/` and build indexes locally. Include your preferred license in this file when ready.
