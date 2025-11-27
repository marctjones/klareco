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

## Current state
- Deterministic parser/deparser (`parser.py`, `deparser.py`) and AST-to-graph converter (`ast_to_graph.py`).
- Language ID + translation front door with graceful fallback (`lang_id.py`, `front_door.py`).
- RAG scaffolding: Tree-LSTM retriever, FAISS indexer, corpus manager/CLI (`rag/retriever.py`, `scripts/index_corpus.py`, `corpus_manager.py`).
- Tracing/logging and symbolic intent gating (`trace.py`, `logging_config.py`, `gating_network.py`), with targeted pytest coverage.

## In-flight redesign
- Replace placeholder orchestrator with AST-first routing + structural retrieval and extractive answers.
- Add canonical slot signatures and grammar-driven tokenizer to drive indexing and search.
- Make training/cleaning scripts resumable with periodic checkpoints and real-time logs.
- Tighten docs, TODO, and design to reflect the Esperanto advantage and reduced-model footprint.

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

### Corpus management
```bash
# Validate or register texts (uses lingua + parser samples)
python -m klareco corpus validate data/raw/book.txt
python -m klareco corpus add data/raw/book.txt --title "My Book" --type literature
python -m klareco corpus list

# Build corpus with sources and index it (requires local texts)
python scripts/build_corpus_with_sources.py --min-length 20 --output data/corpus_with_sources.jsonl
python scripts/index_corpus.py --corpus data/corpus_with_sources.jsonl --output data/corpus_index --batch-size 32
```

### Retrieval (requires index + checkpoint)
```python
from klareco.rag.retriever import create_retriever
retriever = create_retriever(index_dir="data/corpus_index",
                             model_path="models/tree_lstm/checkpoint_epoch_12.pt")
for hit in retriever.retrieve("Kio estas Esperanto?", k=3):
    print(f"{hit['score']:.2f} :: {hit['text']}")
```

### Pipeline status
- The CLI pipeline (`python -m klareco run ...`) currently uses a minimal orchestrator with placeholder answers; being replaced by the AST-first structural retriever + extractive responder.
- `scripts/run_pipeline.py` still reflects the old graph generation path and will be updated after the tokenizer/indexer rewrite.

## Tests
- Fast checks: `python -m pytest tests/test_parser.py -k basic`, `python -m pytest tests/test_gating_network.py -k classify`.
- RAG/generator tests are skipped or fail if `torch-geometric`, `faiss`, and local indexes/models are missing.

## Roadmap (see DESIGN.md for detail)
- Build grammar-driven tokenizer + canonical slot signatures; update indexer/retriever to use structural filtering first.
- Implement extractive answerer from AST contexts; add optional small AST-aware seq2seq.
- Replace orchestrator with intent gating wired to structural retrieval.
- Make all long-running scripts resumable with checkpoints and real-time logs.
- Expand deterministic tests for parser/tokenizer/indexer/pipeline; keep coverage high.

## Data & Licensing
`data/` and `logs/` stay local and untracked; do not commit corpora, checkpoints, or generated logs. Add your own texts under `data/raw/` and build indexes locally. Include your preferred license in this file when ready.
