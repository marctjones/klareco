# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Concept

Klareco is an experiment to **find where deterministic computation stops.** Given
a language with a genuinely regular grammar, how much of "understanding" can be
done with ordinary classical programming — rules, tables, queries, unification,
constraint solving, search — and what is the *irreducible residue* that resists
it? The residue, once characterized, is where machine learning belongs. Nowhere
else.

**This is a boundary-discovery project, not a small-model project.** Do not
pre-assign work to "rules" or "neural" by category. In particular, do **not**
assume reasoning requires learning: transitive inference, type hierarchies,
constraint propagation, quantifiers, arithmetic, and planning are classical CS and
must be attempted deterministically first. Equally, do not assume all of grammar
is deterministic — proper-noun disambiguation in Esperanto provably is not.

**The method** (see `VISION.md` for the full argument):
1. Attempt it deterministically. Implement. Measure.
2. Find where it breaks — and *why*, stated as a property of the problem, not of
   the implementation.
3. Characterize the residue: what information does the deterministic method
   provably lack?
4. Only then introduce a learned component, targeted at that specific residue.
5. Keep it decomposable — the deterministic version still runs, so we can always
   say how much the model added.

**⚠️ Before trusting any number in this repo, read the "Current state" section of
`DESIGN.md`.** Several data artifacts were lost in a June 2026 laptop migration
and the pipeline degrades *silently* without them.

## Key Architecture Principles

**AST-First Pipeline**: Everything operates on structured Abstract Syntax Trees,
not raw text. The AST is the universal contract between stages, and it is what
makes the deterministic/learned boundary *visible*.

```
Text → Parser (16 rules) → AST → Retrieval / Symbolic reasoning → AST → Deparser → Text
       └─ deterministic          └─ deterministic today             └─ deterministic
                                    (learned components: see DESIGN.md "deferred")
```

**Attribution is built in.** Each AST node tracks whether it came from a rule or a
model. Explainability does not require zero learned parameters — it requires
decomposable contributions.

**Function Word Exclusion Principle**:
- **Function words** (kaj, de, en, la, mi, etc.) are grammatical, not semantic
- They are handled by the **deterministic AST layer**, not learned embeddings
- Including them in embedding training causes **embedding collapse** (all words become similar)
- Only **content words** (hundo, tablo, legi, bela) get learned embeddings
- This is a core architectural decision, not a workaround

**A silently-degrading dependency is a bug.** Every artifact the pipeline loads
must fail loudly if absent. The June migration cost weeks of invisible quality
loss because missing files logged a warning and carried on.

## Schema-First Development

**The store is DuckDB, not Kuzu.** Kuzu was removed in the May–June 2026
migration; there are no `kuzu` imports left in `klareco/`, and no Kuzu database on
disk. Everything lives in `data/indexes/duckdb_store.db` (table `sentences`, one
row per sentence, AST carried as an `ast_json` blob) plus a Whoosh BM25 index at
`data/indexes/whoosh_v2`. Any doc, script, or code comment that tells you to write
Cypher is stale — fix it when you find it.

**⚠️ Ontology status: defined in code, NOT loaded at runtime.** The four layers
below are real, but they only ever loaded into Kuzu. In the DuckDB store,
`ontology_nodes` and `ontology_edges` are **empty** and `verb_klaso` is **0%
populated**. So:

- The "always query the ontology" rule below is currently **unfollowable**, and a
  couple of paths (e.g. `klareco/knowledge/synonyms.py`) fall back to hardcoded
  lists. That is **acknowledged debt, not a licence to add more.**
- The class definitions survive as Python literals in
  `scripts/index/extend_kuzu_schema_semantic_ontology.py`. Restoring the ontology
  means emitting a snapshot from those literals into the DuckDB tables — not
  re-deriving it from scratch.
- **Honest caveat:** even when loaded, the ontology is hand-seeded and thin
  (`kreado-26` = `["fond","kre","produk","far"]`; `persono` = `["homo","vir",
  "infan","kuracist"]`). Querying it beats hardcoding a list in Python — one
  source of truth, one place to extend — but it is the *same kind* of knowledge,
  just relocated. Lexical synonymy is a genuine learned residue we are currently
  faking with a list. Don't oversell it.

### What's in the Ontology

**Layer 1: Lexical Semantics**
- Verb classes (50+): kreado, movo, pensado, perceptado, emocio, komunikado, vivo, profesio
- Aspectual classes: stato, aktiveco, plenumigo, atingaĵo
- Noun entity types: persono, loko, tempo, organizaĵo, evento, profesio
- Thematic roles: aganto, paciento, temo, spertanto, instrumento, fonto, celo, loko, tempo

**Layer 2: Frame Semantics**
- Semantic frames (FrameNet-style)
- Event participants and roles

**Layer 3: Discourse Semantics**
- RST relations (detalaĵo, fono, rezulto, kaŭzo, celo, kontrasto)
- Information structure

**Layer 4: Schema Semantics**
- Biographical schema: identigo (1.0), ĉefa_realigo (0.95), naskiĝo_morto (0.85), profesio (0.80), loko (0.70)
- Definitional schema: kategorio (1.0), esenca_eco (0.90), funkcio (0.75)
- Event schema: ĉefa_okazaĵo (1.0), partoprenantoj (0.90), tempo (0.85), loko (0.80)

### Mandatory Rules

**1. AST access — read the `ast_json` blob; never reconstruct, never re-parse**
```python
# ✅ DO THIS — the store carries the parsed AST as a JSON blob (~0.9 ms):
import json, duckdb
con = duckdb.connect('data/indexes/duckdb_store.db', read_only=True)
ast = json.loads(con.execute(
    "SELECT ast_json FROM sentences WHERE sid = ?", [sid]).fetchone()[0])

# ✅ For text that is NOT in the store (e.g. the user's question), parse it:
from klareco.parser import parse
ast = parse(question)          # ~milliseconds

# ❌ NEVER: KuzuASTReconstructor — the Kuzu DB no longer exists, and the
#    reconstructor was measured at ~17,000 ms per AST before removal.
# ❌ NEVER: re-parse a sentence that is already in the store. The blob is
#    ~50x faster and is what was actually indexed.
```

**2. Query the ontology instead of hardcoding lists** *(blocked: see status above)*
```python
# ❌ NEVER DO THIS:
PERSON_WORDS = ['homo', 'vir', 'kuracist']  # Hardcoded gazetteer!

# ✅ DO THIS — ontology_edges maps radiko → class_id:
con.execute("""
    SELECT radiko FROM ontology_edges
    WHERE rel = 'HAVAS_ENTECAN_TIPON' AND class_id = 'persono'
""").fetchall()
# ⚠️ Returns [] today — ontology_edges is empty. If you need this, the
#    correct move is to RESTORE the ontology, not to add another list.
```

**3. Use verb classes for synonyms, not manual lists**
```python
# ❌ NEVER DO THIS:
CREATION_VERBS = ['fond', 'kre', 'produk']  # Manual list!

# ✅ DO THIS:
con.execute("""
    SELECT radiko FROM ontology_edges
    WHERE rel = 'APARTENAS_AL_VERBA_KLASO' AND class_id = 'kreado-26'
""").fetchall()
# The store also has a `verb_klaso` column on `sentences` for filtering —
# ⚠️ currently 0% populated; populating it is part of the ontology restore.
```

**4. Use schema slots for importance ranking, not hardcoded weights**
```python
# ❌ NEVER DO THIS:
if question_type == 'WHO':
    importance = 0.95  # Hardcoded!

# ✅ DO THIS — SkemaSloto nodes carry graveco_pezo:
rows = con.execute(
    "SELECT node_json FROM ontology_nodes WHERE label = 'SkemaSloto'").fetchall()
```

**5. Use grammatical role + thematic role for answer extraction, not string matching**
```python
# ❌ NEVER DO THIS:
if question.startswith('Kiu'):
    return ast['subjekto']  # String matching on the surface form!

# ✅ DO THIS — the interrogative's CASE tells you which slot the answer fills.
#    Esperanto marks this explicitly: "Kiu" (nominative) → the gap is the
#    SUBJECT; "Kiun" (accusative) → the gap is the OBJECT. This is free,
#    deterministic, and English cannot do it. Read `kazo` off the korelativo
#    kerno rather than pattern-matching the question string.
```

### Decision Checklist

Before implementing ANY feature, ask:

- [ ] Does this need entity classification? → Query `ontology_edges` (`persono`, `loko`, …)
- [ ] Does this need verb synonyms? → Query the verb class, don't list roots
- [ ] Does this need importance ranking? → Query `SkemaSloto.graveco_pezo`
- [ ] Am I creating a hardcoded list? → STOP. Restore/extend the ontology instead.
- [ ] Am I re-parsing a sentence already in the store? → STOP, read `ast_json`
- [ ] Am I pattern matching on word forms or question strings? → STOP, use the AST's
      grammatical features (`kazo`, `vortspeco`, `sufiksoj`) — they are already there

### What to Do Instead of Piecemeal Solutions

| Piecemeal Approach | Schema-First Approach |
|--------------------|----------------------|
| Gazetteer of place names | Query `ontology_edges` for `class_id = 'loko'` |
| List of person words | Query `ontology_edges` for `class_id = 'persono'` |
| Verb synonym lists | Query the `VerbaKlaso` members |
| Hardcoded importance weights | Query `SkemaSloto.graveco_pezo` |
| `if question.startswith('Kiu')` | Read `kazo` / `vortspeco` off the interrogative AST node |
| Re-parsing an indexed sentence | `json.loads(ast_json)` from the store (~0.9 ms) |
| `KuzuASTReconstructor` | **Removed.** Kuzu no longer exists. |

### Files That Should NOT Exist

These indicate piecemeal solutions that should use the schema instead:

- ❌ `*_gazetteer.py` (entity lists)
- ❌ `*_patterns.py` (regex/pattern matching)
- ❌ `place_names.json` (hardcoded entities)
- ❌ `person_indicators.py` (hardcoded indicators)
- ❌ Any file re-parsing ASTs that exist in graph

### How to Extend the Schema

If you need a new semantic class:

1. Add it to the appropriate layer in the database
2. Update the taxonomy hierarchically
3. Link roots to the new class
4. Document in `docs/SEMANTIC_ONTOLOGY_REFERENCE.md`

**DO NOT** create standalone files or hardcoded lists.

## Development Commands

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional for graph neural networks:
pip install torch-geometric faiss-cpu
```

### Testing
```bash
# Run all tests
python -m pytest

# Fast checks (no external dependencies)
python -m pytest tests/test_parser.py -k basic
python -m pytest tests/test_gating_network.py -k classify

# Specific test file
python -m pytest tests/test_parser.py -v

# With coverage
python -m pytest --cov=klareco --cov-report=html
```

### Parser & Core Operations
```bash
# Parse Esperanto sentence
python -m klareco parse "Mi amas la hundon."

# Translate to Esperanto
python -m klareco translate "The dog sees the cat." --to eo
```

### Build the corpus + indexes
```bash
# Parse cleaned texts into a unified corpus with ASTs (~5-6 h)
./scripts/parse/parse_corpus.sh

# Build the DuckDB store (sentences + shredded AST columns + ast_json blob)
python scripts/index/build_duckdb_store.py

# Build Whoosh BM25 index on top of the store
python scripts/index/build_whoosh_index.py
```

### End-to-end question answering
```python
# There is NO `python -m klareco run`. The CLI exposes: parse, query,
# translate, corpus, info. To run the full orchestrator, use the factory:
from klareco.orchestrator.factory import build_default_pipeline
pipeline = build_default_pipeline(whoosh_index_dir='data/indexes/whoosh_v2')
result = pipeline.answer("Kiu fondis Esperanton?")
print(result.text)
```

```bash
# Evaluate against a test set (local, single-process)
python scripts/eval/evaluate_extractive_qa.py \
    --test-set data/test_sets/qa_test_diverse_30.jsonl

# A/B the rerankers against each other
python scripts/eval/multi_reranker_bench.py \
    --test-set data/test_sets/synthetic_who_rebuild_17_cleanish.jsonl

# Same evaluator on Modal (parallel workers)
# Push index volume first, then run
./scripts/eval/modal_upload_indexes.sh
modal run scripts/eval/modal_eval.py --test-set data/test_sets/qa_test_diverse_30.jsonl

# Diff two eval result files (regression check)
python scripts/eval/compare_eval_results.py before.json after.json

# Interactive retrieval debugger
python scripts/eval/debug_retrieval.py "Kiu fondis Esperanton?"
```

## Critical Implementation Details

### Parser (klareco/parser.py)
- 16 hand-coded Esperanto grammar rules
- Parses to AST with explicit roles: `subjekto`, `verbo`, `objekto`, `aliaj`
- Each word decomposed: `radiko` (root), `prefikso`, `sufiksoj`, `vortspeco`, `kazo`, `nombro`, `tempo`
- Parse status tracked per word: `success`, `unknown_root`, `proper_name_unknown`

### Compositional Embeddings (klareco/embeddings/compositional.py)
- Decomposes words into learned + programmatic features
- Learned: root (64d), prefix (8d), suffix (8d) - total ~500K params
- Programmatic: ending (8d), grammar (8d) - case, number, tense encoded deterministically
- Total output: 128d per word
- Key: Generalizes to unseen word combinations (e.g., "rehundejo" = re+hund+ej+o)

### ASTToGraphConverter (klareco/ast_to_graph.py)
**IMPORTANT BUG TO KNOW**: Constructor accepts either `int` or `CompositionalEmbedding` as first arg for backwards compatibility. Use keyword args to be explicit:
```python
# Correct
converter = ASTToGraphConverter(compositional_embedding=emb)

# Also works (for legacy code)
converter = ASTToGraphConverter(emb)  # Detects type automatically
```

### Training Scripts
All training scripts now include:
- Checkpoint resume by default (`--fresh` to override)
- File logging to `{output_dir}/training.log`
- Early stopping (patience=3 epochs)
- Atomic checkpoint saves (write to .tmp then rename to avoid corruption)
- Checkpoint rotation (keeps last 2: `best_model.pt` and `best_model.prev.pt`)

## Script Versioning Policy (MANDATORY)

**CRITICAL: Every Python script in `scripts/` MUST include version information.**

### When Creating ANY Script

Before creating a script, you MUST:

1. **Determine version compatibility**: What database version? What models does it depend on?
2. **Add complete docstring** with VERSION, COMPATIBLE WITH, DEPENDENCIES, STAGE
3. **Update `docs/VERSION_COMPATIBILITY.md`** if adding new script type
4. **Validate format** before committing

### Required Docstring Template

```python
"""
<Script Name - Descriptive Title>

VERSION: v2.1 | v3.0
COMPATIBLE WITH: v2.1 database schema, v3.0 AST-annotator protocol
DEPENDENCIES: Root Embeddings v3, M1 v2 (list all model dependencies)
STAGE: Data | Training | Evaluation | Inspection | Utility

Description:
    Brief description of what this script does (1-3 sentences).
    Focus on the "why" not just the "what".

Pipeline Position:
    v2.1 DB → [THIS SCRIPT] → Next Component → ...
    (Show where this fits in the data/training pipeline)

Usage:
    python scripts/path/to/script.py --arg1 value1 --arg2 value2

Inputs:
    - Input 1: Description (format: JSON/JSONL/CSV, location: data/...)
    - Input 2: Description

Outputs:
    - Output 1: Description (format, location)
    - Output 2: Description

Quality Checks:
    - Check 1: What validation is performed
    - Check 2: What quality metrics are computed

Last Updated: YYYY-MM-DD
Author: <name>
Related Issues: #123, #456
See Also: docs/RELATED_DOC.md, other_script.py
"""
```

### When Modifying Existing Scripts

When modifying a script, you MUST:

1. **Update VERSION** if:
   - Architecture changed (e.g., now uses v3.0 AST-annotator)
   - Dependencies changed (e.g., now requires M1 v3)
   - Database schema changed

2. **Update COMPATIBLE WITH** if dependencies changed

3. **Update Last Updated** date (always)

4. **Add comment** at top explaining what changed:
   ```python
   # CHANGELOG:
   # 2025-01-15: Migrated to v3.0 AST-annotator protocol
   # 2024-12-01: Added tier filtering support
   ```

### Script Naming Convention

All scripts follow: `<stage>_<target>_<version>.py`

Examples:
- `data/export_roots_v2.1.py` - Data export for v2.1 database
- `train/roots_v3.py` - Training for v3.0 architecture
- `evaluate/embedding_quality_v2.py` - Evaluation script

### Validation Checklist

Before creating/modifying a script, ASK YOURSELF:

- [ ] What version is this compatible with? (v2.1 database? v3.0 models?)
- [ ] What are ALL dependencies? (models, data, other scripts)
- [ ] What stage is this? (data/train/evaluate/inspect/util)
- [ ] Where does it fit in the pipeline? (input → THIS → output)
- [ ] What quality checks does it perform?
- [ ] What happens if it fails partway? (checkpoint support?)

If you can't answer these, **DON'T create the script** - ask for clarification first.

### Enforcement

This is **NON-NEGOTIABLE**. Scripts without proper version info will be rejected.

See:
- `docs/CLI_ARCHITECTURE.md` - Complete versioning strategy
- `docs/VERSION_COMPATIBILITY.md` - Version compatibility matrix
- `scripts/util/validate_script_versions.py` - Validation tool (to be created)

## Long-Running Scripts Policy

**IMPORTANT: Claude should NEVER run long-running scripts directly.**

Long-running scripts include:
- Training scripts (any model training)
- Corpus building/parsing scripts
- Dataset cleaning/processing scripts
- Index building scripts
- Any script that takes more than ~30 seconds

### What Claude Should Do Instead

1. **Create/update shell wrapper scripts** in `scripts/` that:
   - Activate the Python venv automatically
   - Have checkpoint support for restartability
   - Log output to `logs/` directory
   - Support `--fresh` flag to start over and `--resume` to continue

2. **Tell the user to run it** in a separate terminal:
   ```bash
   ./scripts/parse/parse_corpus.sh --fresh   # Example
   ```

3. **Monitor progress** only if asked, by reading log files

### Script Requirements for Restartability

All long-running Python scripts must have:
```python
# Checkpoint support
parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
parser.add_argument('--fresh', action='store_true', help='Start fresh, ignore checkpoint')

# Atomic checkpoint saves
def save_checkpoint(path, state):
    temp_path = path.with_suffix('.tmp')
    with open(temp_path, 'w') as f:
        json.dump(state, f)
    temp_path.rename(path)  # Atomic rename
```

### Shell Wrapper Template

```bash
#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "No venv found"; exit 1
fi

# Parse --fresh flag
FRESH_FLAG=""
[[ "$1" == "--fresh" ]] && FRESH_FLAG="--fresh"

# Run with logging
LOG_FILE="logs/script_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
python scripts/my_script.py $FRESH_FLAG 2>&1 | tee "$LOG_FILE"
```

### Available Long-Running Scripts

| Task | Shell Script | Description |
|------|--------------|-------------|
| Acquire tier-0 sources | `./scripts/acquire/acquire_all_tier0.sh` | Download authoritative Esperanto sources |
| Clean all texts | `./scripts/clean/clean_all.sh` | Clean Gutenberg + ReVo |
| Extract all | `./scripts/extract/extract_all.sh` | Extract Wikipedia + Books |
| Parse corpus | `./scripts/parse/parse_corpus.sh` | Build unified corpus with ASTs (~5-6 h) |
| Build DuckDB store | `python scripts/index/build_duckdb_store.py` | Corpus → `sentences` table (AST blob + shredded columns) |
| Build Whoosh index | `python scripts/index/build_whoosh_index.py` | Build BM25 index over the store |
| Post-reparse pipeline | `./scripts/pipeline/post_reparse_pipeline.sh` | Schema + Whoosh + eval (after a reparse) |
| Validate store | `python scripts/index/validate_duckdb_store.py` | DuckDB integrity checks |
| Validate all | `./scripts/validate/validate_all.sh` | Run corpus integrity checks |

⚠️ The `*_kuzu_*` scripts under `scripts/index/` are **dead** — Kuzu is gone. The
one exception is `extend_kuzu_schema_semantic_ontology.py`, which is still the
source of truth for the ontology's class definitions (see Schema-First above).

### Pipeline Stages

```
ACQUIRE  → Download raw data           (scripts/acquire/)
CLEAN    → Normalize text              (scripts/clean/)
EXTRACT  → Extract sentence JSONL      (scripts/extract/)
OCR      → PDF → text (PAG only)       (scripts/ocr/)
PARSE    → Parse to ASTs                (scripts/parse/)
INDEX    → DuckDB store + Whoosh       (scripts/index/)
EVAL     → Run evaluators               (scripts/eval/)
VALIDATE → Data-integrity checks       (scripts/validate/)
PIPELINE → Post-parse orchestrator     (scripts/pipeline/)
```

Scripts follow naming convention: `<stage>_<target>.py` and `<stage>_<target>.sh`,
where `<stage>` matches the subdirectory.

## Code Organization

```
klareco/
├── parser.py               # 16 Esperanto rules → AST
├── deparser.py             # AST → text reconstruction
├── proper_nouns.py         # v3 cleaned + Wikipedia-category dictionary
├── cli.py / __main__.py    # CLI entry points
├── orchestrator/           # Immutable QueryContext pipeline (active spine)
│   ├── pipeline.py         #   Orchestrator.answer(question)
│   ├── factory.py          #   build_default_pipeline(whoosh_index_dir=...)
│   ├── context.py          #   QueryContext / ContextDelta dataclasses
│   ├── phase_timer.py      #   Sub-stage timing
│   └── stages/             #   Parse → [Dialog, Math, Planner] → Retrieve
│                           #   → DetRerank → ASTAwareRerank → Rerank(stub)
│                           #   → ExtractGenerate → BiographyFormat → Format
├── rag/                    # Retrieval + extraction
│   ├── whoosh_retriever.py #   Main retriever (BM25 ∩ AST roles)
│   ├── duckdb_retriever.py #   Store-backed retrieval
│   ├── ast_aware_reranker.py #  Structural reranker (#741)
│   ├── entity_fact_retriever.py
│   ├── unified_extractor.py#   Fact extraction
│   ├── extractive_answering.py
│   ├── importance_scorer.py
│   ├── question_classifier.py
│   └── entity_recognizer.py
├── reasoning/              # Forward-chaining inference, path-finding (#749, #761)
├── planning/               # STRIPS-style task planner (#771)
├── generation/             # Biography / definition / comparison generators
├── dialog/                 # Multi-turn conversational state (#767)
├── tools/                  # SymPy math evaluator (#772)
├── knowledge/              # Vocabularies — ontology-backed by design;
│                           #   ⚠️ some hardcoded fallbacks active while
│                           #   the ontology is unpopulated (acknowledged debt)
├── ontology/               # ⚠️ DEAD — semantic_query.py still takes a kuzu_conn
├── eval/qa_metrics.py      # Shared evaluator (local + Modal)
└── (klareco/utils/ was removed with Kuzu; tests/test_kuzu_open.py is stale)

scripts/
├── acquire/      # Raw downloads from upstream sources
├── clean/        # Raw → cleaned text
├── extract/      # Cleaned → sentence JSONL
├── ocr/          # PDF → text (PAG)
├── parse/        # Build unified corpus from extracted sentences
├── index/        # Corpus → DuckDB store + Whoosh BM25 index
├── eval/         # Evaluators, comparators, debug tools
├── validate/     # Data-integrity checks
├── pipeline/     # Post-parse orchestrator
└── util/         # Dev tooling (script-version linter wired to pre-commit)
```

## Data Files (Not in Git)

```
data/
├── raw/                          # Raw source files (not modified)
│   ├── eo/                       # Esperanto raw data
│   │   ├── wikipedia/            # Wikipedia dumps
│   │   ├── gutenberg/            # Project Gutenberg texts
│   │   ├── fundamento/           # Fundamento/Krestomatio
│   │   └── dictionaries/         # ReVo, etc.
│   └── en/                       # English reference data
├── cleaned/                      # Cleaned text files
│   └── eo/                       # Cleaned Esperanto texts
├── extracted/                    # Extracted sentences with metadata
├── corpus/                       # Parsed corpus files with ASTs
│   └── unified_corpus.jsonl      # Main unified corpus
├── indexes/                      # FAISS indexes + metadata
│   ├── compositional/            # Compositional embeddings index
│   └── merged/                   # Merged sources index
├── training/                     # Training-ready filtered data
└── vocabularies/                 # Root/prefix/suffix vocabularies
    ├── root_vocab.json           # Roots from corpus
    ├── prefix_vocab.json        # Esperanto prefixes
    └── suffix_vocab.json        # Esperanto suffixes

models/
└── root_embeddings/              # Stage 1 root embedding model
```

## HuggingFace models (if/when klareco loads them)

klareco's own trained checkpoints live in `models/` (do NOT migrate to
the shared pool — they're project-specific). But if any future klareco
code loads a public HuggingFace model (e.g. a base encoder behind a
fine-tuned head), follow the cross-project convention:

- Use the env var `$AI_MODELS_DIR` (default `~/Projects/aishared/models/`,
  exported in `~/.bashrc`).
- Pre-fetch with `ai-fetch <hf-id>` (bash function in `~/.bashrc`).
- Load by **local path** — `AutoModel.from_pretrained(f"{os.environ['AI_MODELS_DIR']}/<org>_<model>")` —
  never by HF ID. This keeps `~/.cache/huggingface/` empty as a true cache.
- Full loader pattern in `~/.claude/skills/aishared-resources/SKILL.md`.

## Disk-space conventions

The project's data + indexes are large (typical working state ~50-90 GB)
and DuckDB doesn't auto-vacuum. Three hard rules and a maintenance toolkit.

### Never delete (source-of-truth or hard-to-rebuild)

- `data/raw/` — Wikipedia dump, Gutenberg texts, ReVo. Hard to re-acquire.
- `data/dictionaries/` — ReVo, Esperanto dictionaries.
- `data/cleaned/` — Cleaned text feeding extraction.
- `data/extracted/` — Extracted sentence JSONL feeding parsing.
- `data/corpus/` and `data/enhanced_corpus/` — Parsed AST corpus. Re-parse
  is ~5-6 hours.
- `data/indexes/` — Live DuckDB store + Whoosh index.
- `data/vocabularies/` — Root/affix vocabularies.
- `data/proper_nouns_dynamic_v3.json` AND its fallbacks
  `data/proper_nouns_dynamic_v2.json` AND `data/proper_nouns_dynamic.json` —
  the older versions are explicit fallback paths in `klareco/proper_nouns.py`.
  Don't delete superseded versions until the fallback chain in code is removed.
- `models/` — Trained checkpoints. Re-training is expensive.

### Always-regenerable (safe to delete to recover space)

- `data/staging/*.jsonl` — staging outputs from `build_*` scripts. Once the
  `--apply` step has run, the staging file is no longer needed.
- `logs/**/*.log` — per-run logs older than ~30 days.
- `results/bench_*.json{,l}` and `results/eval_*.json` — per-run bench
  outputs. Keep what's referenced from issues; rest auto-prune.
- `data/staging/duckdb_parquet_export/` — temporary EXPORT staging from
  the corruption-recovery or compaction tools.
- Any `*.tmp` file older than an hour — orphaned atomic-write scratch.

### In-place schema changes leave dead pages (the Stage-2 lesson)

DuckDB does **not** auto-vacuum. The pattern `ALTER TABLE ADD COLUMN` →
`UPDATE` → `CREATE INDEX` on a 5M-row table leaves ~30 GB of dead pages
that never get reclaimed. Two rules:

1. **Prefer new-table-swap over in-place ALTER for bulk schema changes:**
   ```sql
   CREATE TABLE sentences_new AS
     SELECT *, <computed_columns> FROM sentences;
   DROP TABLE sentences;
   ALTER TABLE sentences_new RENAME TO sentences;
   CREATE INDEX ... ON sentences(...);
   ```
   This is atomic, leaves no dead pages, and is easy to verify by row count
   before the `DROP TABLE`.

2. **After any unavoidable in-place bulk change, run compaction:**
   `python scripts/util/compact_duckdb.py --apply` does an EXPORT/IMPORT
   round-trip that reclaims the dead pages.

### The maintenance toolkit

- `scripts/util/preflight_disk.sh <min_gb> [reason]` — guard at top of
  long-running scripts. Refuses to start without enough headroom.
- `scripts/util/cleanup_stale.sh` — single entry point for safe deletions.
  Dry-run default; `--apply` to commit. Documents the never-delete list.
- `scripts/util/compact_duckdb.py --dry-run | --apply` — EXPORT/IMPORT
  round-trip to reclaim DuckDB dead pages.

See `docs/MAINTENANCE.md` for the recommended schedule.

## Important Patterns

### Error Handling in Training Scripts
Training scripts should never crash on I/O errors during checkpoint saves. Always:
```python
temp_path = output_dir / 'checkpoint.pt.tmp'
try:
    torch.save(checkpoint, temp_path)
    temp_path.rename(output_dir / 'checkpoint.pt')
except Exception as e:
    logger.error(f"Failed to save: {e}")
    if temp_path.exists():
        temp_path.unlink()
    # Continue training
```

### AST Structure
All ASTs follow this pattern:
```python
{
    'tipo': 'frazo',
    'subjekto': {'tipo': 'vortgrupo', 'kerno': {...}, 'priskriboj': [...]},
    'verbo': {'tipo': 'vorto', 'radiko': '...', 'vortspeco': 'verbo', ...},
    'objekto': {...},
    'aliaj': [...],  # Modifiers, adverbs, etc.
    'parse_statistics': {'total_words': N, 'success_rate': 0.XX}
}
```

### Running Long Processes
For corpus building or large evaluations, run shell scripts in a
separate terminal to keep Claude's context clear:
```bash
./scripts/parse/parse_corpus.sh  # logs under logs/
```

Monitor with: `tail -f logs/<script-name>_*.log`

## Current Development Status

**Active spine**: AST-native orchestrator pipeline (`klareco/orchestrator/`)
with deterministic-first evaluation. See `DESIGN.md` — especially its
**"Current state"** section — for the full picture.

**Working**:
- 16-rule parser + deparser (UD-Prago: 80.3% POS strict, 93.3% scheme-adjusted)
- DuckDB store: 5.39M sentences, `ast_json` blob + shredded AST columns
- Whoosh BM25 index with AST-role matching
- Extractive QA end-to-end: retrieves and answers with citations
- `klareco.eval` shared by local + Modal evaluators

**Broken / degraded — read before trusting any number:**
- ⚠️ Parser data lost in the June 2026 migration (`protected_roots.json`,
  `proper_nouns_dynamic_v*.json`). `Esperanton` parses to root `esper` + suffix
  `ant`. Fails silently.
- ⚠️ Semantic ontology **not loaded**: `ontology_nodes`/`ontology_edges` empty,
  `verb_klaso` 0% populated. Everything downstream of it no-ops.
- ⚠️ `entity_facts` table missing — `BiographyFormatStage` crashes.
- ⚠️ **All nine rerankers are tied.** Everything they *share* (BM25, phrase boost,
  exact radiko match, tense) is alive; everything that makes any one of them
  *different* (verb class, negation, entity-type gating) reads a dead column.
  Identical live inputs → identical rankings. The smart reranker is written; it is
  running on empty. Compounding this, the 17-question test set has no headroom
  (recall@5 = 17/17). Fixing either alone proves nothing — both must land together.

**Unmeasured** (landed just before the migration, never benchmarked): symbolic
reasoning (#749, #761), planner (#771), math tool (#772), dialog (#767),
biography/definition generators (#766, #775).

**In Progress**: restore the lost artifacts, then build a *discriminating* test
set (#736, #737) — one where BM25 fails but the answer is still findable. Until
that exists, reranker work cannot be measured.
See [EPIC #713](https://github.com/marctjones/klareco/issues/713).

**Deferred**: the learned stack was **pruned from the repo** (commits `b68320e`,
`822a3eb`, `313ec3e`) — `klareco/embeddings/`, `klareco/models/`, and
`klareco/summarization/` no longer exist; recover from git history if needed. Only
the neural reranker remains, as a no-op stub in `RerankStage`. Learned components
re-enter the pipeline only when the deterministic floor is stable enough to
attribute a measurable improvement to a specific model.

## Testing Philosophy

Klareco uses a **four-category testing strategy** aligned with the staged pipeline architecture.

### Test Categories

| Category | Purpose | Location |
|----------|---------|----------|
| **Code Tests** | Verify implementation correctness | `tests/test_*.py` |
| **Data Quality Tests** | Validate training data quality | `tests/test_data_quality.py` |
| **Model Quality Tests** | Measure trained model performance | `tests/test_model_quality.py` |
| **Regression Tests** | Prevent quality degradation | `tests/test_regression.py` |

### TDD Workflow

For each stage implementation:
1. **Write failing tests FIRST** (red)
2. **Implement minimal code to pass** (green)
3. **Refactor while tests pass** (refactor)
4. **Verify coverage** (`pytest --cov`)

### Stage-Specific Tests

**Stage 0 (Parser)**: #115
- All 16 grammar rules tested
- Edge cases: compound words, correlatives, numerals
- Parse rate tracking (target: >90%)
- Coverage target: 90%+

**Stage 0 (Data Quality)**: #116
- Corpus quality: parse rate, duplicates, source diversity
- Vocabulary coverage: Fundamento roots, affix completeness
- Training pair quality: function word exclusion, balance

**Stage 1 (Semantic Model)**: #117
- Root similarity accuracy: >85%
- No embedding collapse: mean_sim < 0.5
- Cluster separation: gap > 0.03
- Fundamento coverage: 100%
- Affix consistency: mal- vector similarity > 0.7

**Stage 2 (Grammatical Model)**: #118
- Negation polarity reversal
- Tense temporal ordering
- Mood discrimination: >80%
- Sentence type classification: >95%

**Stage 3 (Discourse Model)**: #119
- Coreference chain coherence: >0.7
- Cross-document discrimination: <0.3
- Discourse relation classification

### Running Tests

```bash
# All tests
python -m pytest

# With coverage
python -m pytest --cov=klareco --cov-report=html

# Code tests only (fast)
python -m pytest tests/test_parser.py tests/test_deparser.py -v

# Model quality tests (requires trained models)
python -m pytest tests/test_model_quality.py -v

# Skip slow tests
python -m pytest -m "not slow"
```

### Coverage Targets

| Module | Current | Target |
|--------|---------|--------|
| Parser | 61% | 90%+ |
| Deparser | TBD | 85%+ |
| Embeddings | TBD | 85%+ |
| Retriever | TBD | 80%+ |

See wiki: **Testing-Strategy.md** for comprehensive documentation.

## Knowledge Management Strategy

**IMPORTANT**: Klareco uses a four-tier content organization system across Wiki, Discussions, Issues, and Markdown files (similar to llmfp and pdfe projects).

### The Four-Tier System

**Tier 1: Wiki** (Educational, Timeless Reference)
- **Purpose**: Explain concepts, algorithms, linguistic theory
- **Content**: Esperanto grammar rules, AST structure, compositional embeddings theory
- **Audience**: Anyone learning about Esperanto-first AI concepts
- **Lifespan**: Timeless - updated when understanding changes
- **Examples**: "Esperanto Grammar Rules", "Compositional Embeddings Theory", "AST Structure"

**Tier 2: Discussions** (Feedback, Ideas, Lab Notes)
- **Purpose**: Unstructured thoughts, feedback, ideas, Q&A, experiment results
- **Content**: Lab notebooks, feature ideas, training experiments, research findings
- **Audience**: Developers, contributors, future collaborators
- **Lifespan**: Permanent but evolving - stays open for ongoing conversation
- **Examples**: "Lab Notebook: Semantic Similarity Training", "Idea: Multi-hop Reasoning", "Training Results Discussion"

**Tier 3: Issues** (Actionable Tasks)
- **Purpose**: Track bugs, features, and tasks with clear completion criteria
- **Content**: Bugs to fix, features to implement, models to train
- **Audience**: Developers implementing changes
- **Lifespan**: Temporary - closed when completed
- **Examples**: "Fix parser bug for compound words (#5)", "Implement AST Trail system (#12)"

**Tier 4: Markdown Files** (Code Documentation)
- **Purpose**: Document code architecture, API, project-specific guides
- **Content**: README, CLAUDE.md, design docs, implementation roadmaps
- **Audience**: Developers working with the codebase
- **Lifespan**: Version-controlled - updates with code changes
- **Examples**: "README.md", "CLAUDE.md", "IMPLEMENTATION_ROADMAP_V2.md"

### Decision Matrix: Where Does Content Go?

| Content Type | Wiki | Discussion | Issue | Markdown |
|--------------|------|------------|-------|----------|
| **Grammar theory** | ✅ Primary | - | - | Reference |
| **Algorithm explanation** | ✅ Primary | - | - | - |
| **Bug to fix** | - | - | ✅ Primary | - |
| **Feature to implement** | - | Discussion→ | ✅ Primary | - |
| **Research question** | Reference | Discussion→ | ✅ Primary | - |
| **Unstructured thoughts** | - | ✅ Primary | - | - |
| **Feature idea (unvalidated)** | - | ✅ Primary | →Issue | - |
| **Training results** | - | ✅ Primary | Reference | Reference |
| **Usage question** | Reference | ✅ Primary | - | - |
| **Code API docs** | - | - | - | ✅ Primary |
| **Lessons learned** | ✅ Primary | ✅ Initial | - | - |

### GitHub CLI Commands

```bash
# Issues
gh issue list                                    # List all open issues
gh issue list --label "priority: high"           # Filter by label
gh issue view 5                                  # View issue details
gh issue create --title "Title" --body "Desc"   # Create new issue
gh issue close 14 --comment "Fixed in abc123"   # Close with comment
gh issue comment 5 --body "Added validation results"  # Add comment to issue

# Pull Requests
gh pr list                                       # List open PRs
gh pr create --title "Title" --body "Desc"      # Create PR
gh pr view 3                                     # View PR details
gh pr checks                                     # View CI status
gh pr merge 3                                    # Merge PR

# Repository
gh repo view                                     # View repo info
gh browse                                        # Open repo in browser
gh label list                                    # List labels

# Discussions (via API - no direct gh command)
gh api repos/marctjones/klareco --jq '.has_discussions'  # Check if enabled
# Create discussions via web UI or API
```

### Wiki Access

The wiki is a **separate git repository**. It cannot be accessed via `gh` CLI.

```bash
# Clone the wiki
git clone https://github.com/marctjones/klareco.wiki.git
cd klareco.wiki

# Edit markdown files (e.g., Home.md, Stage-1-Embeddings.md)
# Wiki pages are plain markdown files

# Push changes
git add .
git commit -m "Update wiki"
git push origin master
```

**Important**: Always `git pull` before editing to avoid merge conflicts if others edited via web UI.

### Content Migration Guidelines

**FROM Issues TO Discussions**:
Migrate if issue is:
- ❌ Not actionable (no clear completion criteria)
- ❌ Open-ended research without specific goal
- ❌ Ideas without implementation plan
- ❌ Placeholder for "someday maybe"

**FROM Discussions TO Issues**:
Convert when discussion leads to:
- ✅ Specific, actionable task
- ✅ Clear success criteria
- ✅ Decision to implement

**FROM Discussions TO Wiki**:
Migrate when discussion crystallizes into:
- ✅ Documented understanding
- ✅ Educational reference material
- ✅ Timeless knowledge

### Issue Management Best Practices

1. **Check Issues Before Creating**:
   - ALWAYS check existing issues: `gh issue list`
   - Check Discussions too (might already be there)
   - Reference relevant issue numbers in commits

2. **Create Issues Proactively**:
   - When discovering bugs → create issue with labels
   - When identifying enhancements → create issue
   - When planning training experiments → create issue
   - Use labels: `bug`, `enhancement`, `training`, `documentation`, `research`

3. **Close Issues When Resolved**:
   - After fixing bug or implementing feature → close issue
   - Reference issue in commit: `git commit -m "Fix parser bug #5"`
   - Use `gh issue close 5` when work is complete

4. **Update Documentation References**:
   - Replace inline TODOs with "See issue #X"
   - Keep docs focused on current capabilities, not future plans

### What NOT to Create

- ❌ `SESSION_SUMMARY.md` - Use issue comments instead
- ❌ `TODO_LIST.md` - Use GitHub issues
- ❌ `/tmp/research_ideas.md` - Create GitHub Discussion instead
- ❌ Ephemeral tracking files - GitHub is source of truth

### docs/ Directory Management

**KEY INSIGHT**: The `docs/` directory should contain ONLY operational guides tied to code, not educational content or session notes.

**What STAYS in docs/ (Tier 4):**
- ✅ Technical guides for running scripts (e.g., `CORPUS_BUILDING.md`)
- ✅ Operational references (e.g., `CORPUS_INVENTORY.md` - simplified list)
- ✅ API documentation
- ✅ Setup/installation guides tied to specific code

**What MOVES to Wiki (Tier 1):**
- ❌ `RAG_SYSTEM.md` → Wiki: "RAG System Overview"
- ❌ `TWO_STAGE_RETRIEVAL.md` → Wiki: "Two-Stage Retrieval Architecture"
- ❌ `RETRIEVAL_GUIDE.md` → Wiki: Merge into retrieval page
- ❌ `CORPUS_MANAGEMENT.md` → Wiki: "Corpus Management Guide"
- **Why**: Explains concepts/algorithms (educational, timeless)

**What MOVES to Discussions (Tier 2):**
- ❌ `SESSION_SUMMARY.md` → Discussion: "Lab Notebook: [Date]"
- ❌ `docs/archive/*.md` → Discussion: "Lab Notebook: Archive"
- **Why**: Session notes, experiment results, historical record

**Decision Rule for docs/**:
- **Keep**: "How do I run this code?" (operational)
- **Move to Wiki**: "How does this work?" (conceptual)
- **Move to Discussion**: "What did we learn?" (results/notes)

**Example**:
- ✅ KEEP: `CORPUS_BUILDING.md` - Step-by-step guide to run `./scripts/extract/extract_wikipedia.sh`
- ❌ MOVE: `RAG_SYSTEM.md` - Explains RAG architecture concepts (→ Wiki)
- ❌ MOVE: `SESSION_SUMMARY.md` - Notes from development session (→ Discussion)

## See Also

- `VISION.md` - Long-term thesis: decomposable contributions, attribution
- `DESIGN.md` - Active architecture: orchestrator stages, schema-first foundation, deferred work
- `README.md` - Setup and quickstart commands
- `AGENTS.md` - Repository guidelines
- `16RULES.MD` - Esperanto grammar specification (reference)
- `docs/VERSION_COMPATIBILITY.md` - Deferred v3.0 model-retraining plan
