# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Concept

Klareco is a **Pure Esperanto AI** that maximizes deterministic processing and minimizes learned parameters. The core thesis: By making grammar, morphology, and linguistic structure 100% programmatic, we can focus all learned capacity on *reasoning*, not language rules.

**The Proof of Concept Plan:**
- Month 1-2: Build symbolic reasoner + deterministic features
- GOAL: Answer 50 questions using ONLY deterministic processing + retrieval (zero learned reasoning)
- NEXT: Add minimal 20M param reasoning core, measure improvement
- THESIS TEST: If 50-100M param core gets 80%+ accuracy on Esperanto Q&A while being fully explainable and grammatically perfect, the thesis is proven

**This is achievable.** The foundation is strong. The key shift: stop trying to learn grammar, focus learned capacity entirely on reasoning.

## Key Architecture Principles

**AST-First Pipeline**: Everything operates on structured Abstract Syntax Trees, not raw text.

```
Text → Parser (rules) → AST → Compositional Embeddings → Retrieval/Reasoning → Linearizer → Text
       ├─ deterministic     ├─ learned (~500K params)                           └─ deterministic
       └─ 16 Esperanto rules
```

**What's Deterministic vs Learned**:
- **100% Deterministic**: Parser, deparser, morphology analyzer, grammar checker, symbolic reasoner, prefix/suffix/ending features, **function word handling**
- **Minimal Learned**: Root embeddings for content words only (320K params), AST Reasoning Core (target 20-100M params), retrieval reranking
- **Goal**: Maximum deterministic processing. Learn reasoning patterns, NOT grammar rules.

**Function Word Exclusion Principle** (see Wiki for details):
- **Function words** (kaj, de, en, la, mi, etc.) are grammatical, not semantic
- They are handled by the **deterministic AST layer**, not learned embeddings
- Including them in embedding training causes **embedding collapse** (all words become similar)
- Only **content words** (hundo, tablo, legi, bela) get learned embeddings
- This is a core architectural decision, not a workaround

**The Big Idea** (see `VISION.md`): Traditional LLMs waste capacity learning grammar. By making grammar explicit through ASTs, we hypothesize a 100M-500M param "reasoning core" could match larger models on structured tasks.

## MANDATORY: Schema-First Development (v2.2+)

**CRITICAL**: Klareco now has a comprehensive 4-layer semantic ontology in the database. You MUST use it instead of creating piecemeal solutions.

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

**1. AST access — do NOT use KuzuASTReconstructor**
```python
# ⚠️ MEASURED 2026-05: KuzuASTReconstructor.reconstruct_ast() is
# ~17,000 ms PER AST (not "<5ms" as previously claimed here — that
# figure was wrong by ~3400x). It issues many unindexed Kuzu
# traversals per sentence. NEVER use it.
#
# Until the store migration lands, re-parse on demand — parse(text)
# is ~milliseconds and is what the active retriever already does
# (commit 3dd0b73 switched off the reconstructor for this reason):
from klareco.parser import parse
ast = parse(text)

# Target architecture (see DuckDB de-risk, 2026-05): a flat store
# carries the parsed AST as a JSON blob; AST access becomes
# json.loads(ast_json) at ~0.9 ms — ~20,000x faster than the
# reconstructor and ~50x faster than re-parsing.
```

**2. ALWAYS query semantic ontology instead of hardcoded lists**
```python
# ❌ NEVER DO THIS:
PERSON_WORDS = ['homo', 'vir', 'kuracist']  # Hardcoded gazetteer!

# ✅ ALWAYS DO THIS:
result = kuzu_conn.execute("""
    MATCH (r:Radiko)-[:HAVAS_ENTECAN_TIPON]->(e:EntecaTipo {tipo_id: 'persono'})
    RETURN r.radiko
""")
```

**3. ALWAYS use verb/noun classes for synonyms**
```python
# ❌ NEVER DO THIS:
CREATION_VERBS = ['fond', 'kre', 'produk']  # Manual list!

# ✅ ALWAYS DO THIS:
result = kuzu_conn.execute("""
    MATCH (r:Radiko)-[:APARTENAS_AL_VERBA_KLASO]->(v:VerbaKlaso {klaso_id: 'kreado-26'})
    RETURN r.radiko
""")
```

**4. ALWAYS use schema slots for importance ranking**
```python
# ❌ NEVER DO THIS:
if question_type == 'WHO':
    importance = 0.95  # Hardcoded!

# ✅ ALWAYS DO THIS:
result = kuzu_conn.execute("""
    MATCH (s:SkemaSloto {sloto_id: 'ĉefa_realigo'})
    RETURN s.graveco_pezo
""")
```

**5. ALWAYS use thematic roles for answer extraction**
```python
# ❌ NEVER DO THIS:
if question.startswith('Kiu'):
    return ast['subjekto']  # Pattern matching!

# ✅ ALWAYS DO THIS:
# Query thematic roles to find Aganto (agent) in creation events
result = kuzu_conn.execute("""
    MATCH (v:Radiko)-[:APARTENAS_AL_VERBA_KLASO]->(vc:VerbaKlaso {klaso_id: 'kreado-26'})
    MATCH (v)-[:HAVAS_TEMAN_ROLON]->(tr:TemaRolo {rolo_id: 'aganto'})
    RETURN tr
""")
```

### Decision Checklist

Before implementing ANY feature, ask:

- [ ] Does this need entity classification? → Query EntecaTipo
- [ ] Does this need verb synonyms? → Query VerbaKlaso
- [ ] Does this need importance ranking? → Query SkemaSloto
- [ ] Am I creating a hardcoded list? → STOP, use ontology
- [ ] Am I re-parsing ASTs? → STOP, use reconstructor
- [ ] Am I pattern matching on word forms? → STOP, use semantic classes

### What to Do Instead of Piecemeal Solutions

| Piecemeal Approach | Schema-First Approach |
|--------------------|----------------------|
| Gazetteer of place names | Query `EntecaTipo {tipo_id: 'loko'}` |
| List of person words | Query `EntecaTipo {tipo_id: 'persono'}` |
| Time word patterns | Query `EntecaTipo {tipo_id: 'tempo'}` |
| Verb synonym lists | Query `VerbaKlaso` members |
| Hardcoded importance weights | Query `SkemaSloto.graveco_pezo` |
| Pattern matching for WHO | Query thematic role `aganto` |
| Slow `KuzuASTReconstructor` (~17,000ms/AST, measured) | Re-parse on demand `parse(text)` (~ms); target store: `json.loads(ast_json)` blob (~0.9ms) |

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
# Parse cleaned texts into a unified corpus with ASTs
./scripts/parse/parse_corpus.sh

# Build Kuzu v2.1 graph (export CSV → load → load ReVo → extend ontology)
./scripts/index/reindex_kuzu_v2.1.sh

# Build Whoosh BM25 index on top of the corpus
python scripts/index/build_whoosh_index.py
```

### End-to-end question answering
```bash
# Run the orchestrator pipeline on one question
python -m klareco run "Kiu fondis Esperanton?"

# Evaluate against a test set (local, single-process)
python scripts/eval/evaluate_extractive_qa.py \
    --test-set data/test_sets/qa_test_diverse_30.jsonl

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
| Parse corpus | `./scripts/parse/parse_corpus.sh` | Build unified corpus with ASTs |
| Re-index Kuzu | `./scripts/index/reindex_kuzu_v2.1.sh` | CSV export → Kuzu load → ReVo + ontology |
| Build Whoosh index | `python scripts/index/build_whoosh_index.py` | Build BM25 index over the corpus |
| Post-reparse pipeline | `./scripts/pipeline/post_reparse_pipeline.sh` | Schema + ReVo + Whoosh + eval (after a reparse) |
| Validate all | `./scripts/validate/validate_all.sh` | Run corpus + Kuzu integrity checks |

### Pipeline Stages

```
ACQUIRE  → Download raw data           (scripts/acquire/)
CLEAN    → Normalize text              (scripts/clean/)
EXTRACT  → Extract sentence JSONL      (scripts/extract/)
OCR      → PDF → text (PAG only)       (scripts/ocr/)
PARSE    → Parse to ASTs                (scripts/parse/)
INDEX    → Kuzu graph + Whoosh         (scripts/index/)
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
│   ├── pipeline.py         #   Orchestrator runner
│   ├── factory.py          #   build_default_pipeline()
│   ├── context.py          #   QueryContext / ContextDelta dataclasses
│   ├── phase_timer.py      #   Sub-stage timing
│   └── stages/             #   Parse → Retrieve → DetRerank → Rerank
│                           #   → ExtractGenerate → FormatOutput
├── rag/                    # Retrieval + extraction
│   ├── whoosh_retriever.py #   Main retriever (BM25 ∩ AST roles)
│   ├── unified_extractor.py#   Fact extraction
│   ├── extractive_answering.py
│   ├── ast_semantic_ranker.py
│   ├── kuzu_ast_reconstructor.py
│   ├── importance_scorer.py
│   ├── grammatical_variants.py
│   ├── discourse_planner.py
│   ├── question_classifier.py
│   └── entity_recognizer.py
├── knowledge/              # Kuzu-backed vocabularies (no hardcoded lists)
├── ontology/               # Kuzu query API for the 4-layer ontology
├── schema/                 # Kuzu v2.1 DDL
├── eval/qa_metrics.py      # Shared evaluator (local + Modal)
└── utils/kuzu_open.py      # Single Kuzu opener with env-var memory caps

scripts/
├── acquire/      # Raw downloads from upstream sources
├── clean/        # Raw → cleaned text
├── extract/      # Cleaned → sentence JSONL
├── ocr/          # PDF → text (PAG)
├── parse/        # Build unified corpus from extracted sentences
├── index/        # Corpus → Kuzu graph + Whoosh BM25 index
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
with deterministic-first evaluation. See `DESIGN.md` for the full picture.

**Production-Ready**:
- 16-rule parser + deparser (91.8% parse rate on 4.2M sentences)
- v2.1 Kuzu graph with 4-layer semantic ontology
- WhooshRetriever with AST-role matching
- DeterministicRerankStage (question-type AST boost)
- UnifiedASTExtractor + ExtractiveAnswerGenerator
- `klareco.eval` shared by local + Modal cloud evaluators
- Proper-noun dictionary v3 (cleaned + Wikipedia-category enriched)

**In Progress**: Iterative QA-accuracy improvements driven by retrieval-rank
metrics — see [EPIC #713](https://github.com/marctjones/klareco/issues/713).

**Deferred** (working code on disk but not in the active loop): Stage 1
root embeddings, M1 selectional preference, neural cross-encoder reranker,
entity classifier, summarization stack. These return once the deterministic
floor is stable enough to attribute a measurable improvement to a specific
learned component.

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
