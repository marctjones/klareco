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

### Corpus Management
```bash
# Build corpus from cleaned texts
python scripts/build_corpus_v2.py \
  --cleaned-dir data/cleaned \
  --output data/corpus_with_sources_v2.jsonl \
  --min-parse-rate 0.5

# Build retrieval index
python scripts/index_corpus.py \
  --corpus data/corpus_with_sources_v2.jsonl \
  --output data/corpus_index_v3 \
  --batch-size 32
```

### Training Models
```bash
# Train root embeddings (Stage 1)
./scripts/run_fundamento_training.sh

# Full training pipeline
./scripts/run_full_training.sh
```

### RAG Demo
```bash
# Interactive retrieval demo
python scripts/demo_rag.py --interactive

# Single query
python scripts/demo_rag.py "Kio estas la Unu Ringo?"
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
   ./scripts/run_corpus_rebuild.sh --fresh   # Example
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
| Corpus rebuild | `./scripts/run_corpus_rebuild.sh` | Full corpus rebuild pipeline |
| Fundamento training | `./scripts/run_fundamento_training.sh` | Train on Fundamento data |
| Full training | `./scripts/run_full_training.sh` | Complete training pipeline |

## Code Organization

```
klareco/
├── parser.py               # 16 Esperanto rules → AST
├── deparser.py             # AST → text reconstruction
├── ast_to_graph.py         # AST → PyG graph for neural models
├── embeddings/
│   └── compositional.py    # Morpheme-level embeddings (500K params)
├── models/
│   └── tree_lstm.py        # TreeLSTMEncoder for sentence embeddings
├── rag/
│   └── retriever.py        # Two-stage retrieval (structural + neural)
├── experts/
│   ├── extractive.py       # Template-based answering
│   └── summarizer.py       # AST-based summarization
└── orchestrator.py         # Intent routing and pipeline coordination

scripts/
├── build_corpus_v2.py      # Corpus builder with quality filtering
├── create_training_corpus.py  # Filter corpus for training
└── index_corpus.py         # Build FAISS retrieval index
```

## Data Files (Not in Git)

```
data/
├── corpus/                       # Parsed corpus files
│   ├── unified_corpus.jsonl     # Main unified corpus with ASTs
│   └── authoritative_corpus.jsonl  # Tier 1-3 authoritative sources
├── training/                     # Training-ready filtered data
│   ├── authoritative_training.jsonl  # High-quality Fundamento/Krestomatio
│   ├── literature_training.jsonl     # Tier 5 literature
│   └── general_training.jsonl        # Tier 6 Wikipedia (filtered)
├── corpus_index_v3/              # FAISS index + metadata
└── vocabularies/                 # Root/prefix/suffix vocabularies
    ├── root_vocab.json          # Roots from corpus
    ├── prefix_vocab.json        # Esperanto prefixes
    └── suffix_vocab.json        # Esperanto suffixes

models/
└── root_embeddings/              # Stage 1 root embedding model
```

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
For training or corpus building, use shell scripts to run in separate terminal to save Claude context:
```bash
./scripts/run_semantic_training.sh  # Logs to models/semantic_similarity/training.log
```

Monitor with: `tail -f models/semantic_similarity/training.log`

## Current Development Status

**Production-Ready**:
- Parser (16 rules, 91.8% parse rate on corpus)
- Two-stage retrieval (structural + neural)
- Compositional embeddings (implemented, 320K params)
- RAG demo with extractive answering

**In Progress** (see `IMPLEMENTATION_ROADMAP_V2.md`):
- Semantic similarity training (data ready, model training)
- Integration into retrieval pipeline

**Next Steps**:
1. Complete semantic similarity model training (running)
2. Integrate semantic model into retriever
3. AST Trail system for explainability
4. AST-based reasoning patterns

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
- ✅ KEEP: `CORPUS_BUILDING.md` - Step-by-step guide to run `./scripts/run_wikipedia_extraction.sh`
- ❌ MOVE: `RAG_SYSTEM.md` - Explains RAG architecture concepts (→ Wiki)
- ❌ MOVE: `SESSION_SUMMARY.md` - Notes from development session (→ Discussion)

## See Also

- `VISION.md` - Long-term architecture goals
- `DESIGN.md` - Technical design decisions
- `IMPLEMENTATION_ROADMAP_V2.md` - Detailed development plan
- `README.md` - Usage examples and current status
- `16RULES.MD` - Specification of Esperanto grammar rules
