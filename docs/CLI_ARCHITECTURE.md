# Klareco CLI Architecture Design

## Vision

A single, coherent CLI (`python -m klareco`) that provides:
1. **Lifecycle Management**: Manage the AI system from data → training → evaluation → deployment
2. **Pipeline Inspection**: Inspect ASTs and annotations at any pipeline stage
3. **Debugging Tools**: Decode tensors, trace annotations, understand model decisions
4. **Testing**: Run validation at every level (data, models, pipeline)

## Current State (2025-01)

### Existing Commands
- `klareco parse` - Parse Esperanto to AST
- `klareco query` - Query corpus with retrieval
- `klareco translate` - Translate to/from Esperanto
- `klareco corpus` - Corpus management
- `klareco info` - System information

### Problems
- **Scattered scripts**: 100+ scripts in `scripts/` with no consistent interface
- **No inspection tools**: Can't inspect ASTs, annotations, or tensors
- **No training commands**: Training is via shell scripts, not CLI
- **No testing commands**: Testing is via pytest, not integrated
- **No pipeline debugging**: Can't trace how annotations flow through models

## Proposed Architecture

### Design Principles

1. **Single Entry Point**: `python -m klareco <command> <subcommand>`
2. **Verb-Noun Pattern**: Commands follow `klareco <action> <target>` (e.g., `klareco train roots`, `klareco inspect ast`)
3. **Pipeline Stages as First-Class**: Every pipeline stage has corresponding CLI commands
4. **JSON I/O**: Everything can output JSON for scripting/piping
5. **Interactive Mode**: Commands support `-i/--interactive` for exploration

### Command Structure

```
klareco
├── parse         # Parse text → AST (existing)
├── translate     # Translate to/from Esperanto (existing)
├── info          # System information (existing)
│
├── data          # Data pipeline commands
│   ├── extract   # Extract sentences from sources
│   ├── validate  # Validate corpus quality
│   ├── stats     # Show corpus statistics
│   └── export    # Export training data from DB
│
├── train         # Training commands
│   ├── roots     # Train root embeddings
│   ├── m1        # Train M1 selectional preference
│   ├── entity    # Train entity classifier
│   ├── reranker  # Train reranker
│   └── status    # Show training status (checkpoints, metrics)
│
├── annotate      # Run annotation pipeline
│   ├── text      # Annotate text through full pipeline
│   ├── file      # Annotate file (batch)
│   └── stages    # Show available annotation stages
│
├── inspect       # Inspection/debugging tools
│   ├── ast       # Inspect AST structure
│   ├── annotations # Show all annotations in AST
│   ├── tensor    # Decode/visualize tensor annotation
│   ├── pipeline  # Trace annotation flow through pipeline
│   └── model     # Inspect model (vocabulary, embeddings, etc.)
│
├── test          # Testing commands
│   ├── parse     # Test parser on examples
│   ├── model     # Test specific model
│   ├── pipeline  # Test full pipeline
│   └── quality   # Run quality metrics
│
├── query         # RAG/retrieval commands (existing, expand)
│   ├── search    # Search corpus (existing functionality)
│   ├── rag       # Full RAG pipeline
│   └── explain   # Explain retrieval decisions
│
└── db            # Database commands
    ├── query     # Query Kuzu database
    ├── stats     # Database statistics
    └── validate  # Validate database integrity
```

## Priority Implementation Order

### Phase 1: Critical Inspection Tools (Epic #637)
**Goal**: Be able to inspect ASTs and annotations during development

1. **`klareco inspect ast`** - Show AST structure with deterministic features
2. **`klareco inspect annotations`** - Show all annotations in AST
3. **`klareco inspect tensor`** - Decode tensor embeddings to similar words
4. **`klareco annotate text`** - Run text through annotation pipeline

**Use case**: During model development, inspect what each model adds

### Phase 2: Training Lifecycle (Epic #638)
**Goal**: Train models via CLI instead of shell scripts

1. **`klareco train roots`** - Train root embeddings
2. **`klareco train m1`** - Train M1 selectional
3. **`klareco train status`** - Show training progress/checkpoints
4. **`klareco test model`** - Test trained model quality

**Use case**: Complete training workflow through CLI

### Phase 3: Data Pipeline (Epic #639)
**Goal**: Generate training data from v2.1 database

1. **`klareco data export roots`** - Export root training data from DB
2. **`klareco data export m1`** - Export M1 training data from DB
3. **`klareco data validate`** - Validate training data quality
4. **`klareco data stats`** - Show data statistics

**Use case**: Single source of truth (database → training data)

### Phase 4: Pipeline Debugging (Epic #640)
**Goal**: Understand annotation flow and model decisions

1. **`klareco inspect pipeline`** - Trace annotations through pipeline
2. **`klareco query explain`** - Explain retrieval ranking decisions
3. **`klareco test pipeline`** - Test full pipeline with examples

**Use case**: Debug why model makes certain predictions

## Key Features

### 1. AST Inspection (`klareco inspect ast`)

```bash
# Show full AST
klareco inspect ast "La hundo vidis la katon."

# Show only deterministic features
klareco inspect ast "La hundo vidis la katon." --deterministic-only

# Output JSON
klareco inspect ast "La hundo vidis la katon." --json > ast.json

# Interactive mode
klareco inspect ast -i
```

**Output Example**:
```
=== AST Structure ===
Sentence type: frazo

Subject: La hundo
  └─ Word: hundo (substantivo)
     ├─ Root: hund
     ├─ Case: nominativo (deterministic from -o)
     ├─ Number: singularo (deterministic, no -j)
     └─ Annotations: (none yet)

Verb: vidis
  └─ Word: vidis (verbo)
     ├─ Root: vid
     ├─ Tense: estinto (deterministic from -is)
     └─ Annotations: (none yet)

Object: la katon
  └─ Word: katon (substantivo)
     ├─ Root: kat
     ├─ Case: akuzativo (deterministic from -n)
     ├─ Number: singularo (deterministic, no -j)
     └─ Annotations: (none yet)
```

### 2. Annotation Inspection (`klareco inspect annotations`)

```bash
# Show all annotations
klareco inspect annotations ast.json

# Show specific annotation
klareco inspect annotations ast.json --key root_embedding

# Show annotation summary (where each annotation appears)
klareco inspect annotations ast.json --summary
```

**Output Example**:
```
=== Annotation Summary ===

root_embedding (tensor: [64])
  ├─ subjekto.kerno (hund)
  ├─ verbo (vid)
  └─ objekto (kat)

M1_plausibility (float: 0.8734)
  └─ <root> (sentence-level)

entity_type (string)
  ├─ subjekto.kerno → ANIMALO
  └─ objekto → ANIMALO
```

### 3. Tensor Decoding (`klareco inspect tensor`)

```bash
# Decode tensor to similar words
klareco inspect tensor ast.json --key root_embedding --node verbo

# Show top-10 similar roots
klareco inspect tensor ast.json --key root_embedding --node verbo --top-k 10

# Visualize embedding space (2D projection)
klareco inspect tensor ast.json --key root_embedding --visualize
```

**Output Example**:
```
=== Tensor Decoding: root_embedding (verbo) ===
Root: vid
Embedding: torch.tensor([0.12, -0.34, 0.56, ...]) (64d)

Most similar roots:
  1. rigar (0.89) - to look at, to gaze
  2. observ (0.84) - to observe
  3. spekti (0.81) - to watch
  4. vidi (0.79) - to see (variant)
  5. rigard (0.77) - look, gaze (noun)
  ...

Can't fully decode: This is a learned semantic representation.
Use --similar to explore semantic neighborhood.
```

### 4. Pipeline Tracing (`klareco inspect pipeline`)

```bash
# Trace annotations through full pipeline
klareco inspect pipeline "La hundo vidis la katon."

# Trace specific stages
klareco inspect pipeline "La hundo vidis la katon." --stages parse,root-embeddings,m1

# Show intermediate ASTs
klareco inspect pipeline "La hundo vidis la katon." --show-asts
```

**Output Example**:
```
=== Pipeline Trace ===

Stage 1: M0 Parser
  Added: AST structure (subjekto, verbo, objekto)
  Added: Deterministic features (kazo, nombro, tempo)
  Time: 2ms

Stage 2: Root Embeddings
  Read: radiko (deterministic)
  Added: root_embedding (64d tensor) to 3 words
  Time: 1ms

Stage 3: Compositional Embeddings
  Read: root_embedding (previous annotation)
  Read: prefikso, sufiksoj (deterministic)
  Added: word_embedding (128d tensor) to 3 words
  Time: 1ms

Stage 4: M1 Selectional
  Read: root_embedding from subjekto, verbo, objekto
  Added: M1_plausibility = 0.8734 (sentence-level)
  Time: 3ms

Stage 5: Entity Classifier
  Read: word_embedding
  Added: entity_type = 'ANIMALO' to subjekto, objekto
  Time: 2ms

Total time: 9ms
Total annotations: 11
```

### 5. Training via CLI (`klareco train`)

```bash
# Train root embeddings
klareco train roots --fresh

# Resume training from checkpoint
klareco train roots --resume

# Train with specific config
klareco train roots --config configs/root_embeddings.yaml

# Show training status
klareco train status roots
```

**Output Example**:
```
=== Training Status: Root Embeddings ===

Checkpoint: models/root_embeddings/best_model.pt
Epoch: 12/50
Loss: 0.0234
Vocabulary: 9,800 roots (tier1a+1b+2)

Last trained: 2025-01-15 14:23:00
Training time: 2h 34m

Validation metrics:
  - Similarity accuracy: 87.3%
  - Mean cosine similarity: 0.42 (no collapse ✓)
  - Fundamento coverage: 100%

Status: ✓ Ready for downstream training
```

## Naming Conventions and Versioning

### Script Naming Strategy

All scripts follow a consistent naming pattern:

```
<stage>_<target>_<version>.py
```

Examples:
- `data_export_roots_v2.py` - Works with v2.1 database schema
- `train_roots_v3.py` - Trains root embeddings for v3 architecture
- `evaluate_m1_v2.py` - Evaluates M1 model for v2 system

### Versioning Scheme

**v1.0** (Deprecated) - Original single-tier corpus, simple embeddings
**v2.0** (Current) - Six-tier classification, AST-native schema
**v2.1** (Current) - Pure Esperanto schema (Radiko, Vorto, Frazo)
**v3.0** (Future) - Full annotation pipeline with deterministic-first architecture

### Docstring Convention

Every script MUST include a header docstring with:

```python
"""
<Script Name>

VERSION: v2.1
COMPATIBLE WITH: v2.1 database schema, AST-native architecture
DEPENDENCIES: Root Embeddings v2, M1 v2
STAGE: Training | Data | Evaluation | Inspection

Description:
    Brief description of what the script does.

Pipeline Position:
    DB (v2.1) → [THIS SCRIPT] → Root Embeddings → M1 → ...

Usage:
    python script.py --arg1 value1

Inputs:
    - Input 1: Description (format, source)
    - Input 2: Description (format, source)

Outputs:
    - Output 1: Description (format, location)
    - Output 2: Description (format, location)

Last Updated: 2025-01-15
Author: <name>
Related Issues: #123, #456
"""
```

### Example: Complete Script Header

```python
"""
Export Root Embedding Training Data from v2.1 Database

VERSION: v2.1
COMPATIBLE WITH: v2.1 database schema (Radiko nodes with tier/source/ofteco)
DEPENDENCIES: None (queries database directly)
STAGE: Data

Description:
    Extracts root words from v2.1 Kuzu database for root embedding training.
    Filters to tier1a+1b+2 (core vocabulary), excludes tier0 function words
    and tier5 parse failures.

Pipeline Position:
    v2.1 DB → [THIS SCRIPT] → data/training/roots_v2.1.json → train_roots_v3.py

Usage:
    python scripts/data_export_roots_v2.py \\
        --kuzu data/indexes/v2.1_kuzu_index_full \\
        --output data/training/roots_v2.1.json \\
        --tiers 1a,1b,2 \\
        --min-ofteco 5

Inputs:
    - Kuzu database: data/indexes/v2.1_kuzu_index_full
    - Schema: Radiko nodes with nivelo, fonto, ofteco properties

Outputs:
    - Training vocabulary: data/training/roots_v2.1.json
      Format: {"radiko": str, "ofteco": int, "fonto": str, "nivelo": str}[]

Quality Checks:
    - No tier0 function words included (validates with tier0_grammatical_words.json)
    - All Fundamento roots present (1,403 tier1a+1b roots)
    - Vocabulary size: ~9,800 roots

Last Updated: 2025-01-15
Author: Claude + Marc
Related Issues: #617 (Root Embeddings Data Export)
See Also: docs/TRAINING_READY_SUMMARY.md, docs/V2.1_DATABASE_CLASSIFICATION_COMPLETE.md
"""
```

### CLI Command Naming

CLI commands follow verb-noun pattern and include version in help text:

```python
@subcommand('train', 'roots')
def cmd_train_roots(args):
    """
    Train root embeddings (v3 architecture).

    Compatible with: v2.1 database
    Outputs: Root embeddings v3 (64d, tier-filtered)
    """
    pass
```

### Version Compatibility Matrix

Maintain a compatibility matrix in `docs/VERSION_COMPATIBILITY.md`:

| Component | v1.0 | v2.0 | v2.1 | v3.0 |
|-----------|------|------|------|------|
| Database Schema | Simple | 6-tier | Pure EO | Pure EO |
| Root Embeddings | 18K vocab | 10K vocab | 9.8K tier-filtered | 9.8K tier-filtered |
| M1 Selectional | N/A | Basic | Tier-aware | AST-annotator |
| Pipeline | Text→Parse | Text→Parse→Embed | DB→Parse→Embed | DB→Parse→Annotate chain |

### Deprecation Strategy

When deprecating scripts:

1. **Mark as deprecated** in docstring:
```python
"""
DEPRECATED: Use `klareco train roots` instead.
This script is kept for compatibility with v1.0 system only.
Will be removed in v3.0.
"""
```

2. **Add runtime warning**:
```python
warnings.warn(
    "train_roots_v1.py is deprecated. Use `klareco train roots` instead.",
    DeprecationWarning
)
```

3. **Move to archive**:
```bash
scripts/
  archive/
    v1.0/
      train_roots_v1.py  # Old v1.0 scripts
    v2.0/
      train_roots_v2.py  # Superseded by v2.1
```

### Script Organization

```
scripts/
├── data/              # Data pipeline scripts
│   ├── export_roots_v2.1.py
│   ├── export_m1_v2.1.py
│   └── validate_corpus_v2.py
│
├── train/             # Training scripts
│   ├── roots_v3.py
│   ├── m1_v3.py
│   └── entity_v2.py
│
├── evaluate/          # Evaluation scripts
│   ├── embedding_quality_v2.py
│   └── m1_accuracy_v2.py
│
├── inspect/           # Inspection/debugging scripts
│   ├── ast_inspector_v2.py
│   └── tensor_decoder_v2.py
│
├── util/              # Utilities (version-agnostic)
│   ├── kuzu_query.py
│   └── vocabulary_tools.py
│
└── archive/           # Deprecated scripts
    ├── v1.0/
    └── v2.0/
```

### Version Tracking in Code

Every model checkpoint includes version metadata:

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'version': 'v3.0',
    'compatible_with': {
        'database_schema': 'v2.1',
        'pipeline': 'v3.0'
    },
    'training_data': {
        'source': 'v2.1 database',
        'tiers': ['1a', '1b', '2'],
        'vocab_size': 9800
    },
    'created': '2025-01-15T14:23:00Z',
    'git_commit': 'abc123...'
}
```

### Migration Checklist

When updating to a new version:

- [ ] Update script docstrings with new version
- [ ] Update VERSION_COMPATIBILITY.md matrix
- [ ] Move superseded scripts to archive/
- [ ] Update CLAUDE.md with new version info
- [ ] Add deprecation warnings to old scripts
- [ ] Update CLI help text
- [ ] Update related issues/PRs

## Implementation Notes

### JSON I/O Convention

All commands support `--json` flag for machine-readable output:

```bash
klareco inspect ast "Text" --json | jq '.verbo.radiko'
# Output: "vid"

# Pipeline commands
klareco inspect annotations ast.json --json | jq '.root_embedding | length'
# Output: 3
```

### Interactive Mode

Commands with `-i/--interactive` support REPL-style exploration:

```bash
$ klareco inspect ast -i
klareco> La hundo vidis la katon.
[Shows AST]
klareco> .annotations
[Shows all annotations]
klareco> .tensor verbo.root_embedding
[Decodes tensor]
klareco> .exit
```

### Tensor Serialization

When outputting ASTs with tensors to JSON:
- Default: Convert tensors to lists (JSON-serializable)
- Use `--keep-tensors` to output special format with tensor metadata

```json
{
  "verbo": {
    "radiko": "vid",
    "annotations": {
      "root_embedding": {
        "_type": "torch.Tensor",
        "_shape": [64],
        "_dtype": "float32",
        "_data": [0.12, -0.34, 0.56, ...]
      }
    }
  }
}
```

## Testing Strategy

### Unit Tests
- Test each CLI command in isolation
- Mock models/annotators for speed
- Test JSON output parsing

### Integration Tests
- Test full pipeline: `klareco annotate text "..." | klareco inspect annotations`
- Test training: `klareco train roots --dry-run`
- Test data export: `klareco data export roots --dry-run`

### End-to-End Tests
- Test realistic workflow: data → train → evaluate → deploy
- Test debugging workflow: inspect → trace → explain

## Migration Plan

### Phase 1: Keep Existing Scripts (Don't Break Things)
- Add new CLI commands alongside existing scripts
- Scripts remain functional, CLI is additive
- Document equivalents: `scripts/train_roots.sh` = `klareco train roots`

### Phase 2: Gradually Deprecate Scripts
- As CLI commands stabilize, mark scripts as deprecated
- Add warnings to scripts: "Use `klareco train roots` instead"
- Move old scripts to `scripts/archive/`

### Phase 3: CLI as Primary Interface
- Update documentation to use CLI commands
- Remove or archive deprecated scripts
- Shell scripts become thin wrappers around CLI

## Documentation

### User Guide
- `docs/CLI_USER_GUIDE.md` - How to use each command
- `docs/CLI_WORKFLOWS.md` - Common workflows (training, debugging, testing)
- `docs/CLI_REFERENCE.md` - Complete command reference

### Developer Guide
- `docs/CLI_DEVELOPMENT.md` - How to add new commands
- `docs/CLI_TESTING.md` - Testing CLI commands

## Related Issues

- Epic #637: Phase 1 - Inspection Tools
- Epic #638: Phase 2 - Training Lifecycle
- Epic #639: Phase 3 - Data Pipeline
- Epic #640: Phase 4 - Pipeline Debugging
- Issue #633-636: AST Annotator Infrastructure (prerequisite)

## Success Metrics

1. **Coverage**: All common workflows have CLI commands (no need for ad-hoc scripts)
2. **Usability**: New contributors can train models using only CLI (no script hunting)
3. **Debuggability**: Can trace any prediction back to source using `inspect` commands
4. **Consistency**: All commands follow same patterns (flags, output, error handling)

## Future Enhancements

- **Web UI**: `klareco serve` - Launch web UI for visual pipeline inspection
- **Logging**: `klareco logs` - View training/inference logs
- **Deployment**: `klareco deploy` - Deploy model to production
- **Monitoring**: `klareco monitor` - Monitor production metrics
