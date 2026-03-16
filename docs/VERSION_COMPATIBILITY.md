# Klareco Version Compatibility Matrix

This document tracks compatibility across major versions of Klareco's components.

## Current Versions

- **Database Schema**: v2.1 (Pure Esperanto)
- **Models**: v2.0 (needs retraining → v3.0)
- **Pipeline**: v2.1 (AST-based)
- **CLI**: v1.0 (basic commands)

## Version History

### v1.0 (Deprecated)

**Status**: Superseded by v2.0
**Maintained**: No
**Scripts Archived**: `scripts/archive/v1.0/`

| Component | Implementation |
|-----------|----------------|
| Database Schema | Simple flat corpus |
| Root Embeddings | 18K vocabulary (includes function words!) |
| M1 Selectional | Not implemented |
| Pipeline | Text → Parse → Embed |
| CLI | Basic parse/query commands |

**Problems**:
- Function words in root embeddings (embedding collapse)
- No tier classification
- No provenance tracking

### v2.0 (Partially Deprecated)

**Status**: Database migrated to v2.1, models need retraining
**Maintained**: Database only (v2.1 is evolution of v2.0)
**Scripts Archived**: `scripts/archive/v2.0/`

| Component | Implementation |
|-----------|----------------|
| Database Schema | 6-tier classification (tier0-tier5) |
| Root Embeddings | 10K vocabulary (still has function words!) |
| M1 Selectional | Basic implementation |
| Pipeline | Text → Parse → Embed → M1 |
| CLI | Parse, query, translate |

**Improvements**:
- Added tier classification (tier0-tier5)
- Added provenance tracking
- Basic M1 selectional preference

**Problems**:
- Still includes function words in embeddings
- Not using AST-annotator pattern
- Mixed language schema (English + Esperanto)

### v2.1 (Current - Database)

**Status**: Production for database, models need retraining → v3.0
**Maintained**: Yes
**Scripts**: `scripts/` (root level, needs organization)

| Component | Implementation |
|-----------|----------------|
| Database Schema | ✅ Pure Esperanto (Radiko, Vorto, Frazo) |
| Root Embeddings | ❌ Needs retraining (v2.0 models incompatible) |
| M1 Selectional | ❌ Needs retraining (v2.0 models incompatible) |
| Pipeline | ⚠️ Partial (parse works, models don't) |
| CLI | ✅ Basic commands (parse, query, translate) |

**Database Schema**:
```cypher
(Radiko {radiko: str, nivelo: str, fonto: str, ofteco: int})
(Vorto {vorto: str})
(Frazoteksto {teksto: str})
(Radiko)-[:APERIS_EN]->(Vorto)
(Vorto)-[:EN]->(Frazoteksto)
```

**Data Statistics**:
- 1,248,082 Radiko nodes
- 77,913,734 Vorto nodes
- 5,442,136 Frazoteksto nodes

**Tier Distribution**:
- Tier 0 (function words): 187 roots
- Tier 1a (Unua Libro): 750 roots
- Tier 1b (Fundamento): 1,403 roots
- Tier 2 (ReVo): 7,646 roots
- Tier 3 (Corpus): 1,237,596 roots
- Tier 5 (Parse failures): 500 roots

**Improvements**:
- Pure Esperanto schema
- Complete provenance tracking (fonto, nivelo)
- Frequency data (ofteco)
- Proper tier classification

### v3.0 (Target - Models + CLI)

**Status**: In development (Epic #616 + CLI Epics #637-642)
**Target Date**: Q1 2025
**Scripts**: Will be in `scripts/data/`, `scripts/train/`, etc.

| Component | Implementation |
|-----------|----------------|
| Database Schema | ✅ v2.1 (no changes) |
| Root Embeddings | 🚧 9.8K tier-filtered (excludes tier0+tier5) |
| M1 Selectional | 🚧 AST-annotator pattern |
| Entity Classifier | 🚧 Deterministic tier1-2 + learned tier3 |
| M2.1 Taxonomy | 🚧 90% deterministic (ReVo+ConceptNet) |
| M2.2 Coreference | 🚧 80% deterministic (grammar matching) |
| Pipeline | 🚧 Full AST-annotator chain |
| CLI | 🚧 Complete lifecycle management |

**Root Embeddings v3.0**:
- Vocabulary: 9,800 roots (tier1a+1b+2)
- **Excludes tier0**: No function words (prevents collapse)
- **Excludes tier5**: No parse failures (garbage data)
- Embedding dimension: 64d
- Training objective: Semantic similarity
- Format: Uses ASTAnnotator protocol

**M1 Selectional v3.0**:
- Input: AST with root_embedding annotations
- Output: M1_plausibility score
- Architecture: Reads embeddings, doesn't re-learn grammar
- Training data: (subject, verb, object) triples from v2.1 DB

**Pipeline v3.0**:
```python
ast = parser.parse(text)              # M0: Deterministic
ast = root_embeddings.annotate(ast)   # Learned: 64d roots
ast = compositional.annotate(ast)     # Learned: 128d words
ast = m1_selectional.annotate(ast)    # Learned: plausibility
ast = entity_classifier.annotate(ast) # Mostly deterministic
ast = taxonomy.annotate(ast)          # 90% deterministic
ast = coreference.annotate(ast)       # 80% deterministic
```

**CLI v3.0**:
- `klareco inspect ast/annotations/tensor` - Inspection tools
- `klareco train roots/m1/entity` - Training commands
- `klareco data export roots/m1` - Data export
- `klareco test pipeline/model` - Testing commands

## Compatibility Matrix

| Component | v1.0 | v2.0 | v2.1 | v3.0 |
|-----------|------|------|------|------|
| **Database Schema** |
| Format | Flat corpus | 6-tier | Pure EO | Pure EO |
| Tier classification | ❌ | ✅ | ✅ | ✅ |
| Provenance tracking | ❌ | ⚠️ | ✅ | ✅ |
| Frequency data | ❌ | ❌ | ✅ | ✅ |
| **Root Embeddings** |
| Vocabulary size | 18K | 10K | 9.8K | 9.8K |
| Function words included | ✅ (BAD) | ✅ (BAD) | ❌ (GOOD) | ❌ (GOOD) |
| Tier filtering | ❌ | ❌ | ✅ | ✅ |
| Embedding dimension | 64d | 64d | 64d | 64d |
| **M1 Selectional** |
| Implemented | ❌ | ✅ | ⚠️ | ✅ |
| Uses AST-annotator | ❌ | ❌ | ❌ | ✅ |
| Tier-aware | ❌ | ⚠️ | ⚠️ | ✅ |
| **Entity Classifier** |
| Implemented | ❌ | ❌ | ❌ | ✅ |
| Deterministic tier1-2 | ❌ | ❌ | ❌ | ✅ |
| **M2.1 Taxonomy** |
| Implemented | ❌ | ❌ | ❌ | ✅ |
| Deterministic % | - | - | - | 90% |
| **M2.2 Coreference** |
| Implemented | ❌ | ❌ | ❌ | ✅ |
| Deterministic % | - | - | - | 80% |
| **Pipeline** |
| Architecture | Text→Parse | Text→Embed | DB→Parse→Embed | DB→AST-annotator chain |
| Explainability | ❌ | ⚠️ | ⚠️ | ✅ |
| Tensor annotations | ❌ | ❌ | ❌ | ✅ |
| **CLI** |
| Commands | 2 | 5 | 5 | 20+ |
| Inspection tools | ❌ | ❌ | ❌ | ✅ |
| Training commands | ❌ | ❌ | ❌ | ✅ |
| Data export | ❌ | ❌ | ❌ | ✅ |

## Migration Paths

### v1.0 → v2.1

**Status**: Not supported
**Reason**: Database schema incompatible, models obsolete
**Recommendation**: Start from scratch with v2.1 database

### v2.0 → v2.1

**Status**: Database migration complete ✅
**Models**: Need retraining (Epic #616)
**Steps**:
1. ✅ Database migrated to Pure Esperanto schema
2. ✅ Tier classification complete (1.2M Radiko nodes)
3. 🚧 Retrain root embeddings with tier filtering (#617-620)
4. 🚧 Retrain M1 with new embeddings (#621-624)
5. 🚧 Implement remaining models (#625-632, #637-645)

### v2.1 → v3.0

**Status**: In progress (Epic #616 + CLI Epics)
**Database**: No changes (v2.1 database works with v3.0 models)
**Models**: Complete retraining required
**CLI**: New commands added (backward compatible)
**Steps**:
1. 🚧 Implement AST-annotator infrastructure (#633-636)
2. 🚧 Implement CLI inspection tools (#637-639)
3. 🚧 Retrain all models with ASTAnnotator pattern (#617-632)
4. 🚧 Implement CLI training/data commands (#640-641)
5. 🚧 Implement CLI debugging tools (#642)

## Deprecation Timeline

| Version | Deprecated Date | Removal Date | Status |
|---------|----------------|--------------|--------|
| v1.0 | 2024-12-01 | 2025-03-01 | Archived |
| v2.0 models | 2025-01-15 | 2025-03-01 | Training new v3.0 |
| v2.0 scripts | 2025-01-15 | 2025-04-01 | Migrating to CLI |
| v2.1 database | - | - | Current |

## Checking Your Version

### Database Schema Version

```bash
# Query schema version
python -c "
import kuzu
db = kuzu.Database('data/indexes/v2.1_kuzu_index_full', read_only=True)
conn = kuzu.Connection(db)

# Check for v2.1 Pure Esperanto schema
result = conn.execute('MATCH (r:Radiko) RETURN r.nivelo LIMIT 1')
if result.has_next():
    print('Database: v2.1 (Pure Esperanto)')
else:
    print('Database: v2.0 or earlier')
"
```

### Model Version

```bash
# Check root embeddings version
python -c "
import torch
checkpoint = torch.load('models/root_embeddings/best_model.pt')

if 'version' in checkpoint:
    print(f'Root Embeddings: {checkpoint[\"version\"]}')
elif checkpoint['vocab_size'] > 15000:
    print('Root Embeddings: v1.0 (deprecated)')
elif checkpoint['vocab_size'] > 9000:
    print('Root Embeddings: v2.0 (needs retraining)')
else:
    print('Root Embeddings: Unknown version')

# Check if function words present (bad!)
if 'mi' in checkpoint['root_to_idx']:
    print('WARNING: Function words in vocabulary (v1.0/v2.0 issue)')
"
```

### CLI Version

```bash
klareco info | grep "CLI Version"
# v1.0: Basic commands only
# v3.0: Full lifecycle management
```

## Related Documentation

- `docs/CLI_ARCHITECTURE.md` - CLI design and versioning
- `docs/V2.1_DATABASE_CLASSIFICATION_COMPLETE.md` - v2.1 database details
- `docs/TRAINING_READY_SUMMARY.md` - v3.0 training preparation
- Epic #616 - v3.0 model retraining plan
- Epics #637-642 - v3.0 CLI implementation

## Questions?

- **Which version should I use?** v2.1 database + v3.0 models (Epic #616)
- **Can I use v2.0 models?** No, they include function words (embedding collapse issue)
- **When will v3.0 be ready?** Target: Q1 2025 (2-3 weeks for Phase 0-1)
- **Can I mix versions?** v2.1 database works with v3.0 models, but NOT with v2.0 models

## Version Updates

- **2025-01-15**: Added v3.0 target, CLI versioning, ASTAnnotator protocol
- **2024-12-15**: Completed v2.1 database migration
- **2024-11-01**: Discovered function word issue in v2.0 embeddings
- **2024-10-01**: Released v2.0 with tier classification
