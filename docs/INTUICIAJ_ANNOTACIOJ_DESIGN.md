# Intuiciaj Annotacioj (Gut Annotations) - Design Document

**Status:** DESIGN PROPOSAL - Not yet implemented
**Created:** 2026-03-22
**Purpose:** Extend v2.1 AST schema with semantic intuition annotations

---

## Executive Summary

This document proposes adding **gut feeling annotations** to Klareco's v2.1 AST system using **pure Esperanto terminology**. These annotations capture semantic intuitions (confidence, salience, coherence, surprise) that guide retrieval and reasoning while maintaining full explainability.

**Key insight:** ASTs represent explicit thoughts (pensoj), gut annotations represent implicit feelings (sentoj) about those thoughts. Together they form a complete cognitive system.

**Implementation strategy:** Uses existing v2.1 annotation system - **NO schema changes needed**. Completely modular and backward compatible.

---

## Philosophical Foundation

### Dual-Process Cognition in AI

**ASTs = Explicit Thoughts (System 2)**
- Verbalizable, inspectable, logical
- Structured, compositional, rule-governed
- Slow, deliberate, conscious-analog

**Gut Annotations = Implicit Feelings (System 1)**
- Subsymbolic, pattern-based, statistical
- Holistic, similarity-driven, confidence-weighted
- Fast, automatic, subconscious-analog

**Integration:** Gut feelings guide which thoughts to pursue, thoughts verify gut feelings.

### Grounding Meaning in "What Matters"

Traditional meaning theories:
- **Referential:** Words refer to objects (struggles with abstract concepts)
- **Embodied:** Words grounded in sensorimotor experience (good for concrete nouns)
- **Distributional:** Words defined by context (statistical patterns)

**Our addition:** Words also grounded in **what matters to the system's goals**:
- **Confidence** (certeco): Does this matter to my knowledge state?
- **Salience** (graveco): Does this matter to the current query?
- **Coherence** (kohereco): Does this matter to semantic harmony?
- **Surprise** (surprizo): Does this matter as unexpected information?

---

## Pure Esperanto Design

### Why Esperanto Annotations?

1. **Circular Vocabulary:** Annotations themselves explained in Esperanto
2. **Self-Describing:** System can explain its own intuitions in Esperanto
3. **Philosophical Purity:** Entire knowledge graph is Esperanto
4. **Root Linking:** Every annotation connects to Root table (v2.1 schema)

### Annotation Type Names

| English | Esperanto | Root | Description |
|---------|-----------|------|-------------|
| Confidence | **certeco** | cert | Epistemic certainty about word knowledge |
| Salience | **graveco** | grav | Importance/relevance for current query |
| Coherence | **kohereco** | koher | Semantic fit with surrounding context |
| Surprise | **surprizo** | surpriz | Unexpectedness in context |

### Annotation Value Names (Qualitative + Numeric)

**Certeco (Confidence):**
- `tre_alta` (0.9-1.0) - Very high confidence
- `alta` (0.7-0.9) - High confidence
- `meza` (0.5-0.7) - Medium confidence
- `malalta` (0.3-0.5) - Low confidence
- `tre_malalta` (0.0-0.3) - Very low confidence

**Graveco (Salience):**
- `gravega` (0.9-1.0) - Critically important
- `grava` (0.7-0.9) - Important
- `meze_grava` (0.5-0.7) - Moderately important
- `malgrava` (0.3-0.5) - Unimportant
- `neglektebla` (0.0-0.3) - Negligible

**Kohereco (Coherence):**
- `tre_koheraj` (0.9-1.0) - Very coherent
- `koheraj` (0.7-0.9) - Coherent
- `meze_koheraj` (0.5-0.7) - Moderately coherent
- `malkoheraj` (0.3-0.5) - Incoherent
- `kaosaj` (0.0-0.3) - Chaotic

**Surprizo (Surprise):**
- `atendata` (0.0-0.2) - Expected
- `malsurpriza` (0.2-0.4) - Unsurprising
- `meze_surpriza` (0.4-0.6) - Moderately surprising
- `surpriza` (0.6-0.8) - Surprising
- `tre_surpriza` (0.8-1.0) - Very surprising

---

## Technical Architecture

### Uses Existing v2.1 Schema (No Changes!)

The v2.1 schema already has an extensible annotation system:

```sql
-- Already exists in v2.1
CREATE NODE TABLE AnnotationType (
    id INT64 PRIMARY KEY,
    type_name STRING,          -- "intuicia_certeco"
    description STRING,
    root STRING,               -- "cert" (links to Root table!)
    value_type STRING,
    created_by STRING
);

CREATE NODE TABLE AnnotationValue (
    id INT64 PRIMARY KEY,
    value_name STRING,         -- "tre_alta"
    root STRING,               -- "alt" (links to Root table!)
    value_type STRING,
    numeric_value DOUBLE,      -- 0.95
    metadata STRING
);

CREATE NODE TABLE Annotation (
    id INT64 PRIMARY KEY,
    annotation_set_id INT64,
    annotation_type_id INT64,
    annotation_value_id INT64,
    confidence DOUBLE,
    metadata STRING
);

-- Can attach to any level
CREATE REL TABLE WORD_HAS_ANNOTATION (FROM Vorto TO Annotation);
CREATE REL TABLE FRAZO_HAS_ANNOTATION (FROM Frazo TO Annotation);
-- etc.
```

**We just register new types and values - no schema changes!**

### Module Structure

```
klareco/annotations/
├── __init__.py
├── tipoj.py                 # IntuitaspectoTipoj (type registry)
├── valoroj.py               # IntuitaspectoValoroj (value registry)
├── agordoj.py               # IntuitaspectoAgordoj (config dataclass)
├── certeco.py               # Confidence computation
├── graveco.py               # Salience computation
├── kohereco.py              # Coherence computation
├── surprizo.py              # Surprise computation
├── annotilo.py              # IntuitaspectoAnnotilo (main class)
└── kuzu_integration.py      # Database operations

scripts/
├── initialize_gut_annotation_types.py
└── index_corpus_with_gut.py  # Optional gut annotations during indexing

tests/test_intuiciaj_annotacioj/
├── test_tipoj.py
├── test_valoroj.py
├── test_agordoj.py
├── test_certeco.py
├── test_graveco.py
├── test_kohereco.py
├── test_surprizo.py
├── test_annotilo.py
└── test_integration.py
```

### Configuration System (Modular Feature Toggles)

```python
@dataclass
class IntuitaspectoAgordoj:
    """Configuration - can enable/disable each feature"""

    # Feature toggles
    ebligi_certecon: bool = True     # Enable confidence
    ebligi_gravecon: bool = True     # Enable salience
    ebligi_koherecon: bool = True    # Enable coherence
    ebligi_surprizon: bool = True    # Enable surprise

    # Model dependencies
    vojo_al_enkorporigilo: Optional[str] = None  # Embedder path
    vojo_al_vortaro: Optional[str] = None        # Vocabulary path

    @classmethod
    def minimuma(cls):
        """Minimal: only confidence (no embeddings needed)"""
        return cls(
            ebligi_certecon=True,
            ebligi_gravecon=False,
            ebligi_koherecon=False,
            ebligi_surprizon=False
        )

    @classmethod
    def malaktivigita(cls):
        """Disabled: no gut annotations"""
        return cls(
            ebligi_certecon=False,
            ebligi_gravecon=False,
            ebligi_koherecon=False,
            ebligi_surprizon=False
        )
```

**Key:** Can start with minimal mode (only confidence), no embeddings required!

---

## Computation Methods

### 1. Certeco (Confidence)

**Based on:**
- Vocabulary tier (v2.1 classification: tier0-tier6)
- Parse status (success vs. unknown_root)
- Corpus frequency

**Algorithm:**
```python
def kalkuli_certecon(vorto: Dict) -> float:
    """Compute epistemic confidence"""

    # Base confidence from v2.1 tier
    nivelo = vorto['radiko_nivelo']

    if nivelo == 'tier0_*':           # Function words
        baza = 1.0
    elif nivelo == 'tier1a_unua_libro':  # Fundamento
        baza = 0.95
    elif nivelo == 'tier2_revo':         # ReVo dictionary
        baza = 0.85
    elif nivelo == 'tier3_corpus':       # Corpus-validated
        baza = 0.70
    else:                                # Unknown
        baza = 0.30

    # Adjust by parse status
    if vorto['parse_status'] != 'success':
        baza *= 0.5

    # Adjust by frequency
    ofteco = vorto['radiko_ofteco']
    if ofteco > 0:
        baza += min(0.1, log(ofteco) / 10)

    return baza
```

**Dependencies:** v2.1 tier classification, corpus stats
**No embeddings needed!**

### 2. Graveco (Salience)

**Based on:**
- Semantic similarity to query words (gut feeling - 60%)
- Structural role importance (explicit rule - 40%)

**Algorithm:**
```python
def kalkuli_gravecon(vorto: Dict, demando_ast: Dict) -> float:
    """Compute salience for query"""

    # Gut: Semantic similarity
    vorto_emb = embedder.compose(vorto)
    demando_embs = [embedder.compose(qw) for qw in demando_ast.words]

    semantic_sim = max([
        cosine_sim(vorto_emb, qe)
        for qe in demando_embs
    ])

    # Thought: Structural role
    role_weights = {
        'subjekto': 0.9,
        'verbo': 0.85,
        'objekto': 0.8,
        'modifilo': 0.4
    }
    structural_weight = role_weights[vorto['role']]

    # Combine (60% gut, 40% thought)
    return 0.6 * semantic_sim + 0.4 * structural_weight
```

**Dependencies:** Compositional embeddings, query AST
**Requires embeddings!**

### 3. Kohereco (Coherence)

**Based on:**
- Average similarity to context words (all other words in sentence)

**Algorithm:**
```python
def kalkuli_koherecon(vorto: Dict, ast: Dict) -> float:
    """Compute coherence with context"""

    vorto_emb = embedder.compose(vorto)

    # Get all other words in sentence
    context_words = [w for w in ast.words if w.id != vorto.id]
    context_embs = [embedder.compose(cw) for cw in context_words]

    # Average similarity to context
    similarities = [
        cosine_sim(vorto_emb, ce)
        for ce in context_embs
    ]

    return mean(similarities)
```

**Dependencies:** Compositional embeddings
**Requires embeddings!**

### 4. Surprizo (Surprise)

**Based on:**
- Inverse frequency (rare words = surprising)
- Future: language model predictions

**Algorithm:**
```python
def kalkuli_surprizon(vorto: Dict) -> float:
    """Compute surprise/unexpectedness"""

    ofteco = vorto['radiko_ofteco']

    if ofteco == 0:
        return 0.9  # Very surprising (unknown)

    # Surprise = -log(frequency)
    surprizo = -log(ofteco / total_words) / 10  # Normalize

    return min(1.0, surprizo)
```

**Dependencies:** Corpus stats
**No embeddings needed!**

---

## Usage Examples

### Initialize Annotation System

```python
# scripts/initialize_gut_annotation_types.py
from klareco.annotations.tipoj import IntuitaspectoTipoj
from klareco.annotations.valoroj import IntuitaspectoValoroj
import kuzu

# Connect to v2.1 database
db = kuzu.Database("data/indexes/v2.1_kuzu_index_full")
conn = kuzu.Connection(db)

# Register 4 annotation types
IntuitaspectoKonfigurilo.initialize_annotation_types(conn)

# Register 20 annotation values
IntuitaspectoKonfigurilo.initialize_annotation_values(conn)

print("✓ Registered Esperanto gut annotations")
```

### Annotate Corpus (Minimal Mode)

```bash
# Only confidence - no embeddings needed!
python scripts/index_corpus_with_gut.py \
  --corpus data/corpus/unified_corpus.jsonl \
  --enable-gut \
  --gut-mode minimal \
  --vocabulary data/vocabularies/root_vocab.json
```

### Annotate Corpus (Full Mode)

```bash
# All features - requires embeddings
python scripts/index_corpus_with_gut.py \
  --corpus data/corpus/unified_corpus.jsonl \
  --enable-gut \
  --gut-mode full \
  --embedder models/root_embeddings/best_model.pt \
  --vocabulary data/vocabularies/root_vocab.json \
  --corpus-stats data/corpus/stats.json
```

### Query Annotations (Esperanto!)

```cypher
-- Find high-confidence words
MATCH (v:Vorto)-[:WORD_HAS_ANNOTATION]->(a:Annotation)
      -[:ANNOTATION_HAS_TYPE]->(t:AnnotationType {type_name: 'intuicia_certeco'})
      -[:ANNOTATION_HAS_VALUE]->(val:AnnotationValue {value_name: 'alta'})
RETURN v.plena_vorto, val.numeric_value

-- Find salient words for query
MATCH (v:Vorto)-[:WORD_HAS_ANNOTATION]->(a:Annotation)
      -[:ANNOTATION_HAS_TYPE]->(t:AnnotationType {type_name: 'intuicia_graveco'})
      -[:ANNOTATION_HAS_VALUE]->(val:AnnotationValue)
WHERE val.numeric_value > 0.7
RETURN v.plena_vorto, val.value_name
ORDER BY val.numeric_value DESC

-- Find incoherent words (potential errors)
MATCH (v:Vorto)-[:WORD_HAS_ANNOTATION]->(a:Annotation)
      -[:ANNOTATION_HAS_TYPE]->(t:AnnotationType {type_name: 'intuicia_kohereco'})
      -[:ANNOTATION_HAS_VALUE]->(val:AnnotationValue)
WHERE val.value_name IN ['malkoheraj', 'kaosaj']
RETURN v.plena_vorto, val.numeric_value

-- Circular vocabulary: what roots are annotations linked to?
MATCH (t:AnnotationType)-[:ANNOTATION_TYPE_IS_ROOT]->(r:Root)
RETURN t.type_name, r.root
```

### Programmatic Usage

```python
from klareco.annotations.annotilo import IntuitaspectoAnnotilo
from klareco.annotations.agordoj import IntuitaspectoAgordoj

# Minimal config (no embeddings)
agordoj = IntuitaspectoAgordoj.minimuma()
annotilo = IntuitaspectoAnnotilo(agordoj)

# Annotate AST
ast = parser.parse("La hundo kuris rapide.")
annotacioj = annotilo.annotacii_vorton(ast.words[1])  # "hundo"

# Result: [{'enteca_tipo': 'Vorto', 'annotacia_tipo': 'intuicia_certeco', ...}]
```

---

## Integration Points

### 1. Corpus Indexing (Optional)

**File to modify:** `scripts/index_corpus_v2.py` or create new wrapper

**Add flag:** `--enable-gut` with modes: `minimal`, `full`, `disabled`

**Backward compatible:** Default is `disabled`

```python
# In index_corpus_v2.py (add optional section)
if args.enable_gut:
    from klareco.annotations.annotilo import IntuitaspectoAnnotilo

    annotilo = IntuitaspectoAnnotilo(
        IntuitaspectoAgordoj.from_mode(args.gut_mode)
    )

    # For each AST
    annotations = annotilo.annotacii_frazon(ast)
    store_annotations_to_kuzu(annotations)
```

### 2. Retrieval (Optional)

**File to create:** `klareco/rag/retriever_with_gut.py`

**Extends:** `ASTRetriever` with salience weighting

```python
class GutAwareRetriever(ASTRetriever):
    """Retriever with optional salience weighting"""

    def retrieve(self, query, top_k=10):
        # Standard retrieval
        candidates = super().retrieve(query, top_k*2)

        # Rerank by salience overlap
        if self.use_salience:
            for passage in candidates:
                salience_score = compute_salience_overlap(query, passage)
                passage.score = 0.7 * passage.score + 0.3 * salience_score

        return candidates[:top_k]
```

### 3. QA System (Future)

**Use confidence for uncertainty:**
```python
if answer.min_confidence < 0.7:
    return "Mi ne estas certa pri mia respondo" (I'm not certain)
```

**Use coherence for error detection:**
```python
if answer.coherence < 0.5:
    flag_for_review("Answer seems incoherent")
```

---

## Implementation Phases

### Phase 1: Minimal Mode (Confidence Only)
**Time:** 8-12 hours
**Dependencies:** v2.1 tier classification, corpus stats
**No embeddings required!**

**Deliverables:**
- `certeco.py` - Confidence computation
- `surprizo.py` - Surprise computation (frequency-based)
- Database initialization
- Backward-compatible corpus indexing

**Risk:** LOW (uses existing v2.1 data, no embeddings)

### Phase 2: Full Mode (All Features)
**Time:** 12-16 hours
**Dependencies:** Compositional embeddings (from other session)

**Deliverables:**
- `graveco.py` - Salience computation
- `kohereco.py` - Coherence computation
- Full annotation pipeline
- Retrieval integration

**Risk:** MEDIUM (depends on embeddings from parallel session)

### Phase 3: Advanced Features
**Time:** 8-12 hours

**Deliverables:**
- Salience-weighted retrieval
- Confidence-aware QA
- Demo notebooks
- Performance benchmarks

---

## Coordination with Parallel Development

### Current Parallel Work (from git status):
- `klareco/embeddings/` - Hybrid embeddings, root embeddings
- Training scripts - Fundamento training, semantic hierarchies
- Documentation - Model naming

### Conflict Avoidance Strategy:

**Files we CREATE (no conflicts):**
- `klareco/annotations/` (new directory)
- `tests/test_intuiciaj_annotacioj/` (new directory)
- `docs/INTUICIAJ_ANNOTACIOJ_*.md` (new files)

**Files we IMPORT (read-only, coordinate):**
- `klareco/embeddings/compositional.py` (for Phase 2)
- `klareco/embeddings/__init__.py` (import only)

**Files we DON'T TOUCH:**
- Training scripts
- Existing embeddings code
- Schema files (already has what we need)

### Coordination Protocol:

1. **Phase 1** can start immediately (no embeddings dependency)
2. **Phase 2** waits for embeddings session to stabilize
3. Use interface pattern for embeddings:
   ```python
   class EmbedderInterface(ABC):
       @abstractmethod
       def compose_word(self, word: Dict) -> np.ndarray:
           pass
   ```
4. Both sessions implement interface independently
5. Merge when both stable

---

## Testing Strategy

### Unit Tests (per component)
- Mock all dependencies
- Test each computation independently
- 85%+ coverage per module

### Integration Tests
- Test full annotation pipeline
- Test database round-trip
- Test with real v2.1 data

### Backward Compatibility Tests
- Ensure disabled mode works (default)
- Ensure existing code unaffected
- Ensure schema unchanged

### Performance Tests
- Annotation speed (target: <1ms per word)
- Database insertion (target: 1000 annotations/sec)
- Retrieval impact (target: <10% slowdown)

---

## Rollback Plan

If implementation becomes problematic:

**Option 1: Deploy minimal only**
- Only implement Phase 1 (confidence + surprise)
- Skip embeddings-dependent features
- Still provides value (vocabulary coverage awareness)

**Option 2: External annotations**
- Compute annotations offline
- Store in separate JSON files
- Don't integrate into database
- Use for analysis only

**Option 3: Defer entirely**
- Document design
- Implement after embeddings stabilize
- No rush - this is an enhancement, not critical

---

## Success Criteria

### Minimum Viable Product (MVP):
- ✅ Confidence annotations working
- ✅ Database integration (4 types, 20 values registered)
- ✅ Backward compatible (disabled by default)
- ✅ Documentation complete
- ✅ 80%+ test coverage

### Full Product:
- ✅ All 4 annotation types working
- ✅ Salience-weighted retrieval
- ✅ Pure Esperanto (all types/values linked to roots)
- ✅ Demo notebooks
- ✅ 90%+ test coverage

---

## Benefits

### For Klareco's Mission:
1. **Explainability:** Gut feelings are named and queryable in Esperanto
2. **Purity:** Entire system (thoughts + feelings) in Esperanto
3. **Modularity:** Can enable/disable features independently
4. **Efficiency:** Guides attention to important parts (salience)
5. **Honesty:** Reports uncertainty (confidence)
6. **Error Detection:** Flags anomalies (coherence)

### For Research:
1. **Dual-Process AI:** First system with explicit thoughts + gut feelings
2. **Meaning Grounding:** Demonstrates grounding in "what matters"
3. **Circular Vocabulary:** Self-describing annotation system
4. **Minimal Learning:** Confidence + surprise need no embeddings!

---

## Questions for Review

1. **Timing:** Implement now or defer until embeddings stable?
2. **Scope:** Start with minimal (Phase 1) or full (Phase 1+2)?
3. **Integration:** Modify existing scripts or create new wrappers?
4. **Naming:** Keep Esperanto names or add English aliases?
5. **Storage:** Always compute on-the-fly or pre-compute and store?

---

## References

### Related Documents:
- `klareco/schema/kuzu_ast_schema.py` - v2.1 annotation system
- `docs/V2.1_DATABASE_CLASSIFICATION_COMPLETE.md` - Tier classification
- `CLAUDE.md` - Project philosophy

### Theoretical Background:
- Damasio's Somatic Marker Hypothesis - Gut feelings guide reasoning
- Dual-Process Theory (Kahneman) - System 1 vs System 2
- Embodied Cognition (Lakoff & Johnson) - Meaning grounded in experience
- Symbol Grounding Problem (Harnad) - Connecting symbols to non-symbolic representations

---

## Appendix: Example Annotation Records

### Word with All Annotations

```json
{
  "vorto": {
    "id": 12345,
    "plena_vorto": "hundo",
    "radiko": "hund",
    "radiko_nivelo": "tier1a_unua_libro",
    "radiko_ofteco": 15420
  },
  "annotacioj": [
    {
      "tipo": "intuicia_certeco",
      "valoro": "tre_alta",
      "nombro": 0.95,
      "radiko_de_tipo": "cert",
      "radiko_de_valoro": "alt"
    },
    {
      "tipo": "intuicia_graveco",
      "valoro": "grava",
      "nombro": 0.82,
      "radiko_de_tipo": "grav",
      "radiko_de_valoro": "grav"
    },
    {
      "tipo": "intuicia_kohereco",
      "valoro": "koheraj",
      "nombro": 0.78,
      "radiko_de_tipo": "koher",
      "radiko_de_valoro": "koher"
    },
    {
      "tipo": "intuicia_surprizo",
      "valoro": "malsurpriza",
      "nombro": 0.25,
      "radiko_de_tipo": "surpriz",
      "radiko_de_valoro": "surpriz"
    }
  ]
}
```

### Interpretation (in Esperanto)

```
La vorto "hundo":
- Mi estas TRE CERTA pri ĝia signifo (fundamento-vorto, ofta)
- Ĝi estas GRAVA por la demando (alta simileco al demando-vortoj)
- Ĝi KOHERAJ kun la kunteksto (similas al aliaj vortoj en frazo)
- Ĝi estas MALSURPRIZA (ofta vorto, atendata)
```

---

**END OF DESIGN DOCUMENT**

**Next Steps:**
1. Review by other Claude session
2. Decide: implement now or defer?
3. If implement: start with Phase 1 (minimal mode)
4. If defer: bookmark for future enhancement
