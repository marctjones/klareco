# Semantic Knowledge Graph Integration

## Overview

Klareco now includes a comprehensive semantic knowledge graph built from two authoritative sources: ReVo (Reta Vortaro) and ConceptNet. This enables semantic reasoning, synonym expansion, and multi-hop inference.

## Database Contents

**Nodes:**
- 14,591 Esperanto roots
- 23,873 tier0 authoritative sentences
- 12,332 external concepts (from ConceptNet)

**Semantic Relations (18,509 total):**
- **ReVo**: 1,982 relations (weight=2.0, authoritative Esperanto dictionary)
  - 653 synonyms
  - 78 antonyms
  - 817 hypernyms (is-a relationships)
  - 299 hyponyms (subtype relationships)
  - 135 part-of relationships

- **ConceptNet**: 16,527 relations (weight=1.0, multilingual knowledge)
  - 12,477 external synonyms (links to other languages)
  - 826 internal Esperanto synonyms
  - 472 is-a relations
  - 366 antonyms
  - 2,029 has-context (topical domain) relations
  - 357 part-of relationships

## Pipeline Scripts

### ReVo Processing Pipeline

**Complete wrapper:**
```bash
./scripts/process_revo.sh          # Run all 3 steps
./scripts/process_revo.sh --fresh  # Rebuild from scratch
```

**Individual steps:**

1. **Extract Relations** (`scripts/extract_revo_semantic_relations.py`)
   - Connects to ReVo SQLite database (data/raw/eo/dictionaries/revo/revo.db)
   - Extracts 5 semantic relation types: synonym, antonym, hypernym, hyponym, part-of
   - Normalizes x-notation (cx→ĉ, gx→ĝ, etc.)
   - Validates all roots with Klareco parser
   - Outputs: `data/raw/eo/dictionaries/revo/revo_semantic_relations.json`

2. **Validate Relations** (`scripts/validate_revo_relations.py`)
   - Checks symmetric consistency (if A syn B, then B syn A)
   - Validates hypernym/hyponym reciprocity
   - Detects cycles in hierarchical relations
   - Checks corpus coverage (62.4% of ReVo roots found in tier0)
   - Outputs: `data/raw/eo/dictionaries/revo/revo_validation_report.json`

3. **Load to Kuzu** (`scripts/load_revo_to_kuzu.py`)
   - Extends Kuzu schema with 5 relation table types
   - Filters relations to only include roots present in corpus
   - Uses CSV bulk loading for performance
   - Sets weight=2.0 (higher than ConceptNet)
   - Outputs: Edges in `data/indexes/kuzu_index/kuzu.db`

### ConceptNet Loading

**Script:** `scripts/load_conceptnet_to_kuzu.py`

```bash
python scripts/load_conceptnet_to_kuzu.py --fresh
```

- Processes 475MB gzipped ConceptNet dump (34M assertions)
- Filters for Esperanto-relevant relations only (66K lines)
- Creates Concept nodes for external language links
- Memory-efficient streaming (line-by-line processing)
- Loaded: 16,769 semantic relations

## Usage Examples

### Python API

```python
import kuzu

db = kuzu.Database('data/indexes/kuzu_index/kuzu.db')
conn = kuzu.Connection(db)

# Find synonyms
result = conn.execute("""
    MATCH (r:Root {root: 'bona'})-[:REVO_SYNONYM]->(s:Root)
    RETURN s.root
""")

# Find what something is (hypernyms)
result = conn.execute("""
    MATCH (r:Root {root: 'hundo'})-[:REVO_HYPERNYM]->(h:Root)
    RETURN h.root
""")

# Multi-hop reasoning
result = conn.execute("""
    MATCH path = (r:Root)-[:REVO_SYNONYM]->(s:Root)-[:REVO_HYPERNYM]->(h:Root)
    RETURN r.root, s.root, h.root
    LIMIT 10
""")
```

### Query Patterns

See IdlerGear reference: "Semantic Query Patterns" for comprehensive examples.

## Integration with RAG System

The semantic relations can enhance retrieval in several ways:

1. **Synonym Expansion**: When user queries "bona", also search for synonyms
2. **Hypernym Generalization**: If no results for "hundo", search for "besto"
3. **Semantic Reranking**: Boost results with semantic similarity to query
4. **Multi-hop Inference**: Answer "What are examples of animals?" by traversing hyponym edges

## Data Sources

**ReVo (Reta Vortaro)**
- Database: `data/raw/eo/dictionaries/revo/revo.db`
- Version: 2017-12-15
- Nodes: 40,373 dictionary entries
- Authoritative Esperanto dictionary with hand-curated semantic relations

**ConceptNet**
- Dump: ConceptNet 5.7 (34M assertions)
- Filtered for Esperanto: 66,492 lines (0.2% of total)
- Multilingual knowledge graph with 12,332 external concept links

## Technical Implementation

**Memory Efficiency:**
- All scripts use streaming line-by-line processing
- Memory usage: <200MB even for 475MB files
- CSV bulk loading for fast Kuzu ingestion

**Restartability:**
- Checkpoint support with `--resume` and `--fresh` flags
- Progress tracking in JSON checkpoint files
- Atomic checkpoint saves (write to .tmp then rename)

**Quality Assurance:**
- Parser validation ensures all roots are compatible with Klareco
- Validation scripts check semantic consistency
- Weight system (ReVo=2.0, ConceptNet=1.0) prioritizes authoritative sources

## Next Steps

Potential enhancements:

1. **Semantic Retrieval Integration**
   - Add synonym expansion to retriever
   - Use hypernym generalization for zero-result queries
   - Implement semantic similarity reranking

2. **Model Training**
   - Retrain M1 selectional model with tier0 weights
   - Use semantic relations as supervision signal
   - Add semantic loss terms to embedding training

3. **Multi-hop Reasoning**
   - Implement graph traversal for question answering
   - Build reasoning chains: "X is-a Y, Y part-of Z"
   - Add explainability via semantic paths

4. **Additional Sources**
   - Wikidata integration for factual knowledge
   - Wikipedia category hierarchy for topic modeling
   - Lernu.net vocabulary for pedagogical relations

## Files Modified/Created

**Created:**
- `scripts/extract_revo_semantic_relations.py`
- `scripts/validate_revo_relations.py`
- `scripts/load_revo_to_kuzu.py`
- `scripts/process_revo.sh`
- `docs/SEMANTIC_KNOWLEDGE_GRAPH.md`

**Modified:**
- `scripts/extract_tier0_literary.py` (added checkpoints)
- `scripts/extract_grammar_works.py` (added checkpoints)
- `scripts/merge_tier0_into_corpus.py` (added checkpoints)

**Data Generated:**
- `data/raw/eo/dictionaries/revo/revo_semantic_relations.json` (3,453 relations)
- `data/raw/eo/dictionaries/revo/revo_validation_report.json`
- `data/indexes/kuzu_index/kuzu.db` (semantic edges loaded)
- `data/enhanced_corpus/corpus_with_tier0.jsonl` (23,873 sentences)
