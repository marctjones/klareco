# Fact-Based Architecture: Should We Query Kuzu Directly?

## The Fundamental Question

**Current design**: Query → RAG retrieves sentences → Parse → Extract facts → Rank → Synthesize

**Alternative**: Query → Kuzu graph query → Facts directly → Synthesize

**User's insight**: We have a knowledge graph - why not use it directly?

## Current vs Alternative Architecture

### Architecture A: RAG-First (Current)

```
┌──────────────────────────────────────────────────┐
│ Query: "Kio estas kato?"                         │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ RAG Retrieval (Root embeddings + M1 + Reranker) │
│   → Top 20 sentences about cats                  │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ Parse sentences to ASTs (16 rules)              │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ Extract facts from ASTs                          │
│   - (kato, est, besto)                          │
│   - (kato, hav, piedo)                          │
│   - etc.                                        │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ Build temporary in-memory fact graph            │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ Rank facts, cluster, synthesize                 │
└──────────────────────────────────────────────────┘
```

**Problems**:
- ❌ Redundant: Extracting facts every time from same sentences
- ❌ Slow: Parse → extract → build graph each query
- ❌ Wasteful: Rebuilding fact graph from scratch
- ❌ Limited: Only facts from retrieved sentences

### Architecture B: Kuzu-First (Alternative)

```
┌──────────────────────────────────────────────────┐
│ Query: "Kio estas kato?"                         │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ Query Kuzu Graph Directly                       │
│   MATCH (f:Fact)-[:ABOUT]->(e:Entity {name: "kato"})│
│   WHERE f.type IN ['definition', 'property']    │
│   RETURN f                                       │
│                                                  │
│   → Returns facts directly from graph!           │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ Rank facts using graph queries                  │
│   - PageRank on fact nodes                      │
│   - Entity salience (already in graph)          │
│   - Co-occurrence counts (pre-computed)         │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ Cluster facts using graph queries               │
│   - Find connected fact subgraphs               │
│   - Group by shared entities                    │
└──────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│ Synthesize from facts (construct ASTs)          │
└──────────────────────────────────────────────────┘
```

**Advantages**:
- ✅ Fast: No parsing, facts already extracted
- ✅ Efficient: Facts pre-computed, stored in graph
- ✅ Comprehensive: Access to ALL facts in corpus, not just retrieved sentences
- ✅ Graph operations: Leverage Kuzu's graph algorithms (PageRank, shortest path, etc.)
- ✅ Deterministic: Pure graph queries, no models needed

## Enhanced Kuzu Schema: Adding Facts

### Current Schema (v2.1)
```
Nodes:
  - Radiko (roots)
  - Vorto (words)
  - Frazoteksto (sentences with ASTs)

Edges:
  - (Vorto)-[:HAS_ROOT]->(Radiko)
  - (Frazoteksto)-[:CONTAINS_WORD]->(Vorto)
```

### Enhanced Schema (v3.0 - Fact-Based)
```
Nodes:
  - Radiko (roots)
  - Vorto (words)
  - Frazoteksto (sentences with ASTs)
  - Entity (named entities: kato, hundo, Zamenhof, etc.)
  - Fact (atomic propositions)
      properties: {
        type: 'predication' | 'property' | 'possession' | 'action',
        subject: string,
        predicate: string,
        object: string,
        importance: float,  // pre-computed
        source_sentence_id: string
      }

Edges:
  - (Vorto)-[:HAS_ROOT]->(Radiko)
  - (Frazoteksto)-[:CONTAINS_WORD]->(Vorto)
  - (Frazoteksto)-[:CONTAINS_FACT]->(Fact)
  - (Fact)-[:ABOUT]->(Entity)
  - (Fact)-[:SUBJECT]->(Entity)
  - (Fact)-[:OBJECT]->(Entity)
  - (Fact)-[:RELATED_TO]->(Fact)  // co-occurrence, similarity
  - (Entity)-[:CO_OCCURS_WITH]->(Entity)  // pre-computed
```

### Pre-Computation: Populate Facts Once

```python
def populate_facts_in_kuzu(corpus, kuzu_db):
    """
    ONE-TIME: Parse entire corpus, extract facts, store in Kuzu.
    Then queries are fast!
    """

    for sentence in corpus:
        # Parse to AST
        ast = parse(sentence['text'])

        # Extract facts
        facts = extract_facts_from_ast(ast)

        for fact in facts:
            # Compute importance (deterministic)
            importance = compute_fact_importance(fact, kuzu_db)

            # Store fact in Kuzu
            kuzu_db.execute("""
                CREATE (f:Fact {
                    type: $type,
                    subject: $subject,
                    predicate: $predicate,
                    object: $object,
                    importance: $importance,
                    source_sentence_id: $sentence_id
                })
            """, fact)

            # Create edges
            kuzu_db.execute("""
                MATCH (s:Frazoteksto {id: $sentence_id})
                MATCH (f:Fact {id: $fact_id})
                CREATE (s)-[:CONTAINS_FACT]->(f)
            """, {'sentence_id': sentence['id'], 'fact_id': fact['id']})

            # Link to entities
            for entity in [fact['subject'], fact['object']]:
                kuzu_db.execute("""
                    MERGE (e:Entity {name: $entity})
                    MATCH (f:Fact {id: $fact_id})
                    CREATE (f)-[:ABOUT]->(e)
                """, {'entity': entity, 'fact_id': fact['id']})

    # Pre-compute co-occurrence
    compute_entity_cooccurrence(kuzu_db)
    compute_fact_relationships(kuzu_db)
```

**This is done ONCE during corpus indexing, not at query time!**

## Query-Time: Fast Graph Queries

### Example Query 1: "Kio estas kato?" (What is a cat?)

```cypher
// Find definition facts about "kato"
MATCH (f:Fact)-[:ABOUT]->(e:Entity {name: "kato"})
WHERE f.type = 'predication' AND f.predicate = 'est'
RETURN f.subject, f.object, f.importance
ORDER BY f.importance DESC
LIMIT 10

// Results:
// (kato, besto, 0.95)
// (kato, karnovoro, 0.85)
// (kato, hejma_besto, 0.80)
```

### Example Query 2: "Kion mangxas katoj?" (What do cats eat?)

```cypher
// Find action facts where kato is subject and verb is mangx/cxas
MATCH (f:Fact)-[:SUBJECT]->(e1:Entity {name: "kato"})
WHERE f.predicate IN ['manĝ', 'ĉas']
MATCH (f)-[:OBJECT]->(e2:Entity)
RETURN e2.name, count(f) as frequency
ORDER BY frequency DESC

// Results:
// (muso, 45)  // cats hunt mice (mentioned 45 times in corpus)
// (birdo, 32)
// (fiŝo, 18)
```

### Example Query 3: Get related facts for clustering

```cypher
// Find facts related to main fact (for clustering)
MATCH (f1:Fact {id: $fact_id})
MATCH (f1)-[:ABOUT]->(e:Entity)
MATCH (f2:Fact)-[:ABOUT]->(e)
WHERE f2 <> f1
RETURN f2, count(e) as shared_entities
ORDER BY shared_entities DESC
LIMIT 20

// Returns facts that share entities with main fact
// → Automatically clusters related facts!
```

## Do We Need Models?

### Question 1: Semantic Fact Clustering

**With Kuzu**: ✅ **NO MODEL NEEDED**

```cypher
// Cluster facts by shared entities (deterministic)
MATCH (f1:Fact)-[:ABOUT]->(e:Entity)<-[:ABOUT]-(f2:Fact)
WHERE f1 <> f2
RETURN f1, f2, e.name as shared_entity

// Or cluster by co-occurrence in same sentences
MATCH (s:Frazoteksto)-[:CONTAINS_FACT]->(f1:Fact)
MATCH (s)-[:CONTAINS_FACT]->(f2:Fact)
WHERE f1 <> f2
RETURN f1, f2, count(s) as co_occurrence
```

**Deterministic clustering using graph structure!**

### Question 2: Ranking Fact Importance

**With Kuzu**: ✅ **NO MODEL NEEDED**

```python
def compute_fact_importance_in_graph(fact_id, kuzu_db):
    """
    Compute importance using graph algorithms (deterministic).
    Pre-computed and stored, not computed at query time.
    """

    # Factor 1: PageRank in fact graph
    pagerank = kuzu_db.execute("""
        CALL pagerank(Fact, RELATED_TO)
        MATCH (f:Fact {id: $fact_id})
        RETURN f.pagerank
    """, {'fact_id': fact_id})

    # Factor 2: Entity salience (already in graph)
    entity_salience = kuzu_db.execute("""
        MATCH (f:Fact {id: $fact_id})-[:ABOUT]->(e:Entity)
        RETURN avg(e.salience) as avg_salience
    """, {'fact_id': fact_id})

    # Factor 3: Fact frequency (how many sentences mention this?)
    frequency = kuzu_db.execute("""
        MATCH (s:Frazoteksto)-[:CONTAINS_FACT]->(f:Fact {id: $fact_id})
        RETURN count(s) as freq
    """, {'fact_id': fact_id})

    # Combine (deterministic formula)
    importance = 0.4 * pagerank + 0.3 * entity_salience + 0.3 * (frequency / 100.0)

    return importance
```

**All deterministic graph queries!**

## Hybrid Architecture: Best of Both?

Maybe we don't have to choose? Use both:

### Scenario A: Fact-Based Questions (Use Kuzu)
```
Query: "Kio estas kato?"
→ Query type: Definition
→ Use Kuzu directly (facts about "kato" with predicate "est")
→ Fast, precise, comprehensive
```

### Scenario B: Complex/Narrative Questions (Use RAG + Facts)
```
Query: "Rakontu al mi pri la historio de Esperanto."
→ Query type: Narrative
→ Use RAG to find relevant passages/paragraphs
→ Extract facts from retrieved sentences
→ Synthesize narrative from facts
```

### Scenario C: Fact Enrichment (Use Both)
```
Query: "Kiu fondis Esperanton?"
→ RAG finds: "Zamenhof fondis Esperanton en 1887."
→ Query Kuzu for related facts about "Zamenhof":
    - (Zamenhof, est, kuracisto)
    - (Zamenhof, naskiĝ_en, Bjalistoko)
    - (Zamenhof, havis, celo_mondpaco)
→ Synthesize richer answer combining RAG + Kuzu facts
```

## Practical Implementation Strategy

### Phase 1: Enhance Kuzu Schema (Week 1-2)

1. Add Fact and Entity nodes to schema
2. Write fact extraction script (from ASTs)
3. Populate facts for entire corpus (one-time operation)
4. Pre-compute:
   - Fact importance (PageRank + entity salience + frequency)
   - Entity co-occurrence
   - Fact relationships

### Phase 2: Implement Fact-Based Queries (Week 3)

1. Write Cypher query templates for common question types:
   - Definition: "Kio estas X?"
   - Property: "Kiel X estas?" (How is X?)
   - Action: "Kion faras X?" (What does X do?)
   - Cause: "Kial X?" (Why X?)

2. Implement fact retrieval from Kuzu

3. Implement fact clustering (graph queries)

4. Implement fact ranking (use pre-computed importance)

### Phase 3: Synthesis (Week 4)

1. Construct ASTs from facts (deterministic)
2. Deparse to sentences
3. Test on 50-100 queries

### Phase 4: Hybrid Approach (Week 5, if needed)

1. Identify when Kuzu facts insufficient
2. Fall back to RAG retrieval
3. Combine RAG + Kuzu facts

## Answering Your Questions

### "Do we need models to do semantic fact clusters?"

**NO!** With Kuzu:
```cypher
// Cluster facts by shared entities (deterministic)
MATCH (f1:Fact)-[:ABOUT]->(e:Entity)<-[:ABOUT]-(f2:Fact)
RETURN f1, f2, e.name
```

### "Do we need a model to rank the importance of facts?"

**NO!** With Kuzu:
```cypher
// Use pre-computed importance from PageRank + entity salience + frequency
MATCH (f:Fact)-[:ABOUT]->(e:Entity {name: $query_entity})
RETURN f
ORDER BY f.importance DESC
```

### "Could we use the kuzu graph to supplement the ranked sentences?"

**YES!** Absolutely:
```cypher
// After RAG retrieves sentences, query Kuzu for related facts
MATCH (s:Frazoteksto {id: $retrieved_sentence_id})-[:CONTAINS_FACT]->(f1:Fact)
MATCH (f1)-[:RELATED_TO]->(f2:Fact)
RETURN f2
```

### "Would it be better to extract the information we need from the kuzu db?"

**YES, for many query types!** Especially:
- Definition queries ("Kio estas X?")
- Property queries ("Kiel X estas?")
- Factual queries ("Kiam X?", "Kie X?")

**Advantages**:
- ✅ Faster (no parsing at query time)
- ✅ More comprehensive (all facts in corpus)
- ✅ More structured (facts already extracted and related)
- ✅ Fully deterministic (graph queries + algorithms)

## Recommendation

**Implement Kuzu-first architecture with RAG fallback**:

1. **Primary**: Query Kuzu for facts directly
   - Fast, deterministic, comprehensive
   - No models needed for clustering or ranking
   - Leverage graph algorithms (PageRank, etc.)

2. **Fallback**: Use RAG for complex/narrative queries
   - When Kuzu facts insufficient
   - When need full sentence context

3. **Hybrid**: Enrich RAG results with Kuzu facts
   - RAG finds relevant sentences
   - Query Kuzu for related facts
   - Synthesize richer answers

**This maximizes deterministic processing and leverages the knowledge graph we already have!**

## Next Steps

1. ✅ Design enhanced Kuzu schema (v3.0 with Facts and Entities)
2. ✅ Write fact extraction script
3. ✅ Populate facts for corpus (one-time operation)
4. ✅ Write Cypher query templates
5. ✅ Implement fact-based summarization
6. ✅ Test and compare vs RAG-only approach

Ready to implement the fact-based architecture?
