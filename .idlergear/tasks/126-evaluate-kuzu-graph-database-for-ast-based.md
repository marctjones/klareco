---
id: 126
title: Evaluate Kuzu graph database for AST-based reasoning
state: closed
created: '2026-01-07T02:58:35.540300Z'
labels:
- enhancement
- reasoning
- future
priority: low
---
## Motivation

Klareco's future reasoning core needs to handle multi-hop queries:
- "Who founded the language that uses -uj- suffix?"
- "What is the capital of the country where Zamenhof was born?"

These require traversing relationships between entities and concepts.

## Proposed Solution

Kuzu is an embedded graph database optimized for:
- Property graphs with typed edges
- Cypher-like query language
- OLAP-style analytics on graphs

## Graph Schema for Klareco

```
Nodes:
- Document(id, text, source, parse_rate)
- Entity(name, type: PERSON|PLACE|CONCEPT|WORK)
- Root(root, embedding[128])
- GrammarFeature(name, value)

Edges:
- MENTIONS(Document -> Entity, slot: SUBJ|VERB|OBJ, position)
- DEFINES(Document -> Entity)  # "Hundo estas mamulo"
- CREATED_BY(Entity -> Entity)  # Esperanto CREATED_BY Zamenhof
- HAS_ROOT(Document -> Root, slot, position)
- HAS_FEATURE(Document -> GrammarFeature)
```

## Example Queries

```cypher
// Find documents that define "hundo"
MATCH (d:Document)-[:DEFINES]->(e:Entity {name: "hundo"})
RETURN d.text

// Multi-hop: Who created the language with -uj- suffix?
MATCH (d:Document)-[:MENTIONS]->(lang:Entity {type: "LANGUAGE"})
WHERE d.text CONTAINS "-uj-"
MATCH (creator:Entity)-[:CREATED]->(lang)
RETURN creator.name

// Find all SUBJ-VERB-OBJ patterns with "fondis"
MATCH (d:Document)-[s:HAS_ROOT {slot: "SUBJ"}]->(subj:Root)
MATCH (d)-[v:HAS_ROOT {slot: "VERB"}]->(verb:Root {root: "fond"})
MATCH (d)-[o:HAS_ROOT {slot: "OBJ"}]->(obj:Root)
RETURN subj.root, obj.root, d.text
```

## Benefits

1. **Multi-hop reasoning** - Native support for relationship traversal
2. **Pattern matching** - Cypher queries for AST patterns
3. **Knowledge graph** - Build entity relationships from corpus
4. **Explainability** - Query paths show reasoning chain

## Challenges

1. **Entity extraction**: Need to build entity recognition pipeline first
2. **Relationship extraction**: Need to identify CREATED_BY, CAPITAL_OF, etc.
3. **Scale**: 4.4M documents → potentially billions of edges
4. **Integration**: How does graph query combine with vector search?

## Phased Approach

**Phase 1** (Current): Vector retrieval + AST pattern matching (done)
**Phase 2** (Near-term): ChromaDB/RocksDB for better storage
**Phase 3** (Future): Kuzu for reasoning when we have:
- Entity recognition pipeline
- Relationship extraction from AST patterns
- Defined reasoning patterns (multi-hop Q&A)

## Tasks

1. [ ] Install Kuzu and prototype with small knowledge graph
2. [ ] Design schema for Esperanto linguistic entities
3. [ ] Write ingestion pipeline: Document → Entities → Relationships
4. [ ] Implement graph-augmented retrieval (combine vector + graph)
5. [ ] Benchmark multi-hop query performance

## Alignment with Architecture

- **Deterministic**: Graph queries are rule-based, not learned
- **Explainable**: Query paths provide reasoning trace
- **Composable**: Graph reasoning + vector retrieval + AST analysis
- **Future reasoning core**: Foundation for the 20-100M param reasoning model
