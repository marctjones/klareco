---
id: 140
title: 'Phase 3: Add sentence adjacency edges for context retrieval'
state: open
created: '2026-01-08T00:48:17.353261Z'
labels:
- enhancement
- kuzu
- retrieval
- coreference
priority: high
---
## Objective

Add NEXT_SENTENCE edges between adjacent sentences to enable context-aware retrieval and coreference resolution.

## The Problem

Current retrieval returns isolated sentences. When the answer is:
```
"Zamenhof kreis sian lingvon en 1887."
```

But the query asks about "Esperanto", we miss it because the sentence uses "sian lingvon" (his language) instead of "Esperanto". The previous sentence likely mentions Zamenhof creating Esperanto explicitly.

## Solution

Add sentence adjacency edges:
```cypher
CREATE REL TABLE NEXT_SENTENCE (FROM Sentence TO Sentence)
CREATE REL TABLE IN_DOCUMENT (FROM Sentence TO Document)
```

## Use Cases

### 1. Context Expansion
Get surrounding sentences for any retrieved sentence:
```cypher
MATCH (s:Sentence {id: $sent_id})
OPTIONAL MATCH (prev:Sentence)-[:NEXT_SENTENCE]->(s)
OPTIONAL MATCH (s)-[:NEXT_SENTENCE]->(next:Sentence)
RETURN prev.text, s.text, next.text
```

### 2. Coreference Resolution
Find if previous sentence mentions an entity:
```cypher
// Retrieved sentence has "Li kreis lingvon"
// Check if previous sentence has "Zamenhof"
MATCH (s:Sentence)-[:HAS_ROOT]->(r:Root {root: 'kre'})
MATCH (prev:Sentence)-[:NEXT_SENTENCE]->(s)
MATCH (prev)-[:HAS_ROOT]->(entity:Root {root: 'zamenhof'})
RETURN s.text, prev.text, entity.root AS antecedent
```

### 3. Context Boosting
Boost score of sentences adjacent to highly-relevant sentences:
```cypher
MATCH (s:Sentence)-[:HAS_ROOT]->(r:Root {root: 'esperant'})
WITH s, 1.0 AS base_score
OPTIONAL MATCH (prev:Sentence)-[:NEXT_SENTENCE]->(s)
OPTIONAL MATCH (s)-[:NEXT_SENTENCE]->(next:Sentence)
RETURN s.id, s.text, base_score,
       prev.id AS prev_id, next.id AS next_id
```
Then boost prev/next by 0.5 * base_score in Python.

### 4. Paragraph Reconstruction
Get full paragraph for a sentence:
```cypher
MATCH (s:Sentence {id: $sent_id})-[:IN_DOCUMENT]->(d:Document)
MATCH (all:Sentence)-[:IN_DOCUMENT]->(d)
WHERE all.sent_idx >= s.sent_idx - 2 
  AND all.sent_idx <= s.sent_idx + 2
RETURN all.text ORDER BY all.sent_idx
```

## Implementation

### Step 1: Extend schema
- Add NEXT_SENTENCE edge table
- Add Document node table
- Add IN_DOCUMENT edge table
- Add sent_idx property to Sentence

### Step 2: Build edges during indexing
- Track previous sentence ID while streaming corpus
- Create NEXT_SENTENCE edge between consecutive sentences
- Create IN_DOCUMENT edges

### Step 3: Add context retrieval methods
- `get_context(sent_id, window=1)` → surrounding sentences
- `get_antecedent(sent_id, entity_root)` → check if entity in previous sentences

### Step 4: Integrate with scoring
- Add context_boost parameter to search
- Optionally expand results with context

## Success Criteria
- [ ] All sentence adjacency edges created
- [ ] `get_context()` returns correct surrounding sentences
- [ ] Coreference query finds antecedents
- [ ] Context boosting improves retrieval on test queries
- [ ] "Kiu fondis Esperanton?" finds Zamenhof via context

## Depends On
- Task #138 (Kuzu migration complete)

## Supersedes
- Task #137 (sentence-level graph for context)
