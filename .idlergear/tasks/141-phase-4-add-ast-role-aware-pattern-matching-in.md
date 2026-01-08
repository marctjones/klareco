---
id: 141
title: 'Phase 4: Add AST role-aware pattern matching in Kuzu'
state: open
created: '2026-01-08T00:48:54.565208Z'
labels:
- enhancement
- kuzu
- retrieval
- ast
priority: high
---
## Objective

Enable queries that match AST roles (subjekto/verbo/objekto), leveraging Esperanto's unique fully-parsed structure.

## The Problem

Current retrieval finds documents containing roots, but ignores their grammatical role:
- Query: "Kiu fondis Esperanton?" (Who founded Esperanto?)
- We want: sentences where "fond" is VERB and "esperant" is OBJECT
- We get: all sentences containing either root in any role

This returns noise like "La fondinto de Esperanto..." where "fond" is a noun (fondinto), not a verb.

## Solution

Query HAS_ROOT edges with role attribute:
```cypher
// Find sentences where fond/kre is VERB and esperant is OBJECT
MATCH (s:Sentence)-[:HAS_ROOT {role: 'verbo'}]->(v:Root)
WHERE v.root IN ['fond', 'kre', 'establ']  // Expanded synonyms
MATCH (s)-[:HAS_ROOT {role: 'objekto'}]->(o:Root)
WHERE o.root IN ['esperant', 'lingv']
RETURN s.id, s.text
```

## Use Cases

### 1. Question-Aligned Retrieval
Match query structure to document structure:

| Question Type | Query Pattern |
|---------------|---------------|
| Kiu (Who) | SUBJ is answer, match VERB+OBJ |
| Kion (What-obj) | OBJ is answer, match SUBJ+VERB |
| Kio estas X | X is SUBJ, answer is predicate |

```cypher
// "Kiu fondis Esperanton?" → find SUBJ where VERB=fond, OBJ=esperant
MATCH (s:Sentence)-[:HAS_ROOT {role: 'subjekto'}]->(subj:Root)
MATCH (s)-[:HAS_ROOT {role: 'verbo'}]->(v:Root {root: 'fond'})
MATCH (s)-[:HAS_ROOT {role: 'objekto'}]->(o:Root {root: 'esperant'})
RETURN subj.root, s.text
```

### 2. Definition Queries
"Kio estas hundo?" → Find "X estas hundo" patterns:
```cypher
MATCH (s:Sentence)-[:HAS_ROOT {role: 'verbo'}]->(v:Root {root: 'est'})
MATCH (s)-[:HAS_ROOT {role: 'subjekto'}]->(subj:Root {root: 'hund'})
MATCH (s)-[:HAS_ROOT {role: 'predikato'}]->(pred:Root)
RETURN pred.root, s.text
// Returns: "besto", "mamulo", etc.
```

### 3. Verb-Object Collocation
Find what objects go with a verb:
```cypher
MATCH (s:Sentence)-[:HAS_ROOT {role: 'verbo'}]->(v:Root {root: 'manĝ'})
MATCH (s)-[:HAS_ROOT {role: 'objekto'}]->(obj:Root)
RETURN obj.root, count(*) AS freq
ORDER BY freq DESC LIMIT 10
// Returns: pan, pom, viand, ...
```

### 4. Passive Voice Handling
In passive, roles flip. Detect and adjust:
```cypher
// "Esperanto estis fondita de Zamenhof"
// Here: esperant is SUBJ, but semantically is OBJECT of founding
MATCH (s:Sentence)-[:HAS_ROOT {role: 'verbo'}]->(v:Root)
WHERE v.root ENDS WITH 'it'  // Past passive participle
MATCH (s)-[:HAS_ROOT {role: 'subjekto'}]->(subj:Root)
MATCH (s)-[:HAS_ROOT {role: 'aliaj'}]->(agent:Root)  // "de Zamenhof"
RETURN subj.root AS patient, agent.root AS agent, s.text
```

## Implementation

### Step 1: Ensure role is indexed
HAS_ROOT edge already has role attribute from Phase 1.
Add index if needed for performance.

### Step 2: Create role-aware search method
```python
def search_by_pattern(
    self,
    verb_roots: List[str],
    obj_roots: List[str] = None,
    subj_roots: List[str] = None,
) -> List[Sentence]:
    # Build Cypher query based on provided roles
    # Execute and return matches
```

### Step 3: Integrate with question classifier
- QuestionClassifier already detects question type
- Use question type to determine which pattern to match
- "Kiu" → match VERB+OBJ, return SUBJ
- "Kion" → match SUBJ+VERB, return OBJ

### Step 4: Add to ASTAwareRetriever
- Parse query to AST
- Extract roles from query AST
- Build pattern query
- Combine with BM25 scoring

## Query AST → Pattern Mapping

```python
def query_ast_to_pattern(query_ast):
    """Convert query AST to Cypher pattern."""
    verb = query_ast.get('verbo', {}).get('radiko')
    obj = query_ast.get('objekto', {}).get('radiko')
    subj = query_ast.get('subjekto', {}).get('radiko')
    
    # Expand each with synonyms
    verb_expanded = expand_synonyms(verb) if verb else None
    obj_expanded = expand_synonyms(obj) if obj else None
    subj_expanded = expand_synonyms(subj) if subj else None
    
    return build_cypher_pattern(verb_expanded, obj_expanded, subj_expanded)
```

## Success Criteria
- [ ] Role-filtered queries return fewer, more relevant results
- [ ] "Kiu fondis Esperanton?" returns sentences with fond as VERB
- [ ] Definition queries ("Kio estas X?") find predicate patterns
- [ ] Q&A benchmark accuracy improves
- [ ] Latency acceptable (< 50ms for pattern queries)

## Depends On
- Task #138 (Kuzu migration complete)
- Task #139 (synonym expansion for role values)

## Related
- Leverages Esperanto's deterministic case marking (-n for accusative)
- AST parser already extracts roles correctly
