---
id: 137
title: Add sentence-level graph index for context boosting
state: closed
created: '2026-01-08T00:20:19.474779Z'
labels:
- enhancement
- retrieval
priority: high
---
Store sentence adjacency and AST role relationships to enable:
1. Coreference propagation - "Li" refers to entity in previous sentence
2. Context boosting - boost neighbors of relevant sentences  
3. Multi-hop retrieval - follow edges to gather evidence
4. **AST-role matching** - match query roles (subjekto/verbo/objekto) to sentence roles

Recommended: Kuzu graph database (embedded, Cypher queries)

Schema:
- Root nodes with vortspeco (from AST)
- Sentence nodes with subjekto_root, verbo_root, objekto_root
- IS_SYNONYM / IS_HYPERNYM edges between roots
- HAS_ROOT edges with role attribute
- NEXT_SENTENCE edges for context
- IN_DOCUMENT edges for grouping

Key advantage: AST annotations make edges meaningful - "hundo as subject" vs "hundo as object" are different semantic relationships.

See conversation for full Cypher examples.
