---
id: 135
title: Improve RootInvertedIndex ranking to prioritize relevant documents
state: open
created: '2026-01-07T23:28:24.298684Z'
labels:
- enhancement
- retrieval
priority: high
---
## Problem
Query "Kiu fondis Esperanton?" returns taxonomy template pages instead of documents about Zamenhof founding Esperanto.

## Current Behavior
- Documents with many "fondita" mentions (taxonomy templates) score higher than actual answers
- Score of 57.0 for irrelevant template vs expected answer not in top 5

## Root Cause Analysis
- Current scoring counts root occurrences without considering document relevance
- Long documents with many root matches outscore short, focused answers
- No penalty for document length or template-like content

## Potential Solutions
1. **TF-IDF weighting**: Penalize common roots like "fond" (66K occurrences) vs rare roots like "esperant" (701)
2. **Document length normalization**: Shorter, focused documents should score higher
3. **Role-based scoring**: Prioritize subject role matches for WHO questions
4. **Co-occurrence boost**: Documents with query roots appearing close together should score higher
5. **Template filtering**: Detect and penalize Wikipedia template/boilerplate content

## Test Cases
- "Kiu fondis Esperanton?" → Should return Zamenhof-related documents
- "Kio estas Esperanto?" → Should return definition/description documents
- "Kie naskiĝis Zamenhof?" → Should return birthplace information

## Related
- SemanticDB has good synonym expansion (fond → kreinto, aŭtoro, fondinto)
- 7,000 occurrences of "zamenhof" in corpus - answer exists, just not ranked correctly
