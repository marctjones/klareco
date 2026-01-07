---
id: 92
title: Add named entity boosting for WHO/WHERE questions in ASTAware
state: open
created: '2026-01-06T22:08:49.431882Z'
labels:
- enhancement
- retrieval
priority: high
---
## Problem

When the query contains "Kiu" (who), the retriever should prioritize documents with proper nouns in subject position. Currently all documents with matching verbs/objects rank similarly regardless of whether they contain person names.

## Evidence

Query: "Kiu fondis Esperanton?"
- Expected: Documents with proper noun subjects like "Zamenhof fondis..."
- Actual top results: "li fondis firmaeton", "ŝi fondis la Constructora Benéfica"
- Problem: Generic pronouns (li, ŝi, oni) rank the same as proper nouns

## Proposed Solution

Add entity-type boosting in the reranking stage:

```python
QUESTION_ENTITY_MAP = {
    'kiu': 'PERSON',      # who → boost proper nouns (likely people)
    'kie': 'PLACE',       # where → boost location entities
    'kiam': 'TIME',       # when → boost date/time expressions
    'kio': 'THING',       # what → neutral
}

def _rerank_with_entity_boost(self, candidates, question_type):
    for score, doc in candidates:
        # Check if document has matching entity type
        doc_subj = doc.get('slots', {}).get('SUBJ')
        
        if question_type == 'PERSON':
            # Boost if subject is proper noun (capitalized, not in common vocab)
            if is_proper_noun(doc_subj):
                score *= 1.3  # 30% boost
        
        elif question_type == 'PLACE':
            # Boost if contains location preposition phrase
            if has_location_prep(doc):
                score *= 1.3
        
        yield (score, doc)
```

## Files to Modify

- `klareco/rag/ast_aware_retriever.py`: Add `_rerank_with_entity_boost()` method
- `klareco/rag/entity_recognizer.py`: May need `is_proper_noun()` helper

## Dependencies

- Related to #228 (entity type annotation for proper nouns)

## Expected Impact

- Factual question accuracy: 20% → 40%+ 
- Better ranking for WHO/WHERE/WHEN questions

## Acceptance Criteria

- [ ] Entity type detection for subject slots
- [ ] Boosting factor applied during reranking
- [ ] Evaluation shows improvement on factual questions
