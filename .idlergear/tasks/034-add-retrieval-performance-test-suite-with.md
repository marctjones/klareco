---
id: 34
title: Add retrieval performance test suite with Wikipedia queries
state: open
created: '2026-01-05T01:09:48.286865Z'
labels:
- testing
- retrieval
priority: high
---
## Problem

No systematic testing of retrieval quality on Wikipedia content. We need benchmark queries to validate:
- Wikipedia data is properly indexed
- Retrieval returns relevant Wikipedia articles
- Ranking is appropriate

## Solution

Create test suite in `tests/test_wikipedia_retrieval.py`:

```python
WIKIPEDIA_TEST_QUERIES = [
    # Factual queries (should find Wikipedia)
    {
        'query': 'Kiu fondis Esperanton?',
        'expected_article': 'L. L. Zamenhof',
        'min_recall': 0.9,
    },
    {
        'query': 'Kio estas la Unu Ringo?',
        'expected_article': 'Unu Ringo',
        'min_recall': 0.9,
    },
    {
        'query': 'Kie loĝas prezidento de Usono?',
        'expected_article': 'Blanka Domo',
        'min_recall': 0.8,
    },
    # Esperanto history queries
    {
        'query': 'Kiam aperis la unua libro pri Esperanto?',
        'expected_article': 'Esperanto',
        'min_recall': 0.9,
    },
    {
        'query': 'Kio estas la himno de Esperanto?',
        'expected_article': 'La Espero',
        'min_recall': 0.9,
    },
]

def test_wikipedia_recall():
    """Test that Wikipedia articles are found for factual queries."""
    retriever = load_retriever('data/indexes/slot_full')
    
    for test_case in WIKIPEDIA_TEST_QUERIES:
        results = retriever.search(test_case['query'], top_k=10)
        
        # Check if expected article appears in top-10
        found = any(
            test_case['expected_article'] in result['source'].get('article_title', '')
            for score, result in results
        )
        
        assert found, f"Expected article not found for: {test_case['query']}"
```

## Test Categories

1. **Factual recall**: Finding specific Wikipedia articles
2. **Ranking quality**: Expected article in top-3
3. **Coverage**: Queries across different domains
4. **Negation**: "Kiu NE fondis Esperanton?"
5. **Temporal**: "Kiam..." queries

## Acceptance Criteria

- [ ] 20+ test queries covering different topics
- [ ] Tests run in CI pipeline
- [ ] Baseline metrics established (recall, MRR, nDCG)
- [ ] Tests fail when Wikipedia data missing (guards against regression)
