# Refactoring Proposal: Unify AST Extraction Systems

## Problem

Currently we have TWO independent AST-based extraction systems:

1. **ASTAnswerExtractor** (`answer_extractor.py`)
   - Extracts answer spans (text fragments)
   - Question-aware
   - Multi-document aggregation
   - Output: `"Zamenhof"` (string)

2. **FactExtractor** (`fact_extractor.py`)
   - Extracts structured facts (triples)
   - Generic (not question-aware)
   - Single-document focus
   - Output: `Fact(subject="Zamenhof", relation="CREATED", object="Esperanto")`

**Both do the same core work:**
- Parse AST structure
- Identify subjects, verbs, objects
- Map verbs to relation types
- Handle special patterns (IS-A, CREATED, BORN, etc.)

**The only difference is output format** (span vs triple), which is wasteful.

## Proposed Solution

### Phase 1: Create Unified Extractor Base Class

```python
class UnifiedASTExtractor:
    """
    Single AST extraction system that can output multiple formats.

    Core extraction logic is shared. Output format is a parameter.
    """

    def extract(self, ast: Dict, mode: str = 'facts') -> Union[List[Fact], Dict]:
        """
        Extract semantic information from AST.

        Args:
            ast: Parsed AST
            mode: 'facts' (structured triples) or 'spans' (answer text)

        Returns:
            List[Fact] if mode='facts'
            Dict with answer span info if mode='spans'
        """
        # Single traversal of AST
        frazo = ast if ast.get('tipo') == 'frazo' else None
        if not frazo:
            return [] if mode == 'facts' else None

        # Common extraction logic
        subject = self._extract_subject(frazo)
        verb = self._extract_verb(frazo)
        object_ = self._extract_object(frazo)
        relation = self._map_verb_to_relation(verb)

        # Format output based on mode
        if mode == 'facts':
            return self._format_as_facts(subject, verb, object_, relation)
        elif mode == 'spans':
            return self._format_as_spans(subject, verb, object_, relation)
```

### Phase 2: Refactor ExtractiveAnswerGenerator

```python
class ExtractiveAnswerGenerator:
    def __init__(self, ...):
        # Single extractor, not two
        self.extractor = UnifiedASTExtractor()

    def generate(self, sentences, query, question_type, ...):
        # Try direct answer extraction first
        query_ast = parse(query)

        # Extract answer spans (question-aware mode)
        answer_span = self.extractor.extract_answer_span(
            query_ast=query_ast,
            sentences=sentences[:20],
            question_type=question_type
        )

        if answer_span and answer_span['confidence'] >= 0.7:
            # High-confidence direct answer
            return self._format_direct_answer(answer_span)

        # Fallback: Extract facts for multi-sentence answer
        all_facts = []
        for sent in sentences:
            facts = self.extractor.extract(sent['ast'], mode='facts')
            all_facts.extend(facts)

        # Continue with existing fact scoring, filtering, discourse planning
        ...
```

### Phase 3: Consolidate Verb Mappings

Currently duplicated in both classes:

**answer_extractor.py:**
```python
VERB_SYNONYMS = {
    'est': ['konstitu', 'konsist'],
    'kre': ['fond', 'establ', 'konstru'],
    ...
}
```

**fact_extractor.py:**
```python
VERB_TO_RELATION = {
    'est': RelationType.IS_A,
    'kre': RelationType.CREATED_BY,
    ...
}
```

**Unified:**
```python
VERB_SEMANTICS = {
    'est': {
        'relation': RelationType.IS_A,
        'synonyms': ['konstitu', 'konsist'],
        'answer_extraction': 'predicate_nominative'
    },
    'kre': {
        'relation': RelationType.CREATED_BY,
        'synonyms': ['fond', 'establ', 'konstru'],
        'answer_extraction': 'subject_agent'
    },
    ...
}
```

## Benefits

1. **Single AST traversal** - Process each sentence once, not twice
2. **Single source of truth** for verb semantics
3. **Easier to maintain** - Change logic in one place
4. **Better performance** - No duplicate work
5. **Clearer architecture** - Output format is a choice, not a system split

## Migration Path

1. Create `UnifiedASTExtractor` with both output modes
2. Update `ExtractiveAnswerGenerator` to use unified extractor
3. Deprecate old `ASTAnswerExtractor` and `FactExtractor`
4. Run evaluation to ensure no accuracy loss
5. Remove deprecated classes

## Expected Impact

- **Code reduction**: ~40% fewer lines (eliminate duplication)
- **Performance**: ~20% faster (single AST traversal)
- **Accuracy**: Same or better (easier to improve one system than two)
