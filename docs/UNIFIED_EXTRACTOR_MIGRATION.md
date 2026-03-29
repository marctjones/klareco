# Unified Extractor Migration Plan

## Status: Phase 1 Complete (Core Architecture)

Created `klareco/rag/unified_extractor.py` with foundational architecture that eliminates duplication between ASTAnswerExtractor and FactExtractor.

## What's Implemented ✅

### 1. Unified Verb Semantics (Lines 72-131)
```python
VERB_SEMANTICS = {
    'est': {
        'relation': RelationType.IS_A,
        'synonyms': ['konstitu', 'konsist'],
        'answer_extraction': 'predicate_nominative',
    },
    'kre': {
        'relation': RelationType.CREATED_BY,
        'synonyms': ['fond', 'establ', 'konstruk'],
        'answer_extraction': 'subject_agent',
    },
    # ... 10 verb patterns total
}
```

**Achievement**: Single source of truth for verb-to-relation mappings + synonyms. No more duplication.

### 2. Common AST Traversal Methods (Lines 275-439)
- `_get_verb_root()` - Extract verb from AST (checks verbo + aliaj fallback)
- `_get_entity_name()` - Extract entity names with proper noun capitalization
- `_vortgrupo_to_text()` - Convert AST node to text for answers
- `_are_verbs_similar()` - Synonym matching using unified semantics
- `_extract_modifiers()` - Extract time/location/manner modifiers from aliaj

**Achievement**: Eliminated ~200 lines of duplicated traversal logic.

### 3. Fact Extraction (mode='facts') - Lines 213-273, 440-647
- `extract()` - Main entry point with mode parameter
- `_extract_as_facts()` - Orchestrates fact extraction
- `_extract_fact_from_frazo()` - Dispatcher by relation type
- Relation-specific extractors:
  - `_extract_is_a_fact()` - "X estas Y" patterns
  - `_extract_created_by_fact()` - "X kreis Y" patterns
  - `_extract_has_fact()` - "X havas Y" patterns
  - `_extract_located_at_fact()` - "X loĝas en Y" patterns
  - `_extract_born_fact()` - "X naskiĝis" patterns
  - `_extract_published_fact()` - "X publikigis Y" patterns
  - `_extract_action_fact()` - Generic action fallback

**Achievement**: Complete fact extraction pipeline using unified verb semantics.

### 4. Answer Span Extraction (mode='spans') - Lines 178-212, 648-826
- `extract_answer()` - Main entry point for question-aware extraction
- `_detect_question_type()` - Auto-detect WHO/WHERE/WHEN/etc from correlatives
- `_check_correlative()` - Parse correlative suffix (u/o/e/am/om/etc)
- `_is_complex_sentence()` - Detect multi-clause sentences
- `_is_clause_boundary()` - Identify participials, conjunctions, relative pronouns
- Answer extractors (currently placeholders):
  - `_extract_who_answer()` - Person/agent extraction
  - `_extract_what_answer()` - Thing/concept extraction
  - `_extract_where_answer()` - Location extraction
  - `_extract_when_answer()` - Time extraction
  - `_extract_how_many_answer()` - Quantity extraction
  - `_extract_why_answer()` - Reason/cause extraction
  - `_extract_how_answer()` - Manner extraction
  - `_extract_whose_answer()` - Possession extraction

**Achievement**: Framework for question-aware extraction with subclause scoring support.

### 5. Validation (Lines 827-865)
- `_validate_answer()` - Type checking for extracted answers
- Rejects correlatives (function words)
- Rejects pronouns for WHO questions
- Additional validation rules expandable

## What Needs Completion 🚧

### Phase 2: Complete Answer Span Extractors (Priority: HIGH)

Copy full implementations from `answer_extractor.py`:

1. **_extract_who_answer()** (lines 636-821)
   - Multi-candidate ranking
   - Passive voice detection
   - Query entity filtering
   - Proximity scoring
   - Validation
   - **Complexity**: 186 lines, uses: `_is_person()`, `_is_passive_voice()`, `_extract_accusative_object_root()`, `_matches_root()`, `_score_candidate_proximity()`

2. **_extract_what_answer()** (lines 823-965)
   - Predicate extraction for "estas" questions
   - Object/subject candidates
   - Multi-candidate ranking
   - **Complexity**: 143 lines, uses: `_is_correlative()`, `_score_candidate_proximity()`

3. **_extract_where_answer()** (lines 967-1088)
   - Location preposition patterns
   - -ej suffix detection
   - Place name gazetteer
   - **Complexity**: 122 lines, uses: `_is_place()`, `_get_suffixes()`

4. **_extract_when_answer()** (lines 1090-1154)
   - Time preposition patterns
   - Year/date recognition
   - Time adverbs
   - **Complexity**: 65 lines, uses: `_looks_like_time()`

5. **_extract_how_many_answer()** (lines 1156-1212)
   - Numeric modifier extraction
   - Number word recognition
   - **Complexity**: 57 lines, uses: `_is_number_word()`

6. **_extract_why_answer()** (lines 1214-1260)
   - Causal marker detection (pro, ĉar, por)
   - **Complexity**: 47 lines

7. **_extract_how_answer()** (lines 1262-1321)
   - Manner preposition patterns
   - Adverb extraction
   - **Complexity**: 60 lines

8. **_extract_whose_answer()** (lines 1345-1386)
   - Possessive "de" patterns
   - **Complexity**: 42 lines

**Total to copy**: ~620 lines of answer extraction logic

### Phase 3: Complete Participial/Nested Clause Extraction (Priority: MEDIUM)

Copy from `fact_extractor.py`:

1. **_extract_from_participial_nouns()** (lines 359-481)
   - Pattern: "La kreinto de Esperanto" → CREATED-BY fact
   - Uses: `_is_participial_noun()`, `_find_prepositional_object()`
   - **Complexity**: 123 lines

2. **_extract_from_nested_clauses()** (lines 525-705)
   - Pattern: "...kiun Zamenhof kreis..." → CREATED-BY fact
   - Relative clause detection
   - Subsequence extraction
   - **Complexity**: 181 lines, uses: `_find_verb_after_position()`, `_extract_from_clause_subsequence()`

**Total to copy**: ~304 lines

### Phase 4: Add Helper Methods (Priority: HIGH)

Copy from `answer_extractor.py`:

1. **_is_person()** (lines 1842-1928) - 87 lines
   - Suffix checks (-ul, -ist, -in, participles)
   - Proper noun detection
   - Place name exclusion
   - Function word exclusion

2. **_is_place()** (lines 1930-1967) - 38 lines
   - -ej suffix check
   - Place name gazetteer
   - Location root patterns

3. **_get_suffixes()** (lines 1969-1979) - 11 lines
   - Extract suffix list from node

4. **_is_passive_voice()** (lines 1773-1816) - 44 lines
   - Detect passive participle in subjekto modifiers

5. **_extract_accusative_object_root()** (lines 1708-1738) - 31 lines
   - Extract object from query for filtering

6. **_matches_root()** (lines 1740-1751) - 12 lines
   - Check if text matches root pattern

7. **_looks_like_time()** (lines 1994-2022) - 29 lines
   - Year/month/time word detection

8. **_is_number_word()** (lines 2024-2039) - 16 lines
   - Numeric word recognition

9. **Position tracking and proximity scoring** (lines 2041-2295) - 255 lines
   - `_get_word_position()`
   - `_find_node_position()`
   - `_nodes_equal()`
   - `_count_words()`
   - `_find_root_positions()`
   - `_find_root_in_node()`
   - `_score_candidate_proximity()`

**Total to copy**: ~523 lines

### Phase 5: Add Subclause Scoring (Priority: MEDIUM)

Copy from `answer_extractor.py`:

1. **_extract_from_best_subclause()** (lines 492-571) - 80 lines
   - Decompose into subclauses
   - Score each subclause
   - Extract from best match

2. **_extract_subclauses()** (lines 1512-1570) - 59 lines
   - Split by clause boundaries

3. **_make_subclause()** (lines 1572-1617) - 46 lines
   - Construct subclause structure

4. **_score_subclause()** (lines 1619-1653) - 35 lines
   - Relevance scoring

5. **_extract_roots()** (lines 1655-1683) - 29 lines
   - Extract roots for scoring

6. **_extract_roots_from_node()** (lines 1685-1702) - 18 lines
   - Recursive root extraction

**Total to copy**: ~267 lines

### Phase 6: Add Multi-Doc Aggregation (Priority: LOW)

Copy from `answer_extractor.py`:

1. **extract_answer_from_multiple_docs()** (lines 312-472) - 161 lines
   - Extract from top-N documents
   - Aggregate candidates by occurrence
   - Weight by multi-doc evidence

## Total Code to Complete

- **Phase 2**: 620 lines (answer extractors)
- **Phase 3**: 304 lines (participial/nested facts)
- **Phase 4**: 523 lines (helpers)
- **Phase 5**: 267 lines (subclause scoring)
- **Phase 6**: 161 lines (multi-doc aggregation)

**Grand Total**: ~1,875 lines to copy/adapt

**Current file size**: ~1,100 lines
**Estimated final size**: ~2,975 lines (vs 3,075 lines combined in old system)

## Migration Strategy

### Step 1: Complete High-Priority Methods
1. Copy Phase 2 (answer extractors) - enables question-aware extraction
2. Copy Phase 4 (helpers) - required by answer extractors
3. **Test on 10-question subset** - verify basic functionality

### Step 2: Complete Medium-Priority Methods
1. Copy Phase 3 (participial/nested) - enhances fact extraction
2. Copy Phase 5 (subclause scoring) - handles complex sentences
3. **Test on full 50-question set** - verify no accuracy regression

### Step 3: Update Pipeline Integration
1. Modify `ExtractiveAnswerGenerator` to import UnifiedASTExtractor
2. Replace FactExtractor + ASTAnswerExtractor with unified version
3. Add compatibility shim if needed
4. **Run evaluation** - compare to baseline (30% accuracy)

### Step 4: Complete Low-Priority Methods
1. Copy Phase 6 (multi-doc aggregation) - optional enhancement
2. **Final evaluation** - measure improvement

### Step 5: Deprecation
1. Add deprecation warnings to `answer_extractor.py` and `fact_extractor.py`
2. Update all imports across codebase
3. Remove old files after 2 weeks of testing

## Expected Benefits

- **Code reduction**: ~100 lines saved (3,075 → 2,975)
- **Maintenance**: Single system to update, not two
- **Performance**: Single AST traversal per sentence
- **Clarity**: Output format is a choice, not an architectural split
- **Consistency**: Same verb semantics across all extraction

## Testing Plan

After each phase:
```bash
# Unit tests
python -m pytest tests/test_unified_extractor.py -v

# Integration test (10 questions)
python scripts/evaluate_extractive_qa.py --limit 10

# Full evaluation (50 questions)
python scripts/evaluate_extractive_qa.py

# Compare to baseline
# Baseline: 15/50 (30% accuracy) with separate extractors
# Target: >= 15/50 (no regression), ideally better
```

## Next Session Tasks

1. **Complete Phase 2**: Copy all 8 answer extraction methods
2. **Complete Phase 4**: Copy all 9 helper methods
3. **Test basic functionality**: Run on 10-question subset
4. **If passing**: Proceed to Phase 3 and Phase 5

## File Status

- ✅ `klareco/rag/unified_extractor.py` - Created with core architecture
- 🚧 `klareco/rag/extractive_answering.py` - Needs update to use unified extractor
- 📋 `klareco/rag/answer_extractor.py` - To be deprecated
- 📋 `klareco/rag/fact_extractor.py` - To be deprecated
