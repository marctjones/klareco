# Plan: Complete AST-First Architecture Enforcement

**Date**: 2026-03-28
**Problem**: We keep reverting to BM25 text matching instead of fully leveraging AST grammatical roles
**Root Cause**: AST constraints are not enforced at retrieval level, only at answer extraction level

---

## The Core Issue

### Current (Broken) Pipeline:
```
Query: "Kiu fondis Esperanton?"
  ↓
Parse to AST: subjekto=kiu, verbo=fond, objekto=esperant
  ↓
Extract roots: ['fond', 'esperant']
  ↓
PROBLEM: Expand & retrieve based on TEXT MATCHING ONLY
  - ANY sentence with "fond*" OR "kre*" OR "esperant*" ANYWHERE
  - BM25 ranks by term frequency, ignoring grammatical roles
  - Retrieves: "fondis Esperanto-grupon", "Esperanto-kongreso", etc.
  ↓
Try answer extraction from wrong documents
  ↓
FAILURE: Extract "Tautorat" (founded a group) not "Zamenhof" (created language)
```

### Correct (AST-First) Pipeline:
```
Query: "Kiu fondis Esperanton?"
  ↓
Parse to AST: subjekto=kiu (WHO), verbo=fond (FOUNDED), objekto=esperant (ESPERANTO)
  ↓
ENFORCE AST ROLES IN RETRIEVAL:
  - Require: verbo contains {fond, kre, establ} (verb synonyms)
  - Require: objekto contains {esperant} (OBJECT must be language, not adjective)
  - Return ONLY sentences matching BOTH role constraints
  ↓
Retrieve: "Zamenhof kreis Esperanton", "Zamenhof fondis Esperanton"
  ↓
Extract answer: subjekto from matched sentences
  ↓
SUCCESS: "Zamenhof"
```

---

## Why We Keep Failing

### Symptom 1: BM25 Bias
- I keep debugging BM25 ranking issues
- I keep expanding synonyms without role constraints
- I keep ignoring that AST roles ARE AVAILABLE but unused

### Symptom 2: Partial Implementation
- `WhooshRetriever.retrieve()` has `query_entity` parameter ✓
- But it's only used for word form expansion, NOT grammatical role filtering
- `ASTAnswerExtractor` has grammatical role matching ✓
- But it operates on ALREADY WRONG documents from BM25

### Symptom 3: No Enforcement Mechanism
- No test that verifies AST roles are used in retrieval
- No architecture review preventing BM25-only implementations
- Easy to "fix" by tweaking expansion/ranking instead of using AST

---

## Comprehensive Fix Plan

### Phase 1: Add AST Role Constraints to Kuzu Queries

**Goal**: Query Kuzu graph for sentences matching grammatical role patterns

**Current Kuzu Schema** (from v2.1):
```cypher
(Frazoteksto)-[:HAVAS_AST]->(FrazoAST)
(FrazoAST)-[:SUBJEKTO]->(Vortgrupo)
(FrazoAST)-[:VERBO]->(Vortgrupo)
(FrazoAST)-[:OBJEKTO]->(Vortgrupo)
(Vortgrupo)-[:RADIKO {radiko: "fond"}]->()
```

**New Retrieval Query**:
```python
def retrieve_with_ast_roles(
    self,
    query_ast: Dict,
    top_k: int = 20
) -> List[Dict]:
    """
    Retrieve sentences matching GRAMMATICAL ROLE constraints from query AST.

    For "Kiu fondis Esperanton?":
    - subjekto: kiu (extract this as answer)
    - verbo: {fond, kre, establ} (verb + synonyms)
    - objekto: {esperant} (entity in accusative)

    Returns ONLY sentences where ALL roles match.
    """
    # Extract constraints from query AST
    question_type = self._detect_question_type(query_ast)

    # Get verb root + synonyms
    verb_node = query_ast.get('aliaj', [])  # or verbo if parser sets it
    verb_root = extract_verb_root(verb_node)
    verb_synonyms = get_synonyms(verb_root, max_count=3)
    verb_constraint = [verb_root] + list(verb_synonyms)

    # Get object entity
    obj_node = find_accusative_object(query_ast)
    obj_root = extract_root(obj_node)

    # Build Kuzu query with role constraints
    kuzu_query = f"""
        MATCH (ft:Frazoteksto)-[:HAVAS_AST]->(ast:FrazoAST)
        WHERE ast.verbo.radiko IN {verb_constraint}
          AND ast.objekto.radiko = '{obj_root}'
          AND ast.objekto.kazo = 'akuzativo'
        RETURN ft.id, ft.teksto, ast
        LIMIT {top_k}
    """

    # Execute and return
    result = self.kuzu_conn.execute(kuzu_query)
    return parse_results(result)
```

**Why This Works**:
- Queries graph by GRAMMATICAL STRUCTURE, not text
- "esperant" in OBJECT role ≠ "esperant" in adjective position
- Can't match "fondis Esperanto-grupon" if "grup" is object, not "esperant"
- Returns ONLY sentences with correct grammatical pattern

---

### Phase 2: Modify WhooshRetriever to Use AST-First Cascade

**Goal**: Try AST role retrieval FIRST, fall back to BM25 only if needed

```python
class WhooshRetriever:
    def retrieve(self, query_roots, top_k, query_ast=None, ...):
        """
        Cascading retrieval: AST-first, BM25 fallback.

        1. If query_ast provided:
           - Try AST role-based retrieval from Kuzu
           - If >= 5 results, return (AST retrieval successful)

        2. Fallback to BM25:
           - Use current Whoosh FTS with expansions
           - Filters meta-content, ranks by BM25
        """
        if query_ast:
            # TRY: AST role-based retrieval
            ast_results = self.retrieve_with_ast_roles(query_ast, top_k)

            if len(ast_results) >= 5:
                logger.info(f"AST role retrieval: {len(ast_results)} matches (sufficient)")
                return ast_results
            else:
                logger.warning(f"AST role retrieval: only {len(ast_results)} matches, falling back to BM25")

        # FALLBACK: BM25 text matching (current implementation)
        return self._retrieve_bm25(query_roots, top_k, ...)
```

**Why This Works**:
- Prioritizes AST structure over text matching
- BM25 fallback handles edge cases (rare verbs, complex structures)
- Logs when fallback happens (visibility into pipeline behavior)

---

### Phase 3: Update demo_extractive_qa.py to Pass query_ast

**Goal**: Ensure AST is passed to retriever, not just roots

```python
def main():
    # ... existing code ...

    # Parse query to AST
    ast = parse(query)

    # Retrieve with AST
    sentences = retriever.retrieve(
        query_roots=query_roots,  # Used for fallback only
        top_k=args.top_k,
        query_ast=ast,  # NEW: Enable AST role retrieval
        question_type=question_type,
        query_entity=query_entity
    )
```

---

### Phase 4: Add Tests to Enforce AST Usage

**Goal**: Prevent backsliding by requiring AST tests to pass

```python
def test_ast_role_retrieval_who_question():
    """
    Test that WHO questions use AST role constraints.

    Query: "Kiu fondis Esperanton?"
    Expected: Retrieve sentences where:
      - verbo.radiko IN ['fond', 'kre', 'establ']
      - objekto.radiko = 'esperant'
      - objekto.kazo = 'akuzativo'

    Should NOT retrieve:
      - "fondis Esperanto-grupon" (objekto = 'grup', not 'esperant')
      - "Esperanto-kongreso okazis" (no matching verb)
    """
    query = "Kiu fondis Esperanton?"
    ast = parse(query)

    retriever = WhooshRetriever(...)
    docs = retriever.retrieve(
        query_roots=['fond', 'esperant'],
        query_ast=ast,
        top_k=10
    )

    # Verify AST role filtering worked
    for doc in docs[:5]:
        doc_ast = doc['ast']

        # Check: objekto must contain "esperant"
        obj_root = extract_root(doc_ast.get('objekto'))
        assert obj_root == 'esperant', \
            f"Object should be 'esperant', got '{obj_root}' in: {doc['text']}"

        # Check: verbo must be creation verb
        verb_root = extract_root(doc_ast.get('verbo'))
        assert verb_root in ['fond', 'kre', 'establ'], \
            f"Verb should be creation verb, got '{verb_root}' in: {doc['text']}"
```

```python
def test_entity_vs_adjective_distinction():
    """
    Test that "Esperanton" (language) ≠ "esperantaj" (adjective).

    Query: "Kiu fondis Esperanton?"
    Should retrieve: "Zamenhof fondis Esperanton" (language)
    Should NOT retrieve: "fondis esperantajn grupojn" (adjective)
    """
    query = "Kiu fondis Esperanton?"
    ast = parse(query)

    retriever = WhooshRetriever(...)
    docs = retriever.retrieve(query_ast=ast, top_k=10)

    # All top docs should have "esperant" as NOUN in OBJECT role
    for doc in docs[:5]:
        assert has_noun_in_object(doc['ast'], 'esperant'), \
            f"Document should have 'esperant' as noun in object: {doc['text']}"
```

**Why This Works**:
- CI fails if AST role constraints are removed
- Forces me to think about grammatical structure
- Documents expected behavior

---

### Phase 5: Architecture Documentation & Linting

**Goal**: Make AST-first approach the ONLY acceptable pattern

#### Update CLAUDE.md:
```markdown
## MANDATORY: AST-First Retrieval

**YOU MUST use AST grammatical role constraints for retrieval.**

DO NOT:
❌ Expand roots and search with BM25 text matching only
❌ Ignore grammatical roles (subject, verb, object)
❌ Use `retriever.retrieve(query_roots=...)` without `query_ast=...`

DO:
✅ Pass `query_ast` to retriever for grammatical role filtering
✅ Use Kuzu queries that match SUBJECT-VERB-OBJECT patterns
✅ Fall back to BM25 only when AST retrieval returns < 5 results

Example (CORRECT):
```python
ast = parse(query)
docs = retriever.retrieve(
    query_roots=roots,       # Fallback only
    query_ast=ast,           # PRIMARY: AST role filtering
    top_k=10
)
```

Example (WRONG):
```python
docs = retriever.retrieve(
    query_roots=roots,  # ❌ Text matching only, ignores grammar
    top_k=10
)
```

**If you find yourself debugging BM25 ranking issues, STOP.**
**The problem is you're not using AST constraints.**
```

#### Add Lint Rule:
```python
# scripts/lint_ast_usage.py
"""
Lint rule: Ensure retrieve() calls pass query_ast parameter.
"""
import ast
import sys

def check_retrieve_calls(file_path):
    """Check that all retriever.retrieve() calls pass query_ast."""
    with open(file_path) as f:
        tree = ast.parse(f.read())

    errors = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check if this is a retrieve() call
            if (hasattr(node.func, 'attr') and
                node.func.attr == 'retrieve'):
                # Check for query_ast keyword argument
                has_ast = any(kw.arg == 'query_ast' for kw in node.keywords)
                if not has_ast:
                    errors.append(f"Line {node.lineno}: retrieve() call missing query_ast parameter")

    return errors

if __name__ == '__main__':
    errors = check_retrieve_calls('scripts/demo_extractive_qa.py')
    if errors:
        print("AST USAGE ERRORS:")
        for err in errors:
            print(f"  {err}")
        sys.exit(1)
```

Run in CI:
```bash
python scripts/lint_ast_usage.py
```

---

## Implementation Order

1. **Phase 1** (2 hours):
   - Implement `retrieve_with_ast_roles()` in WhooshRetriever
   - Test manually with "Kiu fondis Esperanton?"

2. **Phase 2** (1 hour):
   - Add cascading logic (AST-first, BM25 fallback)
   - Add logging for visibility

3. **Phase 3** (30 min):
   - Update demo_extractive_qa.py to pass query_ast
   - Update evaluate_extractive_qa.py

4. **Phase 4** (2 hours):
   - Write test_ast_role_retrieval_who_question()
   - Write test_entity_vs_adjective_distinction()
   - Write tests for WHERE/WHEN/WHAT questions

5. **Phase 5** (1 hour):
   - Update CLAUDE.md with mandatory rules
   - Add lint_ast_usage.py script
   - Add to CI pipeline

**Total estimate**: 6.5 hours

---

## Expected Results

### Before Fix:
```
Query: "Kiu fondis Esperanton?"
Retrieval: 50 docs (BM25 text matching ANY "fond*" OR "kre*" OR "esperant*")
Top docs: About founding Esperanto GROUPS/CONGRESSES
Answer: "Tautorat" (founded a group) ❌
Accuracy: 24% (12/50)
```

### After Fix:
```
Query: "Kiu fondis Esperanton?"
Retrieval: 10 docs (AST role matching: verbo={fond,kre} AND objekto={esperant})
Top docs: "Zamenhof fondis Esperanton", "Zamenhof kreis Esperanton"
Answer: "Zamenhof" ✓
Accuracy: 55-65% (28-33/50) estimated
```

---

## How This Prevents Backsliding

1. **Tests enforce AST usage**
   - CI fails if AST role filtering is removed
   - Can't merge PRs without passing tests

2. **Lint rule catches mistakes**
   - Automatic check for missing `query_ast` parameter
   - Forces AST-first pattern

3. **Documentation makes it mandatory**
   - CLAUDE.md explicitly prohibits BM25-only approaches
   - Provides correct/incorrect examples

4. **Logging provides visibility**
   - Know when AST retrieval succeeds vs falls back
   - Can investigate why fallback happened

5. **Architecture makes AST the default**
   - AST retrieval is PRIMARY path
   - BM25 is explicitly a FALLBACK
   - Can't accidentally skip AST

---

## Open Questions

1. **Kuzu schema completeness**: Does v2.1 Kuzu graph have full AST structure indexed?
   - Check: `MATCH (ast:FrazoAST) RETURN ast LIMIT 1`
   - If not, need to update indexing script

2. **Parser quality**: Does parser correctly identify verbo/objekto for all question types?
   - Current: "Kiu fondis Esperanton?" puts verb+object in `aliaj` (not verbo/objekto)
   - Need: Parser fix OR query logic that handles `aliaj`

3. **Synonym expansion in Kuzu**: Should synonyms be pre-computed in graph?
   - Option A: Query-time expansion: `verbo.radiko IN ['fond', 'kre', 'establ']`
   - Option B: Graph-time links: `(Verbo)-[:SYNONYM]->(Verbo)`
   - Recommendation: Option A (simpler, no graph changes needed)

4. **Performance**: Will AST role queries be fast enough?
   - Kuzu is optimized for graph traversal
   - Test with `EXPLAIN` and benchmark
   - Fallback to BM25 if too slow (but log this!)

---

## Success Criteria

- [ ] `test_ast_role_retrieval_who_question()` passes
- [ ] `test_entity_vs_adjective_distinction()` passes
- [ ] `scripts/lint_ast_usage.py` passes
- [ ] Query "Kiu fondis Esperanton?" returns "Zamenhof" ✓
- [ ] 50-question evaluation: ≥ 50% accuracy (up from 24%)
- [ ] AST retrieval logs show: "AST role retrieval: N matches (sufficient)"

---

## Next Steps

1. Review this plan with user
2. User approves → implement Phase 1
3. Test Phase 1 works → implement Phase 2-5
4. Run full 50-question evaluation
5. Commit all changes with detailed explanation
