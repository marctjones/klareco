# Test Coverage Analysis - Klareco

**Date**: 2025-11-11
**Total Tests**: 147 unit tests + 10 integration tests = **157 tests**
**Current Status**: 140 passing, 7 failing (all due to pipeline refactoring)

---

## Test Results Summary

### Unit Tests: 140/147 Passing (95.2%)

**Failures (7 tests)** - All in `test_pipeline.py` and `test_integration_e2e.py`:
- These tests expect the old pipeline structure (IntentClassifier → Responder)
- Now the pipeline uses Orchestrator instead
- **Action Required**: Update these 7 tests to expect new structure

### Integration Tests: 10/10 Passing (100%) ✅

---

## Coverage by Component

### ✅ Well-Tested Components (Good Coverage)

| Component | Test File | Tests | Lines | Status |
|-----------|-----------|-------|-------|--------|
| Logging | test_logging_config.py | 48 | 498 | ✅ Excellent |
| Integration E2E | test_integration_e2e.py | 15 | 368 | ✅ Good (needs update) |
| Pipeline | test_pipeline.py | 20 | 323 | ✅ Good (needs update) |
| Orchestrator | test_orchestrator_unit.py | 15 | 192 | ✅ Good |
| MathExpert | test_math_expert.py | 18 | 155 | ✅ Good |
| Deparser | test_deparser.py | 5 | 76 | ⚠️  Adequate |
| Parser | test_parser.py | 5 | 74 | ⚠️  **CRITICAL - Needs expansion** |
| Safety | test_safety.py | 5 | 73 | ⚠️  Adequate |
| Trace | test_trace.py | 5 | 73 | ⚠️  Adequate |
| Translator | test_translator.py | 3 | 61 | ⚠️  Adequate |
| Lang ID | test_lang_id.py | 8 | 56 | ✅ Good |
| Front Door | test_front_door.py | 5 | 54 | ⚠️  Adequate |
| Responder | test_responder.py | 1 | 42 | ⚠️  Minimal |
| Intent Classifier | test_intent_classifier.py | 3 | 30 | ⚠️  Minimal |

### ❌ Components with NO Unit Tests

| Component | File | Criticality | Action Required |
|-----------|------|-------------|-----------------|
| **DateExpert** | klareco/experts/date_expert.py | **HIGH** | ❌ Create tests |
| **GrammarExpert** | klareco/experts/grammar_expert.py | **HIGH** | ❌ Create tests |
| **GatingNetwork** | klareco/gating_network.py | **HIGH** | ❌ Create tests |
| ExpertBase | klareco/experts/base.py | MEDIUM | ❌ Create tests |
| AST to Graph | klareco/ast_to_graph.py | MEDIUM | ❌ Create tests |
| DataLoader | klareco/dataloader.py | LOW | ⚠️  Consider tests |
| CLI | klareco/cli.py | LOW | ⚠️  Consider tests |

---

## Critical Gaps Analysis

### 🚨 CRITICAL Priority

#### 1. Parser Tests (CRITICAL - Core Component)
**Current**: Only 5 tests, 74 lines
**Problem**: Parser is the **most critical component** (95.7% accuracy reported) but has minimal test coverage
**Needs Testing**:
- ✅ Edge cases: Unknown roots, ambiguous words
- ✅ Complex sentences: Multiple clauses, nested structures
- ✅ All 16 grammatical rules coverage
- ✅ Prefix combinations (mal-, re-, ge-)
- ✅ Suffix combinations (-ul-, -ej-, -in-, -et-, -ad-, -ig-)
- ✅ All verb tenses (present, past, future, conditional, infinitive)
- ✅ All cases (nominative, accusative)
- ✅ Number (singular, plural)
- ✅ Correlatives (kiu, kio, kie, kiam, etc.)
- ✅ Error handling for invalid input

**Recommendation**: Expand to **50+ tests** covering all morphological features

#### 2. GatingNetwork Tests (HIGH - New Component)
**Current**: NO TESTS
**Problem**: Core routing logic has no unit tests
**Needs Testing**:
- Intent classification for all 7 intents
- AST pattern matching logic
- Edge cases (empty AST, malformed AST)
- Confidence scores
- Fallback behavior

**Recommendation**: Create test_gating_network.py with **15-20 tests**

#### 3. DateExpert Tests (HIGH - Zero Coverage)
**Current**: NO TESTS
**Problem**: One of 3 core experts, completely untested
**Needs Testing**:
- can_handle() detection (temporal keywords)
- execute() for current date, time, day of week
- Different temporal query formats
- Error handling

**Recommendation**: Create test_date_expert.py with **15-20 tests** (similar to MathExpert)

#### 4. GrammarExpert Tests (HIGH - Zero Coverage)
**Current**: NO TESTS
**Problem**: One of 3 core experts, completely untested
**Needs Testing**:
- can_handle() detection (grammar keywords)
- execute() AST analysis
- Subject/verb/object identification
- POS tagging accuracy
- Error handling

**Recommendation**: Create test_grammar_expert.py with **15-20 tests**

### ⚠️ MEDIUM Priority

#### 5. Safety Monitor (Needs Expansion)
**Current**: 5 tests, 73 lines
**Problem**: Critical for security, needs more coverage
**Needs Testing**:
- All input length edge cases
- AST complexity thresholds
- Malicious input handling
- Unicode edge cases
- Performance with large inputs

**Recommendation**: Expand to **15-20 tests**

#### 6. Front Door (Needs Expansion)
**Current**: 5 tests, 54 lines
**Problem**: Entry point for all queries, needs more coverage
**Needs Testing**:
- All supported languages (currently only tests 3)
- Translation quality validation
- Fallback behavior for unsupported languages
- Mixed-language input
- Special characters handling

**Recommendation**: Expand to **15-20 tests**

### 🔧 LOW Priority (But Should Have Tests)

- ExpertBase: Basic interface tests
- AST to Graph: Graph construction validation
- DataLoader: Data loading and preprocessing
- CLI: Command-line interface tests

---

## Test Input Size Analysis

### Current Test Input Characteristics

**Short sentences (mostly)**:
- "La hundo vidas la katon." (23 chars)
- "Mi amas la katon." (17 chars)
- "La bona kato manĝas." (20 chars)

**Medium sentences (some)**:
- "Malgrandaj hundoj vidas la grandan katon." (42 chars)

**Long sentences (rare)**:
- Integration tests use test corpus with longer sentences

### Recommendations for Input Size

#### ✅ Keep short inputs for:
- Unit tests (focused, fast)
- Specific feature testing
- Edge case validation

#### ⚠️ Need more medium/long inputs for:
- **Parser stress testing** - Complex sentences with:
  - Multiple clauses
  - Nested structures
  - Many affixes
  - Correlatives
  - Examples: "La granda malbela hundo, kiu kuris rapide, vidis la malgrandan belan katon."

- **Pipeline integration** - Realistic queries:
  - "Kiom estas la diferenco inter dek kvin kaj ses?"
  - "Kiam hodiaŭ estas, kaj kiu tago de la semajno?"
  - "Eksplik al mi la gramatikon de tiu frazo."

- **Expert system routing** - Complex multi-step queries

#### Recommendations:
1. **Unit tests**: Keep short (10-30 chars) for speed and focus
2. **Integration tests**: Add medium (30-80 chars) and long (80-150 chars) sentences
3. **Stress tests**: Create separate test_parser_stress.py with very complex sentences
4. **Corpus testing**: Expand test_corpus.json from 20 to 100+ diverse sentences

---

## Number of Tests - Is It Enough?

### Current Distribution

| Category | Current Tests | Recommended | Gap |
|----------|---------------|-------------|-----|
| Core Pipeline | 20 | 25-30 | +5-10 |
| Parser | 5 | 50-60 | **+45-55** |
| Front Door | 5 | 15-20 | +10-15 |
| Safety | 5 | 15-20 | +10-15 |
| Experts (Math) | 18 | 15-20 | ✅ Good |
| Experts (Date) | 0 | 15-20 | **+15-20** |
| Experts (Grammar) | 0 | 15-20 | **+15-20** |
| Gating Network | 0 | 15-20 | **+15-20** |
| Orchestrator | 15 | 15-20 | ✅ Good |
| Integration E2E | 15 | 20-25 | +5-10 |
| **TOTAL** | **147** | **250-300** | **+103-153** |

### Recommendation: **Double the test count** to 250-300 tests

---

## Stages That Need More Testing

### Priority Ranking (1 = Highest Priority)

#### Priority 1: CRITICAL (Zero Tolerance for Bugs)
1. **Parser** - Core of the system, currently under-tested
   - Current: 5 tests
   - Needed: 50+ tests
   - **This is the biggest gap**

2. **GatingNetwork** - Routes all queries, no tests
   - Current: 0 tests
   - Needed: 15-20 tests

3. **Safety Monitor** - Security critical
   - Current: 5 tests
   - Needed: 15-20 tests

#### Priority 2: HIGH (Core Functionality)
4. **DateExpert** - User-facing, no tests
   - Current: 0 tests
   - Needed: 15-20 tests

5. **GrammarExpert** - User-facing, no tests
   - Current: 0 tests
   - Needed: 15-20 tests

6. **Front Door** - Entry point, needs expansion
   - Current: 5 tests
   - Needed: 15-20 tests

#### Priority 3: MEDIUM (Important but Stable)
7. **Integration E2E** - End-to-end validation
   - Current: 15 tests
   - Needed: 20-25 tests

8. **Pipeline** - Well-tested but needs update
   - Current: 20 tests (7 failing)
   - Needed: 25-30 tests

---

## Specific Test Gaps by Feature

### Parser - Detailed Gap Analysis

**Missing Tests for Grammatical Features**:

| Feature | Example | Currently Tested? | Priority |
|---------|---------|-------------------|----------|
| Correlatives | kiu, kio, kie, kiam | ❌ | HIGH |
| Compound words | lernejo, hundejo | ❌ | HIGH |
| Multiple prefixes | mal-re-fari | ❌ | MEDIUM |
| Suffix chains | bel-ul-in-et-o | ❌ | HIGH |
| Future tense | kuros, manĝos | ❌ | MEDIUM |
| Conditional | venus, estus | ❌ | MEDIUM |
| Infinitive | iri, esti | ❌ | MEDIUM |
| Imperative | iru! venu! | ❌ | LOW |
| Accusative -n | hundon, katon | ✅ | - |
| Plural -j | hundoj, katoj | ✅ | - |
| Adjectives -a | bona, granda | ✅ | - |
| Adverbs -e | rapide, bone | ❌ | MEDIUM |
| Participles | -anta, -inta, -onta | ❌ | LOW |

**Recommendation**: Create test_parser_comprehensive.py with 45+ tests covering all features

---

## Recommendations Summary

### Immediate Actions (This Week)

1. **Fix failing tests** (7 tests in test_pipeline.py and test_integration_e2e.py)
   - Update to expect Orchestrator instead of IntentClassifier + Responder
   - Should take 1-2 hours

2. **Create missing expert tests**:
   - test_date_expert.py (15-20 tests)
   - test_grammar_expert.py (15-20 tests)
   - Estimated: 2-3 hours

3. **Create GatingNetwork tests**:
   - test_gating_network.py (15-20 tests)
   - Estimated: 1-2 hours

### Short-Term Actions (Next 2 Weeks)

4. **Expand Parser tests** (CRITICAL):
   - test_parser_comprehensive.py (45+ tests)
   - Cover all 16 grammatical rules
   - Cover all morphological combinations
   - Estimated: 4-6 hours

5. **Expand Safety tests**:
   - Add 10-15 more tests for edge cases
   - Estimated: 1-2 hours

6. **Expand Front Door tests**:
   - Add 10-15 more tests for all languages
   - Estimated: 1-2 hours

### Long-Term Actions (Next Month)

7. **Create stress tests**:
   - test_parser_stress.py - Very complex sentences
   - test_pipeline_stress.py - Performance testing
   - Estimated: 2-3 hours

8. **Expand test corpus**:
   - From 20 to 100+ sentences
   - Cover all grammatical features
   - Include edge cases
   - Estimated: 3-4 hours

---

## Test Quality Metrics

### Good Practices Currently Followed ✅
- Clear test names (test_<action>_<condition>_<expected_result>)
- setup_method for initialization
- Focused tests (one assertion per test)
- Good use of pytest features

### Areas for Improvement ⚠️
- **Input diversity**: Add more varied sentence structures
- **Edge case coverage**: More boundary value testing
- **Error path testing**: More tests for failure scenarios
- **Parametrized tests**: Use @pytest.mark.parametrize for multiple inputs
- **Coverage tracking**: Add coverage reporting to CI

---

## Conclusion

### Overall Test Health: 🟡 FAIR (Needs Improvement)

**Strengths**:
- Good infrastructure (pytest, fixtures, clear naming)
- High integration test pass rate (100%)
- Some components well-tested (MathExpert, Orchestrator, Logging)

**Critical Weaknesses**:
- **Parser severely under-tested** (5 tests for most critical component)
- **3 major components have ZERO tests** (DateExpert, GrammarExpert, GatingNetwork)
- **Limited input diversity** (mostly short sentences)
- **7 failing tests** from refactoring (needs immediate fix)

**Priority Actions**:
1. Fix 7 failing tests (immediate)
2. Create DateExpert + GrammarExpert + GatingNetwork tests (this week)
3. Expand Parser tests to 50+ (next 2 weeks)
4. Expand test corpus and add stress tests (next month)

**Target**: **250-300 total tests** with **comprehensive coverage** of all components
