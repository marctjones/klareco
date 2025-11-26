# Phase 4 Design - Orchestrator & Expert System

**Target Duration:** 4 weeks
**Status:** 🎯 Ready to Begin
**Prerequisites:** ✅ Phase 3 Complete (Baseline RAG)

---

## Overview

Phase 4 transforms Klareco from a parsing system into an **agentic AI** by implementing:
1. **Orchestrator**: Routes queries to specialized experts
2. **Gating Network**: Intent classification for routing
3. **First Experts**: Factoid QA and symbolic Tools
4. **Execution Loop**: Multi-step reasoning capability

This builds on Phase 3's baseline RAG to create a **production-ready question-answering system**.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       USER QUERY                              │
│              "Kiam naskiĝis Zamenhof?"                       │
│              (When was Zamenhof born?)                        │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│                   FRONT DOOR (Phase 2)                        │
│   - Language identification                                   │
│   - Translation to Esperanto                                  │
│   - Parse to AST                                              │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│               ORCHESTRATOR (Phase 4 - NEW)                    │
│                                                               │
│   ┌─────────────────────────────────────────────────┐       │
│   │         Gating Network                          │       │
│   │  (Classify intent from AST structure)           │       │
│   └─────────────────────────────────────────────────┘       │
│                        │                                      │
│   ┌────────────────────┼────────────────────┐                │
│   │                    │                    │                │
│   ▼                    ▼                    ▼                │
│ factoid_          calculation          command              │
│ question          _request             _intent              │
└───┬────────────────────┬─────────────────────┬──────────────┘
    │                    │                     │
    │                    │                     │
┌───▼─────────┐   ┌─────▼──────┐   ┌─────────▼──────┐
│  Factoid_QA │   │  Math_Tool │   │  (Future)      │
│  Expert     │   │  Expert    │   │  Experts       │
│             │   │            │   │                │
│  Uses:      │   │  Uses:     │   │                │
│  - RAG      │   │  - sympy   │   │                │
│  - Baseline │   │  - Parser  │   │                │
│  - FAISS    │   │            │   │                │
└─────────────┘   └────────────┘   └────────────────┘
       │                 │                 │
       └─────────────────┴─────────────────┘
                        │
                        ▼
                   RESPONSE
```

---

## Phase 4 Components

### 1. Orchestrator Core

**File**: `klareco/orchestrator.py`

**Responsibilities:**
- Receive parsed AST from Front Door
- Invoke Gating Network for intent classification
- Route to appropriate Expert
- Manage Execution Loop
- Return structured response

**Key Methods:**
```python
class Orchestrator:
    def route(self, ast: dict) -> dict:
        """
        Route query to appropriate expert.

        Args:
            ast: Parsed query AST

        Returns:
            Response from expert
        """
        # 1. Classify intent
        intent = self.gating_network.classify(ast)

        # 2. Select expert
        expert = self.experts[intent]

        # 3. Execute
        response = expert.execute(ast)

        return response

    def execute_loop(self, ast: dict, goal: str) -> list:
        """
        Multi-step execution until goal achieved.

        Args:
            ast: Initial query AST
            goal: Completion criterion

        Returns:
            List of execution steps
        """
        steps = []
        current_ast = ast

        while not self.goal_achieved(goal, steps):
            response = self.route(current_ast)
            steps.append(response)

            # Generate next action
            current_ast = self.plan_next(response, goal)

            if len(steps) > MAX_STEPS:
                break

        return steps
```

---

### 2. Gating Network (Intent Classifier)

**File**: `klareco/gating_network.py`

**Level 1: Symbolic Rules** (Phase 4 - build this first)
Classify intent from AST structure:

```python
def classify_intent_symbolic(ast: dict) -> str:
    """
    Rule-based intent classification from AST.

    Uses grammatical structure to infer intent.
    """
    # Question words (kiu, kio, kiam, kie, kial, kiel, kiom)
    if has_question_word(ast, ["kiam", "kie"]):
        return "factoid_question"

    # Imperative mood (command)
    if ast.get("verbo", {}).get("modo") == "imperativo":
        return "command_intent"

    # Mathematical expressions
    if has_numbers(ast) and has_operators(ast):
        return "calculation_request"

    # Default
    return "general_query"
```

**Examples:**
- "Kiam naskiĝis Zamenhof?" → `factoid_question` (has "kiam" = when)
- "Kalkulu 15 plus 27" → `calculation_request` (numbers + operator)
- "Difinu la vorton 'lingvo'" → `dictionary_lookup` (command "difinu")

**Level 2: Neural Classifier** (Phase 5 - future)
- Train on execution traces
- Learns from usage patterns
- Handles ambiguous cases

---

### 3. Expert Interface

**File**: `klareco/experts/base.py`

All experts implement this interface:

```python
class Expert(ABC):
    """Base class for all experts."""

    @abstractmethod
    def can_handle(self, ast: dict) -> bool:
        """Check if this expert can handle the query."""
        pass

    @abstractmethod
    def execute(self, ast: dict) -> dict:
        """Execute query and return response."""
        pass

    @abstractmethod
    def estimate_confidence(self, ast: dict) -> float:
        """Estimate confidence in handling this query."""
        pass
```

---

### 4. Factoid_QA_Expert

**File**: `klareco/experts/factoid_qa.py`

Uses baseline RAG from Phase 3 to answer factual questions.

**Algorithm:**
```
1. Extract query intent from AST
2. Deparse AST to query text
3. Search RAG index (top-K retrieval)
4. Extract answer from context
5. Return structured response
```

**Implementation:**
```python
class FactoidQAExpert(Expert):
    def __init__(self, rag_index_path: str):
        self.rag = load_baseline_rag(rag_index_path)

    def execute(self, ast: dict) -> dict:
        # 1. Deparse query
        query_text = deparse(ast)

        # 2. Search RAG
        results = self.rag.search(query_text, k=5)

        # 3. Extract answer
        answer = self.extract_answer(ast, results)

        return {
            "answer": answer,
            "confidence": self.compute_confidence(results),
            "sources": [r["text"] for r in results]
        }

    def extract_answer(self, query_ast: dict, contexts: list) -> str:
        """
        Extract answer from retrieved contexts.

        For Phase 4, use simple heuristics:
        - Look for entities matching question type
        - Return first relevant sentence

        For Phase 5, use neural answer extractor.
        """
        question_word = get_question_word(query_ast)

        if question_word == "kiam":  # When
            return extract_temporal_entity(contexts)
        elif question_word == "kie":  # Where
            return extract_location_entity(contexts)
        elif question_word == "kiu":  # Who
            return extract_person_entity(contexts)
        else:
            return contexts[0]["text"]  # Return top result
```

---

### 5. Tool Experts (Symbolic)

**Math_Tool_Expert** (`klareco/experts/math_tool.py`):
```python
class MathToolExpert(Expert):
    def execute(self, ast: dict) -> dict:
        # Extract mathematical expression from AST
        expr = extract_math_expression(ast)

        # Evaluate using sympy
        result = sympy.sympify(expr).evalf()

        return {
            "result": result,
            "expression": expr,
            "confidence": 1.0  # Symbolic = certain
        }
```

**Date_Tool_Expert** (`klareco/experts/date_tool.py`):
```python
class DateToolExpert(Expert):
    def execute(self, ast: dict) -> dict:
        # Extract date operation from AST
        operation = extract_date_operation(ast)

        # Execute using datetime
        result = execute_date_calculation(operation)

        return {
            "result": result,
            "operation": operation,
            "confidence": 1.0
        }
```

**Dictionary_Tool_Expert** (`klareco/experts/dictionary_tool.py`):
```python
class DictionaryToolExpert(Expert):
    def __init__(self, vocab_path: str):
        self.vocab = load_vocabulary(vocab_path)

    def execute(self, ast: dict) -> dict:
        # Extract word from AST
        word = extract_target_word(ast)

        # Look up in vocabulary
        entry = self.vocab.get(word)

        if entry:
            return {
                "definition": entry["definition"],
                "root": entry["root"],
                "part_of_speech": entry["pos"],
                "confidence": 1.0
            }
        else:
            return {
                "error": "Word not found",
                "confidence": 0.0
            }
```

---

## Implementation Plan

### Week 1: Orchestrator Foundation

**Day 1-2: Core Architecture**
- [ ] Create `klareco/orchestrator.py`
- [ ] Implement `Orchestrator.route()`
- [ ] Define Expert interface in `klareco/experts/base.py`
- [ ] Set up expert registry pattern

**Day 3-4: Gating Network (Symbolic)**
- [ ] Create `klareco/gating_network.py`
- [ ] Implement rule-based intent classification
- [ ] Add support for question words (kiu, kio, kiam, kie, kial, kiel, kiom)
- [ ] Handle imperative mood detection
- [ ] Test on sample queries

**Day 5-7: Integration & Testing**
- [ ] Integrate with Front Door (Phase 2)
- [ ] Create end-to-end pipeline test
- [ ] Add execution trace logging
- [ ] Document intent categories

### Week 2: Factoid_QA_Expert

**Day 1-2: RAG Integration**
- [ ] Create `klareco/experts/factoid_qa.py`
- [ ] Load baseline RAG from Phase 3
- [ ] Implement query → RAG search
- [ ] Return top-K contexts

**Day 3-4: Answer Extraction**
- [ ] Implement entity extraction heuristics
- [ ] Handle temporal questions (kiam = when)
- [ ] Handle location questions (kie = where)
- [ ] Handle person questions (kiu = who)

**Day 5-7: Testing & Refinement**
- [ ] Create test suite with diverse questions
- [ ] Measure answer accuracy
- [ ] Tune retrieval parameters (K, similarity threshold)
- [ ] Add confidence scoring

### Week 3: Tool Experts

**Day 1-2: Math_Tool_Expert**
- [ ] Create `klareco/experts/math_tool.py`
- [ ] Parse mathematical expressions from AST
- [ ] Integrate sympy for evaluation
- [ ] Handle arithmetic, algebra, basic calculus

**Day 3-4: Date_Tool_Expert & Dictionary_Tool_Expert**
- [ ] Create `klareco/experts/date_tool.py`
- [ ] Implement date arithmetic (datetime library)
- [ ] Create `klareco/experts/dictionary_tool.py`
- [ ] Load vocabulary from Phase 2

**Day 5-7: Tool Testing**
- [ ] Test all tool experts
- [ ] Create example queries for each
- [ ] Document tool capabilities

### Week 4: Execution Loop & Polish

**Day 1-3: Execution Loop**
- [ ] Implement `Orchestrator.execute_loop()`
- [ ] Add goal completion checking
- [ ] Handle multi-step queries
- [ ] Add max steps safety limit

**Day 4-5: Integration Testing**
- [ ] End-to-end system test
- [ ] Test query routing accuracy
- [ ] Test expert execution
- [ ] Measure latency

**Day 6-7: Documentation & Demo**
- [ ] Write Phase 4 completion report
- [ ] Create demo script
- [ ] Document all intent categories
- [ ] Prepare for Phase 5

---

## Testing Strategy

### Unit Tests

```python
# Test Gating Network
def test_gating_network_question_classification():
    ast = parse("Kiam naskiĝis Zamenhof?")
    intent = classify_intent_symbolic(ast)
    assert intent == "factoid_question"

# Test Expert Selection
def test_orchestrator_routes_to_factoid_qa():
    orchestrator = Orchestrator()
    ast = parse("Kie loĝas la prezidanto?")
    expert = orchestrator.select_expert(ast)
    assert isinstance(expert, FactoidQAExpert)

# Test Math Tool
def test_math_tool_evaluates_expression():
    expert = MathToolExpert()
    ast = parse("Kalkulu 15 plus 27")
    result = expert.execute(ast)
    assert result["result"] == 42
```

### Integration Tests

```python
def test_end_to_end_factoid_question():
    # Full pipeline: query → answer
    query = "Kiam naskiĝis Zamenhof?"
    response = pipeline.run(query)

    assert "1859" in response["answer"]
    assert response["confidence"] > 0.7

def test_end_to_end_calculation():
    query = "Kalkulu la kvadraton de 8"
    response = pipeline.run(query)

    assert response["result"] == 64
    assert response["confidence"] == 1.0
```

### Benchmark Queries

Create `data/phase4_test_queries.json`:
```json
[
  {
    "query": "Kiam naskiĝis Zamenhof?",
    "expected_intent": "factoid_question",
    "expected_expert": "Factoid_QA_Expert",
    "expected_answer_contains": "1859"
  },
  {
    "query": "Kalkulu 15 plus 27",
    "expected_intent": "calculation_request",
    "expected_expert": "Math_Tool_Expert",
    "expected_result": 42
  },
  {
    "query": "Difinu la vorton 'lingvo'",
    "expected_intent": "dictionary_lookup",
    "expected_expert": "Dictionary_Tool_Expert"
  }
]
```

---

## Success Metrics

### Minimum Viable Product (MVP)
- ✅ Orchestrator routes 90%+ of test queries correctly
- ✅ Factoid_QA_Expert answers 70%+ of factual questions
- ✅ Tool Experts handle 100% of symbolic queries (math, date, dictionary)
- ✅ End-to-end latency <2 seconds per query

### Stretch Goals
- 🎯 95%+ routing accuracy on diverse queries
- 🎯 80%+ answer accuracy for factoid questions
- 🎯 Multi-step execution loop handles 3+ step queries
- 🎯 <500ms latency for tool experts

---

## Dependencies

### Python Packages
```bash
# Already installed from Phase 3
- torch
- transformers
- faiss-cpu
- sentence-transformers

# Phase 4 additions
pip install sympy  # Math tool
```

### Data Requirements
- ✅ Baseline RAG index (Phase 3)
- ✅ Parsed AST corpus (Phase 3)
- ✅ Parser vocabulary (Phase 2)

---

## Expert Categories (Phase 4)

| Intent | Expert | Input | Output | Priority |
|--------|--------|-------|--------|----------|
| `factoid_question` | Factoid_QA_Expert | Question AST | Answer + sources | **High** |
| `calculation_request` | Math_Tool_Expert | Math expression | Numeric result | **High** |
| `dictionary_lookup` | Dictionary_Tool_Expert | Word | Definition | **Medium** |
| `date_calculation` | Date_Tool_Expert | Date query | Date result | **Medium** |
| `general_query` | (Fallback - Phase 5) | Any AST | Generic response | **Low** |

---

## Future Expert Ideas (Phase 5+)

- **Summarize_Expert**: Summarize documents using AST structure
- **Grammar_Checker_Expert**: Validate Esperanto grammar
- **Translation_Expert**: Translate between languages
- **Code_Interpreter_Expert**: Execute code snippets
- **Memory_Read/Write_Expert**: Access long-term memory
- **Web_Search_Expert**: Search web for current info
- **Logic_Solver_Expert**: Formal logic reasoning (Prolog)

---

## Execution Trace Format

All operations logged as JSON for learning loop (Phase 9):

```json
{
  "trace_id": "uuid",
  "timestamp": "2025-11-11T21:00:00Z",
  "query": {
    "text": "Kiam naskiĝis Zamenhof?",
    "ast": {...}
  },
  "orchestrator": {
    "gating_network": {
      "intent": "factoid_question",
      "confidence": 0.95
    },
    "selected_expert": "Factoid_QA_Expert"
  },
  "expert_execution": {
    "expert_name": "Factoid_QA_Expert",
    "rag_search": {
      "query": "Kiam naskiĝis Zamenhof?",
      "top_k": 5,
      "results": [...]
    },
    "answer_extraction": {
      "answer": "Zamenhof naskiĝis en 1859.",
      "confidence": 0.85,
      "sources": [...]
    }
  },
  "response": {
    "answer": "Zamenhof naskiĝis en 1859.",
    "confidence": 0.85,
    "latency_ms": 145
  }
}
```

---

## Integration with Existing System

Phase 4 sits between **Phase 2 (Parser)** and **Phase 5 (Neural Components)**:

```
Phase 2 Output (AST)
  → Phase 4 (Orchestrator routes to Expert)
  → Phase 4 Expert executes using Phase 3 (RAG) or symbolic tools
  → Phase 4 returns structured response
```

**Modified Pipeline** (`klareco/pipeline.py`):
```python
class KlarecoPipeline:
    def __init__(self):
        self.front_door = FrontDoor()
        self.parser = Parser()
        self.orchestrator = Orchestrator()  # NEW in Phase 4

    def run(self, query: str) -> dict:
        # Phase 2: Parse
        processed = self.front_door.process(query)
        ast = self.parser.parse(processed)

        # Phase 4: Route & Execute (NEW)
        response = self.orchestrator.route(ast)

        return response
```

---

## Example Queries & Expected Flow

### Example 1: Factoid Question
**Query**: "Kiam naskiĝis Zamenhof?"

**Flow**:
1. Front Door → AST with "kiam" (when)
2. Gating Network → `factoid_question`
3. Orchestrator → Routes to Factoid_QA_Expert
4. Expert:
   - Deparse: "Kiam naskiĝis Zamenhof?"
   - RAG search → Top 5 contexts
   - Extract answer → "1859"
5. Response: {"answer": "Zamenhof naskiĝis en 1859.", "confidence": 0.85}

### Example 2: Math Calculation
**Query**: "Kalkulu 15 plus 27"

**Flow**:
1. Front Door → AST with numbers [15, 27] and operator "plus"
2. Gating Network → `calculation_request`
3. Orchestrator → Routes to Math_Tool_Expert
4. Expert:
   - Extract expression: "15 + 27"
   - Evaluate with sympy: 42
5. Response: {"result": 42, "confidence": 1.0}

### Example 3: Dictionary Lookup
**Query**: "Difinu la vorton 'lingvo'"

**Flow**:
1. Front Door → AST with imperative verb "difinu"
2. Gating Network → `dictionary_lookup`
3. Orchestrator → Routes to Dictionary_Tool_Expert
4. Expert:
   - Extract word: "lingvo"
   - Look up in vocabulary
   - Return definition
5. Response: {"definition": "sistemo de komunikado", "confidence": 1.0}

---

## Risks & Mitigation

### Risk 1: Poor Routing Accuracy
**Problem**: Gating Network misclassifies intents
**Mitigation**:
- Start with simple symbolic rules
- Create comprehensive test suite
- Log all routing decisions for analysis
- Plan neural classifier for Phase 5

### Risk 2: Low Answer Quality (Factoid QA)
**Problem**: Retrieved contexts don't contain answer
**Mitigation**:
- Tune RAG retrieval (K, similarity threshold)
- Improve answer extraction heuristics
- Fall back to returning top context if no entity found
- Plan neural answer extractor for Phase 5

### Risk 3: Execution Loop Doesn't Converge
**Problem**: Multi-step queries loop infinitely
**Mitigation**:
- Add MAX_STEPS safety limit (e.g., 10)
- Implement clear goal completion criteria
- Log all execution steps for debugging

---

## Code Structure

```
klareco/
├── orchestrator.py          # Main orchestrator
├── gating_network.py        # Intent classifier
├── experts/
│   ├── __init__.py
│   ├── base.py              # Expert interface
│   ├── factoid_qa.py        # Factoid QA expert
│   ├── math_tool.py         # Math calculations
│   ├── date_tool.py         # Date operations
│   └── dictionary_tool.py   # Vocabulary lookup
└── utils/
    ├── entity_extraction.py  # Extract entities from text
    └── goal_checking.py      # Check execution loop goals

tests/phase4/
├── test_orchestrator.py
├── test_gating_network.py
├── test_experts/
│   ├── test_factoid_qa.py
│   ├── test_math_tool.py
│   ├── test_date_tool.py
│   └── test_dictionary_tool.py
└── test_integration.py

data/
├── phase4_test_queries.json  # Benchmark queries
└── faiss_production/          # Production RAG (Phase 3)
```

---

## Next Steps After Phase 4

Once Phase 4 is complete, you'll have:
- ✅ Working question-answering system
- ✅ Expert-based architecture
- ✅ Symbolic tool integration
- ✅ Execution loop foundation

**Phase 5** will add:
- Neural answer extraction (improve QA accuracy)
- Summarize_Expert (document summarization)
- Multi-step Blueprint generation
- Neural Gating Network (learn from traces)

---

## Conclusion

Phase 4 transforms Klareco from a **parser** into an **intelligent agent** by:
1. Adding routing logic (Orchestrator + Gating Network)
2. Creating modular experts (QA + Tools)
3. Enabling multi-step reasoning (Execution Loop)

This sets the foundation for future neural components while delivering immediate value through symbolic tools and RAG-based question answering.

---

**Status**: 🎯 Ready to begin
**Prerequisites**: ✅ Phase 3 complete (Baseline RAG built)
**Target Completion**: 2025-12-09 (4 weeks from now)
**Next Action**: Implement Orchestrator core & Gating Network

---

**Last Updated**: 2025-11-11
**Author**: Phase 4 Design Document
**See Also**: `DESIGN.md`, `PHASE3_COMPLETION_REPORT.md`
