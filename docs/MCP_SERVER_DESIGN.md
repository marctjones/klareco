# Klareco MCP Server Design

## Concept

Create an MCP server that exposes Klareco's AST-based retrieval and evaluation capabilities to Claude Code, enabling:
- Interactive testing of retrieval quality
- Automatic relevance evaluation
- Debugging of pipeline components
- Explainable results (AST trails, scores)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│ Claude Code (MCP Client)                           │
│   • Uses klareco_query tool                        │
│   • Uses klareco_evaluate_relevance tool           │
│   • Analyzes results, suggests improvements        │
└─────────────────────────────────────────────────────┘
                        ↓ MCP Protocol
┌─────────────────────────────────────────────────────┐
│ Klareco MCP Server (Python)                        │
│   • Exposes tools via MCP protocol                 │
│   • Wraps Klareco retrieval pipeline               │
│   • Returns structured JSON responses              │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ Klareco System (Backend)                           │
│   • Parser, Retriever, Summarizer                  │
│   • Kuzu database, trained models                  │
│   • AST processing pipeline                        │
└─────────────────────────────────────────────────────┘
```

## MCP Tools Exposed

### 1. klareco_query
**Purpose**: Run a query through the full pipeline and return results

```python
@mcp_tool
def klareco_query(
    query: str,
    top_k: int = 10,
    include_ast: bool = False,
    include_annotations: bool = False
) -> dict:
    """
    Query Klareco system and return relevant sentences.

    Args:
        query: Esperanto question (e.g., "Kiu fondis Esperanton?")
        top_k: Number of results to return
        include_ast: Include full AST in response
        include_annotations: Include model annotations

    Returns:
        {
            "query": "Kiu fondis Esperanton?",
            "results": [
                {
                    "rank": 1,
                    "text": "Zamenhof fondis Esperanton en 1887.",
                    "score": 0.95,
                    "source": "wikipedia_eo_zamenhof.txt",
                    "ast": {...},  # If include_ast=True
                    "annotations": {...}  # If include_annotations=True
                },
                ...
            ],
            "pipeline_stats": {
                "retrieval_time_ms": 45,
                "total_candidates": 150,
                "m1_filtered": 50,
                "reranked": 10
            }
        }
    """
```

### 2. klareco_evaluate_relevance
**Purpose**: Evaluate relevance of retrieved sentences to query

```python
@mcp_tool
def klareco_evaluate_relevance(
    query: str,
    sentences: list[str],
    gold_answer: str = None
) -> dict:
    """
    Evaluate how relevant sentences are to the query.

    This tool uses AST-based semantic analysis to determine:
    - Does the sentence answer the question?
    - What parts of the sentence are relevant?
    - Why did the retriever select this sentence?

    Args:
        query: Esperanto question
        sentences: List of retrieved sentences to evaluate
        gold_answer: Optional known correct answer for comparison

    Returns:
        {
            "query": "Kiu fondis Esperanton?",
            "query_type": "kiu (WHO question)",
            "evaluations": [
                {
                    "sentence": "Zamenhof fondis Esperanton en 1887.",
                    "relevance_score": 0.95,
                    "explanation": "Answers WHO question: subject 'Zamenhof' matches query intent",
                    "ast_overlap": {
                        "shared_roots": ["fond", "esperant"],
                        "query_coverage": 0.8
                    },
                    "answer_extraction": {
                        "extracted": "Zamenhof",
                        "confidence": 0.95,
                        "method": "deterministic (subjekto extraction for 'kiu')"
                    },
                    "gold_match": true  # If gold_answer provided
                },
                {
                    "sentence": "La lingvo havas regulan gramatikon.",
                    "relevance_score": 0.20,
                    "explanation": "Does NOT answer WHO question: no person as subject",
                    "ast_overlap": {
                        "shared_roots": ["esperant"],  # Implied context
                        "query_coverage": 0.2
                    },
                    "answer_extraction": {
                        "extracted": null,
                        "confidence": 0.0,
                        "method": "failed (subject is 'lingvo', not person)"
                    },
                    "gold_match": false
                }
            ],
            "summary": {
                "highly_relevant": 2,  # Score > 0.8
                "somewhat_relevant": 3,  # Score 0.5-0.8
                "not_relevant": 5,  # Score < 0.5
                "answers_found": ["Zamenhof"],
                "coverage": 0.85  # % of query intent covered by results
            }
        }
    """
```

### 3. klareco_parse
**Purpose**: Parse Esperanto sentence to AST for inspection

```python
@mcp_tool
def klareco_parse(
    text: str,
    include_annotations: bool = False
) -> dict:
    """
    Parse Esperanto text to AST.

    Args:
        text: Esperanto sentence
        include_annotations: Include model annotations

    Returns:
        {
            "text": "La hundo kuras rapide.",
            "ast": {
                "tipo": "frazo",
                "subjekto": {...},
                "verbo": {...},
                "aliaj": [...]
            },
            "parse_statistics": {
                "total_words": 4,
                "success_rate": 1.0,
                "unknown_roots": []
            },
            "annotations": {...}  # If include_annotations=True
        }
    """
```

### 4. klareco_explain
**Purpose**: Explain why a sentence was retrieved for a query

```python
@mcp_tool
def klareco_explain(
    query: str,
    sentence: str
) -> dict:
    """
    Explain why a sentence was retrieved for a query.

    Shows the full AST trail: how the retriever scored this sentence.

    Args:
        query: Esperanto question
        sentence: Retrieved sentence to explain

    Returns:
        {
            "query": "Kiu fondis Esperanton?",
            "sentence": "Zamenhof fondis Esperanton en 1887.",
            "explanation": {
                "stage_1_retrieval": {
                    "method": "root embeddings + structural similarity",
                    "score": 0.88,
                    "factors": {
                        "root_overlap": ["fond", "esperant"],
                        "query_root_embedding": [0.12, -0.45, ...],
                        "sentence_root_embedding": [0.15, -0.42, ...],
                        "cosine_similarity": 0.92
                    }
                },
                "stage_2_m1_filtering": {
                    "method": "selectional preferences",
                    "passed": true,
                    "score": 0.91,
                    "factors": {
                        "verb": "fondis",
                        "subject": "Zamenhof (person)",
                        "object": "Esperanton (language)",
                        "plausibility": 0.95  # Person can found language
                    }
                },
                "stage_3_reranking": {
                    "method": "learned reranker",
                    "score": 0.95,
                    "factors": {
                        "query_document_similarity": 0.93,
                        "answer_type_match": true  # Query wants person, sentence has person
                    }
                }
            },
            "final_score": 0.95,
            "rank": 1
        }
    """
```

### 5. klareco_summarize
**Purpose**: Summarize retrieved sentences using AST-based fusion

```python
@mcp_tool
def klareco_summarize(
    query: str,
    sentences: list[str],
    max_sentences: int = 3
) -> dict:
    """
    Summarize retrieved sentences using AST-based fusion.

    Args:
        query: Original query (for relevance context)
        sentences: Retrieved sentences to summarize
        max_sentences: Maximum sentences in summary

    Returns:
        {
            "query": "Kiu fondis Esperanton?",
            "input_sentences": 10,
            "summary": {
                "text": "Zamenhof fondis Esperanton en 1887 kaj estis pola kuracisto. La fundinto vivis en Bjalistoko.",
                "sentences": [
                    "Zamenhof fondis Esperanton en 1887 kaj estis pola kuracisto.",
                    "La fundinto vivis en Bjalistoko."
                ]
            },
            "operations": [
                {
                    "type": "fusion",
                    "input_asts": [
                        "Zamenhof fondis Esperanton en 1887.",
                        "Zamenhof estis pola kuracisto."
                    ],
                    "output_ast": "Zamenhof fondis Esperanton en 1887 kaj estis pola kuracisto.",
                    "reason": "Same subject, combined with 'kaj'"
                },
                {
                    "type": "selection",
                    "kept": "La fundinto vivis en Bjalistoko.",
                    "reason": "High importance score (0.85), adds biographical detail"
                },
                {
                    "type": "deduplication",
                    "removed": "La lingvo estas planlingvo.",
                    "reason": "Not directly relevant to WHO question"
                }
            ]
        }
    """
```

### 6. klareco_inspect_model
**Purpose**: Inspect model state and annotations

```python
@mcp_tool
def klareco_inspect_model(
    model_name: str,
    word: str = None
) -> dict:
    """
    Inspect model state and learned parameters.

    Args:
        model_name: "roots" | "m1" | "entity" | "reranker"
        word: Optional word to inspect (get embedding, predictions)

    Returns:
        {
            "model": "roots",
            "version": "v3.0",
            "stats": {
                "vocab_size": 9873,
                "embedding_dim": 64,
                "trainable_params": 632272,
                "checkpoint": "models/root_embeddings_v3/best_model.pt"
            },
            "word_analysis": {  # If word provided
                "word": "fundinto",
                "root": "fund",
                "root_embedding": [0.12, -0.45, ...],
                "similar_roots": [
                    {"root": "kre", "similarity": 0.85},
                    {"root": "establ", "similarity": 0.78}
                ]
            }
        }
    """
```

## Usage Example: Interactive Evaluation with Claude Code

### Scenario: User wants to test retrieval quality

```markdown
User: "Test the Klareco system with the query 'Kiu fondis Esperanton?'"

Claude Code:
1. Calls klareco_query("Kiu fondis Esperanton?", top_k=10)
2. Receives results
3. Calls klareco_evaluate_relevance(query, [returned sentences])
4. Analyzes evaluation scores
5. Reports to user:

   "Retrieved 10 sentences for query 'Kiu fondis Esperanton?':

   ✅ Highly relevant (score > 0.8): 3 sentences
      1. 'Zamenhof fondis Esperanton en 1887.' (0.95)
         → Correctly answers WHO question
      2. 'Ludoviko Lazaro Zamenhof estis pola kuracisto.' (0.90)
         → Provides biographical detail about answer

   ⚠️ Somewhat relevant (score 0.5-0.8): 2 sentences
      3. 'La fundinto vivis en Bjalistoko.' (0.75)
         → Relevant but indirect (uses 'fundinto' not 'Zamenhof')

   ❌ Not relevant (score < 0.5): 5 sentences
      8. 'La lingvo havas regulan gramatikon.' (0.20)
         → Does not answer WHO question

   Coverage: 85% - Good! The top results directly answer the question.

   Would you like me to:
   - Explain why specific sentences scored high/low?
   - Inspect the AST overlap between query and results?
   - Test with more queries?"
```

### Scenario: Debugging low relevance

```markdown
User: "Why did the sentence 'La lingvo estas planlingvo' get retrieved for 'Kiu fondis Esperanton?'?"

Claude Code:
1. Calls klareco_explain("Kiu fondis Esperanton?", "La lingvo estas planlingvo.")
2. Receives explanation with AST trail
3. Reports:

   "Explanation for why 'La lingvo estas planlingvo.' was retrieved:

   Stage 1 (Root Embeddings): Score 0.60
   - Shared roots: ['esperant'] (implied from context)
   - Root overlap is moderate, but doesn't match query intent (WHO vs WHAT)

   Stage 2 (M1 Filtering): Score 0.40
   - Subject is 'lingvo' (non-person), query asks for person (kiu)
   - Selectional preference mismatch reduces score

   Stage 3 (Reranking): Score 0.20
   - Reranker detects answer type mismatch
   - Final low score → ranked 8th out of 10

   **Analysis**: This sentence was retrieved because it mentions 'Esperanto'
   (shared context), but correctly scored LOW because it doesn't answer
   WHO question. The pipeline is working correctly - it's retrieving
   context but ranking it lower than direct answers.

   Recommendation: If you want to EXCLUDE this type of sentence entirely,
   could add a stricter filter in M1 stage: for 'kiu' queries, only
   keep sentences with person subjects."
```

## Implementation Plan

### Phase 1: Core MCP Server (Week 1)
1. Create `klareco/mcp_server.py` using `mcp` Python library
2. Implement `klareco_query` tool (wrap existing retriever)
3. Implement `klareco_parse` tool (wrap existing parser)
4. Test with Claude Code locally

### Phase 2: Evaluation Tools (Week 2)
5. Implement `klareco_evaluate_relevance` tool
   - AST-based relevance scoring
   - Answer extraction (deterministic for kiu/kio/kiam/etc)
   - Gold answer comparison
6. Test evaluation on sample queries

### Phase 3: Explainability (Week 3)
7. Implement `klareco_explain` tool
   - AST trail tracking through pipeline
   - Score breakdown per stage
8. Implement `klareco_inspect_model` tool
   - Model state inspection
   - Word embedding lookup

### Phase 4: Summarization (Week 4)
9. Implement `klareco_summarize` tool
   - AST-based fusion
   - Show operations performed
10. End-to-end testing with Claude Code

## Technical Details

### MCP Server Setup

```python
# klareco/mcp_server.py
from mcp import MCPServer, Tool
from klareco.rag.retriever import ASTRetriever
from klareco.parser import Parser
from klareco.evaluation.relevance import RelevanceEvaluator

server = MCPServer(name="klareco", version="1.0.0")

@server.tool()
def klareco_query(
    query: str,
    top_k: int = 10,
    include_ast: bool = False
) -> dict:
    """Query Klareco system and return relevant sentences."""
    retriever = ASTRetriever.load("models/retriever_v3")
    results = retriever.retrieve(query, top_k=top_k)

    return {
        "query": query,
        "results": [
            {
                "rank": i + 1,
                "text": r.text,
                "score": r.score,
                "source": r.source,
                "ast": r.ast if include_ast else None
            }
            for i, r in enumerate(results)
        ]
    }

# ... more tools ...

if __name__ == "__main__":
    server.run()
```

### Claude Code Configuration

```json
// .mcp.json (user's Claude Code config)
{
  "mcpServers": {
    "klareco": {
      "command": "python",
      "args": ["-m", "klareco.mcp_server"],
      "env": {
        "KLARECO_DB_PATH": "/home/marc/Projects/klareco/data/indexes/kuzu_v2.1",
        "KLARECO_MODELS_PATH": "/home/marc/Projects/klareco/models"
      }
    }
  }
}
```

## Benefits

1. **Interactive Testing**: Claude Code can test queries and analyze results in real-time
2. **Automatic Evaluation**: No manual inspection of results - Claude analyzes relevance automatically
3. **Explainability**: AST trails show exactly why sentences were retrieved
4. **Debugging**: Inspect model state, annotations, scores at any pipeline stage
5. **Continuous Improvement**: Claude can suggest improvements based on evaluation results

## Example Evaluation Session

```markdown
User: "Run comprehensive evaluation on Klareco retrieval"

Claude Code:
[Calls klareco_query for 20 test queries]
[Calls klareco_evaluate_relevance for each result set]
[Analyzes patterns]

"Evaluation Results (20 queries):

✅ Strong Performance:
- WHO questions (kiu): 90% accuracy, top result correct in 18/20 cases
- WHAT questions (kio): 85% accuracy, top-3 contains answer in 17/20 cases

⚠️ Needs Improvement:
- WHY questions (kial): 60% accuracy, struggles with causal reasoning
  → Recommendation: Add causal relation detection to M1 model

❌ Weak Performance:
- HOW questions (kiel): 40% accuracy, process descriptions not well-ranked
  → Recommendation: Need deterministic rules for procedural content

Detailed Issues Found:
1. Query: 'Kial Esperanto estas facila?'
   - Top result: 'Esperanto havas regulan gramatikon.' (0.75)
   - Issue: Doesn't explain WHY, just states a feature
   - Fix: Add 'ĉar' (because) detection in AST to boost causal sentences

2. Query: 'Kiel funkcias la akuzativo?'
   - Top result: 'La akuzativo uzas -n finaĵon.' (0.70)
   - Issue: States fact, doesn't explain process
   - Fix: Boost sentences with imperatives, examples, step-by-step structure

Overall Assessment: System works well for factual questions (kiu/kio/kiam/kie),
needs improvement for reasoning questions (kial/kiel). These could be addressed
with additional deterministic AST rules, not necessarily learned models."
```

## Next Steps

1. ⬜ Implement core MCP server with `klareco_query` and `klareco_parse`
2. ⬜ Test with Claude Code locally
3. ⬜ Add `klareco_evaluate_relevance` tool
4. ⬜ Create evaluation dataset (20-50 test queries with gold answers)
5. ⬜ Run comprehensive evaluation session with Claude Code
6. ⬜ Iterate on pipeline based on evaluation results
