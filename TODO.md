# Klareco Development Roadmap

This document tracks the current development priorities for the Klareco neuro-symbolic AI system. For the complete 9-phase design, see `DESIGN.md`. For architectural overview, see `CLAUDE.md`.

## Current Phase: Phase 2 - Core Foundation (Final Tasks)

### 🔥 Critical Priority: Parser Perfection

**Goal**: Achieve 100% accuracy on Esperanto grammar before moving to Phase 3.

- [ ] **Expand KNOWN_ROOTS vocabulary**
  - Add all common Esperanto roots (500+ most frequent)
  - Consider loading from external vocabulary file
  - Reference: `scripts/build_morpheme_vocab.py`

- [ ] **Handle participles** (Rule 13)
  - Active/passive: -ant-, -int-, -ont- / -at-, -it-, -ot-
  - Example: "kuranta" (running), "vidita" (seen)

- [ ] **Handle correlatives**
  - kio, kiu, kie, kiam, kiel, kiom, kia, kies, nenio, etc.
  - These are closed-class words, can be hardcoded

- [ ] **Handle prepositions**
  - al, de, en, sur, sub, kun, sen, pro, por, etc.
  - Rule 9: Prepositions govern nominative case

- [ ] **Compound word parsing** (Rule 15)
  - vaporŝipo → vapor + ŝip + o
  - Need heuristics or dictionary for segmentation

- [ ] **Elision handling** (Rule 16)
  - Poetic forms like "bird'" for "birdo"
  - Low priority but document for completeness

- [ ] **Error recovery and reporting**
  - Better error messages when unknown roots found
  - Suggest possible corrections based on edit distance

- [ ] **Parser test coverage**
  - Aim for 100% line coverage in `parser.py`
  - Test all 16 rules with edge cases
  - Test complex sentences with multiple modifiers

### 🛠️ Infrastructure Tasks

- [x] **Create `watch.sh` script**
  - Uses: `tail -f klareco.log` for real-time structured logging
  - Monitors Python logging output during long-running tests
  - Works perfectly with tqdm progress bars (no pipe issues)

- [ ] **Improve test corpus diversity**
  - Add sentences testing each of the 16 rules explicitly
  - Add sentences with complex compound words
  - Add sentences with all verb tenses/moods
  - Add sentences with all participle forms

- [ ] **Documentation**
  - Add inline code comments explaining complex parser logic
  - Document the morpheme stripping order (right-to-left)
  - Document the AST structure fully

## Phase 3: Knowledge Base & GNN Encoder

**Start Condition**: Parser handles 100% of test corpus correctly.

### Corpus Preparation
- [ ] **Corpus acquisition and cleaning**
  - Esperanto Wikipedia dump
  - Tekstaro texts
  - Other open-source Esperanto corpora
  - Target: 10M+ tokens

- [ ] **Corpus validation**
  - Ensure parser can handle all sentences
  - Fix parser bugs discovered during corpus processing
  - Track parsing success rate

- [ ] **AST corpus generation**
  - Parse entire corpus into AST format
  - Store as structured JSON or pickle
  - This becomes training data for GNN

### GNN Encoder (Context Expert)
- [ ] **Architecture design**
  - Graph Attention Network (GAT) or Graph Convolutional Network (GCN)
  - Input: AST graph structure
  - Output: Semantic embedding vector

- [ ] **Training data preparation**
  - Convert ASTs to graph format (nodes, edges)
  - Define node features (morpheme type, POS, etc.)
  - Define edge types (subject-of, object-of, modifies, etc.)

- [ ] **PoC implementation**
  - Use PyTorch Geometric
  - Train on small subset (10K sentences)
  - Validate embeddings capture semantic similarity

- [ ] **RAG integration**
  - Index corpus with GNN embeddings
  - Implement similarity search (FAISS or similar)
  - Test retrieval quality

## Phase 4: Agentic Core (Orchestrator & Tools)

**Start Condition**: GNN encoder trained and RAG system functional.

### Orchestrator & Execution Loop
- [ ] **Design Orchestrator architecture**
  - Input: AST
  - Output: Expert selection + routing
  - Gating network for expert selection

- [ ] **Implement Execution Loop**
  - `while not goal_achieved:` logic
  - Step tracking in ExecutionTrace
  - Max iteration limits for safety

- [ ] **Experts Manifest**
  - JSON file defining all available experts
  - Schema: name, description, input_format, output_format
  - Used by Orchestrator for routing decisions

### Symbolic Tool Experts
- [ ] **Math Expert**
  - Arithmetic, algebra, calculus
  - Parse AST to extract mathematical expressions
  - Use SymPy for symbolic math

- [ ] **Date/Time Expert**
  - Handle temporal queries
  - Parse AST for date/time references
  - Use Python datetime + dateutil

- [ ] **Dictionary Expert**
  - Esperanto-English dictionary lookups
  - Root word definitions
  - Etymology information

- [ ] **Grammar Expert**
  - Explain AST structure to user
  - Grammar rule validation
  - Already partially implemented in current Responder

### Factoid QA Expert (First Neural Decoder)
- [ ] **Dataset creation**
  - Generate Q&A pairs from corpus
  - Format: Esperanto question → Esperanto answer
  - Include AST representations

- [ ] **PoC: Fine-tune Mistral 7B**
  - Use LoRA for efficiency
  - Train to generate AST structure (not just text)
  - Integrate with Writer Loop

- [ ] **Writer Loop implementation**
  - Takes expert output + AST structure
  - Constructs response AST programmatically
  - Validates before deparsing

## Phase 5: Summarization & Multi-Step Planning

**Start Condition**: Factoid QA Expert working, Orchestrator routing successfully.

- [ ] **Summarize Expert dataset**
- [ ] **Summarize Expert training**
- [ ] **Multi-step Blueprint generation**
- [ ] **Neural Clusterer for complex tasks**

## Phase 6: Memory System

**Start Condition**: Multiple experts working, need to maintain context.

- [ ] **Short-Term Memory (STM) design**
  - Store recent interaction ASTs
  - FIFO or recency-based eviction

- [ ] **Long-Term Memory (LTM) database**
  - PostgreSQL or Neo4j
  - Schema for storing consolidated facts as ASTs

- [ ] **Memory Read/Write Tools**
  - STM/LTM query interfaces
  - Integration with Orchestrator

- [ ] **Consolidate Expert**
  - Scheduled task to summarize STM → LTM
  - Detect important facts for persistence

## Phase 7: Goals & Values

**Start Condition**: Memory system functional, need alignment framework.

- [ ] **Goals Manifest design**
  - Priority, completion criteria
  - Stored as ASTs in LTM

- [ ] **Values Manifest design**
  - Name, weight, conflict resolution
  - Stored as ASTs in LTM

- [ ] **Goal/Value Sync Tool**
  - Bidirectional Esperanto ↔ Native language

- [ ] **Pre-Query Goal Check**
  - Orchestrator evaluates goals before processing

- [ ] **Post-Retrieval Reflection**
  - Generate "Weighting Instructions" from Values
  - Writer Loop incorporates during AST construction

## Phase 8: External Tool Integration

**Start Condition**: System has strong internal reasoning, ready for external actions.

- [ ] **Sandboxed execution environment**
  - Docker or similar isolation
  - Security policies

- [ ] **Web Search Tool**
  - API integration (SerpAPI or similar)
  - Parse results into AST format

- [ ] **Code Interpreter Tool**
  - Python execution in sandbox
  - Parse code execution results

- [ ] **Formal Logic Tool**
  - Prolog integration
  - Logical inference on AST-based facts

## Phase 9: Learning Loop

**Start Condition**: Full system operational, ready for self-improvement.

- [ ] **Log Database**
  - Store all ExecutionTraces
  - Indexing for efficient querying

- [ ] **Emergent Intent Analyzer**
  - Pattern detection in logs
  - Identify frequent AST structures

- [ ] **Triage LLM**
  - Classify patterns for rule extraction vs. fine-tuning

- [ ] **Distillation Pipeline**
  - Generate symbolic rules OR training data
  - Output: Python code for new rules

- [ ] **Governance & PR system**
  - Automatic PR creation
  - Human review process
  - Deployment gate

## Development Philosophy

1. **Symbolic first**: Perfect the deterministic components before adding neural ones
2. **Test-driven**: High test coverage, diverse test corpus
3. **Traceable**: Log everything in ExecutionTrace
4. **Incremental**: Complete each phase before moving to next
5. **Documented**: Keep CLAUDE.md, this TODO, and DESIGN.md in sync
6. **Pure Python preference**: Favor pure Python, in-process libraries over external services
7. **Embedded over distributed**: SQLite over PostgreSQL, in-process over Docker

## Quick Reference: What to Work On Now

**Right now (Phase 2 completion)**:
1. Expand parser vocabulary
2. Add participle handling
3. Add correlatives and prepositions
4. Improve test corpus
5. Achieve 100% parsing accuracy

**Next (Phase 3)**:
1. Acquire and clean Esperanto corpus
2. Design GNN architecture
3. Train GNN encoder
4. Implement RAG system

**Then (Phase 4)**:
1. Build Orchestrator
2. Implement symbolic Tool Experts
3. Train Factoid QA Expert
4. Build Writer Loop
