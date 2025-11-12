# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Klareco is a **neuro-symbolic AI agent** that fundamentally reimagines how AI systems process language. By using Esperanto's perfectly regular grammar as an intermediate representation, the system can perform most linguistic operations **symbolically** (programmatically) rather than requiring expensive LLM inference at every step.

### The Core Innovation: The Esperanto AST

Traditional LLM systems are "opaque" - they require neural network inference for grammar checking, intent classification, semantic understanding, and safety validation. Klareco replaces this opacity with **structured, symbolic representations** (ASTs) that can be:
- **Validated programmatically** against the 16 Rules (no LLM needed)
- **Queried and manipulated** using symbolic operations
- **Stored efficiently** in memory as structured data
- **Composed** into multi-step plans deterministically
- **Checked for safety** using rule-based validators

This means **LLMs are only invoked for genuinely creative/generative tasks** (answering complex questions, summarization) while all structural processing happens symbolically. This dramatically reduces cost, increases traceability, and enables capabilities impossible in pure neural systems.

### Full System Architecture (9-Phase Design)

Klareco is not just a parser - it's a complete agentic AI system with:

1. **Expert System Architecture** - Specialized experts routed by an Orchestrator:
   - Factoid_QA_Expert (neural decoder for factual questions)
   - Summarize_Expert (neural decoder for summarization)
   - Tool Experts (Math, Date, Dictionary, Grammar - all symbolic)
   - Memory_Read_Tool / Memory_Write_Tool
   - Orchestrator with Gating Network routes queries to appropriate experts

2. **Planning Capabilities** - Multi-step reasoning:
   - Execution Loop: `while not goal_achieved:` logic
   - Orchestrator generates multi-step "Blueprints" from ASTs
   - Neural Clusterer organizes complex tasks
   - Tool-use intents trigger argument generation from ASTs

3. **Agentic Memory System** - Structured, persistent memory:
   - Short-Term Memory (STM): Recent interactions stored as ASTs (not text!)
   - Long-Term Memory (LTM): SQL/Graph database for consolidated facts
   - Consolidate_Expert: Scheduled summarization of STM → LTM
   - Enables personalization and context maintenance across sessions

4. **Goals System** - Strategic planning:
   - Goals with Priority and Completion Criteria
   - Pre-Query Goal Check by Orchestrator
   - System can pursue long-term objectives

5. **Values System** - Ethical/motivational framework:
   - Values with Name, Weight, and Conflict resolution
   - Post-Retrieval Reflection generates "Weighting Instructions"
   - Writer Loop incorporates values during AST construction
   - Enables alignment without ad-hoc prompting

6. **RAG with GNN Encoder** - Semantic search on structure:
   - Graph Neural Network operates on AST structure (not just text embeddings)
   - More precise semantic representations because grammar is explicit
   - Corpus indexed as ASTs for efficient retrieval

7. **External Tool Integration** - Real-world actions:
   - Web_Search_Tool, Code_Interpreter_Tool, Formal_Logic_Tool
   - Sandboxed execution environment
   - ASTs provide structured input/output formats for tools

8. **Learning Loop** - Self-improvement:
   - All operations logged in Execution Traces (JSON)
   - Emergent Intent Analyzer identifies new patterns
   - Distillation Pipeline creates new symbolic rules or training data
   - Human-in-the-loop governance via PR review

### What the Symbolic AST Enables (Avoiding Expensive LLM Calls)

In a traditional LLM system, you need expensive inference for:
- Grammar checking → Klareco: **Deterministic validation** against 16 Rules
- Intent classification → Klareco: **Rule-based Level 1 filter** from AST structure
- Semantic understanding → Klareco: **GNN encoding** of explicit AST structure
- Memory retrieval → Klareco: **Structured queries** on AST database
- Safety validation → Klareco: **Symbolic checks** on AST complexity/content
- Tool argument extraction → Klareco: **Programmatic traversal** of AST
- Multi-step planning → Klareco: **Symbolic composition** of AST blueprints

The system processes multi-language queries by translating them to Esperanto, parsing them into detailed morpheme-based ASTs, and then processing through a hybrid pipeline where **symbolic operations handle structure** and **neural components handle semantics**.

## Current Implementation Status

The project is currently in **Phase 3-4** (GNN Encoder + Expert System) of the 9-phase roadmap. Here's what's built vs. planned:

### ✅ Completed (Phases 1-3)
- **Front Door**: Language identification (FastText) and translation (Opus-MT)
- **Parser**: Pure Python, morpheme-based parser implementing the 16 Rules (95.7% accuracy on 1.27M sentences)
- **Deparser**: AST-to-text reconstruction
- **Execution Trace**: Complete JSON-based logging system
- **Safety Monitor**: Input length and AST complexity validation
- **Pipeline**: Full orchestration with traceability and expert system integration
- **Test Infrastructure**: Integration tests, unit tests, coverage tracking
- **GNN Encoder**: Tree-LSTM for encoding Esperanto ASTs (98.7% accuracy, 1.7M parameters, training complete)
- **Baseline RAG**: FAISS-based semantic search for comparison

### ✅ Completed (Phase 4 - Expert System) **NEW**
- **Gating Network**: Symbolic intent classifier (Level 1) with 7 intent types
- **Orchestrator**: Expert routing with fallback capability-based selection
- **MathExpert**: Symbolic computation (arithmetic operations, symbolic)
- **DateExpert**: Temporal queries (current date/time, day of week, symbolic)
- **GrammarExpert**: AST grammatical analysis and explanation (symbolic)
- **Expert Integration**: All three experts integrated into main pipeline
- **End-to-End Demo**: Complete demo showing multi-language → AST → expert routing → response

### 🚧 In Development
- **Parser Refinement**: Expanding vocabulary (8,397 roots), handling edge cases
- **Test Corpus**: Building diverse Esperanto test sentences

### 📋 Planned (Phases 4-9)
- **Phase 4 (Remaining)**: Dictionary Expert, Factoid_QA_Expert (neural RAG), Execution Loop, Memory Tools
- **Phase 5**: Summarize_Expert, multi-step Blueprints, Neural Gating Network (Level 2)
- **Phase 6**: STM/LTM memory system, Memory_Read/Write_Tools, Consolidate_Expert
- **Phase 7**: Goals and Values manifests with alignment
- **Phase 8**: External tool integration (Web Search, Code Interpreter, Prolog)
- **Phase 9**: Learning Loop with Emergent Intent Analyzer and Distillation Pipeline

**Development Philosophy**: Build the symbolic foundation perfectly first (parser, AST structure, traceability), then layer neural components on top. Phase 3 GNN and Phase 4 symbolic experts are complete - ready for neural RAG expert integration.

## Core Architecture

### The 16 Rules Foundation
The entire parser is built on Zamenhof's 16 Rules of Esperanto Grammar (detailed in `16RULES.MD`). This perfect regularity allows for deterministic, rule-based parsing without probabilistic models. Every grammatical feature (part of speech, tense, case, number) can be extracted purely from morphological analysis.

### Processing Pipeline (klareco/pipeline.py)
The `KlarecoPipeline` orchestrates all processing through these stages:
1. **SafetyMonitor** - Input validation (length checks)
2. **FrontDoor** - Language identification and translation to Esperanto
3. **Parser** - Morphological analysis producing detailed ASTs
4. **SafetyMonitor** - AST complexity validation
5. **Orchestrator** - Intent classification (Gating Network) + Expert routing
   - **Gating Network**: Symbolic rules classify intent (7 types: calculation_request, temporal_query, grammar_query, factoid_question, dictionary_lookup, command_intent, general_query)
   - **Expert Selection**: Routes to registered expert or uses fallback capability-based selection
   - **Expert Execution**: Specialized handler processes query (MathExpert, DateExpert, GrammarExpert, etc.)
6. **Response Integration** - Expert response integrated into trace

All operations are logged through the `ExecutionTrace` system for complete traceability.

### Parser Architecture (klareco/parser.py)
The parser is a **pure Python, from-scratch implementation** (no Lark or external parsing libraries) that works in layers:
1. **Morphological Layer** (`parse_word`) - Strips grammatical endings right-to-left (-n for accusative, -j for plural, then part-of-speech endings)
2. **Lexical Layer** - Identifies prefixes, root, and suffixes
3. **Syntactic Layer** (`parse_sentence`) - Identifies subject, verb, object using case markers

The parser produces morpheme-based ASTs with these key properties:
- `radiko` (root), `vortspeco` (part of speech), `nombro` (number), `kazo` (case)
- `prefikso`, `sufiksoj` for affixes
- `tempo`/`modo` for verbs

### Deparser (klareco/deparser.py)
Reconstructs Esperanto text from ASTs by assembling morphemes left-to-right: prefix + root + suffixes + POS ending + plural marker + case marker.

### Front Door (klareco/front_door.py)
Uses FastText for language identification (`lang_id.py`) and Opus-MT models for translation (`translator.py`). All input is normalized to Esperanto for internal processing.

### Execution Trace (klareco/trace.py)
A JSON-based logging system that records every pipeline step with inputs, outputs, and descriptions. Essential for debugging and future learning loops.

## Commands

### Environment Setup
```bash
# Activate the conda environment
source /home/marc/miniconda3/bin/activate klareco-env
```

### Unified CLI

**Primary interface for all Klareco functionality:**

```bash
# Run the pipeline on input text
python -m klareco run "La hundo vidas la katon."
python -m klareco run --file input.txt
python -m klareco run "Hello world" --output-format json --debug

# Parse Esperanto text (debugging)
python -m klareco parse "mi amas la hundon"

# Translate text
python -m klareco translate "The dog sees the cat."

# Run tests
python -m klareco test
python -m klareco test --num-sentences 10
python -m klareco test --pytest -v

# System information
python -m klareco info

# Setup (download models, build vocabulary)
python -m klareco setup --all
```

**Available commands:**
- `run` - Run the pipeline on input text
- `test` - Run integration tests (replaces scripts/run_integration_test.py)
- `parse` - Parse Esperanto text into AST (debugging)
- `translate` - Translate text to/from Esperanto
- `corpus` - Corpus management (clean, verify, create-test)
- `setup` - Setup Klareco (download models, build vocabulary)
- `info` - Display system information

**Key options:**
- `--debug` - Enable debug-level logging with full context
- `--stop-after <step>` - Stop pipeline at specific step
- `--output-format` - text, json, or trace
- `-f, --file` - Read input from file

For detailed help: `python -m klareco <command> --help`

### Running Tests
```bash
# Run full integration test suite with coverage
./run.sh

# Run integration tests and stop at a specific stage
python scripts/run_integration_test.py --stop-after Parser

# Run limited number of test sentences
python scripts/run_integration_test.py --num-sentences 5

# Test expert system
python scripts/test_experts.py          # Test individual experts
python scripts/test_orchestrator.py     # Test orchestrator integration

# Run end-to-end demo
python scripts/demo_klareco.py          # Full multi-language demo

# Run unit tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_parser.py

# Run with coverage
python -m coverage run --source=klareco -m pytest tests/
python -m coverage report -m
```

### Development Scripts
```bash
# Build morpheme vocabulary from corpus
python scripts/build_morpheme_vocab.py

# Run parser standalone test
python scripts/run_parser_test.py
```

### Real-Time Log Monitoring with Full Visibility

**For monitoring long-running tests and seeing progress in real-time:**

1. **Terminal 1** - Run tests:
   ```bash
   ./run.sh
   # or for custom runs:
   python scripts/run_integration_test.py --num-sentences 10
   # or with DEBUG mode for full context:
   python scripts/run_integration_test.py --num-sentences 10 --debug
   ```

2. **Terminal 2** - Watch logs in real-time:
   ```bash
   ./watch.sh
   ```

**What You'll See:**

**Normal Mode (INFO level):**
```
2025-11-11 10:25:29,031 - INFO - Testing sentences: 1/5 (20%) - Sentence 1: En la maniero... [ETA: 3s]
2025-11-11 10:25:29,031 - INFO - Running pipeline for sentence 1/5
2025-11-11 10:25:29,032 - INFO - Step 1: SafetyMonitor - Checking input length
2025-11-11 10:25:29,032 - INFO - Step 2: FrontDoor - Processing input text
2025-11-11 10:25:29,373 - INFO - Step 3: Parser - Parsing Esperanto text to AST
2025-11-11 10:25:29,374 - ERROR - FAILED sentence 1: Ne povis trovi validan radikon en 'En'
2025-11-11 10:25:30,031 - INFO - Testing sentences: 2/5 (40%) - Sentence 2: ... [ETA: 9s]
2025-11-11 10:25:35,031 - INFO - PASSED sentence 2
...
2025-11-11 10:25:45,031 - INFO - Integration test COMPLETE: 3 passed, 2 failed out of 5
```

**Debug Mode (--debug flag):**
```
2025-11-11 10:25:29,030 - INFO - DEBUG MODE ENABLED - Verbose logging active
2025-11-11 10:25:29,031 - root - DEBUG - [run_integration_test.py:81] - Full input: En la maniero de mia amiko tujege...
2025-11-11 10:25:29,374 - root - ERROR - [pipeline.py:114] - Pipeline failed with error: Ne povis...
Traceback (most recent call last):
  File "/home/marc/klareco/klareco/pipeline.py", line 65, in run
    ast = parse_esperanto(processed_text)
    ...
ValueError: Ne povis trovi validan radikon en 'En'. Restaĵo: ''
2025-11-11 10:25:29,374 - root - DEBUG - [run_integration_test.py:89] - Full error: Ne povis trovi...
2025-11-11 10:25:29,374 - root - DEBUG - [run_integration_test.py:90] - Trace: {"trace_id": "...", ...}
```

**Features:**
- ✅ **Progress indicators**: "X/Y (Z%) [ETA: Ns]" every 10%
- ✅ **Test results**: "PASSED" or "FAILED" for each test
- ✅ **Pipeline steps**: See each stage (SafetyMonitor, FrontDoor, Parser, etc.)
- ✅ **Error messages**: Immediate error logging with truncated context
- ✅ **Debug mode**: Full inputs, stack traces, AST state, file:line numbers
- ✅ **Summary**: Total passed/failed count at end
- ✅ **Works with tqdm**: Progress bars in console, structured logs in file

**Debug Mode Benefits:**
- See exact input that caused failure
- Full stack traces in log file
- File names and line numbers for each log
- Complete error context
- AST/trace dumps for debugging

**Tip:** Start `watch.sh` before running tests to see everything from the beginning.

### Testing Individual Components
```bash
# Test parser directly
python -c "from klareco.parser import parse; print(parse('La hundo vidas la katon.'))"

# Test deparser
python -c "from klareco.parser import parse; from klareco.deparser import deparse; ast = parse('La hundo vidas la katon.'); print(deparse(ast))"

# Test front door
python -c "from klareco.front_door import FrontDoor; fd = FrontDoor(); print(fd.process('Hello world'))"

# Test orchestrator with experts
python -c "from klareco.parser import parse; from klareco.orchestrator import create_orchestrator_with_experts; o = create_orchestrator_with_experts(); ast = parse('Kiom estas du plus tri?'); print(o.route(ast))"
```

## Key Implementation Details

### Morpheme Analysis Order
When parsing words, the system strips endings **right-to-left**:
1. Accusative -n (if present)
2. Plural -j (if present)
3. Part-of-speech ending (-o, -a, -e, -as, -is, -os, -us, -u, -i)
4. Suffixes (innermost to outermost)
5. Prefix (leftmost)
6. Root (what remains)

When deparsing, it assembles **left-to-right**: prefix + root + suffixes + POS + plural + case.

### AST Structure
```python
# Word-level AST
{
    "tipo": "vorto",
    "plena_vorto": "hundon",
    "radiko": "hund",
    "vortspeco": "substantivo",
    "nombro": "singularo",
    "kazo": "akuzativo",
    "prefikso": None,
    "sufiksoj": []
}

# Sentence-level AST
{
    "tipo": "frazo",
    "subjekto": {...},  # vortgrupo (noun phrase)
    "verbo": {...},     # vorto
    "objekto": {...},   # vortgrupo
    "aliaj": [...]      # other words
}
```

### Vocabulary Management
The parser uses hardcoded vocabulary in `parser.py`:
- `KNOWN_ROOTS` - Core semantic roots
- `KNOWN_PREFIXES` - Grammatical prefixes (mal-, re-, ge-)
- `KNOWN_SUFFIXES` - Derivational suffixes (-ul, -ej, -in, -et, -ad, -ig)
- `KNOWN_ENDINGS` - Grammatical endings with their properties

To extend vocabulary, add roots to these dictionaries or implement the planned vocabulary loading from `scripts/build_morpheme_vocab.py`.

### Future E-Hy Integration
The long-term vision (documented in `eHy.md`) is to make every Esperanto word a programmable Hy object with queryable semantic and grammatical properties, unifying natural language, data, and code.

## Design Documents
- `DESIGN.md` - 9-phase implementation roadmap
- `16RULES.MD` - Complete Esperanto grammar specification
- `eHy.md` - Vision for Esperanto-Hy integration
- `TODO.md` - Current development priorities
- `EXPERT_INTEGRATION_SUMMARY.md` - Expert system architecture and integration (Phase 4)
- `GNN_TRAINING_GUIDE.md` - Guide for training the Tree-LSTM GNN encoder (Phase 3)
- `CHECKPOINT_RESUMPTION_SUMMARY.md` - Checkpoint save/resume implementation details

## Testing Philosophy
The system uses a multi-tiered testing approach:
1. Unit tests for individual components (tests/)
2. Integration tests via `scripts/run_integration_test.py`
3. Test corpus at `data/test_corpus.json` (50+ diverse sentences)
4. Coverage tracking to ensure comprehensive testing
5. All logs go to `run_output.txt` (from run.sh)
