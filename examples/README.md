# Klareco Examples

This directory contains usage examples demonstrating different aspects of Klareco's functionality.

## Quick Start

Make sure you've installed dependencies first:
```bash
pip install -r requirements.txt
```

Then run any example:
```bash
python examples/basic_parsing.py
```

## Available Examples

### 1. `basic_parsing.py` - Start Here!
**What it shows:** Core parsing functionality
- Parse single words into morpheme-based ASTs
- Parse complete sentences
- Explore AST structure
- Understand Esperanto grammar basics

**Run time:** Instant
**Difficulty:** Beginner

```bash
python examples/basic_parsing.py
```

**Sample output:**
```
Input: 'hundon'
Morpheme Analysis:
  Root (radiko):        hund
  Part of Speech:       substantivo
  Number (nombro):      singularo
  Case (kazo):          akuzativo
```

---

### 2. `morpheme_analysis.py` - Deep Dive
**What it shows:** Detailed morphological analysis
- How words are decomposed into morphemes
- Prefix, root, suffix identification
- Grammatical markers (case, number, tense)
- The 16 Rules in action

**Run time:** Instant
**Difficulty:** Intermediate

```bash
python examples/morpheme_analysis.py
```

**Perfect for:** Understanding how Klareco achieves deterministic parsing through Esperanto's regular morphology.

---

### 3. `round_trip.py` - Verify Completeness
**What it shows:** Text → AST → Text conversion
- Parse sentences to ASTs
- Reconstruct original text from ASTs
- Verify lossless representation
- Understand AST structure

**Run time:** Instant
**Difficulty:** Intermediate

```bash
python examples/round_trip.py
```

**Key insight:** The AST preserves ALL linguistic information, enabling symbolic operations without LLMs.

---

### 4. `full_pipeline.py` - End-to-End Processing
**What it shows:** Complete Klareco pipeline
- Language identification
- Translation to Esperanto
- Parsing to AST
- Intent classification
- Response generation
- Execution tracing

**Run time:** ~10-30 seconds first run (downloads models), instant after
**Difficulty:** Intermediate

```bash
python examples/full_pipeline.py
```

**Note:** First run downloads ~1GB of translation models from Hugging Face. Subsequent runs use cached models.

**Sample output:**
```
Input: 'The dog sees the cat'

Pipeline Steps:
  1. SafetyMonitor: Checking input length
  2. FrontDoor: Processing input text
  3. Parser: Parsing Esperanto text to AST
  4. IntentClassifier: Extracting intent from AST
  5. Responder: Generating response

Final Result:
  [Response from system]
```

---

## Learning Path

**New to Klareco?** Follow this order:

1. **Start with `basic_parsing.py`**
   Get familiar with parsing and ASTs.

2. **Then try `morpheme_analysis.py`**
   Understand how morphological decomposition works.

3. **Next: `round_trip.py`**
   See how ASTs preserve complete information.

4. **Finally: `full_pipeline.py`**
   Experience the end-to-end system.

**Want to build something?**
After running examples, check the main README.md for:
- API documentation
- Project structure
- Development guide

## Common Issues

### ModuleNotFoundError: No module named 'klareco'
Make sure you're running from the project root:
```bash
# From klareco/ directory:
python examples/basic_parsing.py
```

### Translation models downloading slowly
First run downloads ~1GB of models. This is normal. Subsequent runs are instant.

Use CPU-only installation if you don't have a GPU:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Unknown root errors when parsing
The current parser has a limited vocabulary (Phase 2 focus). See `TODO.md` for vocabulary expansion plans (Phase 2.5).

## Going Further

After trying these examples:

- **Read the docs:**
  - `DESIGN.md` - Full system architecture (9 phases)
  - `16RULES.MD` - Complete Esperanto grammar specification
  - `CLAUDE.md` - Developer guide for contributing

- **Run tests:**
  ```bash
  ./run.sh  # Full integration test suite
  ```

- **Check the roadmap:**
  - `TODO.md` - Current development priorities
  - Phase 3 (next): GNN Encoder for semantic RAG

- **Explore the code:**
  - `klareco/parser.py` - Pure Python parser implementation
  - `klareco/pipeline.py` - Full processing orchestration
  - `tests/` - Comprehensive test suite

## Contributing Examples

Have a cool example to share? Contributions welcome!

Good example topics:
- Error handling patterns
- Custom intent classification
- AST manipulation/transformation
- Integration with external tools
- Performance benchmarking

See the main README.md Contributing section for guidelines.
