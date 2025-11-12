# Klareco - Neuro-Symbolic AI via Esperanto

Klareco is a **neuro-symbolic AI agent** that fundamentally reimagines language processing by using Esperanto's perfectly regular grammar as an intermediate representation. This allows most linguistic operations to be performed **symbolically** (programmatically) rather than requiring expensive LLM inference at every step.

## The Core Innovation

Traditional LLMs are "opaque" - they require neural network inference for everything: grammar checking, intent classification, semantic understanding, safety validation. **Klareco replaces this opacity with structured, symbolic representations (ASTs)** that can be:
- **Validated programmatically** against the 16 Rules of Esperanto (no LLM needed)
- **Queried and manipulated** using symbolic operations
- **Stored efficiently** in memory as structured data
- **Composed** into multi-step plans deterministically
- **Checked for safety** using rule-based validators

This means **LLMs are only invoked for genuinely creative tasks** while all structural processing happens symbolically, dramatically reducing cost and increasing traceability.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/klareco.git
cd klareco

# Set up environment
conda create -n klareco-env python=3.13
conda activate klareco-env

# For Intel/AMD GPUs (no NVIDIA): Use CPU-only (saves ~1.5GB)
./scripts/ensure_cpu_only.sh
pip install -r requirements-cpu.txt

# For NVIDIA GPUs: Use standard requirements
# pip install -r requirements.txt

# Run tests
./run.sh

# Try an example
python examples/basic_parsing.py
```

## Installation

### Prerequisites
- Python 3.13+
- Conda (recommended) or virtualenv
- 4GB RAM minimum (8GB recommended for translation models)

### Setup

1. **Create conda environment:**
   ```bash
   conda create -n klareco-env python=3.13
   conda activate klareco-env
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download models** (first run will auto-download):
   - Translation models (~1GB) from Hugging Face
   - Lingua language detector (pure Python, no download)

4. **Set up corpus** (optional, for training):
   ```bash
   # Download public domain Esperanto texts
   python scripts/setup_corpus.sh

   # Build vocabulary
   python scripts/build_morpheme_vocab.py
   ```

### CPU-Only Installation (Recommended for Intel/AMD GPUs)

**For systems without NVIDIA GPUs** (Intel integrated, AMD, etc.):

```bash
# Option 1: Use the helper script (recommended)
./scripts/ensure_cpu_only.sh
pip install -r requirements-cpu.txt

# Option 2: Manual installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-cpu.txt
```

**Benefits:**
- ~1.5GB smaller installation
- No unnecessary CUDA dependencies
- Same performance (GPU acceleration not available anyway)

**Verify CPU-only installation:**
```bash
python -c "import torch; print(torch.__version__)"
# Should show +cpu (e.g., 2.1.0+cpu)

python -c "import torch; print(torch.cuda.is_available())"
# Should print: False
```

## Usage

### Command-Line Interface

Klareco provides a unified CLI for all functionality:

```bash
# Run the pipeline on text
python -m klareco run "La hundo vidas la katon."
python -m klareco run --file input.txt
python -m klareco run "Hello world" --output-format json

# Parse Esperanto text (debugging)
python -m klareco parse "mi amas la hundon"

# Translate text
python -m klareco translate "The dog sees the cat."
python -m klareco translate "La hundo vidas la katon." --to en

# Run tests
python -m klareco test
python -m klareco test --num-sentences 10
python -m klareco test --pytest -v

# System information
python -m klareco info

# Setup (download models, build vocabulary)
python -m klareco setup --all
```

**Available Commands:**
- `run` - Run the pipeline on input text (supports text, file, or stdin)
- `test` - Run integration tests (supports both script and pytest)
- `parse` - Parse Esperanto text into AST (debugging utility)
- `translate` - Translate text to/from Esperanto
- `corpus` - Corpus management (clean, verify, create-test)
- `setup` - Setup Klareco (download models, build vocabulary)
- `info` - Display system information

**Common Options:**
- `--debug` - Enable debug-level logging with full context
- `--stop-after <step>` - Stop pipeline at specific step (for debugging)
- `--output-format` - Output format: text, json, or trace
- `-f, --file` - Read input from file
- `-v, --verbose` - Verbose output

For detailed help on any command:
```bash
python -m klareco <command> --help
```

### Basic Pipeline

```python
from klareco.pipeline import KlarecoPipeline

# Initialize the pipeline
pipeline = KlarecoPipeline()

# Process a query (any language)
trace = pipeline.run("The dog sees the cat")

# Access the results
print(trace.final_result)
print(trace.to_json())  # Full execution trace
```

### Parse Esperanto Directly

```python
from klareco.parser import parse

# Parse a sentence
ast = parse("La hundo vidas la katon.")

# Explore the AST
print(ast['subjekto'])  # Subject: "La hundo"
print(ast['verbo'])     # Verb: "vidas"
print(ast['objekto'])   # Object: "la katon"
```

### Convert AST Back to Text

```python
from klareco.parser import parse
from klareco.deparser import deparse

# Round-trip: text → AST → text
ast = parse("Mi amas la grandan hundon.")
reconstructed = deparse(ast)
print(reconstructed)  # "mi amas la grandan hundon"
```

See `examples/` directory for more detailed usage examples.

## Project Structure

```
klareco/
├── klareco/              # Main package
│   ├── parser.py         # Pure Python Esperanto parser (Rule 1-16)
│   ├── deparser.py       # AST-to-text reconstruction
│   ├── pipeline.py       # Full processing orchestration
│   ├── front_door.py     # Language ID + translation
│   ├── intent_classifier.py  # Intent extraction from AST
│   ├── responder.py      # Response generation
│   ├── safety.py         # Input/output validation
│   └── trace.py          # Execution tracing system
├── tests/                # Comprehensive test suite (9 files)
├── scripts/              # Development and setup scripts
├── data/                 # Training corpus (not in git - see DATA_AUDIT.md)
├── examples/             # Usage examples and demos
└── docs/
    ├── DESIGN.md         # 9-phase implementation roadmap
    ├── 16RULES.MD        # Complete Esperanto grammar specification
    ├── CLAUDE.md         # Developer guide (for Claude Code)
    ├── TODO.md           # Current development priorities
    ├── eHy.md            # Vision for Esperanto-Hy integration
    └── DATA_AUDIT.md     # Copyright compliance and data management
```

## Development

### Running Tests

```bash
# Full integration test suite with coverage
./run.sh

# Run limited number of test sentences
python scripts/run_integration_test.py --num-sentences 5

# Stop at a specific pipeline stage
python scripts/run_integration_test.py --stop-after Parser

# Enable debug mode (full error context)
python scripts/run_integration_test.py --debug
```

### Real-Time Log Monitoring

```bash
# Terminal 1: Run tests
./run.sh

# Terminal 2: Watch logs in real-time
./watch.sh
```

### Code Coverage

Coverage reports are generated automatically:
```bash
./run.sh
# Check run_output.txt for coverage report
```

## Current Implementation Status

**Phase 2 (Core Foundation & Traceability)** - ✅ Complete

- ✅ Pure Python morpheme-based parser implementing the 16 Rules
- ✅ AST-to-text deparser
- ✅ Language identification (Lingua)
- ✅ Translation pipeline (Opus-MT)
- ✅ Execution trace system with JSON logging
- ✅ Safety monitors (input/AST complexity)
- ✅ Basic intent classification and response generation
- ✅ Comprehensive test suite with coverage tracking

### Parser Performance

**Current performance on diverse Esperanto corpus (547 MB, 28 texts):**

| Text Type | Success Rate | AST Production | Notes |
|-----------|-------------|----------------|-------|
| Standard Esperanto (Zamenhof) | 99.1% | 100% | Baseline - authoritative texts |
| Literary Fiction (Poe) | 96.3% | 100% | Creative writing |
| Fantasy Literature (Tolkien) | 93.6% | 100% | Specialized vocabulary |
| Wikipedia (Encyclopedia) | 87.6% | 100% | Technical terms, proper nouns |
| **Overall Average** | **93.9%** | **100%** | Across all text types |

**Key Features:**
- ✅ **Graceful degradation** - 100% of sentences produce valid ASTs (zero crashes)
- ✅ **Vocabulary: 192 verified Esperanto roots** (manually audited, no foreign words)
- ✅ **Unknown word categorization** - Proper names, foreign words flagged (not discarded)
- ✅ **Corpus: 547 MB clean Esperanto text** (removed 72% HTML/wiki artifacts from Wikipedia)

See `COMPREHENSIVE_PARSER_REPORT.md`, `PARSER_VOCABULARY_AUDIT.md`, and `WIKIPEDIA_PARSER_VERIFICATION.md` for detailed performance analysis.

**Planned (Phases 3-9):**
- Phase 3: GNN Encoder for semantic RAG
- Phase 4: Expert routing and tool integration
- Phase 5: Multi-step planning with Blueprints
- Phase 6: Agentic memory system (STM/LTM)
- Phase 7: Goals and Values manifests
- Phase 8: External tool integration (Web, Code, Prolog)
- Phase 9: Learning loop with emergent intent analysis

See `DESIGN.md` for the complete roadmap.

## Documentation

- **[DESIGN.md](DESIGN.md)** - Full 9-phase architecture and implementation plan
- **[16RULES.MD](16RULES.MD)** - Complete Esperanto grammar specification (Zamenhof's 16 Rules)
- **[CLAUDE.md](CLAUDE.md)** - Developer guide for working with this codebase
- **[TODO.md](TODO.md)** - Current tasks and development priorities
- **[eHy.md](eHy.md)** - Vision for Esperanto-Hy linguistic programming integration
- **[DATA_AUDIT.md](DATA_AUDIT.md)** - Data management and copyright compliance

## Data and Copyright

This project uses a mix of public domain and copyrighted training data:
- **Public domain sources**: Project Gutenberg, Wikipedia (CC-BY-SA)
- **Copyrighted sources**: Used locally under fair use for research (not distributed)

**Important**: The `data/` directory is excluded from git to protect copyrighted material. Users who clone this repo can:
1. Download public domain sources using `scripts/download_public_data.sh`
2. Optionally add their own copyrighted books for local research use

See `DATA_AUDIT.md` for full details on copyright compliance.

## Why Esperanto?

Esperanto's **perfectly regular grammar** (16 rules, no exceptions) enables deterministic parsing that's impossible with natural languages:

1. **All morphology is transparent**: Every word can be decomposed into prefix + root + suffixes + POS ending + plural + case
2. **Case marking eliminates ambiguity**: Word order flexibility without parsing ambiguity
3. **No irregular verbs/nouns**: Every grammatical transformation is rule-based
4. **Productive affixation**: New words can be validated without a dictionary

This regularity means we can parse **symbolically** with 100% accuracy, then use LLMs only for semantic tasks.

## Technology Stack

- **Parser**: Pure Python (no Lark, no external parser libraries)
- **Language ID**: Lingua (pure Python, 99.9% accuracy)
- **Translation**: Hugging Face Transformers + Opus-MT models
- **Neural Components**: PyTorch (for future GNN encoder)
- **Testing**: pytest + coverage
- **Logging**: Python logging with real-time monitoring

## License

[Your license here - MIT, Apache 2.0, GPL, etc.]

## Contributing

This project is under active development. Current focus is on:
- Expanding parser vocabulary and edge case handling
- Building comprehensive test corpus
- Documenting the 16 Rules implementation

See `TODO.md` for current priorities.

## Citation

If you use Klareco in your research, please cite:

```bibtex
[Your citation format here]
```

## Acknowledgments

- L. L. Zamenhof for creating Esperanto's perfectly regular grammar
- The Esperanto community for public domain texts
- Project Gutenberg for digitized Esperanto literature
