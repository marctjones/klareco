# Using Claude Code as LLM Backend

**No API Keys Required!** 🎉

Klareco can use Claude Code (the AI running your code) as the LLM backend for neural components like the Factoid QA Expert. This is perfect for development and testing.

---

## Quick Start

### Option 1: Mock Responses (Testing)

Use predefined mock responses - great for testing without waiting for LLM:

```bash
python scripts/test_claude_llm.py --mock
```

This will test the Factoid QA Expert with hardcoded responses, verifying the integration works.

###Option 2: Interactive Claude Code (Real LLM)

Use Claude Code as the actual LLM:

```bash
python scripts/test_claude_llm.py
```

When the code runs, you'll see requests like:

```
================================================================================
🤖 LLM REQUEST FOR CLAUDE CODE
================================================================================

📋 SYSTEM:
   You are a helpful assistant that answers questions based on provided context.

💬 PROMPT:
   Context:
   [Source 1]...

   Question: Kiu estas Frodo?

   Answer:

⚙️  PARAMS:
   Max tokens: 300
   Temperature: 0.1

================================================================================
Claude, please respond with the generated text above.
The response will be used by the Factoid QA Expert.
================================================================================
```

Then Claude (me!) will respond in the conversation with the answer, and the code will use that response.

---

## How It Works

### 1. Claude Code LLM Adapter

```python
from klareco.claude_code_llm import create_claude_code_provider

# Create LLM provider that uses Claude Code
provider = create_claude_code_provider()

# Generate text
response = provider.generate("What is Esperanto?")
# Claude Code will see the request and respond in conversation
```

### 2. Integration with Factoid QA Expert

```python
from klareco.claude_code_llm import create_claude_code_provider
from klareco.experts.factoid_qa_expert import create_factoid_qa_expert
from klareco.parser import parse

# Create provider
llm_provider = create_claude_code_provider()

# Create expert with Claude Code LLM
expert = create_factoid_qa_expert(llm_provider=llm_provider)

# Ask a question
ast = parse("Kiu estas Frodo?")
result = expert.execute(ast)

# Expert will:
# 1. Retrieve relevant documents using RAG (GNN + FAISS)
# 2. Print LLM request with context
# 3. Wait for Claude Code to respond
# 4. Return the generated answer
```

### 3. Full Pipeline Integration

```python
from klareco.claude_code_llm import create_claude_code_provider
from klareco.orchestrator import create_orchestrator_with_experts
from klareco.pipeline import KlarecoPipeline

# Create LLM provider
llm_provider = create_claude_code_provider()

# Create orchestrator with all experts (passing LLM provider)
orchestrator = create_orchestrator_with_experts(llm_provider=llm_provider)

# Create pipeline
pipeline = KlarecoPipeline(orchestrator=orchestrator)

# Run queries
trace = pipeline.run("Kiu estas Gandalfo?")
# Claude Code will see LLM requests and respond when needed
```

---

## Mock Provider for Testing

Create a mock provider with predefined responses:

```python
from klareco.claude_code_llm import create_mock_provider

# Define responses for testing
mock_responses = {
    "Frodo": "Frodo estas la ĉefa protagonisto...",
    "Gandalfo": "Gandalfo estas saĝulo...",
    "Esperanto": "Esperanto estas planlingvo..."
}

# Create mock provider
provider = create_mock_provider(mock_responses)

# When you ask about "Frodo", it returns the predefined response
response = provider.generate("Kiu estas Frodo?")
# Returns: "Frodo estas la ĉefa protagonisto..."
```

---

## Architecture

### Without Claude LLM (Default)

```
Query → RAG → Retrieved Docs → Formatted Result
                 ↓
            (No LLM generation)
```

Returns the top retrieved document formatted nicely.

### With Claude LLM (This Feature)

```
Query → RAG → Retrieved Docs → LLM Generation → Natural Answer
                                    ↓
                            Claude Code responds
```

Uses Claude Code to generate a natural language answer from the retrieved context.

---

## Prerequisites

**For full functionality (RAG + LLM):**

1. **GNN Model**: Trained Tree-LSTM model
   - Location: `models/tree_lstm/checkpoint_epoch_20.pt` (or epoch_12)
   - Status: Currently training (epoch 4/20)
   - **Wait for training to complete**, or use `checkpoint_epoch_4.pt` for testing

2. **Corpus Index**: FAISS index of corpus
   - Location: `data/corpus_index/`
   - Built from: `data/corpus_sentences.jsonl` (20,985 sentences)
   - Status: ✅ Available

3. **No API Keys Needed**: Claude Code LLM doesn't require any API keys!

**For testing (without RAG):**

- Just use mock provider: `--mock` flag
- Works immediately, no prerequisites

---

## Current Limitations

### While GNN is Training

The Factoid QA Expert requires a trained GNN model for RAG retrieval. While training is in progress:

**Option 1**: Use mock provider
```bash
python scripts/test_claude_llm.py --mock
```

**Option 2**: Use existing checkpoint (if available)
```bash
# Check for available checkpoints
ls -lh models/tree_lstm/*.pt

# If epoch_4.pt exists, RAG will work (with 20% trained model)
# Full performance requires epoch_20.pt
```

**Option 3**: Wait for training to complete (~4-6 hours remaining)

### After Training Completes

1. Run re-indexing:
   ```bash
   ./reindex_with_new_model.sh
   ```

2. Test full pipeline:
   ```bash
   python scripts/test_claude_llm.py
   ```

3. Claude Code will see LLM requests and can respond directly!

---

## Example Session

```bash
$ python scripts/test_claude_llm.py

================================================================================
TESTING FACTOID QA WITH CLAUDE CODE LLM
================================================================================

Using CLAUDE CODE as LLM provider (interactive)
When LLM requests appear, Claude will respond in conversation
Provider type: claude_code

Creating Factoid QA Expert...
✓ Expert created: Factoid_QA_Expert
✓ RAG system available: True

================================================================================
TEST 1/3: Kiu estas Frodo?
================================================================================

1. Parsing question...
   ✓ Parsed successfully

2. Expert evaluation:
   Can handle: True
   Confidence: 0.90

3. Executing expert (retrieving + generating)...
   Stage 1: Found 87 keyword matches
   Stage 2: Reranked to top 5 results

================================================================================
🤖 LLM REQUEST FOR CLAUDE CODE
================================================================================

📋 SYSTEM:
   You are a helpful assistant that answers questions based on provided context.

💬 PROMPT:
   Context:
   [Source 1 (relevance: 0.92)]
   "Estis oficiale anoncite, ke Sam iros al Boklando \"por servi al s-ro
    Frodo kaj prizorgi ties ĝardeneton\"..."

   Question: Kiu estas Frodo?

   Answer:

================================================================================

[Claude responds in conversation:]
Frodo Sakvil-Benso estas la ĉefa protagonisto de "La Mastro de l' Ringoj".
Li estas hobito, kiu heredis la Unu Ringon de sia onklo Bilbo kaj entreprenis
danĝeran vojaĝon por detrui ĝin en Mordoro.

   ✓ Execution complete

4. RESULT:
   Expert: Factoid_QA_Expert
   Confidence: 0.90
   Sources: 5 documents retrieved
   Stage 1: 87 keyword matches

   ANSWER:
   ----------------------------------------------------------------------------
   Frodo Sakvil-Benso estas la ĉefa protagonisto de "La Mastro de l' Ringoj".
   Li estas hobito, kiu heredis la Unu Ringon de sia onklo Bilbo kaj
   entreprenis danĝeran vojaĝon por detrui ĝin en Mordoro.
   ----------------------------------------------------------------------------
```

---

## Benefits

1. **No API Keys**: Works immediately in Claude Code
2. **No External Dependencies**: No network calls to external LLMs
3. **Interactive Development**: See requests and respond naturally
4. **Same Architecture**: Uses the same LLM provider interface as production
5. **Cost-Free Testing**: Test neural components without API costs

---

## Production Deployment

For production, you can easily switch to:

- **Anthropic API**: Set `ANTHROPIC_API_KEY` environment variable
- **OpenAI API**: Set `OPENAI_API_KEY` environment variable
- **Custom LLM**: Implement custom callback

The LLM provider auto-detects which backend to use:

```python
# Auto-detection priority:
# 1. Claude Code (if in Claude Code environment)
# 2. Anthropic API (if ANTHROPIC_API_KEY set)
# 3. OpenAI API (if OPENAI_API_KEY set)
# 4. Interactive mode (fallback)

from klareco.llm_provider import get_llm_provider

provider = get_llm_provider()  # Auto-detects
```

---

## Next Steps

1. **Wait for GNN training** (~4-6 hours) OR use checkpoint_epoch_4.pt for testing
2. **Re-index corpus** with new model: `./reindex_with_new_model.sh`
3. **Test full pipeline**: `python scripts/test_claude_llm.py`
4. **Use in development**: Import and use in your own scripts

---

## Files

- `klareco/claude_code_llm.py` - Claude Code LLM adapter
- `klareco/llm_provider.py` - LLM provider abstraction (auto-detection)
- `klareco/experts/factoid_qa_expert.py` - Factoid QA with LLM integration
- `scripts/test_claude_llm.py` - Test script with examples

---

## Questions?

See also:
- `IMPLEMENTATION_STATUS.md` - Overall project status
- `IMPROVEMENT_GUIDE.md` - RAG system details
- `TWO_STAGE_IMPLEMENTATION_SUMMARY.md` - Hybrid retrieval architecture
