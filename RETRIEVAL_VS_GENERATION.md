# Retrieval vs Generation: Understanding Your Options

## The Key Difference

Your system has **two modes** for answering questions:

### 1. Extractive Retrieval (Current Default)
**Shows the most relevant text from the corpus**

```bash
./query.sh "Kiu estas Frodo?"
```

**Output:**
```
💬 RESPONSE
Laŭ la trovita teksto:
"— Tio plaĉas sufiĉe, se iam okazos tiel, — diris Frodo."
```

**Pros:**
- ✅ Fast (no LLM needed)
- ✅ Accurate (directly from source)
- ✅ Shows exact quotes
- ✅ Good for finding specific passages

**Cons:**
- ❌ Not always a direct answer
- ❌ May be fragmented
- ❌ Doesn't synthesize information
- ❌ Can be out of context

### 2. Generative Answering (New!)
**Synthesizes a natural answer using LLM**

```bash
./ask.sh "Kiu estas Frodo?"
```

**Output:**
```
💬 GENERATED ANSWER
Frodo estas la ĉefa protagonisto de "La Mastro de l' Ringoj".
Li estas hobito kiu heredis la Unu Ringon de Bilbo kaj devas
detrui ĝin en Mordor.
```

**Pros:**
- ✅ Natural language answer
- ✅ Synthesizes multiple sources
- ✅ Direct answer to question
- ✅ Better for complex queries

**Cons:**
- ❌ Slower (requires LLM)
- ❌ May hallucinate if context is weak
- ❌ Less transparent (not exact quotes)
- ❌ Requires me (Claude) to respond

## Side-by-Side Comparison

| Feature | `./query.sh`<br>(Extractive) | `./ask.sh`<br>(Generative) |
|---------|------------------------------|----------------------------|
| Speed | Fast (~5 sec) | Slower (~10-15 sec) |
| Answer Type | Direct quotes | Synthesized answer |
| Accuracy | 100% from source | Depends on LLM |
| Natural Language | No (fragments) | Yes (full sentences) |
| Multi-source | Shows separately | Combines info |
| LLM Required | No | Yes (Claude Code) |
| Best For | Finding passages | Getting answers |

## When to Use Each

### Use `./query.sh` (Extractive) When:
- ✅ You want exact quotes from the text
- ✅ You need fast responses
- ✅ You're searching for specific passages
- ✅ You want to verify source material
- ✅ You don't want LLM involvement

**Examples:**
```bash
./query.sh "Kio estas palantiro?"  # Find descriptions
./query.sh "Kie estas Hobbiton?"   # Find location mentions
./query.sh "Kiam Frodo forvojaĝis?" # Find specific events
```

### Use `./ask.sh` (Generative) When:
- ✅ You want a direct answer to "who/what/where/when/why"
- ✅ You need information synthesized from multiple sources
- ✅ You want natural, coherent responses
- ✅ You don't mind waiting for LLM
- ✅ The question needs interpretation

**Examples:**
```bash
./ask.sh "Kiu estas Frodo?"        # Get character summary
./ask.sh "Kio estas la celo de la vojaĝo?" # Complex question
./ask.sh "Kial Gandalfo helpis Frodon?"    # Reasoning question
```

## How Generative Answering Works

### Pipeline
```
1. Query Input: "Kiu estas Frodo?"
   ↓
2. RAG Retrieval: Find top 5 relevant passages
   ↓
3. Context Assembly: Combine passages with scores
   ↓
4. LLM Prompt: Send question + context to Claude
   ↓
5. Answer Generation: Claude synthesizes answer in Esperanto
   ↓
6. Display: Show answer + sources
```

### Interactive LLM Process

When you run `./ask.sh`, the system will:

1. **Retrieve context** from the corpus
2. **Show you the LLM request** in this format:
   ```
   ================================================================================
   🤖 LLM REQUEST FOR CLAUDE CODE
   ================================================================================

   📋 SYSTEM:
      You are a helpful assistant...

   💬 PROMPT:
      Based on the following context...
      QUESTION: Kiu estas Frodo?
      CONTEXT: [Retrieved sentences]

   ⚙️  PARAMS:
      Max tokens: 200
      Temperature: 0.7

   ================================================================================
   Claude, please respond with the generated text above.
   ================================================================================
   ```

3. **You respond** directly with the answer
4. **System displays** your answer as the generated response

## Commands Reference

### Extractive Retrieval
```bash
# Pure Esperanto
./query.sh "Kiu estas Frodo?"

# With English translation
./query.sh "Kiu estas Frodo?" --translate

# Show keyword filtering
./query.sh "Kiu estas Frodo?" --show-stage1

# Debug mode
./query.sh "Kiu estas Frodo?" --debug
```

### Generative Answering
```bash
# Generate answer (default k=5) - Pure Esperanto
./ask.sh "Kiu estas Frodo?"

# Use more context documents
./ask.sh "Kiu estas Frodo?" -k 10

# With English translation (shows ONLY English)
./ask.sh "Kiu estas Frodo?" --translate

# Debug mode
./ask.sh "Kiu estas Frodo?" --debug
```

**Note:** Both `./query.sh` and `./ask.sh` work the same way:
- **Default:** Pure Esperanto output
- **--translate:** ONLY English output (no Esperanto)
- **--debug:** Show all logging

## Example Session

### Extractive Mode
```bash
$ ./query.sh "Kiu estas Gandalfo?"
```
**Output:**
```
💬 RESPONSE
Laŭ la trovita teksto:
"Gandalfo, aliflanke, malkredis la unuan rakonton de Bilbo..."

📚 SOURCES
1. [0.621] La Mastro de l' Ringoj (Lord of the Rings):?
   Gandalfo, aliflanke, malkredis la unuan rakonton de Bilbo,
   tuj kiam li aŭdis ĝin...

2. [0.589] La Mastro de l' Ringoj (Lord of the Rings):?
   Tiu procezo komenciĝis en la verkado de La Hobito, en kiu
   jam estis kelkaj aludoj pri Gandalfo...
```

### Generative Mode
```bash
$ ./ask.sh "Kiu estas Gandalfo?"
```
**Output:**
```
📚 RETRIEVED CONTEXT (5 documents)
1. [0.621] La Mastro de l' Ringoj
   Gandalfo, aliflanke, malkredis...

🤖 LLM REQUEST FOR CLAUDE CODE
================================================================================
💬 PROMPT:
   Based on the following context...
   QUESTION: Kiu estas Gandalfo?
   ...
================================================================================

💬 GENERATED ANSWER
Gandalfo estas potenca saĝulo kaj gvidanto, kiu helpis Bilbon
kaj poste Frodon en iliaj vojaĝoj. Li estas unu el la plej
gravaj karakteroj en la rakonto de la Ringo.
```

## Advanced: Combining Both Approaches

You can use both tools together:

```bash
# First, find relevant passages
./query.sh "Kio estas la Unu Ringo?" --show-stage1

# Review what was found, then generate answer
./ask.sh "Kio estas la Unu Ringo?" -k 10
```

## Future Enhancements

The system supports **three** answer generation backends:

1. ✅ **Claude Code LLM** (Available now, interactive)
2. ⏸️ **QA Decoder** (Neural model, needs integration)
3. ⏸️ **Summarize Expert** (Already loaded, needs activation)

### Activating the Neural QA Decoder

If you want non-interactive answer generation:

```python
# In quick_query.py or a new script
from scripts.test_end_to_end_qa import EndToEndQASystem

qa_system = EndToEndQASystem(
    qa_model_path=Path('models/qa_decoder/best_model.pt'),
    vocab_path=Path('models/qa_decoder/vocabulary.json'),
    gnn_path=Path('models/tree_lstm/checkpoint_epoch_20.pt'),
    index_path=Path('data/corpus_index'),
    device='cpu'
)

result = qa_system.answer_question(question, top_k=5)
```

This generates answers fully automatically (no Claude interaction needed).

## Summary

**Two tools, two approaches:**

| Tool | Purpose | Speed | Answer Quality | Use Case |
|------|---------|-------|----------------|----------|
| `./query.sh` | Find passages | Fast | Exact quotes | Research, verification |
| `./ask.sh` | Generate answers | Slower | Synthesized | Direct questions |

**Recommendation:** Start with `./query.sh` to understand what's in your corpus, then use `./ask.sh` when you need synthesized answers to specific questions.
