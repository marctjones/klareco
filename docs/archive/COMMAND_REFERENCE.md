# Klareco Command Reference

## Philosophy: Local-First AI

**Klareco uses LOCAL MODELS as the primary approach.** External LLMs (like Claude Code) are STOPGAP measures only used when local features are not yet implemented. When you see "EXTERNAL LLM" warnings, that indicates temporary code that should eventually be replaced with local models.

## Quick Start

### Three Main Commands

| Command | Purpose | Model Type | Output Type |
|---------|---------|------------|-------------|
| `./query.sh` | **Extractive retrieval** | Local (GNN) | Direct quotes from corpus |
| `./ask.sh` | **Generative answering** | **LOCAL (QA Decoder)** | Synthesized answer |
| `./ask-external.sh` | **Generative answering** | ⚠️ **EXTERNAL LLM (STOPGAP)** | Synthesized answer |

**Use `./ask.sh` (local) by default.** Only use `./ask-external.sh` if the local model fails.

## Common Options (Work for All Commands)

### Translation Modes

```bash
# Pure Esperanto (default)
./query.sh "Kiu estas Frodo?"
./ask.sh "Kiu estas Frodo?"
# Output: Pure Esperanto only

# English translation (shows ONLY English)
./query.sh "Kiu estas Frodo?" --translate
./ask.sh "Kiu estas Frodo?" --translate
# Output: Only English translation
```

### Debug Mode

```bash
# Show all logging and debug info
./query.sh "Kiu estas Frodo?" --debug
./ask.sh "Kiu estas Frodo?" --debug
```

### Additional Options

**Only for `./query.sh`:**
```bash
# Show keyword filtering details (Stage 1 & 2 retrieval)
./query.sh "Kiu estas Frodo?" --show-stage1
```

**Only for `./ask.sh` and `./ask-external.sh`:**
```bash
# Specify number of context documents to use (default: 30)
./ask.sh "Kiu estas Frodo?" -k 50

# Use fewer for faster responses
./ask.sh "Kiu estas Frodo?" -k 10
```

## Detailed Command Reference

### `./query.sh` - Extractive Retrieval (LOCAL)

**Full syntax:**
```bash
./query.sh [QUERY] [OPTIONS]
```

**Options:**
- `--translate` - Show ONLY English (translates output after processing)
- `--show-stage1` - Show keyword filtering details
- `--debug` - Enable debug logging

**Examples:**
```bash
# Basic query
./query.sh "Kiu estas Frodo?"

# With all options
./query.sh "Kiu estas Frodo?" --translate --show-stage1 --debug

# Default query (if no argument)
./query.sh
# Uses: "Kiu estas Gandalfo?"
```

**What it does:**
1. Parses query to AST
2. Retrieves top 5 most relevant passages (hybrid search with LOCAL GNN)
3. Shows best matching sentence as answer
4. Lists top 3 sources with similarity scores

**Models used:**
- ✅ LOCAL: Tree-LSTM GNN encoder (512-dim embeddings)
- ✅ LOCAL: FAISS index (20,985 Esperanto sentences)
- ✅ LOCAL: Esperanto parser (AST-based)

### `./ask.sh` - Generative Answering (LOCAL QA Decoder)

⭐ **PREFERRED METHOD - Uses your trained local model**

**Full syntax:**
```bash
./ask.sh [QUERY] [OPTIONS]
```

**Options:**
- `-k N` - Number of context documents to retrieve (default: 5)
- `--translate` - Show ONLY English (translates output after generation)
- `--debug` - Enable debug logging

**Examples:**
```bash
# Basic query (uses local QA Decoder)
./ask.sh "Kiu estas Frodo?"

# With more context
./ask.sh "Kiu estas Frodo?" -k 10

# English output
./ask.sh "Kiu estas Frodo?" --translate

# All options
./ask.sh "Kiu estas Frodo?" -k 20 --translate --debug

# Default query (if no argument)
./ask.sh
# Uses: "Kiu estas Frodo?"
```

**What it does:**
1. Parses query to AST
2. Retrieves top k relevant passages (hybrid search with LOCAL GNN)
3. Generates answer using LOCAL QA Decoder (8-layer transformer)
4. Shows answer + sources

**Models used:**
- ✅ LOCAL: QA Decoder (417MB, 8 layers, ~15M parameters)
- ✅ LOCAL: Tree-LSTM GNN encoder
- ✅ LOCAL: Vocabulary (74,000+ tokens)

**Output format:**
```
📚 RETRIEVED CONTEXT (k documents)
   1. [score] Source name
      Text snippet...

🤖 Generating answer with local QA Decoder...
✓ Generated N tokens

💬 GENERATED ANSWER (Local QA Decoder)
   [Natural language answer from local model]
```

### `./ask-external.sh` - Generative Answering (⚠️ EXTERNAL LLM - STOPGAP)

⚠️ **TEMPORARY FALLBACK - Uses external Claude Code LLM**

This is a **stopgap measure** only. Use `./ask.sh` (local) instead.

**Full syntax:**
```bash
./ask-external.sh [QUERY] [OPTIONS]
```

**Options:**
- `-k N` - Number of context documents to retrieve (default: 5)
- `--translate` - Show ONLY English (translates output after generation)
- `--debug` - Enable debug logging

**Examples:**
```bash
# Only use if local model fails
./ask-external.sh "Kiu estas Frodo?"
```

**What it does:**
1. Parses query to AST
2. Retrieves top k relevant passages (hybrid search)
3. ⚠️ **Sends context to EXTERNAL Claude Code LLM (interactive)**
4. ⚠️ **Waits for Claude to respond (you must answer)**
5. Shows answer + sources

**Models used:**
- ✅ LOCAL: GNN encoder for retrieval
- ⚠️ **EXTERNAL: Claude Code LLM (stopgap)**

**Output format:**
```
📚 RETRIEVED CONTEXT (k documents)
   1. [score] Source name
      Text snippet...

⚠️ EXTERNAL LLM REQUEST (STOPGAP - Local model preferred)
🤖 LLM REQUEST FOR CLAUDE CODE
   [Shows prompt with context]
   [Wait for Claude to respond - INTERACTIVE]

💬 GENERATED ANSWER (External LLM)
   [Answer synthesized by external LLM]
```

## Option Comparison Table

| Option | `./query.sh` | `./ask.sh`<br>(LOCAL) | `./ask-external.sh`<br>(STOPGAP) | Behavior |
|--------|--------------|----------------------|----------------------------------|----------|
| *(no flags)* | ✅ | ✅ | ✅ | Pure Esperanto output |
| `--translate` | ✅ | ✅ | ✅ | ONLY English output |
| `--debug` | ✅ | ✅ | ✅ | Show all logging |
| `--show-stage1` | ✅ | ❌ | ❌ | Show keyword filtering |
| `-k N` | ❌ | ✅ | ✅ | Number of context docs |

## Model Comparison Table

| Feature | `./query.sh`<br>(LOCAL) | `./ask.sh`<br>(LOCAL) | `./ask-external.sh`<br>(STOPGAP) |
|---------|------------------------|----------------------|----------------------------------|
| Speed | Fast (~5 sec) | Medium (~8 sec) | Slower (~15 sec) |
| Answer Type | Direct quotes | Synthesized | Synthesized |
| Model Type | ✅ LOCAL GNN | ✅ LOCAL QA Decoder | ⚠️ EXTERNAL LLM |
| Automatic | Yes | Yes | No (interactive) |
| Requires External Service | No | No | ⚠️ Yes (Claude Code) |
| Best For | Finding passages | Getting answers | Fallback only |

## Translation Behavior (Consistent!)

All commands handle `--translate` **identically**:

### Default Mode (No Translation)
```bash
$ ./query.sh "Kiu estas Frodo?"
💬 RESPONSE
Laŭ la trovita teksto:
"— Tio plaĉas sufiĉe, se iam okazos tiel, — diris Frodo."

$ ./ask.sh "Kiu estas Frodo?"
💬 GENERATED ANSWER (Local QA Decoder)
Frodo estas hobito, kiu heredis la Unu Ringon...
```

### Translation Mode (--translate)
```bash
$ ./query.sh "Kiu estas Frodo?" --translate
💬 RESPONSE
According to the found text:
"It's nice enough if it ever happens like that, said Frodo."

$ ./ask.sh "Kiu estas Frodo?" --translate
💬 GENERATED ANSWER (Local QA Decoder)
Frodo is a hobbit who inherited the One Ring...
```

**Key point:** All show **ONLY** the target language (Esperanto or English), never both.

## Debug Mode

All commands suppress logging by default. Use `--debug` to see full pipeline logs:

```bash
$ ./ask.sh "Kiu estas Frodo?" --debug
2025-11-14 11:30:00 - INFO - Loading QA Decoder...
2025-11-14 11:30:01 - INFO - Loading GNN encoder...
2025-11-14 11:30:02 - INFO - Generating answer...
...
```

Without `--debug`, output is clean:
```bash
$ ./ask.sh "Kiu estas Frodo?"
======================================================================
KLARECO - LOCAL QA DECODER (Fully Automatic)
======================================================================
...
```

## Recommended Workflows

### Standard Workflow (All Local)

```bash
# 1. First, see what's in the corpus
./query.sh "Kiu estas Frodo?" --show-stage1

# 2. Generate a synthesized answer with LOCAL model
./ask.sh "Kiu estas Frodo?" -k 10
```

### Fallback Workflow (If Local Fails)

```bash
# 1. Try local first
./ask.sh "Kiu estas Frodo?" -k 10

# 2. If local model gives poor results, try external (stopgap)
./ask-external.sh "Kiu estas Frodo?" -k 10
```

### Save Output

```bash
# Save Esperanto output
./query.sh "Kiu estas Frodo?" > frodo_eo.txt
./ask.sh "Kiu estas Frodo?" > frodo_answer_eo.txt

# Save English output
./query.sh "Kiu estas Frodo?" --translate > frodo_en.txt
./ask.sh "Kiu estas Frodo?" --translate > frodo_answer_en.txt
```

### Batch Processing (All Local)

```bash
# Generate answers for multiple questions (local)
for q in "Kiu estas Frodo?" "Kiu estas Gandalfo?" "Kio estas hobito?"; do
    echo "Question: $q"
    ./ask.sh "$q" -k 10
    echo ""
    echo "---"
    echo ""
done
```

### Compare Modes

```bash
# Compare extractive vs generative (both local)
echo "=== EXTRACTIVE (LOCAL) ==="
./query.sh "Kiu estas Frodo?"

echo ""
echo "=== GENERATIVE (LOCAL QA DECODER) ==="
./ask.sh "Kiu estas Frodo?"
```

## When to Use Each

### Use `./query.sh` (Local Extractive) When:
- ✅ You want exact quotes
- ✅ You need fast results (~5 sec)
- ✅ You're verifying source material
- ✅ You want to see keyword matching
- ✅ You need guaranteed accuracy (no synthesis)

### Use `./ask.sh` (Local Generative) When:
- ✅ You want a direct answer
- ✅ You need information synthesized
- ✅ You want natural language
- ✅ Question needs interpretation
- ✅ You want fully automatic operation

### Use `./ask-external.sh` (External Stopgap) When:
- ⚠️ Local QA Decoder gives poor results
- ⚠️ You need very high quality synthesis temporarily
- ⚠️ You're okay with interactive mode
- ⚠️ You understand this is a temporary fallback

## Environment

All scripts work with the same local environment:
- **Corpus:** `data/corpus_index/` (20,985 Esperanto sentences)
- **GNN Model:** `models/tree_lstm/checkpoint_epoch_20.pt` (LOCAL)
- **QA Decoder:** `models/qa_decoder/best_model.pt` (LOCAL, 417MB)
- **Parser:** Pure Esperanto AST parser (LOCAL)
- **Retriever:** Hybrid (keyword + semantic, LOCAL)

## Troubleshooting

### "No results found"
```bash
# Check what keywords are extracted
./query.sh "YOUR_QUERY" --show-stage1

# Try with more context documents (local)
./ask.sh "YOUR_QUERY" -k 20
```

### Local model gives poor answers
```bash
# Try with more context
./ask.sh "YOUR_QUERY" -k 20 --debug

# Fallback to external (stopgap)
./ask-external.sh "YOUR_QUERY" -k 10
```

### Translation issues
```bash
# Use debug to see what's happening
./ask.sh "YOUR_QUERY" --translate --debug
```

### External LLM not responding (ask-external.sh)
When you see:
```
⚠️ EXTERNAL LLM REQUEST (STOPGAP)
🤖 LLM REQUEST FOR CLAUDE CODE
...
```

Respond in the conversation with the answer in Esperanto based on the provided context.

**Note:** This is temporary. Use `./ask.sh` (local) instead.

## Quick Reference Card

```
┌────────────────────────────────────────────────────────────────┐
│ KLARECO COMMAND QUICK REFERENCE (LOCAL-FIRST)                  │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│ ./query.sh "QUERY"              ✅ LOCAL extractive            │
│ ./query.sh "QUERY" --translate  ✅ LOCAL extractive (English)  │
│ ./query.sh "QUERY" --show-stage1  Show keyword matching        │
│                                                                 │
│ ./ask.sh "QUERY"                ✅ LOCAL QA Decoder            │
│ ./ask.sh "QUERY" --translate    ✅ LOCAL QA Decoder (English)  │
│ ./ask.sh "QUERY" -k 10          Use 10 context docs            │
│                                                                 │
│ ./ask-external.sh "QUERY"       ⚠️ EXTERNAL LLM (stopgap)     │
│                                                                 │
│ All: --debug                    Show all logging               │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

## Summary

**Klareco uses LOCAL MODELS as the primary approach:**
- Default: Pure Esperanto, LOCAL models
- `--translate`: ONLY English, LOCAL models
- `--debug`: Full logging
- External LLMs: ⚠️ STOPGAP only

**The difference between commands:**
- `./query.sh`: Shows passages (LOCAL GNN retrieval)
- `./ask.sh`: Synthesizes answers (LOCAL QA Decoder) ⭐ **PREFERRED**
- `./ask-external.sh`: Synthesizes answers (EXTERNAL LLM) ⚠️ **STOPGAP**

**Remember:** Always prefer local models. External LLMs are temporary fallbacks.
