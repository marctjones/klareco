# Quick Query Usage Guide

## Overview

The `quick_query.py` script is designed for **pure Esperanto processing** with optional English translations of the OUTPUT for readability.

**Key Design Principles:**
1. ✅ **Default:** Pure Esperanto (no translation overhead)
2. ✅ **Optional `--translate`:** Show English translations of output
3. ✅ **Never translates input** during processing (keeps pipeline pure)
4. ✅ **Translations only for display** (doesn't affect retrieval or models)

## Usage

### Pure Esperanto Mode (Default - Recommended)

```bash
# Using the script directly
python scripts/quick_query.py "Kiu estas Frodo?"

# Using the wrapper (easier)
./query.sh "Kiu estas Frodo?"
```

**Output:**
```
======================================================================
KLARECO QUICK QUERY - Pure Esperanto Processing
======================================================================

📝 Query: Kiu estas Frodo?
   [Pure Esperanto mode - use --translate to show English]

🔄 PIPELINE STAGES
----------------------------------------------------------------------
2. 🌍 Language: Esperanto ✓
3. 🌲 Parsing → AST created
5. 🎯 Intent: factoid_question
   Stage 1: 1027 keyword matches
   Stage 2: Reranked top 100
   Final: 5 results

======================================================================
💬 RESPONSE
======================================================================

Laŭ la trovita teksto:
"— Tio plaĉas sufiĉe, se iam okazos tiel, — diris Frodo."

(Fonto: La Mastro de l' Ringoj (Lord of the Rings), simileco: 0.588)

======================================================================
📚 SOURCES
======================================================================

1. [0.588] La Mastro de l' Ringoj (Lord of the Rings):?
   — Tio plaĉas sufiĉe, se iam okazos tiel, — diris Frodo.

2. [0.582] La Mastro de l' Ringoj (Lord of the Rings):?
   Do ĝojiĝu, Frodo!
```

### With English Translations (--translate)

For quick readability when you want to see what the Esperanto output means:

```bash
python scripts/quick_query.py "Kiu estas Frodo?" --translate
# or
./query.sh "Kiu estas Frodo?" --translate
```

**Output:**
```
📝 Query: Kiu estas Frodo?
   [English translations enabled for OUTPUT]

💬 RESPONSE
======================================================================

Laŭ la trovita teksto:
→ According to the found text:
"— Tio plaĉas sufiĉe, se iam okazos tiel, — diris Frodo."
→ "It's nice enough if it ever happens like that, said Frodo."

(Fonto: La Mastro de l' Ringoj (Lord of the Rings), simileco: 0.588)
→ (Source: The Master of the Rings (Lord of the Rings), similarity: 0.588)
```

**Note:** Opus-MT translations are approximate. Use `--translate` for quick understanding, not for production.

### Show Keyword Filtering Details (--show-stage1)

See how the two-stage retrieval works:

```bash
./query.sh "Kiu estas Frodo?" --show-stage1
```

Shows:
- Stage 1 keywords extracted from query
- Total candidates found by keyword matching
- Candidates reranked semantically
- Top 10 keyword matches before semantic reranking

### Debug Mode (--debug)

See full pipeline logging:

```bash
./query.sh "Kiu estas Frodo?" --debug
```

## Command Reference

### Basic Usage

```bash
# Default query
./query.sh

# Custom query (pure Esperanto)
./query.sh "Kiu estas Gandalfo?"

# With English translations
./query.sh "Kiu estas Gandalfo?" --translate

# Show keyword filtering
./query.sh "Kiu estas Gandalfo?" --show-stage1

# Combine options
./query.sh "Kiu estas Gandalfo?" --translate --show-stage1

# Debug mode
./query.sh "Kiu estas Gandalfo?" --debug
```

### Full Options

```
python scripts/quick_query.py [OPTIONS] [QUERY]

Arguments:
  QUERY                 Query to process (default: "Kiu estas Gandalfo?")

Options:
  --translate           Translate OUTPUT to English (does NOT affect processing)
  --show-stage1         Show Stage 1 keyword filter results
  --debug               Show debug logging
  -h, --help            Show help message

Examples:
  quick_query.py "Kiu estas Frodo?"              # Pure Esperanto output
  quick_query.py "Kiu estas Frodo?" --translate  # Show English translations
  quick_query.py --show-stage1                   # Show keyword filtering
```

## Why Pure Esperanto by Default?

### Advantages

1. **No Translation Overhead**
   - Faster processing (no MT model loading)
   - Pure Esperanto pipeline (no translation errors)
   - See exactly what the system is working with

2. **Avoids Translation Issues**
   - Input translation can introduce errors
   - Opus-MT Esperanto→English is imperfect
   - Better to work in native language

3. **True System Behavior**
   - See how Klareco actually processes queries
   - Understand AST structure directly
   - Debug issues in the source language

4. **Better for Development**
   - Focus on Esperanto grammar/semantics
   - Catch parser issues early
   - Verify retrieval quality directly

### When to Use --translate

✅ **Good use cases:**
- Quick readability check
- Showing results to non-Esperanto speakers
- Understanding general meaning of output
- Demos and presentations

❌ **Avoid for:**
- Production use (translation quality varies)
- Debugging (use pure Esperanto)
- Performance testing (translation adds overhead)
- Accuracy evaluation (translations can be wrong)

## Example Queries

### Character Questions
```bash
./query.sh "Kiu estas Frodo Baginzo?"
./query.sh "Kiu estas Gandalfo la Griza?"
./query.sh "Kiu estas Bilbo?"
./query.sh "Kiu estas Saurono?"
```

### What Is Questions
```bash
./query.sh "Kio estas hobito?"
./query.sh "Kio estas la Unu Ringo?"
./query.sh "Kio estas Mez-Tero?"
./query.sh "Kio estas palantiro?"
```

### Where Questions
```bash
./query.sh "Kie estas Hobbiton?"
./query.sh "Kie estas Mordor?"
./query.sh "Kie estas Rivendel?"
```

### Complex Queries
```bash
./query.sh "Kial Frodo kaj Bilbo estas specialaj?" --translate
./query.sh "Kiel Gandalfo helpis Frodon?" --show-stage1
```

## Understanding the Output

### Pipeline Stages

1. **Language Detection**
   - `Language: Esperanto ✓` - Input already in Esperanto
   - `Language Detection: en → eo` - Translated from English

2. **Parsing**
   - `Parsing → AST created` - Esperanto parsed to AST structure

3. **Intent Classification**
   - Shows detected intent (factoid_question, calculation_request, etc.)
   - Stage 1: Keyword matches (e.g., "1027 keyword matches")
   - Stage 2: Semantic reranking (e.g., "Reranked top 100")
   - Final: Results returned (e.g., "5 results")

### Response Section

Shows the answer generated by the expert system:
- **Laŭ la trovita teksto:** - "According to the found text:"
- **Quote** - Relevant sentence from corpus
- **Fonto** - Source (e.g., Lord of the Rings)
- **Simileco** - Similarity score (0.0-1.0)

### Sources Section

Top 3 retrieved documents with:
- **Similarity score** - [0.588]
- **Source** - La Mastro de l' Ringoj
- **Text** - Retrieved sentence
- **Count** - "... and 2 more" if more than 3

## Performance Tips

### Faster Queries

```bash
# Default mode (fastest)
./query.sh "Kiu estas Frodo?"

# Avoid --translate unless needed (saves ~2-3 seconds)
# Avoid --debug unless debugging (reduces logging overhead)
```

### Better Results

```bash
# Use --show-stage1 to verify keyword matching
./query.sh "Kiu estas Frodo?" --show-stage1

# Check if query uses good keywords (names, specific terms)
# Example: "Frodo" is an excellent keyword (appears 1027 times)
```

## Troubleshooting

### No Results

```bash
# Check if query is valid Esperanto
./query.sh "Kiu estas XYZ?" --debug

# See what keywords were extracted
./query.sh "Kiu estas XYZ?" --show-stage1
```

### Poor Quality Results

```bash
# Check Stage 1 candidates
./query.sh "YOUR_QUERY" --show-stage1

# Verify similarity scores (should be > 0.5 for good matches)
# Low scores (< 0.3) indicate weak semantic match
```

### Translation Issues

```bash
# Don't rely on --translate for accuracy
# Use it only for quick understanding

# For better understanding, learn basic Esperanto:
# - estas = is
# - kaj = and
# - la = the
# - kiu = who
# - kio = what
# - kie = where
```

## Advanced Usage

### Batch Queries

```bash
# Create a script with multiple queries
for query in \
    "Kiu estas Frodo?" \
    "Kiu estas Gandalfo?" \
    "Kio estas hobito?"; do
    echo ""
    echo "======================================"
    echo "Query: $query"
    echo "======================================"
    ./query.sh "$query"
    sleep 1
done
```

### Save Results

```bash
# Save pure Esperanto output
./query.sh "Kiu estas Frodo?" > results_eo.txt

# Save with translations
./query.sh "Kiu estas Frodo?" --translate > results_en.txt

# Save both versions
./query.sh "Kiu estas Frodo?" | tee results_eo.txt
./query.sh "Kiu estas Frodo?" --translate | tee results_en_eo.txt
```

### Integration

```python
# Use in Python scripts
from scripts.quick_query import main
import sys

# Override arguments
sys.argv = ['quick_query.py', 'Kiu estas Frodo?', '--translate']
main()
```

## Summary

**Default behavior (recommended):**
```bash
./query.sh "Kiu estas Frodo?"
```
- Pure Esperanto processing
- No translation overhead
- See true system behavior

**Optional English output:**
```bash
./query.sh "Kiu estas Frodo?" --translate
```
- Translates OUTPUT only (not input)
- Good for quick understanding
- Not for production/accuracy

**Best practice:** Learn basic Esperanto for better understanding, use `--translate` sparingly for convenience.
