# Quick Query Test Guide

## Your Corpus Content

Currently indexed: **20,985 sentences** from literary Esperanto translations

### Available Sources:
1. **Lord of the Rings** (`la_mastro_de_l_ringoj`) - 12,718 sentences
2. **The Hobbit** (`la_hobito`) - 5,600 sentences  
3. **Six Stories** (`ses_noveloj`) - 1,055 sentences
4. **Edgar Allan Poe** - 1,612 sentences:
   - Kadavrejo Strato (Murders in the Rue Morgue)
   - Puto kaj Pendolo (The Pit and the Pendulum)
   - La Korvo (The Raven)
   - Usxero Domo (The Fall of the House of Usher)

**Note:** No Wikipedia or non-literary content in corpus yet.

---

## Quick Test Commands

### 1. Simple Query (See Retrieval + Generation)

```bash
python scripts/test_end_to_end_qa.py --index data/corpus_index
```

Tests these 3 questions:
- "Kiu estas Frodo?" (Who is Frodo?)
- "Kio estas hobito?" (What is a hobbit?)
- "Kie estas Hobbiton?" (Where is Hobbiton?)

Shows:
- ✓ What context was retrieved (3 sentences)
- ✓ Which source each sentence came from
- ✓ What the QA Decoder generated

### 2. Custom Questions (Tolkien Content)

Ask about hobbits, wizards, rings:

```bash
# Test single question
python scripts/test_end_to_end_qa.py --index data/corpus_index

# Then edit the test_questions list in the script to add your own
```

**Example questions that will retrieve Tolkien content:**
- "Kiu estas Gandalf?" (Who is Gandalf?)
- "Kio estas la Unu Ringo?" (What is the One Ring?)
- "Kie loĝas hobitoj?" (Where do hobbits live?)
- "Kio okazis en Hobbiton?" (What happened in Hobbiton?)

### 3. Edgar Allan Poe Content

Ask about houses, darkness, fear:

```bash
# Edit test_end_to_end_qa.py with these questions:
test_questions = [
    "Kio okazis en la domo?",        # What happened in the house?
    "Kio estas la korvo?",            # What is the raven?
    "Kio estas mallumo?",             # What is darkness?
]
```

---

## Understanding Output

When you run a test, you'll see:

```
Question: Kiu estas Frodo?
✓ Parsed question to AST
✓ Retrieved 3 context sentences          ← RAG retrieval worked
✓ Parsed 3 context ASTs                  ← Context parsed
✓ Encoded question and context with GNN  ← GNN encoding worked
✓ Generated answer tokens: [...]         ← QA Decoder generated

Generated: adjektivo PLURAL malagrabl...  ← Current output (needs tuning)
Context sentences: 3
```

**Current Generation Quality:**
- Model generates repetitive token patterns
- This is expected for first-iteration models
- Infrastructure is working correctly
- Generation quality improves with:
  - More training data
  - Better hyperparameters
  - Dataset diversity

---

## What Each Question Type Retrieves

### Tolkien Questions → Tolkien Content
Questions with: hobito, Gandalf, ringo, Ŝiro, etc.
→ Retrieves from `la_mastro_de_l_ringoj` and `la_hobito`

### Gothic/Mystery Questions → Poe Content  
Questions with: domo, korvo, mallumo, morto, timo, etc.
→ Retrieves from Poe stories

### General Questions
Questions about emotions, objects, actions
→ May retrieve from any source depending on semantic similarity

---

## Next Steps

To add non-literary or Wikipedia content:
1. Download Esperanto Wikipedia dumps
2. Process into sentence-level JSONL
3. Re-run indexing: `python scripts/index_corpus.py`

See `RAG_STATUS.md` for full RAG system documentation.
