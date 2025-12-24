# Klareco Quick Start Guide

**Updated**: 2025-11-27

## What to Do Right Now

### Option 1: Run Improved Corpus Builder (Background Task)

This is running right now or you can start it:

```bash
python scripts/build_corpus_v2.py
```

**What it does**:
- Processes Wikipedia without freezing (0.1s throttle)
- Filters out English sections
- Keeps ALL Esperanto sentences (min-parse-rate=0.0)
- Shows progress every 20 sentences
- Resumes if interrupted

**Expected**: ~45-50k sentences (vs 35k before), completes in 1-3 hours

---

## What to Do Next (This Week)

### Day 1-2: Proper Noun Dictionary (6 hours)

**Why**: Fixes parse rates from 91.8% to 95%+ instantly

**Step 1**: Create extraction script
```bash
# Create scripts/build_proper_noun_dict.py
# (See IMPLEMENTATION_ROADMAP_V2.md for full code)
```

**Step 2**: Run extraction
```bash
python scripts/build_proper_noun_dict.py \
  --corpus data/corpus_with_sources_v2.jsonl \
  --output data/proper_nouns_static.json \
  --min-frequency 3
```

**Step 3**: Integrate into parser
```python
# Add to parser.py in categorize_unknown_word()
if word[0].isupper():
    if proper_noun_dict.is_proper_noun(word):
        ast["parse_status"] = "success"  # No longer "failed"!
```

**Result**: Parse rates jump to 95%+, proper nouns recognized

---

### Day 3-5: Semantic Retrieval (12 hours)

**Why**: Most impressive improvement for least effort!

**What you get**:
```
Query: "Kiu vidas la katon?" (Who sees the cat?)

BEFORE: Returns ANY sentence with "vid" + "kat"
  ❌ "La kato vidas la hundon." (cat is seeing - WRONG!)

AFTER: Returns ONLY sentences where cat is PATIENT
  ✅ "La hundo vidas la katon." (cat is being seen - RIGHT!)
```

**Steps**:

1. **Implement signature extraction** (3 hours)
   - Create `klareco/semantic_signatures.py`
   - Extract (agent, action, patient) from AST
   - ~30 lines of code!

2. **Build semantic index** (4 hours)
   - Create `scripts/build_semantic_index.py`
   - Index corpus by semantic signatures
   - Outputs ~20k unique signatures

3. **Add semantic search** (3 hours)
   - Create `klareco/semantic_search.py`
   - Query with wildcards (None = match anything)
   - ~30 lines of code!

4. **Integrate & demo** (2 hours)
   - Add to retriever as Stage 1.5
   - Create demo script showing improvement
   - Show users the magic!

**Result**: Role-based retrieval working, impressive demos

---

## Week 2-3: Compositional Embeddings (Optional but Awesome)

**Why**: 75% parameter reduction (1.28M → 320K)

**What you get**:
- Smaller models (faster training, inference)
- Better generalization (handles unseen word forms)
- Interpretable dimensions (can probe what model knows)

**Steps**: See `IMPLEMENTATION_ROADMAP_V2.md` Day 6-15

---

## Decision Tree: What Should I Work On?

### If you want IMPRESSIVE RESULTS FAST:
→ **Do Proper Nouns (Day 1-2) + Semantic Retrieval (Day 3-5)**
- Total: 1 week
- Impact: 🔥🔥🔥 (Users love it!)
- Complexity: ⭐ (Easy)

### If you want SMALLER, FASTER MODELS:
→ **Do Compositional Embeddings (Week 2-3)**
- Total: 2 weeks
- Impact: 🔥🔥 (75% fewer params)
- Complexity: ⭐⭐ (Medium)

### If you want to PROVE THE THESIS:
→ **Do AST Reasoning (Week 4-8)**
- Total: 4 weeks
- Impact: 🔥🔥🔥 (Revolutionary!)
- Complexity: ⭐⭐⭐ (Hard)

### If you want ALL OF THE ABOVE:
→ **Follow IMPLEMENTATION_ROADMAP_V2.md in order**
- Total: 8 weeks
- Impact: Complete Esperanto-first AI system
- Demonstrates that structure beats learned weights

---

## Key Documents

### Design & Analysis
- `CORPUS_AND_AST_AUDIT.md` - Current system is good! Don't rebuild.
- `QUICK_WINS_ANALYSIS.md` - Why semantic retrieval first
- `COMPOSITIONAL_EMBEDDINGS.md` - Why Esperanto = free dimensions
- `ESPERANTO_FIRST_IMPLEMENTATION_PLAN.md` - Original detailed plan

### Implementation
- `IMPLEMENTATION_ROADMAP_V2.md` - **Updated roadmap (READ THIS!)**
- `CORPUS_BUILD_IMPROVEMENTS.md` - Corpus builder changes

### Status
- `CORPUS_V2_RESULTS.md` - Current corpus quality
- `README.md` - System overview

---

## Success Metrics

### After Week 1 (Semantic Retrieval)
- ✅ Parse rate > 95%
- ✅ Semantic retrieval working
- ✅ Demo shows role disambiguation
- ✅ Users impressed

### After Week 3 (Compositional Embeddings)
- ✅ 75% parameter reduction
- ✅ Quality maintained
- ✅ Compositional generalization

### After Week 8 (AST Reasoning)
- ✅ Reasoning patterns working
- ✅ 80% accuracy on factoid questions
- ✅ All reasoning explainable via AST trail

---

## The Big Picture

### Traditional LLM
```
50K vocab × 256 dims = 12.8M parameters
Learns everything (including grammar)
```

### Klareco (Current)
```
10K vocab × 128 dims = 1.28M parameters
Parser handles grammar deterministically
```

### Klareco (With Compositional Embeddings)
```
5K roots × 64 dims + 38 affixes × 8 dims + 16 programmatic dims
= 320K parameters (75% reduction!)
Grammar is FREE (16 dimensions, 0 parameters)
```

### Klareco (Full Vision)
```
Compositional embeddings (320K)
+ Semantic retrieval (0 params, deterministic!)
+ AST reasoning (0-5M params for patterns)
+ Tree-LSTM encoder (15M params)
= ~20M total parameters

Competes with 110M+ LLMs through structural efficiency!
```

---

## Philosophy Recap

**We're not building a smaller LLM.**

**We're building a different architecture** that uses Esperanto's regularity to:
- Replace learned attention with deterministic role extraction
- Replace massive embeddings with compositional morphemes
- Replace black-box reasoning with interpretable AST operations

**80% of the work is deterministic structure.**
**20% is tiny learned components where structure can't help.**

**The AST is the consciousness** - every thought happens in explicit Esperanto structures we can inspect and understand.

---

## Next Command to Run

```bash
# If corpus builder is done, extract proper nouns:
python scripts/build_proper_noun_dict.py \
  --corpus data/corpus_with_sources_v2.jsonl \
  --output data/proper_nouns_static.json \
  --min-frequency 3

# If still running, let it finish, then start on proper nouns tomorrow!
```

**Let's build Esperanto-first AI!** 🚀
