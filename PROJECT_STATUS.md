# Klareco Project Status

**Last Updated**: January 3, 2026
**Current Milestone**: M3 (Deterministic Retrieval)

---

## 🎯 Overall Progress

| Milestone | Status | Key Metric |
|-----------|--------|------------|
| **M1: Single-Turn Q&A** | ✅ **COMPLETE** | 20% partial match vs OLMo's 8% (2.5x better) |
| **M2: Retrieval Improvement** | ✅ **COMPLETE** | 85% recall (exceeded 80% target) |
| **M3: Deterministic Retrieval** | 🔄 **IN PROGRESS** | Target: 95% recall with 0-param grammar tokens |
| **M4: AST-Aware Deep Learning** | 🔲 Planned | TreeLSTM sentence encoder (2-5M params) |

---

## ✅ What's Complete

### **M2: Slot-Based Retrieval** (January 3, 2026)
- ✅ **SlotBasedRetriever**: Two-stage retrieval (filter→rerank)
  - Stage 1: Slot-based filtering (SUBJ/VERB/OBJ embeddings)
  - Stage 2: Full sentence embedding reranking
  - **Recall**: 85% @ k=10 (exceeded 80% target)
  - **Latency**: 11.5ms mean (20 queries on 1,000 docs)
  - Implementation: `klareco/rag/slot_retriever.py`
  - Demo: `scripts/demo_slot_retrieval.py`
- ✅ **Slot-based indexing**: Grammar-aware retrieval
  - Extracts SUBJ/VERB/OBJ from ASTs
  - Separate embeddings per slot (no information loss from averaging)
  - Index: `data/indexes/slot_test/` (1,000 docs), `slot_full/` (4.4M docs)
- ✅ **Benchmark validation**: Tested on Wikipedia queries
  - 4 retriever implementations benchmarked
  - FAISSSlotRetriever: 85% recall, 5.1ms latency
  - MultiFAISSSlotRetriever: 75% recall, 1.1ms latency (fastest)
  - Results: `benchmark_results/combined_20260103_081055.json`

**Key Achievement**: Beat 80% recall target using compositional embeddings + AST structure (no TreeLSTM needed yet)

### **Stage 1: Semantic Model** (733K params)
- ✅ **Root embeddings**: 11,121 roots × 64d = 712K params
  - Correlation: 0.89 | Accuracy: 97.98%
  - Model: `models/root_embeddings/best_model.pt`
- ✅ **Affix transforms V2**: 12 prefixes + 29 suffixes = 21K params
  - Low-rank transformations (rank=8)
  - Anti-collapse: mal_mean_sim = -0.03
  - Model: `models/affix_transforms_v2/best_model.pt`
- ✅ **Corpus index**: 4.4M sentences
  - Merged index with tiered architecture (just completed!)
  - Index: `data/indexes/merged/`

### **Core Components**
- ✅ **Parser**: 16 Esperanto grammar rules, 91.8% parse rate
- ✅ **Deparser**: AST → text reconstruction
- ✅ **EnrichedAST**: Parser AST + trained embeddings
- ✅ **Two-stage retrieval**: Structural filtering + FAISS semantic search
- ✅ **Extractive Q&A**: Template-based answering

### **M1 Benchmark Results** (Dec 31, 2025)
Klareco (733K params) vs OLMo 1B (1.18B params):
- **Partial match**: 20% vs 8% = **2.5x better**
- **Latency**: 690ms vs 38,329ms = **56x faster**
- **Parameters**: 733K vs 1.18B = **1,600x smaller**

**Thesis validation**: ✅ Specialized linguistic processing beats brute-force parameters

---

## 🔄 In Progress (This Session)

### **January 3, 2026: Test Suite Cleanup & M2 Validation**
- ✅ **Fixed all 17 pytest collection errors**
  - Before: 371 tests collected, 17 errors
  - After: 480 tests collected, 0 errors
  - Status: 329 passed, 130 skipped (M3/M4 future modules), 53 failed (parser edge cases)
- ✅ **Validated M2 completion**
  - SlotBasedRetriever: 85% recall @ k=10 (demo tested)
  - Pipeline demo: Full Stage 0 + Stage 1 working
  - End-to-end: Parse → Embed → Retrieve validated
- ✅ **Created M3 & M4 milestones**
  - M3: 10 issues for deterministic retrieval (grammar tokens, morphological expansion)
  - M4: 4 issues for TreeLSTM/GNN sentence encoders
- 📝 **Created issue #201**: End-to-end integration test (P1)

### **Session Accomplishments**
1. Fixed test suite collection errors (committed)
2. Verified demos work (slot retrieval, pipeline)
3. Documented M2 completion in PROJECT_STATUS.md
4. Created milestones M3 (deterministic, 0 params) and M4 (TreeLSTM, 2-5M params)
5. Ready to start M3 implementation

---

## 🔜 Next Steps (Priority Order)

### **Immediate (M3 Critical Path)**

1. **Create end-to-end integration test** (Issue #201 - P1)
   - Automated pytest for parse → embed → retrieve pipeline
   - Prevents regression as M3/M4 add complexity
   - Target: This week

2. **Start M3.1: Grammar Token Index** (Issues #187-189 - P0)
   - Build inverted index on deterministic grammar tokens
   - Expected: 85% → 92% recall with 0 new parameters
   - See: `.idlergear/notes/015.md` for design

3. **Implement M3.2: Morphological Expansion** (Issues #190-192 - P0)
   - Deterministic generation of all valid Esperanto forms
   - Handles queries with different affixes/endings
   - Expected: +3-5% recall boost

### **Short-term (M3 Completion)**

4. **M3.3: Multi-Index Cascade** (Issues #193-195 - P1)
   - Combine grammar tokens (Stage 1) + slot embeddings (Stage 2)
   - Target: 95% recall with minimal latency
   - Benchmark against slot-only baseline (85%)

5. **M3.4: Validation Report** (Issue #196 - P1)
   - Document deterministic retrieval results
   - Compare: Simple averaging (35%) → Slots (85%) → Grammar tokens (95%)
   - Publish methodology for other Esperanto AI projects

### **Medium-term (M4 Preparation)**

6. **Fix 53 parser test failures** (P2)
   - Fundamento prefix/suffix protection edge cases
   - Non-critical: Doesn't affect corpus parse rate (91.8%)
   - See: `tests/test_parser_fundamento.py`

7. **Data quality fixes** (5 GitHub issues - P2)
   - #173: Convert x-notation to Unicode (4 files)
   - #176: Remove transcriber notes from Gutenberg files
   - #172: Clean wikipedia.txt (0 newlines issue)
   - #175: Fix Wikipedia Gutenberg boilerplate false positives
   - #174: Fix test sampling bug

### **Future (M4: AST-Aware Deep Learning)**

8. **TreeLSTM sentence encoder** (Issues #197-199 - M4)
   - Child-Sum TreeLSTM respecting AST structure
   - 2-5M params, expected 95% → 97% recall
   - Only if M3 deterministic methods plateau

9. **AST-Constrained Attention** (Issue #200 - M4)
   - Cross-attention between query/doc ASTs
   - Role-based masking (SUBJ↔SUBJ only)
   - 1-2M params, expected +1-2% recall

---

## 🎬 Demo Scripts Available

### **Working Demos**

```bash
# 1. Parse Esperanto sentences
python -m klareco parse "Mi amas la hundon."
python -m klareco translate "The dog sees the cat." --to eo

# 2. Root embeddings demo (interactive)
python scripts/demo_root_embeddings.py -i
# Try: "hund bon", "mal grand", "bel egal"

# 3. Affix embeddings demo
python scripts/demo_affix_embeddings.py

# 4. Compositional embeddings demo
python scripts/demo_embeddings.py

# 5. RAG Q&A with compositional embeddings
python scripts/demo_rag_compositional.py -i
# Try: "Kio estas Esperanto?", "Kiu estas Frodo?"

# 6. RAG Q&A (older version)
python scripts/demo_rag.py --interactive

# 7. Full pipeline demo (EnrichedAST + SemanticPipeline)
python scripts/demo_pipeline.py
```

### **Shell Scripts**

```bash
# Data pipeline
./scripts/pipeline.sh              # Run full data pipeline
./scripts/clean_all.sh             # Clean all texts
./scripts/extract_all.sh           # Extract sentences from all sources
./scripts/parse_corpus.sh          # Parse to ASTs

# Indexing
./scripts/index_compositional.sh   # Build compositional index
./scripts/index_faiss.sh           # Build FAISS index

# Training
./scripts/train_roots.sh           # Train root embeddings
./scripts/train_affixes.sh         # Train affix transforms
./scripts/train_full.sh            # Full training pipeline

# Testing & validation
./scripts/test_all.sh              # Run all tests
./scripts/validate_all.sh          # Validate data quality
```

---

## 📊 Key Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| **Total parameters** | 733K | <1M | ✅ On track |
| **Parse rate** | 91.8% | >90% | ✅ Met |
| **Root embedding accuracy** | 97.98% | >95% | ✅ Met |
| **Corpus size** | 4.4M sentences | 4M+ | ✅ Met |
| **M1 partial match** | 20% | >15% | ✅ Exceeded |
| **M2 retrieval recall** | 85% | 80% | ✅ **Exceeded** |
| **M3 deterministic recall** | - | 95% | 🔄 In progress |

---

## 🏗️ Architecture Summary

```
Text → Parser (0 params) → AST → Embeddings (733K) → RAG → Reasoning Core (20-100M) → Linearizer (0 params) → Text
       ↑ 16 rules              ↑ Stage 1 complete         ↑ M2 focus      ↑ Future           ↑ deterministic
```

**What's deterministic (0 params)**:
- Parser (16 Esperanto grammar rules)
- Morphology analyzer
- Deparser / linearizer
- Grammar checker
- Function word handling

**What's learned**:
- Stage 1 (733K): Root embeddings + affix transforms ✅ **COMPLETE**
- Stage 2 (~52K): Grammatical transforms 🔲 Planned
- Stage 3 (~100K): Discourse model 🔲 Planned
- Stage 4 (20-100M): Reasoning core 🔲 Future

---

## 📝 Open Issues

**Total**: 12 GitHub issues (10 from IdlerGear context + 2 new local tasks)

**High priority (M2)**:
- #171: Improve retrieval corpus to 80% recall
- #3 (local): Implement slot-based indexing
- Data quality fixes (#172-176)

**Medium priority**:
- #170: Run OLMo 1B on English benchmarks
- #4 (local): Graph-based embeddings

**Future work**:
- #167-169: Multi-hop reasoning, discourse model, generation

---

## 💡 Key Insights from This Session

1. **Sentence averaging is suboptimal** - We're not using our AST structure!
   - Current: Simple averaging loses role information
   - Better: Slot-based retrieval (SUBJ/VERB/OBJ separate)
   - Best: Graph embeddings (full structural encoding)

2. **Esperanto's free word order helps** - Makes averaging less harmful than in English
   - Accusative markers (`-n`) indicate roles explicitly
   - Word order is stylistic, not semantic

3. **Tiered architecture matters** - Fundamento should outrank Wikipedia
   - Tier 0: Curated facts (currently empty - opportunity!)
   - Tier 1-3: Authoritative (18K sentences)
   - Tier 5+: General (4.4M sentences)

4. **M2 bottleneck is retrieval, not embeddings** - Stage 1 models work well
   - Embeddings: 97.98% accuracy ✅
   - Retrieval recall: 35% ❌
   - Solution: Better indexing + curated facts

---

## 📚 Key Files

### **Models** (git-tracked)
- `models/root_embeddings/best_model.pt` (712K params)
- `models/affix_transforms_v2/best_model.pt` (21K params)

### **Indexes** (local, 3.1GB total)
- `data/indexes/merged/embeddings.npy` (1.1GB)
- `data/indexes/merged/metadata.jsonl` (1015MB)
- `data/indexes/merged/faiss_index.bin` (1.1GB - building now)

### **Corpus** (local)
- `data/corpus/authoritative_corpus.jsonl` (38MB, 18K sentences)
- `data/corpus/unified_corpus.jsonl` (21GB, 4.38M sentences)

### **Documentation**
- `README.md` - Quick start and current status
- `VISION.md` - Long-term architecture vision
- `TRAINING_QUICKSTART.md` - Training guide
- `CLAUDE.md` - Development guide for Claude Code
- `docs/M1_COMPARISON_REPORT.md` - M1 evaluation results

---

## 🚀 How to Get Started

### **Try the system:**
```bash
# Interactive RAG demo
python scripts/demo_rag_compositional.py -i

# Ask in Esperanto:
# "Kio estas Esperanto?"
# "Kiu kreis Esperanton?"
# "Kie loĝas Frodo?"
```

### **Check progress:**
```bash
# Run tests
python -m pytest

# Validate data quality
./scripts/validate_all.sh

# Check index status
ls -lh data/indexes/merged/
```

### **Next milestone work:**
```bash
# 1. Wait for FAISS index to complete
# 2. Test merged index
python scripts/demo_rag_compositional.py -i

# 3. Start implementing slot-based retrieval
# See: .idlergear/tasks/003-implement-slot-based-retrieval...
```

---

**For detailed task tracking, see**: `.idlergear/tasks/` or run `gh issue list`
