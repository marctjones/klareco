# Getting Started: Implementation Roadmap

**Date**: 2026-03-09
**Status**: Ready to Start
**Goal**: Build schema-based summarization with citations

---

## ✅ What We Already Have

1. **Parser** - `klareco/parser.py` (16 Esperanto rules, 92% parse rate)
2. **Deparser** - `klareco/deparser.py` (AST → text)
3. **Kuzu v2.1 Database** - 13GB corpus loaded (`data/indexes/v2.1_kuzu_index_full`)
4. **Semantic Taxonomy** - `klareco/semantic_enrichment/radiko_semantiko.py` (~270 categories)
5. **RAG System** - `klareco/rag/` (basic retrieval working)
6. **Schema Definition** - `klareco/schema/kuzu_ast_schema_v2_1.py` (Pure Esperanto)

---

## 🎯 Implementation Plan (22 Weeks)

### **PHASE 0: Validation** (2 weeks) - **START HERE**

Goal: Test deterministic baseline before training models

#### Week 1: Extend Schema & Annotate 50 Roots

**Task 1.1: Extend Kuzu Schema** (#655)
```bash
# Add semantic properties to Radiko node
cd /home/marc/Projects/klareco

# Apply schema extensions
python scripts/extend_kuzu_schema.py \
  --schema klareco/schema/semantic_properties.py \
  --database data/indexes/v2.1_kuzu_index_full
```

**Task 1.2: Choose 50 High-Priority Roots** (#656)

Priority formula:
```python
priority = (
    0.30 * funda_score +      # Fundamento status
    0.30 * freq_score +        # Frequency tier
    0.20 * coverage_score +    # Class diversity
    0.20 * importance_score    # Schema importance
) * 100
```

Top candidates (manually verify):
1. **Identification**: est, nom, vid, sci, ...
2. **Creation**: fond, kre, far, konstru, ...
3. **Movement**: ir, ven, don, alport, ...
4. **People**: hom, vir, infan, patr, ...
5. **Time/Place**: jar, lok, urb, land, ...

**Task 1.3: Manually Annotate 50 Roots**
```bash
# Create annotation file
nano data/annotations/phase_0_roots.jsonl

# Format (one per line):
{"radiko": "fond", "verba_klaso": "kreado-26", "aspekta_klaso": "plenumigo",
 "semantika_kampo": "socia", "graveco_biografia": 0.95, "graveco_difina": 0.30,
 "graveco_okazaĵa": 0.90, "funda_stato": "fundamento_kerno", "ofteca_tavolo": 0}
```

**Task 1.4: Load Annotations into Kuzu**
```bash
python scripts/load_semantic_annotations.py \
  --annotations data/annotations/phase_0_roots.jsonl \
  --database data/indexes/v2.1_kuzu_index_full
```

#### Week 2: Implement Deterministic Baseline

**Task 2.1: Implement Schema Classification** (#660, #670)
```bash
# Create new module
touch klareco/summarization/__init__.py
touch klareco/summarization/schema_classifier.py
touch klareco/summarization/importance_scorer.py
```

Implement:
- Schema slot classifier (biographical/definitional/event)
- Deterministic importance formula (schema + RST + novelty)

**Task 2.2: Test on 10 Queries** (#666)
```bash
python scripts/test_deterministic_baseline.py \
  --queries data/test_queries/phase_0.jsonl \
  --output results/phase_0_baseline.json
```

Test queries:
1. "Kiu fondis Esperanton?"
2. "Rakontu pri Zamenhof"
3. "Kio estas Esperanto?"
4. "Kiam estis kreita Esperanto?"
5. "Kie naskiĝis Zamenhof?"
6. "Kial Zamenhof kreis Esperanton?"
7. "Kiuj parolas Esperanton?"
8. "Kio okazis en 1887?"
9. "Kiuj inspiris Zamenhof?"
10. "Kiom da homoj parolas Esperanton?"

**Task 2.3: Evaluate Quality**
- Human evaluation (1-5 scale)
- **Success criteria**: ≥75% quality → proceed to Phase 1

---

### **PHASE 1: Foundation** (8 weeks)

#### Week 1-2: Pure Esperanto Design (#664, #665)
- Complete verb class taxonomy (50-100 classes)
- Complete noun class taxonomy (80-120 classes)
- Define RST relations in Esperanto
- Define all schema slots

#### Week 3: Retrain Root Embeddings (#479)
- Clean vocabulary (remove corrupted tiers)
- Include semantic function words (~125 words)
- Train on full corpus

#### Week 4-5: Annotate 200 Roots (#656, #658)
- Expand to 200 highest-priority roots
- Target: 75% corpus coverage
- Bootstrap from ReVo/Fundamento definitions

#### Week 6-7: Deterministic Pipeline (#670-#674)
Implement:
- RST discourse detection (#661)
- Fact extraction (#657)
- Fact selection (#671)
- Fact clustering (#672)
- Sentence synthesis (#673)
- **Citation tracking** (#674)

#### Week 8: CLI & Evaluation (#675, #666)
- Implement citation lookup commands
- Evaluate on 30 queries
- Measure where deterministic fails

---

### **PHASE 2: Learned Models** (8 weeks)

#### Week 1-2: Train Reranker (#668)
- Collect 10,000-20,000 training examples
- Train 5M param model
- Integrate into RAG pipeline

#### Week 3-4: Train Importance Adjuster (#667)
- Collect 5,000 examples via active learning
- Train 2M param model
- Integrate into summarization

#### Week 5-6: Expand to 500 Roots (#659)
- Target: 90% corpus coverage

#### Week 7-8: Optional - Unknown Root Classifier (#669)
- Only if coverage <90%

---

### **PHASE 3: Optimization** (4 weeks)

#### Week 1-2: Fix M1 Selectional (#475)
- Debug object selectional issues
- Improve to 85%+ accuracy

#### Week 3: Hyperparameter Tuning
- Optimize all models
- Speed improvements

#### Week 4: Final Evaluation (#663)
- 100-question benchmark
- Human evaluation study
- Comparison with baselines

---

## 🚀 Quick Start Commands

### Option 1: Run Phase 0 Script (Automated)
```bash
# Not created yet - we'll build this together
./scripts/run_phase_0.sh
```

### Option 2: Step-by-Step (Manual)
```bash
# Step 1: Extend schema
python scripts/extend_kuzu_schema.py

# Step 2: Annotate 50 roots
nano data/annotations/phase_0_roots.jsonl

# Step 3: Load annotations
python scripts/load_semantic_annotations.py

# Step 4: Test baseline
python scripts/test_deterministic_baseline.py

# Step 5: Evaluate
python scripts/evaluate_phase_0.py
```

---

## 📁 File Structure We'll Create

```
klareco/
├── schema/
│   ├── kuzu_ast_schema_v2_1.py          # ✅ Exists
│   └── semantic_properties.py           # ✅ Just created
├── summarization/                        # 🆕 Create this
│   ├── __init__.py
│   ├── schema_classifier.py             # Schema slot detection
│   ├── rst_detector.py                  # Discourse relations
│   ├── importance_scorer.py             # Deterministic formula
│   ├── fact_selector.py                 # Diversity constraints
│   ├── fact_clusterer.py                # Group for synthesis
│   ├── sentence_synthesizer.py          # AST fusion
│   └── citation_tracker.py              # Source provenance
├── cli.py                               # ✅ Exists, will extend
└── semantic_enrichment/
    ├── radiko_semantiko.py              # ✅ Exists (~270 categories)
    └── enricher.py                      # ✅ Exists

scripts/
├── extend_kuzu_schema.py                # 🆕 Create
├── load_semantic_annotations.py         # 🆕 Create
├── test_deterministic_baseline.py       # 🆕 Create
├── evaluate_phase_0.py                  # 🆕 Create
└── run_phase_0.sh                       # 🆕 Create (orchestrator)

data/
├── annotations/
│   └── phase_0_roots.jsonl              # 🆕 Create (50 roots)
├── test_queries/
│   └── phase_0.jsonl                    # 🆕 Create (10 queries)
└── indexes/
    └── v2.1_kuzu_index_full             # ✅ Exists (13GB)
```

---

## 🎯 Next Immediate Steps

**RIGHT NOW:**

1. **Create scripts directory structure:**
   ```bash
   mkdir -p data/annotations
   mkdir -p data/test_queries
   mkdir -p results
   ```

2. **Create extend_kuzu_schema.py script** (15 min)
   - Read semantic_properties.py
   - Apply ALTER TABLE commands to Kuzu
   - Verify schema extended

3. **Create 10 test queries** (10 min)
   - Simple factoids + complex summaries
   - Save to `data/test_queries/phase_0.jsonl`

4. **Manually annotate 10 roots as proof-of-concept** (30 min)
   - Start with: fond, est, hund, hom, ir, sci, far, parol, vid, nom
   - Save to `data/annotations/phase_0_roots.jsonl`

5. **Test schema extension** (10 min)
   - Run extend_kuzu_schema.py
   - Verify new properties exist

**Total time to see first results: ~1-2 hours**

---

## ⚠️ Important Notes

- **Don't skip Phase 0!** We need to validate the deterministic approach works before training models
- **75% quality threshold**: If Phase 0 baseline achieves ≥75%, we know the approach is sound
- **Manual annotation is OK**: 50 roots is manageable (~2-3 hours of work)
- **Build incrementally**: Test each component before moving to next

---

## 🆘 If You Get Stuck

1. Check existing code in `klareco/rag/` for retrieval examples
2. Check `klareco/semantic_enrichment/` for annotation examples
3. Read `docs/COMPLETE_SYSTEM_DESIGN_WITH_MODELS.md` for architecture
4. Look at issue #666 for Phase 0 details

---

## 📊 Success Metrics

**Phase 0 (2 weeks):**
- ✅ Schema extended with semantic properties
- ✅ 50 roots annotated
- ✅ 10 queries answered with ≥75% quality
- ✅ Decision: Proceed to Phase 1 or revise

**Phase 1 (8 weeks):**
- ✅ 200 roots annotated
- ✅ Deterministic pipeline working
- ✅ Citations generated
- ✅ CLI commands functional

**Phase 2 (8 weeks):**
- ✅ All 5 models trained
- ✅ 500 roots annotated
- ✅ 90% corpus coverage

**Phase 3 (4 weeks):**
- ✅ 100-question benchmark: 85%+ quality
- ✅ Production-ready system

---

## 🎉 Ready to Start?

**First command to run:**
```bash
cd /home/marc/Projects/klareco
mkdir -p data/annotations data/test_queries results
echo "Phase 0 directories created!"
```

Let's build this! 🚀
