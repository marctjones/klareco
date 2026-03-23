# Model Issues Reference

Quick reference for all deep learning models and their GitHub issues.

## Model Status Overview

| Model | Params | Status | Primary Issue | Training Issues | Notes |
|-------|--------|--------|---------------|-----------------|-------|
| **RootEmbedder** | 500K | 🔄 In Progress | #685 | #617-620 | Root embeddings |
| **MorphemeComposer** | 500K | 📝 Planned | #698 | (sub-issues TBD) | Compositional embeddings |
| **PlausibilityFilter** | 2M | 📝 Planned | #687 | #621-623 | Selectional preference |
| **RelevanceRanker** | 5M | 📝 Planned | #686 | #629-632, #668 | Reranker |
| **ASTEncoder** | 8M | ✅ Exists | - | - | TreeLSTM (in code) |
| **NodePredictor** | 12M | 📝 Planned | #692 | (embedded in #692) | Next-node predictor |
| **IntentClassifier** | 5M | 📝 Optional | #693 | (embedded in #693) | Intent classifier |
| **DiscourseClassifier** | 10M | 📝 Optional | #694 | (embedded in #694) | Discourse classifier |

**Legend**:
- ✅ Exists: Already implemented in code
- 🔄 In Progress: Currently being worked on
- 📝 Planned: Issue created, not started
- 📝 Optional: Low priority, deterministic alternatives exist

## Foundation Models (Required by All)

### RootEmbedder: Root Embeddings (500K params)
**Purpose**: Semantic similarity between Esperanto roots

**Issues**:
- **#685**: Semantic retrieval integration (primary)
- **#617**: Stage 1.1 - Generate tier-filtered training data
- **#618**: Stage 1.2 - Train root embeddings with tier-filtered vocab
- **#619**: Stage 1.3 - Test embedding quality
- **#620**: Stage 1.4 - Freeze for downstream models
- **#679**: Define semantic role for embeddings

**Status**: 🔄 In Progress (training data ready, model training started)

**Training Script**: `scripts/train_roots.sh`

**Next Steps**:
1. Complete training (#618)
2. Validate quality (#619)
3. Freeze for downstream use (#620)

---

### MorphemeComposer: Compositional Embeddings (500K params)
**Purpose**: Combine root + affix semantics for novel words

**Issues**:
- **#698**: Train Stage 2 compositional embeddings (primary) ⭐ NEW
- **#615**: Research - Compound root decomposition
- **#590**: Advanced semantic constraints (future)

**Status**: 📝 Planned (issue just created)

**Training Script**: `scripts/train_compositional.sh` (to be created)

**Next Steps**:
1. Create data export script (#698 sub-issue needed)
2. Implement compositional model
3. Train after RootEmbedder completes

**Dependencies**: RootEmbedder must be trained first (frozen root embeddings)

---

## Retrieval Models (Support Reasoning & QA)

### PlausibilityFilter: Selectional Preference (2M params)
**Purpose**: Filter semantically implausible SVO triples

**Issues**:
- **#687**: Train PlausibilityFilter selectional preference model (primary)
- **#621**: PlausibilityFilter.1 - Generate training data from Tier 5 corpus
- **#622**: PlausibilityFilter.2 - Train selectional preference model
- **#623**: PlausibilityFilter.3 - Test model accuracy

**Status**: 📝 Planned

**Training Script**: `scripts/train_m1.sh` (to be created)

**Next Steps**:
1. Generate training data (#621)
2. Train model (#622)
3. Evaluate on test set (#623)

**Dependencies**: RootEmbedder (root embeddings for SVO features)

---

### RelevanceRanker: Reranker (5M params)
**Purpose**: Rank retrieved facts by query relevance

**Issues**:
- **#686**: Train reranker model (primary)
- **#668**: Train Model 4: Reranker (alternative/duplicate?)
- **#629**: Reranker 1 - Generate query-document relevance pairs
- **#630**: Reranker 2 - Train neural reranker with frozen embeddings
- **#631**: Reranker 3 - Test relevance ranking quality
- **#632**: Reranker 4 - Deploy in production RAG pipeline
- **#649**: Data export script for reranker training pairs

**Status**: 📝 Planned

**Training Script**: `scripts/train_reranker.sh` (to be created)

**Next Steps**:
1. Generate training pairs (#629, #649)
2. Train reranker (#630)
3. Test ranking quality (#631)
4. Integrate into RAG (#632)

**Dependencies**: RootEmbedder (frozen embeddings for cross-encoder)

---

## Generation Models

### ASTEncoder: TreeLSTM Encoder (8M params)
**Purpose**: Encode AST structure into dense vector

**Issues**:
- **None** - Already implemented in `klareco/models/tree_lstm.py`

**Status**: ✅ Exists in codebase

**Training**: Trained jointly with downstream models (NodePredictor, IntentClassifier, DiscourseClassifier)

**Used By**:
- NodePredictor (Next-Node Predictor)
- IntentClassifier (Intent Classifier)
- DiscourseClassifier (Discourse Classifier)

---

### NodePredictor: Next-Node Predictor (12M params)
**Purpose**: Predict next AST node for text generation

**Issues**:
- **#692**: Implement AST Generation (includes NodePredictor training)
- **#690**: Abstractive answer generation (uses NodePredictor)

**Status**: 📝 Planned

**Training Script**: `scripts/train_ast_generator.sh` (to be created)

**Next Steps** (from #692):
1. Implement ASTGenerator model
2. Create training data from parsed corpus
3. Train next-node prediction
4. Evaluate generation quality

**Dependencies**:
- MorphemeComposer (compositional embeddings for node features)
- ASTEncoder (TreeLSTM encoder, trained jointly)

**Note**: Training is embedded in #692 (capability issue), not a separate issue

---

## Instruction & Chat Models (Optional)

### IntentClassifier: Intent Classifier (5M params, OPTIONAL)
**Purpose**: Classify ambiguous instruction intent

**Issues**:
- **#693**: Implement Intent Classification (includes IntentClassifier training)

**Status**: 📝 Optional (85% of instructions handled by deterministic patterns!)

**Training Script**: `scripts/train_intent_classifier.sh` (to be created)

**Next Steps** (from #693):
1. Collect instruction-intent pairs (~10K annotations)
2. Implement intent classifier
3. Train on annotated data
4. Compare to deterministic baseline

**Dependencies**: ASTEncoder (TreeLSTM encoder, trained jointly)

**Alternative**: Use deterministic verb patterns + clarification questions (0 params)

**Recommendation**: Start with deterministic approach, only train if clarification questions are unacceptable

---

### DiscourseClassifier: Discourse Classifier (10M params, OPTIONAL)
**Purpose**: Classify discourse relation between conversation turns

**Issues**:
- **#694**: Implement Multi-Turn Chat (includes DiscourseClassifier training)
- **#661**: RST discourse relation detection (related, different focus)

**Status**: 📝 Optional (90% of chat works with deterministic coreference!)

**Training Script**: `scripts/train_discourse_model.sh` (to be created)

**Next Steps** (from #694):
1. Collect multi-turn dialogues (~5K annotations)
2. Annotate discourse relations (6 classes)
3. Implement discourse classifier
4. Train on annotated data

**Dependencies**: ASTEncoder (TreeLSTM encoder, trained jointly)

**Alternative**: Basic chat works with deterministic pronoun resolution + entity tracking (0 params)

**Recommendation**: Start with deterministic approach, only train if response quality is insufficient

---

## Training Order

```
Stage 0: Parser (deterministic) ✅
  ↓
Stage 1: RootEmbedder Root Embeddings (500K) 🔄 #685, #617-620
  ↓
Stage 2: MorphemeComposer Compositional Embeddings (500K) 📝 #698
  ↓
Stage 3: PlausibilityFilter Selectional Preference (2M) 📝 #687, #621-623
  ↓
Stage 4: RelevanceRanker Reranker (5M) 📝 #686, #629-632
  ↓
Stage 5: ASTEncoder+NodePredictor TreeLSTM + Next-Node (8M+12M) 📝 #692
  ↓
Stage 6 (Optional): IntentClassifier Intent Classifier (5M) 📝 #693
  ↓
Stage 7 (Optional): DiscourseClassifier Discourse Classifier (10M) 📝 #694
```

## Configuration Options

### Minimal (21M params) - RECOMMENDED FIRST
**Models**: RootEmbedder, MorphemeComposer, ASTEncoder, NodePredictor

**Issues to Complete**:
- ✅ #685 (RootEmbedder)
- ✅ #698 (MorphemeComposer)
- ✅ #692 (ASTEncoder+NodePredictor)

**Capabilities**:
- ✓ Text generation
- ✓ Instruction following (85% deterministic)
- ✓ Multi-turn chat (90% deterministic)
- ✓ Symbolic reasoning (100% deterministic)

---

### Standard (28M params) - RECOMMENDED FOR PRODUCTION
**Models**: Minimal + PlausibilityFilter, RelevanceRanker

**Additional Issues**:
- ✅ #687 (PlausibilityFilter)
- ✅ #686 (RelevanceRanker)

**Improvements**:
- Better retrieval precision
- Fewer hallucinations
- More accurate reasoning

---

### Full (43M params) - MAXIMUM QUALITY
**Models**: Standard + IntentClassifier, DiscourseClassifier

**Additional Issues**:
- ✅ #693 (IntentClassifier)
- ✅ #694 (DiscourseClassifier)

**Improvements**:
- Automatic intent disambiguation
- Better chat quality
- Smoother conversations

---

## Issue Dependencies Graph

```
#685 (RootEmbedder Root Embeddings)
  ├── #698 (MorphemeComposer Compositional) - requires frozen RootEmbedder
  ├── #687 (PlausibilityFilter Selectional) - uses RootEmbedder features
  ├── #686 (RelevanceRanker Reranker) - uses frozen RootEmbedder
  └── #692 (NodePredictor Generator) - uses MorphemeComposer
        ├── #693 (IntentClassifier Intent) - uses ASTEncoder from #692
        └── #694 (DiscourseClassifier Discourse) - uses ASTEncoder from #692
```

**Critical Path**: #685 → #698 → #692

## Quick Links

### By Training Stage
- **Stage 1**: #685, #617-620 (Root Embeddings)
- **Stage 2**: #698 (Compositional Embeddings) ⭐ NEW
- **Stage 3**: #687, #621-623 (Selectional Preference)
- **Stage 4**: #686, #629-632 (Reranker)
- **Stage 5**: #692 (Text Generation with ASTEncoder+NodePredictor)
- **Stage 6**: #693 (Intent Classifier, optional)
- **Stage 7**: #694 (Discourse Classifier, optional)

### By Capability
- **Text Generation**: #692 (includes NodePredictor)
- **Instruction Following**: #693 (includes IntentClassifier, optional)
- **Multi-Turn Chat**: #694 (includes DiscourseClassifier, optional)
- **Symbolic Reasoning**: #695 (0 params, deterministic!)

### By Model
- **RootEmbedder**: #685, #617-620
- **MorphemeComposer**: #698 ⭐ NEW
- **PlausibilityFilter**: #687, #621-623
- **RelevanceRanker**: #686, #629-632, #668
- **ASTEncoder**: (no issue, already exists)
- **NodePredictor**: #692
- **IntentClassifier**: #693
- **DiscourseClassifier**: #694

### Epic Issues
- **#696**: LLM-Style Capabilities (master epic)
- **#697**: Deterministic vs Learned Analysis

## Next Actions

### Immediate (Week 1-2)
1. Complete RootEmbedder training (#685, #618)
2. Validate RootEmbedder quality (#619)
3. Freeze RootEmbedder for downstream (#620)

### Short-term (Week 3-4)
4. Create compositional data export script (#698 sub-issue)
5. Train MorphemeComposer compositional embeddings (#698)

### Medium-term (Month 2)
6. Train PlausibilityFilter selectional preference (#687)
7. Train RelevanceRanker reranker (#686)

### Long-term (Month 3+)
8. Implement and train NodePredictor generator (#692)
9. Evaluate minimal config (21M params)
10. Decide if IntentClassifier, DiscourseClassifier are needed based on results

## Questions to Resolve

1. **IntentClassifier Intent Classifier**: Do we need it? Or can we use deterministic patterns + clarification?
2. **DiscourseClassifier Discourse Model**: Do we need it? Or is basic chat good enough?
3. **RelevanceRanker vs #668**: Are these duplicate issues? Need to reconcile.
4. **Training data annotation**: Who will annotate ~15K examples for IntentClassifier, DiscourseClassifier?

## Summary

✅ **All 7 models now have issues**:
- RootEmbedder: #685 ✓
- MorphemeComposer: #698 ✓ (just created)
- PlausibilityFilter: #687 ✓
- RelevanceRanker: #686 ✓
- ASTEncoder: (exists in code) ✓
- NodePredictor: #692 ✓
- IntentClassifier: #693 ✓
- DiscourseClassifier: #694 ✓

🎯 **Critical path**: #685 → #698 → #692

📝 **Optional models**: IntentClassifier (#693), DiscourseClassifier (#694) - start with deterministic alternatives

🚀 **Recommended start**: Minimal config (21M params) with RootEmbedder, MorphemeComposer, ASTEncoder, NodePredictor
