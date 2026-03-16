# Phase 1 Implementation Plan

**Goal**: Enhanced Deterministic + Minimal Learning (75% deterministic, 25% learned)

**Status**: Starting Phase 1
**Estimated Time**: 8 weeks
**Date**: 2026-03-09

---

## Phase 1 Components

### 1. AST Deparser Integration (Deterministic) ✅ Available
**Status**: Deparser already exists at `klareco/deparser.py`
**Task**: Integrate into synthesizer for grammatically perfect output
**Estimated**: 2-3 days
**Priority**: HIGH (immediate quality improvement)

**Subtasks**:
- [ ] Update Synthesizer to use deparser instead of templates
- [ ] Convert selected facts back to AST structure
- [ ] Generate grammatically perfect Esperanto sentences
- [ ] Test on Phase 0 queries

---

### 2. Discourse Planning (Deterministic)
**Status**: Not implemented
**Task**: Add RST (Rhetorical Structure Theory) relations for coherent text
**Estimated**: 1 week
**Priority**: MEDIUM

**Subtasks**:
- [ ] Design discourse relation schema (elaboration, contrast, cause-effect)
- [ ] Implement fact aggregation (combine related facts)
- [ ] Add discourse markers (Krome, Tamen, Sekve)
- [ ] Prevent fact repetition (deduplicate)
- [ ] Order facts by discourse coherence

---

### 3. Root Embeddings (Learned - 320K params)
**Status**: Not implemented
**Task**: Train semantic embeddings for content roots
**Estimated**: 2 weeks
**Priority**: HIGH (enables semantic search)

**Subtasks**:
- [ ] Design embedding architecture (64d per root)
- [ ] Create training data pipeline (co-occurrence from corpus)
- [ ] Implement contrastive learning objective
- [ ] Train on 5.4M sentence corpus
- [ ] Evaluate on synonym/antonym tasks
- [ ] Integrate into retriever for semantic search

**Training**: User will specify when to start

---

### 4. Coreference Resolution (Learned - 10M params)
**Status**: Not implemented
**Task**: Track entity mentions across sentences (resolve pronouns)
**Estimated**: 3 weeks
**Priority**: HIGH (60% of sentences have pronouns)

**Subtasks**:
- [ ] Design coreference model architecture
- [ ] Create annotation guidelines for coreference chains
- [ ] Annotate 1K documents with coreference
- [ ] Train coreference resolver
- [ ] Integrate into fact extractor
- [ ] Resolve pronouns to concrete entities

**Training**: User will specify when to start

---

### 5. Annotation Expansion (Data Work)
**Status**: 50 roots annotated (Phase 0)
**Task**: Expand to 200 roots with semantic properties
**Estimated**: 2 weeks
**Priority**: MEDIUM

**Subtasks**:
- [ ] Identify most frequent 200 roots in corpus
- [ ] Create annotation interface/tooling
- [ ] Annotate 150 additional roots with semantic properties
- [ ] Load annotations into Kuzu database
- [ ] Validate annotation quality

---

## Implementation Order (Recommended)

### Week 1-2: Quick Wins (Deterministic)
1. **AST Deparser Integration** (2-3 days)
   - Immediate quality improvement
   - No training needed
2. **Discourse Planning** (1 week)
   - Better text coherence
   - No training needed

### Week 3-4: Semantic Foundation (Learned)
3. **Root Embeddings Infrastructure** (1 week)
   - Design architecture
   - Create training pipeline
   - **User decision point**: Run training?
4. **Semantic Search Integration** (1 week)
   - Integrate embeddings into retriever
   - Test semantic similarity

### Week 5-7: Coreference (Learned)
5. **Coreference Model Design** (1 week)
   - Architecture design
   - Annotation guidelines
6. **Coreference Data Annotation** (1 week)
   - Annotate 1K documents
   - **User decision point**: Use manual or semi-automated?
7. **Coreference Training & Integration** (1 week)
   - **User decision point**: Run training?
   - Integrate into fact extractor

### Week 8: Expansion
8. **Annotation Expansion** (1 week)
   - Annotate 150 more roots
   - Load into database
   - Test on extended vocabulary

---

## Success Criteria (Phase 1 Complete)

- [ ] Deparser integrated (perfect grammar output)
- [ ] Discourse planning working (coherent multi-sentence summaries)
- [ ] Root embeddings trained (semantic similarity >0.85 accuracy)
- [ ] Coreference resolver trained (pronoun resolution >80% accuracy)
- [ ] 200 roots annotated (vs 50 in Phase 0)
- [ ] Quality ≥85% on test queries (vs ≥75% Phase 0 target)

---

## Dependencies

### Deterministic Components
- ✅ Deparser exists at `klareco/deparser.py`
- ✅ Parser working (91.8% parse rate)
- ✅ Schema classifier (100% accuracy)
- ✅ Fact extractor (55% yield)

### Learned Components (Training Required)
- ⏳ Root embeddings (need training pipeline + user approval)
- ⏳ Coreference resolver (need annotation + training + user approval)

### Data Requirements
- ✅ 5.4M sentence corpus available
- ✅ 50 roots annotated
- ⏳ Need 1K documents annotated for coreference
- ⏳ Need 150 more roots annotated

---

## Current Status: Ready to Start

**Next Action**: Implement AST Deparser integration (highest priority, immediate impact)

**Estimated Timeline**:
- Weeks 1-2: Deterministic improvements (deparser + discourse)
- Weeks 3-7: Learned models (embeddings + coreference) - with user approval
- Week 8: Data expansion (200 roots)

**Blockers**: None (can start deterministic work immediately)

---

**Last Updated**: 2026-03-09
**Status**: Phase 1 planning complete, ready to implement
**Next**: Integrate AST deparser into synthesizer

