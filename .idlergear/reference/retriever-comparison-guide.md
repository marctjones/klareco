---
id: 1
title: Retriever Comparison Guide
created: '2026-01-06T19:52:29.713360Z'
updated: '2026-01-06T19:52:29.713379Z'
---
# Klareco Retriever Comparison Guide

## Quick Reference: Which Retriever to Use

**For production (4.4M corpus):**
- **ASTAwareRetriever** - Best for Q&A with full AST analysis
- **HNSWSlotRetriever** - Fastest (~2-3ms latency)
- **FAISSSlotRetriever** - Good balance of speed and accuracy
- **HybridFAISSMmapRetriever** - Best accuracy (~90% recall)

**DO NOT USE with 4.4M corpus:**
- ❌ SlotBasedRetriever - O(n) linear scan (hours per query)
- ❌ Retriever (legacy) - Loads all metadata into RAM (OOM crash)

## Retriever Matrix

| Retriever | AST-Aware | Memory Efficient | Latency | Recall | Status |
|-----------|:---------:|:----------------:|:-------:|:------:|:------:|
| **ASTAwareRetriever** | ✅✅ | ✅ | ~0.4s | ~85% | ✅ Production |
| **HNSWSlotRetriever** | ✅ | ✅ | ~2-3ms | ~85-90% | ✅ Production |
| **FAISSSlotRetriever** | ✅ | ✅ | ~5ms | ~85% | ✅ Production |
| **HybridFAISSMmapRetriever** | ✅ | ✅ | ~3.5ms | ~90% | ✅ Production |
| ScaNNSlotRetriever | ✅ | ✅ | ~3-5ms | ~90-95% | ⚠️ Deprecated |
| MultiFAISSSlotRetriever | ✅ | ✅ | ~8ms | ~88% | ⚠️ Deprecated |
| ~~SQLiteSlotRetriever~~ | - | - | - | - | ❌ Deleted 2026-01-06 |
| ~~MemoryMappedSlotRetriever~~ | - | - | - | - | ❌ Deleted 2026-01-06 |
| ~~SlotBasedRetriever~~ | - | - | - | - | ❌ Deleted 2026-01-06 |
| ~~Retriever (legacy)~~ | - | - | - | - | ❌ Deleted 2026-01-06 |

## Detailed Comparison

### Memory-Efficient Retrievers

All these retrievers can handle the 4.4M corpus without OOM:

#### 1. ASTAwareRetriever
- **File**: `ast_aware_retriever.py`
- **Best for**: Question answering with full AST analysis
- **Features**: 
  - Question type classification (WHO/WHAT/WHERE/WHEN)
  - Entity recognition
  - Pattern matching
  - Semantic relations (synonyms/antonyms from ReVo)
- **Prefilters**: HNSW (fast) or keyword grep (deterministic)
- **Deterministic**: ✅ Yes (rule-based AST analysis)
- **Training**: Uses root embeddings (733K params) for HNSW prefilter

#### 2. HNSWSlotRetriever
- **File**: `slot_retriever_hnsw.py`
- **Best for**: Fastest retrieval with good accuracy
- **Pipeline**: HNSW prefilter → Mmap slot rerank
- **Memory**: HNSW disk-backed + mmap arrays
- **Deterministic**: ✅ Yes (slot matching is rule-based)
- **Training**: Uses root embeddings (733K params)

#### 3. FAISSSlotRetriever
- **File**: `slot_retriever_faiss.py`
- **Best for**: Good balance of speed and simplicity
- **Pipeline**: FAISS prefilter → Slot rerank
- **Memory**: FAISS mmap + lazy doc loading
- **Deterministic**: ✅ Yes
- **Training**: Uses root embeddings (733K params)

#### 4. HybridFAISSMmapRetriever
- **File**: `slot_retriever_hybrid.py`
- **Best for**: Highest accuracy with fast speed
- **Pipeline**: FAISS prefilter → Mmap slot rerank → Final ranking
- **Memory**: FAISS mmap + mmap slot arrays
- **Deterministic**: ✅ Yes
- **Training**: Uses root embeddings (733K params)

### Deleted Retrievers (2026-01-06)

The following retrievers were deleted as redundant or unusable:

- **SQLiteSlotRetriever**: Redundant - HybridFAISSMmapRetriever provides same functionality
- **MemoryMappedSlotRetriever**: Redundant - HybridFAISSMmapRetriever provides same functionality
- **SlotBasedRetriever**: O(n) linear scan = hours per query with 4.4M corpus
- **Retriever (legacy)**: Loaded all metadata into RAM = OOM crash

See GitHub issues #208, #213 (closed as won't fix) for context.

## Training Requirements

All retrievers use the **same 733K param model**:
- Root embeddings: 712K params (11,121 roots × 64d)
- Affix transforms: 21K params (low-rank matrices)

This is NOT per-retriever training - it's shared infrastructure for semantic similarity.

## Deterministic vs Learned Components

| Component | Type | Parameters |
|-----------|------|------------|
| Parser (16 rules) | Deterministic | 0 |
| Slot extraction (SUBJ/VERB/OBJ) | Deterministic | 0 |
| Slot matching | Deterministic | 0 |
| Question classification | Deterministic | 0 |
| Entity recognition | Deterministic | 0 |
| Pattern matching | Deterministic | 0 |
| Semantic relations (ReVo) | Deterministic | 0 |
| **Root embeddings** | **Learned** | **712K** |
| **Affix transforms** | **Learned** | **21K** |

## Usage Examples

```python
# Production: ASTAwareRetriever
from klareco.rag.ast_aware_retriever import ASTAwareRetriever
retriever = ASTAwareRetriever(
    index_path=Path("data/indexes/slot_hybrid"),
    use_prefilter=True,  # Use HNSW for speed
)
results = retriever.search("Kiu fondis Esperanton?", top_k=10)

# Alternative: HNSWSlotRetriever (faster but less AST analysis)
from klareco.rag.slot_retriever_hnsw import HNSWSlotRetriever
retriever = HNSWSlotRetriever(index_path, indexer)
results = retriever.search("Kiu fondis Esperanton?", top_k=10)
```

## See Also

- Issue #89: SlotBasedRetriever fix
- Issue #90: Legacy Retriever fix
- CLAUDE.md: Architecture overview
