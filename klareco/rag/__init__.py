# RAG (Retrieval-Augmented Generation) module
#
# ACTIVE retrievers for hybrid embeddings (128d = 64d linguistic + 64d topical):
# - ASTAwareRetriever: Full AST analysis with question classification (recommended)
# - HNSWSlotRetriever: HNSW prefilter + mmap slots (fastest)
# - FAISSSlotRetriever: FAISS prefilter + slot rerank
# - HybridFAISSMmapRetriever: FAISS + mmap hybrid (best accuracy)
#
# DEPRECATED retrievers (still available for slot_full/ index):
# - MultiFAISSSlotRetriever: Per-slot FAISS indexes (deprecated 2026-01-06)
# - ScaNNSlotRetriever: ScaNN prefilter (deprecated 2026-01-06)
#
# DELETED retrievers (2026-01-06):
# - SQLiteSlotRetriever: Redundant - use HybridFAISSMmapRetriever instead
# - MemoryMappedSlotRetriever: Redundant - use HybridFAISSMmapRetriever instead
# - SlotBasedRetriever: O(n) linear scan = hours per query
# - Retriever (legacy): Loaded all metadata into RAM = OOM
#
# See IdlerGear notes #82, #83 for decision rationale.

# Active retrievers for hybrid embeddings (128d)
from klareco.rag.ast_aware_retriever import ASTAwareRetriever
from klareco.rag.slot_retriever_faiss import FAISSSlotRetriever
from klareco.rag.slot_retriever_hnsw import HNSWSlotRetriever
from klareco.rag.slot_retriever_hybrid import HybridFAISSMmapRetriever

# Deprecated retrievers (still importable for slot_full/ index)
from klareco.rag.slot_retriever_multifaiss import MultiFAISSSlotRetriever

# Optional: ScaNN requires TensorFlow (deprecated - no hybrid indexes)
try:
    from klareco.rag.slot_retriever_scann import ScaNNSlotRetriever
    _SCANN_AVAILABLE = True
except ImportError:
    _SCANN_AVAILABLE = False

# Only export active retrievers by default
__all__ = [
    'ASTAwareRetriever',
    'HNSWSlotRetriever',
    'FAISSSlotRetriever',
    'HybridFAISSMmapRetriever',
]

# Deprecated retrievers still available for slot_full/ index
__deprecated__ = [
    'MultiFAISSSlotRetriever',
]
if _SCANN_AVAILABLE:
    __deprecated__.append('ScaNNSlotRetriever')
