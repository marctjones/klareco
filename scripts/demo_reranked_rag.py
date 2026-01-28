#!/usr/bin/env python3
"""
Demo: RAG with M1 Filtering + Reranker Integration

Demonstrates the full RAG pipeline:
1. AST-aware retrieval (structural matching)
2. M1 plausibility filtering (removes nonsense from synonym expansion)
3. Neural reranking (learned relevance scoring)

Shows results with different pipeline configurations.

Usage:
    PYTHONPATH=. python scripts/demo_reranked_rag.py
    PYTHONPATH=. python scripts/demo_reranked_rag.py --query "Kio estas hundo?"
    PYTHONPATH=. python scripts/demo_reranked_rag.py --no-m1        # Skip M1 filtering
    PYTHONPATH=. python scripts/demo_reranked_rag.py --no-rerank   # Skip reranking
"""

import argparse
import logging
from pathlib import Path
import torch
from typing import List, Tuple, Dict, Optional

from klareco.parser import parse
from klareco.embeddings.compositional import CompositionalEmbedding
from klareco.models.reranker import ASTReranker
from klareco.models.m1_inference import M1Inference
from klareco.rag.ast_aware_retriever import ASTAwareRetriever

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RerankedRetriever:
    """
    Wrapper around ASTAwareRetriever that adds reranking.

    Full Pipeline:
    1. Structural retrieval with M1 query expansion (filters synonyms BEFORE search)
    2. Neural reranking (learned relevance scoring)

    NOTE: M1 is now integrated into the retriever for query expansion,
    not used as a post-hoc filter. This is M1's intended purpose.
    """

    def __init__(
        self,
        retriever: ASTAwareRetriever,
        reranker: ASTReranker,
        rerank_top_k: int = 50,
    ):
        """
        Args:
            retriever: Base retriever (with optional M1 for query expansion)
            reranker: Trained reranker model
            rerank_top_k: Retrieve this many candidates, then rerank
        """
        self.retriever = retriever
        self.reranker = reranker
        self.rerank_top_k = rerank_top_k

    def extract_svo_triple(self, ast: Dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract subject-verb-object triple from AST."""
        def get_root(node):
            if node is None:
                return None
            if isinstance(node, dict):
                if node.get('tipo') == 'vortgrupo':
                    kerno = node.get('kerno', {})
                    return kerno.get('radiko')
                elif node.get('tipo') == 'vorto':
                    return node.get('radiko')
            return None

        subj = get_root(ast.get('subjekto'))
        verb = get_root(ast.get('verbo'))
        obj = get_root(ast.get('objekto'))

        return (subj, verb, obj)

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_m1_expansion: bool = True,
        use_reranking: bool = True,
        m1_min_plausibility: float = 0.5,
    ):
        """
        Search with optional M1 query expansion and reranking.

        Args:
            query: Query text
            top_k: Final number of results
            use_m1_expansion: Whether to use M1 for query expansion (filters synonyms BEFORE search)
            use_reranking: Whether to apply reranking
            m1_min_plausibility: Minimum M1 score for keeping synonym expansions

        Returns:
            List of (score, doc, stats) tuples
        """
        # Stage 1: Structural retrieval with M1 query expansion
        # M1 filters synonym expansions BEFORE search (if enabled in retriever)
        candidates = self.retriever.search(
            query=query,
            top_k=max(self.rerank_top_k, top_k),
            use_m1_expansion=use_m1_expansion,
            m1_min_plausibility=m1_min_plausibility,
        )

        if not candidates:
            return []

        if not use_reranking:
            return candidates[:top_k]

        # Stage 2: Neural reranking
        logger.info(f"  Reranking {len(candidates)} candidates...")

        query_ast = parse(query)
        reranked = []

        for score, doc, stats in candidates:
            try:
                # Parse document text
                doc_text = doc.get('text', '')
                doc_ast = parse(doc_text)

                # Get reranker score
                with torch.no_grad():
                    rerank_score = self.reranker(query_ast, doc_ast).item()

                # Combine scores: retrieval + reranker
                # Weight: 30% retrieval, 70% reranker
                # (M1 already influenced retrieval via query expansion)
                combined_score = 0.3 * score + 0.7 * rerank_score

                reranked.append((combined_score, doc, stats))

            except Exception as e:
                logger.warning(f"Failed to rerank document: {e}")
                # Keep original score if reranking fails
                reranked.append((score, doc, stats))

        # Sort by reranked scores
        reranked.sort(key=lambda x: x[0], reverse=True)

        return reranked[:top_k]


def load_m1(m1_path: Path, stage1_path: Path) -> Optional[M1Inference]:
    """Load M1 plausibility model for query expansion."""
    if not m1_path.exists():
        logger.warning(f"M1 model not found at {m1_path} - M1 query expansion disabled")
        return None

    if not stage1_path.exists():
        logger.warning(f"Stage 1 model not found at {stage1_path} - M1 query expansion disabled")
        return None

    try:
        m1 = M1Inference(
            model_path=m1_path,
            comp_model_path=stage1_path,  # Use comp_model_path parameter
            device='cpu'
        )
        logger.info("✓ M1 model loaded (for query expansion)")
        return m1
    except Exception as e:
        logger.error(f"Failed to load M1: {e}")
        return None


def load_reranker():
    """Load compositional embedding and reranker."""
    # Load compositional embedding
    comp_model_path = Path('models/root_embeddings/best_model.pt')
    checkpoint = torch.load(comp_model_path, map_location='cpu', weights_only=False)

    if 'root_vocab' in checkpoint:
        comp_emb = CompositionalEmbedding(
            root_vocab=checkpoint['root_vocab'],
            prefix_vocab=checkpoint['prefix_vocab'],
            suffix_vocab=checkpoint['suffix_vocab'],
            embed_dim=checkpoint.get('embed_dim', 128),
        )
        comp_emb.load_state_dict(checkpoint['model_state_dict'])
    else:
        root_to_idx = checkpoint['root_to_idx']
        prefix_vocab = {'<NONE>': 0, '<UNK>': 1}
        suffix_vocab = {'<NONE>': 0, '<UNK>': 1}

        comp_emb = CompositionalEmbedding(
            root_vocab=root_to_idx,
            prefix_vocab=prefix_vocab,
            suffix_vocab=suffix_vocab,
            embed_dim=checkpoint.get('embedding_dim', 64),
        )

        if 'embeddings.weight' in checkpoint['model_state_dict']:
            comp_emb.root_embed.weight.data = checkpoint['model_state_dict']['embeddings.weight']
        elif 'weight' in checkpoint['model_state_dict']:
            comp_emb.root_embed.weight.data = checkpoint['model_state_dict']['weight']

    comp_emb.eval()

    # Load reranker
    reranker_path = Path('models/reranker/best_model.pt')
    reranker = ASTReranker.load(reranker_path, comp_emb)
    reranker.eval()

    logger.info("✓ Reranker loaded")
    return reranker


def main():
    parser = argparse.ArgumentParser(
        description="RAG demo with M1 filtering + reranking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (retrieval → M1 → reranker)
  python scripts/demo_reranked_rag.py

  # Skip M1 filtering
  python scripts/demo_reranked_rag.py --no-m1

  # Skip reranking
  python scripts/demo_reranked_rag.py --no-rerank

  # Custom query
  python scripts/demo_reranked_rag.py --query "Kiu fondis Esperanton?"
        """
    )
    parser.add_argument('--query', type=str, help='Query to test')
    parser.add_argument('--index-dir', type=str, default='data/indexes/kuzu_index',
                       help='Path to Kuzu index')
    parser.add_argument('--m1-model', type=str,
                       default='models/m1_semantic_tier_priority/best_model.pt',
                       help='Path to M1 model')
    parser.add_argument('--stage1-model', type=str,
                       default='models/root_embeddings_tier0/best_model.pt',
                       help='Path to Stage 1 embeddings')
    parser.add_argument('--no-m1', action='store_true',
                       help='Disable M1 plausibility filtering')
    parser.add_argument('--no-rerank', action='store_true',
                       help='Disable reranking')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    parser.add_argument('--max-length', type=int, default=None,
                       help='Truncate text to this many characters (default: no truncation)')
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("RAG Demo: Retrieval → M1 Filtering → Reranking")
    logger.info("=" * 70)

    # Load M1 (optional) - for query expansion in retriever
    m1 = None
    if not args.no_m1:
        logger.info("Loading M1 plausibility model...")
        m1 = load_m1(
            m1_path=Path(args.m1_model),
            stage1_path=Path(args.stage1_model),
        )
        if m1 is None:
            logger.warning("M1 loading failed - continuing without M1 query expansion")

    # Load retriever with M1 for query expansion
    logger.info("Loading retriever...")
    base_retriever = ASTAwareRetriever(
        index_path=Path(args.index_dir),
        m1_model=m1,  # Pass M1 for query expansion (filters synonyms BEFORE search)
    )

    # Load reranker
    logger.info("Loading reranker...")
    reranker = load_reranker()

    # Create reranked retriever
    retriever = RerankedRetriever(
        retriever=base_retriever,
        reranker=reranker,
        rerank_top_k=50,
    )

    # Test queries
    test_queries = [
        "Kiu fondis Esperanton?",
        "Kio estas Esperanto?",
        "Kie naskiĝis Zamenhof?",
    ]

    if args.query:
        test_queries = [args.query]

    # Pipeline configuration
    pipeline_config = []
    pipeline_config.append("Retrieval (AST-aware)")
    if m1 and not args.no_m1:
        pipeline_config.append("M1 Query Expansion")
    if not args.no_rerank:
        pipeline_config.append("Reranking")

    logger.info(f"\nPipeline: {' → '.join(pipeline_config)}\n")

    for query in test_queries:
        logger.info("\n" + "=" * 70)
        logger.info(f"Query: {query}")
        logger.info("=" * 70)

        # Search with FULL pipeline (M1 expansion + reranking)
        if not args.no_rerank:
            logger.info("\n--- FULL Pipeline (Retrieval + M1 Expansion + Reranker) ---")
            results_full = retriever.search(
                query=query,
                top_k=args.top_k,
                use_m1_expansion=(m1 is not None and not args.no_m1),
                use_reranking=True,
            )

            for i, (score, doc, _) in enumerate(results_full, 1):
                text = doc.get('text', '')
                if args.max_length:
                    text = text[:args.max_length] + "..." if len(text) > args.max_length else text
                logger.info(f"  {i}. [Combined: {score:.4f}] {text}")

        # Search WITHOUT M1 expansion (for comparison)
        if m1 and not args.no_m1:
            logger.info("\n--- WITHOUT M1 Expansion (Retrieval + Reranker only) ---")
            results_no_m1 = retriever.search(
                query=query,
                top_k=args.top_k,
                use_m1_expansion=False,
                use_reranking=not args.no_rerank,
            )

            for i, (score, doc, _) in enumerate(results_no_m1, 1):
                text = doc.get('text', '')
                if args.max_length:
                    text = text[:args.max_length] + "..." if len(text) > args.max_length else text
                logger.info(f"  {i}. [Score: {score:.4f}] {text}")

        # Search WITHOUT reranking (for comparison)
        if not args.no_rerank:
            logger.info("\n--- WITHOUT Reranking (Retrieval + M1 Expansion only) ---")
            results_no_rerank = retriever.search(
                query=query,
                top_k=args.top_k,
                use_m1_expansion=(m1 is not None and not args.no_m1),
                use_reranking=False,
            )

            for i, (score, doc, _) in enumerate(results_no_rerank, 1):
                text = doc.get('text', '')
                if args.max_length:
                    text = text[:args.max_length] + "..." if len(text) > args.max_length else text
                logger.info(f"  {i}. [Score: {score:.4f}] {text}")

    base_retriever.close()


if __name__ == '__main__':
    main()
