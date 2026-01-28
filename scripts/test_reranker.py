#!/usr/bin/env python3
"""
Test the trained reranker model on sample queries and documents.

Usage:
    python scripts/test_reranker.py
"""

import logging
from pathlib import Path
import torch

from klareco.parser import parse
from klareco.embeddings.compositional import CompositionalEmbedding
from klareco.models.reranker import ASTReranker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_models():
    """Load compositional embedding and reranker models."""
    logger.info("Loading models...")

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
        # Simple root embeddings
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
    logger.info("  ✓ Compositional embedding loaded")

    # Load reranker
    reranker_path = Path('models/reranker/best_model.pt')
    reranker = ASTReranker.load(reranker_path, comp_emb)
    reranker.eval()
    logger.info("  ✓ Reranker loaded")

    return comp_emb, reranker


def test_query_doc_pair(reranker, query_text, doc_text):
    """Test reranker on a single query/doc pair."""
    try:
        query_ast = parse(query_text)
        doc_ast = parse(doc_text)

        with torch.no_grad():
            score = reranker(query_ast, doc_ast)

        return score.item()
    except Exception as e:
        logger.warning(f"Failed to score: {e}")
        return None


def main():
    logger.info("=" * 60)
    logger.info("Reranker Model Test")
    logger.info("=" * 60)

    # Load models
    comp_emb, reranker = load_models()

    # Test cases
    test_cases = [
        # (query, relevant_doc, irrelevant_doc)
        (
            "Kio estas hundo?",
            "Hundo estas domestika besto. Hundoj vivas kun homoj.",
            "Kato estas alia besto. Katoj estas sendependaj.",
        ),
        (
            "Kie vivas la homoj?",
            "Homoj vivas en urboj kaj vilaĝoj. Ili konstruas domojn.",
            "Birdoj flugas en la ĉielo. Ili havas flugilojn.",
        ),
        (
            "Kiu inventis la telefon?",
            "Alexander Graham Bell inventis la telefon. Li estis sciencisto.",
            "La telefon estas utila aparato por komunikado.",
        ),
    ]

    logger.info("\n" + "=" * 60)
    logger.info("Testing Query/Document Pairs")
    logger.info("=" * 60)

    for i, (query, relevant, irrelevant) in enumerate(test_cases, 1):
        logger.info(f"\nTest Case {i}")
        logger.info(f"  Query: {query}")

        # Score relevant document
        rel_score = test_query_doc_pair(reranker, query, relevant)
        logger.info(f"  Relevant doc:   {relevant[:50]}...")
        if rel_score is not None:
            logger.info(f"  Relevance score: {rel_score:.4f}")

        # Score irrelevant document
        irrel_score = test_query_doc_pair(reranker, query, irrelevant)
        logger.info(f"  Irrelevant doc: {irrelevant[:50]}...")
        if irrel_score is not None:
            logger.info(f"  Relevance score: {irrel_score:.4f}")

        # Check if ranking is correct
        if rel_score is not None and irrel_score is not None:
            if rel_score > irrel_score:
                logger.info(f"  ✓ Correct ranking (relevant > irrelevant)")
            else:
                logger.info(f"  ✗ Incorrect ranking (relevant < irrelevant)")

    logger.info("\n" + "=" * 60)
    logger.info("Test complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
