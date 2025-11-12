#!/usr/bin/env python3
"""
Build baseline RAG system using sentence-transformers.

This creates a FAISS index over deparsed Esperanto sentences using
simple text embeddings as a baseline to compare against GNN encoder.

Usage:
    python scripts/build_baseline_rag.py --corpus data/ast_corpus/ --output data/faiss_baseline/

Dependencies:
    pip install sentence-transformers faiss-cpu
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.deparser import deparse
from klareco.logging_config import setup_logging

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError as e:
    print(f"Error: {e}")
    print("\nPlease install required packages:")
    print("  pip install sentence-transformers faiss-cpu")
    sys.exit(1)


def load_ast_corpus(corpus_dir: Path, max_sentences: int = None) -> Tuple[List[Dict], List[str]]:
    """
    Load AST corpus from JSONL files.

    Args:
        corpus_dir: Directory containing *_asts.jsonl files
        max_sentences: Maximum sentences to load (None = all)

    Returns:
        (asts, sentences) tuple
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading AST corpus from {corpus_dir}...")

    asts = []
    sentences = []

    jsonl_files = sorted(corpus_dir.glob('*_asts.jsonl'))
    if not jsonl_files:
        raise FileNotFoundError(f"No *_asts.jsonl files found in {corpus_dir}")

    logger.info(f"Found {len(jsonl_files)} JSONL files")

    for jsonl_file in jsonl_files:
        logger.info(f"  Loading {jsonl_file.name}...")

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    asts.append(data['ast'])
                    sentences.append(data['sentence'])

                    if max_sentences and len(asts) >= max_sentences:
                        logger.info(f"    Reached max_sentences limit ({max_sentences})")
                        return asts, sentences

                except Exception as e:
                    logger.warning(f"    Skipping line {line_num}: {e}")

        logger.info(f"    Loaded {len(asts)} ASTs so far")

    logger.info(f"Total ASTs loaded: {len(asts):,}")
    return asts, sentences


def deparse_corpus(asts: List[Dict], logger) -> List[str]:
    """
    Deparse ASTs to normalized Esperanto text.

    Args:
        asts: List of AST dictionaries

    Returns:
        List of deparsed Esperanto sentences
    """
    logger.info("Deparsing ASTs to normalized Esperanto text...")

    texts = []
    failed = 0

    for i, ast in enumerate(asts, 1):
        try:
            text = deparse(ast)
            texts.append(text)
        except Exception as e:
            # Fallback to original sentence if deparser fails
            logger.warning(f"Deparser failed for AST {i}: {e}")
            texts.append("")  # Empty string as placeholder
            failed += 1

        if i % 10000 == 0:
            logger.info(f"  Deparsed {i:,}/{len(asts):,} ({i/len(asts)*100:.1f}%)")

    logger.info(f"Deparsed {len(texts):,} sentences ({failed} failures)")
    return texts


def build_embeddings(texts: List[str], model_name: str, logger) -> np.ndarray:
    """
    Generate embeddings using sentence-transformers.

    Args:
        texts: List of text strings
        model_name: SentenceTransformer model name

    Returns:
        Numpy array of embeddings (N x D)
    """
    logger.info(f"Loading sentence-transformer model: {model_name}")
    model = SentenceTransformer(model_name)

    logger.info(f"Generating embeddings for {len(texts):,} sentences...")
    logger.info(f"  Model dimension: {model.get_sentence_embedding_dimension()}")

    # Generate embeddings with progress
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )

    logger.info(f"Embeddings generated: shape={embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray, logger) -> faiss.Index:
    """
    Build FAISS index for similarity search.

    Args:
        embeddings: Numpy array of embeddings

    Returns:
        FAISS index
    """
    logger.info("Building FAISS index...")

    dimension = embeddings.shape[1]
    logger.info(f"  Dimension: {dimension}")
    logger.info(f"  Vectors: {len(embeddings):,}")

    # Use L2 (Euclidean) distance for now
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))

    logger.info(f"FAISS index built: {index.ntotal:,} vectors")
    return index


def save_baseline_rag(
    output_dir: Path,
    index: faiss.Index,
    texts: List[str],
    original_sentences: List[str],
    asts: List[Dict],
    model_name: str,
    logger
):
    """
    Save baseline RAG components to disk.

    Args:
        output_dir: Output directory
        index: FAISS index
        texts: Deparsed Esperanto texts
        original_sentences: Original corpus sentences
        asts: AST dictionaries
        model_name: Model name used
    """
    logger.info(f"Saving baseline RAG to {output_dir}...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    index_file = output_dir / 'faiss_index.bin'
    faiss.write_index(index, str(index_file))
    logger.info(f"  Saved FAISS index: {index_file}")

    # Save texts
    texts_file = output_dir / 'texts.json'
    with open(texts_file, 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    logger.info(f"  Saved texts: {texts_file}")

    # Save original sentences
    sentences_file = output_dir / 'original_sentences.json'
    with open(sentences_file, 'w', encoding='utf-8') as f:
        json.dump(original_sentences, f, ensure_ascii=False, indent=2)
    logger.info(f"  Saved original sentences: {sentences_file}")

    # Save ASTs (compressed)
    asts_file = output_dir / 'asts.jsonl'
    with open(asts_file, 'w', encoding='utf-8') as f:
        for ast in asts:
            json.dump(ast, f, ensure_ascii=False)
            f.write('\n')
    logger.info(f"  Saved ASTs: {asts_file}")

    # Save metadata
    metadata = {
        'model_name': model_name,
        'num_vectors': len(texts),
        'dimension': index.d,
        'created': datetime.now().isoformat(),
        'index_type': 'IndexFlatL2',
    }
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Saved metadata: {metadata_file}")

    logger.info("✅ Baseline RAG saved successfully")


def main():
    """Build baseline RAG system."""
    parser = argparse.ArgumentParser(description='Build baseline RAG with sentence-transformers')
    parser.add_argument('--corpus', type=str, default='data/ast_corpus',
                        help='AST corpus directory')
    parser.add_argument('--output', type=str, default='data/faiss_baseline',
                        help='Output directory for FAISS index')
    parser.add_argument('--model', type=str, default='distiluse-base-multilingual-cased-v2',
                        help='SentenceTransformer model name')
    parser.add_argument('--max-sentences', type=int, default=None,
                        help='Maximum sentences to process (for testing)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    args = parser.parse_args()

    # Setup logging
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("="*70)
    logger.info("BASELINE RAG CONSTRUCTION - PHASE 3")
    logger.info("="*70)
    logger.info(f"Corpus: {args.corpus}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Max sentences: {args.max_sentences or 'unlimited'}")
    logger.info("")

    start_time = datetime.now()

    try:
        # Step 1: Load AST corpus
        corpus_dir = Path(args.corpus)
        asts, original_sentences = load_ast_corpus(corpus_dir, args.max_sentences)
        logger.info("")

        # Step 2: Deparse to normalized text
        texts = deparse_corpus(asts, logger)
        logger.info("")

        # Step 3: Generate embeddings
        embeddings = build_embeddings(texts, args.model, logger)
        logger.info("")

        # Step 4: Build FAISS index
        index = build_faiss_index(embeddings, logger)
        logger.info("")

        # Step 5: Save everything
        output_dir = Path(args.output)
        save_baseline_rag(output_dir, index, texts, original_sentences, asts, args.model, logger)
        logger.info("")

        # Summary
        duration = (datetime.now() - start_time).total_seconds()
        logger.info("="*70)
        logger.info("BASELINE RAG COMPLETE")
        logger.info("="*70)
        logger.info(f"Processed: {len(texts):,} sentences")
        logger.info(f"Dimension: {embeddings.shape[1]}")
        logger.info(f"Duration: {duration:.1f}s ({duration/60:.1f} minutes)")
        logger.info(f"Output: {output_dir}")
        logger.info("")
        logger.info("✅ Baseline RAG ready for evaluation!")

        return 0

    except Exception as e:
        logger.error(f"ERROR: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
