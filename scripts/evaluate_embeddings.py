#!/usr/bin/env python3
"""
Evaluate Tree-LSTM embeddings vs Baseline RAG embeddings.

This script:
1. Loads trained Tree-LSTM model
2. Loads baseline FAISS index and embeddings
3. Generates test queries from corpus
4. Compares retrieval performance:
   - Precision@K
   - Recall@K
   - Mean Reciprocal Rank (MRR)
5. Generates comparison report with visualizations

Usage:
    python scripts/evaluate_embeddings.py \\
        --tree-lstm models/tree_lstm/best_model.pt \\
        --baseline data/faiss_baseline \\
        --corpus data/ast_corpus \\
        --output evaluation_results \\
        --num-queries 100
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from klareco.ast_to_graph import ASTToGraphConverter
from klareco.deparser import deparse
from klareco.models.tree_lstm import TreeLSTMEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_asts_from_corpus(corpus_dir: Path, max_asts: int = None) -> List[dict]:
    """Load ASTs from corpus directory."""
    logger.info(f"Loading ASTs from {corpus_dir}...")

    asts = []
    jsonl_files = sorted(corpus_dir.glob("*.jsonl"))

    for jsonl_file in tqdm(jsonl_files, desc="Loading corpus files"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if max_asts and len(asts) >= max_asts:
                    break

                data = json.loads(line.strip())
                asts.append(data['ast'])

        if max_asts and len(asts) >= max_asts:
            break

    logger.info(f"Loaded {len(asts)} ASTs")
    return asts


def load_tree_lstm_model(model_path: Path, device: str = 'cpu') -> TreeLSTMEncoder:
    """Load trained Tree-LSTM model."""
    logger.info(f"Loading Tree-LSTM model from {model_path}...")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Use hardcoded config (matches training parameters)
    config = {
        'vocab_size': 10000,
        'embed_dim': 128,
        'hidden_dim': 256,
        'output_dim': 512
    }

    model = TreeLSTMEncoder(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim']
    )

    # Load weights - handle both checkpoint and state_dict formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Direct state_dict (final_model.pt format)
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    logger.info(f"Model loaded (output_dim={config['output_dim']})")
    return model


def load_baseline_rag(baseline_dir: Path) -> Tuple[faiss.Index, SentenceTransformer, List[str], List[dict]]:
    """Load baseline RAG system."""
    logger.info(f"Loading baseline RAG from {baseline_dir}...")

    # Load FAISS index
    index_path = baseline_dir / "faiss_index.bin"
    index = faiss.read_index(str(index_path))

    # Load sentence transformer model
    with open(baseline_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)
    model_name = metadata['model_name']
    model = SentenceTransformer(model_name)

    # Load texts and ASTs
    with open(baseline_dir / "texts.json", 'r', encoding='utf-8') as f:
        texts = json.load(f)

    with open(baseline_dir / "asts.jsonl", 'r', encoding='utf-8') as f:
        asts = [json.loads(line.strip()) for line in f]

    logger.info(f"Loaded baseline RAG: {len(texts)} vectors, dim={index.d}")
    return index, model, texts, asts


def encode_with_tree_lstm(ast: dict, model: TreeLSTMEncoder, converter: ASTToGraphConverter, device: str = 'cpu') -> np.ndarray:
    """Encode AST using Tree-LSTM."""
    # Convert AST to graph
    graph = converter.ast_to_graph(ast)
    graph = graph.to(device)

    # Encode
    with torch.no_grad():
        embedding = model(graph)

    # Return as numpy array
    return embedding.cpu().numpy()


def encode_with_baseline(ast: dict, model: SentenceTransformer) -> np.ndarray:
    """Encode AST using baseline sentence-transformers."""
    # Deparse AST to text
    text = deparse(ast)

    # Encode text
    embedding = model.encode(text, convert_to_numpy=True)

    return embedding


def generate_test_queries(
    asts: List[dict],
    num_queries: int,
    seed: int = 42
) -> List[Tuple[int, dict]]:
    """Generate test queries by randomly sampling ASTs."""
    random.seed(seed)

    # Sample random indices
    indices = random.sample(range(len(asts)), min(num_queries, len(asts)))

    # Return (index, AST) pairs
    return [(idx, asts[idx]) for idx in indices]


def search_faiss(
    query_embedding: np.ndarray,
    index: faiss.Index,
    k: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Search FAISS index for top-K similar vectors."""
    # Ensure query is 2D
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Search
    distances, indices = index.search(query_embedding, k)

    return distances[0], indices[0]


def calculate_precision_at_k(
    query_idx: int,
    retrieved_indices: np.ndarray,
    k: int,
    threshold: int = 10
) -> float:
    """
    Calculate Precision@K.

    For this evaluation, a "relevant" document is one within `threshold`
    positions of the query in the corpus (i.e., similar context).
    """
    relevant_count = 0
    for idx in retrieved_indices[:k]:
        # Check if retrieved doc is "close" to query doc
        if abs(idx - query_idx) <= threshold:
            relevant_count += 1

    return relevant_count / k


def calculate_recall_at_k(
    query_idx: int,
    retrieved_indices: np.ndarray,
    k: int,
    threshold: int = 10
) -> float:
    """
    Calculate Recall@K.

    Total relevant docs = 2 * threshold + 1 (docs within threshold distance).
    """
    total_relevant = 2 * threshold + 1

    relevant_count = 0
    for idx in retrieved_indices[:k]:
        if abs(idx - query_idx) <= threshold:
            relevant_count += 1

    return relevant_count / min(total_relevant, k)


def calculate_mrr(
    query_idx: int,
    retrieved_indices: np.ndarray,
    threshold: int = 10
) -> float:
    """
    Calculate Mean Reciprocal Rank.

    Finds the rank of the first relevant document.
    """
    for rank, idx in enumerate(retrieved_indices, start=1):
        if abs(idx - query_idx) <= threshold:
            return 1.0 / rank

    return 0.0


def evaluate_system(
    queries: List[Tuple[int, dict]],
    corpus_asts: List[dict],
    encoder_fn,
    index: faiss.Index,
    k_values: List[int] = [1, 5, 10],
    threshold: int = 10
) -> Dict:
    """Evaluate a retrieval system."""
    results = {
        'precision': {k: [] for k in k_values},
        'recall': {k: [] for k in k_values},
        'mrr': []
    }

    for query_idx, query_ast in tqdm(queries, desc="Evaluating"):
        # Encode query
        query_embedding = encoder_fn(query_ast)

        # Search
        _, retrieved_indices = search_faiss(query_embedding, index, k=max(k_values))

        # Calculate metrics
        for k in k_values:
            precision = calculate_precision_at_k(query_idx, retrieved_indices, k, threshold)
            recall = calculate_recall_at_k(query_idx, retrieved_indices, k, threshold)
            results['precision'][k].append(precision)
            results['recall'][k].append(recall)

        mrr = calculate_mrr(query_idx, retrieved_indices, threshold)
        results['mrr'].append(mrr)

    # Average metrics
    averaged = {
        'precision': {k: np.mean(results['precision'][k]) for k in k_values},
        'recall': {k: np.mean(results['recall'][k]) for k in k_values},
        'mrr': np.mean(results['mrr'])
    }

    return averaged


def build_tree_lstm_index(
    asts: List[dict],
    model: TreeLSTMEncoder,
    converter: ASTToGraphConverter,
    device: str = 'cpu'
) -> faiss.Index:
    """Build FAISS index from Tree-LSTM embeddings."""
    logger.info("Building Tree-LSTM FAISS index...")

    # Get output dimension from model (nested in tree_lstm)
    output_dim = model.tree_lstm.output_dim

    # Create index
    index = faiss.IndexFlatL2(output_dim)

    # Encode all ASTs
    embeddings = []
    for ast in tqdm(asts, desc="Encoding ASTs"):
        embedding = encode_with_tree_lstm(ast, model, converter, device)
        embeddings.append(embedding)

    # Add to index
    embeddings = np.vstack(embeddings)
    index.add(embeddings)

    logger.info(f"Built FAISS index: {index.ntotal} vectors, dim={index.d}")
    return index


def generate_comparison_report(
    baseline_results: Dict,
    tree_lstm_results: Dict,
    output_dir: Path
):
    """Generate comparison report."""
    logger.info("Generating comparison report...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create text report
    report_lines = [
        "=" * 80,
        "TREE-LSTM VS BASELINE RAG EVALUATION",
        "=" * 80,
        "",
        "## Precision@K",
        ""
    ]

    for k in sorted(baseline_results['precision'].keys()):
        baseline_p = baseline_results['precision'][k]
        tree_lstm_p = tree_lstm_results['precision'][k]
        improvement = ((tree_lstm_p - baseline_p) / baseline_p * 100) if baseline_p > 0 else 0

        report_lines.extend([
            f"Precision@{k}:",
            f"  Baseline:   {baseline_p:.4f}",
            f"  Tree-LSTM:  {tree_lstm_p:.4f}",
            f"  Improvement: {improvement:+.2f}%",
            ""
        ])

    report_lines.extend([
        "## Recall@K",
        ""
    ])

    for k in sorted(baseline_results['recall'].keys()):
        baseline_r = baseline_results['recall'][k]
        tree_lstm_r = tree_lstm_results['recall'][k]
        improvement = ((tree_lstm_r - baseline_r) / baseline_r * 100) if baseline_r > 0 else 0

        report_lines.extend([
            f"Recall@{k}:",
            f"  Baseline:   {baseline_r:.4f}",
            f"  Tree-LSTM:  {tree_lstm_r:.4f}",
            f"  Improvement: {improvement:+.2f}%",
            ""
        ])

    report_lines.extend([
        "## Mean Reciprocal Rank (MRR)",
        ""
    ])

    baseline_mrr = baseline_results['mrr']
    tree_lstm_mrr = tree_lstm_results['mrr']
    improvement = ((tree_lstm_mrr - baseline_mrr) / baseline_mrr * 100) if baseline_mrr > 0 else 0

    report_lines.extend([
        f"MRR:",
        f"  Baseline:   {baseline_mrr:.4f}",
        f"  Tree-LSTM:  {tree_lstm_mrr:.4f}",
        f"  Improvement: {improvement:+.2f}%",
        "",
        "=" * 80
    ])

    # Save report
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"Report saved to {report_path}")

    # Save JSON results
    results_json = {
        'baseline': baseline_results,
        'tree_lstm': tree_lstm_results
    }

    json_path = output_dir / "evaluation_results.json"
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    logger.info(f"JSON results saved to {json_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\nPrecision@5:")
    print(f"  Baseline:   {baseline_results['precision'][5]:.4f}")
    print(f"  Tree-LSTM:  {tree_lstm_results['precision'][5]:.4f}")
    print(f"\nRecall@5:")
    print(f"  Baseline:   {baseline_results['recall'][5]:.4f}")
    print(f"  Tree-LSTM:  {tree_lstm_results['recall'][5]:.4f}")
    print(f"\nMRR:")
    print(f"  Baseline:   {baseline_results['mrr']:.4f}")
    print(f"  Tree-LSTM:  {tree_lstm_results['mrr']:.4f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Tree-LSTM vs Baseline RAG")
    parser.add_argument('--tree-lstm', type=str, required=True,
                        help='Path to trained Tree-LSTM model')
    parser.add_argument('--baseline', type=str, required=True,
                        help='Path to baseline RAG directory')
    parser.add_argument('--corpus', type=str, required=True,
                        help='Path to AST corpus directory')
    parser.add_argument('--output', type=str, default='evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--num-queries', type=int, default=100,
                        help='Number of test queries')
    parser.add_argument('--max-corpus-asts', type=int, default=10000,
                        help='Max ASTs to load from corpus for evaluation')
    parser.add_argument('--k-values', type=int, nargs='+', default=[1, 5, 10],
                        help='K values for Precision@K and Recall@K')
    parser.add_argument('--threshold', type=int, default=10,
                        help='Relevance threshold (doc distance)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for Tree-LSTM (cpu or cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Convert paths
    tree_lstm_path = Path(args.tree_lstm)
    baseline_dir = Path(args.baseline)
    corpus_dir = Path(args.corpus)
    output_dir = Path(args.output)

    logger.info("=" * 80)
    logger.info("TREE-LSTM VS BASELINE EVALUATION")
    logger.info("=" * 80)

    # Load corpus
    corpus_asts = load_asts_from_corpus(corpus_dir, max_asts=args.max_corpus_asts)

    # Generate test queries
    logger.info(f"Generating {args.num_queries} test queries...")
    queries = generate_test_queries(corpus_asts, args.num_queries, seed=args.seed)

    # Load baseline system
    baseline_index, baseline_model, baseline_texts, baseline_asts = load_baseline_rag(baseline_dir)

    # Evaluate baseline
    logger.info("Evaluating baseline RAG...")
    baseline_encoder = lambda ast: encode_with_baseline(ast, baseline_model)
    baseline_results = evaluate_system(
        queries,
        corpus_asts,
        baseline_encoder,
        baseline_index,
        k_values=args.k_values,
        threshold=args.threshold
    )

    # Load Tree-LSTM model
    tree_lstm_model = load_tree_lstm_model(tree_lstm_path, device=args.device)

    # Create AST to Graph converter
    converter = ASTToGraphConverter()

    # Build Tree-LSTM index
    tree_lstm_index = build_tree_lstm_index(corpus_asts, tree_lstm_model, converter, device=args.device)

    # Evaluate Tree-LSTM
    logger.info("Evaluating Tree-LSTM...")
    tree_lstm_encoder = lambda ast: encode_with_tree_lstm(ast, tree_lstm_model, converter, device=args.device)
    tree_lstm_results = evaluate_system(
        queries,
        corpus_asts,
        tree_lstm_encoder,
        tree_lstm_index,
        k_values=args.k_values,
        threshold=args.threshold
    )

    # Generate comparison report
    generate_comparison_report(baseline_results, tree_lstm_results, output_dir)

    logger.info("")
    logger.info("=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
