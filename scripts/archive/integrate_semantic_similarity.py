#!/usr/bin/env python3
"""
Integrate Semantic Similarity Model into Retrieval Pipeline.

This script demonstrates how to use the trained semantic similarity model
for retrieval tasks. It can be used as a reference for integration.

Usage:
    python scripts/integrate_semantic_similarity.py --query "Kio estas hobito?"
    python scripts/integrate_semantic_similarity.py --interactive
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.embeddings.compositional import CompositionalEmbedding
from klareco import parser as eo_parser_module
from klareco.models.tree_lstm import TreeLSTMEncoder
from klareco.ast_to_graph import ASTToGraphConverter


class SemanticRetriever:
    """Retriever using semantic similarity model."""

    def __init__(
        self,
        model: TreeLSTMEncoder,
        converter: ASTToGraphConverter,
        corpus_path: Path,
        device: torch.device,
        max_corpus_size: int = None,
    ):
        self.model = model
        self.converter = converter
        self.device = device
        self.corpus = []
        self.corpus_embeddings = None

        self._load_corpus(corpus_path, max_corpus_size)

    def _load_corpus(self, corpus_path: Path, max_size: int = None):
        """Load corpus and compute embeddings."""
        print(f"Loading corpus from {corpus_path}...")

        sentences = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_size and i >= max_size:
                    break
                entry = json.loads(line)
                sentences.append(entry.get('text', entry.get('sentence_a', '')))

        print(f"Computing embeddings for {len(sentences)} sentences...")
        embeddings = []
        valid_sentences = []

        self.model.eval()
        with torch.no_grad():
            for sent in sentences:
                try:
                    ast = eo_parser_module.parse(sent)
                    if ast is None:
                        continue

                    graph = self.converter.ast_to_graph(ast)
                    if graph is None:
                        continue

                    graph = graph.to(self.device)
                    emb = self.model(graph)
                    emb = F.normalize(emb, dim=-1)

                    embeddings.append(emb.cpu())
                    valid_sentences.append(sent)
                except Exception:
                    continue

        self.corpus = valid_sentences
        self.corpus_embeddings = torch.stack(embeddings) if embeddings else None
        print(f"Indexed {len(self.corpus)} sentences")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve most similar sentences to query."""
        if self.corpus_embeddings is None or len(self.corpus) == 0:
            return []

        # Encode query
        try:
            ast = eo_parser_module.parse(query)
            if ast is None:
                return []

            graph = self.converter.ast_to_graph(ast)
            if graph is None:
                return []

            self.model.eval()
            with torch.no_grad():
                graph = graph.to(self.device)
                query_emb = self.model(graph)
                query_emb = F.normalize(query_emb, dim=-1)

            # Compute similarities
            similarities = torch.matmul(self.corpus_embeddings, query_emb.cpu())
            similarities = similarities.numpy()

            # Get top-k
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                results.append({
                    'text': self.corpus[idx],
                    'similarity': float(similarities[idx]),
                    'rank': len(results) + 1,
                })

            return results

        except Exception as e:
            print(f"Error processing query: {e}")
            return []


def load_model(checkpoint_path: Path, vocab_dir: Path, device: torch.device):
    """Load model from checkpoint."""
    # Load vocabularies
    compositional_embedding = CompositionalEmbedding.from_vocabulary_files(
        vocab_dir,
        embed_dim=128,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    # Detect input_dim from checkpoint
    w_i_shape = checkpoint['model_state_dict']['tree_lstm.cell.W_i.weight'].shape
    checkpoint_input_dim = w_i_shape[1]

    # Create converter based on checkpoint type
    if checkpoint_input_dim == 19:
        # Legacy checkpoint without compositional features
        converter = ASTToGraphConverter(128)
    else:
        # New checkpoint with compositional features
        converter = ASTToGraphConverter(compositional_embedding)
        compositional_embedding = compositional_embedding.to(device)
        converter.compositional_embedding = compositional_embedding

    # Create model
    model = TreeLSTMEncoder(
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        input_dim=checkpoint_input_dim,
        use_compositional=True,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, converter, checkpoint


def interactive_demo(retriever: SemanticRetriever):
    """Interactive retrieval demo."""
    print("\n" + "=" * 60)
    print("Semantic Retrieval Demo")
    print("=" * 60)
    print("Enter an Esperanto query to find similar sentences.")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("Query: ").strip()
        if query.lower() == 'quit':
            break

        results = retriever.retrieve(query, top_k=5)

        if not results:
            print("  No results found.\n")
            continue

        print(f"\nTop {len(results)} results:")
        for r in results:
            print(f"  [{r['similarity']:.3f}] {r['text']}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Semantic similarity retrieval")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/semantic_similarity/best_model.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/similarity_pairs_train.jsonl"),
        help="Corpus to search (JSONL with 'text' or 'sentence_a' field)",
    )
    parser.add_argument(
        "--vocab-dir",
        type=Path,
        default=Path("data/vocabularies"),
        help="Vocabulary directory",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single query to run",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive demo",
    )
    parser.add_argument(
        "--max-corpus",
        type=int,
        default=5000,
        help="Maximum corpus size to load",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto, cpu, cuda)",
    )

    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Check checkpoint exists
    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print(f"Loading model from: {args.checkpoint}")
    print(f"Device: {device}")

    # Load model
    model, converter, checkpoint = load_model(args.checkpoint, args.vocab_dir, device)

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Val correlation: {checkpoint['val_correlation']:.4f}")

    # Create retriever
    retriever = SemanticRetriever(
        model=model,
        converter=converter,
        corpus_path=args.corpus,
        device=device,
        max_corpus_size=args.max_corpus,
    )

    # Run query or interactive mode
    if args.query:
        results = retriever.retrieve(args.query, top_k=args.top_k)
        print(f"\nQuery: {args.query}")
        print(f"\nTop {len(results)} results:")
        for r in results:
            print(f"  [{r['similarity']:.3f}] {r['text']}")

    elif args.interactive:
        interactive_demo(retriever)

    else:
        # Demo queries
        demo_queries = [
            "Kio estas hundo?",
            "Mi amas vin.",
            "La suno brilas.",
            "Kie estas la libro?",
        ]

        print("\n" + "=" * 60)
        print("Demo Queries")
        print("=" * 60)

        for query in demo_queries:
            results = retriever.retrieve(query, top_k=3)
            print(f"\nQuery: {query}")
            for r in results:
                print(f"  [{r['similarity']:.3f}] {r['text']}")


if __name__ == "__main__":
    main()
