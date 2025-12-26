#!/usr/bin/env python3
"""
Quick evaluation of semantic similarity model.

Run while training is in progress or after completion.
Uses the test set (held out from training).

Usage:
    python scripts/evaluate_semantic_model.py
    python scripts/evaluate_semantic_model.py --checkpoint models/semantic_similarity/best_model.prev.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.embeddings.compositional import CompositionalEmbedding
from klareco import parser as eo_parser_module
from klareco.models.tree_lstm import TreeLSTMEncoder
from klareco.ast_to_graph import ASTToGraphConverter


def load_model(checkpoint_path: Path, vocab_dir: Path, device: torch.device):
    """Load model from checkpoint."""
    # Load vocabularies
    compositional_embedding = CompositionalEmbedding.from_vocabulary_files(
        vocab_dir,
        embed_dim=128,
    )

    # Load checkpoint first to detect input_dim
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    # Detect input_dim from checkpoint weights (W_i weight shape is [hidden_dim, input_dim])
    w_i_shape = checkpoint['model_state_dict']['tree_lstm.cell.W_i.weight'].shape
    checkpoint_input_dim = w_i_shape[1]

    # Create converter - check if checkpoint was trained with old behavior (input_dim=19)
    # Old behavior: compositional_embedding was accidentally stored in embed_dim, not used
    # New behavior: compositional_embedding properly used, input_dim=144
    if checkpoint_input_dim == 19:
        # Old checkpoint - create converter without compositional embedding to match
        print(f"Note: Checkpoint uses legacy input_dim=19 (without compositional features)")
        converter = ASTToGraphConverter(128)  # No compositional embedding, just embed_dim
    else:
        # New checkpoint - use compositional embedding
        converter = ASTToGraphConverter(compositional_embedding)
        compositional_embedding = compositional_embedding.to(device)
        converter.compositional_embedding = compositional_embedding

    # Create model with checkpoint's input_dim
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


def evaluate_test_set(model, converter, test_file: Path, device: torch.device, max_samples: int = None):
    """Evaluate on test set."""
    all_pred_sims = []
    all_target_sims = []
    failed = 0

    with open(test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if max_samples:
        lines = lines[:max_samples]

    print(f"\nEvaluating on {len(lines):,} test samples...")

    with torch.no_grad():
        for line in tqdm(lines, desc="Evaluating"):
            record = json.loads(line)
            sent_a = record['sentence_a']
            sent_b = record['sentence_b']
            target_sim = record['similarity']

            try:
                ast_a = eo_parser_module.parse(sent_a)
                ast_b = eo_parser_module.parse(sent_b)

                if ast_a is None or ast_b is None:
                    failed += 1
                    continue

                graph_a = converter.ast_to_graph(ast_a)
                graph_b = converter.ast_to_graph(ast_b)

                if graph_a is None or graph_b is None:
                    failed += 1
                    continue

                graph_a = graph_a.to(device)
                graph_b = graph_b.to(device)

                emb_a = model(graph_a)
                emb_b = model(graph_b)

                # Compute cosine similarity
                emb_a = F.normalize(emb_a, dim=-1)
                emb_b = F.normalize(emb_b, dim=-1)
                pred_sim = torch.sum(emb_a * emb_b).item()

                all_pred_sims.append(pred_sim)
                all_target_sims.append(target_sim)

            except Exception as e:
                failed += 1
                if failed <= 3:  # Show first few errors
                    print(f"\n  Error on sample: {e}")
                    print(f"    Sentence A: {sent_a[:50]}...")
                continue

    # Compute correlation
    correlation = np.corrcoef(all_pred_sims, all_target_sims)[0, 1]

    # Compute MSE
    mse = np.mean((np.array(all_pred_sims) - np.array(all_target_sims)) ** 2)

    return {
        'correlation': correlation,
        'mse': mse,
        'n_samples': len(all_pred_sims),
        'n_failed': failed,
    }


def interactive_demo(model, converter, device: torch.device):
    """Interactive demo to test sentence pairs."""
    print("\n" + "=" * 60)
    print("Interactive Semantic Similarity Demo")
    print("=" * 60)
    print("Enter two Esperanto sentences to compute their similarity.")
    print("Type 'quit' to exit.\n")

    while True:
        sent_a = input("Sentence A: ").strip()
        if sent_a.lower() == 'quit':
            break

        sent_b = input("Sentence B: ").strip()
        if sent_b.lower() == 'quit':
            break

        try:
            ast_a = eo_parser_module.parse(sent_a)
            ast_b = eo_parser_module.parse(sent_b)

            if ast_a is None or ast_b is None:
                print("  Error: Failed to parse one or both sentences\n")
                continue

            graph_a = converter.ast_to_graph(ast_a)
            graph_b = converter.ast_to_graph(ast_b)

            if graph_a is None or graph_b is None:
                print("  Error: Failed to convert to graph\n")
                continue

            with torch.no_grad():
                graph_a = graph_a.to(device)
                graph_b = graph_b.to(device)

                emb_a = model(graph_a)
                emb_b = model(graph_b)

                emb_a = F.normalize(emb_a, dim=-1)
                emb_b = F.normalize(emb_b, dim=-1)
                similarity = torch.sum(emb_a * emb_b).item()

            print(f"\n  Similarity: {similarity:.4f}")
            if similarity > 0.8:
                print("  Interpretation: Very similar (paraphrases)")
            elif similarity > 0.5:
                print("  Interpretation: Somewhat similar")
            elif similarity > 0.3:
                print("  Interpretation: Slightly related")
            else:
                print("  Interpretation: Different meanings")
            print()

        except Exception as e:
            print(f"  Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate semantic similarity model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/semantic_similarity/best_model.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("data/similarity_pairs_test.jsonl"),
        help="Test data file",
    )
    parser.add_argument(
        "--vocab-dir",
        type=Path,
        default=Path("data/vocabularies"),
        help="Vocabulary directory",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max test samples (default: all)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive demo after evaluation",
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
        print("\nAvailable checkpoints:")
        for f in Path("models/semantic_similarity").glob("*.pt"):
            print(f"  {f}")
        sys.exit(1)

    print(f"Loading model from: {args.checkpoint}")
    print(f"Device: {device}")

    # Load model
    model, converter, checkpoint = load_model(args.checkpoint, args.vocab_dir, device)

    print(f"\nCheckpoint info:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val correlation: {checkpoint['val_correlation']:.4f}")
    print(f"  Val loss: {checkpoint['val_loss']:.4f}")

    # Evaluate on test set
    if args.test_file.exists():
        results = evaluate_test_set(model, converter, args.test_file, device, args.max_samples)

        print(f"\nTest Set Results:")
        print(f"  Samples evaluated: {results['n_samples']:,}")
        print(f"  Failed to parse: {results['n_failed']:,}")
        print(f"  Pearson correlation: {results['correlation']:.4f}")
        print(f"  MSE: {results['mse']:.4f}")

        if results['correlation'] > 0.8:
            print("\n  ✓ Excellent! Correlation > 0.8")
        elif results['correlation'] > 0.6:
            print("\n  ✓ Good. Correlation > 0.6")
        else:
            print("\n  Model still training, correlation < 0.6")
    else:
        print(f"\nTest file not found: {args.test_file}")

    # Interactive demo
    if args.interactive:
        interactive_demo(model, converter, device)


if __name__ == "__main__":
    main()
