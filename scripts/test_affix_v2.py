#!/usr/bin/env python3
"""
Test Affix Transforms V2 Model
==============================
Run after training completes to validate the model fixes embedding collapse.

Usage:
    python scripts/test_affix_v2.py
    python scripts/test_affix_v2.py --rebuild-index  # Also rebuild FAISS index
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_models():
    """Load root embeddings and affix transforms."""
    root_path = PROJECT_ROOT / "models/root_embeddings/best_model.pt"
    affix_path = PROJECT_ROOT / "models/affix_transforms_v2/best_model.pt"

    if not root_path.exists():
        print(f"ERROR: Root model not found: {root_path}")
        return None, None

    if not affix_path.exists():
        print(f"ERROR: Affix V2 model not found: {affix_path}")
        print("Is training still running?")
        return None, None

    root_model = torch.load(root_path, map_location='cpu', weights_only=False)
    affix_model = torch.load(affix_path, map_location='cpu', weights_only=False)

    return root_model, affix_model


def show_model_info(root_model, affix_model):
    """Display model information."""
    print("=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)

    print(f"\nRoot Embeddings:")
    print(f"  Vocabulary size: {root_model['vocab_size']:,} roots")
    print(f"  Embedding dim: {root_model['embedding_dim']}")
    print(f"  Correlation: {root_model.get('correlation', 'N/A')}")

    print(f"\nAffix Transforms V2:")
    print(f"  Embedding dim: {affix_model['embedding_dim']}")
    print(f"  Rank: {affix_model['rank']}")
    print(f"  Prefixes: {len(affix_model['prefixes'])}")
    print(f"  Suffixes: {len(affix_model['suffixes'])}")
    print(f"  Training epochs: {affix_model['epoch']}")
    print(f"  Final loss: {affix_model['loss']:.4f}")

    # Anti-collapse metrics
    if 'metrics' in affix_model:
        m = affix_model['metrics']
        print(f"\nAnti-Collapse Metrics:")
        if 'mal_mean_sim' in m:
            status = "PASS" if m['mal_mean_sim'] < 0.5 else "FAIL"
            print(f"  mal_mean_sim: {m['mal_mean_sim']:.4f} ({status}, target < 0.5)")
        if 'mal_dist' in m:
            print(f"  mal_dist: {m['mal_dist']:.4f}")
        if 'embedding_diversity' in m:
            print(f"  embedding_diversity: {m['embedding_diversity']:.4f}")


def test_transform_behavior(root_model, affix_model):
    """Test that transforms behave correctly."""
    print("\n" + "=" * 60)
    print("TRANSFORM BEHAVIOR TESTS")
    print("=" * 60)

    # Get embeddings and vocab from model checkpoint
    # Root model stores embeddings in model_state_dict['embeddings.weight']
    embeddings = root_model['model_state_dict']['embeddings.weight'].numpy()
    root_to_idx = root_model['root_to_idx']

    # Get transforms from model_state_dict
    prefix_names = affix_model['prefixes']
    suffix_names = affix_model['suffixes']
    state_dict = affix_model['model_state_dict']
    rank = affix_model['rank']
    dim = affix_model['embedding_dim']

    def get_root_embedding(root):
        if root in root_to_idx:
            idx = root_to_idx[root]
            return torch.tensor(embeddings[idx])
        return None

    def get_prefix_transform(prefix_name):
        """Get down/up weights for a prefix from state_dict."""
        down_key = f'prefix_transforms.{prefix_name}.down.weight'
        up_key = f'prefix_transforms.{prefix_name}.up.weight'
        if down_key in state_dict and up_key in state_dict:
            return {
                'down': state_dict[down_key],
                'up': state_dict[up_key]
            }
        return None

    def apply_transform(emb, transform_data):
        """Apply low-rank transform: x + up(down(x))"""
        down = transform_data['down']  # [rank, dim]
        up = transform_data['up']      # [dim, rank]
        projected = emb @ down.T  # [rank]
        delta = projected @ up.T  # [dim]
        return emb + delta

    # Test 1: mal- should create different but related embeddings
    print("\n1. Testing mal- prefix (antonym transformation):")
    test_roots = ['bon', 'grand', 'jun', 'fort', 'bel']

    mal_transform = get_prefix_transform('mal')
    if mal_transform:
        results = []

        for root in test_roots:
            root_emb = get_root_embedding(root)
            if root_emb is not None:
                mal_emb = apply_transform(root_emb, mal_transform)

                # Cosine similarity
                sim = torch.nn.functional.cosine_similarity(
                    root_emb.unsqueeze(0), mal_emb.unsqueeze(0)
                ).item()

                # Distance
                dist = torch.norm(mal_emb - root_emb).item()

                results.append((root, sim, dist))
                print(f"  {root} -> mal{root}: sim={sim:.3f}, dist={dist:.3f}")

        if results:
            avg_sim = np.mean([r[1] for r in results])
            avg_dist = np.mean([r[2] for r in results])
            print(f"\n  Average: sim={avg_sim:.3f}, dist={avg_dist:.3f}")

            if avg_sim < 0.8:
                print("  PASS: mal- creates distinct embeddings (not collapsed)")
            else:
                print("  WARNING: mal- embeddings too similar to roots")
    else:
        print("  WARNING: mal- prefix not found in model")

    # Test 2: Different roots with same affix should stay apart
    print("\n2. Testing separation (different roots with mal-):")
    if mal_transform:
        mal_embeddings = []

        for root in test_roots:
            root_emb = get_root_embedding(root)
            if root_emb is not None:
                mal_emb = apply_transform(root_emb, mal_transform)
                mal_embeddings.append((root, mal_emb))

        if len(mal_embeddings) >= 2:
            # Compute pairwise similarities
            sims = []
            for i, (r1, e1) in enumerate(mal_embeddings):
                for j, (r2, e2) in enumerate(mal_embeddings):
                    if i < j:
                        sim = torch.nn.functional.cosine_similarity(
                            e1.unsqueeze(0), e2.unsqueeze(0)
                        ).item()
                        sims.append(sim)
                        print(f"  mal{r1} vs mal{r2}: sim={sim:.3f}")

            if sims:
                mean_sim = np.mean(sims)
                print(f"\n  Mean pairwise similarity: {mean_sim:.3f}")

                if mean_sim < 0.5:
                    print("  PASS: Different mal- words are NOT collapsed together")
                else:
                    print("  FAIL: Different mal- words are too similar (collapse!)")


def test_rag_queries(rebuild_index=False):
    """Test RAG retrieval with the new model."""
    print("\n" + "=" * 60)
    print("RAG RETRIEVAL TESTS")
    print("=" * 60)

    index_dir = PROJECT_ROOT / "data/corpus_index_compositional"

    if rebuild_index or not (index_dir / "embeddings.npy").exists():
        print("\nRebuilding index with V2 model...")
        import subprocess
        result = subprocess.run(
            ["./scripts/run_compositional_indexing.sh", "--fresh"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Index rebuild failed: {result.stderr}")
            return
        print("Index rebuilt successfully")

    # Test queries
    test_queries = [
        "malbona",      # Opposite of good
        "malgranda",    # Opposite of big
        "malhundo",     # Bad dog (the original failing case)
        "rehavi",       # To have again
        "hundejo",      # Dog place / kennel
    ]

    print("\nTest queries (check these manually after index rebuild):")
    for q in test_queries:
        print(f"  - {q}")

    print("\nTo test interactively, run:")
    print("  python scripts/demo_rag_compositional.py -i")


def main():
    parser = argparse.ArgumentParser(description="Test Affix V2 Model")
    parser.add_argument("--rebuild-index", action="store_true",
                       help="Also rebuild the FAISS index")
    args = parser.parse_args()

    print("=" * 60)
    print("AFFIX TRANSFORMS V2 - VALIDATION")
    print("=" * 60)

    # Load models
    root_model, affix_model = load_models()
    if root_model is None or affix_model is None:
        sys.exit(1)

    # Show info
    show_model_info(root_model, affix_model)

    # Test transforms
    test_transform_behavior(root_model, affix_model)

    # Test RAG
    if args.rebuild_index:
        test_rag_queries(rebuild_index=True)
    else:
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("\n1. Rebuild index:")
        print("   ./scripts/run_compositional_indexing.sh --fresh")
        print("\n2. Test RAG interactively:")
        print("   python scripts/demo_rag_compositional.py -i")
        print("\n3. Or run this script with --rebuild-index:")
        print("   python scripts/test_affix_v2.py --rebuild-index")


if __name__ == "__main__":
    main()
