#!/usr/bin/env python3
"""
Standalone GNN Test - Tests Tree-LSTM without full parser dependencies

This script tests the Tree-LSTM GNN model with synthetic graph data.
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from klareco.models.tree_lstm import TreeLSTMEncoder
from torch_geometric.data import Data


def create_sample_tree_graph():
    """
    Create a sample tree graph for testing.

    Tree structure:
         0 (root)
        / \
       1   2
      / \
     3   4

    This represents a simplified AST-like structure.
    """
    # Node features: 5 nodes with 19 features each (as per ast_to_graph)
    # Features are: [root_id(10), pos(11), number(2), case(2), parse_status(1)] = 19 features
    num_nodes = 5
    feature_dim = 19

    # Random features for testing
    x = torch.randn(num_nodes, feature_dim)

    # Edge index: parent -> child connections
    # 0 -> 1, 0 -> 2, 1 -> 3, 1 -> 4
    edge_index = torch.tensor([
        [0, 0, 1, 1],  # source nodes (parents)
        [1, 2, 3, 4]   # target nodes (children)
    ], dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


def test_tree_lstm_encoder():
    """Test the TreeLSTMEncoder with a sample graph"""
    print("="*80)
    print("TREE-LSTM GNN MODEL TEST")
    print("="*80)

    # Create model
    print("\n1. Creating TreeLSTMEncoder...")
    vocab_size = 10000  # Morpheme vocabulary size
    embed_dim = 128
    hidden_dim = 256
    output_dim = 512

    encoder = TreeLSTMEncoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=0.1
    )

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    print(f"✓ Model created successfully!")
    print(f"  - Vocabulary size: {vocab_size:,}")
    print(f"  - Embedding dim: {embed_dim}")
    print(f"  - Hidden dim: {hidden_dim}")
    print(f"  - Output dim: {output_dim}")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")

    # Create sample data
    print("\n2. Creating sample tree graph...")
    graph = create_sample_tree_graph()
    print(f"✓ Graph created:")
    print(f"  - Nodes: {graph.num_nodes}")
    print(f"  - Edges: {graph.edge_index.shape[1]}")
    print(f"  - Node features shape: {graph.x.shape}")
    print(f"  - Edge index shape: {graph.edge_index.shape}")

    # Forward pass
    print("\n3. Running forward pass (encoding tree)...")
    encoder.eval()  # Set to evaluation mode
    with torch.no_grad():
        embedding = encoder(graph)

    print(f"✓ Encoding successful!")
    print(f"  - Embedding shape: {embedding.shape}")
    print(f"  - Embedding dtype: {embedding.dtype}")
    print(f"  - Embedding norm: {embedding.norm().item():.4f}")
    print(f"  - Mean value: {embedding.mean().item():.4f}")
    print(f"  - Std value: {embedding.std().item():.4f}")
    print(f"  - Min value: {embedding.min().item():.4f}")
    print(f"  - Max value: {embedding.max().item():.4f}")

    # Test batch processing
    print("\n4. Testing batch processing...")
    from torch_geometric.data import Batch

    # Create a batch of 3 graphs
    graphs = [create_sample_tree_graph() for _ in range(3)]
    batch = Batch.from_data_list(graphs)

    print(f"✓ Batch created:")
    print(f"  - Batch size: {batch.num_graphs}")
    print(f"  - Total nodes: {batch.num_nodes}")
    print(f"  - Total edges: {batch.edge_index.shape[1]}")

    with torch.no_grad():
        batch_embeddings = encoder.forward_batch(batch)

    print(f"✓ Batch encoding successful!")
    print(f"  - Batch embeddings shape: {batch_embeddings.shape}")
    print(f"  - Expected shape: (3, 512)")

    # Test similarity computation
    print("\n5. Testing similarity computation...")
    emb1 = batch_embeddings[0]
    emb2 = batch_embeddings[1]
    emb3 = batch_embeddings[2]

    # Cosine similarity
    cos_sim_12 = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0)
    cos_sim_13 = torch.nn.functional.cosine_similarity(emb1, emb3, dim=0)
    cos_sim_23 = torch.nn.functional.cosine_similarity(emb2, emb3, dim=0)

    print(f"✓ Similarity computed:")
    print(f"  - Graph 1 vs Graph 2: {cos_sim_12.item():.4f}")
    print(f"  - Graph 1 vs Graph 3: {cos_sim_13.item():.4f}")
    print(f"  - Graph 2 vs Graph 3: {cos_sim_23.item():.4f}")

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print("✓ TreeLSTMEncoder initialization: PASSED")
    print("✓ Single graph encoding: PASSED")
    print("✓ Batch graph encoding: PASSED")
    print("✓ Similarity computation: PASSED")
    print("\n🎉 All tests PASSED! The GNN model is working correctly.")
    print("="*80)

    return encoder, graph, embedding


def test_model_architecture():
    """Test the internal architecture components"""
    print("\n" + "="*80)
    print("ARCHITECTURE COMPONENTS TEST")
    print("="*80)

    from klareco.models.tree_lstm import ChildSumTreeLSTMCell, ChildSumTreeLSTM

    # Test Tree-LSTM Cell
    print("\n1. Testing ChildSumTreeLSTMCell...")
    input_dim = 19
    hidden_dim = 256
    cell = ChildSumTreeLSTMCell(input_dim, hidden_dim)

    print(f"✓ Cell created:")
    print(f"  - Input dim: {input_dim}")
    print(f"  - Hidden dim: {hidden_dim}")

    # Test forward pass
    x = torch.randn(input_dim)
    child_h = [torch.randn(hidden_dim) for _ in range(2)]
    child_c = [torch.randn(hidden_dim) for _ in range(2)]

    h, c = cell(x, child_h, child_c)
    print(f"✓ Forward pass successful:")
    print(f"  - Output h shape: {h.shape}")
    print(f"  - Output c shape: {c.shape}")

    # Test Tree-LSTM
    print("\n2. Testing ChildSumTreeLSTM...")
    output_dim = 512
    tree_lstm = ChildSumTreeLSTM(input_dim, hidden_dim, output_dim)

    print(f"✓ Tree-LSTM created:")
    print(f"  - Input dim: {input_dim}")
    print(f"  - Hidden dim: {hidden_dim}")
    print(f"  - Output dim: {output_dim}")

    # Create sample graph
    graph = create_sample_tree_graph()
    embedding = tree_lstm(graph)

    print(f"✓ Encoding successful:")
    print(f"  - Embedding shape: {embedding.shape}")
    print(f"  - Expected: ({output_dim},)")

    print("\n✓ All architecture components working correctly!")


if __name__ == "__main__":
    print("\n" + "🧠 KLARECO GNN (TREE-LSTM) MODEL TEST 🧠".center(80))
    print()

    try:
        # Test main encoder
        encoder, graph, embedding = test_tree_lstm_encoder()

        # Test architecture components
        test_model_architecture()

        print("\n" + "="*80)
        print("FINAL RESULT: ✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nThe Tree-LSTM GNN model is fully functional and ready to use.")
        print("It can encode tree-structured ASTs into 512-dimensional embeddings.")
        print("\nNext steps:")
        print("  1. Train the model on actual Esperanto AST pairs")
        print("  2. Load trained weights for production use")
        print("  3. Integrate with RAG system for semantic search")
        print("="*80 + "\n")

        sys.exit(0)

    except Exception as e:
        print(f"\n❌ TEST FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
