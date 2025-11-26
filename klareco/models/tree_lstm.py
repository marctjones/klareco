"""
Tree-LSTM (Long Short-Term Memory) for AST Encoding.

Implementation of Child-Sum Tree-LSTM for encoding Klareco's tree-structured ASTs
into fixed-dimensional semantic embeddings.

References:
    Tai et al. (2015) - "Improved Semantic Representations From Tree-Structured
    Long Short-Term Memory Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import MessagePassing
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


class ChildSumTreeLSTMCell(nn.Module):
    """
    Child-Sum Tree-LSTM cell.

    Composes node representation from its children using LSTM-style gates.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize Tree-LSTM cell.

        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden state
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input gate
        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.U_i = nn.Linear(hidden_dim, hidden_dim)

        # Forget gates (one per child, but we'll use shared weights)
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim)

        # Output gate
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim)

        # Cell input
        self.W_u = nn.Linear(input_dim, hidden_dim)
        self.U_u = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        child_h: List[torch.Tensor],
        child_c: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Tree-LSTM cell.

        Args:
            x: Node feature vector (input_dim,)
            child_h: List of child hidden states [(hidden_dim,), ...]
            child_c: List of child cell states [(hidden_dim,), ...]

        Returns:
            (h, c): Hidden state and cell state for this node
        """
        # Sum child hidden states
        if child_h:
            h_sum = torch.stack(child_h).sum(dim=0)
        else:
            h_sum = torch.zeros(self.hidden_dim, device=x.device)

        # Input gate
        i = torch.sigmoid(self.W_i(x) + self.U_i(h_sum))

        # Output gate
        o = torch.sigmoid(self.W_o(x) + self.U_o(h_sum))

        # Cell input
        u = torch.tanh(self.W_u(x) + self.U_u(h_sum))

        # Forget gates (one per child)
        if child_h:
            f_list = []
            for child_h_k in child_h:
                f_k = torch.sigmoid(self.W_f(x) + self.U_f(child_h_k))
                f_list.append(f_k)

            # Cell state
            c = i * u + sum(f_k * child_c_k for f_k, child_c_k in zip(f_list, child_c))
        else:
            # Leaf node
            c = i * u

        # Hidden state
        h = o * torch.tanh(c)

        return h, c


class ChildSumTreeLSTM(nn.Module):
    """
    Child-Sum Tree-LSTM for encoding tree-structured data.

    Processes a tree bottom-up, composing each node's representation
    from its children's hidden states.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        """
        Initialize Tree-LSTM.

        Args:
            input_dim: Dimension of node feature vectors
            hidden_dim: Dimension of LSTM hidden state
            output_dim: Dimension of final sentence embedding
            dropout: Dropout probability
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Tree-LSTM cell
        self.cell = ChildSumTreeLSTMCell(input_dim, hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward_node(
        self,
        node_idx: int,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        h_dict: dict,
        c_dict: dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single node (recursively processes children first).

        Args:
            node_idx: Index of current node
            x: All node features (num_nodes, input_dim)
            edge_index: Edge connectivity (2, num_edges)
            h_dict: Dictionary to store hidden states
            c_dict: Dictionary to store cell states

        Returns:
            (h, c): Hidden and cell state for this node
        """
        # Check if already computed
        if node_idx in h_dict:
            return h_dict[node_idx], c_dict[node_idx]

        # Find children (nodes that this node points to)
        # Edge format: [source, target] where source -> target
        # For tree, parent -> child, so we want all targets where source == node_idx
        child_mask = edge_index[0] == node_idx
        child_indices = edge_index[1, child_mask].tolist()

        # Recursively compute children
        child_h = []
        child_c = []
        for child_idx in child_indices:
            h_child, c_child = self.forward_node(child_idx, x, edge_index, h_dict, c_dict)
            child_h.append(h_child)
            child_c.append(c_child)

        # Compute this node
        node_features = x[node_idx]
        h, c = self.cell(node_features, child_h, child_c)

        # Store
        h_dict[node_idx] = h
        c_dict[node_idx] = c

        return h, c

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through Tree-LSTM.

        Args:
            data: PyG Data object with x (node features) and edge_index

        Returns:
            Sentence embedding (output_dim,)
        """
        x = data.x
        edge_index = data.edge_index

        # Find root node (node with no incoming edges)
        # For our ASTs, root is always node 0 (sentence node)
        root_idx = 0

        # Process tree bottom-up
        h_dict = {}
        c_dict = {}
        h_root, c_root = self.forward_node(root_idx, x, edge_index, h_dict, c_dict)

        # Project to output dimension
        h_root = self.dropout(h_root)
        embedding = self.output_proj(h_root)

        return embedding

    def forward_batch(self, batch: Batch) -> torch.Tensor:
        """
        Forward pass on a batch of graphs.

        Args:
            batch: PyG Batch object

        Returns:
            Batch embeddings (batch_size, output_dim)
        """
        embeddings = []

        # Process each graph separately (Tree-LSTM is not easily batchable)
        for i in range(batch.num_graphs):
            # Extract single graph
            mask = batch.batch == i
            x_i = batch.x[mask]

            # Remap edge indices to local node indices
            global_to_local = {global_idx.item(): local_idx
                             for local_idx, global_idx in enumerate(torch.where(mask)[0])}

            edge_mask = torch.isin(batch.edge_index[0], torch.where(mask)[0])
            edge_index_i = batch.edge_index[:, edge_mask]

            # Remap to local indices
            edge_index_local = torch.tensor([
                [global_to_local[src.item()], global_to_local[dst.item()]]
                for src, dst in edge_index_i.t()
            ], dtype=torch.long).t()

            # Create Data object for this graph
            data_i = Data(x=x_i, edge_index=edge_index_local)

            # Forward pass
            emb_i = self.forward(data_i)
            embeddings.append(emb_i)

        return torch.stack(embeddings)


class TreeLSTMEncoder(nn.Module):
    """
    Complete Tree-LSTM encoder with morpheme embeddings.

    Wraps ChildSumTreeLSTM with input embedding layer for morphemes.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 512,
        dropout: float = 0.1
    ):
        """
        Initialize encoder.

        Args:
            vocab_size: Size of morpheme vocabulary
            embed_dim: Dimension of morpheme embeddings
            hidden_dim: Dimension of LSTM hidden state
            output_dim: Dimension of sentence embeddings
            dropout: Dropout probability
        """
        super().__init__()

        # Store dimensions
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Morpheme embedding layer
        self.embed = nn.Embedding(vocab_size, embed_dim)

        # Feature dimension: embed_dim + POS (11) + number (2) + case (2) + parse_status (1) = embed_dim + 16
        # But our current features are just concatenated IDs, so we need to adjust
        # For now, use the raw feature dimension from ast_to_graph (19)
        input_dim = 19  # From ast_to_graph.py extract_node_features

        # Tree-LSTM
        self.tree_lstm = ChildSumTreeLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Encode AST graph into sentence embedding.

        Args:
            data: PyG Data object

        Returns:
            Sentence embedding (output_dim,)
        """
        # Note: In production, we would embed the morpheme IDs here
        # For now, we use the raw features from ast_to_graph
        return self.tree_lstm(data)

    def forward_batch(self, batch: Batch) -> torch.Tensor:
        """
        Encode batch of AST graphs.

        Args:
            batch: PyG Batch object

        Returns:
            Batch embeddings (batch_size, output_dim)
        """
        return self.tree_lstm.forward_batch(batch)


def main():
    """Test Tree-LSTM implementation."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from klareco.parser import parse
    from klareco.ast_to_graph import ASTToGraphConverter

    print("Testing Tree-LSTM Implementation")
    print("=" * 70)

    # Parse a sentence
    sentence = "La hundo vidas la grandan katon."
    print(f"Sentence: {sentence}\n")

    ast = parse(sentence)
    print(f"AST parsed: {ast['tipo']}\n")

    # Convert to graph
    converter = ASTToGraphConverter()
    graph = converter.ast_to_graph(ast)
    print(f"Graph created:")
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Edges: {graph.edge_index.shape[1]}")
    print(f"  Node features: {graph.x.shape}\n")

    # Create Tree-LSTM encoder
    encoder = TreeLSTMEncoder(
        vocab_size=1000,  # Placeholder
        embed_dim=128,
        hidden_dim=256,
        output_dim=512
    )
    print("Tree-LSTM Encoder created:")
    print(f"  Hidden dim: 256")
    print(f"  Output dim: 512\n")

    # Encode
    with torch.no_grad():
        embedding = encoder(graph)

    print(f"Sentence embedding:")
    print(f"  Shape: {embedding.shape}")
    print(f"  Norm: {embedding.norm().item():.4f}")
    print(f"  First 10 values: {embedding[:10].tolist()}")

    print("\n✅ Tree-LSTM implementation successful!")


if __name__ == '__main__':
    main()
