"""
Contrastive Learning DataLoader for Tree-LSTM Training.

Loads positive and negative pairs from training data and converts ASTs to graphs
for batch processing with PyTorch Geometric.
"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

from .ast_to_graph import ASTToGraphConverter


class ContrastivePairDataset(Dataset):
    """
    Dataset for contrastive learning pairs.

    Each item is a tuple: (graph1, graph2, label)
    - label = 1 for positive pairs (similar)
    - label = 0 for negative pairs (dissimilar)
    """

    def __init__(
        self,
        positive_pairs_file: Path,
        negative_pairs_file: Path,
        max_pairs: Optional[int] = None,
        converter: Optional[ASTToGraphConverter] = None
    ):
        """
        Initialize dataset.

        Args:
            positive_pairs_file: Path to positive pairs JSONL
            negative_pairs_file: Path to negative pairs JSONL
            max_pairs: Maximum pairs to load (None = all)
            converter: AST to graph converter (creates new one if None)
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch-geometric is required")

        self.converter = converter or ASTToGraphConverter()
        self.pairs = []

        # Load positive pairs
        self._load_pairs(positive_pairs_file, label=1, max_pairs=max_pairs)

        # Load negative pairs
        remaining = max_pairs - len(self.pairs) if max_pairs else None
        self._load_pairs(negative_pairs_file, label=0, max_pairs=remaining)

        # Shuffle pairs
        random.shuffle(self.pairs)

    def _load_pairs(self, filepath: Path, label: int, max_pairs: Optional[int] = None):
        """Load pairs from JSONL file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if max_pairs and len(self.pairs) >= max_pairs:
                    break

                try:
                    data = json.loads(line)
                    self.pairs.append({
                        'ast1': data['ast1']['ast'],
                        'ast2': data['ast2']['ast'],
                        'label': label,
                        'similarity': data.get('similarity', 0.0)
                    })
                except Exception as e:
                    print(f"Warning: Failed to load pair: {e}")

    def __len__(self) -> int:
        """Number of pairs in dataset."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[Data, Data, int]:
        """
        Get a pair by index.

        Returns:
            (graph1, graph2, label)
        """
        pair = self.pairs[idx]

        # Convert ASTs to graphs
        try:
            graph1 = self.converter.ast_to_graph(pair['ast1'])
            graph2 = self.converter.ast_to_graph(pair['ast2'])
            label = pair['label']

            return graph1, graph2, label

        except Exception as e:
            # Fallback: return a dummy pair if conversion fails
            print(f"Warning: Failed to convert pair {idx}: {e}")
            # Create minimal graphs
            x = torch.zeros((1, 19))
            edge_index = torch.empty((2, 0), dtype=torch.long)
            graph = Data(x=x, edge_index=edge_index)
            return graph, graph, 0


def collate_pairs(batch: List[Tuple[Data, Data, int]]) -> Tuple[List[Data], List[Data], torch.Tensor]:
    """
    Collate function for DataLoader.

    Args:
        batch: List of (graph1, graph2, label) tuples

    Returns:
        (graphs1, graphs2, labels)
    """
    graphs1 = [item[0] for item in batch]
    graphs2 = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.float)

    return graphs1, graphs2, labels


def create_dataloader(
    positive_pairs_file: Path,
    negative_pairs_file: Path,
    batch_size: int = 32,
    max_pairs: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a DataLoader for contrastive learning.

    Args:
        positive_pairs_file: Path to positive pairs JSONL
        negative_pairs_file: Path to negative pairs JSONL
        batch_size: Batch size
        max_pairs: Maximum pairs to load
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes

    Returns:
        DataLoader instance
    """
    dataset = ContrastivePairDataset(
        positive_pairs_file=positive_pairs_file,
        negative_pairs_file=negative_pairs_file,
        max_pairs=max_pairs
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pairs
    )

    return dataloader


def test_dataloader():
    """Test the contrastive dataloader."""
    print("Testing Contrastive DataLoader")
    print("="*70)

    # Check if training data exists
    data_dir = Path("data/training_pairs")
    positive_file = data_dir / "positive_pairs.jsonl"
    negative_file = data_dir / "negative_pairs.jsonl"

    if not positive_file.exists() or not negative_file.exists():
        print("ERROR: Training data not found!")
        print(f"  Expected: {positive_file}")
        print(f"  Expected: {negative_file}")
        print("\nRun: python scripts/prepare_training_data.py")
        return

    # Create dataloader
    print(f"Loading training pairs from {data_dir}...")
    dataloader = create_dataloader(
        positive_pairs_file=positive_file,
        negative_pairs_file=negative_file,
        batch_size=4,
        max_pairs=100,  # Small test
        shuffle=True
    )

    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Num batches: {len(dataloader)}")
    print()

    # Test one batch
    print("Testing first batch...")
    graphs1, graphs2, labels = next(iter(dataloader))

    print(f"  Batch size: {len(graphs1)}")
    print(f"  Graph 1 shapes: {[g.x.shape for g in graphs1]}")
    print(f"  Graph 2 shapes: {[g.x.shape for g in graphs2]}")
    print(f"  Labels: {labels}")
    print(f"  Positive pairs: {labels.sum().item()}")
    print(f"  Negative pairs: {(labels == 0).sum().item()}")

    print("\n✅ DataLoader test successful!")


if __name__ == '__main__':
    test_dataloader()
