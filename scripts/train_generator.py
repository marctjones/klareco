#!/usr/bin/env python3
"""
Train the Graph2Seq Generator model.

This script loads a dataset of (question_graph, context_graph, target_text)
and trains the generator model to synthesize the target text from the graphs.
"""

import argparse
import logging
from pathlib import Path
import sys
import json

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.models.generator import Graph2SeqGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SynthesisDataset(Dataset):
    """
    A PyTorch Dataset for loading the synthesis dataset from JSONL.
    """
    def __init__(self, file_path: Path):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # In a real implementation, we would need to combine the question and context
        # graphs into a single graph data object for the GNN.
        # This is a complex step involving re-indexing nodes and edges.
        
        # Placeholder: just use the context graph for now.
        graph_dict = item['context_graph']
        
        # We also need to convert the target_text into a tensor of token indices.
        # This requires a vocabulary.
        
        # Placeholder:
        target_tensor = torch.tensor([0, 1, 2, 3, 1]) # SOS, word, word, word, EOS

        graph_data = Data(
            x=torch.tensor(graph_dict['x'], dtype=torch.long),
            edge_index=torch.tensor(graph_dict['edge_index'], dtype=torch.long)
        )
        
        return graph_data, target_tensor


def train(model, dataloader, optimizer, criterion, device):
    """
    The main training loop for one epoch.
    """
    model.train()
    total_loss = 0

    for graph_data, target_tensor in dataloader:
        graph_data = graph_data.to(device)
        target_tensor = target_tensor.to(device)

        optimizer.zero_grad()

        # The model forward pass
        output = model(graph_data, target_tensor)

        # Loss calculation
        # The output is [batch, seq_len, vocab_size]
        # The target is [batch, seq_len]
        # We need to reshape for CrossEntropyLoss
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        target = target_tensor.contiguous().view(-1)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train the Graph2Seq Generator model.")
    parser.add_argument("--dataset-path", type=Path, default=Path("data/training_pairs/synthesis_dataset.jsonl"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # TODO: Load a vocabulary to get vocab_size
    vocab_size = 1000 # Placeholder
    
    # Instantiate dataset and dataloader
    # Note: The custom collate_fn would be needed for batching graphs.
    dataset = SynthesisDataset(args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True) # collate_fn would be needed here

    # Instantiate model
    model = Graph2SeqGenerator(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        gnn_out_dim=args.hidden_dim
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0) # Assuming 0 is a padding token

    logging.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        loss = train(model, dataloader, optimizer, criterion, device)
        logging.info(f"Epoch {epoch}/{args.epochs}, Loss: {loss:.4f}")
    logging.info("Training complete.")


if __name__ == "__main__":
    main()
