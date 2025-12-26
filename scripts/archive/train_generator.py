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
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/generator_checkpoints"))
    parser.add_argument("--load-checkpoint", type=Path, default=None, help="Path to a checkpoint to resume training from.")
    args = parser.parse_args()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # TODO: Load a vocabulary to get vocab_size
    vocab_size = 1000 # Placeholder
    
    # Instantiate dataset and dataloader
    dataset = SynthesisDataset(args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Instantiate model and optimizer
    model = Graph2SeqGenerator(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        gnn_out_dim=args.hidden_dim
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    start_epoch = 1
    if args.load_checkpoint and args.load_checkpoint.exists():
        logging.info(f"Loading checkpoint from {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logging.info(f"Resuming training from epoch {start_epoch}")

    logging.info("Starting training...")
    for epoch in range(start_epoch, args.epochs + 1):
        loss = train(model, dataloader, optimizer, criterion, device)
        logging.info(f"Epoch {epoch}/{args.epochs}, Loss: {loss:.4f}")

        # Save checkpoint
        checkpoint_path = args.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        logging.info(f"Saving checkpoint to {checkpoint_path}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        
    logging.info("Training complete.")


if __name__ == "__main__":
    main()
