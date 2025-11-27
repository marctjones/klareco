#!/usr/bin/env python3
"""
Train the Graph2Seq model.

Usage:
    python scripts/train_graph2seq.py
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import logging
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch_geometric.data import Batch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.models.graph2seq import Graph2SeqGenerator
from klareco.logging_config import setup_logging

# --- Vocabulary and Dataset ---

class Vocab:
    """Vocabulary class to handle word-to-index mapping."""
    def __init__(self, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        self.word2index = vocab_data['word2index']
        self.index2word = vocab_data['index2word']
        self.n_words = vocab_data['n_words']
        self.pad_token_id = self.word2index['<PAD>']
        self.sos_token_id = self.word2index['<SOS>']
        self.eos_token_id = self.word2index['<EOS>']

    def sequence_to_indices(self, sequence, add_eos=False, add_sos=False):
        indices = []
        if add_sos:
            indices.append(self.sos_token_id)
        
        # Simple space-based tokenization
        tokens = sequence.strip().split(' ')
        indices.extend([self.word2index.get(word, self.word2index["<UNK>"]) for word in tokens])
        
        if add_eos:
            indices.append(self.eos_token_id)
        return indices

class SynthesisDataset(Dataset):
    """Custom dataset for the synthesis_dataset.jsonl file."""
    def __init__(self, file_path, vocab):
        self.file_path = file_path
        self.vocab = vocab
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        question_graph = item['question_graph']
        context_graph = item['context_graph']
        
        # Convert target text to tensor of indices
        target_text = item['target_text']
        target_indices = self.vocab.sequence_to_indices(target_text, add_eos=True)
        
        return {
            'question_graph': question_graph,
            'context_graph': context_graph,
            'target_tensor': torch.tensor(target_indices, dtype=torch.long)
        }



def merge_graphs(q_graph, c_graph):
    """
    Merges a question graph and a context graph into a single graph.
    """
    num_q_nodes = q_graph.num_nodes

    # Combine node features
    x = torch.cat([q_graph.x, c_graph.x], dim=0)

    # Offset context graph edge indices
    c_edge_index = c_graph.edge_index + num_q_nodes
    
    # Combine edge indices
    edge_index = torch.cat([q_graph.edge_index, c_edge_index], dim=1)

    # Combine edge attributes
    edge_attr = torch.cat([q_graph.edge_attr, c_graph.edge_attr], dim=0)

    # Optional: Add an edge connecting the two graphs (e.g., from question root to context root)
    # Assuming root is node 0 in both graphs
    # connection_edge = torch.tensor([[0], [num_q_nodes]], dtype=torch.long)
    # connection_attr = torch.tensor([99], dtype=torch.long) # New edge type
    # edge_index = torch.cat([edge_index, connection_edge], dim=1)
    # edge_attr = torch.cat([edge_attr, connection_attr], dim=0)
    
    from torch_geometric.data import Data
    merged_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return merged_graph

def collate_fn(batch):

    """

    Custom collate function to batch graph data.

    """

    from torch_geometric.data import Data



    merged_graphs = []

    targets = []

    vocab = batch[0]['vocab']



    for item in batch:

        q_graph = Data(

            x=torch.tensor(item['question_graph']['x'], dtype=torch.float),

            edge_index=torch.tensor(item['question_graph']['edge_index'], dtype=torch.long),

            edge_attr=torch.tensor(item['question_graph']['edge_attr'], dtype=torch.long)

        )

        c_graph = Data(

            x=torch.tensor(item['context_graph']['x'], dtype=torch.float),

            edge_index=torch.tensor(item['context_graph']['edge_index'], dtype=torch.long),

            edge_attr=torch.tensor(item['context_graph']['edge_attr'], dtype=torch.long)

        )

        

        merged_graphs.append(merge_graphs(q_graph, c_graph))

        targets.append(item['target_tensor'])



    graph_batch = Batch.from_data_list(merged_graphs)

    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=vocab.pad_token_id)



    return graph_batch, targets_padded





# --- Training and Validation ---



def train_epoch(model, dataloader, optimizer, criterion, device, logger):

    model.train()

    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Training")



    for graph_batch, targets in pbar:

        graph_batch, targets = graph_batch.to(device), targets.to(device)



        optimizer.zero_grad()



        output = model(graph_batch, targets)

        # Loss calculation
        output_flat = output.reshape(-1, output.shape[-1])
        targets_flat = targets.reshape(-1)
        loss = criterion(output_flat, targets_flat)

        loss.backward()

        optimizer.step()



        total_loss += loss.item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})



    avg_loss = total_loss / len(dataloader)

    return avg_loss

def main():
    parser = argparse.ArgumentParser(description='Train Graph2Seq model')
    parser.add_argument('--dataset-path', type=str, default='data/training_pairs/synthesis_dataset.jsonl')
    parser.add_argument('--vocab-path', type=str, default='data/vocab.json')
    parser.add_argument('--output-dir', type=str, default='models/graph2seq')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--embed-dim', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--gnn-out-dim', type=int, default=512)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device(args.device)

    logger.info("Loading vocabulary...")
    vocab = Vocab(args.vocab_path)

    logger.info("Creating dataset and dataloader...")
    dataset = SynthesisDataset(args.dataset_path, vocab)
    
    # Update collate_fn to pass vocab
    def collate_with_vocab(batch):
        # Add vocab to each item to be accessible in collate_fn
        for item in batch:
            item['vocab'] = vocab
        return collate_fn(batch)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_with_vocab)

    logger.info("Initializing model...")
    model = Graph2SeqGenerator(
        vocab_size=vocab.n_words,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        gnn_out_dim=args.gnn_out_dim
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Ignore padding index in loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_token_id) 

    logger.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        logger.info(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_epoch(model, dataloader, optimizer, criterion, device, logger)
        logger.info(f"  Train Loss: {train_loss:.4f}")
        # Validation step would go here

    logger.info("Training complete.")


if __name__ == '__main__':
    main()
