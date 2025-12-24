#!/usr/bin/env python3
"""
Train Tree-LSTM encoder with compositional embeddings.

Uses the new CompositionalEmbedding for morpheme composition,
achieving 92% parameter reduction compared to traditional word embeddings.

Usage:
    python scripts/train_compositional_tree_lstm.py \
        --corpus data/corpus_with_sources_v2.jsonl \
        --vocab-dir data/vocabularies \
        --output models/tree_lstm_compositional \
        --epochs 10
"""

import argparse
import json
import sys
import time
import gc
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Tuple, Optional
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.parser import parse
from klareco.ast_to_graph import ASTToGraphConverter
from klareco.models.tree_lstm import TreeLSTMEncoder
from klareco.embeddings.compositional import CompositionalEmbedding

# Check for torch_geometric
try:
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("ERROR: torch-geometric is required. Install with: pip install torch-geometric")
    sys.exit(1)


class ContrastiveLoss(nn.Module):
    """Contrastive loss for sentence similarity learning."""

    def __init__(self, margin: float = 1.0, temperature: float = 0.07):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """
        InfoNCE-style contrastive loss.

        Args:
            anchor: Anchor embeddings (batch_size, embed_dim)
            positive: Positive embeddings (batch_size, embed_dim)
            negatives: Negative embeddings (batch_size, num_neg, embed_dim)

        Returns:
            Loss scalar
        """
        batch_size = anchor.size(0)

        # Normalize embeddings
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        # Positive similarity
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature  # (batch,)

        # Negative similarities
        neg_sim = torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1) / self.temperature  # (batch, num_neg)

        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # (batch, 1 + num_neg)
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
        loss = F.cross_entropy(logits, labels)

        return loss


class CorpusDataset(Dataset):
    """Dataset for loading sentences from corpus."""

    def __init__(
        self,
        corpus_path: Path,
        converter: ASTToGraphConverter,
        max_samples: int = 100000,
        seed: int = 42,
    ):
        self.corpus_path = corpus_path
        self.converter = converter
        self.graphs = []
        self.sentences = []

        print(f"Loading corpus from {corpus_path}...")
        random.seed(seed)

        # Count lines
        with open(corpus_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)

        # Sample lines if too many
        sample_rate = min(1.0, max_samples / total_lines)

        loaded = 0
        failed = 0
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc="Loading"):
                if random.random() > sample_rate:
                    continue

                try:
                    entry = json.loads(line)
                    ast = entry.get('ast')
                    sentence = entry.get('sentence', '')

                    if not ast or ast.get('tipo') != 'frazo':
                        continue

                    graph = converter.ast_to_graph(ast)
                    if graph is not None:
                        self.graphs.append(graph)
                        self.sentences.append(sentence)
                        loaded += 1

                        if loaded >= max_samples:
                            break

                except Exception as e:
                    failed += 1
                    continue

        print(f"Loaded {len(self.graphs):,} graphs ({failed:,} failed)")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.sentences[idx]


def collate_contrastive(batch, num_negatives: int = 7):
    """
    Collate function for contrastive learning.

    Creates triplets: anchor, positive (same sentence augmented), negatives (other sentences).
    """
    graphs = [item[0] for item in batch]
    batch_size = len(graphs)

    # For now, use same graph as positive (self-supervised)
    # In production, would use paraphrases or augmentations
    anchors = graphs
    positives = graphs  # Same graph (add noise in training)

    # Negatives: random other graphs from batch
    negatives_list = []
    for i in range(batch_size):
        neg_indices = [j for j in range(batch_size) if j != i]
        random.shuffle(neg_indices)
        neg_indices = neg_indices[:num_negatives]
        # Pad if not enough negatives
        while len(neg_indices) < num_negatives:
            neg_indices.append(random.choice([j for j in range(batch_size) if j != i]))
        negatives_list.append([graphs[j] for j in neg_indices])

    return anchors, positives, negatives_list


def train_epoch(
    model: TreeLSTMEncoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: ContrastiveLoss,
    device: torch.device,
    epoch: int,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for anchors, positives, negatives_list in pbar:
        optimizer.zero_grad()

        # Encode anchors
        anchor_embs = []
        for graph in anchors:
            graph = graph.to(device)
            emb = model(graph)
            anchor_embs.append(emb)
        anchor_embs = torch.stack(anchor_embs)

        # Encode positives (with dropout for augmentation)
        positive_embs = []
        for graph in positives:
            graph = graph.to(device)
            emb = model(graph)
            positive_embs.append(emb)
        positive_embs = torch.stack(positive_embs)

        # Encode negatives
        batch_size = len(anchors)
        num_neg = len(negatives_list[0])
        negative_embs = []
        for i in range(batch_size):
            neg_embs_i = []
            for graph in negatives_list[i]:
                graph = graph.to(device)
                emb = model(graph)
                neg_embs_i.append(emb)
            negative_embs.append(torch.stack(neg_embs_i))
        negative_embs = torch.stack(negative_embs)  # (batch, num_neg, embed_dim)

        # Compute loss
        loss = criterion(anchor_embs, positive_embs, negative_embs)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / max(num_batches, 1)


def evaluate(
    model: TreeLSTMEncoder,
    dataloader: DataLoader,
    criterion: ContrastiveLoss,
    device: torch.device,
) -> float:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for anchors, positives, negatives_list in dataloader:
            # Encode anchors
            anchor_embs = []
            for graph in anchors:
                graph = graph.to(device)
                emb = model(graph)
                anchor_embs.append(emb)
            anchor_embs = torch.stack(anchor_embs)

            # Encode positives
            positive_embs = []
            for graph in positives:
                graph = graph.to(device)
                emb = model(graph)
                positive_embs.append(emb)
            positive_embs = torch.stack(positive_embs)

            # Encode negatives
            batch_size = len(anchors)
            negative_embs = []
            for i in range(batch_size):
                neg_embs_i = []
                for graph in negatives_list[i]:
                    graph = graph.to(device)
                    emb = model(graph)
                    neg_embs_i.append(emb)
                negative_embs.append(torch.stack(neg_embs_i))
            negative_embs = torch.stack(negative_embs)

            loss = criterion(anchor_embs, positive_embs, negative_embs)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Train Tree-LSTM with compositional embeddings"
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/corpus_with_sources_v2.jsonl"),
        help="Corpus JSONL file"
    )
    parser.add_argument(
        "--vocab-dir",
        type=Path,
        default=Path("data/vocabularies"),
        help="Vocabulary directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/tree_lstm_compositional"),
        help="Output directory for model"
    )
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=256, help="LSTM hidden dimension")
    parser.add_argument("--output-dim", type=int, default=512, help="Output embedding dimension")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=50000, help="Max training samples")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load compositional embedding
    logger.info(f"Loading vocabularies from {args.vocab_dir}")
    comp_embedding = CompositionalEmbedding.from_vocabulary_files(
        args.vocab_dir,
        embed_dim=args.embed_dim,
        composition_method='sum',
    )
    logger.info(f"Compositional embedding stats: {comp_embedding.get_vocab_stats()}")

    # Create converter with compositional embedding
    converter = ASTToGraphConverter(
        embed_dim=args.embed_dim,
        compositional_embedding=comp_embedding,
    )
    input_dim = converter.get_feature_dim()
    logger.info(f"Feature dimension: {input_dim}")

    # Load dataset
    dataset = CorpusDataset(
        args.corpus,
        converter,
        max_samples=args.max_samples,
    )

    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_contrastive,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_contrastive,
        num_workers=0,
    )

    # Create model
    model = TreeLSTMEncoder(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        input_dim=input_dim,
        use_compositional=True,
    )
    model = model.to(device)

    # Also move compositional embedding to device
    comp_embedding = comp_embedding.to(device)
    converter.compositional_embedding = comp_embedding

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = ContrastiveLoss(temperature=0.07)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_loss = float('inf')
    logger.info("Starting training...")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': {
                    'embed_dim': args.embed_dim,
                    'hidden_dim': args.hidden_dim,
                    'output_dim': args.output_dim,
                    'input_dim': input_dim,
                }
            }, args.output / 'best_model.pt')
            logger.info(f"  Saved best model (val_loss={val_loss:.4f})")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, args.output / 'checkpoint.pt')

        gc.collect()

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'config': {
            'embed_dim': args.embed_dim,
            'hidden_dim': args.hidden_dim,
            'output_dim': args.output_dim,
            'input_dim': input_dim,
        }
    }, args.output / 'final_model.pt')

    logger.info(f"Training complete. Best val_loss: {best_val_loss:.4f}")
    logger.info(f"Models saved to {args.output}")


if __name__ == '__main__':
    main()
