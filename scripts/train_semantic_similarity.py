#!/usr/bin/env python3
"""
Train Compositional Embeddings on Semantic Similarity.

Uses the similarity pairs generated from parallel corpus to train
embeddings that capture semantic meaning, not just structural similarity.

Training objective:
- Pairs with high similarity (paraphrases) should have similar embeddings
- Pairs with low similarity should have different embeddings
- Uses cosine similarity loss to match predicted similarity to target
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from klareco.embeddings.compositional import CompositionalEmbedding
from klareco import parser as eo_parser_module
from klareco.models.tree_lstm import TreeLSTMEncoder
from klareco.ast_to_graph import ASTToGraphConverter

def setup_logging(output_dir: Path):
    """Setup logging to both console and file."""
    log_file = output_dir / 'training.log'

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return logging.getLogger(__name__)

# Placeholder - will be properly initialized in main()
logger = logging.getLogger(__name__)


class SimilarityDataset(Dataset):
    """Dataset of Esperanto sentence pairs with similarity scores."""

    def __init__(
        self,
        data_path: Path,
        converter: ASTToGraphConverter,
        max_samples: Optional[int] = None,
    ):
        self.converter = converter
        self.samples = []  # List of (graph_a, graph_b, similarity)

        logger.info(f"Loading similarity pairs from {data_path}")

        failed = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Loading")):
                if max_samples and i >= max_samples:
                    break

                record = json.loads(line)
                sent_a = record['sentence_a']
                sent_b = record['sentence_b']
                similarity = record['similarity']

                # Parse and convert both sentences
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

                    self.samples.append((graph_a, graph_b, similarity))

                except Exception as e:
                    failed += 1
                    continue

        logger.info(f"Loaded {len(self.samples):,} pairs ({failed:,} failed to parse)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Collate function for DataLoader."""
    graphs_a = [item[0] for item in batch]
    graphs_b = [item[1] for item in batch]
    similarities = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    return graphs_a, graphs_b, similarities


class CosineSimilarityLoss(nn.Module):
    """Loss that pushes embeddings to match target similarity."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        target_sim: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            emb_a: Embeddings for sentence A (batch, dim)
            emb_b: Embeddings for sentence B (batch, dim)
            target_sim: Target similarity scores (batch,) in [0, 1]
        """
        # Normalize embeddings
        emb_a = F.normalize(emb_a, dim=-1)
        emb_b = F.normalize(emb_b, dim=-1)

        # Compute cosine similarity
        pred_sim = torch.sum(emb_a * emb_b, dim=-1)  # (batch,)

        # MSE loss between predicted and target similarity
        loss = F.mse_loss(pred_sim, target_sim)

        return loss


def train_epoch(
    model: TreeLSTMEncoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CosineSimilarityLoss,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """Train for one epoch. Returns (loss, correlation)."""
    model.train()
    total_loss = 0.0
    all_pred_sims = []
    all_target_sims = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for graphs_a, graphs_b, target_sims in pbar:
        optimizer.zero_grad()

        # Encode both sentences
        emb_a_list = []
        emb_b_list = []

        for graph_a, graph_b in zip(graphs_a, graphs_b):
            graph_a = graph_a.to(device)
            graph_b = graph_b.to(device)

            emb_a = model(graph_a)
            emb_b = model(graph_b)

            emb_a_list.append(emb_a)
            emb_b_list.append(emb_b)

        emb_a = torch.stack(emb_a_list)
        emb_b = torch.stack(emb_b_list)
        target_sims = target_sims.to(device)

        # Compute loss
        loss = criterion(emb_a, emb_b, target_sims)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

        # Track predictions for correlation
        with torch.no_grad():
            pred_sim = torch.sum(F.normalize(emb_a, dim=-1) * F.normalize(emb_b, dim=-1), dim=-1)
            all_pred_sims.extend(pred_sim.cpu().tolist())
            all_target_sims.extend(target_sims.cpu().tolist())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)

    # Compute Pearson correlation
    import numpy as np
    correlation = np.corrcoef(all_pred_sims, all_target_sims)[0, 1]

    return avg_loss, correlation


def evaluate(
    model: TreeLSTMEncoder,
    dataloader: DataLoader,
    criterion: CosineSimilarityLoss,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model. Returns (loss, correlation)."""
    model.eval()
    total_loss = 0.0
    all_pred_sims = []
    all_target_sims = []

    with torch.no_grad():
        for graphs_a, graphs_b, target_sims in tqdm(dataloader, desc="Evaluating"):
            emb_a_list = []
            emb_b_list = []

            for graph_a, graph_b in zip(graphs_a, graphs_b):
                graph_a = graph_a.to(device)
                graph_b = graph_b.to(device)

                emb_a = model(graph_a)
                emb_b = model(graph_b)

                emb_a_list.append(emb_a)
                emb_b_list.append(emb_b)

            emb_a = torch.stack(emb_a_list)
            emb_b = torch.stack(emb_b_list)
            target_sims = target_sims.to(device)

            loss = criterion(emb_a, emb_b, target_sims)
            total_loss += loss.item()

            pred_sim = torch.sum(F.normalize(emb_a, dim=-1) * F.normalize(emb_b, dim=-1), dim=-1)
            all_pred_sims.extend(pred_sim.cpu().tolist())
            all_target_sims.extend(target_sims.cpu().tolist())

    avg_loss = total_loss / len(dataloader)

    import numpy as np
    correlation = np.corrcoef(all_pred_sims, all_target_sims)[0, 1]

    return avg_loss, correlation


def main():
    parser = argparse.ArgumentParser(
        description="Train compositional embeddings on semantic similarity"
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/similarity_pairs_train.jsonl"),
        help="Training data file",
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        default=Path("data/similarity_pairs_val.jsonl"),
        help="Validation data file",
    )
    parser.add_argument(
        "--vocab-dir",
        type=Path,
        default=Path("data/vocabularies"),
        help="Vocabulary directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/semantic_similarity"),
        help="Output directory for model",
    )
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--output-dim", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=10000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (epochs without improvement)")

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory first (needed for logging)
    args.output.mkdir(parents=True, exist_ok=True)

    # Setup logging to file and console
    global logger
    logger = setup_logging(args.output)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load vocabularies and create embedding
    logger.info(f"Loading vocabularies from {args.vocab_dir}")
    compositional_embedding = CompositionalEmbedding.from_vocabulary_files(
        args.vocab_dir,
        embed_dim=args.embed_dim,
    )
    logger.info(f"Vocabulary: {len(compositional_embedding.root_vocab):,} roots")

    # Create converter
    converter = ASTToGraphConverter(compositional_embedding)

    # Load datasets
    logger.info("Loading training data...")
    train_dataset = SimilarityDataset(
        args.train_file,
        converter,
        max_samples=args.max_train_samples,
    )

    logger.info("Loading validation data...")
    val_dataset = SimilarityDataset(
        args.val_file,
        converter,
        max_samples=args.max_val_samples,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Create model
    # The compositional embedding provides features to the graph via the converter
    # TreeLSTMEncoder takes the feature dimension as input_dim
    input_dim = converter.get_feature_dim()
    model = TreeLSTMEncoder(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        input_dim=input_dim,
        use_compositional=True,
    ).to(device)

    # Move compositional embedding to device
    compositional_embedding = compositional_embedding.to(device)
    converter.compositional_embedding = compositional_embedding

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # Loss and optimizer
    criterion = CosineSimilarityLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume from checkpoint if requested
    start_epoch = 1
    best_val_corr = -1.0
    checkpoint_path = args.output / 'best_model.pt'

    if args.resume and checkpoint_path.exists():
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_corr = checkpoint['val_correlation']
        # Advance scheduler to correct position
        for _ in range(checkpoint['epoch']):
            scheduler.step()
        logger.info(f"Resumed from epoch {checkpoint['epoch']} (val_corr={best_val_corr:.4f})")
    elif args.resume:
        logger.info("No checkpoint found, starting from scratch")

    # Early stopping tracking
    epochs_without_improvement = 0

    # Training loop
    logger.info("Starting training...")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_corr = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_loss, val_corr = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        logger.info(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_corr={train_corr:.4f}, "
            f"val_loss={val_loss:.4f}, val_corr={val_corr:.4f}"
        )

        # Save best model and track early stopping
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            epochs_without_improvement = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_correlation': val_corr,
                'config': {
                    'embed_dim': args.embed_dim,
                    'hidden_dim': args.hidden_dim,
                    'output_dim': args.output_dim,
                    'num_roots': len(compositional_embedding.root_vocab),
                },
            }
            torch.save(checkpoint, args.output / 'best_model.pt')
            logger.info(f"  Saved best model (val_corr={val_corr:.4f})")
        else:
            epochs_without_improvement += 1
            logger.info(f"  No improvement for {epochs_without_improvement} epoch(s)")

            if epochs_without_improvement >= args.patience:
                logger.info(f"Early stopping: no improvement for {args.patience} epochs")
                break

    logger.info(f"\nTraining complete. Best val_correlation: {best_val_corr:.4f}")
    logger.info(f"Model saved to {args.output / 'best_model.pt'}")


if __name__ == "__main__":
    main()
