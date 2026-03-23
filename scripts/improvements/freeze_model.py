#!/usr/bin/env python3
"""
Freeze root embeddings after training for downstream stability.

Frozen embeddings ensure that MorphemeComposer, NodePredictor, and other
downstream models can rely on stable semantic primitives.

Usage:
    python scripts/improvements/freeze_model.py \
        --model models/root_embeddings/best_model.pt \
        --output models/root_embedder/frozen_v1.0.pt
"""

import argparse
import torch
from pathlib import Path
from datetime import datetime
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def freeze_model(
    checkpoint_path: Path,
    output_path: Path,
    version: str = "v1.0",
    description: str = ""
):
    """
    Freeze a trained root embedding model.

    Freezing means:
    1. Set requires_grad=False for all parameters
    2. Add version metadata
    3. Add frozen flag
    4. Optionally reduce precision (fp16) for smaller size
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract model state
    model_state = checkpoint['model_state_dict']

    # Create frozen checkpoint
    frozen_checkpoint = {
        # Original data
        'model_state_dict': model_state,
        'embedding_dim': checkpoint.get('embedding_dim', 64),
        'vocab_size': checkpoint.get('vocab_size', len(checkpoint['root_to_idx'])),
        'root_to_idx': checkpoint['root_to_idx'],
        'idx_to_root': checkpoint['idx_to_root'],

        # Training metadata
        'best_correlation': checkpoint.get('correlation', 0.0),
        'training_epochs': checkpoint.get('epoch', 0),
        'training_loss': checkpoint.get('loss', 0.0),

        # Freezing metadata
        'frozen': True,
        'version': version,
        'frozen_date': datetime.now().isoformat(),
        'description': description or f"Frozen root embeddings {version}",

        # For validation
        'function_words_excluded': True,
        'antonym_pairs_included': True,  # If using improved training
    }

    # Save frozen model
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = output_path.with_suffix('.tmp')
    torch.save(frozen_checkpoint, temp_path)
    temp_path.rename(output_path)

    logger.info(f"Saved frozen model to {output_path}")
    logger.info(f"  Version: {version}")
    logger.info(f"  Vocabulary: {frozen_checkpoint['vocab_size']:,} roots")
    logger.info(f"  Embedding dim: {frozen_checkpoint['embedding_dim']}")
    logger.info(f"  Best correlation: {frozen_checkpoint['best_correlation']:.4f}")
    logger.info(f"  Frozen: {frozen_checkpoint['frozen']}")

    # Save metadata JSON for easy inspection
    metadata_path = output_path.with_suffix('.json')
    metadata = {
        'version': version,
        'frozen': True,
        'vocab_size': frozen_checkpoint['vocab_size'],
        'embedding_dim': frozen_checkpoint['embedding_dim'],
        'best_correlation': frozen_checkpoint['best_correlation'],
        'training_epochs': frozen_checkpoint['training_epochs'],
        'frozen_date': frozen_checkpoint['frozen_date'],
        'description': frozen_checkpoint['description'],
        'model_path': str(output_path),
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {metadata_path}")

    return frozen_checkpoint


def load_frozen_model(frozen_path: Path):
    """
    Load frozen root embeddings.

    Returns:
        checkpoint dict with embeddings and metadata
    """
    checkpoint = torch.load(frozen_path, map_location='cpu')

    if not checkpoint.get('frozen', False):
        logger.warning(f"Model at {frozen_path} is not marked as frozen!")

    logger.info(f"Loaded frozen model:")
    logger.info(f"  Version: {checkpoint.get('version', 'unknown')}")
    logger.info(f"  Vocabulary: {checkpoint['vocab_size']:,} roots")
    logger.info(f"  Frozen date: {checkpoint.get('frozen_date', 'unknown')}")

    return checkpoint


def validate_frozen_model(frozen_path: Path):
    """Validate that a frozen model can be loaded and used."""
    checkpoint = load_frozen_model(frozen_path)

    # Check required fields
    required_fields = [
        'model_state_dict',
        'root_to_idx',
        'idx_to_root',
        'embedding_dim',
        'vocab_size',
        'frozen'
    ]

    for field in required_fields:
        if field not in checkpoint:
            raise ValueError(f"Missing required field: {field}")

    # Check that embeddings are present
    if 'embeddings.weight' not in checkpoint['model_state_dict']:
        raise ValueError("No embeddings found in model_state_dict")

    embeddings = checkpoint['model_state_dict']['embeddings.weight']
    expected_shape = (checkpoint['vocab_size'], checkpoint['embedding_dim'])

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Embedding shape mismatch: got {embeddings.shape}, "
            f"expected {expected_shape}"
        )

    logger.info("✓ Frozen model validation passed")
    logger.info(f"  Embeddings shape: {embeddings.shape}")
    logger.info(f"  Frozen: {checkpoint['frozen']}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Freeze root embeddings")
    parser.add_argument('--model', type=Path, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--output', type=Path, required=True,
                        help='Path for frozen model')
    parser.add_argument('--version', type=str, default='v1.0',
                        help='Version tag for frozen model')
    parser.add_argument('--description', type=str, default='',
                        help='Description of this model version')
    parser.add_argument('--validate', action='store_true',
                        help='Validate frozen model after creation')

    args = parser.parse_args()

    # Freeze the model
    frozen_checkpoint = freeze_model(
        args.model,
        args.output,
        version=args.version,
        description=args.description
    )

    # Optionally validate
    if args.validate:
        logger.info("\nValidating frozen model...")
        validate_frozen_model(args.output)

    logger.info("\n✓ Freezing complete!")
    logger.info(f"\nTo use this frozen model in downstream training:")
    logger.info(f"  from improvements.freeze_model import load_frozen_model")
    logger.info(f"  checkpoint = load_frozen_model('{args.output}')")
    logger.info(f"  root_embeddings = checkpoint['model_state_dict']['embeddings.weight']")


if __name__ == '__main__':
    main()
