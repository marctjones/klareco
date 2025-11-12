#!/usr/bin/env python3
"""
Pre-flight check for Tree-LSTM training.

Validates all requirements before starting training overnight.

Usage:
    python scripts/preflight_check_training.py --training-data data/training_pairs
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

def check_file_exists(filepath: Path, description: str) -> Tuple[bool, str]:
    """Check if a file exists."""
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        return True, f"✅ {description}: {filepath} ({size_mb:.1f} MB)"
    else:
        return False, f"❌ {description}: {filepath} NOT FOUND"


def check_dependencies() -> Tuple[bool, str]:
    """Check if all required Python packages are installed."""
    missing = []

    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        missing.append("torch")
        torch_version = None

    try:
        import torch_geometric
        pyg_version = torch_geometric.__version__
    except ImportError:
        missing.append("torch-geometric")
        pyg_version = None

    if missing:
        return False, f"❌ Missing dependencies: {', '.join(missing)}"
    else:
        return True, f"✅ Dependencies installed (torch={torch_version}, torch-geometric={pyg_version})"


def validate_training_data(data_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate training data format and content.

    Returns:
        (success, info_dict)
    """
    info = {
        'positive_count': 0,
        'negative_count': 0,
        'total_count': 0,
        'valid_format': True,
        'errors': []
    }

    positive_file = data_dir / 'positive_pairs.jsonl'
    negative_file = data_dir / 'negative_pairs.jsonl'

    # Check files exist
    if not positive_file.exists():
        info['errors'].append(f"Positive pairs file not found: {positive_file}")
        info['valid_format'] = False
        return False, info

    if not negative_file.exists():
        info['errors'].append(f"Negative pairs file not found: {negative_file}")
        info['valid_format'] = False
        return False, info

    # Validate positive pairs
    try:
        with open(positive_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 5:  # Just check first 5 lines
                    break
                try:
                    data = json.loads(line)
                    # Check required fields
                    if 'ast1' not in data or 'ast2' not in data or 'label' not in data:
                        info['errors'].append(f"Missing fields in positive pair line {i+1}")
                        info['valid_format'] = False
                    if data.get('label') != 1:
                        info['errors'].append(f"Positive pair has wrong label: {data.get('label')}")
                        info['valid_format'] = False
                except json.JSONDecodeError as e:
                    info['errors'].append(f"Invalid JSON in positive pairs line {i+1}: {e}")
                    info['valid_format'] = False

        # Count total lines
        with open(positive_file, 'r') as f:
            info['positive_count'] = sum(1 for _ in f)

    except Exception as e:
        info['errors'].append(f"Error reading positive pairs: {e}")
        info['valid_format'] = False
        return False, info

    # Validate negative pairs
    try:
        with open(negative_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 5:  # Just check first 5 lines
                    break
                try:
                    data = json.loads(line)
                    # Check required fields
                    if 'ast1' not in data or 'ast2' not in data or 'label' not in data:
                        info['errors'].append(f"Missing fields in negative pair line {i+1}")
                        info['valid_format'] = False
                    if data.get('label') != 0:
                        info['errors'].append(f"Negative pair has wrong label: {data.get('label')}")
                        info['valid_format'] = False
                except json.JSONDecodeError as e:
                    info['errors'].append(f"Invalid JSON in negative pairs line {i+1}: {e}")
                    info['valid_format'] = False

        # Count total lines
        with open(negative_file, 'r') as f:
            info['negative_count'] = sum(1 for _ in f)

    except Exception as e:
        info['errors'].append(f"Error reading negative pairs: {e}")
        info['valid_format'] = False
        return False, info

    info['total_count'] = info['positive_count'] + info['negative_count']

    # Check class balance
    if info['positive_count'] == 0:
        info['errors'].append("No positive pairs found")
        info['valid_format'] = False

    if info['negative_count'] == 0:
        info['errors'].append("No negative pairs found")
        info['valid_format'] = False

    if info['positive_count'] > 0 and info['negative_count'] > 0:
        ratio = info['negative_count'] / info['positive_count']
        if ratio > 20:
            info['errors'].append(f"WARNING: High class imbalance ({ratio:.1f}:1)")

    return info['valid_format'], info


def test_dataloader(data_dir: Path) -> Tuple[bool, str]:
    """Test loading a batch with the dataloader."""
    try:
        # Add parent directory to path
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from klareco.dataloader import create_dataloader

        positive_file = data_dir / 'positive_pairs.jsonl'
        negative_file = data_dir / 'negative_pairs.jsonl'

        # Try to load a small batch
        dataloader = create_dataloader(
            positive_pairs_file=positive_file,
            negative_pairs_file=negative_file,
            batch_size=4,
            max_pairs=10,  # Only test with 10 pairs
            shuffle=False
        )

        # Try to get one batch
        batch = next(iter(dataloader))
        graphs1, graphs2, labels = batch

        if len(graphs1) > 0 and len(graphs2) > 0:
            return True, f"✅ DataLoader test successful (batch size: {len(labels)})"
        else:
            return False, "❌ DataLoader returned empty batch"

    except Exception as e:
        return False, f"❌ DataLoader test failed: {e}"


def test_model_creation() -> Tuple[bool, str]:
    """Test creating the Tree-LSTM model."""
    try:
        # Add parent directory to path
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from klareco.models.tree_lstm import TreeLSTMEncoder
        import torch

        # Create model
        model = TreeLSTMEncoder(
            vocab_size=1000,
            embed_dim=64,
            hidden_dim=128,
            output_dim=256
        )

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return True, f"✅ Model creation successful ({num_params:,} parameters)"

    except Exception as e:
        return False, f"❌ Model creation failed: {e}"


def main():
    """Run pre-flight checks."""
    parser = argparse.ArgumentParser(description='Pre-flight check for Tree-LSTM training')
    parser.add_argument('--training-data', type=str, default='data/training_pairs',
                        help='Training data directory')
    args = parser.parse_args()

    print("=" * 70)
    print("TREE-LSTM TRAINING PRE-FLIGHT CHECK")
    print("=" * 70)
    print()

    all_checks_passed = True

    # Check 1: Dependencies
    print("1. Checking dependencies...")
    success, message = check_dependencies()
    print(f"   {message}")
    if not success:
        all_checks_passed = False
    print()

    # Check 2: Data files
    print("2. Checking data files...")
    data_dir = Path(args.training_data)

    success, message = check_file_exists(data_dir / 'positive_pairs.jsonl', 'Positive pairs')
    print(f"   {message}")
    if not success:
        all_checks_passed = False

    success, message = check_file_exists(data_dir / 'negative_pairs.jsonl', 'Negative pairs')
    print(f"   {message}")
    if not success:
        all_checks_passed = False

    success, message = check_file_exists(data_dir / 'metadata.json', 'Metadata')
    print(f"   {message}")
    if not success:
        print("   ⚠️  Metadata file missing (optional)")
    print()

    # Check 3: Validate data format
    print("3. Validating data format...")
    success, info = validate_training_data(data_dir)
    print(f"   Positive pairs: {info['positive_count']:,}")
    print(f"   Negative pairs: {info['negative_count']:,}")
    print(f"   Total pairs: {info['total_count']:,}")

    if info['positive_count'] > 0 and info['negative_count'] > 0:
        ratio = info['negative_count'] / info['positive_count']
        print(f"   Class ratio: {ratio:.1f}:1 (negative:positive)")

    if success:
        print(f"   ✅ Data format valid")
    else:
        print(f"   ❌ Data format invalid")
        for error in info['errors']:
            print(f"      - {error}")
        all_checks_passed = False
    print()

    # Check 4: Test dataloader
    print("4. Testing dataloader...")
    success, message = test_dataloader(data_dir)
    print(f"   {message}")
    if not success:
        all_checks_passed = False
    print()

    # Check 5: Test model creation
    print("5. Testing model creation...")
    success, message = test_model_creation()
    print(f"   {message}")
    if not success:
        all_checks_passed = False
    print()

    # Check 6: Check output directory
    print("6. Checking output directory...")
    output_dir = Path('models/tree_lstm')
    if output_dir.exists():
        checkpoints = list(output_dir.glob('checkpoint_epoch_*.pt'))
        if checkpoints:
            print(f"   ⚠️  Found {len(checkpoints)} existing checkpoints")
            print(f"      Use --resume auto to continue training")
        else:
            print(f"   ✅ Output directory exists (no checkpoints)")
    else:
        print(f"   ✅ Output directory will be created")
    print()

    # Final verdict
    print("=" * 70)
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED - READY FOR TRAINING")
        print()
        print("Recommended command:")
        print()
        print("  python scripts/train_tree_lstm.py \\")
        print("      --training-data data/training_pairs \\")
        print("      --output models/tree_lstm \\")
        print("      --epochs 10 \\")
        print("      --batch-size 16 \\")
        print("      --lr 0.001")
        print()
        print("To resume from checkpoint if interrupted:")
        print()
        print("  python scripts/train_tree_lstm.py \\")
        print("      --training-data data/training_pairs \\")
        print("      --output models/tree_lstm \\")
        print("      --resume auto \\")
        print("      --epochs 10 \\")
        print("      --batch-size 16 \\")
        print("      --lr 0.001")
        print()
        return 0
    else:
        print("❌ SOME CHECKS FAILED - FIX ISSUES BEFORE TRAINING")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
