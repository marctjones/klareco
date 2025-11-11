#!/bin/bash
# ============================================================================
# Ensure CPU-Only PyTorch Installation
# ============================================================================
# This script checks your PyTorch installation and ensures you're using the
# CPU-only version (no CUDA). This is ideal for:
# - Intel integrated GPUs (no NVIDIA GPU)
# - Smaller disk footprint (~1.5GB savings)
# - Faster installation
# - No unnecessary CUDA dependencies
#
# Usage:
#   ./scripts/ensure_cpu_only.sh
#
# Or run automatically during setup:
#   conda activate klareco-env
#   ./scripts/ensure_cpu_only.sh
#   pip install -r requirements-cpu.txt
# ============================================================================

set -e  # Exit on error

echo ""
echo "========================================================================"
echo "  Klareco: CPU-Only PyTorch Installation Helper"
echo "========================================================================"
echo ""

# Check if we're in a conda/virtual environment
if [[ -z "$CONDA_DEFAULT_ENV" ]] && [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  WARNING: No conda or virtual environment detected!"
    echo ""
    echo "Please activate your environment first:"
    echo "  conda activate klareco-env"
    echo "  # or"
    echo "  source venv/bin/activate"
    echo ""
    exit 1
fi

if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
    echo "✓ Active conda environment: $CONDA_DEFAULT_ENV"
else
    echo "✓ Active virtual environment: $VIRTUAL_ENV"
fi
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ ERROR: Python not found in PATH"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"
echo ""

# Check if PyTorch is installed
echo "Checking PyTorch installation..."
if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo "✓ PyTorch version: $TORCH_VERSION"

    # Check if CUDA is available in this build
    HAS_CUDA=$(python -c "import torch; print('yes' if torch.cuda.is_available() or '+cu' in torch.__version__ else 'no')")

    if [[ "$HAS_CUDA" == "yes" ]]; then
        echo ""
        echo "⚠️  CUDA-enabled PyTorch detected!"
        echo ""
        echo "Current installation includes CUDA support, which:"
        echo "  - Increases disk usage by ~1.5GB"
        echo "  - Won't work without an NVIDIA GPU"
        echo "  - Is not needed for Intel integrated GPUs"
        echo ""
        echo "Recommendation: Switch to CPU-only version"
        echo ""
        read -p "Uninstall CUDA version and install CPU-only? (y/n) " -n 1 -r
        echo

        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo ""
            echo "Uninstalling current PyTorch..."
            pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

            echo ""
            echo "Installing CPU-only PyTorch..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

            echo ""
            echo "✅ CPU-only PyTorch installed successfully!"

            # Verify
            NEW_VERSION=$(python -c "import torch; print(torch.__version__)")
            echo "   New version: $NEW_VERSION"

            HAS_CUDA_NOW=$(python -c "import torch; print('yes' if torch.cuda.is_available() or '+cu' in torch.__version__ else 'no')")
            if [[ "$HAS_CUDA_NOW" == "no" ]]; then
                echo "   ✓ Verified: CPU-only (no CUDA)"
            else
                echo "   ⚠️  Warning: CUDA still detected (this shouldn't happen)"
            fi
        else
            echo ""
            echo "Keeping current CUDA-enabled installation."
            echo ""
            echo "Note: You can run this script again anytime to switch to CPU-only."
        fi
    else
        echo "✅ CPU-only PyTorch detected (no CUDA)"
        echo ""
        echo "Your installation is already optimized for CPU-only usage!"
        echo "This is the correct setup for Intel integrated GPUs."
    fi
else
    echo "⚠️  PyTorch not installed yet"
    echo ""
    echo "For CPU-only installation, run:"
    echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    echo "  pip install -r requirements-cpu.txt"
    echo ""
fi

echo ""
echo "========================================================================"
echo "  Summary"
echo "========================================================================"
echo ""
echo "Your system:"
echo "  - Intel integrated GPU (no NVIDIA GPU)"
echo "  - Best configuration: CPU-only PyTorch"
echo ""
echo "Benefits of CPU-only:"
echo "  ✓ ~1.5GB smaller installation"
echo "  ✓ No unnecessary CUDA libraries"
echo "  ✓ Faster pip install times"
echo "  ✓ Same performance (no GPU acceleration available anyway)"
echo ""
echo "To install/reinstall everything with CPU-only:"
echo "  conda activate klareco-env"
echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
echo "  pip install -r requirements-cpu.txt"
echo ""
echo "========================================================================"
echo ""
