# Python Environment Setup for Klareco

**Philosophy**: Use standard Python tools only (pip + venv). No conda/anaconda.

## Requirements

- **Python 3.8+** (recommend 3.11 or 3.13)
- **pip** (usually included with Python)
- **venv** (usually included, may need separate package on some systems)

## Setup Methods

### Method 1: Automatic (Recommended)

The `run_corpus_builder.sh` script handles everything automatically:

```bash
./scripts/run_corpus_builder.sh
```

It will:
1. Find system Python 3
2. Create virtual environment in `.venv/`
3. Install all requirements
4. Run the corpus builder

### Method 2: Manual Setup

If you prefer manual control:

```bash
# 1. Create virtual environment
python3 -m venv .venv

# 2. Activate it
source .venv/bin/activate

# 3. Upgrade pip
python3 -m pip install --upgrade pip

# 4. Install requirements
pip install -r requirements.txt

# 5. Verify installation
python3 -c "from klareco.parser import parse; print('OK')"
```

## System-Specific Instructions

### Ubuntu/Debian

```bash
# Install Python and venv
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv

# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Fedora/RHEL/CentOS

```bash
# Install Python and venv
sudo dnf install python3 python3-pip python3-venv

# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### macOS

```bash
# Python 3 should be pre-installed (or use Homebrew)
# If not: brew install python3

# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Arch Linux

```bash
# Install Python
sudo pacman -S python python-pip

# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Verifying Your Setup

### Check Python Version

```bash
python3 --version
# Should be 3.8 or higher (3.11+ recommended)
```

### Check Virtual Environment

```bash
source .venv/bin/activate
which python3
# Should show: /home/marc/klareco/.venv/bin/python3
```

### Check Installed Packages

```bash
pip list
# Should include: torch, transformers, pytest, etc.
```

### Test Klareco Import

```bash
python3 -c "from klareco.parser import parse; print('Parser OK')"
python3 -c "from klareco.deparser import deparse; print('Deparser OK')"
```

## Troubleshooting

### "python3: command not found"

**Problem**: Python 3 not installed

**Solution**: Install Python 3 using your system package manager (see above)

### "No module named 'venv'"

**Problem**: venv module not installed (common on Debian/Ubuntu)

**Solution**:
```bash
sudo apt-get install python3-venv
```

### "pip: command not found"

**Problem**: pip not installed

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install python3-pip

# Fedora
sudo dnf install python3-pip
```

### "Failed to import klareco modules"

**Problem**: Requirements not installed or wrong Python

**Solutions**:

1. **Make sure virtual environment is activated**:
   ```bash
   source .venv/bin/activate
   which python3  # Should show .venv path
   ```

2. **Reinstall requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Check PYTHONPATH**:
   ```bash
   python3 -c "import sys; print('\n'.join(sys.path))"
   # Should include project root
   ```

### "torch" installation fails

**Problem**: PyTorch installation can be slow or fail on some systems

**Solutions**:

1. **CPU-only version** (smaller, faster to install):
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Skip optional dependencies** (minimal installation):
   ```bash
   pip install lingua-language-detector transformers pytest tqdm
   ```

### Still using conda Python

**Problem**: Virtual environment using conda's Python instead of system Python

**Check**:
```bash
source .venv/bin/activate
which python3
python3 -c "import sys; print(sys.prefix)"
```

**Solutions**:

1. **Remove venv and recreate with system Python**:
   ```bash
   rm -rf .venv
   /usr/bin/python3 -m venv .venv  # Use explicit system Python
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Deactivate conda** (if installed):
   ```bash
   conda deactivate
   # Then recreate venv
   ```

## Dependencies Overview

### Core (Required)

- **lingua-language-detector**: Language detection (Esperanto support)
- **transformers**: Translation models (MarianMT)
- **torch**: PyTorch (backend for transformers)
- **pytest**: Testing framework
- **tqdm**: Progress bars

### Optional (Phase 3+)

- **faiss-cpu**: Vector search (for RAG)
- **torch-geometric**: Graph neural networks
- **networkx**: Graph visualization
- **matplotlib**: Plotting

## CPU-Only Installation

If you don't have a GPU or want a smaller installation:

```bash
# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install CPU-only PyTorch first
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Then install other requirements
pip install lingua-language-detector transformers requests pytest pytest-cov hypothesis coverage tqdm
```

This saves ~2GB of disk space and installs faster.

## Development Workflow

### Activate Environment

```bash
# Every time you start a new terminal
source .venv/bin/activate
```

### Run Scripts

```bash
# With activated environment
python scripts/build_corpus_v2.py

# Or use the wrapper script (auto-activates)
./scripts/run_corpus_builder.sh
```

### Run Tests

```bash
source .venv/bin/activate
pytest
```

### Deactivate Environment

```bash
deactivate
```

## VS Code Integration

Add to `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.terminal.activateEnvironment": true
}
```

VS Code will automatically use the venv.

## CI/CD Integration

For automated builds/tests:

```yaml
# .github/workflows/test.yml
- name: Set up Python
  uses: actions/setup-python@v4
  with:
    python-version: '3.11'

- name: Create venv and install dependencies
  run: |
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

- name: Run tests
  run: |
    source .venv/bin/activate
    pytest
```

## Summary

**Standard Python tools used**:
- ✅ `python3 -m venv` - Virtual environment
- ✅ `pip` - Package installation
- ✅ `requirements.txt` - Dependency management

**NOT used**:
- ❌ conda
- ❌ anaconda
- ❌ miniconda
- ❌ poetry (could add later if desired)
- ❌ pipenv (could add later if desired)

**Why venv?**
- Built into Python 3.3+
- No extra installation needed
- Lightweight
- Standard and well-supported
- Works everywhere Python works

**Next steps**:
1. Verify your setup: `python3 --version`
2. Create venv: `python3 -m venv .venv`
3. Activate: `source .venv/bin/activate`
4. Install: `pip install -r requirements.txt`
5. Run: `./scripts/run_corpus_builder.sh`
