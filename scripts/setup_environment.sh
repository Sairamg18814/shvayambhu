#!/bin/bash

# Shvayambhu Environment Setup Script
# Sets up development environment for Apple Silicon Macs

set -e  # Exit on error

echo "=€ Setting up Shvayambhu development environment..."

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "L Error: This script requires macOS"
    exit 1
fi

# Check for Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "   Warning: Not running on Apple Silicon. Performance will be limited."
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.11"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo "L Error: Python 3.11+ required. Current version: $PYTHON_VERSION"
    echo "Install with: brew install python@3.11"
    exit 1
fi

echo " Python version: $PYTHON_VERSION"

# Check for Xcode Command Line Tools
if ! xcode-select -p &> /dev/null; then
    echo "=æ Installing Xcode Command Line Tools..."
    xcode-select --install
    echo "Please complete the installation and run this script again."
    exit 1
fi

echo " Xcode Command Line Tools installed"

# Check Metal compiler
if ! command -v metal &> /dev/null; then
    echo "L Error: Metal compiler not found. Please install Xcode."
    exit 1
fi

echo " Metal compiler available"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "= Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "=æ Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install requirements
echo "=æ Installing Python dependencies..."
pip install -r requirements.txt

# Install development requirements
if [ -f "requirements-dev.txt" ]; then
    echo "=æ Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

# Install MLX (Apple's ML framework)
echo "=æ Installing MLX..."
pip install mlx || echo "   Warning: MLX installation failed. Some features may be unavailable."

# Set up pre-commit hooks
echo "=' Setting up pre-commit hooks..."
pre-commit install

# Check PyTorch Metal support
echo "= Checking PyTorch Metal support..."
python3 -c "
import torch
if torch.backends.mps.is_available():
    print(' PyTorch Metal Performance Shaders (MPS) available')
    print(f'   MPS device: {torch.backends.mps.is_built()}')
else:
    print('   Warning: PyTorch MPS not available. GPU acceleration disabled.')
"

# Create necessary directories
echo "=Á Creating project directories..."
mkdir -p data/{seed,validation,cache}
mkdir -p logs
mkdir -p checkpoints
mkdir -p weights

# Set up git configuration
echo "=' Configuring git..."
git config core.hooksPath .git/hooks

# Compile Metal shaders
echo "=( Compiling Metal shaders..."
if [ -d "metal/shaders" ]; then
    for shader in metal/shaders/*.metal; do
        if [ -f "$shader" ]; then
            output="${shader%.metal}.air"
            echo "  Compiling $(basename $shader)..."
            xcrun -sdk macosx metal -c "$shader" -o "$output" 2>/dev/null || echo "     Warning: Failed to compile $(basename $shader)"
        fi
    done
fi

# Run initial tests
echo ">ê Running initial tests..."
pytest tests/unit/test_blt_encoder.py -v || echo "   Warning: Some tests failed. This is expected for initial setup."

# Display environment info
echo ""
echo "=Ê Environment Summary:"
echo "========================"
python3 -c "
import platform
import torch
import sys

print(f'Platform: {platform.platform()}')
print(f'Processor: {platform.processor()}')
print(f'Python: {sys.version.split()[0]}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'MPS available: {torch.backends.mps.is_available()}')
"

echo ""
echo "( Setup complete! To activate the environment, run:"
echo "   source venv/bin/activate"
echo ""
echo "=Ö Next steps:"
echo "   1. Review the documentation in docs/"
echo "   2. Run 'make test' to verify setup"
echo "   3. Start with examples/ to understand the codebase"
echo ""
echo "=€ Happy coding!"