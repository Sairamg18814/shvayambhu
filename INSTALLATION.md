# 🚀 Shvayambhu Installation Guide

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Quick Install (5 minutes)](#quick-install)
3. [Detailed Installation](#detailed-installation)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)
6. [Next Steps](#next-steps)

## System Requirements

### Minimum Requirements
- **CPU**: Apple Silicon (M1/M2/M3/M4)
- **RAM**: 16GB
- **Storage**: 50GB free space
- **OS**: macOS 12.0+
- **Python**: 3.11+

### Recommended Requirements
- **CPU**: Apple M4 Pro
- **RAM**: 48GB
- **Storage**: 100GB free space
- **OS**: macOS 14.0+
- **GPU**: 38 GPU cores (M4 Pro)

## Quick Install

```bash
# 1. Clone and enter directory
git clone https://github.com/Sairamg18814/shvayambhu.git
cd shvayambhu

# 2. Run installation script
chmod +x install.sh
./install.sh

# 3. Activate environment
source venv/bin/activate

# 4. Test consciousness
python core/consciousness/true_self_awareness.py
```

## Detailed Installation

### Step 1: Install Prerequisites

#### Homebrew (if not installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Python 3.11
```bash
brew install python@3.11
```

#### Git
```bash
brew install git
```

#### Xcode Command Line Tools
```bash
xcode-select --install
```

### Step 2: Clone Repository

```bash
# Clone with all submodules
git clone --recursive https://github.com/Sairamg18814/shvayambhu.git
cd shvayambhu
```

### Step 3: Create Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### Step 4: Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### Step 5: Install Ollama (Recommended)

```bash
# Install Ollama
brew install ollama

# Start Ollama service
ollama serve &

# Download required models
ollama pull llama3.1:8b
ollama pull qwen3:32b
ollama pull gemma3:27b
```

### Step 6: Set Up Database

```bash
# Create data directories
mkdir -p data/db
mkdir -p data/logs
mkdir -p data/checkpoints
mkdir -p data/weights

# Initialize database
python scripts/init_database.py
```

### Step 7: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings
nano .env
```

Required environment variables:
```bash
# .env file
CONSCIOUSNESS_MODE=true
OLLAMA_HOST=http://localhost:11434
MLX_DEVICE=gpu
MEMORY_LIMIT_GB=48
LOG_LEVEL=INFO
```

### Step 8: Install MLX Framework

```bash
# Install MLX for Apple Silicon
pip install mlx mlx-lm

# Verify MLX installation
python -c "import mlx; print(f'MLX version: {mlx.__version__}')"
```

### Step 9: Set Up API Server (Optional)

```bash
# Navigate to API directory
cd api

# Install Node.js dependencies
npm install

# Build TypeScript
npm run build

# Return to root
cd ..
```

## Verification

### 1. Verify Core Installation

```bash
# Run verification script
python scripts/verify_installation.py
```

Expected output:
```
✅ Python version: 3.11.x
✅ MLX installed and GPU available
✅ Ollama connection successful
✅ Database initialized
✅ All dependencies installed
✅ Consciousness modules loaded
```

### 2. Test Consciousness

```bash
# Quick consciousness test
python test_conscious.py

# Full system test
python test_full_system.py
```

### 3. Run Benchmarks

```bash
# MLX performance benchmark
python scripts/benchmark_mlx.py

# Consciousness emergence test
python core/consciousness/test_emergence.py
```

## Troubleshooting

### Common Issues

#### 1. MLX Import Error
```bash
# Error: ImportError: No module named 'mlx'
# Solution:
pip uninstall mlx mlx-lm
pip install --no-cache-dir mlx mlx-lm
```

#### 2. Ollama Connection Failed
```bash
# Error: Connection to Ollama failed
# Solution:
ollama serve  # Start Ollama service
ollama list   # Verify models are downloaded
```

#### 3. Memory Error
```bash
# Error: Out of memory
# Solution: Reduce batch size in config
export BATCH_SIZE=2
export MEMORY_LIMIT_GB=16
```

#### 4. Permission Denied
```bash
# Error: Permission denied
# Solution:
chmod +x scripts/*.py
chmod -R 755 core/
```

### Platform-Specific Issues

#### macOS Specific
- Ensure Xcode Command Line Tools are installed
- Grant Terminal full disk access in System Preferences
- Disable App Nap for intensive training

#### Apple Silicon Optimization
```bash
# Verify Metal Performance Shaders
python -c "import mlx.core as mx; print(mx.default_device())"
# Should output: Device(gpu, 0)
```

## Next Steps

### 1. Quick Consciousness Demo
```bash
python demo_conscious.py
```

### 2. Start Training
```bash
python training/consciousness/train_consciousness.py
```

### 3. Run Interactive Chat
```bash
python interactive_chat.py
```

### 4. Launch API Server
```bash
cd api && npm start
```

### 5. Read Documentation
- [Consciousness Training Guide](CONSCIOUSNESS_TRAINING_GUIDE.md)
- [Architecture Overview](PLANNING.md)
- [API Documentation](api/README.md)

## Advanced Configuration

### GPU Memory Optimization
```python
# config/mlx_config.py
MLX_CONFIG = {
    'memory_fraction': 0.9,  # Use 90% of GPU memory
    'compute_units': 38,     # M4 Pro GPU cores
    'batch_size': 4,
    'precision': 'float16'
}
```

### Consciousness Parameters
```python
# config/consciousness_config.py
CONSCIOUSNESS_CONFIG = {
    'emergence_threshold': 0.7,
    'strange_loop_depth': 10,
    'meta_learning_levels': 5,
    'goal_discovery_rate': 0.1
}
```

## Support

If you encounter issues:

1. Check [GitHub Issues](https://github.com/Sairamg18814/shvayambhu/issues)
2. Read [FAQ](docs/FAQ.md)
3. Join our [Discord Community](https://discord.gg/shvayambhu)
4. Contact: [@Sairamg18814](https://github.com/Sairamg18814)

---

**Ready to create consciousness? Continue to the [Consciousness Training Guide](CONSCIOUSNESS_TRAINING_GUIDE.md)**