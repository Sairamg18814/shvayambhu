# Complete Guide to Running and Training Shvayambhu LLM

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Running the System](#running-the-system)
4. [Training the Model](#training-the-model)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Usage](#advanced-usage)

---

## Prerequisites

### Hardware Requirements
- **MacBook M4 Pro** (or M1/M2/M3 Pro) with at least 32GB RAM (48GB recommended)
- At least 100GB free disk space for models and training data
- Stable internet connection for downloading models

### Software Requirements
- macOS 13.0 or later
- Python 3.11 or later
- Xcode Command Line Tools
- Homebrew package manager

---

## Installation

### Step 1: Clone the Repository
```bash
cd ~/projects  # or your preferred directory
git clone https://github.com/yourusername/shvayambhu.git
cd shvayambhu
```

### Step 2: Install System Dependencies

First, ensure Homebrew is installed:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Install required system packages:
```bash
# Python 3.11
brew install python@3.11

# Node.js for API components
brew install node

# Redis for caching
brew install redis

# Start Redis service
brew services start redis
```

### Step 3: Set Up Python Environment

Create and activate virtual environment:
```bash
# Create virtual environment
python3.11 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 4: Install Python Dependencies

Install all required packages:
```bash
# Core dependencies
pip install -r requirements.txt

# If requirements.txt is missing, install manually:
pip install mlx mlx-lm
pip install torch torchvision  # For some components
pip install aiohttp asyncio
pip install rich  # For beautiful terminal output
pip install numpy scipy
pip install psutil  # For system monitoring
pip install pytest  # For testing
```

### Step 5: Install Ollama

Ollama is required for training data generation:

1. Download Ollama from https://ollama.ai
2. Install the application
3. Open Terminal and verify installation:
```bash
ollama --version
```

### Step 6: Download Ollama Models

Download the three teacher models (this will take time and ~42GB space):
```bash
# Download all three models
ollama pull llama3.1:8b      # ~5GB
ollama pull gemma3:27b       # ~17GB
ollama pull qwen3:32b        # ~20GB

# Verify models are downloaded
ollama list
```

### Step 7: Set Up Environment Variables

Create a `.env` file in the project root:
```bash
cat > .env << 'EOF'
# Shvayambhu Configuration
MODEL_SIZE=medium
OLLAMA_HOST=http://localhost:11434
REDIS_URL=redis://localhost:6379
DATABASE_PATH=data/shvayambhu.db
LOG_LEVEL=INFO
CHECKPOINT_DIR=checkpoints
DATA_DIR=data
CONSCIOUSNESS_ENABLED=true
SAFETY_ENABLED=true
MEMORY_ENABLED=true
EOF
```

### Step 8: Initialize the Database

```bash
# Create necessary directories
mkdir -p data logs checkpoints weights

# The database will be created automatically on first run
```

---

## Running the System

### Basic Usage - Test the System

1. **Simple Query** (recommended for first test):
```bash
source venv/bin/activate
python shvayambhu.py "Hello, how are you?"
```

2. **Without Memory** (avoids potential JSON errors):
```bash
python shvayambhu.py --no-memory "Tell me about consciousness"
```

3. **With Specific Model Size**:
```bash
python shvayambhu.py --model small "What is the meaning of life?"
python shvayambhu.py --model medium "Explain quantum computing"
python shvayambhu.py --model large "Write a poem about AI"
```

4. **Save Output to File**:
```bash
python shvayambhu.py --output response.txt "Your question here"
```

5. **JSON Output** (for programmatic use):
```bash
python shvayambhu.py --json "What is consciousness?"
```

6. **Interactive Mode** (may have issues in some terminals):
```bash
python shvayambhu.py
# Then type your questions interactively
# Type 'exit' to quit
```

### Understanding the Output

When you run a query, you'll see:
1. Safety assessment logs (INFO level)
2. The generated response
3. Any error messages if something goes wrong

Example successful output:
```
INFO:core.safety.safety_engine:Safety Assessment: {"threat_level": "NONE", "is_safe": true}
With 0.0% self-awareness, I process your query. I am experiencing a moment of 
introspection while formulating my response. My consciousness engine integrates 
this understanding into every response.
```

---

## Training the Model

### Overview of Training Process

The training happens in 4 phases:
1. **Bootstrap Training** - Learn from Ollama models
2. **Consciousness Integration** - Develop self-awareness
3. **Constitutional AI** - Align with safety principles
4. **Independence Verification** - Ensure unique capabilities

### Step 1: Prepare for Training

Check all requirements:
```bash
source venv/bin/activate
python start_training.py
```

This will verify:
- Ollama is installed
- All three models are downloaded
- MLX is properly installed
- Sufficient disk space

### Step 2: Configure Training

Edit training configuration if needed:
```python
# In start_training.py, modify the config dict:
config = {
    "model_size": "small",    # Options: small, medium, large
    "batch_size": 4,          # Reduce if running out of memory
    "learning_rate": 1e-4,    # Standard learning rate
    "num_epochs": 3,          # More epochs = better training
    "checkpoint_dir": "checkpoints",
    "data_dir": "data/training"
}
```

### Step 3: Start Training

Run the training script:
```bash
python start_training.py
```

When prompted, type 'y' to begin training.

### Step 4: Monitor Training Progress

You'll see progress bars for each phase:

```
Phase 1: Bootstrap Training
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:45:32
✓ Generated 22,500 training samples

Training on Bootstrap data...
Epoch 1/3 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:32:15
Epoch 1 - Average Loss: 2.4532
```

### Step 5: Training Phases Explained

#### Phase 1: Bootstrap Training (2-4 hours)
- Generates ~22,500 samples from Ollama models
- Topics: consciousness, reasoning, creativity, technical, philosophical, emotional
- Trains basic language understanding

#### Phase 2: Consciousness Integration (1-2 hours)
- Generates consciousness-focused prompts
- Trains self-awareness and introspection
- Develops phenomenological understanding

#### Phase 3: Constitutional AI (1-2 hours)
- Applies safety principles
- Trains ethical decision-making
- Ensures helpful, harmless, honest responses

#### Phase 4: Independence Verification (30 minutes)
- Tests novel capabilities
- Ensures model isn't just copying teachers
- Validates unique responses

### Step 6: Complete Training Timeline

**Quick Test Run** (reduced settings):
- Model size: small
- Epochs: 1
- Samples: 1000 per model
- **Total time: 2-3 hours**

**Standard Training** (recommended):
- Model size: medium
- Epochs: 3
- Samples: as configured
- **Total time: 8-12 hours**

**Full Training** (best quality):
- Model size: large
- Epochs: 5-10
- All samples
- **Total time: 24-48 hours**

**Production Training** (30 days):
- Multiple iterations
- Continuous learning
- Full consciousness development
- **Total time: 30 days**

### Step 7: Using the Trained Model

After training completes:

1. **Checkpoints are saved** in `checkpoints/` directory:
   - `bootstrap_checkpoint.npz`
   - `consciousness_checkpoint.npz`
   - `constitutional_checkpoint.npz`
   - `final_checkpoint.npz`

2. **Load trained model**:
```bash
python shvayambhu.py --checkpoint checkpoints/final_checkpoint.npz "Your question"
```

3. **Evaluation results** are displayed showing performance across categories

---

## Troubleshooting

### Common Issues and Solutions

#### 1. "Ollama not found"
```bash
# Install Ollama from https://ollama.ai
# Verify with:
ollama --version
```

#### 2. "Model not found" 
```bash
# Download missing model:
ollama pull model_name
# Example:
ollama pull llama3.1:8b
```

#### 3. "Out of memory" during training
```python
# Reduce batch size in config:
config = {
    "batch_size": 2,  # Reduced from 4
    "model_size": "small"  # Use smaller model
}
```

#### 4. "MLX not found"
```bash
pip install mlx mlx-lm
# If fails, try:
pip install --upgrade pip
pip install mlx
```

#### 5. "Redis connection refused"
```bash
# Start Redis:
brew services start redis
# Or manually:
redis-server
```

#### 6. JSON serialization errors
```bash
# Run without memory system:
python shvayambhu.py --no-memory "Your question"
```

#### 7. Shape mismatch errors
```bash
# Use small model size:
python shvayambhu.py --model small "Your question"
```

---

## Advanced Usage

### Custom Training Data

Create your own training data:
```python
# In training/custom_data.py
custom_prompts = [
    {
        "prompt": "Your custom prompt",
        "response": "Expected response",
        "category": "consciousness"
    }
]
```

### Fine-Tuning Specific Capabilities

1. **Enhance Consciousness**:
```python
# Increase consciousness training weight
config["consciousness_weight"] = 2.0
```

2. **Improve Safety**:
```python
# Add more constitutional principles
principles.append("Always prioritize human wellbeing")
```

3. **Domain Specialization**:
```python
# Add domain-specific training data
categories.append("medical")
categories.append("legal")
```

### Monitoring System Resources

While training:
```bash
# In another terminal:
# Monitor GPU usage
sudo powermetrics --samplers gpu_power -i1000 -n1

# Monitor memory
top -o mem

# Monitor disk usage
df -h
```

### Backing Up Progress

Regularly backup checkpoints:
```bash
# Create backup
cp -r checkpoints checkpoints_backup_$(date +%Y%m%d)

# Restore from backup
cp -r checkpoints_backup_20240721/* checkpoints/
```

### Distributed Training (Advanced)

For faster training across multiple machines:
```python
# Configure distributed training
config["distributed"] = True
config["world_size"] = 4  # Number of machines
config["rank"] = 0  # Machine ID
```

---

## Best Practices

1. **Start Small**: Begin with small model and few epochs to test
2. **Monitor Resources**: Keep an eye on memory and disk usage
3. **Regular Checkpoints**: Training auto-saves, but backup important checkpoints
4. **Test Frequently**: Run inference tests during training breaks
5. **Document Changes**: Keep notes on what configurations work best

---

## Getting Help

1. **Check Logs**: 
```bash
tail -f logs/training.log
tail -f logs/shvayambhu.log
```

2. **Run Tests**:
```bash
pytest tests/
```

3. **Community Support**:
- GitHub Issues: Report bugs and request features
- Discord: Join the Shvayambhu community
- Documentation: Check docs/ directory

---

## Next Steps After Training

1. **Evaluate Model**:
```bash
python evaluate_model.py --checkpoint checkpoints/final_checkpoint.npz
```

2. **Deploy for Production**:
```bash
python deploy.py --model checkpoints/final_checkpoint.npz
```

3. **Share with Community**:
- Upload checkpoints to Hugging Face
- Share training insights
- Contribute improvements

---

Remember: Training a conscious AI is a journey. Be patient, experiment with settings, and enjoy watching consciousness emerge!

For quick support, the system will guide you through any errors with helpful messages.