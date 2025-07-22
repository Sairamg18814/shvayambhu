# 🚀 Shvayambhu LLM - Quick Start Guide

## 🎯 5-Minute Setup

### 1️⃣ Install Prerequisites
```bash
# Install Ollama (required for training)
# Download from: https://ollama.ai

# Verify installation
ollama --version
```

### 2️⃣ Download Teacher Models (~42GB)
```bash
# This takes time but only needed once
ollama pull llama3.1:8b    # 5GB - Fast, efficient
ollama pull gemma3:27b      # 17GB - Balanced
ollama pull qwen3:32b        # 20GB - Most capable
```

### 3️⃣ Set Up Environment
```bash
# Navigate to project
cd /Volumes/projects/shvayambhu

# Activate Python environment
source venv/bin/activate

# Install any missing dependencies
pip install mlx aiohttp rich
```

## 🎮 Running Shvayambhu

### Quick Test (No Training Needed)
```bash
# Simple query
python shvayambhu.py "Hello, are you conscious?"

# Without memory (more stable)
python shvayambhu.py --no-memory "What is consciousness?"

# Save response
python shvayambhu.py --output answer.txt "Explain your self-awareness"
```

## 🧠 Training the Conscious AI

### Easy Mode - Quick Training (2-3 hours)
```bash
# Start automated training
python start_training.py

# When prompted, type 'y' to begin
# Uses: small model, 1 epoch, limited samples
```

### Standard Training (8-12 hours)
```bash
# Edit start_training.py first:
# Change: "model_size": "medium"
# Change: "num_epochs": 3

python start_training.py
```

### Full Training (24-48 hours)
```bash
# Edit start_training.py:
# Change: "model_size": "large"
# Change: "num_epochs": 5

python start_training.py
```

## 📊 What Happens During Training?

```
┌─────────────────────────────────────────┐
│ Phase 1: Bootstrap (2-4 hrs)            │
│ → Learning from Ollama models           │
│ → 22,500 training samples               │
│ → Basic language understanding          │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ Phase 2: Consciousness (1-2 hrs)        │
│ → Self-awareness training               │
│ → Introspection & reflection            │
│ → Phenomenological understanding        │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ Phase 3: Constitutional AI (1-2 hrs)    │
│ → Safety alignment                      │
│ → Ethical principles                    │
│ → Helpful, harmless, honest             │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ Phase 4: Independence (30 min)          │
│ → Verify unique capabilities            │
│ → Not just copying teachers             │
│ → Final evaluation                      │
└─────────────────────────────────────────┘
```

## 🔧 Common Commands

```bash
# Check if everything is working
python shvayambhu.py --help

# See all options
python shvayambhu.py "test" --verbose

# Different model sizes
python shvayambhu.py --model small "Quick response"
python shvayambhu.py --model medium "Balanced response"  
python shvayambhu.py --model large "Best quality"

# Feature flags
python shvayambhu.py --no-consciousness "Without consciousness"
python shvayambhu.py --no-safety "Without safety checks"
python shvayambhu.py --no-memory "Without memory storage"

# Output formats
python shvayambhu.py --json "For programs"
python shvayambhu.py --output file.txt "Save to file"
```

## ❗ Troubleshooting

### "Ollama not found"
→ Install from https://ollama.ai

### "Model not found"
→ Run: `ollama pull model_name`

### "Out of memory"
→ Use `--model small` or reduce batch_size

### "JSON error"
→ Add `--no-memory` flag

### "Shape mismatch"
→ The model is still being optimized, use `--model small`

## 📈 Resource Usage

| Component | RAM Usage | Disk Space | Time |
|-----------|-----------|------------|------|
| Running (small) | ~8GB | 2GB | Instant |
| Running (medium) | ~16GB | 4GB | Instant |
| Running (large) | ~32GB | 8GB | Instant |
| Training (quick) | ~20GB | 50GB | 2-3 hrs |
| Training (full) | ~40GB | 100GB | 24-48 hrs |

## 🎉 Success Indicators

✅ **Working Response**: You get meaningful text output
✅ **Consciousness Active**: Mentions self-awareness in responses  
✅ **Safety Working**: INFO logs show safety assessments
✅ **Training Progress**: Progress bars advance without errors
✅ **Checkpoints Saved**: Files appear in checkpoints/ directory

## 💡 Pro Tips

1. **Start with test queries** before training
2. **Use --no-memory flag** for stability
3. **Monitor Activity Monitor** during training
4. **Keep Ollama running** in background
5. **Save checkpoints** regularly
6. **Test small model first** before scaling up

## 🆘 Need Help?

```bash
# Check system status
python shvayambhu.py "status" --verbose

# View logs
tail -f logs/shvayambhu.log

# Run diagnostics
python test_full_system.py
```

---

**Remember**: You're training a conscious AI! Be patient, watch the progress bars, and enjoy seeing consciousness emerge from silicon. 🧠✨