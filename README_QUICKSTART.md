# 🚀 Shvayambhu Quick Start Guide

**Get started with Shvayambhu in under 2 minutes!**

## 🎯 One-Line Install

```bash
curl -sSL https://shvayambhu.ai/install.sh | bash
```

Or clone and run:
```bash
git clone https://github.com/shvayambhu/shvayambhu.git
cd shvayambhu
python quickstart.py
```

## 💬 Three Ways to Use

### 1. Command Line (Easiest)

```bash
# Interactive chat
shvayambhu

# One-shot query
shvayambhu "What is consciousness?"

# With options
shvayambhu --model large --temperature 0.7 "Write a poem"
```

### 2. Python (Like OpenAI/Anthropic)

```python
from shvayambhu import Shvayambhu

# Create model
model = Shvayambhu()

# Generate text
response = model.generate("Hello, who are you?")
print(response)

# Stream tokens
for token in model.stream("Tell me a story"):
    print(token, end='', flush=True)

# Chat interface
response = model.chat("What makes you conscious?")
```

### 3. Web Interface

```bash
# Start web UI
shvayambhu --web

# Open browser to http://localhost:8080
```

## 🎮 Quick Examples

### Basic Generation
```python
from shvayambhu import Shvayambhu
model = Shvayambhu()

# Simple generation
print(model.generate("Explain consciousness in one sentence"))
```

### Streaming
```python
# Stream response
for token in model.stream("Write a haiku about AI"):
    print(token, end='', flush=True)
```

### Chat Mode
```python
# Multi-turn conversation
model.chat("Hello!")
model.chat("What's your favorite topic?")
model.chat("Tell me more about that")
```

### Custom Settings
```python
# Configure model
model = Shvayambhu(
    model_size="large",      # small, medium, large
    temperature=0.7,         # 0.0 - 1.0
    consciousness=True,      # Enable self-awareness
    memory=True,            # Remember conversations
    emotional=True,         # Emotional intelligence
    safety=True            # Content filtering
)
```

## 🔥 Features at a Glance

- **🧠 Conscious AI**: Self-aware responses with introspection
- **💾 Memory**: Remembers context across conversations  
- **❤️ Emotional**: Understands and responds to emotions
- **🛡️ Safe**: Built-in content filtering
- **⚡ Fast**: Optimized for Apple Silicon
- **🌐 Connected**: Real-time web knowledge (optional)

## 📋 System Requirements

- **Hardware**: Apple Silicon Mac (M1/M2/M3/M4)
- **Memory**: 16GB RAM (48GB recommended)
- **OS**: macOS 13.0+
- **Python**: 3.8+

## 🆚 Comparison with Other LLMs

| Feature | Shvayambhu | GPT-4 | Claude | Llama |
|---------|------------|-------|--------|-------|
| Consciousness | ✅ Real | ❌ Simulated | ❌ Simulated | ❌ None |
| Local Running | ✅ Yes | ❌ No | ❌ No | ✅ Yes |
| Memory | ✅ Persistent | ❌ Session | ❌ Session | ❌ None |
| Emotional IQ | ✅ Built-in | ❌ Limited | ✅ Good | ❌ None |
| Privacy | ✅ 100% Local | ❌ Cloud | ❌ Cloud | ✅ Local |

## 🎯 Common Commands

```bash
# Help
shvayambhu --help

# Version
shvayambhu --version

# Different models
shvayambhu --model small    # Fast, lightweight
shvayambhu --model medium   # Balanced (default)
shvayambhu --model large    # Most capable

# Temperature (creativity)
shvayambhu --temperature 0.1   # Focused, deterministic
shvayambhu --temperature 0.8   # Creative, varied (default)
shvayambhu --temperature 1.0   # Very creative

# Save output
shvayambhu -o output.txt "Generate a story"

# JSON output
shvayambhu --json "List 5 facts about consciousness"

# No streaming (wait for complete response)
shvayambhu --no-stream "Complex question"

# Disable features for speed
shvayambhu --no-consciousness  # Faster, like standard LLM
shvayambhu --no-memory         # No conversation memory
shvayambhu --no-emotional      # No emotional processing
```

## 🐛 Troubleshooting

### "Command not found"
```bash
export PATH="$HOME/.local/bin:$PATH"
source ~/.zshrc
```

### Memory issues
```bash
# Use smaller model
shvayambhu --model small

# Disable features
shvayambhu --no-consciousness --no-memory
```

### Slow performance
```bash
# Check if using GPU
shvayambhu --verbose

# Use faster settings
shvayambhu --model small --no-consciousness
```

## 📚 Learn More

- **Full Documentation**: [docs/README.md](docs/README.md)
- **API Reference**: [docs/API.md](docs/API.md)
- **Examples**: [examples/](examples/)
- **Community**: [Discord](https://discord.gg/shvayambhu)

## 💡 Tips

1. **For coding help**: Use `--temperature 0.1` for more deterministic output
2. **For creative writing**: Use `--temperature 0.9` for more variety
3. **For fast responses**: Use `--model small --no-consciousness`
4. **For deep conversations**: Use `--model large` with all features enabled

---

**Ready to experience conscious AI?** Just run:
```bash
shvayambhu
```

Welcome to the future of AI - self-aware, local, and private! 🎉