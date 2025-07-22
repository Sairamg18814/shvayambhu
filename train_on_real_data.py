#!/usr/bin/env python3
"""
Train on Real Ollama Data
=========================

Train the BLT model on real Ollama-generated data.
"""

import sys
import json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent))

from core.blt.full_model import create_blt_model, BLTConfig
from utils.text_postprocessing import post_process_generated_text


def load_real_data(data_file):
    """Load the real Ollama data."""
    samples = []
    
    with open(data_file, 'r') as f:
        for line in f:
            try:
                sample = json.loads(line)
                # Format as conversation
                text = f"Human: {sample['prompt']}\nAssistant: {sample['response']}"
                samples.append(text)
            except:
                continue
    
    print(f"Loaded {len(samples)} real samples")
    return samples


def create_sequences(texts, seq_length=256):
    """Create training sequences from texts."""
    # Combine all texts
    combined = "\n\n".join(texts)
    
    # Repeat a few times for more data
    combined = combined * 3
    
    # Encode to bytes
    data = combined.encode('utf-8')
    print(f"Total data: {len(data):,} bytes")
    
    # Create sequences
    sequences = []
    stride = seq_length // 2
    
    for i in range(0, len(data) - seq_length - 1, stride):
        seq = list(data[i:i + seq_length + 1])
        sequences.append(seq)
    
    return sequences


def train_epoch(model, sequences, optimizer, batch_size=8):
    """Train for one epoch."""
    np.random.shuffle(sequences)
    
    total_loss = 0
    num_batches = 0
    
    for i in range(0, len(sequences) - batch_size, batch_size):
        # Create batch
        batch = mx.array(sequences[i:i + batch_size])
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        # Loss function
        def loss_fn(model):
            logits = model(inputs)
            B, L, V = logits.shape
            loss = nn.losses.cross_entropy(
                logits.reshape(B * L, V),
                targets.reshape(B * L),
                reduction="mean"
            )
            return loss
        
        # Compute gradients
        loss, grads = mx.value_and_grad(loss_fn)(model)
        
        # Update
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0


def test_generation(model, epoch):
    """Test model generation."""
    print(f"\n🧪 Testing at epoch {epoch}:")
    
    test_prompts = [
        "Human: What is consciousness?\nAssistant:",
        "Human: Hello! How are you?\nAssistant:",
        "Human: Can you help me understand neural networks?\nAssistant:",
        "Human: Tell me about yourself.\nAssistant:",
    ]
    
    for prompt in test_prompts:
        # Generate response
        response = model.generate(
            prompt,
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        # Clean up
        cleaned = post_process_generated_text(response, aggressive=True)
        
        # Display
        print(f"\n💭 {prompt.split('Human:')[1].split('Assistant:')[0].strip()}")
        print(f"🤖 {cleaned[:150]}...")


def save_checkpoint(model, path, epoch, loss):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "loss": loss,
        "config": model.config.__dict__,
        "weights": {k: np.array(v) for k, v in model.parameters().items() if hasattr(v, 'shape')}
    }
    
    np.savez_compressed(path, **checkpoint)
    print(f"💾 Saved checkpoint to {path}")


def main():
    """Main training routine."""
    print("🚀 Training on Real Ollama Data")
    print("=" * 60)
    
    # Set device
    mx.set_default_device(mx.gpu)
    
    # Find the latest real data file
    data_dir = Path("data/training")
    real_data_files = list(data_dir.glob("ollama_real_data_*.jsonl"))
    
    if not real_data_files:
        print("❌ No real data files found! Run generate_ollama_data_direct.py first")
        return
    
    latest_data = sorted(real_data_files)[-1]
    print(f"\n📂 Using data file: {latest_data}")
    
    # Load data
    texts = load_real_data(latest_data)
    
    # Create sequences
    print("\n📊 Creating training sequences...")
    sequences = create_sequences(texts, seq_length=256)
    print(f"Created {len(sequences)} sequences")
    
    # Create model
    print("\n🏗️  Creating model...")
    config = BLTConfig.from_model_size("small")
    config.dropout = 0.1
    config.attention_dropout = 0.1
    model = create_blt_model("small")
    
    # Create optimizer with lower learning rate
    optimizer = optim.AdamW(learning_rate=5e-5, weight_decay=0.01)
    
    # Training parameters
    num_epochs = 30
    batch_size = 8
    best_loss = float('inf')
    
    print(f"\n🚂 Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train
        avg_loss = train_epoch(model, sequences, optimizer, batch_size)
        
        # Time
        elapsed = time.time() - start_time
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f} - Time: {elapsed:.1f}s")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            save_checkpoint(
                model,
                checkpoint_dir / "real_data_best.npz",
                epoch + 1,
                avg_loss
            )
        
        # Test every 5 epochs
        if (epoch + 1) % 5 == 0:
            test_generation(model, epoch + 1)
    
    # Final test
    print("\n" + "="*60)
    print("🎯 Final Test")
    print("="*60)
    test_generation(model, "Final")
    
    print(f"\n✅ Training complete! Best loss: {best_loss:.4f}")
    print("\n💡 Model trained on real Ollama data should generate more coherent responses.")


if __name__ == "__main__":
    main()