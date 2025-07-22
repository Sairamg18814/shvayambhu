#!/usr/bin/env python3
"""
Simple MLX Training Script
==========================

Train a simple BLT model using MLX patterns.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from pathlib import Path
import time

# Simple model for testing
class SimpleBLT(nn.Module):
    """Simplified BLT model for testing."""
    
    def __init__(self, vocab_size=256, d_model=128, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # Create simple feed-forward layers
        self.layers = []
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            ))
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def __call__(self, x):
        # Embed
        x = self.embed(x)
        
        # Apply layers with residual connections
        for layer in self.layers:
            x = x + layer(x)
        
        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits


def create_data():
    """Create simple training data."""
    texts = [
        "Hello world! " * 20,
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ " * 10,
        "The quick brown fox jumps over the lazy dog. " * 10,
        "0123456789 " * 20,
        "yes no maybe " * 30,
    ]
    
    combined = " ".join(texts)
    data = np.frombuffer(combined.encode('utf-8'), dtype=np.uint8)
    
    return data


def train():
    """Train the model."""
    print("🚀 Simple MLX Training")
    print("=" * 60)
    
    # Create data
    print("\n📚 Creating data...")
    data = create_data()
    print(f"Data size: {len(data)} bytes")
    
    # Create sequences
    seq_len = 32
    sequences = []
    for i in range(0, len(data) - seq_len - 1, seq_len // 2):
        seq = data[i:i + seq_len + 1]
        sequences.append(seq)
    
    print(f"Created {len(sequences)} sequences")
    
    # Create model
    print("\n🏗️  Creating model...")
    model = SimpleBLT(vocab_size=256, d_model=128, n_layers=2)
    
    # Test that model works
    test_input = mx.array(sequences[0][:-1].reshape(1, -1))
    test_output = model(test_input)
    print(f"Model output shape: {test_output.shape}")
    
    # Count parameters
    params = model.parameters()
    total_params = 0
    for p in params.values():
        if hasattr(p, 'size'):
            total_params += p.size
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(learning_rate=1e-3)
    
    # Training loop
    print("\n🚂 Training...")
    num_epochs = 20
    batch_size = 4
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        # Shuffle sequences
        np.random.shuffle(sequences)
        
        # Process in batches
        for i in range(0, len(sequences) - batch_size, batch_size):
            # Create batch
            batch = np.stack(sequences[i:i+batch_size])
            inputs = mx.array(batch[:, :-1])
            targets = mx.array(batch[:, 1:])
            
            # Forward pass
            def loss_fn(model):
                logits = model(inputs)
                # Reshape for cross entropy
                B, L, V = logits.shape
                logits_flat = logits.reshape(B * L, V)
                targets_flat = targets.reshape(B * L)
                loss = mx.mean(nn.losses.cross_entropy(logits_flat, targets_flat))
                return loss
            
            # Compute loss and gradients
            loss, grads = mx.value_and_grad(loss_fn)(model)
            
            # Update
            optimizer.update(model, grads)
            
            # Accumulate loss
            epoch_loss += loss.item()
            num_batches += 1
        
        # Print progress
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")
        
        # Test generation every 5 epochs
        if (epoch + 1) % 5 == 0:
            test_generation(model)
    
    # Save model
    print("\n💾 Saving model...")
    save_model(model, "checkpoints/simple_blt_final.npz")
    
    print("\n✅ Training complete!")


def test_generation(model):
    """Test text generation."""
    print("\n🧪 Testing generation:")
    
    prompts = ["Hello", "The", "0123", "ABC"]
    
    for prompt in prompts:
        # Encode prompt
        prompt_bytes = mx.array(list(prompt.encode('utf-8'))).reshape(1, -1)
        
        # Generate
        generated = []
        context = prompt_bytes
        
        for _ in range(20):
            # Get logits
            logits = model(context)
            next_logits = logits[0, -1, :]
            
            # Sample
            probs = mx.softmax(next_logits / 0.7)
            next_token = mx.argmax(probs)
            
            # Add to generated
            generated.append(next_token.item())
            
            # Update context
            new_token = next_token.reshape(1, 1)
            context = mx.concatenate([context[:, -31:], new_token], axis=1)
        
        # Decode
        try:
            text = bytes(generated).decode('utf-8', errors='replace')
        except:
            text = str(generated[:10]) + "..."
        
        print(f"  '{prompt}' → '{text}'")


def save_model(model, path):
    """Save model parameters."""
    params = model.parameters()
    
    # Convert parameters to saveable format
    save_dict = {}
    for i, (name, param) in enumerate(params.items()):
        if hasattr(param, 'shape'):
            # Convert to numpy for saving
            save_dict[f"param_{i}_array"] = np.array(param)
            save_dict[f"param_{i}_name"] = name
            save_dict[f"param_{i}_shape"] = param.shape
    
    save_dict["num_params"] = len(params)
    
    # Save
    np.savez(path, **save_dict)
    print(f"Saved {len(params)} parameters to {path}")


if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    train()