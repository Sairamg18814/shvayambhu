#!/usr/bin/env python3
"""
Train and Save Model Properly
=============================

Train the BLT model and save weights correctly.
"""

import sys
import time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from core.blt.full_model import create_blt_model, BLTConfig
from utils.text_postprocessing import post_process_generated_text


def create_simple_dataset():
    """Create a simple dataset for testing."""
    # Simple patterns to learn
    texts = [
        # Basic repetitive patterns
        "Hello world! " * 10,
        "The quick brown fox jumps over the lazy dog. " * 5,
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ " * 10,
        "0123456789 " * 20,
        
        # Simple conversations
        "Q: What is your name? A: I am Shvayambhu. " * 5,
        "Q: How are you? A: I am doing well, thank you. " * 5,
        "Q: Can you help me? A: Yes, I can help you. " * 5,
        
        # Basic patterns
        "AAABBBCCC " * 10,
        "123123123 " * 10,
        "yesyesyes nonono " * 10,
    ]
    
    # Combine and encode
    combined = "\n".join(texts)
    data = combined.encode('utf-8')
    
    return data


def create_batches(data, seq_length=64, batch_size=4):
    """Create training batches."""
    sequences = []
    
    # Create overlapping sequences
    stride = seq_length // 2
    for i in range(0, len(data) - seq_length - 1, stride):
        seq = list(data[i:i + seq_length + 1])
        sequences.append(seq)
    
    # Shuffle
    np.random.shuffle(sequences)
    
    # Create batches
    batches = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        if len(batch) == batch_size:
            batches.append(mx.array(batch))
    
    return batches


def train_step(model, batch, optimizer):
    """Single training step."""
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    
    def loss_fn(model):
        logits = model(inputs)
        # Ensure logits are the right shape
        B, L, V = logits.shape
        loss = nn.losses.cross_entropy(
            logits.reshape(B * L, V),
            targets.reshape(B * L),
            reduction="mean"
        )
        return loss
    
    # Compute loss and gradients
    loss, grads = mx.value_and_grad(loss_fn)(model)
    
    # Update parameters
    optimizer.update(model, grads)
    
    # Force evaluation
    mx.eval(model.parameters(), optimizer.state)
    
    return loss.item()


def save_model(model, path, epoch, loss):
    """Save model weights properly."""
    # Get all parameters
    params = model.parameters()
    
    # MLX needs to save arrays, so we'll save the flattened parameters
    # Create a list of all parameter arrays
    param_list = []
    param_shapes = []
    param_names = []
    
    for name, param in params.items():
        if hasattr(param, 'shape'):
            param_list.append(param)
            param_shapes.append(param.shape)
            param_names.append(name)
    
    # Save using numpy format which MLX can load
    import numpy as np
    np.savez(path, 
             epoch=epoch,
             loss=loss,
             param_names=param_names,
             param_shapes=param_shapes,
             **{f"param_{i}": p for i, p in enumerate(param_list)})
    
    print(f"💾 Saved model to {path} ({len(param_list)} parameter tensors)")


def test_generation(model, epoch):
    """Test model generation."""
    print(f"\n🧪 Testing at epoch {epoch}:")
    
    prompts = ["Hello", "The", "123", "Q: "]
    
    for prompt in prompts:
        # Encode prompt
        prompt_bytes = prompt.encode('utf-8')
        prompt_array = mx.array([list(prompt_bytes)])
        
        # Generate
        generated = []
        context = prompt_array
        
        for _ in range(20):
            # Get logits
            logits = model(context)
            
            # Sample from last position
            last_logits = logits[0, -1, :]
            
            # Apply temperature
            probs = mx.softmax(last_logits / 0.7)
            
            # Sample
            next_byte = mx.random.categorical(mx.log(probs))
            
            # Append
            generated.append(next_byte.item())
            
            # Update context (keep last 63 tokens for context window)
            new_context = mx.concatenate([context[:, -63:], next_byte.reshape(1, 1)], axis=1)
            context = new_context
        
        # Decode
        try:
            generated_text = bytes(generated).decode('utf-8', errors='replace')
        except:
            generated_text = str(generated)
        
        print(f"  '{prompt}' → '{generated_text}'")


def main():
    """Main training routine."""
    print("🚀 Training BLT Model with Proper Saving")
    print("=" * 60)
    
    # Set device
    mx.set_default_device(mx.gpu)
    
    # Create simple dataset
    print("\n📚 Creating dataset...")
    data = create_simple_dataset()
    print(f"Dataset size: {len(data):,} bytes")
    
    # Create batches
    print("\n📦 Creating batches...")
    batches = create_batches(data, seq_length=64, batch_size=4)
    print(f"Number of batches: {len(batches)}")
    
    # Create model
    print("\n🏗️  Creating model...")
    # Create a small model for testing
    from core.blt.full_model import BLTModel
    config = BLTConfig(
        vocab_size=256,
        d_model=256,
        n_layers=4,
        n_heads=4,
        mlp_dim=512,
        dropout=0.1
    )
    model = BLTModel(config)
    
    # Count parameters
    param_count = sum(np.prod(p.shape) for p in model.parameters().values() if hasattr(p, 'shape'))
    print(f"Model parameters: {param_count:,}")
    
    # Create optimizer
    optimizer = optim.AdamW(learning_rate=3e-4, weight_decay=0.01)
    
    # Training
    print("\n🚂 Starting training...")
    num_epochs = 50
    best_loss = float('inf')
    
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        # Shuffle batches
        np.random.shuffle(batches)
        
        # Train on batches
        for batch in batches:
            loss = train_step(model, batch, optimizer)
            epoch_loss += loss
        
        # Average loss
        avg_loss = epoch_loss / len(batches)
        
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")
        
        # Save if best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = checkpoint_dir / f"blt_model_epoch{epoch+1}_loss{avg_loss:.4f}.npz"
            save_model(model, str(save_path), epoch + 1, avg_loss)
            
            # Also save as 'best'
            best_path = checkpoint_dir / "blt_model_best.npz"
            save_model(model, str(best_path), epoch + 1, avg_loss)
        
        # Test every 10 epochs
        if (epoch + 1) % 10 == 0:
            test_generation(model, epoch + 1)
    
    # Final test
    print("\n" + "="*60)
    print("🎯 Final Test")
    test_generation(model, "Final")
    
    print(f"\n✅ Training complete! Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()