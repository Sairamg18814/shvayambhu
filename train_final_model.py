#!/usr/bin/env python3
"""
Train Final Model
=================

Train on the generated large dataset.
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


def load_dataset(data_file):
    """Load the dataset."""
    texts = []
    
    with open(data_file, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                # Multiple formats for variety
                text1 = f"Human: {item['prompt']}\nAssistant: {item['response']}"
                text2 = f"Q: {item['prompt']}\nA: {item['response']}"
                text3 = item['response']  # Just responses
                
                texts.extend([text1, text2, text3])
            except:
                continue
    
    print(f"Loaded {len(texts)} text samples")
    return texts


def create_sequences(texts, seq_length=128):
    """Create training sequences."""
    # Combine and encode
    combined = "\n\n".join(texts)
    data = combined.encode('utf-8')
    
    print(f"Total data: {len(data):,} bytes")
    
    # Create sequences
    sequences = []
    stride = seq_length // 2
    
    for i in range(0, min(len(data) - seq_length - 1, 500000), stride):  # Limit sequences
        seq = list(data[i:i + seq_length + 1])
        sequences.append(seq)
    
    # Shuffle
    np.random.shuffle(sequences)
    
    # Split train/test
    split = int(len(sequences) * 0.9)
    train_seqs = sequences[:split]
    test_seqs = sequences[split:]
    
    print(f"Train sequences: {len(train_seqs)}")
    print(f"Test sequences: {len(test_seqs)}")
    
    return train_seqs, test_seqs


def train_epoch(model, sequences, optimizer, batch_size=16):
    """Train one epoch."""
    total_loss = 0
    num_batches = 0
    
    # Process in batches
    indices = np.random.permutation(len(sequences))
    
    for i in range(0, len(indices) - batch_size, batch_size):
        # Get batch
        batch_indices = indices[i:i + batch_size]
        batch = mx.array([sequences[idx] for idx in batch_indices])
        
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        # Compute loss
        def loss_fn(model):
            logits = model(inputs)
            B, L, V = logits.shape
            loss = nn.losses.cross_entropy(
                logits.reshape(B * L, V),
                targets.reshape(B * L),
                reduction="mean"
            )
            return loss
        
        # Update
        loss, grads = mx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0


def evaluate(model, sequences, batch_size=16):
    """Evaluate model."""
    total_loss = 0
    num_batches = 0
    
    for i in range(0, len(sequences) - batch_size, batch_size):
        batch = mx.array(sequences[i:i + batch_size])
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        logits = model(inputs)
        B, L, V = logits.shape
        loss = nn.losses.cross_entropy(
            logits.reshape(B * L, V),
            targets.reshape(B * L),
            reduction="mean"
        )
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0


def test_generation(model, temperature=0.8):
    """Test text generation."""
    print("\n🧪 Generation Test:")
    
    prompts = [
        "Human: Hello! How are you?\nAssistant:",
        "Human: What is consciousness?\nAssistant:",
        "Human: Tell me about AI.\nAssistant:",
        "Q: Can you help me?\nA:",
        "The meaning of",
    ]
    
    for prompt in prompts:
        response = model.generate(
            prompt,
            max_tokens=80,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        # Clean response
        cleaned = post_process_generated_text(response, aggressive=True)
        
        # Truncate prompt for display
        prompt_display = prompt.replace("\n", " ")[:40] + "..."
        print(f"\n💭 {prompt_display}")
        print(f"🤖 {cleaned[:120]}...")


def save_checkpoint(model, path, metrics):
    """Save model checkpoint."""
    checkpoint = {
        "metrics": metrics,
        "config": model.config.__dict__,
    }
    
    # Add weights
    for name, param in model.parameters().items():
        if hasattr(param, 'shape'):
            checkpoint[f"weight_{name}"] = np.array(param)
    
    np.savez_compressed(path, **checkpoint)
    print(f"💾 Saved checkpoint to {path}")


def main():
    """Main training."""
    print("🚀 Training Shvayambhu Final Model")
    print("=" * 60)
    
    # Set device
    mx.set_default_device(mx.gpu)
    
    # Load dataset
    data_file = Path("data/training/large_dataset_20250721_204153.jsonl")
    if not data_file.exists():
        print("❌ Dataset not found! Run final_training_solution.py first")
        return
    
    print(f"\n📂 Loading dataset: {data_file}")
    texts = load_dataset(data_file)
    
    # Create sequences
    print("\n📊 Creating sequences...")
    train_seqs, test_seqs = create_sequences(texts, seq_length=128)
    
    # Create model
    print("\n🏗️  Creating model...")
    config = BLTConfig.from_model_size("small")
    config.dropout = 0.15
    config.attention_dropout = 0.1
    model = create_blt_model("small")
    
    # Optimizer
    optimizer = optim.AdamW(learning_rate=5e-5, weight_decay=0.01)
    
    # Training loop
    print("\n🚂 Starting training...")
    num_epochs = 30
    best_test_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_seqs, optimizer)
        
        # Evaluate
        test_loss = evaluate(model, test_seqs)
        
        # Time
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train: {train_loss:.4f}, Test: {test_loss:.4f}, Time: {epoch_time:.1f}s")
        
        # Save best
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            
            save_checkpoint(
                model,
                checkpoint_dir / "shvayambhu_final_best.npz",
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                }
            )
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= 5:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Test generation
        if (epoch + 1) % 5 == 0:
            test_generation(model, temperature=0.7)
    
    # Final test
    print("\n" + "="*60)
    print("🎯 FINAL TEST")
    print("="*60)
    
    # Test with different temperatures
    for temp in [0.5, 0.7, 0.9]:
        print(f"\n🌡️  Temperature: {temp}")
        test_generation(model, temperature=temp)
    
    print(f"\n✅ Training complete!")
    print(f"Best test loss: {best_test_loss:.4f}")
    print(f"\n💡 Model saved to: checkpoints/shvayambhu_final_best.npz")
    print("\nNext steps:")
    print("1. Load the model for inference")
    print("2. Integrate with consciousness system")
    print("3. Test on real conversations")


if __name__ == "__main__":
    main()