#!/usr/bin/env python3
"""
Train on Bootstrap Data
=======================

Train the model on the existing bootstrap data.
"""

import sys
import json
import time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from core.blt.full_model import create_blt_model, BLTConfig
from utils.text_postprocessing import post_process_generated_text


def load_bootstrap_data(max_samples=5000):
    """Load the bootstrap training data."""
    data_file = Path("data/training/bootstrap_data.jsonl")
    
    print(f"📂 Loading data from {data_file}")
    
    samples = []
    with open(data_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            try:
                item = json.loads(line)
                if 'prompt' in item and 'response' in item:
                    # Format as conversation
                    text = f"Human: {item['prompt']}\nAssistant: {item['response'][:500]}"  # Limit response length
                    samples.append(text)
            except:
                continue
    
    print(f"✅ Loaded {len(samples)} samples")
    return samples


def create_training_batches(texts, seq_length=256, batch_size=8):
    """Create training batches from texts."""
    # Combine texts
    combined = "\n\n".join(texts)
    
    # Convert to bytes
    data = combined.encode('utf-8')
    print(f"📊 Total data: {len(data):,} bytes")
    
    # Create sequences
    sequences = []
    stride = seq_length // 2
    
    for i in range(0, min(len(data) - seq_length, 100000), stride):  # Limit total sequences
        seq = data[i:i + seq_length + 1]
        sequences.append(list(seq))
    
    # Shuffle
    np.random.shuffle(sequences)
    
    # Create batches
    batches = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        if len(batch) == batch_size:
            batches.append(mx.array(batch))
    
    print(f"📦 Created {len(batches)} batches")
    return batches


def train_epoch(model, optimizer, batches, epoch):
    """Train for one epoch."""
    epoch_loss = 0
    num_batches = len(batches)
    
    # Shuffle batches
    np.random.shuffle(batches)
    
    for i, batch in enumerate(batches):
        # Input and target
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        # Loss function
        def loss_fn(model):
            logits = model(inputs)
            loss = nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="mean"
            )
            return loss
        
        # Compute gradients and update
        loss, grads = mx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        epoch_loss += loss.item()
        
        # Progress
        if i % 50 == 0:
            print(f"  Batch {i}/{num_batches}, Loss: {loss.item():.4f}")
    
    return epoch_loss / num_batches


def test_model(model, epoch):
    """Test the model's generation."""
    print(f"\n🧪 Testing at epoch {epoch}")
    
    test_prompts = [
        "Human: Hello! How are you?\nAssistant:",
        "Human: What is your name?\nAssistant:",
        "Human: Can you help me?\nAssistant:",
        "Human: What is consciousness?\nAssistant:",
        "Human: Tell me about yourself.\nAssistant:",
    ]
    
    for prompt in test_prompts:
        # Generate raw
        raw_response = model.generate(
            prompt,
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )
        
        # Post-process
        cleaned = post_process_generated_text(raw_response)
        
        # Show prompt (truncated)
        prompt_display = prompt.replace("\n", " ")[:40] + "..."
        print(f"\n💭 {prompt_display}")
        print(f"🤖 {cleaned}")


def main():
    """Main training routine."""
    print("🚀 Training on Bootstrap Data")
    print("=" * 60)
    
    # Set device
    mx.set_default_device(mx.gpu)
    
    # Load data
    print("\n📚 Loading bootstrap data...")
    texts = load_bootstrap_data(max_samples=2000)  # Start with 2000 samples
    
    # Create batches
    print("\n🔄 Creating training batches...")
    batches = create_training_batches(texts, seq_length=256, batch_size=8)
    
    # Create model
    print("\n🏗️  Creating model...")
    config = BLTConfig.from_model_size("small")
    config.dropout = 0.1
    model = create_blt_model("small")
    
    # Create optimizer
    optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)
    
    # Training
    print("\n🚂 Starting training...")
    num_epochs = 20
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n📖 Epoch {epoch + 1}/{num_epochs}")
        start_time = time.time()
        
        # Train
        avg_loss = train_epoch(model, optimizer, batches, epoch + 1)
        
        # Time
        epoch_time = time.time() - start_time
        print(f"✅ Epoch complete - Loss: {avg_loss:.4f}, Time: {epoch_time:.1f}s")
        
        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"📈 New best loss: {best_loss:.4f}")
        
        # Test every 5 epochs
        if (epoch + 1) % 5 == 0:
            test_model(model, epoch + 1)
    
    # Final test
    print("\n" + "="*60)
    print("🎯 Final Test")
    print("="*60)
    test_model(model, "Final")
    
    print(f"\n✅ Training complete! Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()