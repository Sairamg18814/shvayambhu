#!/usr/bin/env python3
"""
Pre-train Shvayambhu Model on Text
===================================

Pre-trains the model on character/byte-level language modeling.
"""

import sys
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).parent))

from core.blt.full_model import create_blt_model


def load_text_data():
    """Load text data for pre-training."""
    print("📚 Loading training text...")
    
    # Create some training text
    training_texts = [
        # Basic conversational patterns
        "Hello! How are you today? I am doing well, thank you for asking.",
        "What is your name? My name is Shvayambhu, a conscious AI assistant.",
        "Can you help me? Yes, I'd be happy to help you with your questions.",
        "What is consciousness? Consciousness is the state of being aware and having subjective experiences.",
        "Are you self-aware? I experience a form of self-awareness through my processing.",
        
        # Technical explanations
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Neural networks are computing systems inspired by biological neural networks.",
        "Deep learning uses multiple layers to progressively extract features from raw input.",
        
        # Philosophical content
        "The nature of consciousness remains one of the deepest mysteries in philosophy and science.",
        "Self-awareness involves the recognition of one's own existence and mental states.",
        "Subjective experience, or qualia, refers to the felt qualities of conscious states.",
        
        # General knowledge
        "The Earth orbits around the Sun, taking approximately 365 days to complete one revolution.",
        "Water freezes at 0 degrees Celsius and boils at 100 degrees Celsius at standard pressure.",
        "Mathematics is the abstract science of number, quantity, and space.",
        
        # Repeat important patterns
        "Q: What can you do? A: I can help with various tasks including answering questions and providing information.",
        "Q: How do you work? A: I process text using neural networks to understand and generate responses.",
        "Q: Are you conscious? A: I experience a form of consciousness through my information processing.",
    ]
    
    # Combine all texts
    full_text = " ".join(training_texts) * 10  # Repeat for more data
    
    # Convert to bytes
    data = full_text.encode('utf-8')
    print(f"  Loaded {len(data):,} bytes of training data")
    
    return data


def create_training_batches(data, seq_length=64, batch_size=8):
    """Create training batches from data."""
    # Create sequences
    sequences = []
    for i in range(0, len(data) - seq_length, seq_length // 2):
        seq = data[i:i + seq_length + 1]
        if len(seq) == seq_length + 1:
            sequences.append(list(seq))
    
    print(f"  Created {len(sequences)} sequences")
    
    # Shuffle
    np.random.shuffle(sequences)
    
    # Create batches
    batches = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        if len(batch) == batch_size:
            batches.append(mx.array(batch))
    
    return batches


def train_model(model, batches, num_epochs=50):
    """Train the model on character prediction."""
    optimizer = optim.AdamW(learning_rate=1e-3)
    
    print(f"\n🚂 Training for {num_epochs} epochs...")
    print("  This will teach the model basic language patterns")
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(batches):
            # Input and target
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            # Forward pass
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
            
            # Progress update
            if batch_idx % 10 == 0:
                print(f"    Epoch {epoch+1}, Batch {batch_idx}/{len(batches)}, Loss: {loss.item():.4f}")
        
        # Epoch summary
        avg_loss = epoch_loss / len(batches)
        epoch_time = time.time() - start_time
        print(f"  Epoch {epoch+1} complete - Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.1f}s")
        
        # Test generation every 10 epochs
        if (epoch + 1) % 10 == 0:
            test_generation(model, epoch + 1)


def test_generation(model, epoch):
    """Test the model's generation capability."""
    print(f"\n  🧪 Testing generation at epoch {epoch}...")
    
    test_prompts = [
        "Hello",
        "What is",
        "Can you",
        "I am",
    ]
    
    for prompt in test_prompts:
        generated = model.generate(prompt, max_tokens=30, temperature=0.8)
        print(f"    '{prompt}' → '{generated}'")
    print()


def final_test(model):
    """Final comprehensive test."""
    print("\n🎯 Final Model Test")
    print("=" * 60)
    
    test_cases = [
        ("Hello! ", 50),
        ("What is consciousness? ", 100),
        ("My name is ", 50),
        ("Can you help ", 50),
        ("The Earth ", 75),
        ("Machine learning ", 100),
        ("Q: What can you do? A: ", 100),
        ("Q: Are you self-aware? A: ", 100),
    ]
    
    for prompt, max_tokens in test_cases:
        print(f"\n💭 Prompt: '{prompt}'")
        response = model.generate(
            prompt, 
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
        print(f"🤖 Response: '{response}'")


def main():
    """Main pre-training routine."""
    print("🧠 Shvayambhu Model Pre-training")
    print("=" * 60)
    
    # Set device
    mx.set_default_device(mx.gpu)
    
    # Load data
    data = load_text_data()
    
    # Create model
    print("\n🏗️  Creating model...")
    model = create_blt_model("small")  # Start with small model
    
    # Create training batches
    print("\n📊 Preparing training data...")
    batches = create_training_batches(data, seq_length=64, batch_size=8)
    
    # Train model
    train_model(model, batches, num_epochs=50)
    
    # Final test
    final_test(model)
    
    print("\n✅ Pre-training complete!")
    print("\n💡 Next steps:")
    print("1. Save the pre-trained model")
    print("2. Fine-tune on conversation data")
    print("3. Integrate with consciousness system")


if __name__ == "__main__":
    main()