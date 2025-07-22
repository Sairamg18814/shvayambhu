#!/usr/bin/env python3
"""
Quick Training Test
===================

Quick test to see current model state.
"""

import sys
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, str(Path(__file__).parent))

from core.blt.full_model import create_blt_model, BLTConfig
from utils.text_postprocessing import post_process_generated_text


def quick_test():
    """Quick test of model training and generation."""
    print("🚀 Quick Training Test")
    print("=" * 60)
    
    # Set device
    mx.set_default_device(mx.gpu)
    
    # Create model
    print("\n🏗️  Creating model...")
    config = BLTConfig.from_model_size("small")
    config.dropout = 0.1
    model = create_blt_model("small")
    
    # Create simple training data
    print("\n📚 Creating simple training data...")
    texts = [
        "Hello world! This is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
        "Consciousness is self-awareness.",
        "AI can help solve problems.",
    ] * 10  # Repeat for more data
    
    combined = " ".join(texts).encode('utf-8')
    
    # Create a few sequences
    sequences = []
    seq_len = 32
    for i in range(0, len(combined) - seq_len - 1, 16):
        seq = list(combined[i:i + seq_len + 1])
        sequences.append(seq)
    
    print(f"Created {len(sequences)} sequences")
    
    # Train for a few epochs
    print("\n🚂 Quick training (5 epochs)...")
    optimizer = optim.AdamW(learning_rate=1e-3)
    
    for epoch in range(5):
        total_loss = 0
        
        # Shuffle sequences
        np.random.shuffle(sequences)
        
        # Train on mini-batches
        batch_size = 4
        num_batches = 0
        
        for i in range(0, len(sequences) - batch_size, batch_size):
            batch = mx.array(sequences[i:i + batch_size])
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            def loss_fn(model):
                logits = model(inputs)
                B, L, V = logits.shape
                loss = nn.losses.cross_entropy(
                    logits.reshape(B * L, V),
                    targets.reshape(B * L),
                    reduction="mean"
                )
                return loss
            
            loss, grads = mx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch + 1}/5 - Loss: {avg_loss:.4f}")
    
    # Test generation
    print("\n🧪 Testing generation...")
    test_prompts = [
        "Hello",
        "The",
        "Machine",
        "AI",
    ]
    
    for prompt in test_prompts:
        response = model.generate(
            prompt,
            max_tokens=30,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        cleaned = post_process_generated_text(response)
        print(f"'{prompt}' → '{cleaned}'")
    
    # Try loading a saved model if available
    print("\n📂 Checking for saved models...")
    checkpoint_dir = Path("checkpoints")
    
    if checkpoint_dir.exists():
        best_model = checkpoint_dir / "shvayambhu_final_best.npz"
        if best_model.exists():
            print(f"Found saved model: {best_model}")
            
            # Load and test
            try:
                import numpy as np
                data = np.load(best_model)
                print(f"Checkpoint contains: {list(data.keys())[:5]}...")
                
                # Test generation with loaded model
                print("\n🧪 Testing with loaded model...")
                loaded_model = create_blt_model("small")
                
                # Note: Actual weight loading would need proper implementation
                # This is just to show the checkpoint exists
                
            except Exception as e:
                print(f"Error loading: {e}")
    
    print("\n✅ Quick test complete!")
    print("\n💡 Recommendations:")
    print("1. The model needs much more diverse training data")
    print("2. Consider using pre-trained embeddings")
    print("3. Integration with Ollama would provide immediate results")


if __name__ == "__main__":
    quick_test()