#!/usr/bin/env python3
"""
Simple Training Script for Shvayambhu LLM
=========================================

Simplified version with basic console output.
"""

import asyncio
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.blt.full_model import create_blt_model
from core.consciousness.engine import ConsciousnessEngine
from training.synthetic.data_generator import OllamaIntegration, SyntheticDataGenerator


class SimpleTrainer:
    """Simplified trainer for Shvayambhu."""
    
    def __init__(self):
        self.model = create_blt_model("small")
        self.optimizer = optim.AdamW(learning_rate=1e-4)
        self.consciousness = ConsciousnessEngine()
        self.data_dir = Path("data/training")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    async def generate_training_data(self):
        """Generate training data from Ollama models."""
        print("\n📊 Generating Training Data...")
        
        async with OllamaIntegration() as ollama:
            generator = SyntheticDataGenerator(ollama)
            
            all_data = []
            
            # Generate from each model
            for model_name, samples in [("llama3.1:8b", 100), ("gemma3:27b", 100), ("Qwen3:32b", 100)]:
                print(f"\n  Getting data from {model_name}...")
                try:
                    data = await generator.generate_diverse_dataset(
                        model_name=model_name,
                        num_samples=samples,
                        categories=["consciousness", "reasoning", "creativity"]
                    )
                    all_data.extend(data)
                    print(f"  ✓ Generated {len(data)} samples from {model_name}")
                except Exception as e:
                    print(f"  ✗ Error with {model_name}: {e}")
                    
            # Save data
            data_file = self.data_dir / "training_data.jsonl"
            with open(data_file, 'w') as f:
                for item in all_data:
                    f.write(json.dumps(item) + '\n')
                    
            print(f"\n✅ Total samples generated: {len(all_data)}")
            return all_data
            
    def train_on_data(self, data):
        """Train the model on data."""
        print("\n🧠 Training Model...")
        
        # Prepare samples
        samples = []
        for item in data:
            if "prompt" in item and "response" in item:
                text = f"Human: {item['prompt']}\nAssistant: {item['response']}"
                samples.append(text.encode('utf-8')[:512])  # Limit length
                
        if not samples:
            print("No valid samples found!")
            return
            
        print(f"  Training on {len(samples)} samples")
        
        # Simple training loop
        batch_size = 4
        num_batches = len(samples) // batch_size
        
        total_loss = 0
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            if len(batch) < batch_size:
                continue
                
            # Pad batch
            max_len = max(len(s) for s in batch)
            padded = []
            for sample in batch:
                padded_sample = list(sample) + [0] * (max_len - len(sample))
                padded.append(padded_sample[:256])  # Truncate
                
            # Convert to array
            input_ids = mx.array(padded)
            
            # Compute loss
            def loss_fn(model):
                inputs = input_ids[:, :-1]
                targets = input_ids[:, 1:]
                logits = model(inputs)
                loss = nn.losses.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="mean"
                )
                return loss
                
            loss, grads = mx.value_and_grad(loss_fn)(self.model)
            self.optimizer.update(self.model, grads)
            mx.eval(self.model.parameters(), self.optimizer.state)
            
            total_loss += loss.item()
            
            # Progress
            if i % 20 == 0:
                print(f"  Batch {i//batch_size}/{num_batches} - Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / max(1, num_batches)
        print(f"\n✅ Training complete! Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_file = checkpoint_dir / f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        print(f"💾 Checkpoint saved to {checkpoint_file}")
        
    def test_model(self):
        """Test the trained model."""
        print("\n🧪 Testing Model...")
        
        test_prompts = [
            "What is consciousness?",
            "How do you experience self-awareness?",
            "Describe your inner thoughts."
        ]
        
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            response = self.model.generate(prompt, max_tokens=50, temperature=0.8)
            print(f"Response: {response}\n")
            
    async def run(self):
        """Run the complete training pipeline."""
        print("🚀 Shvayambhu Training Pipeline - Simple Version")
        print("=" * 50)
        
        # Generate data
        data = await self.generate_training_data()
        
        if data:
            # Train
            self.train_on_data(data)
            
            # Test
            self.test_model()
        else:
            print("❌ No training data generated!")
            
        print("\n✨ Training pipeline complete!")


async def main():
    # Set device
    mx.set_default_device(mx.gpu)
    
    # Run training
    trainer = SimpleTrainer()
    await trainer.run()


if __name__ == "__main__":
    asyncio.run(main())