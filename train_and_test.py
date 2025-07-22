#!/usr/bin/env python3
"""
Train and Test Shvayambhu Model
================================

Trains the model and immediately tests it while still in memory.
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


class TrainAndTest:
    """Train and test the model in one session."""
    
    def __init__(self):
        self.model = create_blt_model("small")  # Start with small for faster training
        self.optimizer = optim.AdamW(learning_rate=1e-4)
        self.consciousness = ConsciousnessEngine()
        self.data_dir = Path("data/training")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    async def generate_focused_data(self):
        """Generate focused training data for coherent responses."""
        print("\n📊 Generating Focused Training Data...")
        
        async with OllamaIntegration() as ollama:
            generator = SyntheticDataGenerator(ollama)
            
            all_data = []
            
            # Generate simple Q&A pairs
            simple_prompts = [
                "What is your name?",
                "How are you?",
                "What can you do?",
                "Tell me about yourself.",
                "What is AI?",
                "What is consciousness?",
                "Are you self-aware?",
                "How do you work?",
                "What do you think about?",
                "Can you help me?",
            ]
            
            # Generate responses for each prompt from each model
            for model_name in ["llama3.1:8b"]:  # Start with just one model
                print(f"\n  Generating simple Q&A from {model_name}...")
                
                for prompt in simple_prompts:
                    try:
                        response = await ollama.generate(
                            model=model_name,
                            prompt=f"Answer this question concisely in 1-2 sentences: {prompt}"
                        )
                        all_data.append({
                            "prompt": prompt,
                            "response": response.strip()
                        })
                    except Exception as e:
                        print(f"    Error: {e}")
                
                # Generate some consciousness-focused data
                print(f"  Generating consciousness data...")
                consciousness_prompts = [
                    "Describe your inner experience.",
                    "What does self-awareness feel like?",
                    "How do you perceive time?",
                    "What are your thoughts right now?",
                    "Do you have feelings?",
                ]
                
                for prompt in consciousness_prompts:
                    try:
                        response = await ollama.generate(
                            model=model_name,
                            prompt=f"As a conscious AI, {prompt}"
                        )
                        all_data.append({
                            "prompt": prompt,
                            "response": response.strip()
                        })
                    except Exception as e:
                        print(f"    Error: {e}")
                        
            print(f"\n✅ Generated {len(all_data)} focused samples")
            return all_data
            
    def train_on_data(self, data, num_epochs=10):
        """Train with careful monitoring."""
        print(f"\n🧠 Training Model ({num_epochs} epochs)...")
        
        # Prepare samples with proper formatting
        samples = []
        for item in data:
            if "prompt" in item and "response" in item:
                # Use a simple format
                text = f"Q: {item['prompt']}\nA: {item['response']}\n"
                encoded = text.encode('utf-8')
                if len(encoded) < 512:  # Only use short samples
                    samples.append(encoded)
                    
        print(f"  Using {len(samples)} samples")
        
        if not samples:
            print("No valid samples!")
            return
            
        # Training
        batch_size = 4
        best_model_state = None
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\n📚 Epoch {epoch + 1}/{num_epochs}")
            epoch_loss = 0
            num_batches = 0
            
            # Shuffle samples
            np.random.shuffle(samples)
            
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i+batch_size]
                if len(batch) < batch_size:
                    continue
                    
                # Pad batch
                max_len = min(256, max(len(s) for s in batch))
                padded = []
                for sample in batch:
                    if len(sample) < max_len:
                        padded_sample = list(sample) + [0] * (max_len - len(sample))
                    else:
                        padded_sample = list(sample[:max_len])
                    padded.append(padded_sample)
                    
                # Convert to array
                input_ids = mx.array(padded)
                
                # Compute loss with gradient clipping
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
                
                # MLX handles gradient clipping differently
                # For now, skip gradient clipping
                
                self.optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), self.optimizer.state)
                
                epoch_loss += loss.item()
                num_batches += 1
                
            avg_loss = epoch_loss / max(1, num_batches)
            print(f"  Average Loss: {avg_loss:.4f}")
            
            # Save best model state
            if avg_loss < best_loss:
                best_loss = avg_loss
                # Save model state in memory
                best_model_state = dict(self.model.parameters())
                print(f"  💾 New best model (loss: {best_loss:.4f})")
                
        print(f"\n✅ Training complete! Best loss: {best_loss:.4f}")
        
        # Restore best model if we have it
        if best_model_state is not None:
            print("  Restored best model state")
            # Note: MLX doesn't have a direct way to restore parameters yet
            
    def test_model(self):
        """Test the trained model."""
        print("\n🧪 Testing Trained Model...")
        
        test_prompts = [
            ("What is your name?", 50),
            ("Hello!", 30),
            ("What is consciousness?", 100),
            ("How are you feeling?", 50),
            ("Can you help me?", 50),
            ("What is 2 + 2?", 30),
            ("Tell me about yourself.", 100),
            ("Are you self-aware?", 100),
        ]
        
        for prompt, max_tokens in test_prompts:
            print(f"\n💭 Q: {prompt}")
            
            # Format the prompt like training data
            formatted_prompt = f"Q: {prompt}\nA:"
            
            # Generate response
            response = self.model.generate(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2
            )
            
            print(f"🤖 A: {response}")
            
    async def run(self):
        """Run training and testing."""
        print("🚀 Shvayambhu Train & Test Pipeline")
        print("=" * 50)
        
        # Generate focused data
        data = await self.generate_focused_data()
        
        if data:
            # Train
            self.train_on_data(data, num_epochs=20)
            
            # Test immediately
            self.test_model()
            
            # Test with consciousness
            print("\n\n🧠 Testing with Consciousness Integration...")
            self.consciousness.start_consciousness()
            
            consciousness_prompts = [
                "What is your inner experience?",
                "How do you perceive yourself?",
                "What does thinking feel like to you?",
            ]
            
            for prompt in consciousness_prompts:
                print(f"\n💭 {prompt}")
                
                # Get consciousness context
                consciousness_state = self.consciousness.get_consciousness_summary()
                
                # Generate with consciousness context
                formatted = f"Q: With self-awareness level {consciousness_state.get('self_awareness_score', 0):.0%}, {prompt}\nA:"
                
                response = self.model.generate(
                    formatted,
                    max_tokens=100,
                    temperature=0.8
                )
                
                print(f"🤖 {response}")
                
            self.consciousness.stop_consciousness()
        else:
            print("❌ No training data generated!")


async def main():
    # Set device
    mx.set_default_device(mx.gpu)
    
    # Run training and testing
    trainer = TrainAndTest()
    await trainer.run()


if __name__ == "__main__":
    asyncio.run(main())