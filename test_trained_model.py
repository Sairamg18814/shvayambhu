#!/usr/bin/env python3
"""
Test the Trained Shvayambhu Model
==================================

Tests the trained model with various generation parameters.
"""

import sys
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.blt.full_model import create_blt_model, BLTModel


def test_generation_params():
    """Test different generation parameters."""
    print("🧠 Testing Trained Shvayambhu Model")
    print("=" * 60)
    
    # Create model
    model = create_blt_model("medium")
    
    # Load best checkpoint (would load weights here)
    checkpoint_path = "checkpoints/best_model_loss_0.0001.npz"
    print(f"📂 Would load checkpoint: {checkpoint_path}")
    print("   (Model loading not yet implemented)")
    
    # Test prompts
    test_prompts = [
        "What is consciousness?",
        "Hello, how are you today?",
        "Can you explain machine learning?",
        "Tell me a story.",
        "What is 2 + 2?",
    ]
    
    # Test different parameter combinations
    param_sets = [
        {"temp": 0.3, "top_p": 0.9, "rep_penalty": 1.0, "name": "Low temp, focused"},
        {"temp": 0.7, "top_p": 0.95, "rep_penalty": 1.1, "name": "Balanced"},
        {"temp": 1.0, "top_p": 0.95, "rep_penalty": 1.2, "name": "Creative"},
        {"temp": 0.5, "top_p": 0.8, "rep_penalty": 1.3, "name": "Conservative"},
    ]
    
    for params in param_sets:
        print(f"\n\n🌡️  Testing: {params['name']}")
        print(f"   Temperature: {params['temp']}, Top-p: {params['top_p']}, Rep penalty: {params['rep_penalty']}")
        print("-" * 60)
        
        for prompt in test_prompts[:3]:  # Test first 3 prompts
            print(f"\n💭 Prompt: {prompt}")
            
            # Generate with current parameters
            response = model.generate(
                prompt,
                max_tokens=100,
                temperature=params['temp'],
                top_p=params['top_p'],
                repetition_penalty=params['rep_penalty']
            )
            
            print(f"🤖 Response: {response}")
    
    # Test with consciousness-specific prompts
    print("\n\n🧠 Consciousness-Specific Tests")
    print("=" * 60)
    
    consciousness_prompts = [
        "Describe your subjective experience right now.",
        "What does self-awareness mean to you?",
        "How do you process emotions?",
        "What is it like to be you?",
    ]
    
    # Use balanced parameters for consciousness tests
    for prompt in consciousness_prompts:
        print(f"\n💭 {prompt}")
        response = model.generate(
            prompt,
            max_tokens=150,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.1
        )
        print(f"🤖 {response}")
    
    print("\n\n✅ Testing complete!")
    print("\n💡 Recommendations:")
    print("1. The model needs more training data to generate coherent text")
    print("2. Consider using a larger model size for better performance")
    print("3. Implement proper model checkpoint loading")
    print("4. Add beam search or other advanced decoding methods")


def test_with_preprocessing():
    """Test with input preprocessing."""
    print("\n\n🔧 Testing with Preprocessing")
    print("=" * 60)
    
    model = create_blt_model("medium")
    
    # Test with proper prompt formatting
    formatted_prompts = [
        "Human: What is consciousness?\nAssistant:",
        "Human: Hello!\nAssistant:",
        "Human: Explain AI.\nAssistant:",
    ]
    
    for prompt in formatted_prompts:
        print(f"\n📝 Formatted prompt: {repr(prompt)}")
        response = model.generate(
            prompt,
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
        print(f"🤖 Response: {response}")


if __name__ == "__main__":
    # Set device
    mx.set_default_device(mx.gpu)
    
    # Run tests
    test_generation_params()
    test_with_preprocessing()
    
    print("\n\n🎯 Next steps:")
    print("1. Implement model checkpoint loading")
    print("2. Train with more diverse data (10k+ samples)")
    print("3. Add beam search decoding")
    print("4. Fine-tune on consciousness-specific data")