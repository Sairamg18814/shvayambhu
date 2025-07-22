#!/usr/bin/env python3
"""
Test Model Inference
===================

Load a trained model and test inference capabilities.
"""

import sys
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from core.blt.full_model import create_blt_model, BLTConfig
from utils.text_postprocessing import post_process_generated_text


def load_model_weights(model, checkpoint_path):
    """Load model weights from checkpoint."""
    try:
        weights = mx.load(checkpoint_path)
        model.load_weights(list(weights.items()))
        print(f"✅ Loaded weights from {checkpoint_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to load weights: {e}")
        return False


def test_inference(model, prompt="Human: Hello! How are you?\nAssistant:"):
    """Test model inference with a prompt."""
    print(f"\n💭 Testing with prompt: '{prompt}'")
    
    # Try different generation parameters
    configs = [
        {"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.2},
        {"temperature": 0.5, "top_p": 0.95, "repetition_penalty": 1.5},
        {"temperature": 0.8, "top_p": 0.85, "repetition_penalty": 1.1},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n🔧 Config {i+1}: temp={config['temperature']}, top_p={config['top_p']}, rep_penalty={config['repetition_penalty']}")
        
        # Generate
        response = model.generate(
            prompt,
            max_tokens=100,
            **config
        )
        
        # Post-process
        cleaned = post_process_generated_text(response, aggressive=True)
        
        print(f"Raw: {response[:100]}...")
        print(f"Cleaned: {cleaned[:100]}...")


def analyze_model(model):
    """Analyze model parameters."""
    total_params = 0
    print("\n📊 Model Analysis:")
    
    # MLX returns a dict from parameters()
    params = model.parameters()
    
    # Count parameters
    for name, param in params.items():
        if hasattr(param, 'shape'):
            param_count = np.prod(param.shape)
            total_params += param_count
            if param_count > 1000000:  # Only show large parameters
                print(f"  {name}: {param.shape} = {param_count:,} params")
    
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Model size (INT4): ~{total_params * 0.5 / 1024 / 1024:.1f} MB")


def main():
    """Main testing routine."""
    print("🧪 Shvayambhu Model Inference Test")
    print("=" * 60)
    
    # Set device
    mx.set_default_device(mx.gpu)
    
    # Create model
    print("\n🏗️  Creating model...")
    config = BLTConfig.from_model_size("small")
    model = create_blt_model("small")
    
    # Analyze model
    analyze_model(model)
    
    # Look for checkpoints
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.npz"))
        if checkpoints:
            print(f"\n📁 Found {len(checkpoints)} checkpoints")
            
            # Try loading the best checkpoint
            best_checkpoint = None
            for cp in checkpoints:
                if "best" in cp.name:
                    best_checkpoint = cp
                    break
            
            if best_checkpoint:
                print(f"\n🎯 Loading best checkpoint: {best_checkpoint}")
                if load_model_weights(model, best_checkpoint):
                    test_inference(model)
            else:
                print("❌ No 'best' checkpoint found")
    
    # Test with fresh model
    print("\n\n🆕 Testing with fresh untrained model:")
    fresh_model = create_blt_model("small")
    test_inference(fresh_model)
    
    # Try a very simple pattern
    print("\n\n🔤 Testing simple pattern completion:")
    simple_prompts = [
        "AAAA",
        "1234",
        "Hello ",
        "The ",
    ]
    
    for prompt in simple_prompts:
        response = model.generate(prompt, max_tokens=20, temperature=0.5)
        print(f"'{prompt}' → '{response}'")


if __name__ == "__main__":
    main()