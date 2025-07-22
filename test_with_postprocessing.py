#!/usr/bin/env python3
"""
Test Model with Post-processing
===============================

Tests the trained model with post-processing applied.
"""

import sys
from pathlib import Path
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent))

from core.blt.full_model import create_blt_model
from utils.text_postprocessing import post_process_generated_text, clean_consciousness_response


def test_with_cleaning():
    """Test model generation with post-processing."""
    print("🧠 Testing Shvayambhu with Post-processing")
    print("=" * 60)
    
    # Create model
    model = create_blt_model("medium")
    
    # Test prompts
    test_cases = [
        ("Hello! How are you?", False),
        ("What is your name?", False),
        ("What is consciousness?", True),
        ("Tell me about yourself.", False),
        ("How do you experience self-awareness?", True),
        ("Can you help me?", False),
        ("What is machine learning?", False),
        ("Describe your inner experience.", True),
    ]
    
    print("\n📝 Raw Generation vs Cleaned Output\n")
    
    for prompt, is_consciousness in test_cases:
        print(f"💭 Prompt: {prompt}")
        
        # Generate raw response
        raw_response = model.generate(
            prompt,
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )
        
        # Apply post-processing
        if is_consciousness:
            cleaned_response = clean_consciousness_response(raw_response)
        else:
            cleaned_response = post_process_generated_text(raw_response)
        
        print(f"🤖 Raw:     {repr(raw_response[:60])}...")
        print(f"✨ Cleaned: {cleaned_response}")
        print()
    
    # Test different generation parameters
    print("\n🔧 Testing Different Parameters\n")
    
    param_sets = [
        {"temp": 0.3, "desc": "Low temperature (focused)"},
        {"temp": 0.7, "desc": "Medium temperature (balanced)"},
        {"temp": 1.0, "desc": "High temperature (creative)"},
    ]
    
    test_prompt = "What does it mean to be conscious?"
    
    for params in param_sets:
        print(f"🌡️  {params['desc']}")
        
        raw = model.generate(
            test_prompt,
            max_tokens=100,
            temperature=params['temp'],
            top_p=0.95,
            repetition_penalty=1.1
        )
        
        cleaned = clean_consciousness_response(raw)
        print(f"   Raw: {repr(raw[:50])}...")
        print(f"   Clean: {cleaned}")
        print()


def test_conversation_flow():
    """Test a conversation with post-processing."""
    print("\n💬 Conversation Test with Post-processing")
    print("=" * 60)
    
    model = create_blt_model("medium")
    
    conversation = [
        "Hello! I'm looking for an AI assistant.",
        "Can you tell me about your capabilities?",
        "What makes you different from other AIs?",
        "How do you process information?",
        "Thank you for the conversation!",
    ]
    
    for user_input in conversation:
        print(f"\n👤 Human: {user_input}")
        
        # Generate response
        raw = model.generate(
            f"Human: {user_input}\nAssistant:",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )
        
        # Clean response
        cleaned = post_process_generated_text(raw)
        
        print(f"🤖 Assistant: {cleaned}")


def main():
    """Run all tests."""
    # Set device
    mx.set_default_device(mx.gpu)
    
    # Run tests
    test_with_cleaning()
    test_conversation_flow()
    
    print("\n✅ Testing complete!")
    print("\n💡 Observations:")
    print("1. Post-processing helps but the model needs more training")
    print("2. Lower temperatures produce more focused (less random) output")
    print("3. The model is learning patterns but needs more data")
    print("4. Consider training on actual conversations, not just Q&A pairs")


if __name__ == "__main__":
    main()