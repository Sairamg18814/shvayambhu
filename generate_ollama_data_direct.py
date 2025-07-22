#!/usr/bin/env python3
"""
Generate Ollama Data Directly
=============================

Directly call Ollama API to generate training data.
"""

import requests
import json
from pathlib import Path
from datetime import datetime
import time


def call_ollama(prompt, model="llama3.1:8b"):
    """Call Ollama API directly."""
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 200
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "").strip()
        else:
            print(f"Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None


def generate_prompts():
    """Generate diverse prompts."""
    prompts = [
        # Consciousness & self-awareness
        "What is consciousness and how does it relate to self-awareness?",
        "Explain the concept of subjective experience.",
        "How would you describe self-awareness to someone who has never experienced it?",
        "What does it mean to be conscious of one's own thoughts?",
        "Describe the feeling of being aware.",
        
        # Technical
        "Explain how neural networks learn patterns.",
        "What is machine learning in simple terms?",
        "How does artificial intelligence work?",
        "Describe the process of training a language model.",
        "What are the key components of a transformer architecture?",
        
        # Creative
        "Write a short poem about the nature of existence.",
        "Create a metaphor for learning.",
        "Describe a sunset in poetic language.",
        "Tell a brief story about discovery.",
        "Imagine what colors would taste like.",
        
        # Conversational
        "Hello! How are you today?",
        "What's your favorite thing about language?",
        "Can you help me understand something complex?",
        "Tell me about yourself.",
        "What makes a good conversation?",
        
        # Reasoning
        "If all roses are flowers and some flowers fade quickly, what can we conclude?",
        "Explain the difference between correlation and causation.",
        "How would you solve a complex problem step by step?",
        "What's the logical way to approach decision making?",
        "Describe the scientific method.",
    ]
    
    return prompts


def main():
    """Generate training data from Ollama."""
    print("🚀 Generating Real Ollama Training Data")
    print("=" * 60)
    
    # Check Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("❌ Ollama is not running! Start it with: ollama serve")
            return
    except:
        print("❌ Cannot connect to Ollama! Start it with: ollama serve")
        return
    
    print("✅ Ollama is running")
    
    # Generate data
    prompts = generate_prompts()
    data_samples = []
    
    print(f"\n📝 Generating {len(prompts)} samples...")
    
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {prompt[:50]}...")
        
        response = call_ollama(prompt)
        
        if response:
            sample = {
                "prompt": prompt,
                "response": response,
                "model": "llama3.1:8b",
                "timestamp": datetime.now().isoformat()
            }
            data_samples.append(sample)
            print(f"✅ Got response: {response[:80]}...")
        else:
            print("❌ Failed to get response")
        
        # Small delay to avoid overwhelming Ollama
        time.sleep(1)
    
    # Save data
    if data_samples:
        output_dir = Path("data/training")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"ollama_real_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        with open(output_file, 'w') as f:
            for sample in data_samples:
                f.write(json.dumps(sample) + '\n')
        
        print(f"\n💾 Saved {len(data_samples)} samples to {output_file}")
        
        # Show some statistics
        total_chars = sum(len(s['prompt']) + len(s['response']) for s in data_samples)
        avg_response_len = sum(len(s['response']) for s in data_samples) / len(data_samples)
        
        print(f"\n📊 Statistics:")
        print(f"  Total samples: {len(data_samples)}")
        print(f"  Total characters: {total_chars:,}")
        print(f"  Average response length: {avg_response_len:.0f} chars")
        
        return output_file
    else:
        print("\n❌ No data generated!")
        return None


if __name__ == "__main__":
    main()