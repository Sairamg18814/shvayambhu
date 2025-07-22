#!/usr/bin/env python3
"""
Quick test to generate real training data from Ollama
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from training.synthetic.data_generator import OllamaIntegration, SyntheticDataGenerator


async def generate_test_data():
    """Generate a small test dataset."""
    print("🚀 Generating Test Training Data")
    print("=" * 60)
    
    async with OllamaIntegration() as ollama:
        generator = SyntheticDataGenerator(ollama)
        
        # Generate just a few samples to test
        print("\n📝 Generating 10 test samples from llama3.1:8b...")
        
        data = await generator.generate_diverse_dataset(
            model_name="llama3.1:8b",
            num_samples=10,
            categories=["consciousness", "reasoning"]
        )
        
        # Check if we got real data
        if data and len(data) > 0:
            print(f"\n✅ Generated {len(data)} samples")
            
            # Show first sample
            sample = data[0]
            print("\n📋 First sample:")
            print(f"Category: {sample['category']}")
            print(f"Prompt: {sample['prompt']}")
            print(f"Response preview: {sample['response'][:200]}...")
            
            # Save the test data
            output_dir = Path("data/training")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"test_real_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            
            with open(output_file, 'w') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
            
            print(f"\n💾 Saved to {output_file}")
            
            return data
        else:
            print("\n❌ No data generated!")
            return None


if __name__ == "__main__":
    asyncio.run(generate_test_data())