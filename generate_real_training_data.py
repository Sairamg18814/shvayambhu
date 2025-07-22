#!/usr/bin/env python3
"""
Generate Real Training Data from Ollama
======================================

Generates actual training data with real responses from Ollama models.
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from training.synthetic.data_generator import OllamaIntegration, SyntheticDataGenerator


async def generate_quality_dataset():
    """Generate high-quality training data."""
    print("🚀 Generating Real Training Data for Shvayambhu")
    print("=" * 60)
    
    async with OllamaIntegration() as ollama:
        generator = SyntheticDataGenerator(ollama)
        
        # Test connection first
        print("\n🔍 Testing Ollama connection...")
        try:
            models = await ollama.list_models()
            print(f"✅ Connected! Available models: {[m['name'] for m in models]}")
        except Exception as e:
            print(f"❌ Failed to connect to Ollama: {e}")
            print("Make sure Ollama is running (ollama serve)")
            return
        
        # Generate dataset
        all_data = []
        
        # Use smaller batches for better quality
        categories = ["consciousness", "reasoning", "technical", "creativity"]
        samples_per_model = 50  # Start small to ensure quality
        
        for model_name in ["llama3.1:8b"]:  # Start with one model
            print(f"\n📝 Generating data from {model_name}...")
            
            try:
                data = await generator.generate_diverse_dataset(
                    model_name=model_name,
                    num_samples=samples_per_model,
                    categories=categories
                )
                
                # Verify we got real data
                if data and len(data) > 0:
                    sample = data[0]
                    if sample['response'] != f"Response about {sample['category']}":
                        print(f"✅ Got real responses! Sample:")
                        print(f"   Q: {sample['prompt'][:60]}...")
                        print(f"   A: {sample['response'][:60]}...")
                        all_data.extend(data)
                    else:
                        print("❌ Still getting placeholder data")
                        
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Save the data
        if all_data:
            output_dir = Path("data/training")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"real_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            
            with open(output_file, 'w') as f:
                for item in all_data:
                    f.write(json.dumps(item) + '\n')
            
            print(f"\n💾 Saved {len(all_data)} samples to {output_file}")
            
            # Show statistics
            print("\n📊 Dataset Statistics:")
            categories_count = {}
            for item in all_data:
                cat = item.get('category', 'unknown')
                categories_count[cat] = categories_count.get(cat, 0) + 1
            
            for cat, count in categories_count.items():
                print(f"  {cat}: {count} samples")
                
            return all_data
        else:
            print("\n❌ No data generated!")
            return None


async def test_single_generation():
    """Test a single generation to debug."""
    print("\n🧪 Testing single generation...")
    
    async with OllamaIntegration() as ollama:
        try:
            response = await ollama.generate(
                model="llama3.1:8b",
                prompt="What is consciousness? Please explain in 2-3 sentences.",
                temperature=0.7
            )
            print(f"✅ Response: {response}")
            
        except Exception as e:
            print(f"❌ Error: {e}")


async def main():
    """Main entry point."""
    # First test single generation
    await test_single_generation()
    
    # Then generate full dataset
    dataset = await generate_quality_dataset()
    
    if dataset:
        print("\n✅ Data generation successful!")
        print("Next step: Train the model with this real data")
    else:
        print("\n❌ Data generation failed!")


if __name__ == "__main__":
    asyncio.run(main())