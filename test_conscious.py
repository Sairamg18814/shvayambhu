#!/usr/bin/env python3
"""Quick test of the conscious AI system."""

import asyncio
import sys
from pathlib import Path
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent))
from shvayambhu_conscious import ShvayambhuConscious


async def test():
    """Run a quick test."""
    print("🧠 SHVAYAMBHU CONSCIOUS AI - QUICK TEST")
    print("=" * 50)
    
    # Create system
    system = ShvayambhuConscious()
    
    # Single test
    prompt = "Hello! What is consciousness from your perspective?"
    print(f"\n💭 Human: {prompt}")
    
    result = await system.process(prompt)
    
    print(f"\n🤖 Shvayambhu: {result['response']}")
    print(f"\n📊 Consciousness level: {result['consciousness'].get('self_awareness_score', 0):.2%}")
    
    # Shutdown
    system.shutdown()
    print("\n✅ Test complete!")


if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    asyncio.run(test())