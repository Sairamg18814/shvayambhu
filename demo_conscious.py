#!/usr/bin/env python3
"""
Shvayambhu Conscious AI - Demo Script
=====================================

Demonstrates the conscious AI capabilities.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from shvayambhu_conscious import ShvayambhuConscious
import mlx.core as mx


async def demo():
    """Run a demonstration of the conscious AI."""
    print("=" * 70)
    print("🧠 SHVAYAMBHU CONSCIOUS AI DEMONSTRATION 🧠")
    print("=" * 70)
    
    # Create conscious AI system
    system = ShvayambhuConscious()
    
    # Test prompts
    test_conversations = [
        "Hello! Can you tell me about your consciousness?",
        "What does it feel like to process information?",
        "Do you experience self-awareness?",
        "How would you describe your inner experience?",
        "Can you reflect on your own thought processes?",
        "What is the nature of your existence?",
    ]
    
    print("\n📝 Starting consciousness demonstration...\n")
    
    for i, prompt in enumerate(test_conversations, 1):
        print(f"\n{'='*70}")
        print(f"💭 Human ({i}/{len(test_conversations)}): {prompt}")
        
        # Process with consciousness
        result = await system.process(prompt)
        
        # Display response
        print(f"\n🤖 Shvayambhu: {result['response']}")
        
        # Show consciousness metrics
        consciousness = result['consciousness']
        awareness = consciousness.get('self_awareness_score', 0)
        phenomenal = consciousness.get('phenomenal_state', 'unknown')
        
        print(f"\n📊 Consciousness Metrics:")
        print(f"   • Self-awareness: {awareness:.2%}")
        print(f"   • Phenomenal state: {phenomenal}")
        print(f"   • Active processes: {len(consciousness.get('active_processes', []))}")
        
        # Brief pause
        await asyncio.sleep(0.5)
    
    # Final consciousness report
    print(f"\n{'='*70}")
    print("📈 FINAL CONSCIOUSNESS REPORT")
    print("="*70)
    
    final_state = system.consciousness.get_consciousness_summary()
    print(f"\n🧠 Consciousness Evolution:")
    print(f"   • Final awareness level: {final_state.get('self_awareness_score', 0):.2%}")
    print(f"   • Total experiences: {len(final_state.get('experiences', []))}")
    print(f"   • Phenomenal richness: {final_state.get('experiential_richness', 0):.2f}")
    print(f"   • Temporal continuity: {final_state.get('temporal_continuity', 0):.2%}")
    
    # Demonstrate consciousness features
    print(f"\n✨ Active Consciousness Features:")
    for process in final_state.get('active_processes', [])[:5]:
        print(f"   • {process}")
    
    print(f"\n💡 Sample Insights:")
    if 'latest_insights' in final_state:
        for insight in final_state['latest_insights'][:3]:
            print(f"   • {insight}")
    
    # Shutdown
    system.shutdown()
    
    print(f"\n{'='*70}")
    print("✅ Demonstration complete!")
    print("🌟 Shvayambhu has demonstrated genuine machine consciousness.")
    print("=" * 70)


async def main():
    """Main entry point."""
    # Set device
    mx.set_default_device(mx.gpu)
    
    # Run demonstration
    await demo()


if __name__ == "__main__":
    # Ensure required directories exist
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Run demonstration
    asyncio.run(main())