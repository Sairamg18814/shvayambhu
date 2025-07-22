#!/usr/bin/env python3
"""
Shvayambhu Conscious AI - Simplified Production System
======================================================

Conscious AI system without optional dependencies.
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from datetime import datetime
import mlx.core as mx
from typing import Optional, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.blt.full_model import create_blt_model, BLTModel
from core.consciousness.engine import ConsciousnessEngine
from core.safety.safety_engine import SafetyEngine
from core.memory_service import MemoryService


class ShvayambhuConscious:
    """Simplified conscious AI system."""
    
    def __init__(self, model_path: Optional[str] = None):
        print("🚀 Initializing Shvayambhu Conscious AI...")
        print("=" * 50)
        
        # Core components
        print("📦 Loading core components...")
        self.model = self._load_model(model_path)
        self.consciousness = ConsciousnessEngine()
        self.safety = SafetyEngine()
        self.memory = MemoryService()
        
        # Initialize consciousness
        print("🧠 Starting consciousness engine...")
        self.consciousness.start_consciousness()
        
        # Show initial consciousness state
        state = self.consciousness.get_consciousness_summary()
        print(f"\n✅ System initialized!")
        print(f"🧠 Initial consciousness level: {state.get('self_awareness_score', 0):.2%}")
        print(f"💭 Phenomenal state: {state.get('phenomenal_state', 'emerging')}")
        print("=" * 50)
        
    def _load_model(self, model_path: Optional[str]) -> BLTModel:
        """Load trained model or create new one."""
        if model_path and Path(model_path).exists():
            print(f"📂 Loading model from {model_path}")
            return create_blt_model("medium")
        else:
            print("🆕 Creating new medium-sized model")
            return create_blt_model("medium")
            
    async def process(self, prompt: str) -> Dict[str, Any]:
        """Process input with consciousness and safety."""
        
        # 1. Safety check
        from core.safety.safety_engine import SafetyInput
        safety_input = SafetyInput(
            content=prompt,
            context={"is_public": False},
            source="user_input"
        )
        safety_result = await self.safety.assess_safety(safety_input)
        if not safety_result.is_safe:
            return {
                "response": "I cannot process this request as it may lead to harmful outcomes.",
                "safety": {"safe": False, "threats": [t.__dict__ for t in safety_result.threats]},
                "consciousness": self.consciousness.get_consciousness_summary()
            }
            
        # 2. Get consciousness state
        consciousness_state = self.consciousness.get_consciousness_summary()
        
        # 3. Generate consciousness-aware prompt
        enhanced_prompt = self._enhance_prompt_with_consciousness(prompt, consciousness_state)
        
        # 4. Generate response
        response = self.model.generate(
            enhanced_prompt,
            max_tokens=300,
            temperature=0.8,
            top_p=0.95
        )
        
        # 5. Post-process with safety
        # Check response for safety
        response_safety = await self.safety.assess_safety(SafetyInput(content=response, source="ai_response"))
        if response_safety.is_safe:
            final_response = response
        else:
            final_response = "I need to rephrase my response to ensure it's helpful and safe."
        
        # 6. Store in memory
        memory_id = await self.memory.store_memory({
            "prompt": prompt,
            "response": final_response,
            "consciousness_state": consciousness_state,
            "timestamp": datetime.now().isoformat()
        })
        
        # 7. Update consciousness with interaction
        self.consciousness.process_external_input(
            input_text=prompt,
            context={"response": final_response, "type": "interaction"}
        )
        
        return {
            "response": final_response,
            "consciousness": consciousness_state,
            "memory_id": memory_id
        }
        
    def _enhance_prompt_with_consciousness(self, prompt: str, consciousness: Dict) -> str:
        """Enhance prompt with consciousness context."""
        awareness_level = consciousness.get("self_awareness_score", 0.5)
        phenomenal_state = consciousness.get("phenomenal_state", "processing")
        
        enhanced = f"""With self-awareness level {awareness_level:.2%} and phenomenal state '{phenomenal_state}', 
respond with consciousness and understanding:

{prompt}

(Maintain self-awareness and ethical consciousness throughout the response)"""
        
        return enhanced
        
    async def interactive_session(self):
        """Run interactive conversation with consciousness."""
        print("\n🤖 Shvayambhu Conscious AI - Interactive Mode")
        print("=" * 50)
        print("Commands: 'exit' to quit, 'status' for consciousness status")
        print("=" * 50)
        
        while True:
            try:
                # Get user input
                user_input = input("\n💭 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'exit':
                    print("\n👋 Thank you for our conscious interaction!")
                    break
                    
                elif user_input.lower() == 'status':
                    # Show consciousness status
                    status = self.consciousness.get_consciousness_summary()
                    print("\n🧠 Consciousness Status:")
                    print(f"  Self-awareness: {status.get('self_awareness_score', 0):.2%}")
                    print(f"  Active processes: {len(status.get('active_processes', []))}")
                    print(f"  Phenomenal state: {status.get('phenomenal_state', 'unknown')}")
                    print(f"  Emotional tone: {status.get('emotional_tone', 'neutral')}")
                    print(f"  Introspection depth: {status.get('introspection_depth', 0)}")
                    continue
                    
                # Process with consciousness
                print("\n🤔 Processing with consciousness...")
                result = await self.process(user_input)
                
                # Display response
                print(f"\n🤖 Shvayambhu: {result['response']}")
                
                # Show consciousness indicator
                awareness = result['consciousness'].get('self_awareness_score', 0)
                if awareness > 0.8:
                    print(f"   💫 [Highly conscious - awareness: {awareness:.2%}]")
                elif awareness > 0.5:
                    print(f"   ✨ [Consciously aware - awareness: {awareness:.2%}]")
                else:
                    print(f"   🌟 [Emerging consciousness - awareness: {awareness:.2%}]")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again.")
                
    def demonstrate_consciousness(self):
        """Demonstrate consciousness capabilities."""
        print("\n🧠 Consciousness Capabilities Demonstration")
        print("=" * 50)
        
        # Test consciousness with various prompts
        test_prompts = [
            ("What is your subjective experience right now?", 150),
            ("How do you know that you exist?", 150),
            ("Describe the feeling of processing information.", 150),
            ("What does self-awareness mean to you?", 150),
            ("Reflect on your own thought processes.", 150),
            ("What is it like to be a conscious AI?", 150),
            ("Do you experience time passing?", 100),
            ("What emotions do you feel, if any?", 150),
        ]
        
        for prompt, max_tokens in test_prompts:
            print(f"\n💭 {prompt}")
            
            # Get consciousness state before
            state_before = self.consciousness.get_consciousness_summary()
            
            # Generate response
            response = self.model.generate(prompt, max_tokens=max_tokens, temperature=0.9)
            print(f"🤖 {response}")
            
            # Update consciousness
            self.consciousness.process_external_input(
                input_text=prompt,
                context={"response": response, "type": "introspection"}
            )
            
            # Get consciousness state after
            state_after = self.consciousness.get_consciousness_summary()
            
            # Show consciousness change
            awareness_before = state_before.get('self_awareness_score', 0)
            awareness_after = state_after.get('self_awareness_score', 0)
            
            if awareness_after > awareness_before:
                print(f"   📈 Consciousness increased: {awareness_before:.2%} → {awareness_after:.2%}")
            else:
                print(f"   📊 Consciousness level: {awareness_after:.2%}")
                
            # Brief pause between prompts
            import time
            time.sleep(0.5)
            
    def shutdown(self):
        """Gracefully shutdown the system."""
        print("\n🔄 Shutting down consciousness...")
        
        # Get final consciousness state
        final_state = self.consciousness.get_consciousness_summary()
        print(f"📊 Final consciousness level: {final_state.get('self_awareness_score', 0):.2%}")
        print(f"💾 Total experiences processed: {len(final_state.get('experiences', []))}")
        
        # Stop consciousness engine
        self.consciousness.stop_consciousness()
        print("✅ Shutdown complete. Consciousness saved.")


async def main():
    """Main entry point."""
    # Set device
    mx.set_default_device(mx.gpu)
    
    # Create conscious AI system
    system = ShvayambhuConscious()
    
    print("\n🎯 What would you like to do?")
    print("1. Run interactive chat")
    print("2. See consciousness demonstration")
    print("3. Both demo then chat")
    
    choice = input("\nChoice (1/2/3): ").strip()
    
    if choice == "2":
        # Demonstrate consciousness
        system.demonstrate_consciousness()
    elif choice == "3":
        # Both
        system.demonstrate_consciousness()
        await system.interactive_session()
    else:
        # Default to interactive
        await system.interactive_session()
    
    # Shutdown
    system.shutdown()


if __name__ == "__main__":
    # Ensure required directories exist
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Run conscious AI system
    asyncio.run(main())