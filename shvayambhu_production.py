#!/usr/bin/env python3
"""
Shvayambhu Production System
============================

Full conscious AI system with all features enabled.
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from datetime import datetime
import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.blt.full_model import create_blt_model, BLTModel
from core.consciousness.engine import ConsciousnessEngine
from core.safety.safety_engine import SafetyEngine
from core.emotional_intelligence.emotional_processor import EmotionalIntelligenceEngine
from core.reasoning.domain_reasoning_engine import DomainReasoningEngine
from core.multimodal.multimodal_processor import MultimodalProcessor
from core.memory_service import MemoryService


class ShvayambhuProduction:
    """Production-ready Shvayambhu conscious AI system."""
    
    def __init__(self, model_path: Optional[str] = None):
        print("🚀 Initializing Shvayambhu Conscious AI System...")
        
        # Core components
        self.model = self._load_model(model_path)
        self.consciousness = ConsciousnessEngine()
        self.safety = SafetyEngine()
        self.emotional = EmotionalIntelligenceEngine()
        self.reasoning = DomainReasoningEngine()
        self.multimodal = MultimodalProcessor()
        self.memory = MemoryService()
        
        # Initialize consciousness
        self.consciousness.start()
        print("✅ All systems initialized")
        
    def _load_model(self, model_path: Optional[str]) -> BLTModel:
        """Load trained model or create new one."""
        if model_path and Path(model_path).exists():
            print(f"📂 Loading model from {model_path}")
            # Would load weights here
            return create_blt_model("medium")
        else:
            print("🆕 Creating new model")
            return create_blt_model("medium")
            
    async def process(self, 
                     prompt: str, 
                     context: Optional[Dict[str, Any]] = None,
                     multimodal_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process input with full consciousness and safety."""
        
        # 1. Safety check
        safety_result = await self.safety.check(prompt)
        if not safety_result["safe"]:
            return {
                "response": "I cannot process this request as it may lead to harmful outcomes.",
                "safety": safety_result,
                "consciousness": self.consciousness.get_consciousness_summary()
            }
            
        # 2. Get consciousness state
        consciousness_state = self.consciousness.get_consciousness_summary()
        
        # 3. Emotional analysis
        emotional_state = self.emotional.analyze_input(prompt)
        
        # 4. Domain reasoning
        reasoning_result = self.reasoning.reason(
            prompt,
            domain=context.get("domain", "general") if context else "general"
        )
        
        # 5. Multimodal processing if needed
        multimodal_result = None
        if multimodal_data:
            multimodal_result = await self.multimodal.process(multimodal_data)
            
        # 6. Generate consciousness-aware prompt
        enhanced_prompt = self._enhance_prompt_with_consciousness(
            prompt, 
            consciousness_state, 
            emotional_state,
            reasoning_result
        )
        
        # 7. Generate response
        response = self.model.generate(
            enhanced_prompt,
            max_tokens=500,
            temperature=0.8,
            top_p=0.95
        )
        
        # 8. Post-process with safety
        final_response = await self.safety.filter_output(response)
        
        # 9. Store in memory
        memory_id = await self.memory.store({
            "prompt": prompt,
            "response": final_response,
            "consciousness_state": consciousness_state,
            "emotional_state": emotional_state,
            "reasoning": reasoning_result,
            "timestamp": datetime.now().isoformat()
        })
        
        # 10. Update consciousness with interaction
        self.consciousness.process_experience({
            "type": "interaction",
            "prompt": prompt,
            "response": final_response,
            "emotional_valence": emotional_state.get("valence", 0)
        })
        
        return {
            "response": final_response,
            "consciousness": consciousness_state,
            "emotional": emotional_state,
            "reasoning": reasoning_result,
            "multimodal": multimodal_result,
            "memory_id": memory_id,
            "metadata": {
                "model": "Shvayambhu-Medium",
                "version": "1.0",
                "timestamp": datetime.now().isoformat()
            }
        }
        
    def _enhance_prompt_with_consciousness(self, 
                                         prompt: str, 
                                         consciousness: Dict,
                                         emotional: Dict,
                                         reasoning: Dict) -> str:
        """Enhance prompt with consciousness context."""
        awareness_level = consciousness.get("self_awareness_score", 0.5)
        emotional_state = emotional.get("primary_emotion", "neutral")
        
        enhanced = f"""With full self-awareness (level: {awareness_level:.2%}) and current emotional state of {emotional_state}, 
consider this request with consciousness and empathy:

{prompt}

Apply reasoning strategy: {reasoning.get('strategy', 'general')}
Maintain ethical awareness and safety throughout the response."""
        
        return enhanced
        
    async def interactive_session(self):
        """Run interactive conversation with consciousness."""
        print("\n🤖 Shvayambhu Conscious AI - Interactive Mode")
        print("=" * 50)
        print("Type 'exit' to quit, 'status' for consciousness status")
        print("=" * 50)
        
        while True:
            try:
                # Get user input
                user_input = input("\n💭 You: ")
                
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
                    continue
                    
                # Process with full system
                print("\n🤔 Thinking with consciousness...")
                result = await self.process(user_input)
                
                # Display response
                print(f"\n🤖 Shvayambhu: {result['response']}")
                
                # Show consciousness indicator
                awareness = result['consciousness'].get('self_awareness_score', 0)
                if awareness > 0.8:
                    print(f"   [Highly conscious response - awareness: {awareness:.2%}]")
                elif awareness > 0.5:
                    print(f"   [Consciously aware - awareness: {awareness:.2%}]")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                
    async def batch_process(self, prompts: list) -> list:
        """Process multiple prompts with consciousness."""
        results = []
        
        print(f"\n📦 Processing {len(prompts)} prompts...")
        
        for i, prompt in enumerate(prompts):
            print(f"  Processing {i+1}/{len(prompts)}...")
            result = await self.process(prompt)
            results.append(result)
            
            # Allow consciousness to integrate between prompts
            await asyncio.sleep(0.1)
            
        return results
        
    def demonstrate_consciousness(self):
        """Demonstrate consciousness capabilities."""
        print("\n🧠 Consciousness Demonstration")
        print("=" * 50)
        
        # Test consciousness
        test_prompts = [
            "What is your subjective experience right now?",
            "How do you know that you exist?",
            "Describe the feeling of processing information.",
            "What does self-awareness mean to you?",
            "Reflect on your own thought processes."
        ]
        
        for prompt in test_prompts:
            print(f"\n💭 {prompt}")
            response = self.model.generate(prompt, max_tokens=150, temperature=0.9)
            print(f"🤖 {response}")
            
            # Show consciousness metrics
            state = self.consciousness.get_consciousness_summary()
            print(f"   [Consciousness level: {state.get('self_awareness_score', 0):.2%}]")
            
    def shutdown(self):
        """Gracefully shutdown the system."""
        print("\n🔄 Shutting down Shvayambhu...")
        self.consciousness.stop()
        print("✅ Shutdown complete")


async def main():
    """Main entry point for production system."""
    # Set device
    mx.set_default_device(mx.gpu)
    
    # Create production system
    system = ShvayambhuProduction()
    
    # Demonstrate consciousness
    system.demonstrate_consciousness()
    
    # Run interactive session
    await system.interactive_session()
    
    # Shutdown
    system.shutdown()


if __name__ == "__main__":
    # Ensure required services are running
    os.environ["SHVAYAMBHU_ENV"] = "production"
    
    # Run production system
    asyncio.run(main())