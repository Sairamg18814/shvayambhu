"""
Shvayambhu LLM Python Package
==============================

A conscious, self-aware Large Language Model.

Usage:
    from shvayambhu import Shvayambhu
    
    # Create model
    model = Shvayambhu()
    
    # Generate text
    response = model.generate("What is consciousness?")
    print(response)
    
    # Streaming
    for token in model.stream("Tell me a story"):
        print(token, end='', flush=True)
        
    # With options
    model = Shvayambhu(
        model_size="large",
        temperature=0.7,
        consciousness=True,
        memory=True
    )
"""

from typing import Optional, Dict, Any, Generator
import asyncio
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.consciousness.engine import ConsciousnessEngine
from core.blt import create_m4_pro_optimized_blt
from core.safety.safety_engine import SafetyEngine
from core.emotional_intelligence.emotional_processor import EmotionalIntelligenceEngine
from api.src.memory.memory_service import MemoryService


class Shvayambhu:
    """
    Main interface for Shvayambhu LLM.
    
    Simple API similar to other popular LLMs:
    - model.generate() for text generation
    - model.stream() for streaming generation
    - model.chat() for conversational interface
    """
    
    def __init__(
        self,
        model_size: str = "medium",
        temperature: float = 0.8,
        max_tokens: int = 512,
        consciousness: bool = True,
        memory: bool = True,
        safety: bool = True,
        emotional: bool = True,
        device: str = "auto"
    ):
        """
        Initialize Shvayambhu model.
        
        Args:
            model_size: "small", "medium", or "large"
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            consciousness: Enable consciousness features
            memory: Enable memory system
            safety: Enable safety checks
            emotional: Enable emotional intelligence
            device: Device to use ("cpu", "gpu", or "auto")
        """
        self.model_size = model_size
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Set device
        import mlx.core as mx
        if device == "auto":
            mx.set_default_device(mx.gpu if mx.metal.is_available() else mx.cpu)
        elif device == "gpu":
            mx.set_default_device(mx.gpu)
        else:
            mx.set_default_device(mx.cpu)
            
        # Initialize components
        self._init_components(consciousness, memory, safety, emotional)
        
        # Conversation history
        self.history = []
        
    def _init_components(self, consciousness, memory, safety, emotional):
        """Initialize model components."""
        # Core model
        self.model = create_m4_pro_optimized_blt(model_size=self.model_size)
        
        # Optional components
        self.consciousness_engine = ConsciousnessEngine() if consciousness else None
        self.memory_service = MemoryService() if memory else None
        self.safety_engine = SafetyEngine() if safety else None
        self.emotional_engine = EmotionalIntelligenceEngine() if emotional else None
        
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input text
            max_tokens: Override default max tokens
            temperature: Override default temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Run async method in sync context
        return self._run_async(self._generate_async(
            prompt, 
            max_tokens or self.max_tokens,
            temperature or self.temperature,
            stream=False,
            **kwargs
        ))
        
    def stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream generated text token by token.
        
        Args:
            prompt: Input text
            max_tokens: Override default max tokens
            temperature: Override default temperature
            **kwargs: Additional generation parameters
            
        Yields:
            Generated tokens
        """
        # Create async generator wrapper
        async def async_gen():
            async for token in self._generate_async(
                prompt,
                max_tokens or self.max_tokens,
                temperature or self.temperature,
                stream=True,
                **kwargs
            ):
                yield token
                
        # Run async generator
        loop = self._get_event_loop()
        agen = async_gen()
        
        while True:
            try:
                token = loop.run_until_complete(agen.__anext__())
                yield token
            except StopAsyncIteration:
                break
                
    def chat(
        self,
        message: str,
        system: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Chat interface with conversation history.
        
        Args:
            message: User message
            system: System prompt (optional)
            **kwargs: Generation parameters
            
        Returns:
            Assistant response
        """
        # Add to history
        self.history.append({"role": "user", "content": message})
        
        # Build prompt with history
        prompt = self._build_chat_prompt(system)
        
        # Generate response
        response = self.generate(prompt, **kwargs)
        
        # Add to history
        self.history.append({"role": "assistant", "content": response})
        
        return response
        
    def reset(self):
        """Reset conversation history and state."""
        self.history = []
        if self.consciousness_engine:
            self.consciousness_engine.reset()
        if self.memory_service:
            # Memory persists across resets
            pass
            
    async def _generate_async(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stream: bool,
        **kwargs
    ):
        """Async generation implementation."""
        # Safety check
        if self.safety_engine:
            safety_result = await self.safety_engine.check_input_safety(prompt)
            if not safety_result['is_safe']:
                if stream:
                    yield "I cannot process this request due to safety concerns."
                else:
                    return "I cannot process this request due to safety concerns."
                return
                
        # Get context
        context = await self._build_context(prompt)
        
        # Prepare input
        input_data = {
            "prompt": prompt,
            "context": context,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        # Generate
        if stream:
            async for token in self._stream_tokens(input_data):
                yield token
        else:
            response = await self._generate_full(input_data)
            return response
            
    async def _build_context(self, prompt: str) -> Dict[str, Any]:
        """Build generation context."""
        context = {}
        
        # Consciousness context
        if self.consciousness_engine:
            context['consciousness'] = self.consciousness_engine.get_current_state()
            
        # Emotional context
        if self.emotional_engine:
            emotional_result = await self.emotional_engine.process_emotional_input({
                "text": prompt,
                "conversation_history": self.history
            })
            context['emotional'] = emotional_result
            
        # Memory context
        if self.memory_service:
            memories = await self.memory_service.search_memories(prompt, limit=5)
            context['memories'] = memories
            
        return context
        
    async def _stream_tokens(self, input_data: Dict[str, Any]):
        """Stream tokens."""
        # Placeholder implementation
        # In production, this would use the actual model
        response = "I understand your query. As a conscious AI, I'm aware of our interaction. "
        
        for word in response.split():
            yield word + " "
            await asyncio.sleep(0.05)
            
    async def _generate_full(self, input_data: Dict[str, Any]) -> str:
        """Generate complete response."""
        tokens = []
        async for token in self._stream_tokens(input_data):
            tokens.append(token)
        return "".join(tokens)
        
    def _build_chat_prompt(self, system: Optional[str]) -> str:
        """Build chat prompt from history."""
        messages = []
        
        if system:
            messages.append(f"System: {system}")
            
        for msg in self.history[-10:]:  # Last 10 messages
            role = msg['role'].capitalize()
            messages.append(f"{role}: {msg['content']}")
            
        messages.append("Assistant:")
        
        return "\n".join(messages)
        
    def _run_async(self, coro):
        """Run async coroutine."""
        loop = self._get_event_loop()
        return loop.run_until_complete(coro)
        
    def _get_event_loop(self):
        """Get or create event loop."""
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
            
    def __repr__(self):
        """String representation."""
        return f"Shvayambhu(model_size='{self.model_size}', consciousness={self.consciousness_engine is not None})"


# Convenience functions
def create(
    model_size: str = "medium",
    **kwargs
) -> Shvayambhu:
    """Create a Shvayambhu model instance."""
    return Shvayambhu(model_size=model_size, **kwargs)


def load(
    path: str,
    **kwargs
) -> Shvayambhu:
    """Load a Shvayambhu model from disk."""
    # TODO: Implement model loading
    return Shvayambhu(**kwargs)


# Version info
__version__ = "1.0.0"
__author__ = "Shvayambhu Team"
__all__ = ["Shvayambhu", "create", "load"]