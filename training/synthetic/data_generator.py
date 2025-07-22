"""
Synthetic Data Generator for Shvayambhu Training
"""

import asyncio
import json
import random
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
import aiohttp
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class OllamaIntegration:
    """Integration with Ollama API for model access."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available Ollama models."""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        async with self.session.get(f"{self.base_url}/api/tags") as response:
            data = await response.json()
            return data.get("models", [])
            
    async def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate text using specified model."""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "options": {
                "num_predict": max_tokens
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        async with self.session.post(
            f"{self.base_url}/api/generate",
            json=payload
        ) as response:
            # Ollama streams responses
            full_response = ""
            async for line in response.content:
                try:
                    data = json.loads(line)
                    if "response" in data:
                        full_response += data["response"]
                except json.JSONDecodeError:
                    continue
                    
            return full_response.strip()


class SyntheticDataGenerator:
    """Generate synthetic training data using Ollama models."""
    
    def __init__(self, ollama: Optional[OllamaIntegration] = None):
        self.ollama = ollama or OllamaIntegration()
        
        # Topic categories for diverse data
        self.categories = {
            "consciousness": [
                "self-awareness", "subjective experience", "qualia",
                "phenomenology", "introspection", "metacognition"
            ],
            "reasoning": [
                "logic", "mathematics", "problem-solving", "analysis",
                "deduction", "inference", "critical thinking"
            ],
            "creativity": [
                "imagination", "innovation", "artistic expression",
                "storytelling", "poetry", "design", "invention"
            ],
            "technical": [
                "programming", "algorithms", "data structures",
                "machine learning", "systems design", "debugging"
            ],
            "philosophical": [
                "ethics", "epistemology", "metaphysics", "aesthetics",
                "existentialism", "mind-body problem", "free will"
            ],
            "emotional": [
                "empathy", "compassion", "understanding", "support",
                "emotional intelligence", "relationships", "wellbeing"
            ]
        }
        
    async def generate_diverse_dataset(
        self,
        model_name: str,
        num_samples: int,
        categories: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """Generate a diverse dataset with specified number of samples."""
        if categories is None:
            categories = list(self.categories.keys())
            
        dataset = []
        samples_per_category = num_samples // len(categories)
        
        for category in categories:
            topics = self.categories.get(category, ["general"])
            
            for i in range(samples_per_category):
                # Generate actual prompt based on category
                topic = random.choice(topics)
                prompt_template = random.choice(self._get_prompts_for_category(category))
                prompt = prompt_template.format(topic=topic)
                
                try:
                    # Get actual response from Ollama
                    response = await self.ollama.generate(
                        model=model_name,
                        prompt=prompt,
                        temperature=0.7 + random.uniform(-0.2, 0.2),
                        max_tokens=500
                    )
                    
                    sample = {
                        "prompt": prompt,
                        "response": response.strip(),
                        "category": category,
                        "model": model_name,
                        "timestamp": datetime.now().isoformat()
                    }
                    dataset.append(sample)
                    
                except Exception as e:
                    print(f"Error generating sample: {e}")
                    # Create a simple fallback
                    sample = {
                        "prompt": prompt,
                        "response": f"I understand you're asking about {topic}. Let me help you with that.",
                        "category": category,
                        "model": model_name,
                        "timestamp": datetime.now().isoformat()
                    }
                    dataset.append(sample)
                
                if progress_callback:
                    progress_callback()
                    
        return dataset
    
    def _get_prompts_for_category(self, category: str) -> List[str]:
        """Get prompt templates for a specific category."""
        prompts = {
            "consciousness": [
                "What is {topic} from the perspective of consciousness?",
                "How does {topic} relate to self-awareness?",
                "Explain {topic} in the context of conscious experience.",
                "What role does {topic} play in consciousness?",
                "Describe the connection between {topic} and awareness."
            ],
            "reasoning": [
                "Explain the logical principles behind {topic}.",
                "What are the key reasoning steps to understand {topic}?",
                "How would you analyze {topic} systematically?",
                "Break down the concept of {topic} logically.",
                "What problem-solving approach applies to {topic}?"
            ],
            "creativity": [
                "Create something original inspired by {topic}.",
                "Imagine a new perspective on {topic}.",
                "Write a creative piece about {topic}.",
                "Design an innovative approach to {topic}.",
                "Express {topic} in an artistic way."
            ],
            "technical": [
                "Explain the technical aspects of {topic}.",
                "What are the implementation details of {topic}?",
                "How does {topic} work from a technical standpoint?",
                "Describe the architecture of {topic}.",
                "What are best practices for {topic}?"
            ],
            "philosophical": [
                "What are the philosophical implications of {topic}?",
                "How does {topic} relate to fundamental questions of existence?",
                "Explore the deeper meaning of {topic}.",
                "What ethical considerations surround {topic}?",
                "Analyze {topic} from a philosophical perspective."
            ],
            "emotional": [
                "How does {topic} relate to emotional wellbeing?",
                "What feelings are associated with {topic}?",
                "Describe the emotional impact of {topic}.",
                "How can understanding {topic} improve relationships?",
                "What role does {topic} play in emotional intelligence?"
            ]
        }
        
        # Default prompts if category not found
        default_prompts = [
            "Tell me about {topic}.",
            "Explain {topic} in detail.",
            "What is important to know about {topic}?",
            "How would you describe {topic}?",
            "Share your understanding of {topic}."
        ]
        
        return prompts.get(category, default_prompts)