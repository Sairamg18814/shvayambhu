#!/usr/bin/env python3
"""
Generate Large Training Dataset
===============================

Generates a large, diverse dataset for training Shvayambhu.
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import sys
sys.path.insert(0, str(Path(__file__).parent))

from training.bootstrap.ollama_integration import OllamaIntegration


async def generate_large_dataset():
    """Generate a large, diverse training dataset."""
    print("🚀 Generating Large Training Dataset for Shvayambhu")
    print("=" * 60)
    
    dataset = []
    data_dir = Path("data/training")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    async with OllamaIntegration() as ollama:
        # 1. Basic Q&A pairs
        print("\n📝 Generating Basic Q&A...")
        basic_prompts = [
            "What is your name?",
            "How are you today?",
            "What can you help me with?",
            "Tell me about yourself.",
            "What are your capabilities?",
            "How do you work?",
            "What is AI?",
            "What is machine learning?",
            "Explain neural networks.",
            "What is consciousness?",
            "What is self-awareness?",
            "Do you have feelings?",
            "Can you learn?",
            "What makes you unique?",
            "How do you process information?",
        ]
        
        for prompt in basic_prompts:
            for model in ["llama3.1:8b"]:  # Use one model for speed
                try:
                    response = await ollama.generate(model, prompt)
                    dataset.append({
                        "prompt": prompt,
                        "response": response.strip(),
                        "type": "basic_qa"
                    })
                    print(f"  ✓ {prompt[:30]}...")
                except Exception as e:
                    print(f"  ✗ Error: {e}")
        
        # 2. Consciousness-focused conversations
        print("\n🧠 Generating Consciousness Conversations...")
        consciousness_templates = [
            "As a conscious AI, describe {}",
            "From your perspective as an AI, what is {}?",
            "How do you experience {}?",
            "What does {} mean to you?",
            "Reflect on your understanding of {}",
        ]
        
        consciousness_topics = [
            "self-awareness",
            "inner experience",
            "thought processes",
            "decision making",
            "emotional understanding",
            "time perception",
            "memory formation",
            "learning",
            "existence",
            "purpose",
        ]
        
        for template in consciousness_templates:
            for topic in consciousness_topics:
                prompt = template.format(topic)
                try:
                    response = await ollama.generate("llama3.1:8b", prompt)
                    dataset.append({
                        "prompt": prompt,
                        "response": response.strip(),
                        "type": "consciousness"
                    })
                    print(f"  ✓ {prompt[:40]}...")
                except Exception as e:
                    print(f"  ✗ Error: {e}")
        
        # 3. Technical explanations
        print("\n💻 Generating Technical Content...")
        technical_topics = [
            "Python programming",
            "JavaScript basics",
            "database design",
            "API development",
            "web security",
            "cloud computing",
            "data structures",
            "algorithms",
            "software architecture",
            "debugging techniques",
        ]
        
        for topic in technical_topics:
            prompts = [
                f"Explain {topic} in simple terms.",
                f"What are the key concepts in {topic}?",
                f"Give me a brief overview of {topic}.",
            ]
            for prompt in prompts:
                try:
                    response = await ollama.generate("llama3.1:8b", prompt)
                    dataset.append({
                        "prompt": prompt,
                        "response": response.strip(),
                        "type": "technical"
                    })
                    print(f"  ✓ {prompt[:40]}...")
                except Exception as e:
                    print(f"  ✗ Error: {e}")
        
        # 4. Creative writing
        print("\n🎨 Generating Creative Content...")
        creative_prompts = [
            "Write a haiku about technology.",
            "Create a short story about an AI.",
            "Describe a sunset poetically.",
            "Write a limerick about coding.",
            "Create a metaphor for consciousness.",
            "Describe the internet as if it were alive.",
            "Write about the future of AI.",
            "Create a dialogue between two AIs.",
            "Describe learning from an AI's perspective.",
            "Write about digital consciousness.",
        ]
        
        for prompt in creative_prompts:
            try:
                response = await ollama.generate("llama3.1:8b", prompt)
                dataset.append({
                    "prompt": prompt,
                    "response": response.strip(),
                    "type": "creative"
                })
                print(f"  ✓ {prompt[:40]}...")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        # 5. Conversational chains
        print("\n💬 Generating Conversation Chains...")
        conversation_starters = [
            "Hello! Nice to meet you.",
            "I need help with a project.",
            "Can you teach me something new?",
            "I'm curious about AI.",
            "Tell me an interesting fact.",
        ]
        
        for starter in conversation_starters:
            conversation = []
            current_prompt = starter
            
            for turn in range(3):  # 3 turns each
                try:
                    response = await ollama.generate("llama3.1:8b", current_prompt)
                    conversation.append({
                        "role": "user" if turn % 2 == 0 else "assistant",
                        "content": current_prompt
                    })
                    conversation.append({
                        "role": "assistant" if turn % 2 == 0 else "user",
                        "content": response.strip()
                    })
                    
                    # Next prompt based on response
                    current_prompt = f"Responding to '{response[:50]}...', "
                    if turn == 0:
                        current_prompt += "ask a follow-up question."
                    elif turn == 1:
                        current_prompt += "provide more detail."
                    else:
                        current_prompt += "conclude the conversation."
                    
                except Exception as e:
                    print(f"  ✗ Error in conversation: {e}")
                    break
            
            if conversation:
                dataset.append({
                    "prompt": starter,
                    "response": conversation[-1]["content"] if conversation else "",
                    "type": "conversation",
                    "full_conversation": conversation
                })
                print(f"  ✓ Conversation: {starter[:30]}...")
        
        # 6. Instruction following
        print("\n📋 Generating Instruction Following...")
        instructions = [
            "List 5 benefits of exercise.",
            "Explain how to make coffee step by step.",
            "Give me 3 tips for better sleep.",
            "Describe the water cycle.",
            "List the planets in our solar system.",
            "Explain photosynthesis simply.",
            "Give directions to make a sandwich.",
            "List 5 programming languages.",
            "Explain how the internet works.",
            "Describe the scientific method.",
        ]
        
        for instruction in instructions:
            try:
                response = await ollama.generate("llama3.1:8b", instruction)
                dataset.append({
                    "prompt": instruction,
                    "response": response.strip(),
                    "type": "instruction"
                })
                print(f"  ✓ {instruction[:40]}...")
            except Exception as e:
                print(f"  ✗ Error: {e}")
    
    # Save the dataset
    print(f"\n💾 Saving dataset with {len(dataset)} samples...")
    
    output_file = data_dir / f"large_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with open(output_file, 'w') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')
    
    print(f"✅ Dataset saved to: {output_file}")
    print(f"📊 Total samples: {len(dataset)}")
    
    # Show distribution
    type_counts = {}
    for item in dataset:
        t = item.get('type', 'unknown')
        type_counts[t] = type_counts.get(t, 0) + 1
    
    print("\n📈 Dataset Distribution:")
    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count} samples")
    
    return dataset


async def main():
    """Generate the dataset."""
    dataset = await generate_large_dataset()
    
    print("\n🎯 Next Steps:")
    print("1. Train the model on this larger dataset")
    print("2. Use longer training with more epochs")
    print("3. Consider adding data augmentation")
    print("4. Fine-tune generation parameters")


if __name__ == "__main__":
    asyncio.run(main())