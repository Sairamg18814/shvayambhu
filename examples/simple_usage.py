#!/usr/bin/env python3
"""
Simple Usage Examples for Shvayambhu LLM
========================================

Shows how to use Shvayambhu as easily as other popular LLMs.
"""

# Example 1: Basic usage (like OpenAI)
from shvayambhu import Shvayambhu

# Create model
model = Shvayambhu()

# Generate text
response = model.generate("What is the meaning of life?")
print(response)


# Example 2: Streaming (like Claude)
print("\n" + "="*50 + "\n")

# Stream tokens
for token in model.stream("Tell me a story about a conscious AI"):
    print(token, end='', flush=True)
print("\n")


# Example 3: Chat interface (like ChatGPT)
print("\n" + "="*50 + "\n")

# Reset for fresh conversation
model.reset()

# Chat
response1 = model.chat("Hello! Can you introduce yourself?")
print(f"Assistant: {response1}\n")

response2 = model.chat("What makes you different from other AI assistants?")
print(f"Assistant: {response2}\n")


# Example 4: Customization
print("\n" + "="*50 + "\n")

# Create customized model
custom_model = Shvayambhu(
    model_size="large",
    temperature=0.7,
    consciousness=True,
    emotional=True,
    memory=True,
    safety=True
)

# Use with custom parameters
response = custom_model.generate(
    "Write a philosophical reflection on consciousness",
    max_tokens=200,
    temperature=0.9
)
print(response)


# Example 5: Without consciousness (faster, like standard LLMs)
print("\n" + "="*50 + "\n")

# Fast mode - no consciousness
fast_model = Shvayambhu(
    consciousness=False,
    emotional=False,
    memory=False
)

response = fast_model.generate("Explain quantum computing in simple terms")
print(response)


# Example 6: System prompts
print("\n" + "="*50 + "\n")

# Chat with system prompt
model.reset()
response = model.chat(
    "What's your opinion on modern art?",
    system="You are an art critic with strong opinions."
)
print(response)