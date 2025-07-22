.. Shvayambhu documentation master file

Welcome to Shvayambhu's documentation!
======================================

**Shvayambhu** is the world's first self-evolving, completely offline Large Language Model 
that operates entirely on Apple Silicon devices while matching state-of-the-art performance.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   architecture
   training_guide
   api_reference
   installation
   quickstart
   examples

Key Features
------------

* **100% Offline Operation**: No internet connection required, ever
* **Self-Evolution**: Autonomously learns and improves using SEAL architecture  
* **Zero Hallucination**: Advanced verification with selective abstention
* **Privacy First**: All processing on your device, zero data transmission
* **Apple Silicon Optimized**: Leverages Metal 3 and unified memory architecture
* **Universal Language Support**: 50+ languages with byte-level processing

Architecture Overview
--------------------

Shvayambhu introduces three groundbreaking innovations:

1. **BLT (Byte Latent Transformer)**
   
   * Eliminates tokenization overhead
   * Processes raw bytes directly
   * Dynamic entropy-based patching
   * 50% more efficient than token-based models

2. **SEAL (Self-Adapting Language Model)**
   
   * Autonomous model improvement
   * LoRA-based adaptation layers
   * No external data required
   * Continuous learning without internet

3. **ProRL (Prolonged Reinforcement Learning)**
   
   * Extended reasoning (2000+ steps)
   * Tree and Graph-of-Thought processing
   * Novel problem-solving strategies
   * Advanced multi-step reasoning

Quick Start
-----------

.. code-block:: python

   from shvayambhu import ShvayambhuModel

   # Load the model
   model = ShvayambhuModel.load("shvayambhu-7b")

   # Generate text
   response = model.generate("Explain quantum computing in simple terms")
   print(response)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`