# Training Guide

This guide covers the training process for Shvayambhu models.

## Overview

Shvayambhu uses a unique self-training approach with four phases:

1. **Bootstrap Training**: Initial training on minimal seed data
2. **Synthetic Generation**: Creating high-quality synthetic data
3. **Active Learning**: Identifying and filling knowledge gaps
4. **SEAL Evolution**: Continuous self-improvement

## Requirements

- Apple Silicon Mac with 32GB+ RAM
- 500GB+ free storage for training data
- Python 3.11+ environment

## Training Process

### Phase 1: Bootstrap

```python
from shvayambhu.training import bootstrap_training

model = bootstrap_training(
    seed_data_path="data/seed/",
    model_size="7B",
    objectives=["mlm", "next_byte"],
    epochs=10
)
```

### Phase 2: Synthetic Data

The model generates its own training data while maintaining quality.

### Phase 3: Active Learning

The system identifies areas of uncertainty and focuses learning there.

### Phase 4: SEAL Evolution

Autonomous improvement through self-edits and adaptation.