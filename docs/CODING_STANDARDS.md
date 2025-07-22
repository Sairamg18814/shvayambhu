# Shvayambhu Coding Standards and Conventions

## Overview

This document outlines the coding standards and conventions for the Shvayambhu LLM project. All contributors must follow these guidelines to ensure code consistency, maintainability, and quality.

## Python Style Guide

### General Principles

1. **Readability counts** - Code is read more often than it's written
2. **Explicit is better than implicit** - Be clear about intentions
3. **Simple is better than complex** - Avoid over-engineering
4. **Performance matters** - But profile before optimizing

### Code Formatting

We use **Black** for automatic code formatting with these settings:
- Line length: 88 characters
- Python target version: 3.11+

```python
# pyproject.toml configuration
[tool.black]
line-length = 88
target-version = ['py311']
```

### Import Organization

Use **isort** with Black-compatible settings:

```python
# Correct import order
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from shvayambhu.core.blt import LocalEncoder
from shvayambhu.utils.metrics import calculate_entropy
```

### Naming Conventions

#### Variables and Functions
- Use `snake_case` for variables and functions
- Be descriptive but concise
- Avoid single-letter variables except for indices

```python
# Good
def calculate_patch_entropy(byte_sequence: bytes, window_size: int = 16) -> float:
    """Calculate entropy for dynamic patching."""
    pass

# Bad
def calc(b, w=16):
    pass
```

#### Classes
- Use `PascalCase` for class names
- Suffix with purpose when appropriate (e.g., `Encoder`, `Manager`, `Handler`)

```python
class LocalEncoder:
    """Converts byte sequences to patches."""
    pass

class MemoryManager:
    """Manages unified memory allocation."""
    pass
```

#### Constants
- Use `UPPER_SNAKE_CASE` for constants
- Define at module level

```python
DEFAULT_PATCH_SIZE = 16
MIN_PATCH_SIZE = 4
MAX_PATCH_SIZE = 32
ENTROPY_THRESHOLD = 0.7
```

### Type Hints

**Always use type hints** for function signatures and class attributes:

```python
from typing import List, Optional, Tuple, Union, Dict
import torch
from torch import Tensor

def process_bytes(
    input_bytes: bytes,
    patch_size: int = 16,
    return_entropy: bool = False
) -> Union[Tensor, Tuple[Tensor, float]]:
    """Process input bytes into patches.
    
    Args:
        input_bytes: Raw UTF-8 bytes to process
        patch_size: Target patch size
        return_entropy: Whether to return entropy value
        
    Returns:
        Processed tensor or tuple of (tensor, entropy)
    """
    pass
```

### Docstrings

Use **Google-style docstrings** for all public functions, classes, and modules:

```python
def create_patches(
    byte_sequence: bytes,
    min_size: int = 4,
    max_size: int = 32
) -> List[bytes]:
    """Create variable-size patches based on content entropy.
    
    This function analyzes the entropy of byte sequences and creates
    patches of varying sizes. High-entropy regions get smaller patches
    for better resolution, while low-entropy regions use larger patches
    for efficiency.
    
    Args:
        byte_sequence: Input bytes to patch
        min_size: Minimum patch size in bytes
        max_size: Maximum patch size in bytes
        
    Returns:
        List of byte patches
        
    Raises:
        ValueError: If min_size > max_size
        
    Example:
        >>> patches = create_patches(b"Hello, world!", min_size=2, max_size=8)
        >>> len(patches)
        3
    """
    if min_size > max_size:
        raise ValueError(f"min_size ({min_size}) must be <= max_size ({max_size})")
    
    # Implementation here
    pass
```

### Error Handling

Be explicit about error conditions and use appropriate exception types:

```python
def load_model(path: Path) -> ShvayambhuModel:
    """Load model from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    if not path.suffix == ".safetensors":
        raise ValueError(f"Invalid model format: {path.suffix}")
    
    try:
        model = ShvayambhuModel.from_file(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e
    
    return model
```

### Comments

- Use comments sparingly - code should be self-documenting
- Explain **why**, not **what**
- Update comments when code changes

```python
# Good: Explains why
# Use smaller patches for high-entropy regions to capture more detail
if entropy > ENTROPY_THRESHOLD:
    patch_size = min(patch_size // 2, MIN_PATCH_SIZE)

# Bad: States the obvious
# Increment counter by 1
counter += 1
```

## Project-Specific Conventions

### File Organization

```python
# Standard module structure
"""Module docstring explaining purpose."""

# Standard library imports
import os
from pathlib import Path

# Third-party imports
import torch
import numpy as np

# Local imports
from shvayambhu.core import base
from shvayambhu.utils import metrics

# Constants
DEFAULT_VALUE = 42

# Module-level functions
def helper_function():
    pass

# Classes
class MainClass:
    pass

# Script execution
if __name__ == "__main__":
    main()
```

### Logging

Use structured logging instead of print statements:

```python
import logging

logger = logging.getLogger(__name__)

def process_data(data: bytes) -> bytes:
    logger.debug(f"Processing {len(data)} bytes")
    
    try:
        result = transform(data)
        logger.info(f"Successfully processed {len(result)} bytes")
        return result
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise
```

### Testing

Write tests alongside code with descriptive names:

```python
# tests/unit/test_entropy.py
import pytest
from shvayambhu.core.blt.entropy import calculate_entropy

class TestEntropy:
    """Test entropy calculation."""
    
    def test_uniform_distribution_high_entropy(self):
        """Uniform byte distribution should have high entropy."""
        data = bytes(range(256))  # All possible byte values
        entropy = calculate_entropy(data)
        assert entropy > 0.9
    
    def test_repeated_bytes_low_entropy(self):
        """Repeated bytes should have low entropy."""
        data = b"A" * 1000
        entropy = calculate_entropy(data)
        assert entropy < 0.1
    
    @pytest.mark.parametrize("data,expected_range", [
        (b"Hello, world!", (0.3, 0.7)),
        (b"\x00\x01\x02\x03", (0.4, 0.6)),
    ])
    def test_entropy_ranges(self, data, expected_range):
        """Test entropy falls within expected ranges."""
        entropy = calculate_entropy(data)
        assert expected_range[0] <= entropy <= expected_range[1]
```

### Performance Considerations

1. **Profile before optimizing** - Use `py-spy` or `cProfile`
2. **Prefer vectorized operations** - Use PyTorch/NumPy operations
3. **Cache expensive computations** - Use `functools.lru_cache` when appropriate
4. **Document performance-critical code**

```python
from functools import lru_cache
import torch

@lru_cache(maxsize=1024)
def compute_expensive_metric(data_hash: int) -> float:
    """Compute metric with caching for repeated inputs."""
    # Expensive computation here
    pass

def process_batch(batch: torch.Tensor) -> torch.Tensor:
    """Process batch using vectorized operations.
    
    Performance: O(n) time, optimized for GPU execution.
    Memory: Requires 2x batch memory for intermediate results.
    """
    # Vectorized implementation
    return torch.nn.functional.softmax(batch, dim=-1)
```

### Metal/GPU Code

Document Metal-specific optimizations:

```python
def matmul_metal(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication optimized for Apple Silicon.
    
    Uses Metal Performance Shaders for acceleration.
    Falls back to CPU if Metal unavailable.
    
    Note:
        - Input tensors must be on MPS device
        - Supports float32 and float16 only
        - Optimized for matrices > 512x512
    """
    if not torch.backends.mps.is_available():
        logger.warning("Metal not available, using CPU")
        return torch.matmul(a.cpu(), b.cpu())
    
    return torch.matmul(a.to("mps"), b.to("mps"))
```

### Security and Privacy

1. **Never log sensitive data** - No user inputs in logs
2. **Sanitize file paths** - Use `pathlib.Path` for path operations
3. **No network calls** - Enforce offline operation

```python
from pathlib import Path

def safe_load_file(file_path: str) -> bytes:
    """Safely load file with path validation."""
    # Convert to Path and resolve to prevent path traversal
    path = Path(file_path).resolve()
    
    # Ensure path is within allowed directory
    allowed_dir = Path("/data/shvayambhu").resolve()
    if not str(path).startswith(str(allowed_dir)):
        raise SecurityError(f"Access denied: {path}")
    
    # Never log file contents
    logger.info(f"Loading file: {path.name}")  # Only log filename
    
    return path.read_bytes()
```

## Code Review Checklist

Before submitting code for review, ensure:

- [ ] Code passes all linters (`make lint`)
- [ ] All tests pass (`make test`)
- [ ] Type hints are complete
- [ ] Docstrings are updated
- [ ] No commented-out code
- [ ] No debug print statements
- [ ] Performance-critical sections are documented
- [ ] Security considerations are addressed
- [ ] Changes are covered by tests

## Tool Configuration

Ensure your development environment is properly configured:

```bash
# Install pre-commit hooks
pre-commit install

# Run all checks
make check

# Format code
make format

# Run linters
make lint

# Type checking
make type-check
```

## Version Control

### Commit Messages

Follow conventional commit format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Example:
```
feat(blt): implement entropy-based dynamic patching

- Add entropy calculation for byte sequences
- Implement adaptive patch size selection
- Add comprehensive unit tests
- Optimize for Metal acceleration

Closes #42
```

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `perf/description` - Performance improvements
- `refactor/description` - Code refactoring

## Conclusion

These standards ensure our codebase remains clean, efficient, and maintainable. When in doubt, prioritize readability and follow the existing patterns in the codebase.

Remember: **Shvayambhu's mission is to democratize AI while preserving privacy**. Every line of code should support complete offline operation and user data sovereignty.