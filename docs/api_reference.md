# API Reference

## Core Classes

### ShvayambhuModel

The main model class for inference.

```python
class ShvayambhuModel:
    def __init__(self, config: ModelConfig):
        """Initialize a Shvayambhu model.
        
        Args:
            config: Model configuration object
        """
        
    def generate(self, prompt: str, max_length: int = 512) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum generation length
            
        Returns:
            Generated text
        """
        
    def reason(self, query: str, strategy: str = "auto") -> ReasoningResult:
        """Perform multi-step reasoning.
        
        Args:
            query: Complex query requiring reasoning
            strategy: One of "chain", "tree", "graph", or "auto"
            
        Returns:
            ReasoningResult with answer and trace
        """
```

### BLT Components

#### LocalEncoder

```python
class LocalEncoder:
    def __init__(self, patch_size: int = 16):
        """Initialize byte-to-patch encoder."""
        
    def encode(self, bytes: bytes) -> torch.Tensor:
        """Convert bytes to patch embeddings."""
```

## Configuration

### ModelConfig

Configuration class for model initialization.

```python
@dataclass
class ModelConfig:
    model_size: str  # "7B", "13B", or "30B"
    device: str = "mps"  # Metal Performance Shaders
    quantization: str = "int8"  # Quantization type
    max_seq_length: int = 2048
```