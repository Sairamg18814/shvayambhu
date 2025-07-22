"""MLX utility wrapper classes for Shvayambhu project."""

import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for MLX models."""
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    vocab_size: int
    max_position_embeddings: int = 2048
    intermediate_size: Optional[int] = None
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    use_cache: bool = True
    
    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size

class MLXModelWrapper:
    """Base wrapper for MLX models with common utilities."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = mx.default_device()
        self._model = None
        self._optimizer = None
        logger.info(f"MLX Model initialized on device: {self.device}")
        
    def save_checkpoint(self, path: Union[str, Path], metadata: Optional[Dict] = None):
        """Save model checkpoint with metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state': self.get_state_dict(),
            'config': self.config.__dict__,
            'metadata': metadata or {}
        }
        
        mx.save(str(path), checkpoint)
        logger.info(f"Checkpoint saved to {path}")
        
    def load_checkpoint(self, path: Union[str, Path]):
        """Load model checkpoint."""
        path = Path(path)
        checkpoint = mx.load(str(path))
        
        self.set_state_dict(checkpoint['model_state'])
        logger.info(f"Checkpoint loaded from {path}")
        return checkpoint.get('metadata', {})
        
    def get_state_dict(self) -> Dict:
        """Get model state dictionary."""
        if self._model is None:
            raise ValueError("Model not initialized")
        return self._model.parameters()
        
    def set_state_dict(self, state_dict: Dict):
        """Set model state dictionary."""
        if self._model is None:
            raise ValueError("Model not initialized")
        self._model.load_weights(state_dict)
        
    def count_parameters(self) -> int:
        """Count total model parameters."""
        if self._model is None:
            return 0
            
        total = 0
        for param in self._model.parameters().values():
            if isinstance(param, dict):
                for sub_param in param.values():
                    total += sub_param.size
            else:
                total += param.size
        return total
        
    def memory_usage(self) -> Dict[str, float]:
        """Estimate memory usage in GB."""
        param_count = self.count_parameters()
        
        return {
            'parameters_fp32': param_count * 4 / 1e9,
            'parameters_fp16': param_count * 2 / 1e9,
            'parameters_int8': param_count / 1e9,
            'parameters_int4': param_count * 0.5 / 1e9,
        }

class TensorUtils:
    """Utility functions for tensor operations."""
    
    @staticmethod
    def safe_softmax(x: mx.array, axis: int = -1, temperature: float = 1.0) -> mx.array:
        """Numerically stable softmax with temperature scaling."""
        x = x / temperature
        x_max = mx.max(x, axis=axis, keepdims=True)
        exp_x = mx.exp(x - x_max)
        return exp_x / mx.sum(exp_x, axis=axis, keepdims=True)
        
    @staticmethod
    def apply_rotary_pos_emb(x: mx.array, pos: mx.array) -> mx.array:
        """Apply rotary position embeddings."""
        # Simplified rotary embedding for demonstration
        # In practice, this would be more complex
        cos = mx.cos(pos)
        sin = mx.sin(pos)
        return x * cos + mx.roll(x, 1, axis=-1) * sin
        
    @staticmethod
    def create_causal_mask(seq_len: int) -> mx.array:
        """Create causal attention mask."""
        mask = mx.tril(mx.ones((seq_len, seq_len)))
        return mask
        
    @staticmethod
    def batch_matmul(a: mx.array, b: mx.array) -> mx.array:
        """Efficient batch matrix multiplication."""
        return mx.matmul(a, b)

class MemoryManager:
    """Manage memory for large models."""
    
    def __init__(self, memory_limit_gb: float = 40.0):
        self.memory_limit_gb = memory_limit_gb
        self.allocated_arrays: List[mx.array] = []
        
    def allocate(self, shape: Tuple[int, ...], dtype: mx.Dtype = mx.float32) -> mx.array:
        """Allocate array with memory tracking."""
        array = mx.zeros(shape, dtype=dtype)
        self.allocated_arrays.append(array)
        
        current_usage = self.get_memory_usage()
        if current_usage > self.memory_limit_gb:
            logger.warning(f"Memory usage ({current_usage:.2f}GB) exceeds limit ({self.memory_limit_gb}GB)")
            
        return array
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        total_bytes = 0
        for array in self.allocated_arrays:
            total_bytes += array.nbytes
        return total_bytes / 1e9
        
    def clear(self):
        """Clear tracked arrays."""
        self.allocated_arrays.clear()
        mx.eval()  # Force evaluation to free memory

class AttentionCache:
    """KV cache for attention layers."""
    
    def __init__(self, batch_size: int, max_seq_len: int, num_heads: int, head_dim: int):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Pre-allocate cache
        cache_shape = (batch_size, num_heads, max_seq_len, head_dim)
        self.key_cache = mx.zeros(cache_shape)
        self.value_cache = mx.zeros(cache_shape)
        self.cache_len = 0
        
    def update(self, keys: mx.array, values: mx.array) -> Tuple[mx.array, mx.array]:
        """Update cache with new keys and values."""
        seq_len = keys.shape[2]
        
        if self.cache_len + seq_len > self.max_seq_len:
            # Implement sliding window or other strategy
            logger.warning("Cache overflow, implementing sliding window")
            self.cache_len = 0
            
        # Update cache
        self.key_cache[:, :, self.cache_len:self.cache_len + seq_len] = keys
        self.value_cache[:, :, self.cache_len:self.cache_len + seq_len] = values
        self.cache_len += seq_len
        
        # Return full cache up to current position
        return (
            self.key_cache[:, :, :self.cache_len],
            self.value_cache[:, :, :self.cache_len]
        )
        
    def clear(self):
        """Clear the cache."""
        self.cache_len = 0

class GradientAccumulator:
    """Accumulate gradients for large batch training."""
    
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.accumulated_grads = {}
        self.step_count = 0
        
    def accumulate(self, grads: Dict[str, mx.array]):
        """Accumulate gradients."""
        if not self.accumulated_grads:
            self.accumulated_grads = {k: mx.zeros_like(v) for k, v in grads.items()}
            
        for key, grad in grads.items():
            self.accumulated_grads[key] += grad / self.accumulation_steps
            
        self.step_count += 1
        
    def should_update(self) -> bool:
        """Check if we should update parameters."""
        return self.step_count >= self.accumulation_steps
        
    def get_accumulated_grads(self) -> Dict[str, mx.array]:
        """Get accumulated gradients and reset."""
        grads = self.accumulated_grads.copy()
        self.accumulated_grads = {}
        self.step_count = 0
        return grads

class PerformanceMonitor:
    """Monitor model performance metrics."""
    
    def __init__(self):
        self.metrics = {
            'throughput': [],
            'latency': [],
            'memory': [],
            'flops': []
        }
        
    def measure_throughput(self, tokens: int, time_seconds: float):
        """Measure tokens per second."""
        throughput = tokens / time_seconds
        self.metrics['throughput'].append(throughput)
        return throughput
        
    def measure_latency(self, batch_size: int, seq_len: int, time_seconds: float):
        """Measure latency per token."""
        total_tokens = batch_size * seq_len
        latency = time_seconds / total_tokens * 1000  # ms per token
        self.metrics['latency'].append(latency)
        return latency
        
    def get_summary(self) -> Dict[str, float]:
        """Get performance summary."""
        summary = {}
        for metric, values in self.metrics.items():
            if values:
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_std'] = np.std(values)
                summary[f'{metric}_min'] = np.min(values)
                summary[f'{metric}_max'] = np.max(values)
        return summary

class MLXOptimizer:
    """Wrapper for MLX optimizers with additional features."""
    
    def __init__(self, optimizer_type: str = "adamw", learning_rate: float = 1e-4, **kwargs):
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        
        if optimizer_type == "adamw":
            self.optimizer = optim.AdamW(learning_rate=learning_rate, **kwargs)
        elif optimizer_type == "adam":
            self.optimizer = optim.Adam(learning_rate=learning_rate, **kwargs)
        elif optimizer_type == "sgd":
            self.optimizer = optim.SGD(learning_rate=learning_rate, **kwargs)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
            
    def update(self, model: nn.Module, grads: Dict[str, mx.array]):
        """Update model parameters."""
        self.optimizer.update(model, grads)
        
    def set_learning_rate(self, lr: float):
        """Update learning rate."""
        self.learning_rate = lr
        # MLX optimizers don't have a direct LR setter, so we recreate
        self.__init__(self.optimizer_type, lr)

# Utility functions
def estimate_model_size(config: ModelConfig) -> Dict[str, float]:
    """Estimate model size for different quantizations."""
    # Rough estimation based on transformer architecture
    vocab_params = config.vocab_size * config.hidden_size
    attention_params = config.num_layers * (4 * config.hidden_size * config.hidden_size)
    ffn_params = config.num_layers * (2 * config.hidden_size * config.intermediate_size)
    total_params = vocab_params + attention_params + ffn_params
    
    return {
        'total_parameters': total_params,
        'size_fp32_gb': total_params * 4 / 1e9,
        'size_fp16_gb': total_params * 2 / 1e9,
        'size_int8_gb': total_params / 1e9,
        'size_int4_gb': total_params * 0.5 / 1e9,
    }

def create_model_config(model_size: str) -> ModelConfig:
    """Create model configuration for standard sizes."""
    configs = {
        '7b': ModelConfig(
            hidden_size=4096,
            num_layers=32,
            num_attention_heads=32,
            vocab_size=50000
        ),
        '13b': ModelConfig(
            hidden_size=5120,
            num_layers=40,
            num_attention_heads=40,
            vocab_size=50000
        ),
        '30b': ModelConfig(
            hidden_size=7168,
            num_layers=48,
            num_attention_heads=56,
            vocab_size=50000
        )
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}")
        
    return configs[model_size]