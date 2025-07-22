"""MLX utility wrapper classes for Shvayambhu LLM project."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict, Union, Callable
import json
import time
from pathlib import Path


class DeviceManager:
    """Manage MLX device operations and memory."""
    
    def __init__(self):
        self.device = mx.default_device()
        self._memory_stats = {}
    
    def get_device_info(self) -> Dict[str, any]:
        """Get device information."""
        return {
            'device': str(self.device),
            'type': 'GPU' if 'gpu' in str(self.device).lower() else 'CPU',
            'unified_memory': True,  # M4 Pro always has unified memory
            'memory_gb': 48  # Hardcoded for M4 Pro
        }
    
    def synchronize(self):
        """Synchronize device operations."""
        # MLX handles this automatically, but we provide for compatibility
        mx.eval(mx.array([0]))
    
    def clear_cache(self):
        """Clear any cached memory."""
        # Force garbage collection
        import gc
        gc.collect()


class TensorOps:
    """Common tensor operations with MLX."""
    
    @staticmethod
    def create_tensor(data: Union[list, np.ndarray], dtype=mx.float32) -> mx.array:
        """Create MLX tensor from data."""
        if isinstance(data, np.ndarray):
            return mx.array(data, dtype=dtype)
        return mx.array(data, dtype=dtype)
    
    @staticmethod
    def zeros(shape: Tuple[int, ...], dtype=mx.float32) -> mx.array:
        """Create zero tensor."""
        return mx.zeros(shape, dtype=dtype)
    
    @staticmethod
    def ones(shape: Tuple[int, ...], dtype=mx.float32) -> mx.array:
        """Create ones tensor."""
        return mx.ones(shape, dtype=dtype)
    
    @staticmethod
    def randn(shape: Tuple[int, ...], mean: float = 0.0, std: float = 1.0) -> mx.array:
        """Create random normal tensor."""
        return mx.random.normal(shape) * std + mean
    
    @staticmethod
    def cast(tensor: mx.array, dtype) -> mx.array:
        """Cast tensor to different dtype."""
        return tensor.astype(dtype)
    
    @staticmethod
    def to_numpy(tensor: mx.array) -> np.ndarray:
        """Convert MLX tensor to numpy."""
        return np.array(tensor)


class ModelCheckpoint:
    """Handle model checkpointing for MLX models."""
    
    def __init__(self, save_dir: str = "checkpoints"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def save(self, model: nn.Module, name: str, metadata: Optional[Dict] = None):
        """Save model checkpoint."""
        checkpoint_path = self.save_dir / f"{name}.npz"
        
        # Get model state
        state = model.parameters()
        
        # Flatten nested dictionaries
        flat_state = self._flatten_dict(state)
        
        # Save weights
        np.savez(checkpoint_path, **flat_state)
        
        # Save metadata if provided
        if metadata:
            meta_path = self.save_dir / f"{name}_meta.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def load(self, model: nn.Module, name: str) -> Dict:
        """Load model checkpoint."""
        checkpoint_path = self.save_dir / f"{name}.npz"
        
        # Load weights
        weights = np.load(checkpoint_path)
        
        # Unflatten to nested structure
        state = self._unflatten_dict(dict(weights))
        
        # Load into model
        model.update(state)
        
        # Load metadata if exists
        meta_path = self.save_dir / f"{name}_meta.json"
        metadata = {}
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        
        return metadata
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _unflatten_dict(self, d: Dict, sep: str = '.') -> Dict:
        """Unflatten dictionary."""
        result = {}
        for key, value in d.items():
            parts = key.split(sep)
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = mx.array(value)
        return result


class MemoryTracker:
    """Track memory usage during training/inference."""
    
    def __init__(self):
        self.snapshots = []
        
    def snapshot(self, label: str = ""):
        """Take memory snapshot."""
        # For M4 Pro, we track theoretical usage
        snapshot = {
            'label': label,
            'timestamp': time.time(),
            # Add actual memory tracking here if MLX provides it
        }
        self.snapshots.append(snapshot)
    
    def report(self) -> Dict:
        """Generate memory usage report."""
        return {
            'num_snapshots': len(self.snapshots),
            'snapshots': self.snapshots
        }


class GradientUtils:
    """Utilities for gradient computation and manipulation."""
    
    @staticmethod
    def compute_gradient(loss_fn: Callable, model: nn.Module, inputs: mx.array, 
                        targets: mx.array) -> Tuple[float, Dict]:
        """Compute loss and gradients."""
        loss_and_grad_fn = mx.value_and_grad(loss_fn)
        loss, grads = loss_and_grad_fn(model, inputs, targets)
        return loss.item(), grads
    
    @staticmethod
    def clip_gradients(grads: Dict, max_norm: float = 1.0) -> Dict:
        """Clip gradients by norm."""
        # Flatten gradients
        flat_grads = []
        shapes = []
        for g in grads.values():
            if isinstance(g, dict):
                for gg in g.values():
                    flat_grads.append(gg.reshape(-1))
                    shapes.append(gg.shape)
            else:
                flat_grads.append(g.reshape(-1))
                shapes.append(g.shape)
        
        # Compute norm
        all_grads = mx.concatenate(flat_grads)
        grad_norm = mx.sqrt(mx.sum(all_grads * all_grads))
        
        # Clip if needed
        if grad_norm > max_norm:
            scale = max_norm / grad_norm
            # Apply scaling
            clipped_grads = {}
            for key, grad in grads.items():
                if isinstance(grad, dict):
                    clipped_grads[key] = {k: g * scale for k, g in grad.items()}
                else:
                    clipped_grads[key] = grad * scale
            return clipped_grads
        
        return grads


class DataUtils:
    """Utilities for data handling."""
    
    @staticmethod
    def create_batches(data: mx.array, batch_size: int, shuffle: bool = True) -> List[mx.array]:
        """Create batches from data."""
        n_samples = data.shape[0]
        indices = mx.arange(n_samples)
        
        if shuffle:
            indices = mx.random.permutation(indices)
        
        batches = []
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batches.append(data[batch_indices])
        
        return batches
    
    @staticmethod
    def pad_sequences(sequences: List[mx.array], pad_value: int = 0) -> mx.array:
        """Pad sequences to same length."""
        max_len = max(seq.shape[0] for seq in sequences)
        batch_size = len(sequences)
        
        # Create padded array
        padded = mx.full((batch_size, max_len), pad_value)
        
        # Fill with sequences
        for i, seq in enumerate(sequences):
            padded[i, :seq.shape[0]] = seq
        
        return padded


class Profiler:
    """Simple profiler for MLX operations."""
    
    def __init__(self):
        self.timings = {}
    
    def profile(self, name: str, fn: Callable, *args, **kwargs):
        """Profile a function call."""
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        mx.eval(result)  # Force evaluation
        end = time.perf_counter()
        
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(end - start)
        
        return result
    
    def report(self) -> Dict:
        """Generate profiling report."""
        report = {}
        for name, times in self.timings.items():
            times_array = np.array(times)
            report[name] = {
                'count': len(times),
                'total': np.sum(times_array),
                'mean': np.mean(times_array),
                'std': np.std(times_array),
                'min': np.min(times_array),
                'max': np.max(times_array)
            }
        return report


# Convenience functions
def get_device_manager() -> DeviceManager:
    """Get singleton device manager."""
    if not hasattr(get_device_manager, '_instance'):
        get_device_manager._instance = DeviceManager()
    return get_device_manager._instance


def format_memory_size(bytes: int) -> str:
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} PB"