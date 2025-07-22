"""Memory-efficient LoRA implementation for SEAL architecture.

This module provides memory-optimized LoRA adapters that minimize GPU memory
usage while maintaining training efficiency and quality.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import warnings
import gc
import threading
from concurrent.futures import ThreadPoolExecutor

from .lora_adapter import LoRALinear, LoRAEmbedding, LoRAConfig


@dataclass
class MemoryConfig:
    """Configuration for memory-efficient LoRA operations."""
    # Memory optimization strategies
    use_gradient_checkpointing: bool = True
    use_cpu_offload: bool = True
    use_activation_checkpointing: bool = True
    
    # Memory limits
    max_gpu_memory_mb: float = 8192  # 8GB default
    memory_threshold: float = 0.85   # Use 85% of available memory
    
    # Optimization settings
    enable_quantization: bool = True
    quantization_bits: int = 8
    use_mixed_precision: bool = True
    
    # CPU offloading
    offload_optimizer_states: bool = True
    offload_gradients: bool = True
    pin_memory: bool = True
    
    # Memory monitoring
    enable_memory_profiling: bool = False
    memory_check_interval: int = 100  # steps
    auto_garbage_collection: bool = True


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    current_gpu_memory_mb: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    cpu_memory_mb: float = 0.0
    offloaded_parameters_mb: float = 0.0
    
    # Component breakdown
    lora_adapters_memory_mb: float = 0.0
    gradients_memory_mb: float = 0.0
    optimizer_states_memory_mb: float = 0.0
    activations_memory_mb: float = 0.0
    
    # Efficiency metrics
    memory_efficiency: float = 0.0  # Useful memory / total memory
    offload_ratio: float = 0.0      # Offloaded / total parameters


class MemoryEfficientLoRA(nn.Module):
    """Memory-efficient LoRA adapter with CPU offloading and quantization."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        config: MemoryConfig,
        alpha: float = 1.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.config = config
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Initialize LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / np.sqrt(in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Quantization setup
        if config.enable_quantization:
            self._setup_quantization()
        
        # CPU offloading setup
        self.offloaded_params = {}
        self.param_locations = {}  # Track where parameters are stored
        
        # Memory tracking
        self.memory_stats = MemoryStats()
        self._setup_memory_monitoring()
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def _setup_quantization(self):
        """Setup quantization for memory efficiency."""
        if self.config.quantization_bits == 8:
            self.register_buffer('lora_A_scale', torch.ones(self.rank))
            self.register_buffer('lora_B_scale', torch.ones(self.out_features))
        elif self.config.quantization_bits == 4:
            self.register_buffer('lora_A_scale', torch.ones(self.rank))
            self.register_buffer('lora_A_zero_point', torch.zeros(self.rank, dtype=torch.int8))
            self.register_buffer('lora_B_scale', torch.ones(self.out_features))
            self.register_buffer('lora_B_zero_point', torch.zeros(self.out_features, dtype=torch.int8))
    
    def _setup_memory_monitoring(self):
        """Setup memory monitoring and profiling."""
        if self.config.enable_memory_profiling:
            self.memory_timeline = []
            self.step_counter = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Memory-efficient forward pass."""
        if self.config.enable_memory_profiling:
            self._record_memory_usage()
        
        # Ensure parameters are on GPU
        self._ensure_parameters_available()
        
        # Apply gradient checkpointing if enabled
        if self.config.use_gradient_checkpointing and self.training:
            return self._checkpointed_forward(x)
        else:
            return self._standard_forward(x)
    
    def _standard_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        # Get quantized parameters if needed
        lora_A, lora_B = self._get_dequantized_parameters()
        
        # Forward pass through LoRA path
        if self.dropout and self.training:
            x = self.dropout(x)
        
        # A @ x
        result = torch.matmul(x, lora_A.T)
        
        # Optional activation checkpointing
        if self.config.use_activation_checkpointing and self.training:
            result = torch.utils.checkpoint.checkpoint(
                lambda r: torch.matmul(r, lora_B.T), result
            )
        else:
            # B @ (A @ x)
            result = torch.matmul(result, lora_B.T)
        
        return result * self.scaling
    
    def _checkpointed_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with gradient checkpointing."""
        return torch.utils.checkpoint.checkpoint(self._standard_forward, x)
    
    def _get_dequantized_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get dequantized parameters for computation."""
        if self.config.enable_quantization:
            if self.config.quantization_bits == 8:
                lora_A = self._dequantize_int8(self.lora_A, self.lora_A_scale)
                lora_B = self._dequantize_int8(self.lora_B, self.lora_B_scale)
            elif self.config.quantization_bits == 4:
                lora_A = self._dequantize_int4(
                    self.lora_A, self.lora_A_scale, self.lora_A_zero_point
                )
                lora_B = self._dequantize_int4(
                    self.lora_B, self.lora_B_scale, self.lora_B_zero_point
                )
            else:
                lora_A, lora_B = self.lora_A, self.lora_B
        else:
            lora_A, lora_B = self.lora_A, self.lora_B
        
        return lora_A, lora_B
    
    def _dequantize_int8(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize INT8 parameters."""
        return quantized.float() * scale.unsqueeze(-1)
    
    def _dequantize_int4(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize INT4 parameters."""
        return (quantized.float() - zero_point.unsqueeze(-1)) * scale.unsqueeze(-1)
    
    def _ensure_parameters_available(self):
        """Ensure parameters are available on GPU when needed."""
        if not self.config.use_cpu_offload:
            return
        
        device = next(self.parameters()).device
        
        # Move offloaded parameters back to GPU
        for param_name in ['lora_A', 'lora_B']:
            if param_name in self.offloaded_params:
                param = getattr(self, param_name)
                if param.device != device:
                    param.data = self.offloaded_params[param_name].to(
                        device, non_blocking=self.config.pin_memory
                    )
                    self.param_locations[param_name] = 'gpu'
    
    def offload_to_cpu(self):
        """Offload parameters to CPU to save GPU memory."""
        if not self.config.use_cpu_offload:
            return
        
        for param_name in ['lora_A', 'lora_B']:
            param = getattr(self, param_name)
            if param.device.type == 'cuda':
                # Store on CPU
                self.offloaded_params[param_name] = param.detach().cpu()
                if self.config.pin_memory:
                    self.offloaded_params[param_name] = self.offloaded_params[param_name].pin_memory()
                
                # Create placeholder on GPU (minimal memory)
                param.data = torch.empty_like(param)
                self.param_locations[param_name] = 'cpu'
    
    def quantize_parameters(self):
        """Quantize parameters to save memory."""
        if not self.config.enable_quantization:
            return
        
        if self.config.quantization_bits == 8:
            self._quantize_to_int8()
        elif self.config.quantization_bits == 4:
            self._quantize_to_int4()
    
    def _quantize_to_int8(self):
        """Quantize parameters to INT8."""
        # Quantize lora_A
        scale_A = self.lora_A.abs().max(dim=-1, keepdim=True)[0] / 127.0
        quantized_A = torch.round(self.lora_A / scale_A).clamp(-128, 127)
        self.lora_A.data = quantized_A.to(torch.int8)
        self.lora_A_scale.data = scale_A.squeeze(-1)
        
        # Quantize lora_B
        scale_B = self.lora_B.abs().max(dim=-1, keepdim=True)[0] / 127.0
        quantized_B = torch.round(self.lora_B / scale_B).clamp(-128, 127)
        self.lora_B.data = quantized_B.to(torch.int8)
        self.lora_B_scale.data = scale_B.squeeze(-1)
    
    def _quantize_to_int4(self):
        """Quantize parameters to INT4."""
        # Quantize lora_A
        min_val_A = self.lora_A.min(dim=-1, keepdim=True)[0]
        max_val_A = self.lora_A.max(dim=-1, keepdim=True)[0]
        scale_A = (max_val_A - min_val_A) / 15.0
        zero_point_A = torch.round(-min_val_A / scale_A).clamp(0, 15)
        quantized_A = torch.round(self.lora_A / scale_A + zero_point_A).clamp(0, 15)
        
        self.lora_A.data = quantized_A.to(torch.int8)
        self.lora_A_scale.data = scale_A.squeeze(-1)
        self.lora_A_zero_point.data = zero_point_A.squeeze(-1).to(torch.int8)
        
        # Quantize lora_B (similar process)
        min_val_B = self.lora_B.min(dim=-1, keepdim=True)[0]
        max_val_B = self.lora_B.max(dim=-1, keepdim=True)[0]
        scale_B = (max_val_B - min_val_B) / 15.0
        zero_point_B = torch.round(-min_val_B / scale_B).clamp(0, 15)
        quantized_B = torch.round(self.lora_B / scale_B + zero_point_B).clamp(0, 15)
        
        self.lora_B.data = quantized_B.to(torch.int8)
        self.lora_B_scale.data = scale_B.squeeze(-1)
        self.lora_B_zero_point.data = zero_point_B.squeeze(-1).to(torch.int8)
    
    def _record_memory_usage(self):
        """Record current memory usage for profiling."""
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            self.memory_stats.current_gpu_memory_mb = current_memory
            self.memory_stats.peak_gpu_memory_mb = peak_memory
            
            if hasattr(self, 'memory_timeline'):
                self.memory_timeline.append({
                    'step': self.step_counter,
                    'gpu_memory_mb': current_memory,
                    'peak_memory_mb': peak_memory
                })
        
        self.step_counter += 1
        
        # Garbage collection
        if (self.config.auto_garbage_collection and 
            self.step_counter % self.config.memory_check_interval == 0):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        return self.lora_A.numel() + self.lora_B.numel()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get detailed memory usage breakdown."""
        param_memory = 0
        for param in self.parameters():
            param_memory += param.numel() * param.element_size()
        
        # Convert to MB
        param_memory_mb = param_memory / 1024 / 1024
        
        return {
            'parameter_memory_mb': param_memory_mb,
            'current_gpu_memory_mb': self.memory_stats.current_gpu_memory_mb,
            'peak_gpu_memory_mb': self.memory_stats.peak_gpu_memory_mb,
            'offloaded_memory_mb': sum(
                p.numel() * p.element_size() for p in self.offloaded_params.values()
            ) / 1024 / 1024
        }


class MemoryEfficientLoRAManager:
    """Manages multiple memory-efficient LoRA adapters."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.adapters: Dict[str, MemoryEfficientLoRA] = {}
        self.memory_monitor = MemoryMonitor(config)
        self.offload_scheduler = OffloadScheduler(config)
    
    def add_adapter(
        self,
        name: str,
        in_features: int,
        out_features: int,
        rank: int,
        **kwargs
    ) -> MemoryEfficientLoRA:
        """Add a new memory-efficient LoRA adapter."""
        adapter = MemoryEfficientLoRA(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            config=self.config,
            **kwargs
        )
        
        self.adapters[name] = adapter
        self.memory_monitor.register_adapter(name, adapter)
        
        return adapter
    
    def remove_adapter(self, name: str):
        """Remove an adapter and free its memory."""
        if name in self.adapters:
            adapter = self.adapters[name]
            adapter.offload_to_cpu()
            del self.adapters[name]
            self.memory_monitor.unregister_adapter(name)
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def optimize_memory_usage(self):
        """Optimize memory usage across all adapters."""
        # Get current memory usage
        total_memory = self.memory_monitor.get_total_memory_usage()
        
        if total_memory > self.config.max_gpu_memory_mb * self.config.memory_threshold:
            # Need to offload some adapters
            candidates = self.offload_scheduler.select_offload_candidates(self.adapters)
            
            for adapter_name in candidates:
                if adapter_name in self.adapters:
                    self.adapters[adapter_name].offload_to_cpu()
                    
                    # Check if we're below threshold
                    new_total = self.memory_monitor.get_total_memory_usage()
                    if new_total <= self.config.max_gpu_memory_mb * self.config.memory_threshold:
                        break
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report."""
        total_adapters = len(self.adapters)
        total_memory = 0
        offloaded_count = 0
        
        adapter_stats = {}
        for name, adapter in self.adapters.items():
            stats = adapter.get_memory_usage()
            adapter_stats[name] = stats
            total_memory += stats['parameter_memory_mb']
            
            if adapter.param_locations.get('lora_A') == 'cpu':
                offloaded_count += 1
        
        return {
            'total_adapters': total_adapters,
            'offloaded_adapters': offloaded_count,
            'total_memory_mb': total_memory,
            'gpu_memory_mb': self.memory_monitor.get_gpu_memory_usage(),
            'memory_efficiency': self.memory_monitor.get_memory_efficiency(),
            'adapter_breakdown': adapter_stats
        }


class MemoryMonitor:
    """Monitors memory usage across LoRA adapters."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.adapters: Dict[str, MemoryEfficientLoRA] = {}
        self.memory_history = []
    
    def register_adapter(self, name: str, adapter: MemoryEfficientLoRA):
        """Register an adapter for monitoring."""
        self.adapters[name] = adapter
    
    def unregister_adapter(self, name: str):
        """Unregister an adapter."""
        if name in self.adapters:
            del self.adapters[name]
    
    def get_total_memory_usage(self) -> float:
        """Get total memory usage across all adapters (MB)."""
        total = 0.0
        for adapter in self.adapters.values():
            stats = adapter.get_memory_usage()
            total += stats['parameter_memory_mb']
        return total
    
    def get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage (MB)."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def get_memory_efficiency(self) -> float:
        """Calculate memory efficiency score."""
        if not torch.cuda.is_available():
            return 1.0
        
        used_memory = self.get_gpu_memory_usage()
        available_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        
        if available_memory == 0:
            return 0.0
        
        return 1.0 - (used_memory / available_memory)


class OffloadScheduler:
    """Schedules parameter offloading to optimize memory usage."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.access_counts = defaultdict(int)
        self.last_access = defaultdict(float)
    
    def record_access(self, adapter_name: str):
        """Record access to an adapter."""
        import time
        self.access_counts[adapter_name] += 1
        self.last_access[adapter_name] = time.time()
    
    def select_offload_candidates(
        self,
        adapters: Dict[str, MemoryEfficientLoRA]
    ) -> List[str]:
        """Select adapters for CPU offloading based on usage patterns."""
        import time
        current_time = time.time()
        
        # Score adapters for offloading (lower score = better candidate)
        scores = {}
        for name in adapters.keys():
            access_count = self.access_counts.get(name, 0)
            last_access_time = self.last_access.get(name, 0)
            time_since_access = current_time - last_access_time
            
            # Score = (1 / access_frequency) + time_since_access
            frequency_score = 1.0 / (access_count + 1)
            recency_score = time_since_access / 3600  # Hours since last access
            
            scores[name] = frequency_score + recency_score
        
        # Sort by score (highest first = best candidates for offloading)
        candidates = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        
        # Return top candidates (limit to avoid offloading too many)
        return candidates[:len(candidates) // 2]


# Utility functions for memory management
def estimate_memory_requirement(
    model_size: str,
    num_adapters: int,
    avg_rank: int = 16
) -> Dict[str, float]:
    """Estimate memory requirements for LoRA setup."""
    # Model size mapping (parameters in billions)
    size_mapping = {
        "7b": 7e9,
        "13b": 13e9,
        "30b": 30e9,
        "70b": 70e9
    }
    
    if model_size.lower() not in size_mapping:
        raise ValueError(f"Unknown model size: {model_size}")
    
    base_params = size_mapping[model_size.lower()]
    
    # Estimate LoRA parameters (assuming ~20% of layers get LoRA)
    lora_fraction = 0.2
    avg_layer_size = 4096  # Typical hidden dimension
    lora_params_per_adapter = 2 * avg_rank * avg_layer_size * lora_fraction
    total_lora_params = num_adapters * lora_params_per_adapter
    
    # Memory calculations (4 bytes per FP32 parameter)
    base_memory_gb = base_params * 4 / 1024**3
    lora_memory_gb = total_lora_params * 4 / 1024**3
    
    # Add overhead for gradients, optimizer states, activations
    gradient_memory_gb = lora_memory_gb  # Gradients same size as parameters
    optimizer_memory_gb = lora_memory_gb * 2  # Adam requires ~2x parameter memory
    activation_memory_gb = base_memory_gb * 0.3  # Rough estimate
    
    total_memory_gb = (
        base_memory_gb + lora_memory_gb + gradient_memory_gb + 
        optimizer_memory_gb + activation_memory_gb
    )
    
    return {
        "base_model_gb": base_memory_gb,
        "lora_adapters_gb": lora_memory_gb,
        "gradients_gb": gradient_memory_gb,
        "optimizer_states_gb": optimizer_memory_gb,
        "activations_gb": activation_memory_gb,
        "total_estimate_gb": total_memory_gb,
        "with_offloading_gb": base_memory_gb + lora_memory_gb + activation_memory_gb
    }


def create_memory_efficient_config(
    available_memory_gb: float,
    target_efficiency: float = 0.8
) -> MemoryConfig:
    """Create a memory configuration based on available hardware."""
    max_memory_mb = available_memory_gb * 1024 * target_efficiency
    
    # Determine optimization strategies based on available memory
    if available_memory_gb < 8:
        # Very constrained environment
        return MemoryConfig(
            use_gradient_checkpointing=True,
            use_cpu_offload=True,
            use_activation_checkpointing=True,
            max_gpu_memory_mb=max_memory_mb,
            memory_threshold=0.9,
            enable_quantization=True,
            quantization_bits=4,
            offload_optimizer_states=True,
            offload_gradients=True,
            auto_garbage_collection=True,
            memory_check_interval=50
        )
    elif available_memory_gb < 16:
        # Moderately constrained
        return MemoryConfig(
            use_gradient_checkpointing=True,
            use_cpu_offload=True,
            use_activation_checkpointing=False,
            max_gpu_memory_mb=max_memory_mb,
            memory_threshold=0.85,
            enable_quantization=True,
            quantization_bits=8,
            offload_optimizer_states=True,
            offload_gradients=False,
            auto_garbage_collection=True,
            memory_check_interval=100
        )
    else:
        # Ample memory
        return MemoryConfig(
            use_gradient_checkpointing=False,
            use_cpu_offload=False,
            use_activation_checkpointing=False,
            max_gpu_memory_mb=max_memory_mb,
            memory_threshold=0.8,
            enable_quantization=False,
            quantization_bits=16,
            offload_optimizer_states=False,
            offload_gradients=False,
            auto_garbage_collection=False,
            memory_check_interval=1000
        )