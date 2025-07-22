"""Optimization utilities for efficient training.

This module provides gradient accumulation, memory-efficient training,
and advanced optimization techniques.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple, Iterator
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import functools
import warnings


@dataclass
class GradientAccumulatorConfig:
    """Configuration for gradient accumulation."""
    accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    sync_gradients_at_step: bool = True
    use_dynamic_accumulation: bool = False
    target_batch_size: int = 128
    min_accumulation_steps: int = 1
    max_accumulation_steps: int = 32


class GradientAccumulator:
    """Gradient accumulation manager."""
    
    def __init__(self, config: GradientAccumulatorConfig):
        self.config = config
        self.step_count = 0
        self.accumulated_loss = 0.0
        self.gradient_norms = []
        
        # Dynamic accumulation state
        self.current_accumulation_steps = config.accumulation_steps
        self.accumulation_history = []
    
    def should_accumulate(self) -> bool:
        """Check if gradients should be accumulated."""
        return (self.step_count + 1) % self.current_accumulation_steps != 0
    
    def should_sync(self) -> bool:
        """Check if gradients should be synchronized (for distributed training)."""
        if self.config.sync_gradients_at_step:
            return not self.should_accumulate()
        return True
    
    def accumulate_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Accumulate loss with proper scaling."""
        scaled_loss = loss / self.current_accumulation_steps
        self.accumulated_loss += scaled_loss.item()
        return scaled_loss
    
    def step(self) -> Tuple[bool, float]:
        """Increment step counter and return if optimizer should step."""
        self.step_count += 1
        should_step = not self.should_accumulate()
        
        if should_step:
            avg_loss = self.accumulated_loss
            self.accumulated_loss = 0.0
            return True, avg_loss
        
        return False, 0.0
    
    def clip_gradients(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[Any] = None
    ) -> float:
        """Clip gradients and return norm."""
        if scaler is not None:
            scaler.unscale_(optimizer)
        
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.config.max_grad_norm
        )
        
        self.gradient_norms.append(grad_norm.item())
        return grad_norm
    
    def adjust_accumulation_steps(
        self,
        current_batch_size: int,
        memory_usage: float,
        throughput: float
    ):
        """Dynamically adjust accumulation steps based on performance."""
        if not self.config.use_dynamic_accumulation:
            return
        
        # Calculate effective batch size
        effective_batch_size = current_batch_size * self.current_accumulation_steps
        
        # Adjust based on target batch size
        if effective_batch_size < self.config.target_batch_size * 0.8:
            # Increase accumulation
            new_steps = min(
                self.current_accumulation_steps + 1,
                self.config.max_accumulation_steps
            )
        elif effective_batch_size > self.config.target_batch_size * 1.2:
            # Decrease accumulation
            new_steps = max(
                self.current_accumulation_steps - 1,
                self.config.min_accumulation_steps
            )
        else:
            new_steps = self.current_accumulation_steps
        
        # Adjust based on memory usage (if > 90%, increase accumulation)
        if memory_usage > 0.9:
            new_steps = min(
                new_steps + 2,
                self.config.max_accumulation_steps
            )
        
        if new_steps != self.current_accumulation_steps:
            print(f"Adjusting accumulation steps: {self.current_accumulation_steps} -> {new_steps}")
            self.current_accumulation_steps = new_steps
    
    def get_stats(self) -> Dict[str, float]:
        """Get accumulation statistics."""
        if not self.gradient_norms:
            return {}
        
        return {
            "avg_gradient_norm": np.mean(self.gradient_norms),
            "max_gradient_norm": np.max(self.gradient_norms),
            "current_accumulation_steps": self.current_accumulation_steps
        }


class MemoryEfficientTrainer:
    """Memory-efficient training utilities."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradient_checkpointing_enabled = False
        self.cpu_offload_enabled = False
        self.activation_cache = {}
    
    def enable_gradient_checkpointing(self, modules: Optional[List[str]] = None):
        """Enable gradient checkpointing for specific modules."""
        if modules is None:
            # Enable for all transformer blocks
            modules = ["transformer", "encoder", "decoder"]
        
        for name, module in self.model.named_modules():
            if any(target in name for target in modules):
                if hasattr(module, 'gradient_checkpointing'):
                    module.gradient_checkpointing = True
                    print(f"Enabled gradient checkpointing for {name}")
        
        self.gradient_checkpointing_enabled = True
    
    def enable_cpu_offload(self, modules: Optional[List[str]] = None):
        """Enable CPU offloading for specific modules."""
        if modules is None:
            modules = ["lm_head", "output_projection"]
        
        for name, module in self.model.named_modules():
            if any(target in name for target in modules):
                # Move module to CPU and wrap with offload logic
                self._wrap_with_cpu_offload(name, module)
        
        self.cpu_offload_enabled = True
    
    def _wrap_with_cpu_offload(self, name: str, module: nn.Module):
        """Wrap module with CPU offload logic."""
        original_forward = module.forward
        
        @functools.wraps(original_forward)
        def offloaded_forward(*args, **kwargs):
            # Move to GPU for computation
            module.cuda()
            output = original_forward(*args, **kwargs)
            # Move back to CPU
            module.cpu()
            torch.cuda.empty_cache()
            return output
        
        module.forward = offloaded_forward
        print(f"Enabled CPU offload for {name}")
    
    def optimize_memory_usage(self):
        """Optimize memory usage during training."""
        # Clear activation cache
        self.activation_cache.clear()
        
        # Empty CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Run garbage collection
        import gc
        gc.collect()
    
    def compute_gradient_statistics(self) -> Dict[str, float]:
        """Compute gradient statistics for monitoring."""
        stats = defaultdict(list)
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                stats["grad_norms"].append(grad_norm)
                
                # Check for gradient issues
                if torch.isnan(param.grad).any():
                    warnings.warn(f"NaN gradient detected in {name}")
                    stats["nan_gradients"].append(name)
                
                if torch.isinf(param.grad).any():
                    warnings.warn(f"Inf gradient detected in {name}")
                    stats["inf_gradients"].append(name)
        
        return {
            "mean_grad_norm": np.mean(stats["grad_norms"]) if stats["grad_norms"] else 0.0,
            "max_grad_norm": np.max(stats["grad_norms"]) if stats["grad_norms"] else 0.0,
            "min_grad_norm": np.min(stats["grad_norms"]) if stats["grad_norms"] else 0.0,
            "num_nan_gradients": len(stats["nan_gradients"]),
            "num_inf_gradients": len(stats["inf_gradients"])
        }


class AdaptiveOptimizer:
    """Adaptive optimization strategies."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        warmup_steps: int = 10000,
        min_lr: float = 1e-6,
        max_lr: float = 1e-3
    ):
        self.optimizer = optimizer
        self.model = model
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        
        # Adaptive state
        self.loss_history = []
        self.lr_history = []
        self.step = 0
        
        # Layer-wise learning rates
        self.layer_lr_scales = {}
        self._init_layer_lr_scales()
    
    def _init_layer_lr_scales(self):
        """Initialize layer-wise learning rate scales."""
        # Use discriminative fine-tuning
        num_layers = len([n for n, _ in self.model.named_modules() if 'layer' in n])
        
        for i, (name, _) in enumerate(self.model.named_modules()):
            if 'layer' in name:
                # Lower layers get smaller learning rates
                layer_idx = int(name.split('.')[-1]) if name[-1].isdigit() else i
                scale = 0.5 + 0.5 * (layer_idx / max(num_layers, 1))
                self.layer_lr_scales[name] = scale
    
    def step_with_warmup(self, loss: float):
        """Take optimizer step with warmup."""
        self.step += 1
        self.loss_history.append(loss)
        
        # Calculate warmup factor
        if self.step < self.warmup_steps:
            warmup_factor = self.step / self.warmup_steps
        else:
            warmup_factor = 1.0
        
        # Apply warmup to all parameter groups
        for group in self.optimizer.param_groups:
            group['lr'] = group.get('initial_lr', self.max_lr) * warmup_factor
        
        self.lr_history.append(self.optimizer.param_groups[0]['lr'])
        self.optimizer.step()
    
    def adjust_learning_rate(self, metrics: Dict[str, float]):
        """Adjust learning rate based on metrics."""
        # Simple adaptive strategy based on loss plateau
        if len(self.loss_history) > 100:
            recent_losses = self.loss_history[-100:]
            
            # Check for plateau
            loss_std = np.std(recent_losses)
            loss_mean = np.mean(recent_losses)
            
            if loss_std / loss_mean < 0.01:  # Plateau detected
                # Reduce learning rate
                for group in self.optimizer.param_groups:
                    new_lr = max(group['lr'] * 0.5, self.min_lr)
                    if new_lr < group['lr']:
                        print(f"Reducing learning rate: {group['lr']:.6f} -> {new_lr:.6f}")
                        group['lr'] = new_lr
    
    def apply_layer_wise_lr(self):
        """Apply layer-wise learning rates."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Find matching layer scale
                scale = 1.0
                for layer_name, layer_scale in self.layer_lr_scales.items():
                    if layer_name in name:
                        scale = layer_scale
                        break
                
                # Apply scale to parameter's learning rate
                param.lr_scale = scale
    
    def get_effective_lr(self, param_name: str) -> float:
        """Get effective learning rate for a parameter."""
        base_lr = self.optimizer.param_groups[0]['lr']
        
        # Find layer scale
        scale = 1.0
        for layer_name, layer_scale in self.layer_lr_scales.items():
            if layer_name in param_name:
                scale = layer_scale
                break
        
        return base_lr * scale


class LossScaler:
    """Dynamic loss scaling for mixed precision training."""
    
    def __init__(
        self,
        init_scale: float = 2**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        min_scale: float = 1.0,
        max_scale: float = 2**32
    ):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        # Tracking
        self.steps_since_update = 0
        self.found_inf_history = []
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for backward pass."""
        return loss * self.scale
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients before optimizer step."""
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.div_(self.scale)
    
    def update_scale(self, found_inf: bool):
        """Update loss scale based on gradient overflow."""
        self.found_inf_history.append(found_inf)
        
        if found_inf:
            # Decrease scale
            self.scale = max(self.scale * self.backoff_factor, self.min_scale)
            self.steps_since_update = 0
            print(f"Gradient overflow detected, reducing scale to {self.scale}")
        else:
            self.steps_since_update += 1
            
            # Increase scale if stable
            if self.steps_since_update >= self.growth_interval:
                self.scale = min(self.scale * self.growth_factor, self.max_scale)
                self.steps_since_update = 0
                print(f"Increasing scale to {self.scale}")
    
    def check_gradients(self, model: nn.Module) -> bool:
        """Check for inf/nan in gradients."""
        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scaler statistics."""
        return {
            "current_scale": self.scale,
            "steps_since_update": self.steps_since_update,
            "overflow_rate": np.mean(self.found_inf_history[-100:]) if self.found_inf_history else 0.0
        }


def create_optimizer_with_weight_decay(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    no_decay_names: Optional[List[str]] = None
) -> torch.optim.AdamW:
    """Create AdamW optimizer with proper weight decay."""
    if no_decay_names is None:
        no_decay_names = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]
    
    # Separate parameters into weight decay and no weight decay groups
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if any(nd in name for nd in no_decay_names):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
            "name": "decay"
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
            "name": "no_decay"
        }
    ]
    
    # Log parameter counts
    num_decay = sum(p.numel() for p in decay_params)
    num_no_decay = sum(p.numel() for p in no_decay_params)
    print(f"Optimizer groups: decay={num_decay:,}, no_decay={num_no_decay:,}")
    
    return torch.optim.AdamW(
        param_groups,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon
    )