"""LoRA merging system for SEAL architecture.

This module provides capabilities for merging LoRA adapters with base models
and combining multiple LoRA adapters efficiently.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import warnings

from .lora_adapter import LoRALinear, LoRAEmbedding, LoRAConfig
from .parameter_diff import ParameterDiff


@dataclass
class MergingConfig:
    """Configuration for LoRA merging operations."""
    # Merging strategies
    merge_strategy: str = "weighted_sum"  # "weighted_sum", "svd_merge", "selective_merge"
    
    # Weights and scaling
    base_weight: float = 1.0
    adapter_weights: Dict[str, float] = field(default_factory=dict)
    auto_scale_adapters: bool = True
    
    # Quality control
    quality_threshold: float = 0.95  # Minimum quality after merge
    max_rank_reduction: float = 0.5  # Maximum rank reduction during merge
    
    # Memory optimization
    enable_quantization: bool = False
    quantization_bits: int = 8
    use_gradient_checkpointing: bool = False
    
    # Validation settings
    validate_merge: bool = True
    validation_samples: int = 100


@dataclass
class MergeResult:
    """Result of a LoRA merge operation."""
    success: bool
    merged_parameters: Dict[str, torch.Tensor]
    quality_score: float
    compression_ratio: float
    
    # Merge statistics
    original_param_count: int = 0
    merged_param_count: int = 0
    memory_saved_mb: float = 0.0
    
    # Quality metrics
    reconstruction_error: float = 0.0
    validation_loss_change: float = 0.0
    
    # Warnings and notes
    warnings: List[str] = field(default_factory=list)
    merge_details: Dict[str, Any] = field(default_factory=dict)


class LoRAMerger:
    """Merges LoRA adapters with base models and other adapters."""
    
    def __init__(self, config: MergingConfig):
        self.config = config
        self.merge_history: List[MergeResult] = []
    
    def merge_adapter_to_base(
        self,
        base_model: nn.Module,
        lora_adapters: Dict[str, Union[LoRALinear, LoRAEmbedding]],
        adapter_weight: float = 1.0
    ) -> MergeResult:
        """Merge LoRA adapters back into base model parameters."""
        merged_parameters = {}
        original_param_count = 0
        merged_param_count = 0
        total_reconstruction_error = 0.0
        warnings_list = []
        
        for name, adapter in lora_adapters.items():
            # Find corresponding parameter in base model
            base_param = None
            for param_name, param in base_model.named_parameters():
                if param_name == name or param_name.endswith(f".{name}"):
                    base_param = param
                    break
            
            if base_param is None:
                warnings_list.append(f"Could not find base parameter for adapter {name}")
                continue
            
            # Merge adapter with base parameter
            if isinstance(adapter, LoRALinear):
                merged_param, error = self._merge_linear_adapter(
                    base_param, adapter, adapter_weight
                )
            elif isinstance(adapter, LoRAEmbedding):
                merged_param, error = self._merge_embedding_adapter(
                    base_param, adapter, adapter_weight
                )
            else:
                warnings_list.append(f"Unknown adapter type for {name}")
                continue
            
            merged_parameters[name] = merged_param
            original_param_count += base_param.numel() + adapter.get_parameter_count()
            merged_param_count += merged_param.numel()
            total_reconstruction_error += error
        
        # Calculate quality metrics
        quality_score = self._calculate_merge_quality(
            lora_adapters, merged_parameters
        )
        
        compression_ratio = (
            1.0 - (merged_param_count / original_param_count)
            if original_param_count > 0 else 0.0
        )
        
        # Memory calculation
        memory_saved_mb = (original_param_count - merged_param_count) * 4 / 1024 / 1024
        
        result = MergeResult(
            success=len(merged_parameters) > 0,
            merged_parameters=merged_parameters,
            quality_score=quality_score,
            compression_ratio=compression_ratio,
            original_param_count=original_param_count,
            merged_param_count=merged_param_count,
            memory_saved_mb=memory_saved_mb,
            reconstruction_error=total_reconstruction_error,
            warnings=warnings_list,
            merge_details={
                "adapter_count": len(lora_adapters),
                "adapter_weight": adapter_weight,
                "merge_strategy": "base_merge"
            }
        )
        
        self.merge_history.append(result)
        return result
    
    def _merge_linear_adapter(
        self,
        base_param: torch.Tensor,
        adapter: LoRALinear,
        weight: float
    ) -> Tuple[torch.Tensor, float]:
        """Merge a LoRA linear adapter with base parameter."""
        # Compute LoRA contribution: weight * A @ B
        lora_delta = weight * torch.mm(adapter.lora_A, adapter.lora_B)
        
        # Scale by LoRA scaling factor
        if hasattr(adapter, 'scaling'):
            lora_delta = lora_delta * adapter.scaling
        
        # Merge with base parameter
        merged_param = base_param.data + lora_delta
        
        # Calculate reconstruction error
        reconstruction_error = torch.norm(lora_delta).item()
        
        return merged_param, reconstruction_error
    
    def _merge_embedding_adapter(
        self,
        base_param: torch.Tensor,
        adapter: LoRAEmbedding,
        weight: float
    ) -> Tuple[torch.Tensor, float]:
        """Merge a LoRA embedding adapter with base parameter."""
        # Similar to linear but for embedding layers
        lora_delta = weight * torch.mm(adapter.lora_A, adapter.lora_B)
        
        if hasattr(adapter, 'scaling'):
            lora_delta = lora_delta * adapter.scaling
        
        merged_param = base_param.data + lora_delta
        reconstruction_error = torch.norm(lora_delta).item()
        
        return merged_param, reconstruction_error
    
    def merge_multiple_adapters(
        self,
        base_model: nn.Module,
        adapter_groups: Dict[str, Dict[str, Union[LoRALinear, LoRAEmbedding]]],
        group_weights: Optional[Dict[str, float]] = None
    ) -> MergeResult:
        """Merge multiple groups of LoRA adapters."""
        if not group_weights:
            group_weights = {group: 1.0 / len(adapter_groups) 
                           for group in adapter_groups}
        
        # Normalize weights
        total_weight = sum(group_weights.values())
        group_weights = {k: v / total_weight for k, v in group_weights.items()}
        
        merged_parameters = {}
        original_param_count = 0
        merged_param_count = 0
        total_reconstruction_error = 0.0
        warnings_list = []
        
        # Find all parameter names across groups
        all_param_names = set()
        for adapters in adapter_groups.values():
            all_param_names.update(adapters.keys())
        
        for param_name in all_param_names:
            # Find base parameter
            base_param = None
            for name, param in base_model.named_parameters():
                if name == param_name or name.endswith(f".{param_name}"):
                    base_param = param
                    break
            
            if base_param is None:
                warnings_list.append(f"Could not find base parameter {param_name}")
                continue
            
            # Accumulate contributions from all groups
            total_delta = torch.zeros_like(base_param.data)
            param_error = 0.0
            
            for group_name, adapters in adapter_groups.items():
                if param_name in adapters:
                    adapter = adapters[param_name]
                    weight = group_weights[group_name]
                    
                    if isinstance(adapter, LoRALinear):
                        delta = weight * torch.mm(adapter.lora_A, adapter.lora_B)
                    elif isinstance(adapter, LoRAEmbedding):
                        delta = weight * torch.mm(adapter.lora_A, adapter.lora_B)
                    else:
                        continue
                    
                    if hasattr(adapter, 'scaling'):
                        delta = delta * adapter.scaling
                    
                    total_delta += delta
                    param_error += torch.norm(delta).item()
                    original_param_count += adapter.get_parameter_count()
            
            # Merge with base parameter
            merged_param = base_param.data + total_delta
            merged_parameters[param_name] = merged_param
            merged_param_count += merged_param.numel()
            total_reconstruction_error += param_error
            original_param_count += base_param.numel()
        
        # Calculate quality and compression
        quality_score = self._calculate_multi_merge_quality(
            adapter_groups, merged_parameters
        )
        
        compression_ratio = (
            1.0 - (merged_param_count / original_param_count)
            if original_param_count > 0 else 0.0
        )
        
        memory_saved_mb = (original_param_count - merged_param_count) * 4 / 1024 / 1024
        
        result = MergeResult(
            success=len(merged_parameters) > 0,
            merged_parameters=merged_parameters,
            quality_score=quality_score,
            compression_ratio=compression_ratio,
            original_param_count=original_param_count,
            merged_param_count=merged_param_count,
            memory_saved_mb=memory_saved_mb,
            reconstruction_error=total_reconstruction_error,
            warnings=warnings_list,
            merge_details={
                "group_count": len(adapter_groups),
                "group_weights": group_weights,
                "merge_strategy": "multi_merge"
            }
        )
        
        self.merge_history.append(result)
        return result
    
    def create_merged_model(
        self,
        base_model: nn.Module,
        merge_result: MergeResult
    ) -> nn.Module:
        """Create a new model with merged parameters."""
        # Clone the base model
        merged_model = type(base_model)()  # Create new instance
        merged_model.load_state_dict(base_model.state_dict())
        
        # Update parameters with merged values
        with torch.no_grad():
            for param_name, merged_param in merge_result.merged_parameters.items():
                # Find the parameter in the model
                param_found = False
                for name, param in merged_model.named_parameters():
                    if name == param_name or name.endswith(f".{param_name}"):
                        param.data = merged_param.clone()
                        param_found = True
                        break
                
                if not param_found:
                    warnings.warn(f"Could not update parameter {param_name} in merged model")
        
        return merged_model
    
    def selective_merge(
        self,
        base_model: nn.Module,
        lora_adapters: Dict[str, Union[LoRALinear, LoRAEmbedding]],
        importance_scores: Dict[str, float],
        merge_threshold: float = 0.5
    ) -> MergeResult:
        """Selectively merge only important LoRA adapters."""
        # Filter adapters by importance
        selected_adapters = {
            name: adapter for name, adapter in lora_adapters.items()
            if importance_scores.get(name, 0.0) >= merge_threshold
        }
        
        if not selected_adapters:
            return MergeResult(
                success=False,
                merged_parameters={},
                quality_score=0.0,
                compression_ratio=0.0,
                warnings=["No adapters met the importance threshold"]
            )
        
        # Merge selected adapters with importance-weighted scaling
        merged_parameters = {}
        original_param_count = 0
        merged_param_count = 0
        total_reconstruction_error = 0.0
        warnings_list = []
        
        for name, adapter in selected_adapters.items():
            # Find corresponding base parameter
            base_param = None
            for param_name, param in base_model.named_parameters():
                if param_name == name or param_name.endswith(f".{name}"):
                    base_param = param
                    break
            
            if base_param is None:
                warnings_list.append(f"Could not find base parameter for {name}")
                continue
            
            # Weight by importance score
            importance_weight = importance_scores[name]
            
            if isinstance(adapter, LoRALinear):
                merged_param, error = self._merge_linear_adapter(
                    base_param, adapter, importance_weight
                )
            elif isinstance(adapter, LoRAEmbedding):
                merged_param, error = self._merge_embedding_adapter(
                    base_param, adapter, importance_weight
                )
            else:
                continue
            
            merged_parameters[name] = merged_param
            original_param_count += base_param.numel() + adapter.get_parameter_count()
            merged_param_count += merged_param.numel()
            total_reconstruction_error += error
        
        quality_score = self._calculate_merge_quality(selected_adapters, merged_parameters)
        compression_ratio = (
            1.0 - (merged_param_count / original_param_count)
            if original_param_count > 0 else 0.0
        )
        
        memory_saved_mb = (original_param_count - merged_param_count) * 4 / 1024 / 1024
        
        result = MergeResult(
            success=len(merged_parameters) > 0,
            merged_parameters=merged_parameters,
            quality_score=quality_score,
            compression_ratio=compression_ratio,
            original_param_count=original_param_count,
            merged_param_count=merged_param_count,
            memory_saved_mb=memory_saved_mb,
            reconstruction_error=total_reconstruction_error,
            warnings=warnings_list,
            merge_details={
                "selected_adapters": len(selected_adapters),
                "total_adapters": len(lora_adapters),
                "merge_threshold": merge_threshold,
                "importance_scores": importance_scores,
                "merge_strategy": "selective_merge"
            }
        )
        
        self.merge_history.append(result)
        return result
    
    def svd_merge(
        self,
        base_model: nn.Module,
        lora_adapters: Dict[str, Union[LoRALinear, LoRAEmbedding]],
        target_rank: Optional[int] = None
    ) -> MergeResult:
        """Merge adapters using SVD for optimal rank reduction."""
        merged_parameters = {}
        original_param_count = 0
        merged_param_count = 0
        total_reconstruction_error = 0.0
        warnings_list = []
        
        for name, adapter in lora_adapters.items():
            # Find corresponding base parameter
            base_param = None
            for param_name, param in base_model.named_parameters():
                if param_name == name or param_name.endswith(f".{name}"):
                    base_param = param
                    break
            
            if base_param is None:
                warnings_list.append(f"Could not find base parameter for {name}")
                continue
            
            # Compute full LoRA delta
            if isinstance(adapter, (LoRALinear, LoRAEmbedding)):
                lora_delta = torch.mm(adapter.lora_A, adapter.lora_B)
                if hasattr(adapter, 'scaling'):
                    lora_delta = lora_delta * adapter.scaling
            else:
                continue
            
            # Combine with base parameter
            full_param = base_param.data + lora_delta
            
            # Apply SVD if target rank is specified
            if target_rank and target_rank < min(full_param.shape):
                try:
                    U, S, Vh = torch.svd(full_param)
                    # Keep only top components
                    U_reduced = U[:, :target_rank]
                    S_reduced = S[:target_rank]
                    Vh_reduced = Vh[:target_rank, :]
                    
                    # Reconstruct with reduced rank
                    merged_param = torch.mm(
                        U_reduced * S_reduced.unsqueeze(0), Vh_reduced
                    )
                    
                    # Calculate reconstruction error
                    error = torch.norm(full_param - merged_param).item()
                    
                except Exception as e:
                    warnings_list.append(f"SVD failed for {name}: {str(e)}")
                    merged_param = full_param
                    error = 0.0
            else:
                merged_param = full_param
                error = torch.norm(lora_delta).item()
            
            merged_parameters[name] = merged_param
            original_param_count += base_param.numel() + adapter.get_parameter_count()
            merged_param_count += merged_param.numel()
            total_reconstruction_error += error
        
        quality_score = self._calculate_merge_quality(lora_adapters, merged_parameters)
        compression_ratio = (
            1.0 - (merged_param_count / original_param_count)
            if original_param_count > 0 else 0.0
        )
        
        memory_saved_mb = (original_param_count - merged_param_count) * 4 / 1024 / 1024
        
        result = MergeResult(
            success=len(merged_parameters) > 0,
            merged_parameters=merged_parameters,
            quality_score=quality_score,
            compression_ratio=compression_ratio,
            original_param_count=original_param_count,
            merged_param_count=merged_param_count,
            memory_saved_mb=memory_saved_mb,
            reconstruction_error=total_reconstruction_error,
            warnings=warnings_list,
            merge_details={
                "target_rank": target_rank,
                "merge_strategy": "svd_merge"
            }
        )
        
        self.merge_history.append(result)
        return result
    
    def _calculate_merge_quality(
        self,
        adapters: Dict[str, Union[LoRALinear, LoRAEmbedding]],
        merged_parameters: Dict[str, torch.Tensor]
    ) -> float:
        """Calculate quality score for a merge operation."""
        if not adapters or not merged_parameters:
            return 0.0
        
        total_score = 0.0
        param_count = 0
        
        for name, adapter in adapters.items():
            if name in merged_parameters:
                # Simple quality metric based on parameter stability
                merged_param = merged_parameters[name]
                param_norm = torch.norm(merged_param).item()
                
                # Check for NaN or inf values
                if torch.isnan(merged_param).any() or torch.isinf(merged_param).any():
                    continue
                
                # Normalize by parameter magnitude
                if param_norm > 0:
                    stability_score = 1.0 / (1.0 + param_norm / merged_param.numel())
                    total_score += stability_score
                    param_count += 1
        
        return total_score / param_count if param_count > 0 else 0.0
    
    def _calculate_multi_merge_quality(
        self,
        adapter_groups: Dict[str, Dict[str, Union[LoRALinear, LoRAEmbedding]]],
        merged_parameters: Dict[str, torch.Tensor]
    ) -> float:
        """Calculate quality score for multi-adapter merge."""
        # Flatten all adapters
        all_adapters = {}
        for adapters in adapter_groups.values():
            all_adapters.update(adapters)
        
        return self._calculate_merge_quality(all_adapters, merged_parameters)
    
    def get_merge_statistics(self) -> Dict[str, Any]:
        """Get statistics about merge operations."""
        if not self.merge_history:
            return {}
        
        stats = {
            "total_merges": len(self.merge_history),
            "successful_merges": sum(1 for r in self.merge_history if r.success),
            "average_quality": np.mean([r.quality_score for r in self.merge_history]),
            "average_compression": np.mean([r.compression_ratio for r in self.merge_history]),
            "total_memory_saved_mb": sum(r.memory_saved_mb for r in self.merge_history),
            "merge_strategies": defaultdict(int)
        }
        
        for result in self.merge_history:
            strategy = result.merge_details.get("merge_strategy", "unknown")
            stats["merge_strategies"][strategy] += 1
        
        stats["merge_strategies"] = dict(stats["merge_strategies"])
        
        return stats
    
    def optimize_merge_weights(
        self,
        base_model: nn.Module,
        adapter_groups: Dict[str, Dict[str, Union[LoRALinear, LoRAEmbedding]]],
        validation_data: Optional[Any] = None,
        optimization_steps: int = 10
    ) -> Dict[str, float]:
        """Optimize merge weights for best performance."""
        best_weights = {group: 1.0 / len(adapter_groups) for group in adapter_groups}
        best_score = 0.0
        
        for step in range(optimization_steps):
            # Generate candidate weights
            candidate_weights = {}
            for group in adapter_groups:
                # Add some noise to current best weights
                noise = np.random.normal(0, 0.1)
                weight = max(0.0, best_weights[group] + noise)
                candidate_weights[group] = weight
            
            # Normalize weights
            total_weight = sum(candidate_weights.values())
            if total_weight > 0:
                candidate_weights = {k: v / total_weight for k, v in candidate_weights.items()}
            
            # Test merge with candidate weights
            try:
                merge_result = self.merge_multiple_adapters(
                    base_model, adapter_groups, candidate_weights
                )
                
                # Simple scoring function (could be improved with actual validation)
                score = merge_result.quality_score * (1.0 + merge_result.compression_ratio)
                
                if score > best_score:
                    best_score = score
                    best_weights = candidate_weights.copy()
                    
            except Exception as e:
                warnings.warn(f"Weight optimization step {step} failed: {str(e)}")
                continue
        
        return best_weights