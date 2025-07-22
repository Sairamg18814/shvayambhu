"""Dynamic LoRA rank selection and optimization.

This module provides dynamic rank selection capabilities for LoRA adapters,
optimizing the rank based on data characteristics and performance requirements.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import time

from .lora_adapter import LoRALinear, LoRAEmbedding, LoRAConfig


@dataclass
class RankOptimizationConfig:
    """Configuration for dynamic rank optimization."""
    min_rank: int = 1
    max_rank: int = 128
    target_compression_ratio: float = 0.8  # Target compression vs full rank
    
    # Analysis settings
    svd_analysis_samples: int = 1000
    rank_search_method: str = "binary"  # "linear", "binary", "adaptive"
    quality_threshold: float = 0.95  # Minimum quality retention
    
    # Performance thresholds
    max_inference_overhead: float = 0.1  # 10% overhead
    max_memory_overhead: float = 0.15   # 15% overhead
    
    # Optimization frequency
    reoptimization_interval: int = 1000  # Steps between rank reoptimization
    min_performance_gain: float = 0.02  # Minimum gain to change rank


@dataclass
class RankAnalysis:
    """Analysis results for rank selection."""
    parameter_name: str
    original_shape: Tuple[int, ...]
    
    # SVD analysis
    singular_values: torch.Tensor
    explained_variance_ratio: torch.Tensor
    optimal_rank: int
    quality_at_rank: Dict[int, float] = field(default_factory=dict)
    
    # Performance metrics
    compression_ratios: Dict[int, float] = field(default_factory=dict)
    inference_times: Dict[int, float] = field(default_factory=dict)
    memory_usage: Dict[int, float] = field(default_factory=dict)
    
    # Recommendations
    recommended_rank: int = 16
    confidence: float = 0.0
    reasoning: str = ""


class DynamicRankSelector:
    """Selects optimal LoRA ranks dynamically."""
    
    def __init__(self, config: RankOptimizationConfig):
        self.config = config
        self.rank_cache: Dict[str, RankAnalysis] = {}
        self.performance_history: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self.last_optimization: Dict[str, float] = {}
    
    def analyze_parameter_rank_requirements(
        self,
        parameter: torch.Tensor,
        parameter_name: str,
        data_samples: Optional[torch.Tensor] = None
    ) -> RankAnalysis:
        """Analyze a parameter to determine optimal LoRA rank."""
        if len(parameter.shape) != 2:
            # Only 2D parameters (Linear layers) support LoRA
            return RankAnalysis(
                parameter_name=parameter_name,
                original_shape=parameter.shape,
                singular_values=torch.tensor([]),
                explained_variance_ratio=torch.tensor([]),
                optimal_rank=1,
                recommended_rank=1,
                confidence=0.0,
                reasoning="Non-2D parameter, LoRA not applicable"
            )
        
        # Perform SVD analysis
        try:
            U, S, Vh = torch.svd(parameter)
            
            # Calculate explained variance ratio
            total_variance = torch.sum(S**2)
            cumulative_variance = torch.cumsum(S**2, dim=0)
            explained_variance_ratio = cumulative_variance / total_variance
            
            # Determine quality at different ranks
            quality_at_rank = {}
            compression_ratios = {}
            
            max_rank = min(self.config.max_rank, min(parameter.shape))
            
            for rank in range(self.config.min_rank, max_rank + 1):
                # Quality: explained variance
                quality = explained_variance_ratio[rank - 1].item()
                quality_at_rank[rank] = quality
                
                # Compression ratio
                original_params = parameter.numel()
                lora_params = rank * sum(parameter.shape)
                compression = 1.0 - (lora_params / original_params)
                compression_ratios[rank] = compression
            
            # Find optimal rank based on quality threshold
            optimal_rank = self._find_optimal_rank(
                quality_at_rank, compression_ratios
            )
            
            # Additional analysis with data if available
            if data_samples is not None:
                optimal_rank = self._refine_rank_with_data(
                    parameter, data_samples, optimal_rank, quality_at_rank
                )
            
            analysis = RankAnalysis(
                parameter_name=parameter_name,
                original_shape=parameter.shape,
                singular_values=S,
                explained_variance_ratio=explained_variance_ratio,
                optimal_rank=optimal_rank,
                quality_at_rank=quality_at_rank,
                compression_ratios=compression_ratios,
                recommended_rank=optimal_rank,
                confidence=self._calculate_confidence(S, optimal_rank),
                reasoning=self._generate_reasoning(
                    optimal_rank, quality_at_rank, compression_ratios
                )
            )
            
            # Cache the analysis
            self.rank_cache[parameter_name] = analysis
            
            return analysis
        
        except Exception as e:
            return RankAnalysis(
                parameter_name=parameter_name,
                original_shape=parameter.shape,
                singular_values=torch.tensor([]),
                explained_variance_ratio=torch.tensor([]),
                optimal_rank=16,  # Default fallback
                recommended_rank=16,
                confidence=0.0,
                reasoning=f"SVD analysis failed: {str(e)}"
            )
    
    def _find_optimal_rank(
        self,
        quality_at_rank: Dict[int, float],
        compression_ratios: Dict[int, float]
    ) -> int:
        """Find optimal rank balancing quality and compression."""
        if self.config.rank_search_method == "binary":
            return self._binary_search_rank(quality_at_rank, compression_ratios)
        elif self.config.rank_search_method == "adaptive":
            return self._adaptive_search_rank(quality_at_rank, compression_ratios)
        else:  # linear
            return self._linear_search_rank(quality_at_rank, compression_ratios)
    
    def _binary_search_rank(
        self,
        quality_at_rank: Dict[int, float],
        compression_ratios: Dict[int, float]
    ) -> int:
        """Binary search for optimal rank."""
        ranks = sorted(quality_at_rank.keys())
        left, right = 0, len(ranks) - 1
        optimal_rank = ranks[right]  # Start with max rank
        
        while left <= right:
            mid = (left + right) // 2
            rank = ranks[mid]
            quality = quality_at_rank[rank]
            compression = compression_ratios[rank]
            
            if (quality >= self.config.quality_threshold and 
                compression >= self.config.target_compression_ratio):
                optimal_rank = rank
                right = mid - 1  # Try smaller ranks
            else:
                left = mid + 1   # Need larger ranks
        
        return optimal_rank
    
    def _linear_search_rank(
        self,
        quality_at_rank: Dict[int, float],
        compression_ratios: Dict[int, float]
    ) -> int:
        """Linear search for optimal rank."""
        for rank in sorted(quality_at_rank.keys()):
            quality = quality_at_rank[rank]
            compression = compression_ratios[rank]
            
            if (quality >= self.config.quality_threshold and 
                compression >= self.config.target_compression_ratio):
                return rank
        
        # Fallback to max rank if no suitable rank found
        return max(quality_at_rank.keys())
    
    def _adaptive_search_rank(
        self,
        quality_at_rank: Dict[int, float],
        compression_ratios: Dict[int, float]
    ) -> int:
        """Adaptive search considering multiple criteria."""
        scores = {}
        
        for rank in quality_at_rank.keys():
            quality = quality_at_rank[rank]
            compression = compression_ratios[rank]
            
            # Multi-objective score
            quality_score = min(quality / self.config.quality_threshold, 1.0)
            compression_score = min(compression / self.config.target_compression_ratio, 1.0)
            
            # Penalty for very low ranks (might be unstable)
            rank_penalty = 1.0 if rank >= 4 else 0.5
            
            # Combined score
            score = 0.6 * quality_score + 0.3 * compression_score + 0.1 * rank_penalty
            scores[rank] = score
        
        # Return rank with highest score
        return max(scores.keys(), key=lambda k: scores[k])
    
    def _refine_rank_with_data(
        self,
        parameter: torch.Tensor,
        data_samples: torch.Tensor,
        initial_rank: int,
        quality_at_rank: Dict[int, float]
    ) -> int:
        """Refine rank selection using actual data samples."""
        # Test ranks around the initial estimate
        test_ranks = [
            max(1, initial_rank - 2),
            max(1, initial_rank - 1),
            initial_rank,
            min(self.config.max_rank, initial_rank + 1),
            min(self.config.max_rank, initial_rank + 2)
        ]
        
        performance_scores = {}
        
        for rank in test_ranks:
            if rank in quality_at_rank:
                # Simulate LoRA reconstruction
                U, S, Vh = torch.svd(parameter)
                
                # Create LoRA matrices
                lora_A = U[:, :rank] * torch.sqrt(S[:rank])
                lora_B = torch.sqrt(S[:rank]).unsqueeze(0) * Vh[:rank, :]
                
                # Reconstruct and measure error on data
                reconstructed = torch.mm(lora_A, lora_B)
                reconstruction_error = torch.norm(parameter - reconstructed).item()
                
                # Score combines quality and error
                quality_score = quality_at_rank[rank]
                error_score = 1.0 / (1.0 + reconstruction_error)
                
                performance_scores[rank] = 0.7 * quality_score + 0.3 * error_score
        
        if performance_scores:
            return max(performance_scores.keys(), key=lambda k: performance_scores[k])
        else:
            return initial_rank
    
    def _calculate_confidence(
        self,
        singular_values: torch.Tensor,
        optimal_rank: int
    ) -> float:
        """Calculate confidence in the rank selection."""
        if len(singular_values) == 0 or optimal_rank <= 0:
            return 0.0
        
        # Confidence based on singular value distribution
        if optimal_rank >= len(singular_values):
            return 1.0
        
        # Check how much variance is captured
        total_variance = torch.sum(singular_values**2)
        captured_variance = torch.sum(singular_values[:optimal_rank]**2)
        variance_ratio = (captured_variance / total_variance).item()
        
        # Check singular value decay
        if optimal_rank < len(singular_values):
            sv_ratio = singular_values[optimal_rank].item() / singular_values[0].item()
            decay_confidence = 1.0 - sv_ratio
        else:
            decay_confidence = 1.0
        
        # Combined confidence
        confidence = 0.6 * variance_ratio + 0.4 * decay_confidence
        return confidence
    
    def _generate_reasoning(
        self,
        optimal_rank: int,
        quality_at_rank: Dict[int, float],
        compression_ratios: Dict[int, float]
    ) -> str:
        """Generate human-readable reasoning for rank selection."""
        if optimal_rank not in quality_at_rank:
            return f"Selected rank {optimal_rank} as fallback"
        
        quality = quality_at_rank[optimal_rank]
        compression = compression_ratios[optimal_rank]
        
        reasoning = f"Selected rank {optimal_rank}: "
        reasoning += f"quality={quality:.3f} (target≥{self.config.quality_threshold}), "
        reasoning += f"compression={compression:.3f} (target≥{self.config.target_compression_ratio})"
        
        if quality < self.config.quality_threshold:
            reasoning += " - WARNING: Below quality threshold"
        if compression < self.config.target_compression_ratio:
            reasoning += " - WARNING: Below compression target"
        
        return reasoning
    
    def should_reoptimize_rank(
        self,
        parameter_name: str,
        current_performance: float
    ) -> bool:
        """Check if rank should be reoptimized."""
        current_time = time.time()
        
        # Check time-based reoptimization
        if parameter_name in self.last_optimization:
            time_since_last = current_time - self.last_optimization[parameter_name]
            if time_since_last < self.config.reoptimization_interval:
                return False
        
        # Check performance-based reoptimization
        if parameter_name in self.performance_history:
            history = self.performance_history[parameter_name]
            if len(history) >= 10:  # Need enough history
                recent_avg = np.mean([p for _, p in history[-5:]])
                older_avg = np.mean([p for _, p in history[-10:-5]])
                
                performance_degradation = (older_avg - recent_avg) / older_avg
                if performance_degradation > self.config.min_performance_gain:
                    return True
        
        # Default: reoptimize if enough time has passed
        return parameter_name not in self.last_optimization
    
    def update_performance_history(
        self,
        parameter_name: str,
        rank: int,
        performance: float
    ):
        """Update performance history for a parameter."""
        self.performance_history[parameter_name].append((rank, performance))
        
        # Keep only recent history
        if len(self.performance_history[parameter_name]) > 100:
            self.performance_history[parameter_name] = \
                self.performance_history[parameter_name][-50:]
        
        self.last_optimization[parameter_name] = time.time()
    
    def get_rank_recommendations(
        self,
        model: nn.Module,
        target_parameters: Optional[List[str]] = None
    ) -> Dict[str, RankAnalysis]:
        """Get rank recommendations for model parameters."""
        recommendations = {}
        
        for name, param in model.named_parameters():
            if target_parameters and name not in target_parameters:
                continue
            
            if len(param.shape) == 2:  # Only Linear layers
                analysis = self.analyze_parameter_rank_requirements(param, name)
                recommendations[name] = analysis
        
        return recommendations
    
    def optimize_existing_lora_ranks(
        self,
        lora_layers: Dict[str, Union[LoRALinear, LoRAEmbedding]]
    ) -> Dict[str, int]:
        """Optimize ranks of existing LoRA layers."""
        optimized_ranks = {}
        
        for name, layer in lora_layers.items():
            if isinstance(layer, LoRALinear):
                # Get the original weight
                original_weight = layer.weight + torch.mm(layer.lora_A, layer.lora_B)
                
                # Analyze optimal rank
                analysis = self.analyze_parameter_rank_requirements(
                    original_weight, name
                )
                
                current_rank = layer.lora_A.shape[1]
                optimal_rank = analysis.recommended_rank
                
                if optimal_rank != current_rank:
                    optimized_ranks[name] = optimal_rank
        
        return optimized_ranks
    
    def create_adaptive_lora_config(
        self,
        model: nn.Module,
        base_config: LoRAConfig
    ) -> Dict[str, LoRAConfig]:
        """Create adaptive LoRA configurations for different layers."""
        recommendations = self.get_rank_recommendations(model)
        adaptive_configs = {}
        
        for param_name, analysis in recommendations.items():
            # Create customized config for this parameter
            custom_config = LoRAConfig(
                r=analysis.recommended_rank,
                alpha=base_config.alpha,
                dropout=base_config.dropout,
                bias=base_config.bias
            )
            adaptive_configs[param_name] = custom_config
        
        return adaptive_configs
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get statistics about rank optimization."""
        stats = {
            "total_parameters_analyzed": len(self.rank_cache),
            "average_recommended_rank": 0.0,
            "rank_distribution": defaultdict(int),
            "high_confidence_recommendations": 0,
            "compression_ratios": []
        }
        
        if not self.rank_cache:
            return stats
        
        total_rank = 0
        for analysis in self.rank_cache.values():
            rank = analysis.recommended_rank
            total_rank += rank
            stats["rank_distribution"][rank] += 1
            
            if analysis.confidence > 0.8:
                stats["high_confidence_recommendations"] += 1
            
            if rank in analysis.compression_ratios:
                stats["compression_ratios"].append(
                    analysis.compression_ratios[rank]
                )
        
        stats["average_recommended_rank"] = total_rank / len(self.rank_cache)
        
        if stats["compression_ratios"]:
            stats["average_compression_ratio"] = np.mean(stats["compression_ratios"])
        
        return dict(stats)