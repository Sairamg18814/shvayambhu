"""Patch size optimization for BLT.

This module implements dynamic patch size optimization based on
content analysis and performance metrics.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import json
import pickle
from pathlib import Path

from .entropy import calculate_byte_entropy, EntropyCalculator
from .patching import DynamicPatcher


@dataclass
class PatchStatistics:
    """Statistics for patch performance."""
    content_type: str  # text, code, mixed, binary
    avg_patch_size: float
    entropy_mean: float
    entropy_std: float
    reconstruction_accuracy: float
    processing_time_ms: float
    memory_usage_mb: float
    compression_ratio: float
    
    def score(self) -> float:
        """Calculate overall performance score."""
        # Higher accuracy is better
        accuracy_score = self.reconstruction_accuracy
        
        # Lower processing time is better
        time_score = 1.0 / (1.0 + self.processing_time_ms / 100)
        
        # Better compression is good
        compression_score = min(1.0, self.compression_ratio / 10)
        
        # Balanced memory usage
        memory_score = 1.0 / (1.0 + self.memory_usage_mb / 100)
        
        # Weighted combination
        return (
            0.4 * accuracy_score +
            0.3 * time_score +
            0.2 * compression_score +
            0.1 * memory_score
        )


@dataclass
class OptimizationConfig:
    """Configuration for patch optimization."""
    # Size constraints
    min_patch_size: int = 4
    max_patch_size: int = 32
    initial_target_size: int = 16
    
    # Optimization parameters
    learning_rate: float = 0.1
    momentum: float = 0.9
    adaptation_window: int = 1000
    
    # Performance targets
    target_accuracy: float = 0.99
    target_compression: float = 5.0
    max_latency_ms: float = 10.0
    
    # Content-specific settings
    content_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "text": {"target_size": 16, "entropy_threshold": 4.0},
        "code": {"target_size": 24, "entropy_threshold": 5.0},
        "mixed": {"target_size": 20, "entropy_threshold": 4.5},
        "binary": {"target_size": 32, "entropy_threshold": 6.0},
    })


class PatchOptimizer:
    """Dynamic patch size optimizer."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.entropy_calc = EntropyCalculator()
        
        # Optimization state
        self.patch_size_params = {
            content_type: {
                "target_size": cfg["target_size"],
                "entropy_threshold": cfg["entropy_threshold"],
                "momentum": 0.0
            }
            for content_type, cfg in self.config.content_configs.items()
        }
        
        # Performance history
        self.performance_history = deque(maxlen=self.config.adaptation_window)
        self.content_type_stats: Dict[str, List[PatchStatistics]] = {
            "text": [], "code": [], "mixed": [], "binary": []
        }
        
        # Adaptive parameters
        self.global_scale = 1.0
        self.entropy_sensitivity = 1.0
        
    def optimize_patch_sizes(
        self,
        byte_sequence: bytes,
        content_type: Optional[str] = None
    ) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
        """Optimize patch sizes for given byte sequence."""
        # Detect content type if not provided
        if content_type is None:
            content_type = self._detect_content_type(byte_sequence)
        
        # Get current parameters for content type
        params = self.patch_size_params[content_type]
        target_size = int(params["target_size"] * self.global_scale)
        entropy_threshold = params["entropy_threshold"]
        
        # Calculate entropy profile
        entropy_profile = self._calculate_entropy_profile(byte_sequence)
        
        # Determine optimal boundaries
        boundaries = self._optimize_boundaries(
            byte_sequence,
            entropy_profile,
            target_size,
            entropy_threshold
        )
        
        # Collect statistics
        stats = self._collect_statistics(
            byte_sequence,
            boundaries,
            entropy_profile,
            content_type
        )
        
        return boundaries, {
            "content_type": content_type,
            "statistics": stats,
            "entropy_profile": entropy_profile,
            "parameters": params.copy()
        }
    
    def _detect_content_type(self, byte_sequence: bytes) -> str:
        """Detect content type from byte sequence."""
        try:
            text = byte_sequence.decode('utf-8')
            
            # Check for code patterns
            code_indicators = [
                'def ', 'class ', 'import ', 'function', 'var ', 'const ',
                '{}', '[]', '()', '=>', '==', '!=', '&&', '||'
            ]
            code_score = sum(1 for ind in code_indicators if ind in text) / len(code_indicators)
            
            # Check for binary patterns
            non_printable = sum(1 for b in byte_sequence if b < 32 or b > 126)
            binary_score = non_printable / len(byte_sequence)
            
            # Entropy-based detection
            entropy = calculate_byte_entropy(byte_sequence)
            
            # Classify
            if binary_score > 0.3:
                return "binary"
            elif code_score > 0.2:
                return "code"
            elif entropy > 5.0:
                return "mixed"
            else:
                return "text"
                
        except UnicodeDecodeError:
            return "binary"
    
    def _calculate_entropy_profile(self, byte_sequence: bytes) -> np.ndarray:
        """Calculate detailed entropy profile."""
        window_sizes = [8, 16, 32, 64]
        profiles = []
        
        for window_size in window_sizes:
            if len(byte_sequence) >= window_size:
                profile = self.entropy_calc.calculate_sliding_window_entropy(
                    byte_sequence,
                    window_size
                )
                profiles.append(profile)
        
        # Combine profiles with different weights
        if profiles:
            # Weight smaller windows more for fine-grained detail
            weights = [2.0, 1.5, 1.0, 0.5][:len(profiles)]
            weighted_profile = np.zeros_like(profiles[0])
            
            for profile, weight in zip(profiles, weights):
                if len(profile) == len(weighted_profile):
                    weighted_profile += weight * profile
                else:
                    # Resample to match size
                    resampled = np.interp(
                        np.linspace(0, len(profile)-1, len(weighted_profile)),
                        np.arange(len(profile)),
                        profile
                    )
                    weighted_profile += weight * resampled
            
            return weighted_profile / sum(weights)
        else:
            # Fallback for very short sequences
            return np.array([calculate_byte_entropy(byte_sequence)])
    
    def _optimize_boundaries(
        self,
        byte_sequence: bytes,
        entropy_profile: np.ndarray,
        target_size: int,
        entropy_threshold: float
    ) -> List[Tuple[int, int]]:
        """Optimize patch boundaries using dynamic programming."""
        n = len(byte_sequence)
        min_size = self.config.min_patch_size
        max_size = self.config.max_patch_size
        
        # Dynamic programming for optimal segmentation
        # dp[i] = (min_cost, prev_boundary)
        dp = [(float('inf'), -1) for _ in range(n + 1)]
        dp[0] = (0, -1)
        
        for i in range(min_size, n + 1):
            for patch_size in range(min_size, min(max_size + 1, i + 1)):
                j = i - patch_size
                
                # Calculate cost for this patch
                patch_entropy = np.mean(entropy_profile[j:i]) if j < len(entropy_profile) else 0
                
                # Cost function considers:
                # 1. Deviation from target size
                size_cost = abs(patch_size - target_size) / target_size
                
                # 2. Entropy variance within patch
                if i - j > 1 and j < len(entropy_profile):
                    entropy_var = np.var(entropy_profile[j:i])
                else:
                    entropy_var = 0
                
                # 3. Boundary placement (prefer boundaries at low entropy)
                if j > 0 and j < len(entropy_profile):
                    boundary_cost = entropy_profile[j] / 8.0
                else:
                    boundary_cost = 0
                
                # 4. Adaptive sizing based on entropy
                if patch_entropy > entropy_threshold:
                    # High entropy: prefer smaller patches
                    size_penalty = max(0, patch_size - target_size * 0.7) / target_size
                else:
                    # Low entropy: prefer larger patches
                    size_penalty = max(0, target_size * 1.3 - patch_size) / target_size
                
                # Combined cost
                cost = (
                    0.3 * size_cost +
                    0.2 * entropy_var +
                    0.2 * boundary_cost +
                    0.3 * size_penalty
                )
                
                total_cost = dp[j][0] + cost
                
                if total_cost < dp[i][0]:
                    dp[i] = (total_cost, j)
        
        # Reconstruct boundaries
        boundaries = []
        i = n
        while i > 0:
            j = dp[i][1]
            if j >= 0:
                boundaries.append((j, i))
                i = j
            else:
                break
        
        boundaries.reverse()
        
        # Ensure we cover the entire sequence
        if not boundaries or boundaries[0][0] > 0:
            boundaries.insert(0, (0, boundaries[0][0] if boundaries else n))
        if boundaries[-1][1] < n:
            boundaries.append((boundaries[-1][1], n))
        
        return boundaries
    
    def _collect_statistics(
        self,
        byte_sequence: bytes,
        boundaries: List[Tuple[int, int]],
        entropy_profile: np.ndarray,
        content_type: str
    ) -> PatchStatistics:
        """Collect performance statistics."""
        # Patch size statistics
        patch_sizes = [end - start for start, end in boundaries]
        avg_patch_size = np.mean(patch_sizes)
        
        # Entropy statistics
        entropy_mean = np.mean(entropy_profile)
        entropy_std = np.std(entropy_profile)
        
        # Compression ratio
        num_patches = len(boundaries)
        original_size = len(byte_sequence)
        compressed_size = num_patches * 4  # Assuming 4 bytes per patch pointer
        compression_ratio = original_size / compressed_size
        
        # Placeholder values for actual performance metrics
        # In practice, these would be measured during encoding/decoding
        reconstruction_accuracy = 0.99  # Would be measured
        processing_time_ms = avg_patch_size * 0.1  # Rough estimate
        memory_usage_mb = (original_size + num_patches * 768 * 4) / (1024 * 1024)
        
        stats = PatchStatistics(
            content_type=content_type,
            avg_patch_size=avg_patch_size,
            entropy_mean=entropy_mean,
            entropy_std=entropy_std,
            reconstruction_accuracy=reconstruction_accuracy,
            processing_time_ms=processing_time_ms,
            memory_usage_mb=memory_usage_mb,
            compression_ratio=compression_ratio
        )
        
        # Update history
        self.performance_history.append(stats)
        self.content_type_stats[content_type].append(stats)
        
        return stats
    
    def update_parameters(self, feedback: Dict[str, Any]):
        """Update optimization parameters based on feedback."""
        content_type = feedback.get("content_type", "text")
        actual_accuracy = feedback.get("accuracy", 1.0)
        actual_latency = feedback.get("latency_ms", 0.0)
        
        params = self.patch_size_params[content_type]
        
        # Calculate gradients
        accuracy_error = self.config.target_accuracy - actual_accuracy
        latency_error = actual_latency - self.config.max_latency_ms
        
        # Update target size
        if accuracy_error > 0.01:  # Accuracy too low
            # Decrease patch size for better granularity
            size_gradient = -self.config.learning_rate * accuracy_error * 10
        elif latency_error > 0:  # Latency too high
            # Increase patch size for faster processing
            size_gradient = self.config.learning_rate * latency_error * 0.5
        else:
            size_gradient = 0
        
        # Apply momentum
        params["momentum"] = (
            self.config.momentum * params["momentum"] +
            (1 - self.config.momentum) * size_gradient
        )
        
        # Update parameters
        params["target_size"] = np.clip(
            params["target_size"] + params["momentum"],
            self.config.min_patch_size,
            self.config.max_patch_size
        )
        
        # Adapt entropy threshold based on performance
        if len(self.content_type_stats[content_type]) > 10:
            recent_stats = self.content_type_stats[content_type][-10:]
            avg_entropy = np.mean([s.entropy_mean for s in recent_stats])
            params["entropy_threshold"] = 0.9 * params["entropy_threshold"] + 0.1 * avg_entropy
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate optimization report."""
        report = {
            "global_scale": self.global_scale,
            "content_type_parameters": self.patch_size_params,
            "performance_summary": {}
        }
        
        # Summarize performance by content type
        for content_type, stats_list in self.content_type_stats.items():
            if stats_list:
                recent_stats = stats_list[-100:]  # Last 100 samples
                
                report["performance_summary"][content_type] = {
                    "avg_patch_size": np.mean([s.avg_patch_size for s in recent_stats]),
                    "avg_accuracy": np.mean([s.reconstruction_accuracy for s in recent_stats]),
                    "avg_latency_ms": np.mean([s.processing_time_ms for s in recent_stats]),
                    "avg_compression": np.mean([s.compression_ratio for s in recent_stats]),
                    "samples": len(recent_stats)
                }
        
        return report
    
    def save_state(self, path: str):
        """Save optimizer state."""
        state = {
            "patch_size_params": self.patch_size_params,
            "global_scale": self.global_scale,
            "entropy_sensitivity": self.entropy_sensitivity,
            "content_type_stats": {
                ct: [s.__dict__ for s in stats]
                for ct, stats in self.content_type_stats.items()
            }
        }
        
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path_obj, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, path: str):
        """Load optimizer state."""
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.patch_size_params = state["patch_size_params"]
        self.global_scale = state["global_scale"]
        self.entropy_sensitivity = state["entropy_sensitivity"]
        
        # Reconstruct statistics
        for ct, stats_dicts in state["content_type_stats"].items():
            self.content_type_stats[ct] = [
                PatchStatistics(**s) for s in stats_dicts
            ]


class AdaptivePatcher:
    """Adaptive patcher using optimization."""
    
    def __init__(
        self,
        optimizer: Optional[PatchOptimizer] = None,
        fallback_patcher: Optional[DynamicPatcher] = None
    ):
        self.optimizer = optimizer or PatchOptimizer()
        self.fallback_patcher = fallback_patcher or DynamicPatcher()
        self.feedback_buffer = []
    
    def create_patches(
        self,
        byte_sequence: bytes,
        content_type: Optional[str] = None
    ) -> Tuple[List[bytes], Dict[str, Any]]:
        """Create optimized patches."""
        # Get optimized boundaries
        boundaries, optimization_info = self.optimizer.optimize_patch_sizes(
            byte_sequence,
            content_type
        )
        
        # Create patches
        patches = []
        for start, end in boundaries:
            patches.append(byte_sequence[start:end])
        
        # Return patches and info
        return patches, {
            "boundaries": boundaries,
            "optimization": optimization_info,
            "num_patches": len(patches),
            "compression_ratio": len(byte_sequence) / len(patches)
        }
    
    def provide_feedback(
        self,
        content_type: str,
        accuracy: float,
        latency_ms: float,
        **kwargs
    ):
        """Provide performance feedback for optimization."""
        feedback = {
            "content_type": content_type,
            "accuracy": accuracy,
            "latency_ms": latency_ms,
            **kwargs
        }
        
        self.feedback_buffer.append(feedback)
        
        # Update optimizer periodically
        if len(self.feedback_buffer) >= 10:
            for fb in self.feedback_buffer:
                self.optimizer.update_parameters(fb)
            self.feedback_buffer.clear()