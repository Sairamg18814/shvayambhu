"""Parameter diff system for SEAL architecture.

This module provides functionality to compute, analyze, and apply
parameter differences for self-editing operations.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import json
from pathlib import Path
import time

from .lora_adapter import LoRALinear, LoRAEmbedding


@dataclass
class ParameterDiff:
    """Container for parameter differences."""
    name: str
    shape: Tuple[int, ...]
    diff_type: str  # 'full', 'lora', 'sparse'
    
    # Full parameter diff
    full_diff: Optional[torch.Tensor] = None
    
    # LoRA diff
    lora_A: Optional[torch.Tensor] = None
    lora_B: Optional[torch.Tensor] = None
    lora_rank: Optional[int] = None
    
    # Sparse diff (indices and values)
    sparse_indices: Optional[torch.Tensor] = None
    sparse_values: Optional[torch.Tensor] = None
    sparsity_ratio: Optional[float] = None
    
    # Metadata
    magnitude: float = 0.0
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"


@dataclass
class DiffConfig:
    """Configuration for parameter diff computation."""
    # Diff types to try (in order of preference)
    diff_types: List[str] = field(default_factory=lambda: ['lora', 'sparse', 'full'])
    
    # LoRA settings
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_threshold: float = 0.8  # Quality threshold for LoRA approximation
    
    # Sparse settings
    sparsity_threshold: float = 0.01  # Values below this are zeroed
    max_sparsity: float = 0.9  # Maximum allowed sparsity
    
    # General settings
    magnitude_threshold: float = 1e-6  # Minimum magnitude to consider
    compression_ratio_threshold: float = 0.5  # Minimum compression ratio


class ParameterDiffSystem:
    """System for computing and managing parameter differences."""
    
    def __init__(self, config: DiffConfig):
        self.config = config
        self.diff_history: List[ParameterDiff] = []
        self.cached_decompositions: Dict[str, Any] = {}
    
    def compute_diff(
        self,
        original_params: Dict[str, torch.Tensor],
        updated_params: Dict[str, torch.Tensor],
        param_name: str
    ) -> ParameterDiff:
        """Compute the most efficient diff for a parameter."""
        if param_name not in original_params or param_name not in updated_params:
            raise ValueError(f"Parameter {param_name} not found in both parameter sets")
        
        orig_param = original_params[param_name]
        updated_param = updated_params[param_name]
        
        if orig_param.shape != updated_param.shape:
            raise ValueError(f"Parameter shapes don't match: {orig_param.shape} vs {updated_param.shape}")
        
        # Compute raw difference
        raw_diff = updated_param - orig_param
        magnitude = torch.norm(raw_diff).item()
        
        if magnitude < self.config.magnitude_threshold:
            # No significant change
            return ParameterDiff(
                name=param_name,
                shape=orig_param.shape,
                diff_type='none',
                magnitude=magnitude
            )
        
        # Try different diff types
        best_diff = None
        best_compression_ratio = 0.0
        
        for diff_type in self.config.diff_types:
            if diff_type == 'lora':
                diff = self._compute_lora_diff(raw_diff, param_name)
            elif diff_type == 'sparse':
                diff = self._compute_sparse_diff(raw_diff, param_name)
            elif diff_type == 'full':
                diff = self._compute_full_diff(raw_diff, param_name)
            else:
                continue
            
            # Calculate compression ratio
            compression_ratio = self._calculate_compression_ratio(diff, orig_param.numel())
            
            if (compression_ratio > best_compression_ratio and 
                compression_ratio >= self.config.compression_ratio_threshold):
                best_diff = diff
                best_compression_ratio = compression_ratio
        
        # Fall back to full diff if no compression achieved
        if best_diff is None:
            best_diff = self._compute_full_diff(raw_diff, param_name)
        
        best_diff.magnitude = magnitude
        self.diff_history.append(best_diff)
        
        return best_diff
    
    def _compute_lora_diff(
        self,
        diff: torch.Tensor,
        param_name: str
    ) -> ParameterDiff:
        """Compute LoRA approximation of parameter diff."""
        # Only applicable to 2D tensors (Linear layers)
        if len(diff.shape) != 2:
            return ParameterDiff(
                name=param_name,
                shape=diff.shape,
                diff_type='invalid',
                magnitude=torch.norm(diff).item()
            )
        
        # Perform SVD
        try:
            U, S, Vh = torch.svd(diff)
            
            # Determine optimal rank
            rank = min(self.config.lora_rank, min(diff.shape))
            
            # Check quality of approximation
            total_energy = torch.sum(S**2)
            kept_energy = torch.sum(S[:rank]**2)
            quality = (kept_energy / total_energy).item()
            
            if quality < self.config.lora_threshold:
                return ParameterDiff(
                    name=param_name,
                    shape=diff.shape,
                    diff_type='low_quality',
                    magnitude=torch.norm(diff).item()
                )
            
            # Create LoRA matrices
            lora_A = U[:, :rank] * torch.sqrt(S[:rank])
            lora_B = torch.sqrt(S[:rank]).unsqueeze(0) * Vh[:rank, :]
            
            return ParameterDiff(
                name=param_name,
                shape=diff.shape,
                diff_type='lora',
                lora_A=lora_A,
                lora_B=lora_B,
                lora_rank=rank,
                confidence=quality
            )
        
        except Exception:
            return ParameterDiff(
                name=param_name,
                shape=diff.shape,
                diff_type='svd_failed',
                magnitude=torch.norm(diff).item()
            )
    
    def _compute_sparse_diff(
        self,
        diff: torch.Tensor,
        param_name: str
    ) -> ParameterDiff:
        """Compute sparse representation of parameter diff."""
        # Apply threshold
        magnitude = torch.abs(diff)
        threshold = self.config.sparsity_threshold * torch.max(magnitude)
        
        # Create mask
        mask = magnitude > threshold
        sparsity_ratio = 1.0 - (mask.sum().float() / mask.numel()).item()
        
        if sparsity_ratio < self.config.max_sparsity:
            return ParameterDiff(
                name=param_name,
                shape=diff.shape,
                diff_type='not_sparse',
                magnitude=torch.norm(diff).item()
            )
        
        # Extract sparse representation
        indices = torch.nonzero(mask, as_tuple=False)
        values = diff[mask]
        
        return ParameterDiff(
            name=param_name,
            shape=diff.shape,
            diff_type='sparse',
            sparse_indices=indices,
            sparse_values=values,
            sparsity_ratio=sparsity_ratio
        )
    
    def _compute_full_diff(
        self,
        diff: torch.Tensor,
        param_name: str
    ) -> ParameterDiff:
        """Store full parameter diff."""
        return ParameterDiff(
            name=param_name,
            shape=diff.shape,
            diff_type='full',
            full_diff=diff.clone()
        )
    
    def _calculate_compression_ratio(
        self,
        diff: ParameterDiff,
        original_params: int
    ) -> float:
        """Calculate compression ratio for a diff."""
        if diff.diff_type == 'lora' and diff.lora_A is not None and diff.lora_B is not None:
            compressed_params = diff.lora_A.numel() + diff.lora_B.numel()
            return 1.0 - (compressed_params / original_params)
        
        elif diff.diff_type == 'sparse' and diff.sparse_values is not None:
            compressed_params = diff.sparse_values.numel() + diff.sparse_indices.numel()
            return 1.0 - (compressed_params / original_params)
        
        elif diff.diff_type == 'full':
            return 0.0  # No compression
        
        else:
            return 0.0
    
    def apply_diff(
        self,
        original_param: torch.Tensor,
        diff: ParameterDiff
    ) -> torch.Tensor:
        """Apply a parameter diff to get updated parameters."""
        if diff.name and original_param.shape != diff.shape:
            raise ValueError(f"Shape mismatch: {original_param.shape} vs {diff.shape}")
        
        if diff.diff_type == 'lora':
            # Apply LoRA diff: param + A @ B
            lora_diff = torch.mm(diff.lora_A, diff.lora_B)
            return original_param + lora_diff
        
        elif diff.diff_type == 'sparse':
            # Apply sparse diff
            updated_param = original_param.clone()
            if diff.sparse_indices.dim() == 2:
                # Multi-dimensional indices
                updated_param[diff.sparse_indices[:, 0], diff.sparse_indices[:, 1]] += diff.sparse_values
            else:
                # 1D indices
                updated_param.view(-1)[diff.sparse_indices] += diff.sparse_values
            return updated_param
        
        elif diff.diff_type == 'full':
            return original_param + diff.full_diff
        
        else:
            # No change or invalid diff
            return original_param
    
    def merge_diffs(
        self,
        diffs: List[ParameterDiff],
        param_name: str
    ) -> ParameterDiff:
        """Merge multiple diffs for the same parameter."""
        if not diffs:
            raise ValueError("No diffs to merge")
        
        if len(diffs) == 1:
            return diffs[0]
        
        # Check compatibility
        base_shape = diffs[0].shape
        for diff in diffs[1:]:
            if diff.shape != base_shape:
                raise ValueError("Cannot merge diffs with different shapes")
        
        # Convert all to full diffs for merging
        accumulated_diff = torch.zeros(base_shape)
        
        for diff in diffs:
            if diff.diff_type == 'lora':
                diff_tensor = torch.mm(diff.lora_A, diff.lora_B)
            elif diff.diff_type == 'sparse':
                diff_tensor = torch.zeros(base_shape)
                if diff.sparse_indices.dim() == 2:
                    diff_tensor[diff.sparse_indices[:, 0], diff.sparse_indices[:, 1]] = diff.sparse_values
                else:
                    diff_tensor.view(-1)[diff.sparse_indices] = diff.sparse_values
            elif diff.diff_type == 'full':
                diff_tensor = diff.full_diff
            else:
                continue  # Skip invalid diffs
            
            accumulated_diff += diff_tensor
        
        # Create merged diff (try to compress again)
        fake_original = torch.zeros(base_shape)
        fake_updated = accumulated_diff
        
        return self.compute_diff(
            {param_name: fake_original},
            {param_name: fake_updated},
            param_name
        )
    
    def get_diff_statistics(self) -> Dict[str, Any]:
        """Get statistics about computed diffs."""
        if not self.diff_history:
            return {}
        
        stats = {
            'total_diffs': len(self.diff_history),
            'diff_types': defaultdict(int),
            'avg_magnitude': 0.0,
            'avg_confidence': 0.0,
            'compression_ratios': []
        }
        
        total_magnitude = 0.0
        total_confidence = 0.0
        
        for diff in self.diff_history:
            stats['diff_types'][diff.diff_type] += 1
            total_magnitude += diff.magnitude
            if diff.confidence > 0:
                total_confidence += diff.confidence
        
        stats['avg_magnitude'] = total_magnitude / len(self.diff_history)
        stats['avg_confidence'] = total_confidence / max(1, sum(1 for d in self.diff_history if d.confidence > 0))
        
        return dict(stats)
    
    def save_diff(self, diff: ParameterDiff, filepath: Path):
        """Save a parameter diff to disk."""
        diff_data = {
            'name': diff.name,
            'shape': list(diff.shape),
            'diff_type': diff.diff_type,
            'magnitude': diff.magnitude,
            'confidence': diff.confidence,
            'timestamp': diff.timestamp,
            'source': diff.source
        }
        
        # Save tensors separately
        tensors = {}
        if diff.full_diff is not None:
            tensors['full_diff'] = diff.full_diff
        if diff.lora_A is not None:
            tensors['lora_A'] = diff.lora_A
            tensors['lora_B'] = diff.lora_B
            diff_data['lora_rank'] = diff.lora_rank
        if diff.sparse_indices is not None:
            tensors['sparse_indices'] = diff.sparse_indices
            tensors['sparse_values'] = diff.sparse_values
            diff_data['sparsity_ratio'] = diff.sparsity_ratio
        
        # Save metadata as JSON
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(diff_data, f, indent=2)
        
        # Save tensors as PyTorch file
        if tensors:
            torch.save(tensors, filepath.with_suffix('.pt'))
    
    def load_diff(self, filepath: Path) -> ParameterDiff:
        """Load a parameter diff from disk."""
        # Load metadata
        with open(filepath.with_suffix('.json'), 'r') as f:
            diff_data = json.load(f)
        
        # Create base diff object
        diff = ParameterDiff(
            name=diff_data['name'],
            shape=tuple(diff_data['shape']),
            diff_type=diff_data['diff_type'],
            magnitude=diff_data['magnitude'],
            confidence=diff_data['confidence'],
            timestamp=diff_data['timestamp'],
            source=diff_data['source']
        )
        
        # Load tensors if they exist
        tensor_file = filepath.with_suffix('.pt')
        if tensor_file.exists():
            tensors = torch.load(tensor_file)
            
            if 'full_diff' in tensors:
                diff.full_diff = tensors['full_diff']
            
            if 'lora_A' in tensors:
                diff.lora_A = tensors['lora_A']
                diff.lora_B = tensors['lora_B']
                diff.lora_rank = diff_data.get('lora_rank')
            
            if 'sparse_indices' in tensors:
                diff.sparse_indices = tensors['sparse_indices']
                diff.sparse_values = tensors['sparse_values']
                diff.sparsity_ratio = diff_data.get('sparsity_ratio')
        
        return diff