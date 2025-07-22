"""Entropy calculation for dynamic patching in BLT.

This module implements entropy-based calculations to determine optimal
patch sizes for byte sequences. High entropy regions (more random/diverse)
get smaller patches for better resolution, while low entropy regions
(repetitive/structured) use larger patches for efficiency.
"""

import math
from typing import Dict, List, Tuple
import numpy as np


def calculate_byte_entropy(
    byte_sequence: bytes, 
    window_size: int = 256
) -> float:
    """Calculate Shannon entropy for a byte sequence.
    
    Entropy is calculated as: H(X) = -Σ p(x) * log2(p(x))
    where p(x) is the probability of byte value x.
    
    Args:
        byte_sequence: Input bytes to analyze
        window_size: Number of bytes to consider (default: 256)
        
    Returns:
        Entropy value between 0 (uniform) and 1 (maximum randomness)
        
    Example:
        >>> entropy = calculate_byte_entropy(b"AAAA")
        >>> assert entropy < 0.1  # Low entropy for repeated bytes
        >>> entropy = calculate_byte_entropy(bytes(range(256)))
        >>> assert entropy > 0.9  # High entropy for uniform distribution
    """
    if len(byte_sequence) == 0:
        return 0.0
    
    # Limit to window size
    sequence = byte_sequence[:window_size]
    
    # Count byte frequencies
    byte_counts: Dict[int, int] = {}
    for byte in sequence:
        byte_counts[byte] = byte_counts.get(byte, 0) + 1
    
    # Calculate probabilities
    total_bytes = len(sequence)
    entropy = 0.0
    
    for count in byte_counts.values():
        if count > 0:
            probability = count / total_bytes
            entropy -= probability * math.log2(probability)
    
    # Normalize to [0, 1] range (max entropy is log2(256) = 8)
    normalized_entropy = entropy / 8.0
    
    return min(1.0, normalized_entropy)


def calculate_sliding_window_entropy(
    byte_sequence: bytes,
    window_size: int = 64,
    stride: int = 16
) -> List[float]:
    """Calculate entropy using sliding windows across the sequence.
    
    This provides local entropy measurements at different positions,
    enabling dynamic patch size decisions based on local complexity.
    
    Args:
        byte_sequence: Input bytes to analyze
        window_size: Size of sliding window
        stride: Step size between windows
        
    Returns:
        List of entropy values for each window position
    """
    if len(byte_sequence) < window_size:
        return [calculate_byte_entropy(byte_sequence)]
    
    entropies = []
    for i in range(0, len(byte_sequence) - window_size + 1, stride):
        window = byte_sequence[i:i + window_size]
        entropy = calculate_byte_entropy(window)
        entropies.append(entropy)
    
    return entropies


def determine_patch_boundaries(
    byte_sequence: bytes,
    min_patch_size: int = 4,
    max_patch_size: int = 32,
    entropy_threshold_low: float = 0.3,
    entropy_threshold_high: float = 0.7
) -> List[int]:
    """Determine optimal patch boundaries based on entropy.
    
    High entropy regions get smaller patches (down to min_patch_size),
    while low entropy regions get larger patches (up to max_patch_size).
    
    Args:
        byte_sequence: Input bytes to patch
        min_patch_size: Minimum allowed patch size
        max_patch_size: Maximum allowed patch size
        entropy_threshold_low: Below this, use larger patches
        entropy_threshold_high: Above this, use smaller patches
        
    Returns:
        List of patch boundary positions
    """
    if len(byte_sequence) <= min_patch_size:
        return [len(byte_sequence)]
    
    boundaries = []
    position = 0
    
    while position < len(byte_sequence):
        # Calculate local entropy
        window_end = min(position + max_patch_size * 2, len(byte_sequence))
        local_sequence = byte_sequence[position:window_end]
        entropy = calculate_byte_entropy(local_sequence, window_size=64)
        
        # Determine patch size based on entropy
        if entropy < entropy_threshold_low:
            # Low entropy: use larger patches
            patch_size = max_patch_size
        elif entropy > entropy_threshold_high:
            # High entropy: use smaller patches
            patch_size = min_patch_size
        else:
            # Medium entropy: interpolate
            ratio = (entropy - entropy_threshold_low) / (entropy_threshold_high - entropy_threshold_low)
            patch_size = int(max_patch_size - ratio * (max_patch_size - min_patch_size))
        
        # Ensure we don't exceed sequence length
        patch_size = min(patch_size, len(byte_sequence) - position)
        
        # Add boundary
        position += patch_size
        boundaries.append(position)
    
    return boundaries


def adaptive_entropy_patching(
    byte_sequence: bytes,
    target_patches: int = 64,
    min_patch_size: int = 4,
    max_patch_size: int = 32
) -> List[Tuple[int, int]]:
    """Create patches with adaptive sizing based on content entropy.
    
    This algorithm aims to create approximately target_patches patches
    while respecting entropy-based size constraints.
    
    Args:
        byte_sequence: Input bytes to patch
        target_patches: Desired number of patches (approximate)
        min_patch_size: Minimum allowed patch size
        max_patch_size: Maximum allowed patch size
        
    Returns:
        List of (start, end) tuples defining patch boundaries
    """
    if len(byte_sequence) == 0:
        return []
    
    # Calculate average patch size for target
    avg_patch_size = len(byte_sequence) / target_patches
    avg_patch_size = max(min_patch_size, min(max_patch_size, int(avg_patch_size)))
    
    # Calculate entropy profile
    entropies = calculate_sliding_window_entropy(
        byte_sequence, 
        window_size=avg_patch_size,
        stride=max(1, avg_patch_size // 4)
    )
    
    # Smooth entropy values to avoid abrupt changes
    if len(entropies) > 3:
        entropies = np.convolve(entropies, np.ones(3)/3, mode='same').tolist()
    
    patches = []
    position = 0
    entropy_idx = 0
    
    while position < len(byte_sequence):
        # Get current entropy
        if entropy_idx < len(entropies):
            current_entropy = entropies[entropy_idx]
        else:
            current_entropy = entropies[-1] if entropies else 0.5
        
        # Map entropy to patch size (inverse relationship)
        # High entropy -> small patches, Low entropy -> large patches
        entropy_factor = 1.0 - current_entropy
        patch_size = int(min_patch_size + entropy_factor * (max_patch_size - min_patch_size))
        
        # Ensure we don't exceed boundaries
        patch_size = min(patch_size, len(byte_sequence) - position)
        
        # Create patch
        patches.append((position, position + patch_size))
        
        # Update position and entropy index
        position += patch_size
        entropy_idx = min(len(entropies) - 1, int(position / avg_patch_size))
    
    return patches


def compute_patch_statistics(
    byte_sequence: bytes,
    patches: List[Tuple[int, int]]
) -> Dict[str, float]:
    """Compute statistics about the patching result.
    
    Args:
        byte_sequence: Original byte sequence
        patches: List of (start, end) patch boundaries
        
    Returns:
        Dictionary with statistics including average size, entropy, etc.
    """
    if not patches:
        return {
            "num_patches": 0,
            "avg_patch_size": 0.0,
            "min_patch_size": 0,
            "max_patch_size": 0,
            "avg_entropy": 0.0,
            "compression_ratio": 1.0
        }
    
    patch_sizes = [end - start for start, end in patches]
    patch_entropies = [
        calculate_byte_entropy(byte_sequence[start:end])
        for start, end in patches
    ]
    
    return {
        "num_patches": len(patches),
        "avg_patch_size": sum(patch_sizes) / len(patch_sizes),
        "min_patch_size": min(patch_sizes),
        "max_patch_size": max(patch_sizes),
        "avg_entropy": sum(patch_entropies) / len(patch_entropies),
        "compression_ratio": len(patches) / len(byte_sequence)
    }


# Optimized version for production use
class EntropyCalculator:
    """Optimized entropy calculator with caching and vectorization."""
    
    def __init__(self, cache_size: int = 1024):
        """Initialize with optional caching.
        
        Args:
            cache_size: Number of entropy calculations to cache
        """
        self._cache: Dict[bytes, float] = {}
        self._cache_size = cache_size
        self._byte_probs = np.zeros(256)
    
    def calculate(self, byte_sequence: bytes) -> float:
        """Calculate entropy with caching."""
        # Check cache
        if byte_sequence in self._cache:
            return self._cache[byte_sequence]
        
        # Calculate entropy
        entropy = self._calculate_vectorized(byte_sequence)
        
        # Update cache
        if len(self._cache) < self._cache_size:
            self._cache[byte_sequence] = entropy
        
        return entropy
    
    def _calculate_vectorized(self, byte_sequence: bytes) -> float:
        """Vectorized entropy calculation for performance."""
        if len(byte_sequence) == 0:
            return 0.0
        
        # Convert to numpy array for vectorized operations
        byte_array = np.frombuffer(byte_sequence, dtype=np.uint8)
        
        # Count occurrences
        unique, counts = np.unique(byte_array, return_counts=True)
        
        # Calculate probabilities
        probs = counts / len(byte_array)
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log2(probs))
        
        # Normalize
        return min(1.0, entropy / 8.0)