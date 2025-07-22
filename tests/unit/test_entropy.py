"""Unit tests for entropy calculation module."""

import pytest
import numpy as np
from shvayambhu.core.blt.entropy import (
    calculate_byte_entropy,
    determine_patch_boundaries,
    adaptive_entropy_patching,
    EntropyCalculator
)


class TestByteEntropy:
    """Test byte entropy calculation."""
    
    def test_empty_sequence(self):
        """Test that empty sequence returns zero entropy."""
        assert calculate_byte_entropy(b"") == 0.0
    
    def test_single_byte(self):
        """Test that single repeated byte has zero entropy."""
        assert calculate_byte_entropy(b"a" * 100) == 0.0
    
    def test_uniform_distribution(self):
        """Test that uniform distribution has maximum entropy."""
        # All bytes 0-255 equally represented
        uniform_bytes = bytes(range(256)) * 4
        entropy = calculate_byte_entropy(uniform_bytes)
        assert 7.9 < entropy < 8.0  # Should be close to 8 bits
    
    def test_binary_distribution(self):
        """Test entropy of binary distribution."""
        # Equal mix of two values
        binary_bytes = b"01" * 100
        entropy = calculate_byte_entropy(binary_bytes)
        assert 0.9 < entropy < 1.1  # Should be close to 1 bit
    
    def test_increasing_entropy(self):
        """Test that more diverse sequences have higher entropy."""
        seq1 = b"a" * 100
        seq2 = b"ab" * 50
        seq3 = b"abc" * 33
        seq4 = b"abcd" * 25
        
        e1 = calculate_byte_entropy(seq1)
        e2 = calculate_byte_entropy(seq2)
        e3 = calculate_byte_entropy(seq3)
        e4 = calculate_byte_entropy(seq4)
        
        assert e1 < e2 < e3 < e4


class TestPatchBoundaries:
    """Test patch boundary determination."""
    
    def test_uniform_boundaries(self):
        """Test boundaries on uniform data."""
        # Uniform data should result in maximum patch sizes
        uniform_data = b"a" * 100
        boundaries = determine_patch_boundaries(
            uniform_data,
            target_patch_size=16,
            min_patch_size=4,
            max_patch_size=32
        )
        
        # Should have fewer, larger patches
        assert len(boundaries) <= 10
        
        # Verify boundaries are valid
        for start, end in boundaries:
            assert 0 <= start < end <= len(uniform_data)
    
    def test_high_entropy_boundaries(self):
        """Test boundaries on high-entropy data."""
        # Random data should result in smaller patches
        np.random.seed(42)
        random_data = bytes(np.random.randint(0, 256, 100))
        
        boundaries = determine_patch_boundaries(
            random_data,
            target_patch_size=16,
            min_patch_size=4,
            max_patch_size=32
        )
        
        # Should have more, smaller patches
        assert len(boundaries) >= 5
    
    def test_boundary_constraints(self):
        """Test that boundaries respect min/max constraints."""
        data = b"Hello, World!" * 10
        boundaries = determine_patch_boundaries(
            data,
            target_patch_size=8,
            min_patch_size=4,
            max_patch_size=16
        )
        
        for start, end in boundaries:
            patch_size = end - start
            assert 4 <= patch_size <= 16
    
    def test_complete_coverage(self):
        """Test that boundaries cover entire sequence."""
        data = b"Test sequence for boundary coverage"
        boundaries = determine_patch_boundaries(
            data,
            target_patch_size=8,
            min_patch_size=4,
            max_patch_size=16
        )
        
        # First boundary should start at 0
        assert boundaries[0][0] == 0
        
        # Last boundary should end at sequence length
        assert boundaries[-1][1] == len(data)
        
        # No gaps between boundaries
        for i in range(len(boundaries) - 1):
            assert boundaries[i][1] == boundaries[i + 1][0]


class TestAdaptivePatching:
    """Test adaptive entropy-based patching."""
    
    def test_adaptive_patching_basic(self):
        """Test basic adaptive patching functionality."""
        text = "Hello world! This is a test. " * 5
        patches, boundaries, entropies = adaptive_entropy_patching(
            text.encode('utf-8')
        )
        
        assert len(patches) == len(boundaries) == len(entropies)
        assert all(isinstance(p, bytes) for p in patches)
        assert all(0 <= e <= 8 for e in entropies)
    
    def test_mixed_content_patching(self):
        """Test patching on mixed entropy content."""
        # Mix of repetitive and varied content
        mixed = b"aaaaaaa" + b"Hello World!" + b"bbbbbbb" + b"1234567890"
        
        patches, boundaries, entropies = adaptive_entropy_patching(mixed)
        
        # Repetitive sections should have lower entropy
        assert entropies[0] < 1.0  # "aaaaaaa" section
        
        # Varied sections should have higher entropy
        varied_indices = [i for i, (s, e) in enumerate(boundaries) 
                         if b"Hello" in mixed[s:e] or b"1234" in mixed[s:e]]
        for idx in varied_indices:
            assert entropies[idx] > 1.0


class TestEntropyCalculator:
    """Test EntropyCalculator class."""
    
    def test_calculator_initialization(self):
        """Test calculator initialization."""
        calc = EntropyCalculator(window_size=256)
        assert calc.window_size == 256
        assert calc.cache is not None
    
    def test_calculator_caching(self):
        """Test that calculator caches results."""
        calc = EntropyCalculator()
        data = b"Test data for caching"
        
        # First call
        result1 = calc.calculate_entropy(data)
        cache_size_1 = len(calc.cache)
        
        # Second call with same data
        result2 = calc.calculate_entropy(data)
        cache_size_2 = len(calc.cache)
        
        assert result1 == result2
        assert cache_size_2 == cache_size_1  # Cache size unchanged
    
    def test_sliding_window_entropy(self):
        """Test sliding window entropy calculation."""
        calc = EntropyCalculator(window_size=10)
        
        # Create data with changing entropy
        data = b"a" * 20 + b"abcdefghij" * 2 + b"z" * 20
        
        entropies = calc.calculate_sliding_window_entropy(data)
        
        # Should have entropy value for each position
        assert len(entropies) == len(data)
        
        # Beginning and end (uniform) should have low entropy
        assert np.mean(entropies[:10]) < 0.5
        assert np.mean(entropies[-10:]) < 0.5
        
        # Middle (varied) should have higher entropy
        assert np.mean(entropies[20:40]) > 2.0
    
    def test_find_boundaries(self):
        """Test boundary finding in calculator."""
        calc = EntropyCalculator()
        
        # Create data with clear entropy transitions
        data = b"a" * 50 + b"Hello, World!" + b"b" * 50
        
        boundaries = calc.find_entropy_boundaries(
            data,
            threshold=1.0,
            min_distance=10
        )
        
        # Should detect transitions around the varied middle section
        assert len(boundaries) >= 2
        
        # Boundaries should be around positions 50 and 63
        assert any(45 <= b <= 55 for b in boundaries)
        assert any(58 <= b <= 68 for b in boundaries)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_unicode_handling(self):
        """Test handling of various Unicode sequences."""
        # Test various Unicode scenarios
        test_cases = [
            "Hello 世界 🌍",  # Mixed scripts with emoji
            "Привет мир",     # Cyrillic
            "مرحبا بالعالم",  # Arabic (RTL)
            "🎉🎊🎈",        # Only emojis
        ]
        
        for text in test_cases:
            byte_seq = text.encode('utf-8')
            entropy = calculate_byte_entropy(byte_seq)
            assert 0 <= entropy <= 8
            
            # Should successfully create patches
            patches, boundaries, _ = adaptive_entropy_patching(byte_seq)
            assert len(patches) > 0
    
    def test_large_sequences(self):
        """Test handling of large byte sequences."""
        # Create a large sequence with varying patterns
        large_seq = b""
        for i in range(100):
            if i % 3 == 0:
                large_seq += b"a" * 100
            elif i % 3 == 1:
                large_seq += bytes(range(256))
            else:
                large_seq += b"Hello World! " * 10
        
        # Should handle large sequences efficiently
        calc = EntropyCalculator()
        entropy = calc.calculate_entropy(large_seq)
        assert 0 <= entropy <= 8
        
        # Patching should work
        patches, boundaries, _ = adaptive_entropy_patching(
            large_seq,
            target_patch_size=64,
            max_patch_size=128
        )
        assert len(patches) > 50  # Should create many patches
    
    def test_pathological_cases(self):
        """Test pathological input cases."""
        # All zeros
        assert calculate_byte_entropy(bytes(1000)) == 0.0
        
        # All 255s
        assert calculate_byte_entropy(bytes([255] * 1000)) == 0.0
        
        # Alternating extremes
        alternating = bytes([0, 255] * 500)
        entropy = calculate_byte_entropy(alternating)
        assert 0.9 < entropy < 1.1  # Binary entropy
    
    def test_boundary_edge_cases(self):
        """Test edge cases in boundary determination."""
        # Very short sequence
        short = b"Hi"
        boundaries = determine_patch_boundaries(
            short,
            target_patch_size=16,
            min_patch_size=4,
            max_patch_size=32
        )
        assert boundaries == [(0, 2)]
        
        # Sequence shorter than min_patch_size
        tiny = b"X"
        boundaries = determine_patch_boundaries(
            tiny,
            target_patch_size=16,
            min_patch_size=4,
            max_patch_size=32
        )
        assert boundaries == [(0, 1)]  # Should still create one patch