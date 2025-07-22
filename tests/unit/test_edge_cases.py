"""Edge case testing for BLT components.

This module tests handling of edge cases including emojis, special characters,
invalid inputs, and extreme conditions.
"""

import pytest
import torch
import numpy as np
from typing import List, Optional
import struct

from shvayambhu.core.blt.pipeline import BLTPipeline
from shvayambhu.core.blt.patching import ByteProcessor, DynamicPatcher, BLTInputProcessor
from shvayambhu.core.blt.entropy import calculate_byte_entropy


class TestEdgeCases:
    """Comprehensive edge case testing."""
    
    @pytest.fixture
    def pipeline(self):
        """Create BLT pipeline for testing."""
        config = {
            "vocab_size": 256,
            "hidden_dim": 512,
            "num_layers": 4,
            "num_heads": 8,
            "patch_embedding_dim": 512,
            "max_patch_size": 32,
            "device": torch.device("cpu")
        }
        return BLTPipeline(config)
    
    @pytest.fixture
    def byte_processor(self):
        """Create byte processor."""
        return ByteProcessor()
    
    @pytest.fixture
    def patcher(self):
        """Create dynamic patcher."""
        return DynamicPatcher(min_patch_size=4, max_patch_size=32)
    
    def test_empty_input(self, pipeline, byte_processor):
        """Test handling of empty input."""
        # Empty string
        with pytest.raises(ValueError):
            pipeline.process_bytes(b"", mode='inference')
        
        # Empty after processing
        assert byte_processor.process_text("") == b""
    
    def test_single_byte_input(self, pipeline):
        """Test single byte inputs."""
        single_bytes = [
            b"a",
            b"\x00",  # Null byte
            b"\xff",  # Max byte value
            "🦊".encode('utf-8')[:1],  # Partial emoji (invalid)
        ]
        
        for byte_seq in single_bytes:
            if len(byte_seq) == 1 and byte_seq != b"\x00":
                try:
                    output = pipeline.process_bytes(byte_seq, mode='inference')
                    # Should handle gracefully
                    assert 'reconstructed_text' in output or 'error' in output
                except Exception as e:
                    # Should raise meaningful error for invalid UTF-8
                    assert "UTF-8" in str(e) or "invalid" in str(e).lower()
    
    def test_very_long_input(self, pipeline):
        """Test very long inputs."""
        # Create very long text
        long_text = "a" * 100000  # 100k characters
        byte_seq = long_text.encode('utf-8')
        
        # Should handle without memory explosion
        output = pipeline.process_bytes(byte_seq, mode='inference')
        assert output['reconstructed_text'] == long_text
    
    def test_repetitive_patterns(self, pipeline):
        """Test highly repetitive patterns."""
        patterns = [
            "a" * 1000,  # Single character
            "ab" * 500,  # Two character pattern
            "abc" * 333,  # Three character pattern
            "test " * 200,  # Word pattern
            "\n" * 100,  # Newlines
            "\t" * 100,  # Tabs
        ]
        
        for pattern in patterns:
            byte_seq = pattern.encode('utf-8')
            output = pipeline.process_bytes(byte_seq, mode='inference')
            assert output['reconstructed_text'] == pattern
    
    def test_special_unicode_characters(self, pipeline):
        """Test special Unicode characters."""
        special_chars = [
            # Emoji sequences
            "👨‍👩‍👧‍👦",  # Family with ZWJ
            "🏴󠁧󠁢󠁳󠁣󠁴󠁿",  # Flag with tag characters
            "👋🏻👋🏼👋🏽👋🏾👋🏿",  # Skin tone modifiers
            
            # Special spaces
            "Hello World",  # Regular space
            "Hello World",  # Non-breaking space
            "Hello　World",  # Full-width space
            "Hello World",  # Em space
            
            # Control characters
            "Hello\x00World",  # Null
            "Hello\x07World",  # Bell
            "Hello\x1bWorld",  # Escape
            
            # Direction marks
            "Hello\u200eWorld",  # Left-to-right mark
            "Hello\u200fWorld",  # Right-to-left mark
            
            # Mathematical symbols
            "∀x∈ℝ: x²≥0",
            "∑ᵢ₌₁ⁿ i = n(n+1)/2",
            
            # Box drawing
            "┌─┬─┐\n├─┼─┤\n└─┴─┘",
        ]
        
        for text in special_chars:
            byte_seq = text.encode('utf-8')
            output = pipeline.process_bytes(byte_seq, mode='inference')
            
            # Some control characters might be normalized
            if not any(ord(c) < 32 for c in text):
                assert output['reconstructed_text'] == text
    
    def test_invalid_utf8_sequences(self, byte_processor):
        """Test handling of invalid UTF-8 sequences."""
        invalid_sequences = [
            b'\x80',  # Invalid start byte
            b'\xc0\x80',  # Overlong encoding
            b'\xf5\x80\x80\x80',  # Out of range
            b'\xed\xa0\x80',  # Surrogate half
            b'\xc2',  # Incomplete sequence
            b'\xe0\x80',  # Incomplete 3-byte sequence
            b'\xf0\x80\x80',  # Incomplete 4-byte sequence
        ]
        
        for seq in invalid_sequences:
            with pytest.raises(ValueError):
                byte_processor.process_text(seq)
    
    def test_mixed_valid_invalid_bytes(self, pipeline):
        """Test mixed valid and invalid byte sequences."""
        # Valid UTF-8 with invalid bytes in middle
        mixed = b"Hello" + b'\xff\xfe' + b"World"
        
        # Should either handle gracefully or raise error
        try:
            output = pipeline.process_bytes(mixed, mode='inference')
            # If it processes, check it handles the invalid bytes somehow
            assert 'reconstructed_text' in output
        except ValueError as e:
            assert "UTF-8" in str(e) or "invalid" in str(e).lower()
    
    def test_extreme_entropy_cases(self, patcher):
        """Test extreme entropy cases."""
        # Zero entropy (all same byte)
        zero_entropy = b'\x00' * 100
        patches = patcher.create_patches(zero_entropy)
        assert len(patches) < 10  # Should create large patches
        
        # Maximum entropy (all different bytes)
        max_entropy = bytes(range(256))
        patches = patcher.create_patches(max_entropy)
        assert len(patches) > 10  # Should create smaller patches
        
        # Sudden entropy change
        mixed_entropy = b'a' * 50 + bytes(range(50)) + b'b' * 50
        patches = patcher.create_patches(mixed_entropy)
        boundaries = patcher.get_patch_boundaries()
        
        # Should have different patch sizes in different regions
        first_patch_size = boundaries[0][1] - boundaries[0][0]
        middle_patch_idx = len(patches) // 2
        middle_patch_size = boundaries[middle_patch_idx][1] - boundaries[middle_patch_idx][0]
        
        assert first_patch_size != middle_patch_size
    
    def test_boundary_conditions(self, patcher):
        """Test boundary conditions for patching."""
        # Exactly minimum patch size
        min_size_data = b'abcd'  # 4 bytes
        patches = patcher.create_patches(min_size_data)
        assert len(patches) == 1
        
        # Just over minimum
        over_min = b'abcde'  # 5 bytes
        patches = patcher.create_patches(over_min)
        assert len(patches) in [1, 2]  # Depends on entropy
        
        # Exactly maximum patch size
        max_size_data = b'a' * 32
        patches = patcher.create_patches(max_size_data)
        assert all(len(p) <= 32 for p in patches)
    
    def test_null_bytes_handling(self, pipeline):
        """Test handling of null bytes in text."""
        texts_with_nulls = [
            "Hello\x00World",
            "\x00Start",
            "End\x00",
            "Multiple\x00Null\x00Bytes\x00",
            "Binary\x00\x01\x02\x03\x04Data",
        ]
        
        for text in texts_with_nulls:
            byte_seq = text.encode('utf-8')
            output = pipeline.process_bytes(byte_seq, mode='inference')
            assert output['reconstructed_text'] == text
    
    def test_whitespace_variations(self, pipeline):
        """Test various whitespace characters."""
        whitespace_texts = [
            "Space Separated",
            "Tab\tSeparated",
            "Newline\nSeparated",
            "Carriage\rReturn",
            "Mixed \t\n\r Whitespace",
            "Trailing spaces    ",
            "    Leading spaces",
            "Multiple   Spaces",
            "\n\n\nMultiple newlines",
        ]
        
        for text in whitespace_texts:
            byte_seq = text.encode('utf-8')
            output = pipeline.process_bytes(byte_seq, mode='inference')
            assert output['reconstructed_text'] == text
    
    def test_language_edge_cases(self, pipeline):
        """Test edge cases in various languages."""
        edge_cases = [
            # Arabic with diacritics
            "مُحَمَّد",
            
            # Hebrew with vowel points
            "שָׁלוֹם",
            
            # Devanagari with complex conjuncts
            "क्ष्म्य",
            
            # Thai with tone marks
            "น้ำ",
            
            # Zalgo text
            "H̸̡̪̯ͨ͊́̂̓̀͗͏̮̮̦̠̖ͅĘ̷̬̩̣̆̍̑̿̉ͦ͐ͬ͢L̓",
            
            # Vertical text marker
            "縦書き",
        ]
        
        for text in edge_cases:
            byte_seq = text.encode('utf-8')
            output = pipeline.process_bytes(byte_seq, mode='inference')
            # May not preserve exactly due to normalization
            assert len(output['reconstructed_text']) > 0
    
    def test_binary_data_rejection(self, byte_processor):
        """Test that pure binary data is handled appropriately."""
        # Create binary data that's definitely not text
        binary_data = struct.pack('>IIHH', 0xDEADBEEF, 0xCAFEBABE, 0x1234, 0x5678)
        
        # Should reject or handle specially
        with pytest.raises(ValueError):
            byte_processor.process_text(binary_data)
    
    def test_memory_stress(self, pipeline):
        """Test behavior under memory stress."""
        # Create large input that might stress memory
        large_input = ("Large text chunk " * 1000 + "\n") * 100
        byte_seq = large_input.encode('utf-8')
        
        # Should complete without crashing
        output = pipeline.process_bytes(byte_seq, mode='inference')
        assert 'reconstructed_text' in output
    
    def test_concurrent_unicode_normalization(self, pipeline):
        """Test different Unicode normalization forms."""
        import unicodedata
        
        test_string = "é"  # Can be one or two codepoints
        
        forms = {
            'NFC': unicodedata.normalize('NFC', test_string),
            'NFD': unicodedata.normalize('NFD', test_string),
            'NFKC': unicodedata.normalize('NFKC', test_string),
            'NFKD': unicodedata.normalize('NFKD', test_string),
        }
        
        for form_name, normalized in forms.items():
            byte_seq = normalized.encode('utf-8')
            output = pipeline.process_bytes(byte_seq, mode='inference')
            # Should preserve the normalization form
            assert output['reconstructed_text'] == normalized
    
    def test_performance_edge_cases(self, patcher):
        """Test performance-related edge cases."""
        # Adversarial pattern that might cause poor performance
        adversarial = b""
        for i in range(100):
            # Alternating high and low entropy
            if i % 2 == 0:
                adversarial += b'a' * 10
            else:
                adversarial += bytes(np.random.randint(0, 256, 10))
        
        # Should complete in reasonable time
        import time
        start = time.time()
        patches = patcher.create_patches(adversarial)
        elapsed = time.time() - start
        
        assert elapsed < 1.0  # Should complete within 1 second
        assert len(patches) > 0
    
    def test_surrogate_pair_edge_cases(self, pipeline):
        """Test surrogate pair edge cases."""
        # Valid surrogate pairs (in UTF-16, converted to UTF-8)
        surrogate_texts = [
            "𝕳𝖊𝖑𝖑𝖔",  # Mathematical bold
            "🏳️‍🌈",  # Rainbow flag
            "👨‍👨‍👧‍👧",  # Complex emoji family
        ]
        
        for text in surrogate_texts:
            byte_seq = text.encode('utf-8')
            output = pipeline.process_bytes(byte_seq, mode='inference')
            assert output['reconstructed_text'] == text


class TestInputValidation:
    """Test input validation across components."""
    
    def test_none_input_handling(self):
        """Test None input handling."""
        processor = ByteProcessor()
        
        with pytest.raises(TypeError):
            processor.process_text(None)
        
        with pytest.raises(AttributeError):
            processor.is_valid_utf8(None)
    
    def test_wrong_type_handling(self):
        """Test wrong type handling."""
        processor = ByteProcessor()
        
        # Numbers
        with pytest.raises(TypeError):
            processor.process_text(12345)
        
        # Lists
        with pytest.raises(TypeError):
            processor.process_text(['hello', 'world'])
        
        # Dicts
        with pytest.raises(TypeError):
            processor.process_text({'text': 'hello'})
    
    def test_extreme_lengths(self):
        """Test extreme length inputs."""
        patcher = DynamicPatcher(min_patch_size=4, max_patch_size=32)
        
        # Very short (less than min patch size)
        short = b"ab"
        patches = patcher.create_patches(short)
        assert len(patches) == 1
        assert patches[0] == short
        
        # Exactly at boundaries
        exact_min = b"abcd"
        patches = patcher.create_patches(exact_min)
        assert len(patches) == 1
        
        # Very long single patch request
        patcher_large = DynamicPatcher(min_patch_size=4, max_patch_size=10000)
        long_data = b'a' * 50000
        patches = patcher_large.create_patches(long_data)
        assert all(len(p) <= 10000 for p in patches)


def run_edge_case_tests():
    """Run comprehensive edge case tests."""
    print("Running Edge Case Tests...")
    print("=" * 60)
    
    # Run specific test scenarios
    test = TestEdgeCases()
    pipeline = BLTPipeline({
        "vocab_size": 256,
        "hidden_dim": 512,
        "num_layers": 4,
        "num_heads": 8,
        "patch_embedding_dim": 512,
        "max_patch_size": 32,
        "device": torch.device("cpu")
    })
    
    # Test each category
    print("\n--- Testing Special Characters ---")
    test.test_special_unicode_characters(pipeline)
    
    print("\n--- Testing Extreme Cases ---")
    test.test_very_long_input(pipeline)
    
    print("\n--- Testing Invalid Inputs ---")
    processor = ByteProcessor()
    test.test_invalid_utf8_sequences(processor)
    
    print("\n" + "=" * 60)
    print("Edge case tests completed!")


if __name__ == "__main__":
    run_edge_case_tests()