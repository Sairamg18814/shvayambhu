"""Unit tests for the patching module."""

import pytest
import torch
import numpy as np
from shvayambhu.core.blt.patching import (
    ByteProcessor,
    DynamicPatcher,
    PatchEmbedder,
    BLTInputProcessor
)


class TestByteProcessor:
    """Test ByteProcessor functionality."""
    
    def test_process_string_input(self):
        """Test processing string input."""
        processor = ByteProcessor()
        
        # ASCII string
        result = processor.process_text("Hello, World!")
        assert isinstance(result, bytes)
        assert result == b"Hello, World!"
        
        # Unicode string
        unicode_text = "Hello 世界 🌍"
        result = processor.process_text(unicode_text)
        assert isinstance(result, bytes)
        assert result.decode('utf-8') == unicode_text
    
    def test_process_bytes_input(self):
        """Test processing bytes input."""
        processor = ByteProcessor()
        
        # Valid UTF-8 bytes
        valid_bytes = "Test".encode('utf-8')
        result = processor.process_text(valid_bytes)
        assert result == valid_bytes
        
        # Invalid UTF-8 bytes should raise error
        invalid_bytes = b'\xff\xfe'
        with pytest.raises(ValueError, match="Invalid UTF-8"):
            processor.process_text(invalid_bytes)
    
    def test_is_valid_utf8(self):
        """Test UTF-8 validation."""
        processor = ByteProcessor()
        
        # Valid cases
        assert processor.is_valid_utf8(b"Hello")
        assert processor.is_valid_utf8("世界".encode('utf-8'))
        assert processor.is_valid_utf8("🌍".encode('utf-8'))
        
        # Invalid cases
        assert not processor.is_valid_utf8(b'\xff\xfe')
        assert not processor.is_valid_utf8(b'\xc0\x80')  # Overlong encoding
    
    def test_normalize_bytes(self):
        """Test byte normalization."""
        processor = ByteProcessor()
        
        # Test normalization preserves valid sequences
        test_bytes = b"Hello\x00World"
        normalized = processor.normalize_bytes(test_bytes)
        assert normalized == test_bytes  # Should preserve null bytes
        
        # Test with various byte values
        all_bytes = bytes(range(256))
        normalized = processor.normalize_bytes(all_bytes)
        assert len(normalized) == 256
    
    def test_to_tensor(self):
        """Test conversion to tensor."""
        processor = ByteProcessor()
        
        byte_seq = b"Hello"
        tensor = processor.to_tensor(byte_seq)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.long
        assert tensor.shape == (5,)
        assert tensor.tolist() == [72, 101, 108, 108, 111]  # ASCII values
    
    def test_edge_cases(self):
        """Test edge cases."""
        processor = ByteProcessor()
        
        # Empty input
        assert processor.process_text("") == b""
        assert processor.process_text(b"") == b""
        
        # Very long input
        long_text = "a" * 10000
        result = processor.process_text(long_text)
        assert len(result) == 10000
        
        # Special characters
        special = "\n\t\r\x00"
        result = processor.process_text(special)
        assert result == special.encode('utf-8')


class TestDynamicPatcher:
    """Test DynamicPatcher functionality."""
    
    def test_initialization(self):
        """Test patcher initialization."""
        patcher = DynamicPatcher(
            min_patch_size=4,
            max_patch_size=32,
            target_patch_size=16
        )
        
        assert patcher.min_patch_size == 4
        assert patcher.max_patch_size == 32
        assert patcher.target_patch_size == 16
    
    def test_create_patches_uniform(self):
        """Test patch creation on uniform data."""
        patcher = DynamicPatcher()
        
        # Uniform data should create larger patches
        uniform_data = b"a" * 100
        patches = patcher.create_patches(uniform_data)
        
        assert len(patches) > 0
        assert all(isinstance(p, bytes) for p in patches)
        
        # Verify patches reconstruct original
        reconstructed = b"".join(patches)
        assert reconstructed == uniform_data
    
    def test_create_patches_varied(self):
        """Test patch creation on varied data."""
        patcher = DynamicPatcher()
        
        # Varied data should create smaller patches
        varied_data = bytes(range(100))
        patches = patcher.create_patches(varied_data)
        
        assert len(patches) > 3  # Should have multiple patches
        
        # Verify reconstruction
        reconstructed = b"".join(patches)
        assert reconstructed == varied_data
    
    def test_patch_size_constraints(self):
        """Test that patches respect size constraints."""
        patcher = DynamicPatcher(
            min_patch_size=4,
            max_patch_size=8,
            target_patch_size=6
        )
        
        test_data = b"Hello, World! This is a test sequence."
        patches = patcher.create_patches(test_data)
        
        for patch in patches:
            # Allow last patch to be smaller than min
            if patch != patches[-1]:
                assert 4 <= len(patch) <= 8
            else:
                assert len(patch) <= 8
    
    def test_get_patch_metadata(self):
        """Test patch metadata extraction."""
        patcher = DynamicPatcher()
        
        test_data = b"Test data"
        patches = patcher.create_patches(test_data)
        boundaries = patcher.get_patch_boundaries()
        entropies = patcher.get_patch_entropies()
        
        assert len(patches) == len(boundaries) == len(entropies)
        
        # Verify boundaries
        for i, (start, end) in enumerate(boundaries):
            assert test_data[start:end] == patches[i]
    
    def test_adaptive_patching(self):
        """Test adaptive patching based on content."""
        patcher = DynamicPatcher(adaptive=True)
        
        # Create data with different entropy regions
        low_entropy = b"a" * 50
        high_entropy = bytes(np.random.randint(0, 256, 50))
        mixed_data = low_entropy + high_entropy + low_entropy
        
        patches = patcher.create_patches(mixed_data)
        boundaries = patcher.get_patch_boundaries()
        
        # Low entropy regions should have larger patches
        first_patch_size = boundaries[0][1] - boundaries[0][0]
        
        # Find patches in high entropy region
        high_entropy_patches = [
            (s, e) for s, e in boundaries 
            if s >= 50 and e <= 100
        ]
        
        if high_entropy_patches:
            avg_high_entropy_size = np.mean([
                e - s for s, e in high_entropy_patches
            ])
            assert first_patch_size > avg_high_entropy_size


class TestPatchEmbedder:
    """Test PatchEmbedder functionality."""
    
    def test_initialization(self):
        """Test embedder initialization."""
        embedder = PatchEmbedder(
            vocab_size=256,
            embedding_dim=512,
            max_patch_size=32
        )
        
        assert embedder.vocab_size == 256
        assert embedder.embedding_dim == 512
        assert embedder.byte_embeddings.num_embeddings == 256
        assert embedder.byte_embeddings.embedding_dim == 512
    
    def test_embed_single_patch(self):
        """Test embedding a single patch."""
        embedder = PatchEmbedder(embedding_dim=128)
        
        patch = b"Hello"
        embedding = embedder.embed_patch(patch)
        
        assert isinstance(embedding, torch.Tensor)
        assert embedding.shape == (128,)
        assert embedding.dtype == torch.float32
    
    def test_embed_multiple_patches(self):
        """Test embedding multiple patches."""
        embedder = PatchEmbedder(embedding_dim=128)
        
        patches = [b"Hello", b"World", b"Test"]
        embeddings = embedder.embed_patches(patches)
        
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (3, 128)
    
    def test_position_encoding(self):
        """Test positional encoding in patches."""
        embedder = PatchEmbedder(
            embedding_dim=128,
            use_positional=True
        )
        
        # Same content at different positions should have different embeddings
        patch = b"Test"
        emb1 = embedder.embed_patch(patch, position=0)
        emb2 = embedder.embed_patch(patch, position=10)
        
        # Embeddings should be different due to position
        assert not torch.allclose(emb1, emb2)
    
    def test_patch_padding(self):
        """Test handling of variable-length patches."""
        embedder = PatchEmbedder(embedding_dim=128)
        
        # Patches of different lengths
        patches = [b"Hi", b"Hello", b"Test", b"LongerPatch"]
        embeddings = embedder.embed_patches(patches)
        
        # All should have same embedding dimension
        assert embeddings.shape == (4, 128)
        
        # Embeddings should be different
        for i in range(len(patches)):
            for j in range(i + 1, len(patches)):
                assert not torch.allclose(embeddings[i], embeddings[j])
    
    def test_aggregation_methods(self):
        """Test different aggregation methods."""
        # Test mean aggregation
        embedder_mean = PatchEmbedder(
            embedding_dim=128,
            aggregation='mean'
        )
        patch = b"Hello"
        emb_mean = embedder_mean.embed_patch(patch)
        assert emb_mean.shape == (128,)
        
        # Test sum aggregation
        embedder_sum = PatchEmbedder(
            embedding_dim=128,
            aggregation='sum'
        )
        emb_sum = embedder_sum.embed_patch(patch)
        assert emb_sum.shape == (128,)
        
        # Different aggregations should give different results
        assert not torch.allclose(emb_mean, emb_sum)


class TestBLTInputProcessor:
    """Test complete BLT input processing pipeline."""
    
    def test_initialization(self):
        """Test processor initialization."""
        processor = BLTInputProcessor(
            min_patch_size=4,
            max_patch_size=32,
            embedding_dim=512
        )
        
        assert processor.byte_processor is not None
        assert processor.patcher is not None
        assert processor.embedder is not None
    
    def test_process_text_input(self):
        """Test processing text input end-to-end."""
        processor = BLTInputProcessor(embedding_dim=256)
        
        text = "Hello, World! This is a test."
        embeddings, metadata = processor.process_input(text)
        
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.dim() == 2  # (num_patches, embedding_dim)
        assert embeddings.shape[1] == 256
        
        assert 'patches' in metadata
        assert 'boundaries' in metadata
        assert 'entropies' in metadata
        assert 'byte_sequence' in metadata
    
    def test_process_batch(self):
        """Test batch processing."""
        processor = BLTInputProcessor(embedding_dim=256)
        
        texts = [
            "First text",
            "Second text is longer",
            "Third"
        ]
        
        batch_embeddings, batch_metadata = processor.process_batch(texts)
        
        assert len(batch_embeddings) == 3
        assert len(batch_metadata) == 3
        
        # Each item should have embeddings
        for emb in batch_embeddings:
            assert isinstance(emb, torch.Tensor)
            assert emb.shape[1] == 256
    
    def test_reproducibility(self):
        """Test that processing is deterministic."""
        processor = BLTInputProcessor(embedding_dim=128)
        
        text = "Test reproducibility"
        
        # Process same text twice
        emb1, meta1 = processor.process_input(text)
        emb2, meta2 = processor.process_input(text)
        
        # Should get identical results
        assert torch.allclose(emb1, emb2)
        assert meta1['boundaries'] == meta2['boundaries']
    
    def test_unicode_handling(self):
        """Test handling of Unicode text."""
        processor = BLTInputProcessor(embedding_dim=128)
        
        unicode_texts = [
            "Hello 世界",
            "Привет мир",
            "مرحبا بالعالم",
            "🎉🎊🎈"
        ]
        
        for text in unicode_texts:
            embeddings, metadata = processor.process_input(text)
            assert embeddings.shape[0] > 0  # Should create patches
            assert embeddings.shape[1] == 128
            
            # Should be able to track byte sequence
            assert metadata['byte_sequence'] == text.encode('utf-8')
    
    def test_empty_input(self):
        """Test handling of empty input."""
        processor = BLTInputProcessor(embedding_dim=128)
        
        embeddings, metadata = processor.process_input("")
        
        # Should handle gracefully
        assert embeddings.shape[0] == 0  # No patches
        assert embeddings.shape[1] == 128
        assert metadata['byte_sequence'] == b""
    
    def test_device_handling(self):
        """Test device placement."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        
        processor = BLTInputProcessor(
            embedding_dim=128,
            device=device
        )
        
        text = "Test device placement"
        embeddings, _ = processor.process_input(text)
        
        assert embeddings.device == device