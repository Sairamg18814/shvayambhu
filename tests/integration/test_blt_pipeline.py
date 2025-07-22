"""Integration tests for the complete BLT pipeline.

This module tests the end-to-end functionality of the Byte Latent Transformer,
ensuring all components work together correctly.
"""

import pytest
import torch
import numpy as np
from typing import List, Tuple

from shvayambhu.core.blt.encoder import LocalEncoder
from shvayambhu.core.blt.transformer import LatentTransformer
from shvayambhu.core.blt.decoder import LocalDecoder, AdaptiveDecoder
from shvayambhu.core.blt.pipeline import BLTPipeline
from shvayambhu.core.blt.patching import BLTInputProcessor


class TestBLTPipelineIntegration:
    """Integration tests for the BLT pipeline."""
    
    @pytest.fixture
    def blt_pipeline(self):
        """Create a BLT pipeline for testing."""
        config = {
            "vocab_size": 256,
            "hidden_dim": 768,
            "num_layers": 6,
            "num_heads": 12,
            "patch_embedding_dim": 768,
            "max_patch_size": 32,
            "device": torch.device("cpu")
        }
        return BLTPipeline(config)
    
    def test_end_to_end_reconstruction(self, blt_pipeline):
        """Test that text can be encoded and decoded accurately."""
        test_texts = [
            "Hello, World!",
            "The quick brown fox jumps over the lazy dog.",
            "Testing BLT pipeline with various inputs.",
            "1234567890 !@#$%^&*() Special chars test"
        ]
        
        for text in test_texts:
            # Process through pipeline
            byte_seq = text.encode('utf-8')
            output = blt_pipeline.process_bytes(byte_seq, mode='inference')
            
            # Verify reconstruction
            reconstructed = output['reconstructed_text']
            assert reconstructed == text, f"Failed to reconstruct: {text}"
    
    def test_batch_processing(self, blt_pipeline):
        """Test batch processing capability."""
        batch_texts = [
            "First text in batch",
            "Second text with different length",
            "Third",
            "Fourth text is a bit longer than the others in this batch"
        ]
        
        # Process as batch
        byte_sequences = [text.encode('utf-8') for text in batch_texts]
        outputs = blt_pipeline.process_batch(byte_sequences, mode='inference')
        
        # Verify all texts reconstructed correctly
        for i, text in enumerate(batch_texts):
            assert outputs['reconstructed_texts'][i] == text
    
    def test_encoder_decoder_consistency(self, blt_pipeline):
        """Test that encoder and decoder maintain consistency."""
        text = "Testing encoder-decoder consistency"
        byte_seq = text.encode('utf-8')
        
        # Get intermediate representations
        encoder_output = blt_pipeline.encoder(byte_seq)
        transformer_output = blt_pipeline.transformer(
            encoder_output['patch_embeddings'].unsqueeze(0)
        )
        decoder_output = blt_pipeline.decoder(
            transformer_output['hidden_states'],
            encoder_output['patch_boundaries'],
            encoder_output['original_length']
        )
        
        # Verify shapes and consistency
        assert encoder_output['patch_embeddings'].shape[0] == len(encoder_output['patch_boundaries'])
        assert transformer_output['hidden_states'].shape[2] == blt_pipeline.config['hidden_dim']
        assert len(decoder_output['byte_logits']) == encoder_output['original_length']
    
    def test_variable_length_inputs(self, blt_pipeline):
        """Test handling of variable length inputs."""
        test_cases = [
            "",  # Empty string
            "a",  # Single character
            "ab",  # Two characters
            "a" * 100,  # Repetitive content
            "".join(chr(i) for i in range(32, 127)),  # ASCII range
            "A" * 1000,  # Long repetitive
            "The " * 250,  # Long repetitive pattern
        ]
        
        for text in test_cases:
            if text:  # Skip empty string for now
                byte_seq = text.encode('utf-8')
                output = blt_pipeline.process_bytes(byte_seq, mode='inference')
                reconstructed = output['reconstructed_text']
                assert reconstructed == text, f"Failed on length {len(text)}"
    
    def test_unicode_handling(self, blt_pipeline):
        """Test handling of various Unicode characters."""
        unicode_tests = [
            "Hello 世界",  # Chinese
            "Привет мир",  # Russian
            "مرحبا بالعالم",  # Arabic
            "🎉🎊🎈",  # Emojis
            "café",  # Accented characters
            "𝓗𝓮𝓵𝓵𝓸",  # Mathematical alphanumeric
            "¡Hola! ¿Cómo estás?",  # Spanish punctuation
        ]
        
        for text in unicode_tests:
            byte_seq = text.encode('utf-8')
            output = blt_pipeline.process_bytes(byte_seq, mode='inference')
            reconstructed = output['reconstructed_text']
            assert reconstructed == text, f"Failed on Unicode: {text}"
    
    def test_training_mode(self, blt_pipeline):
        """Test pipeline in training mode."""
        text = "Training mode test"
        byte_seq = text.encode('utf-8')
        
        # Process in training mode
        output = blt_pipeline.process_bytes(byte_seq, mode='training')
        
        # Check that we get loss values
        assert 'loss' in output
        assert isinstance(output['loss'], torch.Tensor)
        assert output['loss'].requires_grad
    
    def test_attention_patterns(self, blt_pipeline):
        """Test that attention patterns are reasonable."""
        text = "The cat sat on the mat."
        byte_seq = text.encode('utf-8')
        
        # Get attention patterns
        output = blt_pipeline.process_bytes(
            byte_seq, 
            mode='inference',
            return_attention=True
        )
        
        if 'attention_weights' in output:
            attention = output['attention_weights']
            # Check attention shape and values
            assert len(attention) == blt_pipeline.config['num_layers']
            for layer_attention in attention:
                assert torch.all(layer_attention >= 0)
                assert torch.all(layer_attention <= 1)
                # Check attention sums to 1
                assert torch.allclose(
                    layer_attention.sum(dim=-1),
                    torch.ones_like(layer_attention.sum(dim=-1))
                )
    
    def test_memory_efficiency(self, blt_pipeline):
        """Test memory usage is within expected bounds."""
        import tracemalloc
        
        text = "Memory test " * 100
        byte_seq = text.encode('utf-8')
        
        # Measure memory usage
        tracemalloc.start()
        output = blt_pipeline.process_bytes(byte_seq, mode='inference')
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Check memory usage is reasonable (less than 100MB for this test)
        assert peak < 100 * 1024 * 1024, f"Peak memory too high: {peak / 1024 / 1024:.2f} MB"
    
    def test_deterministic_output(self, blt_pipeline):
        """Test that outputs are deterministic."""
        text = "Deterministic test"
        byte_seq = text.encode('utf-8')
        
        # Set random seeds
        torch.manual_seed(42)
        output1 = blt_pipeline.process_bytes(byte_seq, mode='inference')
        
        torch.manual_seed(42)
        output2 = blt_pipeline.process_bytes(byte_seq, mode='inference')
        
        # Outputs should be identical
        assert output1['reconstructed_text'] == output2['reconstructed_text']
    
    def test_gradient_flow(self, blt_pipeline):
        """Test gradient flow through the pipeline."""
        text = "Gradient flow test"
        byte_seq = text.encode('utf-8')
        
        # Enable gradients
        blt_pipeline.train()
        
        # Forward pass
        output = blt_pipeline.process_bytes(byte_seq, mode='training')
        loss = output['loss']
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist and are not zero
        for name, param in blt_pipeline.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.all(param.grad == 0), f"Zero gradients in {name}"
                assert not torch.any(torch.isnan(param.grad)), f"NaN gradients in {name}"
    
    def test_adaptive_patching(self, blt_pipeline):
        """Test that adaptive patching works correctly."""
        # High entropy text (should create smaller patches)
        high_entropy = "aB3$xY9!pQ2&mN5*vC8#"
        
        # Low entropy text (should create larger patches)
        low_entropy = "aaaaaaaabbbbbbbbcccccccc"
        
        # Process both
        he_output = blt_pipeline.encoder(high_entropy.encode('utf-8'))
        le_output = blt_pipeline.encoder(low_entropy.encode('utf-8'))
        
        # High entropy should have more patches
        assert len(he_output['patch_boundaries']) > len(le_output['patch_boundaries'])
    
    def test_error_handling(self, blt_pipeline):
        """Test error handling for invalid inputs."""
        # Test with invalid UTF-8
        invalid_bytes = bytes([0xFF, 0xFE, 0xFD])
        with pytest.raises(Exception):
            blt_pipeline.process_bytes(invalid_bytes, mode='inference')
        
        # Test with None input
        with pytest.raises(Exception):
            blt_pipeline.process_bytes(None, mode='inference')
    

class TestBLTComponentIntegration:
    """Test integration between specific BLT components."""
    
    def test_encoder_transformer_integration(self):
        """Test encoder and transformer work together."""
        encoder = LocalEncoder(
            patch_embedding_dim=768,
            hidden_dim=768,
            num_layers=2
        )
        transformer = LatentTransformer(
            hidden_dim=768,
            num_layers=6,
            num_heads=12
        )
        
        # Process text through encoder
        text = "Testing component integration"
        byte_seq = text.encode('utf-8')
        encoder_output = encoder(byte_seq)
        
        # Process through transformer
        patch_embeddings = encoder_output['patch_embeddings'].unsqueeze(0)
        transformer_output = transformer(patch_embeddings)
        
        # Verify output shapes
        assert transformer_output['hidden_states'].shape[0] == 1  # batch size
        assert transformer_output['hidden_states'].shape[1] == patch_embeddings.shape[1]  # seq len
        assert transformer_output['hidden_states'].shape[2] == 768  # hidden dim
    
    def test_transformer_decoder_integration(self):
        """Test transformer and decoder work together."""
        transformer = LatentTransformer(
            hidden_dim=768,
            num_layers=6,
            num_heads=12
        )
        decoder = LocalDecoder(
            hidden_dim=768,
            vocab_size=256,
            num_layers=2
        )
        
        # Create dummy transformer output
        batch_size = 2
        seq_len = 10
        hidden_states = torch.randn(batch_size, seq_len, 768)
        
        # Create dummy patch boundaries
        patch_boundaries = [[(0, 4), (4, 8), (8, 12)] for _ in range(batch_size)]
        original_lengths = [12, 12]
        
        # Process through decoder
        output = decoder(hidden_states, patch_boundaries, original_lengths)
        
        # Verify output
        assert 'byte_logits' in output
        assert len(output['byte_logits']) == batch_size
        assert all(len(logits) == length for logits, length in zip(output['byte_logits'], original_lengths))
    
    def test_input_processor_pipeline_integration(self):
        """Test BLTInputProcessor integrates with pipeline."""
        processor = BLTInputProcessor(
            min_patch_size=4,
            max_patch_size=32,
            embedding_dim=768
        )
        
        pipeline = BLTPipeline({
            "vocab_size": 256,
            "hidden_dim": 768,
            "num_layers": 6,
            "num_heads": 12,
            "patch_embedding_dim": 768,
            "max_patch_size": 32,
            "device": torch.device("cpu")
        })
        
        # Process text
        text = "Testing processor integration"
        embeddings, metadata = processor.process_input(text)
        
        # Use metadata in pipeline
        output = pipeline.process_bytes(
            metadata['byte_sequence'],
            mode='inference'
        )
        
        assert output['reconstructed_text'] == text


class TestMultilingualIntegration:
    """Test multilingual support across the pipeline."""
    
    @pytest.fixture
    def multilingual_pipeline(self):
        """Create pipeline for multilingual testing."""
        config = {
            "vocab_size": 256,
            "hidden_dim": 768,
            "num_layers": 6,
            "num_heads": 12,
            "patch_embedding_dim": 768,
            "max_patch_size": 32,
            "device": torch.device("cpu")
        }
        return BLTPipeline(config)
    
    def test_language_preservation(self, multilingual_pipeline):
        """Test that different languages are preserved correctly."""
        test_cases = [
            ("English", "The quick brown fox"),
            ("Chinese", "快速的棕色狐狸"),
            ("Arabic", "الثعلب البني السريع"),
            ("Hindi", "तेज़ भूरी लोमड़ी"),
            ("Russian", "Быстрая коричневая лиса"),
            ("Japanese", "速い茶色のキツネ"),
            ("Korean", "빠른 갈색 여우"),
            ("Hebrew", "השועל החום המהיר"),
            ("Greek", "Η γρήγορη καφέ αλεπού"),
            ("Thai", "สุนัขจิ้งจอกสีน้ำตาลที่รวดเร็ว")
        ]
        
        for language, text in test_cases:
            byte_seq = text.encode('utf-8')
            output = multilingual_pipeline.process_bytes(byte_seq, mode='inference')
            reconstructed = output['reconstructed_text']
            assert reconstructed == text, f"Failed to preserve {language}: {text}"
    
    def test_mixed_language_documents(self, multilingual_pipeline):
        """Test documents with multiple languages."""
        mixed_texts = [
            "Hello world! 你好世界！ مرحبا بالعالم!",
            "Python编程 is very popular في العالم",
            "Machine Learning (機械学習) ist sehr wichtig"
        ]
        
        for text in mixed_texts:
            byte_seq = text.encode('utf-8')
            output = multilingual_pipeline.process_bytes(byte_seq, mode='inference')
            reconstructed = output['reconstructed_text']
            assert reconstructed == text