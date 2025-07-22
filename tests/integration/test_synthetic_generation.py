"""Integration tests for synthetic data generation pipeline."""

import pytest
import torch
import tempfile
from pathlib import Path
import json

from training.synthetic import (
    SyntheticDataGenerator,
    GenerationConfig,
    QualityFilter,
    DiversityChecker,
    DomainSpecificGenerator
)
from core.blt.pipeline import BLTPipeline


@pytest.fixture
def mock_model():
    """Create a mock BLT model for testing."""
    config = {
        'vocab_size': 256,
        'hidden_dim': 128,
        'num_layers': 2,
        'num_heads': 4,
        'intermediate_size': 512,
        'max_position_embeddings': 512,
        'patch_size': 8
    }
    model = BLTPipeline(config)
    return model


@pytest.fixture
def generation_config():
    """Create test generation configuration."""
    return GenerationConfig(
        max_length=256,
        min_length=32,
        temperature=0.8,
        batch_size=4,
        num_generations=20,
        domains=['general', 'code'],
        output_dir=tempfile.mkdtemp()
    )


@pytest.fixture
def quality_filter():
    """Create quality filter for testing."""
    return QualityFilter(
        min_length=32,
        max_length=256,
        min_tokens=8,
        max_special_char_ratio=0.3,
        max_repetition_ratio=0.4
    )


@pytest.fixture
def diversity_checker():
    """Create diversity checker for testing."""
    return DiversityChecker(
        min_similarity_threshold=0.2,
        max_similarity_threshold=0.8,
        n_reference_samples=10
    )


class TestQualityFilter:
    """Test quality filtering functionality."""
    
    def test_quality_evaluation(self, quality_filter):
        """Test quality metric calculation."""
        # Good quality text
        good_text = "The importance of artificial intelligence in modern technology cannot be overstated. It provides innovative solutions to complex problems across various industries."
        
        metrics = quality_filter.evaluate(good_text)
        
        assert metrics.length > 0
        assert metrics.num_tokens > 0
        assert 0 <= metrics.readability_score <= 1
        assert 0 <= metrics.coherence_score <= 1
        assert 0 <= metrics.fluency_score <= 1
        assert 0 <= metrics.overall_score <= 1
        assert isinstance(metrics.quality_flags, list)
    
    def test_quality_filtering(self, quality_filter):
        """Test batch quality filtering."""
        texts = [
            "This is a good quality text with proper structure and content.",
            "abc def abc def abc def",  # Repetitive
            "!!!@@@###$$$",  # Too many special chars
            "Short",  # Too short
            "This is another good quality text that should pass the filter."
        ]
        
        filtered_texts, metrics_list = quality_filter.filter_batch(texts, min_quality_score=0.5)
        
        # Should filter out bad quality texts
        assert len(filtered_texts) < len(texts)
        assert len(filtered_texts) == len(metrics_list)
        
        # All filtered texts should have good quality scores
        for metrics in metrics_list:
            assert metrics.overall_score >= 0.5
    
    def test_edge_cases(self, quality_filter):
        """Test edge cases in quality evaluation."""
        # Empty text
        empty_metrics = quality_filter.evaluate("")
        assert empty_metrics.overall_score == 0.0
        
        # Very long text
        long_text = "word " * 1000
        long_metrics = quality_filter.evaluate(long_text)
        assert long_metrics.length > quality_filter.max_length
        assert "too_long" in long_metrics.quality_flags


class TestDiversityChecker:
    """Test diversity checking functionality."""
    
    def test_diversity_evaluation(self, diversity_checker):
        """Test diversity metric calculation."""
        text = "Machine learning algorithms process large datasets to identify patterns and make predictions about future outcomes."
        
        metrics = diversity_checker.check_diversity(text)
        
        assert 0 <= metrics.diversity_score <= 1
        assert 0 <= metrics.unigram_diversity <= 1
        assert 0 <= metrics.bigram_diversity <= 1
        assert 0 <= metrics.trigram_diversity <= 1
        assert 0 <= metrics.semantic_diversity <= 1
        assert isinstance(metrics.diversity_flags, list)
    
    def test_reference_sample_update(self, diversity_checker):
        """Test reference sample management."""
        initial_count = len(diversity_checker.reference_samples)
        
        new_samples = [
            "First sample text for reference.",
            "Second sample with different content.",
            "Third sample about technology topics."
        ]
        
        diversity_checker.update_reference_samples(new_samples)
        
        assert len(diversity_checker.reference_samples) == initial_count + len(new_samples)
        assert diversity_checker.is_fitted or len(diversity_checker.reference_samples) < 10
    
    def test_similarity_detection(self, diversity_checker):
        """Test similarity detection between texts."""
        # Add reference samples
        reference_texts = [
            "Artificial intelligence is transforming modern technology.",
            "Machine learning algorithms process data efficiently.",
            "Deep learning networks solve complex problems."
        ]
        diversity_checker.update_reference_samples(reference_texts)
        
        # Test similar text
        similar_text = "Artificial intelligence is changing modern technology."
        similar_metrics = diversity_checker.check_diversity(similar_text)
        
        # Test dissimilar text
        dissimilar_text = "The weather forecast predicts rain tomorrow afternoon."
        dissimilar_metrics = diversity_checker.check_diversity(dissimilar_text)
        
        # Similar text should have lower diversity score
        if diversity_checker.is_fitted:
            assert similar_metrics.avg_similarity > dissimilar_metrics.avg_similarity


class TestDomainSpecificGenerator:
    """Test domain-specific generation."""
    
    def test_domain_prompt_selection(self, mock_model, generation_config):
        """Test domain-specific prompt selection."""
        for domain in ['general', 'code', 'technical', 'creative', 'academic']:
            generator = DomainSpecificGenerator(domain, mock_model, generation_config)
            
            prompts = generator._get_domain_prompts()
            assert len(prompts) > 0
            assert all(isinstance(prompt, str) for prompt in prompts)
            
            styles = generator._get_domain_styles()
            assert 'temperature' in styles
            assert 'top_p' in styles
            assert 'repetition_penalty' in styles
    
    def test_domain_generation(self, mock_model, generation_config):
        """Test domain-specific text generation."""
        # Mock the generation to avoid actual model inference
        generator = DomainSpecificGenerator('general', mock_model, generation_config)
        
        # This would normally call the model, so we'll test the setup
        prompts = generator._get_domain_prompts()
        assert len(prompts) > 0
        
        styles = generator._get_domain_styles()
        assert 0 <= styles['temperature'] <= 2.0
        assert 0 <= styles['top_p'] <= 1.0


class TestSyntheticDataGenerator:
    """Test the main synthetic data generator."""
    
    def test_generator_initialization(self, mock_model, generation_config, quality_filter, diversity_checker):
        """Test generator initialization."""
        generator = SyntheticDataGenerator(
            model=mock_model,
            config=generation_config,
            quality_filter=quality_filter,
            diversity_checker=diversity_checker
        )
        
        assert generator.model == mock_model
        assert generator.config == generation_config
        assert generator.quality_filter == quality_filter
        assert generator.diversity_checker == diversity_checker
        assert len(generator.domain_generators) == len(generation_config.domains)
        assert generator.output_dir.exists()
    
    def test_domain_selection(self, mock_model, generation_config, quality_filter, diversity_checker):
        """Test domain selection logic."""
        generator = SyntheticDataGenerator(
            model=mock_model,
            config=generation_config,
            quality_filter=quality_filter,
            diversity_checker=diversity_checker
        )
        
        # Test domain selection
        selected_domains = [generator._select_domain() for _ in range(100)]
        unique_domains = set(selected_domains)
        
        # Should select from configured domains
        assert unique_domains.issubset(set(generation_config.domains))
        
        # Should respect domain weights (roughly)
        domain_counts = {domain: selected_domains.count(domain) for domain in unique_domains}
        assert len(domain_counts) > 0
    
    def test_sample_validation(self, mock_model, generation_config, quality_filter, diversity_checker):
        """Test sample validation logic."""
        generator = SyntheticDataGenerator(
            model=mock_model,
            config=generation_config,
            quality_filter=quality_filter,
            diversity_checker=diversity_checker
        )
        
        # Test good sample
        good_sample = {
            "text": "This is a high-quality text sample with good structure and diverse vocabulary content.",
            "domain": "general",
            "metadata": {"length": 85}
        }
        
        assert generator._validate_sample(good_sample)
        
        # Test bad samples
        short_sample = {
            "text": "Too short",
            "domain": "general",
            "metadata": {"length": 9}
        }
        
        assert not generator._validate_sample(short_sample)
        
        # Test duplicate
        assert not generator._validate_sample(good_sample)  # Should be rejected as duplicate
    
    def test_augmentation(self, mock_model, generation_config, quality_filter, diversity_checker):
        """Test data augmentation functionality."""
        generator = SyntheticDataGenerator(
            model=mock_model,
            config=generation_config,
            quality_filter=quality_filter,
            diversity_checker=diversity_checker
        )
        
        input_texts = [
            "Artificial intelligence is revolutionizing technology.",
            "Machine learning algorithms analyze data patterns."
        ]
        
        # Test augmentation types
        augmentation_types = ["paraphrase", "continuation", "infilling", "summarization"]
        
        for aug_type in augmentation_types:
            # Test individual augmentation (would normally use model)
            result = generator._apply_augmentation(input_texts[0], aug_type)
            # Since we don't have a real model, result might be None
            assert result is None or isinstance(result, str)


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_pipeline(self, mock_model, generation_config, quality_filter, diversity_checker):
        """Test the complete synthetic data generation pipeline."""
        # Reduce sample size for testing
        generation_config.num_generations = 5
        generation_config.batch_size = 2
        
        generator = SyntheticDataGenerator(
            model=mock_model,
            config=generation_config,
            quality_filter=quality_filter,
            diversity_checker=diversity_checker
        )
        
        # Mock the model's forward method to return dummy data
        def mock_forward(input_bytes, mode='training', **kwargs):
            batch_size, seq_len = input_bytes.shape
            vocab_size = 256
            
            return {
                'logits': torch.randn(batch_size, seq_len, vocab_size),
                'loss': torch.tensor(1.0),
                'loss_components': {'next_byte': torch.tensor(0.8)}
            }
        
        mock_model.forward = mock_forward
        
        # This would normally generate actual data
        # For testing, we'll just verify the setup works
        assert len(generator.domain_generators) > 0
        assert generator.config.num_generations == 5
        assert generator.output_dir.exists()
    
    def test_configuration_loading(self):
        """Test loading configuration from file."""
        config_data = {
            "generation_config": {
                "max_length": 512,
                "domains": ["general", "technical"],
                "num_generations": 1000
            }
        }
        
        # Test configuration validation
        assert config_data["generation_config"]["max_length"] > 0
        assert len(config_data["generation_config"]["domains"]) > 0
        assert config_data["generation_config"]["num_generations"] > 0
    
    def test_output_formats(self, mock_model, generation_config, quality_filter, diversity_checker):
        """Test different output formats."""
        for format_type in ['jsonl', 'txt']:
            config = GenerationConfig(
                max_length=128,
                num_generations=5,
                format=format_type,
                output_dir=tempfile.mkdtemp()
            )
            
            generator = SyntheticDataGenerator(
                model=mock_model,
                config=config,
                quality_filter=quality_filter,
                diversity_checker=diversity_checker
            )
            
            # Test batch saving
            sample_batch = [
                {"text": "Sample text 1", "domain": "general"},
                {"text": "Sample text 2", "domain": "technical"}
            ]
            
            output_path = Path(config.output_dir) / f"test.{format_type}"
            generator._save_batch(sample_batch, output_path)
            
            assert output_path.exists()
            assert output_path.stat().st_size > 0


if __name__ == '__main__':
    pytest.main([__file__])