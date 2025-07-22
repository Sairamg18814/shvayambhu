"""Diversity metrics and checking for synthetic data.

This module provides diversity assessment to ensure generated
synthetic data covers a wide range of patterns and content.
"""

import numpy as np
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
import hashlib
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch


@dataclass
class DiversityMetrics:
    """Metrics for diversity assessment."""
    # Similarity metrics
    avg_similarity: float = 0.0
    max_similarity: float = 0.0
    min_similarity: float = 1.0
    
    # N-gram diversity
    unigram_diversity: float = 0.0
    bigram_diversity: float = 0.0
    trigram_diversity: float = 0.0
    
    # Semantic diversity
    semantic_diversity: float = 0.0
    topic_diversity: float = 0.0
    
    # Length diversity
    length_variance: float = 0.0
    length_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Pattern diversity
    syntactic_diversity: float = 0.0
    structural_diversity: float = 0.0
    
    # Overall diversity score
    diversity_score: float = 0.0
    diversity_flags: List[str] = field(default_factory=list)


class DiversityChecker:
    """Check and ensure diversity in synthetic data."""
    
    def __init__(
        self,
        min_similarity_threshold: float = 0.3,
        max_similarity_threshold: float = 0.9,
        n_reference_samples: int = 100,
        use_semantic_similarity: bool = True
    ):
        self.min_similarity_threshold = min_similarity_threshold
        self.max_similarity_threshold = max_similarity_threshold
        self.n_reference_samples = n_reference_samples
        self.use_semantic_similarity = use_semantic_similarity
        
        # Reference samples for comparison
        self.reference_samples: List[str] = []
        self.reference_embeddings: Optional[np.ndarray] = None
        
        # TF-IDF vectorizer for similarity
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        self.is_fitted = False
        
        # N-gram collectors
        self.global_unigrams: Set[str] = set()
        self.global_bigrams: Set[Tuple[str, str]] = set()
        self.global_trigrams: Set[Tuple[str, str, str]] = set()
    
    def check_diversity(
        self,
        text: str,
        existing_hashes: Optional[Set[str]] = None,
        reference_texts: Optional[List[str]] = None
    ) -> DiversityMetrics:
        """Check diversity of a text against existing samples."""
        metrics = DiversityMetrics()
        
        # Update reference samples if provided
        if reference_texts:
            self.update_reference_samples(reference_texts)
        
        # Calculate n-gram diversity
        ngram_metrics = self._calculate_ngram_diversity(text)
        metrics.unigram_diversity = ngram_metrics['unigram']
        metrics.bigram_diversity = ngram_metrics['bigram']
        metrics.trigram_diversity = ngram_metrics['trigram']
        
        # Calculate similarity with reference samples
        if self.reference_samples:
            similarity_metrics = self._calculate_similarity(text)
            metrics.avg_similarity = similarity_metrics['avg']
            metrics.max_similarity = similarity_metrics['max']
            metrics.min_similarity = similarity_metrics['min']
        
        # Calculate semantic diversity if enabled
        if self.use_semantic_similarity and self.reference_embeddings is not None:
            metrics.semantic_diversity = self._calculate_semantic_diversity(text)
        
        # Calculate structural diversity
        metrics.syntactic_diversity = self._calculate_syntactic_diversity(text)
        metrics.structural_diversity = self._calculate_structural_diversity(text)
        
        # Length analysis
        length_metrics = self._analyze_length_diversity(text)
        metrics.length_variance = length_metrics['variance']
        metrics.length_distribution = length_metrics['distribution']
        
        # Calculate overall diversity score
        metrics.diversity_score = self._calculate_overall_diversity(metrics)
        
        # Set diversity flags
        metrics.diversity_flags = self._get_diversity_flags(metrics)
        
        return metrics
    
    def update_reference_samples(self, texts: List[str]):
        """Update reference samples for diversity comparison."""
        # Add to reference samples (keep most recent)
        self.reference_samples.extend(texts)
        if len(self.reference_samples) > self.n_reference_samples:
            self.reference_samples = self.reference_samples[-self.n_reference_samples:]
        
        # Update TF-IDF vectorizer
        if len(self.reference_samples) >= 10:
            self.tfidf_vectorizer.fit(self.reference_samples)
            self.is_fitted = True
        
        # Update n-gram sets
        for text in texts:
            tokens = text.lower().split()
            self.global_unigrams.update(tokens)
            
            if len(tokens) >= 2:
                self.global_bigrams.update(zip(tokens[:-1], tokens[1:]))
            
            if len(tokens) >= 3:
                self.global_trigrams.update(zip(tokens[:-2], tokens[1:-1], tokens[2:]))
    
    def _calculate_ngram_diversity(self, text: str) -> Dict[str, float]:
        """Calculate n-gram diversity metrics."""
        tokens = text.lower().split()
        
        # Unigram diversity
        unique_unigrams = set(tokens)
        new_unigrams = unique_unigrams - self.global_unigrams
        unigram_diversity = len(new_unigrams) / max(len(unique_unigrams), 1)
        
        # Bigram diversity
        bigrams = list(zip(tokens[:-1], tokens[1:])) if len(tokens) >= 2 else []
        unique_bigrams = set(bigrams)
        new_bigrams = unique_bigrams - self.global_bigrams
        bigram_diversity = len(new_bigrams) / max(len(unique_bigrams), 1) if bigrams else 0
        
        # Trigram diversity
        trigrams = list(zip(tokens[:-2], tokens[1:-1], tokens[2:])) if len(tokens) >= 3 else []
        unique_trigrams = set(trigrams)
        new_trigrams = unique_trigrams - self.global_trigrams
        trigram_diversity = len(new_trigrams) / max(len(unique_trigrams), 1) if trigrams else 0
        
        return {
            'unigram': unigram_diversity,
            'bigram': bigram_diversity,
            'trigram': trigram_diversity
        }
    
    def _calculate_similarity(self, text: str) -> Dict[str, float]:
        """Calculate similarity with reference samples."""
        if not self.is_fitted:
            return {'avg': 0.5, 'max': 0.5, 'min': 0.5}
        
        # Vectorize the text
        try:
            text_vector = self.tfidf_vectorizer.transform([text])
            reference_vectors = self.tfidf_vectorizer.transform(self.reference_samples)
            
            # Calculate cosine similarities
            similarities = cosine_similarity(text_vector, reference_vectors)[0]
            
            return {
                'avg': np.mean(similarities),
                'max': np.max(similarities),
                'min': np.min(similarities)
            }
        except:
            # Fallback if vectorization fails
            return {'avg': 0.5, 'max': 0.5, 'min': 0.5}
    
    def _calculate_semantic_diversity(self, text: str) -> float:
        """Calculate semantic diversity using embeddings."""
        # Simplified semantic diversity based on vocabulary
        # In production, this would use actual embeddings
        tokens = set(text.lower().split())
        
        # Calculate semantic categories (simplified)
        categories = {
            'technical': ['algorithm', 'function', 'data', 'system', 'code'],
            'creative': ['story', 'character', 'plot', 'narrative', 'scene'],
            'academic': ['research', 'study', 'analysis', 'hypothesis', 'conclusion'],
            'general': ['people', 'time', 'world', 'day', 'life']
        }
        
        category_scores = {}
        for cat, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in tokens)
            category_scores[cat] = score
        
        # Diversity is higher when multiple categories are represented
        active_categories = sum(1 for score in category_scores.values() if score > 0)
        semantic_diversity = active_categories / len(categories)
        
        return semantic_diversity
    
    def _calculate_syntactic_diversity(self, text: str) -> float:
        """Calculate syntactic pattern diversity."""
        # Analyze sentence structures
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Analyze sentence patterns
        patterns = []
        for sentence in sentences:
            # Simple pattern: length and punctuation
            pattern = {
                'length': len(sentence.split()),
                'has_comma': ',' in sentence,
                'has_question': '?' in sentence,
                'has_exclamation': '!' in sentence,
                'starts_with_caps': sentence[0].isupper() if sentence else False
            }
            patterns.append(tuple(pattern.values()))
        
        # Diversity is the ratio of unique patterns
        unique_patterns = set(patterns)
        syntactic_diversity = len(unique_patterns) / max(len(patterns), 1)
        
        return syntactic_diversity
    
    def _calculate_structural_diversity(self, text: str) -> float:
        """Calculate structural diversity (paragraphs, sections, etc.)."""
        # Analyze text structure
        lines = text.split('\n')
        
        structural_features = {
            'num_paragraphs': len([l for l in lines if l.strip()]),
            'has_headers': any(l.strip().startswith('#') for l in lines),
            'has_lists': any(l.strip().startswith(('-', '*', '1.')) for l in lines),
            'has_code_blocks': '```' in text or '    ' in text,
            'has_quotes': '"' in text or "'" in text,
            'avg_paragraph_length': np.mean([len(l.split()) for l in lines if l.strip()]) if lines else 0
        }
        
        # Count active structural elements
        active_elements = sum(1 for key, value in structural_features.items() 
                            if (isinstance(value, bool) and value) or 
                            (isinstance(value, (int, float)) and value > 0))
        
        structural_diversity = active_elements / len(structural_features)
        
        return structural_diversity
    
    def _analyze_length_diversity(self, text: str) -> Dict[str, Any]:
        """Analyze length-based diversity."""
        words = text.split()
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Calculate length statistics
        word_lengths = [len(w) for w in words]
        sentence_lengths = [len(s.split()) for s in sentences]
        
        # Length distribution
        distribution = {
            'short_sentences': sum(1 for l in sentence_lengths if l < 10),
            'medium_sentences': sum(1 for l in sentence_lengths if 10 <= l < 20),
            'long_sentences': sum(1 for l in sentence_lengths if l >= 20),
            'short_words': sum(1 for l in word_lengths if l < 4),
            'medium_words': sum(1 for l in word_lengths if 4 <= l < 8),
            'long_words': sum(1 for l in word_lengths if l >= 8)
        }
        
        # Calculate variance
        variance = np.var(sentence_lengths) if sentence_lengths else 0
        
        return {
            'variance': variance,
            'distribution': distribution
        }
    
    def _calculate_overall_diversity(self, metrics: DiversityMetrics) -> float:
        """Calculate overall diversity score."""
        # Weight different components
        weights = {
            'ngram': 0.2,
            'similarity': 0.2,
            'semantic': 0.15,
            'syntactic': 0.15,
            'structural': 0.15,
            'length': 0.15
        }
        
        # Calculate n-gram component
        ngram_score = np.mean([
            metrics.unigram_diversity,
            metrics.bigram_diversity,
            metrics.trigram_diversity
        ])
        
        # Calculate similarity component (inverse similarity for diversity)
        similarity_score = 1.0 - metrics.avg_similarity
        
        # Length diversity component
        length_score = min(metrics.length_variance / 50, 1.0)  # Normalize variance
        
        # Calculate weighted score
        score = (
            weights['ngram'] * ngram_score +
            weights['similarity'] * similarity_score +
            weights['semantic'] * metrics.semantic_diversity +
            weights['syntactic'] * metrics.syntactic_diversity +
            weights['structural'] * metrics.structural_diversity +
            weights['length'] * length_score
        )
        
        return score
    
    def _get_diversity_flags(self, metrics: DiversityMetrics) -> List[str]:
        """Get diversity flags for the text."""
        flags = []
        
        if metrics.max_similarity > self.max_similarity_threshold:
            flags.append("too_similar")
        
        if metrics.unigram_diversity < 0.1:
            flags.append("low_vocabulary_novelty")
        
        if metrics.semantic_diversity < 0.2:
            flags.append("narrow_semantic_range")
        
        if metrics.syntactic_diversity < 0.3:
            flags.append("repetitive_syntax")
        
        if metrics.structural_diversity < 0.2:
            flags.append("uniform_structure")
        
        if metrics.length_variance < 5:
            flags.append("uniform_lengths")
        
        return flags
    
    def get_diversity_report(self, texts: List[str]) -> Dict[str, Any]:
        """Generate a diversity report for a collection of texts."""
        all_metrics = []
        
        for text in texts:
            metrics = self.check_diversity(text)
            all_metrics.append(metrics)
        
        # Aggregate statistics
        report = {
            'num_texts': len(texts),
            'avg_diversity_score': np.mean([m.diversity_score for m in all_metrics]),
            'min_diversity_score': np.min([m.diversity_score for m in all_metrics]),
            'max_diversity_score': np.max([m.diversity_score for m in all_metrics]),
            'avg_unigram_diversity': np.mean([m.unigram_diversity for m in all_metrics]),
            'avg_bigram_diversity': np.mean([m.bigram_diversity for m in all_metrics]),
            'avg_trigram_diversity': np.mean([m.trigram_diversity for m in all_metrics]),
            'avg_semantic_diversity': np.mean([m.semantic_diversity for m in all_metrics]),
            'avg_syntactic_diversity': np.mean([m.syntactic_diversity for m in all_metrics]),
            'avg_structural_diversity': np.mean([m.structural_diversity for m in all_metrics])
        }
        
        # Count flags
        all_flags = []
        for m in all_metrics:
            all_flags.extend(m.diversity_flags)
        
        flag_counts = Counter(all_flags)
        report['diversity_issues'] = dict(flag_counts)
        
        return report
    
    def reset_global_statistics(self):
        """Reset global n-gram and reference statistics."""
        self.global_unigrams.clear()
        self.global_bigrams.clear()
        self.global_trigrams.clear()
        self.reference_samples.clear()
        self.reference_embeddings = None
        self.is_fitted = False