"""Quality filtering for synthetic data generation.

This module provides quality assessment and filtering capabilities
for generated synthetic data.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import re
from collections import Counter
import statistics

from ...core.blt.entropy import calculate_byte_entropy


@dataclass
class QualityMetrics:
    """Quality metrics for generated text."""
    # Basic metrics
    length: int = 0
    num_tokens: int = 0
    avg_token_length: float = 0.0
    
    # Quality scores
    readability_score: float = 0.0
    coherence_score: float = 0.0
    fluency_score: float = 0.0
    
    # Content metrics
    vocabulary_diversity: float = 0.0
    sentence_complexity: float = 0.0
    grammar_score: float = 0.0
    
    # Anomaly metrics
    repetition_ratio: float = 0.0
    special_char_ratio: float = 0.0
    uppercase_ratio: float = 0.0
    
    # Statistical metrics
    entropy: float = 0.0
    perplexity: float = 0.0
    
    # Overall score
    overall_score: float = 0.0
    quality_flags: List[str] = None
    
    def __post_init__(self):
        if self.quality_flags is None:
            self.quality_flags = []


class QualityFilter:
    """Filter synthetic data based on quality metrics."""
    
    def __init__(
        self,
        min_length: int = 50,
        max_length: int = 4096,
        min_tokens: int = 10,
        max_special_char_ratio: float = 0.2,
        max_repetition_ratio: float = 0.3,
        min_vocabulary_diversity: float = 0.3,
        language_model: Optional[Any] = None
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.min_tokens = min_tokens
        self.max_special_char_ratio = max_special_char_ratio
        self.max_repetition_ratio = max_repetition_ratio
        self.min_vocabulary_diversity = min_vocabulary_diversity
        self.language_model = language_model
        
        # Compile regex patterns
        self.sentence_pattern = re.compile(r'[.!?]+')
        self.word_pattern = re.compile(r'\b\w+\b')
        self.special_char_pattern = re.compile(r'[^a-zA-Z0-9\s.,!?;:\'"()-]')
    
    def evaluate(self, text: str) -> QualityMetrics:
        """Evaluate text quality and return metrics."""
        metrics = QualityMetrics()
        
        # Basic metrics
        metrics.length = len(text)
        
        # Tokenization (simple word-based)
        tokens = self.word_pattern.findall(text.lower())
        metrics.num_tokens = len(tokens)
        metrics.avg_token_length = np.mean([len(t) for t in tokens]) if tokens else 0
        
        # Calculate individual quality scores
        metrics.readability_score = self._calculate_readability(text, tokens)
        metrics.coherence_score = self._calculate_coherence(text, tokens)
        metrics.fluency_score = self._calculate_fluency(text)
        
        # Content metrics
        metrics.vocabulary_diversity = self._calculate_vocabulary_diversity(tokens)
        metrics.sentence_complexity = self._calculate_sentence_complexity(text)
        metrics.grammar_score = self._calculate_grammar_score(text)
        
        # Anomaly metrics
        metrics.repetition_ratio = self._calculate_repetition_ratio(text, tokens)
        metrics.special_char_ratio = len(self.special_char_pattern.findall(text)) / max(len(text), 1)
        metrics.uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Statistical metrics
        metrics.entropy = calculate_byte_entropy(np.array(list(text.encode('utf-8'))))
        
        # Calculate overall score
        metrics.overall_score = self._calculate_overall_score(metrics)
        
        # Set quality flags
        metrics.quality_flags = self._get_quality_flags(metrics)
        
        return metrics
    
    def _calculate_readability(self, text: str, tokens: List[str]) -> float:
        """Calculate readability score (simplified Flesch Reading Ease)."""
        sentences = self.sentence_pattern.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences or not tokens:
            return 0.0
        
        # Average sentence length
        avg_sentence_length = len(tokens) / len(sentences)
        
        # Average syllables per word (approximation)
        syllable_count = sum(self._count_syllables(token) for token in tokens)
        avg_syllables_per_word = syllable_count / max(len(tokens), 1)
        
        # Simplified Flesch Reading Ease formula
        score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
        
        # Normalize to 0-1 range
        return max(0, min(100, score)) / 100
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximation)."""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Adjust for silent 'e'
        if word.endswith('e'):
            syllable_count -= 1
        
        # Ensure at least one syllable
        return max(1, syllable_count)
    
    def _calculate_coherence(self, text: str, tokens: List[str]) -> float:
        """Calculate coherence score based on word transitions."""
        if len(tokens) < 2:
            return 0.0
        
        # Calculate transition probabilities
        bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
        bigram_counts = Counter(bigrams)
        
        # Coherence based on bigram frequency
        total_bigrams = len(bigrams)
        unique_bigrams = len(bigram_counts)
        
        # More repeated bigrams = higher coherence
        coherence = 1.0 - (unique_bigrams / max(total_bigrams, 1))
        
        # Adjust for very short texts
        if len(tokens) < 50:
            coherence *= (len(tokens) / 50)
        
        return coherence
    
    def _calculate_fluency(self, text: str) -> float:
        """Calculate fluency score based on punctuation and structure."""
        # Check for proper punctuation
        sentences = self.sentence_pattern.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        fluency_score = 1.0
        
        # Check sentence beginnings (should start with capital)
        capital_starts = sum(1 for s in sentences if s and s[0].isupper())
        fluency_score *= (capital_starts / len(sentences))
        
        # Check for balanced punctuation
        punct_counts = {
            '(': text.count('('),
            ')': text.count(')'),
            '[': text.count('['),
            ']': text.count(']'),
            '{': text.count('{'),
            '}': text.count('}'),
            '"': text.count('"'),
            "'": text.count("'")
        }
        
        # Check paired punctuation
        if punct_counts['('] != punct_counts[')']:
            fluency_score *= 0.9
        if punct_counts['['] != punct_counts[']']:
            fluency_score *= 0.9
        if punct_counts['{'] != punct_counts['}']:
            fluency_score *= 0.9
        if punct_counts['"'] % 2 != 0:
            fluency_score *= 0.95
        
        return fluency_score
    
    def _calculate_vocabulary_diversity(self, tokens: List[str]) -> float:
        """Calculate vocabulary diversity (type-token ratio)."""
        if not tokens:
            return 0.0
        
        unique_tokens = set(tokens)
        return len(unique_tokens) / len(tokens)
    
    def _calculate_sentence_complexity(self, text: str) -> float:
        """Calculate sentence complexity based on structure."""
        sentences = self.sentence_pattern.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        complexity_scores = []
        
        for sentence in sentences:
            # Count subordinate clauses (approximation)
            clause_indicators = ['because', 'although', 'while', 'when', 'if', 'since', 'that', 'which']
            clause_count = sum(1 for indicator in clause_indicators if indicator in sentence.lower())
            
            # Count commas (indicates complexity)
            comma_count = sentence.count(',')
            
            # Sentence length factor
            word_count = len(self.word_pattern.findall(sentence))
            
            # Calculate complexity score for sentence
            complexity = (clause_count * 0.3 + comma_count * 0.2 + word_count * 0.01)
            complexity = min(1.0, complexity)  # Cap at 1.0
            complexity_scores.append(complexity)
        
        return np.mean(complexity_scores)
    
    def _calculate_grammar_score(self, text: str) -> float:
        """Calculate basic grammar score."""
        grammar_score = 1.0
        
        # Check for common grammar patterns
        # Double spaces
        if '  ' in text:
            grammar_score *= 0.95
        
        # Sentence endings
        if not text.strip().endswith(('.', '!', '?')):
            grammar_score *= 0.9
        
        # Capitalization after periods
        sentences = text.split('. ')
        for i in range(1, len(sentences)):
            if sentences[i] and not sentences[i][0].isupper():
                grammar_score *= 0.98
        
        # Basic subject-verb patterns (very simplified)
        # This would need a proper parser for accurate results
        common_patterns = [
            r'\b(I|you|he|she|it|we|they)\s+(am|is|are|was|were)\b',
            r'\b(the|a|an)\s+\w+\s+(is|are|was|were)\b'
        ]
        
        pattern_found = False
        for pattern in common_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                pattern_found = True
                break
        
        if not pattern_found and len(text) > 100:
            grammar_score *= 0.95
        
        return grammar_score
    
    def _calculate_repetition_ratio(self, text: str, tokens: List[str]) -> float:
        """Calculate repetition ratio for n-grams."""
        if len(tokens) < 3:
            return 0.0
        
        repetition_scores = []
        
        # Check different n-gram sizes
        for n in [2, 3, 4]:
            if len(tokens) < n:
                continue
            
            ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
            ngram_counts = Counter(ngrams)
            
            # Count repeated n-grams
            repeated = sum(1 for count in ngram_counts.values() if count > 1)
            repetition_ratio = repeated / max(len(ngrams), 1)
            repetition_scores.append(repetition_ratio)
        
        return np.mean(repetition_scores) if repetition_scores else 0.0
    
    def _calculate_overall_score(self, metrics: QualityMetrics) -> float:
        """Calculate overall quality score."""
        # Weight different components
        weights = {
            'readability': 0.2,
            'coherence': 0.2,
            'fluency': 0.15,
            'vocabulary': 0.15,
            'grammar': 0.15,
            'complexity': 0.1,
            'anomaly': 0.05
        }
        
        # Calculate anomaly penalty
        anomaly_penalty = 1.0
        if metrics.repetition_ratio > self.max_repetition_ratio:
            anomaly_penalty *= 0.8
        if metrics.special_char_ratio > self.max_special_char_ratio:
            anomaly_penalty *= 0.9
        if metrics.uppercase_ratio > 0.3:
            anomaly_penalty *= 0.95
        
        # Calculate weighted score
        score = (
            weights['readability'] * metrics.readability_score +
            weights['coherence'] * metrics.coherence_score +
            weights['fluency'] * metrics.fluency_score +
            weights['vocabulary'] * metrics.vocabulary_diversity +
            weights['grammar'] * metrics.grammar_score +
            weights['complexity'] * min(metrics.sentence_complexity, 1.0) +
            weights['anomaly'] * anomaly_penalty
        )
        
        # Apply length penalty
        if metrics.length < self.min_length:
            score *= 0.5
        elif metrics.length > self.max_length:
            score *= 0.8
        
        return score
    
    def _get_quality_flags(self, metrics: QualityMetrics) -> List[str]:
        """Get quality flags for the text."""
        flags = []
        
        if metrics.length < self.min_length:
            flags.append("too_short")
        elif metrics.length > self.max_length:
            flags.append("too_long")
        
        if metrics.num_tokens < self.min_tokens:
            flags.append("insufficient_tokens")
        
        if metrics.repetition_ratio > self.max_repetition_ratio:
            flags.append("high_repetition")
        
        if metrics.special_char_ratio > self.max_special_char_ratio:
            flags.append("high_special_chars")
        
        if metrics.vocabulary_diversity < self.min_vocabulary_diversity:
            flags.append("low_vocabulary_diversity")
        
        if metrics.readability_score < 0.3:
            flags.append("low_readability")
        
        if metrics.coherence_score < 0.3:
            flags.append("low_coherence")
        
        if metrics.grammar_score < 0.5:
            flags.append("grammar_issues")
        
        return flags
    
    def filter_batch(
        self,
        texts: List[str],
        min_quality_score: float = 0.7
    ) -> Tuple[List[str], List[QualityMetrics]]:
        """Filter a batch of texts based on quality."""
        filtered_texts = []
        metrics_list = []
        
        for text in texts:
            metrics = self.evaluate(text)
            if metrics.overall_score >= min_quality_score and not metrics.quality_flags:
                filtered_texts.append(text)
                metrics_list.append(metrics)
        
        return filtered_texts, metrics_list
    
    def get_statistics(self, metrics_list: List[QualityMetrics]) -> Dict[str, float]:
        """Get statistics from a list of quality metrics."""
        if not metrics_list:
            return {}
        
        stats = {
            'avg_length': statistics.mean(m.length for m in metrics_list),
            'avg_tokens': statistics.mean(m.num_tokens for m in metrics_list),
            'avg_readability': statistics.mean(m.readability_score for m in metrics_list),
            'avg_coherence': statistics.mean(m.coherence_score for m in metrics_list),
            'avg_fluency': statistics.mean(m.fluency_score for m in metrics_list),
            'avg_vocabulary_diversity': statistics.mean(m.vocabulary_diversity for m in metrics_list),
            'avg_overall_score': statistics.mean(m.overall_score for m in metrics_list),
            'min_overall_score': min(m.overall_score for m in metrics_list),
            'max_overall_score': max(m.overall_score for m in metrics_list)
        }
        
        # Count quality flags
        all_flags = []
        for m in metrics_list:
            all_flags.extend(m.quality_flags)
        
        flag_counts = Counter(all_flags)
        for flag, count in flag_counts.items():
            stats[f'flag_{flag}_count'] = count
        
        return stats