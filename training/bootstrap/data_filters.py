"""Data Quality Filters for Bootstrap Training.

This module implements comprehensive data filtering and quality assessment
for the self-training pipeline, ensuring high-quality seed data.
"""

import re
import json
import hashlib
from typing import Dict, List, Set, Tuple, Optional, Any, Iterator
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from collections import Counter, defaultdict
import numpy as np

from ...core.blt.entropy import calculate_byte_entropy
from ...core.blt.patching import ByteProcessor

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Container for document quality metrics."""
    # Basic metrics
    length: int
    char_count: int
    word_count: int
    line_count: int
    
    # Language and encoding
    language: str
    language_confidence: float
    encoding: str
    is_valid_utf8: bool
    
    # Content quality
    entropy: float
    repetition_ratio: float
    punctuation_ratio: float
    uppercase_ratio: float
    digit_ratio: float
    special_char_ratio: float
    
    # Structure quality
    avg_word_length: float
    avg_sentence_length: float
    paragraph_count: int
    
    # Content type classification
    content_type: str  # 'text', 'code', 'structured', 'mixed'
    content_confidence: float
    
    # Quality scores
    readability_score: float
    diversity_score: float
    overall_quality: float
    
    # Flags
    is_duplicate: bool = False
    is_spam: bool = False
    is_toxic: bool = False
    has_pii: bool = False


class LanguageDetector:
    """Simple language detection based on character frequencies."""
    
    def __init__(self):
        # Character frequency patterns for major languages
        self.language_patterns = {
            'en': {
                'common_chars': set('etaoinshrdlucmfwypvbgkqjxz'),
                'common_words': {'the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'that'},
                'char_ranges': [(ord('a'), ord('z')), (ord('A'), ord('Z'))]
            },
            'es': {
                'common_chars': set('eaosrnidltucmpbvhgfyqzjxñáéíóúü'),
                'common_words': {'el', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no'},
                'char_ranges': [(ord('a'), ord('z')), (ord('A'), ord('Z'))]
            },
            'fr': {
                'common_chars': set('esaitnrulodcpmvqgbfhzjyxwkàâäçéèêëïîôùûüÿæœ'),
                'common_words': {'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir'},
                'char_ranges': [(ord('a'), ord('z')), (ord('A'), ord('Z'))]
            },
            'zh': {
                'common_chars': set(),  # Will use Unicode ranges
                'common_words': set(),
                'char_ranges': [(0x4e00, 0x9fff)]  # CJK Unified Ideographs
            },
            'ja': {
                'common_chars': set(),
                'common_words': set(),
                'char_ranges': [(0x3040, 0x309f), (0x30a0, 0x30ff), (0x4e00, 0x9fff)]  # Hiragana, Katakana, Kanji
            },
            'ar': {
                'common_chars': set(),
                'common_words': set(),
                'char_ranges': [(0x0600, 0x06ff)]  # Arabic
            },
            'ru': {
                'common_chars': set('оеаинтсрвлкмдпуяыьгзбчйхжшюцщэфёъ'),
                'common_words': {'и', 'в', 'не', 'на', 'я', 'быть', 'с', 'он', 'а', 'то'},
                'char_ranges': [(0x0400, 0x04ff)]  # Cyrillic
            }
        }
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language of text with confidence score."""
        if not text.strip():
            return 'unknown', 0.0
        
        text_lower = text.lower()
        char_counts = Counter(text_lower)
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        language_scores = {}
        
        for lang, patterns in self.language_patterns.items():
            score = 0.0
            
            # Character frequency scoring
            if patterns['common_chars']:
                common_char_count = sum(char_counts.get(c, 0) for c in patterns['common_chars'])
                score += common_char_count / len(text) * 0.4
            
            # Unicode range scoring
            range_chars = 0
            for start, end in patterns['char_ranges']:
                range_chars += sum(1 for c in text if start <= ord(c) <= end)
            score += range_chars / len(text) * 0.4
            
            # Common words scoring
            if patterns['common_words']:
                word_matches = len(words & patterns['common_words'])
                score += word_matches / max(len(words), 1) * 0.2
            
            language_scores[lang] = score
        
        if not language_scores:
            return 'unknown', 0.0
        
        best_lang = max(language_scores, key=language_scores.get)
        confidence = language_scores[best_lang]
        
        return best_lang, min(confidence, 1.0)


class ContentTypeClassifier:
    """Classify content type (text, code, structured data, etc.)."""
    
    def __init__(self):
        # Code indicators
        self.code_patterns = [
            r'def\s+\w+\s*\(',  # Python functions
            r'function\s+\w+\s*\(',  # JavaScript functions
            r'class\s+\w+\s*\{',  # Class definitions
            r'import\s+\w+',  # Import statements
            r'#include\s*<',  # C/C++ includes
            r'<\?php',  # PHP tags
            r'public\s+class\s+\w+',  # Java classes
            r'console\.log\(',  # JavaScript console
            r'System\.out\.print',  # Java print
            r'printf\s*\(',  # C printf
        ]
        
        # Structured data patterns
        self.structured_patterns = [
            r'^\s*\{.*\}\s*$',  # JSON-like
            r'^\s*<\w+.*</\w+>\s*$',  # XML-like
            r'^\w+:\s*\w+',  # Key-value pairs
            r'^\s*\|.*\|.*\|',  # Table format
            r'^\s*\w+,\w+,\w+',  # CSV-like
        ]
    
    def classify_content(self, text: str) -> Tuple[str, float]:
        """Classify content type with confidence."""
        if not text.strip():
            return 'empty', 1.0
        
        lines = text.split('\n')
        total_lines = len(lines)
        
        # Check for code patterns
        code_matches = 0
        for pattern in self.code_patterns:
            code_matches += len(re.findall(pattern, text, re.IGNORECASE))
        
        code_ratio = code_matches / max(total_lines, 1)
        
        # Check for structured data patterns
        structured_matches = 0
        for pattern in self.structured_patterns:
            for line in lines:
                if re.match(pattern, line.strip()):
                    structured_matches += 1
        
        structured_ratio = structured_matches / total_lines
        
        # Determine content type
        if code_ratio > 0.1:
            return 'code', min(code_ratio * 2, 1.0)
        elif structured_ratio > 0.5:
            return 'structured', structured_ratio
        elif structured_ratio > 0.2:
            return 'mixed', structured_ratio
        else:
            return 'text', 1.0 - code_ratio - structured_ratio


class DataQualityFilter:
    """Comprehensive data quality filter for training data."""
    
    def __init__(
        self,
        min_length: int = 100,
        max_length: int = 50000,
        min_entropy: float = 0.3,
        max_entropy: float = 0.9,
        max_repetition_ratio: float = 0.3,
        min_readability: float = 0.3,
        enable_toxicity_filter: bool = True,
        enable_pii_filter: bool = True
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.min_entropy = min_entropy
        self.max_entropy = max_entropy
        self.max_repetition_ratio = max_repetition_ratio
        self.min_readability = min_readability
        self.enable_toxicity_filter = enable_toxicity_filter
        self.enable_pii_filter = enable_pii_filter
        
        # Initialize components
        self.byte_processor = ByteProcessor()
        self.language_detector = LanguageDetector()
        self.content_classifier = ContentTypeClassifier()
        
        # Toxicity patterns (simplified)
        self.toxicity_patterns = [
            r'\b(hate|stupid|idiot|kill|die|death)\b',
            r'\b(spam|scam|fake|fraud|cheat)\b',
        ]
        
        # PII patterns
        self.pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{4}-\d{4}-\d{4}-\d{4}\b',  # Credit card pattern
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone number
        ]
        
        # Duplicate detection
        self.seen_hashes = set()
        self.seen_content = set()
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'accepted': 0,
            'rejected_length': 0,
            'rejected_entropy': 0,
            'rejected_repetition': 0,
            'rejected_readability': 0,
            'rejected_toxicity': 0,
            'rejected_pii': 0,
            'rejected_duplicate': 0,
            'language_distribution': defaultdict(int),
            'content_type_distribution': defaultdict(int)
        }
    
    def calculate_metrics(self, text: str) -> QualityMetrics:
        """Calculate comprehensive quality metrics for text."""
        # Basic metrics
        length = len(text)
        char_count = len(text)
        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)
        lines = text.split('\n')
        line_count = len(lines)
        
        # Language detection
        language, lang_confidence = self.language_detector.detect_language(text)
        
        # Encoding validation
        try:
            text_bytes = text.encode('utf-8')
            is_valid_utf8 = self.byte_processor.is_valid_utf8(text_bytes)
            encoding = 'utf-8'
        except UnicodeEncodeError:
            is_valid_utf8 = False
            encoding = 'unknown'
        
        # Entropy calculation
        if is_valid_utf8:
            entropy = calculate_byte_entropy(text_bytes)
        else:
            entropy = 0.0
        
        # Character ratios
        uppercase_count = sum(1 for c in text if c.isupper())
        digit_count = sum(1 for c in text if c.isdigit())
        punct_count = sum(1 for c in text if c in '.,!?;:')
        special_count = sum(1 for c in text if not c.isalnum() and not c.isspace() and c not in '.,!?;:')
        
        uppercase_ratio = uppercase_count / max(char_count, 1)
        digit_ratio = digit_count / max(char_count, 1)
        punctuation_ratio = punct_count / max(char_count, 1)
        special_char_ratio = special_count / max(char_count, 1)
        
        # Repetition analysis
        repetition_ratio = self._calculate_repetition_ratio(text)
        
        # Structure metrics
        avg_word_length = sum(len(word) for word in words) / max(word_count, 1)
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        paragraphs = text.split('\n\n')
        paragraph_count = len([p for p in paragraphs if p.strip()])
        
        # Content type classification
        content_type, content_confidence = self.content_classifier.classify_content(text)
        
        # Quality scores
        readability_score = self._calculate_readability(text, words, sentences)
        diversity_score = self._calculate_diversity(text)
        
        # Flags
        is_spam = self._detect_spam(text)
        is_toxic = self._detect_toxicity(text)
        has_pii = self._detect_pii(text)
        
        # Overall quality score
        overall_quality = self._calculate_overall_quality(
            entropy, repetition_ratio, readability_score, diversity_score,
            is_spam, is_toxic, has_pii
        )
        
        return QualityMetrics(
            length=length,
            char_count=char_count,
            word_count=word_count,
            line_count=line_count,
            language=language,
            language_confidence=lang_confidence,
            encoding=encoding,
            is_valid_utf8=is_valid_utf8,
            entropy=entropy,
            repetition_ratio=repetition_ratio,
            punctuation_ratio=punctuation_ratio,
            uppercase_ratio=uppercase_ratio,
            digit_ratio=digit_ratio,
            special_char_ratio=special_char_ratio,
            avg_word_length=avg_word_length,
            avg_sentence_length=avg_sentence_length,
            paragraph_count=paragraph_count,
            content_type=content_type,
            content_confidence=content_confidence,
            readability_score=readability_score,
            diversity_score=diversity_score,
            overall_quality=overall_quality,
            is_spam=is_spam,
            is_toxic=is_toxic,
            has_pii=has_pii
        )
    
    def _calculate_repetition_ratio(self, text: str) -> float:
        """Calculate repetition ratio in text."""
        words = text.lower().split()
        if len(words) < 10:
            return 0.0
        
        # Check for repeated n-grams
        repetition_count = 0
        window_size = min(5, len(words) // 4)
        
        for i in range(len(words) - window_size):
            ngram = ' '.join(words[i:i + window_size])
            count = text.lower().count(ngram)
            if count > 1:
                repetition_count += count - 1
        
        return repetition_count / max(len(words), 1)
    
    def _calculate_readability(self, text: str, words: List[str], sentences: List[str]) -> float:
        """Calculate readability score (simplified Flesch Reading Ease)."""
        if not words or not sentences:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables = sum(self._count_syllables(word) for word in words) / len(words)
        
        # Simplified Flesch formula
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        return max(0.0, min(score / 100.0, 1.0))
    
    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count in a word."""
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel
        
        # Adjust for silent e
        if word.endswith('e') and count > 1:
            count -= 1
        
        return max(count, 1)
    
    def _calculate_diversity(self, text: str) -> float:
        """Calculate lexical diversity (Type-Token Ratio)."""
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0
        
        unique_words = set(words)
        return len(unique_words) / len(words)
    
    def _detect_spam(self, text: str) -> bool:
        """Detect spam-like content."""
        # Simple spam indicators
        spam_indicators = [
            text.count('!') > 10,
            text.count('$') > 5,
            'click here' in text.lower(),
            'buy now' in text.lower(),
            'limited time' in text.lower(),
            len(re.findall(r'[A-Z]{3,}', text)) > 5  # Too many caps
        ]
        
        return sum(spam_indicators) >= 2
    
    def _detect_toxicity(self, text: str) -> bool:
        """Detect toxic content."""
        if not self.enable_toxicity_filter:
            return False
        
        text_lower = text.lower()
        for pattern in self.toxicity_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _detect_pii(self, text: str) -> bool:
        """Detect personally identifiable information."""
        if not self.enable_pii_filter:
            return False
        
        for pattern in self.pii_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _calculate_overall_quality(
        self,
        entropy: float,
        repetition_ratio: float,
        readability_score: float,
        diversity_score: float,
        is_spam: bool,
        is_toxic: bool,
        has_pii: bool
    ) -> float:
        """Calculate overall quality score."""
        score = 0.0
        
        # Entropy contribution (0.3 weight)
        if self.min_entropy <= entropy <= self.max_entropy:
            score += 0.3
        else:
            score += 0.3 * (1.0 - abs(entropy - (self.min_entropy + self.max_entropy) / 2))
        
        # Repetition contribution (0.2 weight)
        score += 0.2 * (1.0 - min(repetition_ratio / self.max_repetition_ratio, 1.0))
        
        # Readability contribution (0.2 weight)
        score += 0.2 * readability_score
        
        # Diversity contribution (0.2 weight)
        score += 0.2 * diversity_score
        
        # Penalty for negative flags (0.1 weight)
        penalty = 0.0
        if is_spam:
            penalty += 0.5
        if is_toxic:
            penalty += 0.5
        if has_pii:
            penalty += 0.3
        
        score -= 0.1 * penalty
        
        return max(0.0, min(score, 1.0))
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is a duplicate."""
        # Quick hash check
        text_hash = hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()
        if text_hash in self.seen_hashes:
            return True
        
        # Content similarity check (simplified)
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        if normalized in self.seen_content:
            return True
        
        # Store hashes
        self.seen_hashes.add(text_hash)
        self.seen_content.add(normalized)
        
        return False
    
    def filter_document(self, text: str) -> Tuple[bool, QualityMetrics]:
        """Filter a single document.
        
        Returns:
            Tuple of (is_accepted, quality_metrics)
        """
        self.stats['total_processed'] += 1
        
        # Calculate metrics
        metrics = self.calculate_metrics(text)
        
        # Check for duplicates
        if self.is_duplicate(text):
            metrics.is_duplicate = True
            self.stats['rejected_duplicate'] += 1
            return False, metrics
        
        # Length filter
        if not (self.min_length <= metrics.length <= self.max_length):
            self.stats['rejected_length'] += 1
            return False, metrics
        
        # Entropy filter
        if not (self.min_entropy <= metrics.entropy <= self.max_entropy):
            self.stats['rejected_entropy'] += 1
            return False, metrics
        
        # Repetition filter
        if metrics.repetition_ratio > self.max_repetition_ratio:
            self.stats['rejected_repetition'] += 1
            return False, metrics
        
        # Readability filter
        if metrics.readability_score < self.min_readability:
            self.stats['rejected_readability'] += 1
            return False, metrics
        
        # Toxicity filter
        if metrics.is_toxic:
            self.stats['rejected_toxicity'] += 1
            return False, metrics
        
        # PII filter
        if metrics.has_pii:
            self.stats['rejected_pii'] += 1
            return False, metrics
        
        # Document accepted
        self.stats['accepted'] += 1
        self.stats['language_distribution'][metrics.language] += 1
        self.stats['content_type_distribution'][metrics.content_type] += 1
        
        return True, metrics
    
    def filter_batch(
        self,
        documents: List[str],
        return_metrics: bool = False
    ) -> List[Union[str, Tuple[str, QualityMetrics]]]:
        """Filter a batch of documents.
        
        Args:
            documents: List of text documents
            return_metrics: Whether to return metrics with documents
            
        Returns:
            List of accepted documents (with metrics if requested)
        """
        filtered = []
        
        for doc in documents:
            is_accepted, metrics = self.filter_document(doc)
            
            if is_accepted:
                if return_metrics:
                    filtered.append((doc, metrics))
                else:
                    filtered.append(doc)
        
        return filtered
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get filtering statistics."""
        if self.stats['total_processed'] > 0:
            acceptance_rate = self.stats['accepted'] / self.stats['total_processed']
        else:
            acceptance_rate = 0.0
        
        return {
            'total_processed': self.stats['total_processed'],
            'accepted': self.stats['accepted'],
            'acceptance_rate': acceptance_rate,
            'rejection_reasons': {
                'length': self.stats['rejected_length'],
                'entropy': self.stats['rejected_entropy'],
                'repetition': self.stats['rejected_repetition'],
                'readability': self.stats['rejected_readability'],
                'toxicity': self.stats['rejected_toxicity'],
                'pii': self.stats['rejected_pii'],
                'duplicate': self.stats['rejected_duplicate']
            },
            'language_distribution': dict(self.stats['language_distribution']),
            'content_type_distribution': dict(self.stats['content_type_distribution'])
        }
    
    def save_statistics(self, filepath: str):
        """Save statistics to file."""
        stats = self.get_statistics()
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)


def create_quality_filter(config: Dict[str, Any]) -> DataQualityFilter:
    """Create a quality filter from configuration."""
    return DataQualityFilter(**config)


def filter_dataset(
    input_path: str,
    output_path: str,
    filter_config: Optional[Dict[str, Any]] = None,
    batch_size: int = 1000,
    save_metrics: bool = True
) -> Dict[str, Any]:
    """Filter an entire dataset.
    
    Args:
        input_path: Path to input dataset
        output_path: Path to save filtered dataset
        filter_config: Filter configuration
        batch_size: Processing batch size
        save_metrics: Whether to save metrics
        
    Returns:
        Filtering statistics
    """
    # Create filter
    filter_config = filter_config or {}
    quality_filter = create_quality_filter(filter_config)
    
    # Process dataset
    input_file = Path(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        batch = []
        for line in infile:
            try:
                data = json.loads(line.strip())
                text = data.get('text', '')
                batch.append((data, text))
                
                if len(batch) >= batch_size:
                    # Process batch
                    _process_batch(batch, quality_filter, outfile, save_metrics)
                    batch = []
                    
            except json.JSONDecodeError:
                continue
        
        # Process remaining batch
        if batch:
            _process_batch(batch, quality_filter, outfile, save_metrics)
    
    # Save statistics
    stats = quality_filter.get_statistics()
    stats_path = output_file.with_suffix('.stats.json')
    quality_filter.save_statistics(str(stats_path))
    
    logger.info(f"Filtered {stats['total_processed']} documents, "
               f"accepted {stats['accepted']} ({stats['acceptance_rate']:.2%})")
    
    return stats


def _process_batch(
    batch: List[Tuple[Dict, str]],
    quality_filter: DataQualityFilter,
    outfile,
    save_metrics: bool
):
    """Process a batch of documents."""
    for data, text in batch:
        is_accepted, metrics = quality_filter.filter_document(text)
        
        if is_accepted:
            # Add quality metrics to data if requested
            if save_metrics:
                data['quality_metrics'] = asdict(metrics)
            
            # Write to output
            outfile.write(json.dumps(data) + '\n')