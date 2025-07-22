"""
Web Data Compression System

Efficient compression for web content storage with specialized algorithms
for different content types and consciousness-aware compression strategies.
"""

import gzip
import zlib
import lz4.frame
import brotli
import json
import pickle
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import hashlib
import numpy as np
from collections import defaultdict, Counter
import logging

from web.connectivity.web_monitor import WebContent

logger = logging.getLogger(__name__)


@dataclass
class CompressionConfig:
    """Configuration for web content compression"""
    # Compression algorithms to use by priority
    compression_algorithms: List[str] = field(default_factory=lambda: ['brotli', 'lz4', 'gzip', 'zlib'])
    
    # Content type specific settings
    text_compression_level: int = 6
    json_compression_level: int = 9  
    html_compression_level: int = 6
    default_compression_level: int = 6
    
    # Size thresholds
    min_compression_size: int = 100  # Don't compress content smaller than this
    max_content_size: int = 10 * 1024 * 1024  # 10MB limit
    
    # Semantic compression settings
    enable_semantic_compression: bool = True
    semantic_similarity_threshold: float = 0.8
    semantic_deduplication: bool = True
    
    # Cache settings
    compression_cache_size: int = 1000
    enable_compression_stats: bool = True
    
    # Consciousness-aware settings
    consciousness_content_weight: float = 1.5  # Higher weight for consciousness content
    preserve_consciousness_keywords: bool = True


@dataclass
class CompressionResult:
    """Result of compression operation"""
    original_size: int
    compressed_size: int
    compression_ratio: float
    algorithm_used: str
    compression_time: float
    is_semantic: bool = False
    semantic_similarity: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticCompressor:
    """Performs semantic compression and deduplication"""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        # Simple word frequency based semantic compression
        self.common_phrases = {}
        self.phrase_replacements = {}
        self.consciousness_keywords = {
            'consciousness', 'aware', 'experience', 'subjective', 'qualia',
            'phenomenal', 'introspection', 'self-aware', 'sentient', 'cognitive'
        }
    
    def learn_common_phrases(self, content_list: List[str]):
        """Learn common phrases for compression"""
        
        phrase_counter = Counter()
        
        for content in content_list:
            # Extract phrases (3-5 words)
            words = content.lower().split()
            
            for i in range(len(words) - 2):
                for phrase_len in [3, 4, 5]:
                    if i + phrase_len <= len(words):
                        phrase = ' '.join(words[i:i + phrase_len])
                        if len(phrase) > 15:  # Only count substantial phrases
                            phrase_counter[phrase] += 1
        
        # Keep phrases that occur frequently
        min_frequency = max(2, len(content_list) // 10)
        self.common_phrases = {
            phrase: f"__P{idx}__"
            for idx, (phrase, count) in enumerate(phrase_counter.most_common(200))
            if count >= min_frequency
        }
        
        self.phrase_replacements = {v: k for k, v in self.common_phrases.items()}
        logger.info(f"Learned {len(self.common_phrases)} common phrases for semantic compression")
    
    def compress_semantically(self, content: str) -> Tuple[str, float, Dict[str, Any]]:
        """Apply semantic compression to content"""
        
        if not self.config.enable_semantic_compression:
            return content, 0.0, {}
        
        original_length = len(content)
        compressed_content = content
        consciousness_score = 0.0
        
        # Calculate consciousness score for special handling
        content_lower = content.lower()
        consciousness_matches = sum(1 for kw in self.consciousness_keywords if kw in content_lower)
        consciousness_score = min(1.0, consciousness_matches / len(self.consciousness_keywords))
        
        # Apply phrase replacement (if not consciousness content or preserve setting is off)
        if consciousness_score < 0.3 or not self.config.preserve_consciousness_keywords:
            for phrase, replacement in self.common_phrases.items():
                if phrase in compressed_content.lower():
                    # Case-insensitive replacement while preserving original case
                    import re
                    pattern = re.escape(phrase)
                    compressed_content = re.sub(
                        pattern, replacement, compressed_content, 
                        flags=re.IGNORECASE
                    )
        
        # Remove redundant whitespace
        import re
        compressed_content = re.sub(r'\s+', ' ', compressed_content)
        compressed_content = compressed_content.strip()
        
        compression_ratio = 1.0 - (len(compressed_content) / original_length) if original_length > 0 else 0.0
        
        metadata = {
            'consciousness_score': consciousness_score,
            'phrases_replaced': len([p for p in self.common_phrases.keys() if p in content.lower()]),
            'original_length': original_length,
            'semantic_compressed_length': len(compressed_content)
        }
        
        return compressed_content, compression_ratio, metadata
    
    def decompress_semantically(self, compressed_content: str, metadata: Dict[str, Any]) -> str:
        """Decompress semantically compressed content"""
        
        if not self.config.enable_semantic_compression:
            return compressed_content
        
        # Restore phrases
        decompressed = compressed_content
        for replacement, phrase in self.phrase_replacements.items():
            decompressed = decompressed.replace(replacement, phrase)
        
        return decompressed
    
    def find_similar_content(
        self, 
        new_content: str, 
        existing_content: List[Tuple[str, Dict[str, Any]]]
    ) -> Optional[Tuple[str, float, Dict[str, Any]]]:
        """Find similar existing content for deduplication"""
        
        if not self.config.semantic_deduplication:
            return None
        
        new_words = set(new_content.lower().split())
        if len(new_words) < 5:  # Too short for meaningful comparison
            return None
        
        best_similarity = 0.0
        best_content = None
        best_metadata = None
        
        for existing, metadata in existing_content:
            existing_words = set(existing.lower().split())
            
            if len(existing_words) < 5:
                continue
            
            # Jaccard similarity
            intersection = new_words.intersection(existing_words)
            union = new_words.union(existing_words)
            
            similarity = len(intersection) / len(union) if union else 0.0
            
            if similarity > best_similarity and similarity >= self.config.semantic_similarity_threshold:
                best_similarity = similarity
                best_content = existing
                best_metadata = metadata
        
        if best_content:
            return best_content, best_similarity, best_metadata
        
        return None


class WebContentCompressor:
    """Main web content compression system"""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.semantic_compressor = SemanticCompressor(config)
        
        # Compression statistics
        self.stats = {
            'total_compressed': 0,
            'total_original_size': 0,
            'total_compressed_size': 0,
            'algorithm_usage': defaultdict(int),
            'content_type_stats': defaultdict(lambda: {'count': 0, 'avg_ratio': 0.0}),
            'semantic_compressions': 0,
            'deduplications': 0
        }
        
        # Cache for compressed content
        self.compression_cache: Dict[str, Tuple[bytes, CompressionResult]] = {}
        self.semantic_cache: List[Tuple[str, Dict[str, Any]]] = []
    
    def compress_web_content(self, content: WebContent) -> Tuple[bytes, CompressionResult]:
        """Compress web content with optimal algorithm selection"""
        
        # Generate cache key
        cache_key = hashlib.md5(f"{content.url}{content.content}".encode()).hexdigest()
        
        # Check cache first
        if cache_key in self.compression_cache:
            logger.debug(f"Using cached compression for {content.url}")
            return self.compression_cache[cache_key]
        
        start_time = datetime.now()
        text_content = content.content
        original_size = len(text_content.encode('utf-8'))
        
        # Skip compression for very small content
        if original_size < self.config.min_compression_size:
            result = CompressionResult(
                original_size=original_size,
                compressed_size=original_size,
                compression_ratio=0.0,
                algorithm_used='none',
                compression_time=0.0,
                metadata={'reason': 'content_too_small'}
            )
            return text_content.encode('utf-8'), result
        
        # Skip compression for very large content
        if original_size > self.config.max_content_size:
            logger.warning(f"Content too large for compression: {original_size} bytes")
            result = CompressionResult(
                original_size=original_size,
                compressed_size=original_size,
                compression_ratio=0.0,
                algorithm_used='none',
                compression_time=0.0,
                metadata={'reason': 'content_too_large'}
            )
            return text_content.encode('utf-8'), result
        
        # Check for semantic deduplication
        similar_content = self.semantic_compressor.find_similar_content(
            text_content, self.semantic_cache[-100:]  # Check recent content
        )
        
        if similar_content:
            similar_text, similarity, similar_metadata = similar_content
            
            # Store reference to similar content instead
            reference_data = {
                'type': 'semantic_reference',
                'similarity': similarity,
                'reference_hash': hashlib.md5(similar_text.encode()).hexdigest(),
                'differences': self._compute_differences(text_content, similar_text)
            }
            
            reference_bytes = json.dumps(reference_data).encode('utf-8')
            compressed_size = len(reference_bytes)
            compression_ratio = 1.0 - (compressed_size / original_size)
            
            result = CompressionResult(
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=compression_ratio,
                algorithm_used='semantic_deduplication',
                compression_time=(datetime.now() - start_time).total_seconds(),
                is_semantic=True,
                semantic_similarity=similarity,
                metadata=reference_data
            )
            
            self.stats['deduplications'] += 1
            return reference_bytes, result
        
        # Apply semantic compression first
        semantically_compressed, semantic_ratio, semantic_metadata = self.semantic_compressor.compress_semantically(text_content)
        
        if semantic_ratio > 0.1:  # Significant semantic compression achieved
            self.stats['semantic_compressions'] += 1
            text_content = semantically_compressed
        
        # Select compression level based on content type
        compression_level = self._get_compression_level(content.content_type)
        
        # Try different compression algorithms
        best_result = None
        best_compressed = None
        
        for algorithm in self.config.compression_algorithms:
            try:
                if algorithm == 'gzip':
                    compressed = gzip.compress(text_content.encode('utf-8'), compresslevel=compression_level)
                elif algorithm == 'zlib':
                    compressed = zlib.compress(text_content.encode('utf-8'), level=compression_level)
                elif algorithm == 'lz4':
                    compressed = lz4.frame.compress(text_content.encode('utf-8'), compression_level=compression_level)
                elif algorithm == 'brotli':
                    compressed = brotli.compress(text_content.encode('utf-8'), quality=compression_level)
                else:
                    continue
                
                compressed_size = len(compressed)
                ratio = 1.0 - (compressed_size / original_size)
                
                if best_result is None or compressed_size < best_result.compressed_size:
                    compression_time = (datetime.now() - start_time).total_seconds()
                    
                    best_result = CompressionResult(
                        original_size=original_size,
                        compressed_size=compressed_size,
                        compression_ratio=ratio,
                        algorithm_used=algorithm,
                        compression_time=compression_time,
                        is_semantic=semantic_ratio > 0.1,
                        metadata={
                            'compression_level': compression_level,
                            'semantic_compression_ratio': semantic_ratio,
                            **semantic_metadata
                        }
                    )
                    best_compressed = compressed
                    
            except Exception as e:
                logger.error(f"Compression failed with {algorithm}: {e}")
                continue
        
        # Fallback to no compression if all algorithms failed
        if best_result is None:
            best_compressed = text_content.encode('utf-8')
            best_result = CompressionResult(
                original_size=original_size,
                compressed_size=original_size,
                compression_ratio=0.0,
                algorithm_used='none',
                compression_time=(datetime.now() - start_time).total_seconds(),
                metadata={'reason': 'compression_failed'}
            )
        
        # Update statistics
        self._update_stats(content.content_type, best_result)
        
        # Cache result
        if len(self.compression_cache) < self.config.compression_cache_size:
            self.compression_cache[cache_key] = (best_compressed, best_result)
        
        # Add to semantic cache
        if len(self.semantic_cache) < 1000:
            self.semantic_cache.append((content.content, semantic_metadata))
        
        logger.debug(f"Compressed {content.url}: {original_size} -> {best_result.compressed_size} bytes "
                    f"({best_result.compression_ratio:.2%} reduction, {best_result.algorithm_used})")
        
        return best_compressed, best_result
    
    def decompress_web_content(self, compressed_data: bytes, compression_result: CompressionResult) -> str:
        """Decompress web content"""
        
        algorithm = compression_result.algorithm_used
        
        try:
            # Handle semantic deduplication references
            if algorithm == 'semantic_deduplication':
                reference_data = json.loads(compressed_data.decode('utf-8'))
                # In a real implementation, would look up the referenced content
                # For now, return a placeholder
                return f"[SEMANTIC REFERENCE: similarity={reference_data['similarity']:.3f}]"
            
            # Handle regular compression algorithms
            if algorithm == 'gzip':
                decompressed = gzip.decompress(compressed_data).decode('utf-8')
            elif algorithm == 'zlib':
                decompressed = zlib.decompress(compressed_data).decode('utf-8')
            elif algorithm == 'lz4':
                decompressed = lz4.frame.decompress(compressed_data).decode('utf-8')
            elif algorithm == 'brotli':
                decompressed = brotli.decompress(compressed_data).decode('utf-8')
            elif algorithm == 'none':
                decompressed = compressed_data.decode('utf-8')
            else:
                raise ValueError(f"Unknown compression algorithm: {algorithm}")
            
            # Apply semantic decompression if needed
            if compression_result.is_semantic:
                metadata = compression_result.metadata
                decompressed = self.semantic_compressor.decompress_semantically(decompressed, metadata)
            
            return decompressed
            
        except Exception as e:
            logger.error(f"Decompression failed for algorithm {algorithm}: {e}")
            raise
    
    def _get_compression_level(self, content_type: str) -> int:
        """Get appropriate compression level for content type"""
        
        if content_type == 'json':
            return self.config.json_compression_level
        elif content_type in ['html', 'news', 'article']:
            return self.config.html_compression_level
        elif content_type in ['text', 'social']:
            return self.config.text_compression_level
        else:
            return self.config.default_compression_level
    
    def _compute_differences(self, content1: str, content2: str) -> List[str]:
        """Compute differences between two pieces of content for semantic references"""
        
        # Simple word-level differences
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        differences = list(words1.symmetric_difference(words2))
        return differences[:50]  # Limit size
    
    def _update_stats(self, content_type: str, result: CompressionResult):
        """Update compression statistics"""
        
        if not self.config.enable_compression_stats:
            return
        
        self.stats['total_compressed'] += 1
        self.stats['total_original_size'] += result.original_size
        self.stats['total_compressed_size'] += result.compressed_size
        self.stats['algorithm_usage'][result.algorithm_used] += 1
        
        # Update content type stats
        type_stats = self.stats['content_type_stats'][content_type]
        prev_avg = type_stats['avg_ratio']
        prev_count = type_stats['count']
        
        type_stats['count'] += 1
        type_stats['avg_ratio'] = (prev_avg * prev_count + result.compression_ratio) / type_stats['count']
    
    def train_semantic_compressor(self, training_content: List[WebContent]):
        """Train the semantic compressor with sample content"""
        
        content_texts = [content.content for content in training_content]
        self.semantic_compressor.learn_common_phrases(content_texts)
        
        logger.info(f"Trained semantic compressor with {len(content_texts)} content items")
    
    def get_compression_statistics(self) -> Dict[str, Any]:
        """Get comprehensive compression statistics"""
        
        if self.stats['total_original_size'] == 0:
            overall_ratio = 0.0
        else:
            overall_ratio = 1.0 - (self.stats['total_compressed_size'] / self.stats['total_original_size'])
        
        return {
            'total_items_compressed': self.stats['total_compressed'],
            'total_original_size_mb': self.stats['total_original_size'] / (1024 * 1024),
            'total_compressed_size_mb': self.stats['total_compressed_size'] / (1024 * 1024),
            'overall_compression_ratio': overall_ratio,
            'space_saved_mb': (self.stats['total_original_size'] - self.stats['total_compressed_size']) / (1024 * 1024),
            'algorithm_usage': dict(self.stats['algorithm_usage']),
            'content_type_stats': {
                ct: {'count': stats['count'], 'avg_compression_ratio': stats['avg_ratio']}
                for ct, stats in self.stats['content_type_stats'].items()
            },
            'semantic_compressions': self.stats['semantic_compressions'],
            'semantic_deduplications': self.stats['deduplications'],
            'cache_size': len(self.compression_cache),
            'semantic_cache_size': len(self.semantic_cache)
        }
    
    def optimize_compression_settings(self):
        """Optimize compression settings based on observed performance"""
        
        # Reorder algorithms by effectiveness
        algorithm_effectiveness = {}
        
        for algorithm, usage_count in self.stats['algorithm_usage'].items():
            if algorithm != 'none' and usage_count > 0:
                # Simple effectiveness score based on usage
                algorithm_effectiveness[algorithm] = usage_count
        
        if algorithm_effectiveness:
            # Reorder by effectiveness
            sorted_algorithms = sorted(
                algorithm_effectiveness.keys(),
                key=lambda x: algorithm_effectiveness[x],
                reverse=True
            )
            
            self.config.compression_algorithms = sorted_algorithms
            logger.info(f"Optimized algorithm order: {sorted_algorithms}")
    
    def export_semantic_model(self, filepath: str):
        """Export trained semantic compression model"""
        
        model_data = {
            'common_phrases': self.semantic_compressor.common_phrases,
            'phrase_replacements': self.semantic_compressor.phrase_replacements,
            'consciousness_keywords': list(self.semantic_compressor.consciousness_keywords),
            'stats': self.get_compression_statistics(),
            'config': self.config.__dict__
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"Exported semantic compression model to {filepath}")
    
    def import_semantic_model(self, filepath: str):
        """Import trained semantic compression model"""
        
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.semantic_compressor.common_phrases = model_data['common_phrases']
        self.semantic_compressor.phrase_replacements = model_data['phrase_replacements']
        self.semantic_compressor.consciousness_keywords = set(model_data['consciousness_keywords'])
        
        logger.info(f"Imported semantic compression model from {filepath}")


# Testing and example usage
def test_web_compression():
    """Test web content compression system"""
    
    # Create test content
    test_contents = [
        WebContent(
            id="test_1",
            url="https://example.com/ai-news",
            title="Latest AI Developments",
            content="Artificial intelligence continues to advance rapidly. Machine learning models are becoming more sophisticated. Researchers are exploring consciousness in AI systems. The development of conscious AI represents a significant breakthrough in artificial intelligence research.",
            source="example.com",
            published_at=datetime.now(),
            retrieved_at=datetime.now(),
            content_type="news",
            relevance_score=0.9,
            keywords=["AI", "consciousness", "research"],
            summary="AI research advances with conscious systems",
            language="en"
        ),
        WebContent(
            id="test_2",
            url="https://example.com/consciousness-study",
            title="Consciousness in AI Systems",
            content="The study of consciousness in artificial intelligence systems is gaining momentum. Researchers are developing new methods to detect self-awareness in AI. Machine consciousness represents the next frontier in artificial intelligence development.",
            source="example.com",
            published_at=datetime.now(),
            retrieved_at=datetime.now(),
            content_type="article",
            relevance_score=0.8,
            keywords=["consciousness", "AI", "self-awareness"],
            summary="Research on AI consciousness advances",
            language="en"
        )
    ]
    
    # Create compressor
    config = CompressionConfig(
        enable_semantic_compression=True,
        semantic_deduplication=True
    )
    compressor = WebContentCompressor(config)
    
    # Train semantic compressor
    compressor.train_semantic_compressor(test_contents)
    
    # Test compression
    results = []
    for content in test_contents:
        compressed_data, result = compressor.compress_web_content(content)
        results.append((content, compressed_data, result))
        
        print(f"\nCompressed: {content.title}")
        print(f"Original size: {result.original_size} bytes")
        print(f"Compressed size: {result.compressed_size} bytes")
        print(f"Compression ratio: {result.compression_ratio:.2%}")
        print(f"Algorithm: {result.algorithm_used}")
        
        # Test decompression
        try:
            decompressed = compressor.decompress_web_content(compressed_data, result)
            print(f"Decompression successful: {len(decompressed)} chars")
        except Exception as e:
            print(f"Decompression failed: {e}")
    
    # Print statistics
    stats = compressor.get_compression_statistics()
    print("\nCompression Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    test_web_compression()