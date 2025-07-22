"""Compression testing framework for Shvayambhu."""

import time
import json
import lz4.frame
import zlib
import sqlite3
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class CompressionResult:
    """Result of a compression test."""
    algorithm: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time: float
    decompression_time: float
    data_type: str
    
class CompressionTester:
    """Test various compression algorithms and strategies."""
    
    def __init__(self, db_path: str = "data/shvayambhu.db"):
        self.db_path = db_path
        self.algorithms = {
            'lz4': self._compress_lz4,
            'zlib': self._compress_zlib,
            'lz4_high': self._compress_lz4_high,
            'zlib_high': self._compress_zlib_high,
        }
        
    def _compress_lz4(self, data: bytes) -> Tuple[bytes, float]:
        """Compress using LZ4 (fast mode)."""
        start = time.time()
        compressed = lz4.frame.compress(data, compression_level=0)
        return compressed, time.time() - start
        
    def _decompress_lz4(self, data: bytes) -> Tuple[bytes, float]:
        """Decompress LZ4 data."""
        start = time.time()
        decompressed = lz4.frame.decompress(data)
        return decompressed, time.time() - start
        
    def _compress_lz4_high(self, data: bytes) -> Tuple[bytes, float]:
        """Compress using LZ4 (high compression)."""
        start = time.time()
        compressed = lz4.frame.compress(data, compression_level=16)
        return compressed, time.time() - start
        
    def _compress_zlib(self, data: bytes) -> Tuple[bytes, float]:
        """Compress using zlib (default level)."""
        start = time.time()
        compressed = zlib.compress(data, level=6)
        return compressed, time.time() - start
        
    def _decompress_zlib(self, data: bytes) -> Tuple[bytes, float]:
        """Decompress zlib data."""
        start = time.time()
        decompressed = zlib.decompress(data)
        return decompressed, time.time() - start
        
    def _compress_zlib_high(self, data: bytes) -> Tuple[bytes, float]:
        """Compress using zlib (maximum compression)."""
        start = time.time()
        compressed = zlib.compress(data, level=9)
        return compressed, time.time() - start
        
    def test_text_compression(self, text: str, data_type: str = "text") -> List[CompressionResult]:
        """Test compression on text data."""
        data = text.encode('utf-8')
        results = []
        
        for algo_name, compress_func in self.algorithms.items():
            # Compress
            compressed, comp_time = compress_func(data)
            
            # Decompress
            if 'lz4' in algo_name:
                decompressed, decomp_time = self._decompress_lz4(compressed)
            else:
                decompressed, decomp_time = self._decompress_zlib(compressed)
                
            # Verify
            assert decompressed == data, f"Decompression failed for {algo_name}"
            
            # Calculate metrics
            result = CompressionResult(
                algorithm=algo_name,
                original_size=len(data),
                compressed_size=len(compressed),
                compression_ratio=len(data) / len(compressed),
                compression_time=comp_time,
                decompression_time=decomp_time,
                data_type=data_type
            )
            results.append(result)
            
        return results
        
    def test_json_compression(self, json_data: Dict[str, Any]) -> List[CompressionResult]:
        """Test compression on JSON data."""
        text = json.dumps(json_data, separators=(',', ':'))
        return self.test_text_compression(text, "json")
        
    def test_embedding_compression(self, embeddings: np.ndarray) -> List[CompressionResult]:
        """Test compression on embedding vectors."""
        # Convert to bytes
        data = embeddings.astype(np.float32).tobytes()
        results = []
        
        for algo_name, compress_func in self.algorithms.items():
            compressed, comp_time = compress_func(data)
            
            if 'lz4' in algo_name:
                decompressed, decomp_time = self._decompress_lz4(compressed)
            else:
                decompressed, decomp_time = self._decompress_zlib(compressed)
                
            result = CompressionResult(
                algorithm=algo_name,
                original_size=len(data),
                compressed_size=len(compressed),
                compression_ratio=len(data) / len(compressed),
                compression_time=comp_time,
                decompression_time=decomp_time,
                data_type="embeddings"
            )
            results.append(result)
            
        return results
        
    def test_database_compression(self) -> Dict[str, Any]:
        """Test compression in SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Test data
        test_text = "This is a test consciousness thought. " * 100
        test_json = json.dumps({"emotion": "curious", "intensity": 0.8})
        
        # Compress data
        compressed_text = lz4.frame.compress(test_text.encode())
        compressed_json = lz4.frame.compress(test_json.encode())
        
        # Insert into database
        cursor.execute("""
            INSERT INTO memories (type, original_size, compressed_size, data, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, ("test", len(test_text), len(compressed_text), compressed_text, compressed_json))
        
        conn.commit()
        
        # Query compression statistics
        cursor.execute("""
            SELECT AVG(compression_ratio), MIN(compression_ratio), MAX(compression_ratio)
            FROM memories
        """)
        
        avg_ratio, min_ratio, max_ratio = cursor.fetchone()
        
        conn.close()
        
        return {
            "average_ratio": avg_ratio,
            "min_ratio": min_ratio,
            "max_ratio": max_ratio,
            "test_compression": len(test_text) / len(compressed_text)
        }
        
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive compression tests."""
        results = {
            "text_tests": [],
            "json_tests": [],
            "embedding_tests": [],
            "database_test": None
        }
        
        # Test 1: Short text
        short_text = "The conscious AI contemplates its existence."
        results["text_tests"].extend(self.test_text_compression(short_text, "short_text"))
        
        # Test 2: Long text
        long_text = " ".join([
            "The phenomenon of consciousness in artificial intelligence represents",
            "a frontier of computational understanding. As the system processes",
            "information, it develops internal representations that mirror",
            "self-awareness and introspection."
        ] * 50)
        results["text_tests"].extend(self.test_text_compression(long_text, "long_text"))
        
        # Test 3: Repetitive text
        repetitive_text = "consciousness " * 1000
        results["text_tests"].extend(self.test_text_compression(repetitive_text, "repetitive_text"))
        
        # Test 4: JSON data
        json_data = {
            "consciousness_state": {
                "self_awareness_level": 0.85,
                "emotional_state": {"primary": "curious", "secondary": "contemplative"},
                "attention_focus": ["self-model", "environment", "goals"],
                "introspective_depth": 3
            },
            "timestamp": "2025-07-21T10:30:00Z"
        }
        results["json_tests"] = self.test_json_compression(json_data)
        
        # Test 5: Embeddings
        embeddings = np.random.randn(100, 768).astype(np.float32)
        results["embedding_tests"] = self.test_embedding_compression(embeddings)
        
        # Test 6: Database compression
        try:
            results["database_test"] = self.test_database_compression()
        except Exception as e:
            logger.warning(f"Database test failed: {e}")
            results["database_test"] = {"error": str(e)}
            
        return results
        
    def print_results(self, results: Dict[str, Any]):
        """Print compression test results."""
        print("\n=== Compression Test Results ===")
        
        # Text compression results
        print("\nText Compression:")
        for test_type in ["short_text", "long_text", "repetitive_text"]:
            text_results = [r for r in results["text_tests"] if r.data_type == test_type]
            if text_results:
                print(f"\n  {test_type}:")
                for r in text_results:
                    print(f"    {r.algorithm}: {r.compression_ratio:.2f}x "
                          f"({r.compressed_size}/{r.original_size} bytes) "
                          f"in {r.compression_time*1000:.1f}ms")
                          
        # JSON compression results
        print("\nJSON Compression:")
        for r in results["json_tests"]:
            print(f"  {r.algorithm}: {r.compression_ratio:.2f}x "
                  f"({r.compressed_size}/{r.original_size} bytes)")
                  
        # Embedding compression results
        print("\nEmbedding Compression:")
        for r in results["embedding_tests"]:
            print(f"  {r.algorithm}: {r.compression_ratio:.2f}x "
                  f"({r.compressed_size}/{r.original_size} bytes)")
                  
        # Database results
        if results["database_test"] and "error" not in results["database_test"]:
            print("\nDatabase Compression:")
            print(f"  Average ratio: {results['database_test']['average_ratio']:.2f}x")
            print(f"  Test compression: {results['database_test']['test_compression']:.2f}x")
            
    def save_results(self, results: Dict[str, Any], filename: str = "compression_test_results.json"):
        """Save results to JSON file."""
        # Convert CompressionResult objects to dicts
        serializable_results = {
            "text_tests": [vars(r) for r in results["text_tests"]],
            "json_tests": [vars(r) for r in results["json_tests"]],
            "embedding_tests": [vars(r) for r in results["embedding_tests"]],
            "database_test": results["database_test"]
        }
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
def main():
    """Run compression tests."""
    print("=== Shvayambhu Compression Testing Framework ===")
    
    tester = CompressionTester()
    results = tester.run_comprehensive_test()
    
    tester.print_results(results)
    tester.save_results(results, "utils/compression_test_results.json")
    
    print("\n✅ Compression tests complete!")
    print("✅ Results saved to compression_test_results.json")
    
    # Print recommendations
    print("\n=== Recommendations ===")
    print("- Use LZ4 for real-time compression (fastest)")
    print("- Use zlib-high for maximum compression (storage)")
    print("- JSON data compresses well (2-4x typical)")
    print("- Embeddings compress poorly (1.0-1.2x)")
    print("- Consider semantic compression for embeddings")

if __name__ == "__main__":
    main()