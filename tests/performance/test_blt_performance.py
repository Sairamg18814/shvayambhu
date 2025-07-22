"""Performance benchmarks for BLT architecture.

This module benchmarks the BLT pipeline against traditional tokenization
approaches and measures throughput, latency, and resource usage.
"""

import time
import pytest
import torch
import numpy as np
from typing import List, Dict, Tuple
import psutil
import os
from dataclasses import dataclass

from shvayambhu.core.blt.pipeline import BLTPipeline
from shvayambhu.core.blt.encoder import LocalEncoder
from shvayambhu.core.blt.transformer import LatentTransformer
from shvayambhu.core.blt.decoder import LocalDecoder


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    throughput_bytes_per_sec: float
    throughput_tokens_per_sec: float
    latency_ms: float
    memory_mb: float
    first_token_latency_ms: float
    
    def __repr__(self):
        return (
            f"Throughput: {self.throughput_bytes_per_sec:.2f} bytes/s, "
            f"{self.throughput_tokens_per_sec:.2f} tokens/s | "
            f"Latency: {self.latency_ms:.2f}ms | "
            f"Memory: {self.memory_mb:.2f}MB | "
            f"First token: {self.first_token_latency_ms:.2f}ms"
        )


class BLTPerformanceBenchmark:
    """Comprehensive performance benchmarks for BLT."""
    
    def __init__(self):
        self.device = torch.device("cpu")  # Change to "mps" for Metal on Mac
        self.config = {
            "vocab_size": 256,
            "hidden_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "patch_embedding_dim": 768,
            "max_patch_size": 32,
            "device": self.device
        }
        self.pipeline = BLTPipeline(self.config)
        self.pipeline.eval()
    
    def benchmark_throughput(self, texts: List[str], batch_size: int = 1) -> BenchmarkResult:
        """Benchmark throughput for given texts."""
        total_bytes = sum(len(text.encode('utf-8')) for text in texts)
        total_chars = sum(len(text) for text in texts)
        
        # Warm up
        for _ in range(3):
            self.pipeline.process_bytes(texts[0].encode('utf-8'), mode='inference')
        
        # Measure throughput
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        first_token_time = None
        processed = 0
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            if batch_size == 1:
                for text in batch:
                    output = self.pipeline.process_bytes(
                        text.encode('utf-8'), 
                        mode='inference'
                    )
                    if first_token_time is None:
                        first_token_time = time.time() - start_time
                    processed += 1
            else:
                byte_sequences = [text.encode('utf-8') for text in batch]
                outputs = self.pipeline.process_batch(byte_sequences, mode='inference')
                if first_token_time is None:
                    first_token_time = time.time() - start_time
                processed += len(batch)
        
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate metrics
        total_time = end_time - start_time
        throughput_bytes = total_bytes / total_time
        throughput_tokens = total_chars / total_time  # Approximate
        latency = (total_time / len(texts)) * 1000  # ms per text
        memory_used = end_memory - start_memory
        first_token_latency = first_token_time * 1000 if first_token_time else 0
        
        return BenchmarkResult(
            throughput_bytes_per_sec=throughput_bytes,
            throughput_tokens_per_sec=throughput_tokens,
            latency_ms=latency,
            memory_mb=memory_used,
            first_token_latency_ms=first_token_latency
        )
    
    def benchmark_vs_tokenization(self):
        """Compare BLT performance vs traditional tokenization."""
        # Test texts of various lengths
        test_texts = [
            "Short text",
            "Medium length text that is a bit longer than the short one",
            "Long text " * 50,
            "Very long text " * 200,
        ]
        
        print("\n=== BLT vs Tokenization Benchmark ===")
        
        for text in test_texts:
            print(f"\nText length: {len(text)} chars, {len(text.encode('utf-8'))} bytes")
            
            # BLT timing
            blt_start = time.time()
            byte_seq = text.encode('utf-8')
            blt_output = self.pipeline.process_bytes(byte_seq, mode='inference')
            blt_time = time.time() - blt_start
            
            # Simulate tokenization timing (using simple split for comparison)
            tok_start = time.time()
            tokens = text.split()  # Simple tokenization
            token_ids = [hash(token) % 50000 for token in tokens]  # Simulate token IDs
            tok_time = time.time() - tok_start
            
            print(f"BLT time: {blt_time*1000:.2f}ms")
            print(f"Tokenization time: {tok_time*1000:.2f}ms")
            print(f"BLT/Tokenization ratio: {blt_time/tok_time:.2f}x")
    
    def benchmark_memory_usage(self):
        """Benchmark memory usage patterns."""
        print("\n=== Memory Usage Benchmark ===")
        
        text_sizes = [100, 1000, 10000, 50000]
        
        for size in text_sizes:
            text = "a" * size
            byte_seq = text.encode('utf-8')
            
            # Measure memory before
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024
            
            # Process text
            output = self.pipeline.process_bytes(byte_seq, mode='inference')
            
            # Measure memory after
            mem_after = process.memory_info().rss / 1024 / 1024
            memory_used = mem_after - mem_before
            
            print(f"Text size: {size} bytes, Memory used: {memory_used:.2f} MB")
            print(f"Memory per byte: {memory_used * 1024 * 1024 / size:.2f} bytes")
    
    def benchmark_latency_distribution(self):
        """Benchmark latency distribution across different text types."""
        print("\n=== Latency Distribution Benchmark ===")
        
        test_cases = {
            "ASCII": "Hello world " * 10,
            "Unicode": "你好世界 " * 10,
            "Mixed": "Hello 世界 مرحبا world " * 10,
            "Code": "def func(x): return x * 2 " * 10,
            "Numbers": "1234567890 " * 10,
        }
        
        latencies = {}
        
        for name, text in test_cases.items():
            # Run multiple times to get distribution
            times = []
            for _ in range(100):
                start = time.time()
                self.pipeline.process_bytes(text.encode('utf-8'), mode='inference')
                times.append((time.time() - start) * 1000)
            
            latencies[name] = {
                "mean": np.mean(times),
                "std": np.std(times),
                "min": np.min(times),
                "max": np.max(times),
                "p50": np.percentile(times, 50),
                "p95": np.percentile(times, 95),
                "p99": np.percentile(times, 99),
            }
            
            print(f"\n{name}:")
            print(f"  Mean: {latencies[name]['mean']:.2f}ms ± {latencies[name]['std']:.2f}ms")
            print(f"  Min/Max: {latencies[name]['min']:.2f}ms / {latencies[name]['max']:.2f}ms")
            print(f"  P50/P95/P99: {latencies[name]['p50']:.2f}ms / {latencies[name]['p95']:.2f}ms / {latencies[name]['p99']:.2f}ms")
    
    def benchmark_batch_processing(self):
        """Benchmark batch processing efficiency."""
        print("\n=== Batch Processing Benchmark ===")
        
        # Create test texts
        num_texts = 100
        texts = [f"Test text number {i} with some content" for i in range(num_texts)]
        
        batch_sizes = [1, 4, 8, 16, 32]
        
        for batch_size in batch_sizes:
            result = self.benchmark_throughput(texts, batch_size=batch_size)
            print(f"\nBatch size {batch_size}: {result}")
    
    def benchmark_component_performance(self):
        """Benchmark individual component performance."""
        print("\n=== Component Performance Benchmark ===")
        
        text = "Component benchmark test " * 100
        byte_seq = text.encode('utf-8')
        
        # Benchmark encoder
        encoder = self.pipeline.encoder
        start = time.time()
        for _ in range(100):
            encoder_output = encoder(byte_seq)
        encoder_time = (time.time() - start) / 100 * 1000
        
        # Benchmark transformer
        transformer = self.pipeline.transformer
        patch_embeddings = encoder_output['patch_embeddings'].unsqueeze(0)
        start = time.time()
        for _ in range(100):
            transformer_output = transformer(patch_embeddings)
        transformer_time = (time.time() - start) / 100 * 1000
        
        # Benchmark decoder
        decoder = self.pipeline.decoder
        start = time.time()
        for _ in range(100):
            decoder_output = decoder(
                transformer_output['hidden_states'],
                [encoder_output['patch_boundaries']],
                [encoder_output['original_length']]
            )
        decoder_time = (time.time() - start) / 100 * 1000
        
        total_time = encoder_time + transformer_time + decoder_time
        
        print(f"Encoder: {encoder_time:.2f}ms ({encoder_time/total_time*100:.1f}%)")
        print(f"Transformer: {transformer_time:.2f}ms ({transformer_time/total_time*100:.1f}%)")
        print(f"Decoder: {decoder_time:.2f}ms ({decoder_time/total_time*100:.1f}%)")
        print(f"Total: {total_time:.2f}ms")
    
    def benchmark_scalability(self):
        """Benchmark scalability with text length."""
        print("\n=== Scalability Benchmark ===")
        
        lengths = [10, 50, 100, 500, 1000, 5000]
        results = []
        
        for length in lengths:
            text = "a" * length
            texts = [text] * 10  # Process 10 times for average
            
            result = self.benchmark_throughput(texts, batch_size=1)
            results.append((length, result))
            
            print(f"\nLength {length}: {result.throughput_bytes_per_sec:.2f} bytes/s, "
                  f"Latency: {result.latency_ms:.2f}ms")
        
        # Check if performance scales linearly
        if len(results) > 1:
            # Calculate scaling factor
            first_length, first_result = results[0]
            last_length, last_result = results[-1]
            
            length_ratio = last_length / first_length
            latency_ratio = last_result.latency_ms / first_result.latency_ms
            
            print(f"\nScaling: {length_ratio:.1f}x length -> {latency_ratio:.1f}x latency")
            print(f"Scaling efficiency: {length_ratio / latency_ratio:.2f}")


class TestBLTPerformance:
    """Pytest test cases for performance benchmarks."""
    
    @pytest.fixture
    def benchmark(self):
        """Create benchmark instance."""
        return BLTPerformanceBenchmark()
    
    def test_throughput_targets(self, benchmark):
        """Test that throughput meets targets."""
        # Test on realistic text
        texts = [
            "The quick brown fox jumps over the lazy dog. " * 10
            for _ in range(100)
        ]
        
        result = benchmark.benchmark_throughput(texts, batch_size=8)
        
        # Check targets (adjust based on hardware)
        assert result.throughput_bytes_per_sec > 10000  # 10KB/s minimum
        assert result.latency_ms < 100  # Less than 100ms per text
        assert result.first_token_latency_ms < 50  # Less than 50ms first token
    
    def test_memory_efficiency(self, benchmark):
        """Test memory usage is reasonable."""
        # Process large text
        large_text = "Large text content " * 1000
        
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024
        
        benchmark.pipeline.process_bytes(large_text.encode('utf-8'), mode='inference')
        
        mem_after = process.memory_info().rss / 1024 / 1024
        memory_used = mem_after - mem_before
        
        # Should use less than 100MB for this text
        assert memory_used < 100
    
    def test_batch_efficiency(self, benchmark):
        """Test that batching improves throughput."""
        texts = ["Test text " * 10 for _ in range(32)]
        
        # Single processing
        single_result = benchmark.benchmark_throughput(texts, batch_size=1)
        
        # Batch processing
        batch_result = benchmark.benchmark_throughput(texts, batch_size=8)
        
        # Batch should be more efficient
        assert batch_result.throughput_bytes_per_sec > single_result.throughput_bytes_per_sec
    
    def test_linear_scaling(self, benchmark):
        """Test that performance scales reasonably with input size."""
        # Test different sizes
        small_text = "a" * 100
        large_text = "a" * 1000
        
        # Time small text
        start = time.time()
        benchmark.pipeline.process_bytes(small_text.encode('utf-8'), mode='inference')
        small_time = time.time() - start
        
        # Time large text
        start = time.time()
        benchmark.pipeline.process_bytes(large_text.encode('utf-8'), mode='inference')
        large_time = time.time() - start
        
        # Should scale sub-linearly (due to efficiencies)
        time_ratio = large_time / small_time
        size_ratio = len(large_text) / len(small_text)
        
        assert time_ratio < size_ratio * 1.5  # Allow 50% overhead


def run_all_benchmarks():
    """Run all benchmarks and print results."""
    benchmark = BLTPerformanceBenchmark()
    
    print("Running BLT Performance Benchmarks...")
    print("=" * 60)
    
    benchmark.benchmark_vs_tokenization()
    benchmark.benchmark_memory_usage()
    benchmark.benchmark_latency_distribution()
    benchmark.benchmark_batch_processing()
    benchmark.benchmark_component_performance()
    benchmark.benchmark_scalability()
    
    print("\n" + "=" * 60)
    print("Benchmarks completed!")


if __name__ == "__main__":
    run_all_benchmarks()