#!/usr/bin/env python3
"""Benchmark script for BLT performance optimization.

This script runs comprehensive benchmarks on BLT components including:
- Patching strategies
- Caching performance
- Metal acceleration
- Batch processing
- Memory usage
"""

import argparse
import json
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import psutil
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shvayambhu.core.blt.pipeline import BLTPipeline
from shvayambhu.core.blt.patching import DynamicPatcher, BLTInputProcessor
from shvayambhu.core.blt.patch_optimizer import PatchOptimizer, OptimizationConfig
from shvayambhu.core.blt.patch_cache import create_patch_cache
from shvayambhu.core.blt.batch_processor import create_optimized_batch_processor
from shvayambhu.inference.engine.metal_blt_ops import MetalBLTOps, HAS_METAL
from shvayambhu.utils.profiling.memory_profiler import MemoryProfiler, profile_memory_usage


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    component: str
    operation: str
    configuration: Dict[str, Any]
    throughput_bytes_per_sec: float
    throughput_sequences_per_sec: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    memory_peak_mb: float
    memory_avg_mb: float
    cache_hit_rate: float
    error_rate: float
    timestamp: float


class BLTBenchmark:
    """Comprehensive BLT benchmarking suite."""
    
    def __init__(self, output_dir: str = "benchmarks/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []
        self.memory_profiler = MemoryProfiler()
        
        # Load test data
        self._load_test_data()
    
    def _load_test_data(self):
        """Load diverse test data."""
        self.test_data = {
            "text_short": [
                b"Hello, world!",
                b"The quick brown fox jumps over the lazy dog.",
                b"Machine learning is transforming the world.",
            ] * 100,
            
            "text_long": [
                b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 50,
                b"In the realm of artificial intelligence, " * 100,
            ] * 10,
            
            "code": [
                b"def hello_world():\n    print('Hello, world!')\n",
                b"class Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n",
                b"for i in range(100):\n    x = process_data(i)\n    results.append(x)\n",
            ] * 50,
            
            "multilingual": [
                "你好世界".encode('utf-8'),
                "مرحبا بالعالم".encode('utf-8'),
                "Здравствуй, мир!".encode('utf-8'),
                "こんにちは世界".encode('utf-8'),
            ] * 25,
            
            "high_entropy": [
                bytes(np.random.randint(0, 256, 1000))
                for _ in range(20)
            ],
            
            "low_entropy": [
                b"a" * 1000,
                b"0" * 1000,
                b"\x00" * 1000,
            ] * 10,
        }
    
    def benchmark_patching_strategies(self):
        """Benchmark different patching strategies."""
        print("\n=== Benchmarking Patching Strategies ===")
        
        strategies = [
            ("fixed_16", {"min_patch_size": 16, "max_patch_size": 16}),
            ("fixed_32", {"min_patch_size": 32, "max_patch_size": 32}),
            ("dynamic_entropy", {"min_patch_size": 4, "max_patch_size": 32}),
            ("dynamic_optimized", {"use_optimizer": True}),
        ]
        
        for strategy_name, config in strategies:
            print(f"\nTesting {strategy_name}...")
            
            if config.get("use_optimizer"):
                patcher = None
                optimizer = PatchOptimizer()
            else:
                patcher = DynamicPatcher(**config)
                optimizer = None
            
            for data_type, sequences in self.test_data.items():
                result = self._benchmark_patching(
                    sequences, patcher, optimizer, strategy_name, data_type
                )
                self.results.append(result)
    
    def _benchmark_patching(
        self,
        sequences: List[bytes],
        patcher: Optional[DynamicPatcher],
        optimizer: Optional[PatchOptimizer],
        strategy: str,
        data_type: str
    ) -> BenchmarkResult:
        """Benchmark patching performance."""
        latencies = []
        total_bytes = 0
        
        with self.memory_profiler.profile_component(f"patching_{strategy}_{data_type}"):
            start_time = time.time()
            
            for seq in sequences:
                seq_start = time.time()
                
                if optimizer:
                    boundaries, _ = optimizer.optimize_patch_sizes(seq)
                    patches = [seq[s:e] for s, e in boundaries]
                else:
                    patches = patcher.create_patches(seq)
                
                latencies.append((time.time() - seq_start) * 1000)
                total_bytes += len(seq)
            
            total_time = time.time() - start_time
        
        # Calculate metrics
        latencies = np.array(latencies)
        memory_stats = self.memory_profiler.analyze_component(f"patching_{strategy}_{data_type}")
        
        return BenchmarkResult(
            component="patching",
            operation=strategy,
            configuration={"data_type": data_type},
            throughput_bytes_per_sec=total_bytes / total_time,
            throughput_sequences_per_sec=len(sequences) / total_time,
            latency_p50_ms=np.percentile(latencies, 50),
            latency_p95_ms=np.percentile(latencies, 95),
            latency_p99_ms=np.percentile(latencies, 99),
            memory_peak_mb=memory_stats.get("peak_rss_mb", 0),
            memory_avg_mb=memory_stats.get("avg_rss_mb", 0),
            cache_hit_rate=0.0,  # Not applicable
            error_rate=0.0,
            timestamp=time.time()
        )
    
    def benchmark_caching(self):
        """Benchmark caching strategies."""
        print("\n=== Benchmarking Caching Strategies ===")
        
        cache_configs = [
            ("lru_small", {"cache_type": "lru", "max_size_mb": 10}),
            ("lru_large", {"cache_type": "lru", "max_size_mb": 100}),
            ("frequency", {"cache_type": "frequency", "max_size_mb": 50}),
            ("hierarchical", {"cache_type": "hierarchical", "l1_size_mb": 10, "l2_size_mb": 100}),
            ("content_aware", {"cache_type": "content_aware", "max_size_mb": 50}),
        ]
        
        for cache_name, config in cache_configs:
            print(f"\nTesting {cache_name}...")
            cache = create_patch_cache(**config)
            
            # Create repeated sequences for cache testing
            test_sequences = []
            for _ in range(10):  # Repeat to test cache hits
                test_sequences.extend(self.test_data["text_short"][:20])
                test_sequences.extend(self.test_data["code"][:10])
            
            result = self._benchmark_cache(cache, test_sequences, cache_name)
            self.results.append(result)
    
    def _benchmark_cache(
        self,
        cache: Any,
        sequences: List[bytes],
        cache_type: str
    ) -> BenchmarkResult:
        """Benchmark cache performance."""
        latencies = []
        hits = 0
        total_bytes = 0
        
        # Create dummy embeddings
        embedding_dim = 768
        
        with self.memory_profiler.profile_component(f"cache_{cache_type}"):
            start_time = time.time()
            
            for seq in sequences:
                seq_start = time.time()
                
                # Try to get from cache
                cached = cache.get(seq)
                if cached is not None:
                    hits += 1
                else:
                    # Simulate embedding computation
                    embedding = torch.randn(len(seq), embedding_dim)
                    cache.put(seq, embedding)
                
                latencies.append((time.time() - seq_start) * 1000)
                total_bytes += len(seq)
            
            total_time = time.time() - start_time
        
        # Calculate metrics
        latencies = np.array(latencies)
        memory_stats = self.memory_profiler.analyze_component(f"cache_{cache_type}")
        cache_stats = cache.get_stats() if hasattr(cache, 'get_stats') else {}
        
        return BenchmarkResult(
            component="caching",
            operation=cache_type,
            configuration=cache_stats,
            throughput_bytes_per_sec=total_bytes / total_time,
            throughput_sequences_per_sec=len(sequences) / total_time,
            latency_p50_ms=np.percentile(latencies, 50),
            latency_p95_ms=np.percentile(latencies, 95),
            latency_p99_ms=np.percentile(latencies, 99),
            memory_peak_mb=memory_stats.get("peak_rss_mb", 0),
            memory_avg_mb=memory_stats.get("avg_rss_mb", 0),
            cache_hit_rate=hits / len(sequences),
            error_rate=0.0,
            timestamp=time.time()
        )
    
    def benchmark_metal_acceleration(self):
        """Benchmark Metal acceleration if available."""
        print("\n=== Benchmarking Metal Acceleration ===")
        
        if not HAS_METAL:
            print("Metal not available on this system. Skipping Metal benchmarks.")
            return
        
        metal_ops = MetalBLTOps()
        
        # Test byte embedding
        print("\nTesting Metal byte embedding...")
        for data_type, sequences in list(self.test_data.items())[:3]:
            result = self._benchmark_metal_embedding(
                metal_ops, sequences[:50], data_type
            )
            self.results.append(result)
        
        # Test entropy calculation
        print("\nTesting Metal entropy calculation...")
        for data_type, sequences in list(self.test_data.items())[:3]:
            result = self._benchmark_metal_entropy(
                metal_ops, sequences[:50], data_type
            )
            self.results.append(result)
    
    def _benchmark_metal_embedding(
        self,
        metal_ops: MetalBLTOps,
        sequences: List[bytes],
        data_type: str
    ) -> BenchmarkResult:
        """Benchmark Metal byte embedding."""
        latencies = []
        total_bytes = 0
        embedding_dim = 768
        embedding_table = torch.randn(256, embedding_dim)
        
        start_time = time.time()
        
        for seq in sequences:
            seq_array = np.frombuffer(seq, dtype=np.uint8)
            seq_start = time.time()
            
            # Metal embedding
            _ = metal_ops.byte_embedding_metal(
                seq_array, embedding_table, embedding_dim
            )
            
            latencies.append((time.time() - seq_start) * 1000)
            total_bytes += len(seq)
        
        total_time = time.time() - start_time
        latencies = np.array(latencies)
        
        return BenchmarkResult(
            component="metal",
            operation="byte_embedding",
            configuration={"data_type": data_type},
            throughput_bytes_per_sec=total_bytes / total_time,
            throughput_sequences_per_sec=len(sequences) / total_time,
            latency_p50_ms=np.percentile(latencies, 50),
            latency_p95_ms=np.percentile(latencies, 95),
            latency_p99_ms=np.percentile(latencies, 99),
            memory_peak_mb=0.0,  # Metal memory not tracked here
            memory_avg_mb=0.0,
            cache_hit_rate=0.0,
            error_rate=0.0,
            timestamp=time.time()
        )
    
    def _benchmark_metal_entropy(
        self,
        metal_ops: MetalBLTOps,
        sequences: List[bytes],
        data_type: str
    ) -> BenchmarkResult:
        """Benchmark Metal entropy calculation."""
        latencies = []
        total_bytes = 0
        window_size = 256
        
        start_time = time.time()
        
        for seq in sequences:
            if len(seq) < window_size:
                continue
                
            seq_array = np.frombuffer(seq, dtype=np.uint8)
            seq_start = time.time()
            
            # Metal entropy
            _ = metal_ops.calculate_entropy_metal(seq_array, window_size)
            
            latencies.append((time.time() - seq_start) * 1000)
            total_bytes += len(seq)
        
        total_time = time.time() - start_time
        latencies = np.array(latencies) if latencies else np.array([0])
        
        return BenchmarkResult(
            component="metal",
            operation="entropy_calculation",
            configuration={"data_type": data_type, "window_size": window_size},
            throughput_bytes_per_sec=total_bytes / max(total_time, 0.001),
            throughput_sequences_per_sec=len(sequences) / max(total_time, 0.001),
            latency_p50_ms=np.percentile(latencies, 50),
            latency_p95_ms=np.percentile(latencies, 95),
            latency_p99_ms=np.percentile(latencies, 99),
            memory_peak_mb=0.0,
            memory_avg_mb=0.0,
            cache_hit_rate=0.0,
            error_rate=0.0,
            timestamp=time.time()
        )
    
    def benchmark_batch_processing(self):
        """Benchmark batch processing."""
        print("\n=== Benchmarking Batch Processing ===")
        
        batch_configs = [
            ("small_batch", {"max_batch_size": 8, "num_workers": 2}),
            ("medium_batch", {"max_batch_size": 32, "num_workers": 4}),
            ("large_batch", {"max_batch_size": 128, "num_workers": 8}),
            ("dynamic_batch", {"max_batch_size": 64, "dynamic_batching": True}),
        ]
        
        for batch_name, config in batch_configs:
            print(f"\nTesting {batch_name}...")
            processor = create_optimized_batch_processor(config)
            processor.start()
            
            # Mix different sequence types
            test_sequences = []
            test_sequences.extend(self.test_data["text_short"][:100])
            test_sequences.extend(self.test_data["code"][:50])
            test_sequences.extend(self.test_data["multilingual"][:50])
            
            result = self._benchmark_batch_processing(
                processor, test_sequences, batch_name
            )
            self.results.append(result)
            
            processor.stop()
    
    def _benchmark_batch_processing(
        self,
        processor: Any,
        sequences: List[bytes],
        batch_type: str
    ) -> BenchmarkResult:
        """Benchmark batch processing performance."""
        start_time = time.time()
        
        # Process all sequences
        results = processor.process_sequences(sequences)
        
        total_time = time.time() - start_time
        
        # Get batch statistics
        stats = processor.get_stats()
        
        # Calculate per-sequence latency
        avg_latency = (total_time / len(sequences)) * 1000
        
        return BenchmarkResult(
            component="batch_processing",
            operation=batch_type,
            configuration={
                "avg_batch_size": stats.avg_batch_size,
                "num_batches": stats.num_batches
            },
            throughput_bytes_per_sec=stats.throughput_bytes_per_sec,
            throughput_sequences_per_sec=stats.throughput_sequences_per_sec,
            latency_p50_ms=avg_latency,  # Approximation
            latency_p95_ms=avg_latency * 1.5,  # Approximation
            latency_p99_ms=avg_latency * 2.0,  # Approximation
            memory_peak_mb=stats.memory_peak_mb,
            memory_avg_mb=0.0,
            cache_hit_rate=stats.cache_hits / max(stats.total_sequences, 1),
            error_rate=0.0,
            timestamp=time.time()
        )
    
    def benchmark_end_to_end(self):
        """Benchmark complete BLT pipeline."""
        print("\n=== Benchmarking End-to-End Pipeline ===")
        
        # Create pipeline with different configurations
        configs = [
            ("baseline", {
                "use_metal": False,
                "use_cache": False,
                "batch_size": 1
            }),
            ("cached", {
                "use_metal": False,
                "use_cache": True,
                "batch_size": 1
            }),
            ("metal", {
                "use_metal": HAS_METAL,
                "use_cache": False,
                "batch_size": 1
            }),
            ("optimized", {
                "use_metal": HAS_METAL,
                "use_cache": True,
                "batch_size": 32
            }),
        ]
        
        for config_name, config in configs:
            print(f"\nTesting {config_name} configuration...")
            
            # Create pipeline
            pipeline_config = {
                "vocab_size": 256,
                "hidden_dim": 768,
                "num_layers": 12,
                "num_heads": 12,
                "patch_embedding_dim": 768,
                "max_patch_size": 32,
                "device": torch.device("cpu"),  # For benchmarking
                **config
            }
            
            pipeline = BLTPipeline(pipeline_config)
            
            # Test on different data types
            for data_type, sequences in list(self.test_data.items())[:3]:
                result = self._benchmark_pipeline(
                    pipeline, sequences[:20], config_name, data_type
                )
                self.results.append(result)
    
    def _benchmark_pipeline(
        self,
        pipeline: BLTPipeline,
        sequences: List[bytes],
        config_name: str,
        data_type: str
    ) -> BenchmarkResult:
        """Benchmark end-to-end pipeline."""
        latencies = []
        total_bytes = 0
        errors = 0
        
        with self.memory_profiler.profile_component(f"pipeline_{config_name}_{data_type}"):
            start_time = time.time()
            
            for seq in sequences:
                seq_start = time.time()
                
                try:
                    output = pipeline.process_bytes(seq, mode='inference')
                    if 'error' in output:
                        errors += 1
                except Exception:
                    errors += 1
                
                latencies.append((time.time() - seq_start) * 1000)
                total_bytes += len(seq)
            
            total_time = time.time() - start_time
        
        # Calculate metrics
        latencies = np.array(latencies)
        memory_stats = self.memory_profiler.analyze_component(f"pipeline_{config_name}_{data_type}")
        
        return BenchmarkResult(
            component="pipeline",
            operation=config_name,
            configuration={"data_type": data_type},
            throughput_bytes_per_sec=total_bytes / total_time,
            throughput_sequences_per_sec=len(sequences) / total_time,
            latency_p50_ms=np.percentile(latencies, 50),
            latency_p95_ms=np.percentile(latencies, 95),
            latency_p99_ms=np.percentile(latencies, 99),
            memory_peak_mb=memory_stats.get("peak_rss_mb", 0),
            memory_avg_mb=memory_stats.get("avg_rss_mb", 0),
            cache_hit_rate=0.0,  # Could be extracted from pipeline
            error_rate=errors / len(sequences),
            timestamp=time.time()
        )
    
    def generate_report(self):
        """Generate comprehensive benchmark report."""
        print("\n=== Generating Benchmark Report ===")
        
        # Convert results to DataFrame
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        # Save raw results
        results_file = self.output_dir / f"benchmark_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        print(f"Raw results saved to: {results_file}")
        
        # Generate visualizations
        self._plot_throughput_comparison(df)
        self._plot_latency_comparison(df)
        self._plot_memory_usage(df)
        self._plot_cache_effectiveness(df)
        
        # Generate summary report
        self._generate_summary_report(df)
    
    def _plot_throughput_comparison(self, df: pd.DataFrame):
        """Plot throughput comparison."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Bytes per second by component
        component_throughput = df.groupby(['component', 'operation'])['throughput_bytes_per_sec'].mean()
        component_throughput.unstack().plot(kind='bar', ax=ax1)
        ax1.set_title('Throughput (Bytes/sec) by Component and Operation')
        ax1.set_ylabel('Bytes per Second')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Sequences per second
        seq_throughput = df.groupby(['component', 'operation'])['throughput_sequences_per_sec'].mean()
        seq_throughput.unstack().plot(kind='bar', ax=ax2)
        ax2.set_title('Throughput (Sequences/sec) by Component and Operation')
        ax2.set_ylabel('Sequences per Second')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'throughput_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_latency_comparison(self, df: pd.DataFrame):
        """Plot latency comparison."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Group by component and operation
        latency_data = df.groupby(['component', 'operation']).agg({
            'latency_p50_ms': 'mean',
            'latency_p95_ms': 'mean',
            'latency_p99_ms': 'mean'
        })
        
        # Plot grouped bar chart
        x = np.arange(len(latency_data))
        width = 0.25
        
        ax.bar(x - width, latency_data['latency_p50_ms'], width, label='P50')
        ax.bar(x, latency_data['latency_p95_ms'], width, label='P95')
        ax.bar(x + width, latency_data['latency_p99_ms'], width, label='P99')
        
        ax.set_xlabel('Component - Operation')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Latency Distribution by Component and Operation')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{idx[0]}-{idx[1]}" for idx in latency_data.index], rotation=45, ha='right')
        ax.legend()
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'latency_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_usage(self, df: pd.DataFrame):
        """Plot memory usage comparison."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Memory usage by component
        memory_data = df.groupby(['component', 'operation'])['memory_peak_mb'].mean()
        memory_data.plot(kind='bar', ax=ax)
        
        ax.set_title('Peak Memory Usage by Component and Operation')
        ax.set_ylabel('Memory (MB)')
        ax.set_xlabel('Component - Operation')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'memory_usage.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cache_effectiveness(self, df: pd.DataFrame):
        """Plot cache effectiveness."""
        cache_df = df[df['component'] == 'caching']
        
        if len(cache_df) == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Cache hit rate
        cache_df.set_index('operation')['cache_hit_rate'].plot(kind='bar', ax=ax1)
        ax1.set_title('Cache Hit Rate by Strategy')
        ax1.set_ylabel('Hit Rate')
        ax1.set_ylim(0, 1)
        
        # Throughput improvement
        cache_throughput = cache_df.set_index('operation')['throughput_bytes_per_sec']
        cache_throughput.plot(kind='bar', ax=ax2)
        ax2.set_title('Throughput by Cache Strategy')
        ax2.set_ylabel('Bytes per Second')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cache_effectiveness.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_report(self, df: pd.DataFrame):
        """Generate text summary report."""
        report_path = self.output_dir / 'benchmark_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("BLT Performance Benchmark Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total benchmarks run: {len(df)}\n\n")
            
            # Best configurations
            f.write("Best Configurations:\n")
            f.write("-" * 30 + "\n")
            
            # Highest throughput
            best_throughput = df.loc[df['throughput_bytes_per_sec'].idxmax()]
            f.write(f"Highest Throughput: {best_throughput['component']}-{best_throughput['operation']}\n")
            f.write(f"  {best_throughput['throughput_bytes_per_sec']:.2f} bytes/sec\n\n")
            
            # Lowest latency
            best_latency = df.loc[df['latency_p50_ms'].idxmin()]
            f.write(f"Lowest Latency: {best_latency['component']}-{best_latency['operation']}\n")
            f.write(f"  P50: {best_latency['latency_p50_ms']:.2f} ms\n\n")
            
            # Memory efficiency
            memory_df = df[df['memory_peak_mb'] > 0]
            if len(memory_df) > 0:
                best_memory = memory_df.loc[memory_df['memory_peak_mb'].idxmin()]
                f.write(f"Most Memory Efficient: {best_memory['component']}-{best_memory['operation']}\n")
                f.write(f"  Peak: {best_memory['memory_peak_mb']:.2f} MB\n\n")
            
            # Component summaries
            f.write("Component Performance Summary:\n")
            f.write("-" * 30 + "\n")
            
            for component in df['component'].unique():
                comp_df = df[df['component'] == component]
                f.write(f"\n{component.upper()}:\n")
                f.write(f"  Avg Throughput: {comp_df['throughput_bytes_per_sec'].mean():.2f} bytes/sec\n")
                f.write(f"  Avg Latency (P50): {comp_df['latency_p50_ms'].mean():.2f} ms\n")
                f.write(f"  Avg Memory: {comp_df['memory_peak_mb'].mean():.2f} MB\n")
            
            # Recommendations
            f.write("\n\nRecommendations:\n")
            f.write("-" * 30 + "\n")
            
            # Metal acceleration
            if HAS_METAL:
                metal_df = df[df['component'] == 'metal']
                if len(metal_df) > 0:
                    metal_speedup = metal_df['throughput_bytes_per_sec'].mean()
                    baseline = df[(df['component'] == 'patching') & (df['operation'] == 'fixed_16')]['throughput_bytes_per_sec'].mean()
                    if baseline > 0:
                        speedup = metal_speedup / baseline
                        f.write(f"- Metal acceleration provides {speedup:.1f}x speedup\n")
            
            # Caching
            cache_df = df[df['component'] == 'caching']
            if len(cache_df) > 0:
                best_cache = cache_df.loc[cache_df['cache_hit_rate'].idxmax()]
                f.write(f"- Best cache strategy: {best_cache['operation']} (hit rate: {best_cache['cache_hit_rate']:.2%})\n")
            
            # Batching
            batch_df = df[df['component'] == 'batch_processing']
            if len(batch_df) > 0:
                best_batch = batch_df.loc[batch_df['throughput_sequences_per_sec'].idxmax()]
                f.write(f"- Optimal batch configuration: {best_batch['operation']}\n")
        
        print(f"Summary report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark BLT performance")
    parser.add_argument("--output-dir", default="benchmarks/", help="Output directory for results")
    parser.add_argument("--components", nargs="+", 
                       choices=["patching", "caching", "metal", "batch", "pipeline", "all"],
                       default=["all"], help="Components to benchmark")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks with reduced data")
    
    args = parser.parse_args()
    
    # Create benchmark suite
    benchmark = BLTBenchmark(args.output_dir)
    
    # Run selected benchmarks
    components = set(args.components)
    if "all" in components:
        components = {"patching", "caching", "metal", "batch", "pipeline"}
    
    if "patching" in components:
        benchmark.benchmark_patching_strategies()
    
    if "caching" in components:
        benchmark.benchmark_caching()
    
    if "metal" in components:
        benchmark.benchmark_metal_acceleration()
    
    if "batch" in components:
        benchmark.benchmark_batch_processing()
    
    if "pipeline" in components:
        benchmark.benchmark_end_to_end()
    
    # Generate report
    benchmark.generate_report()
    
    print("\n=== Benchmark Complete ===")
    print(f"Results saved to: {benchmark.output_dir}")


if __name__ == "__main__":
    main()