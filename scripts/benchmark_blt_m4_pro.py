#!/usr/bin/env python3
"""
Benchmark BLT M4 Pro Optimizations
==================================

This script benchmarks the M4 Pro optimized BLT implementation
and compares it with the standard implementation.
"""

import time
import mlx.core as mx
import numpy as np
from typing import Dict, List
import json
from datetime import datetime

# Add project root to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.blt import (
    BLTPipeline,
    create_blt_model,
    M4ProOptimizedBLT,
    create_m4_pro_optimized_blt
)
from utils.hardware.memory_manager import M4ProMemoryManager


def benchmark_model(model, input_data: mx.array, num_runs: int = 50) -> Dict[str, float]:
    """Benchmark a model with given input data."""
    # Warmup
    for _ in range(5):
        if hasattr(model, 'forward'):
            _ = model.forward(input_data, use_cache=False)
        else:
            _ = model(input_data)
        mx.eval(mx.array(0))  # Force synchronization
    
    # Benchmark
    times = []
    memory_usage = []
    memory_manager = M4ProMemoryManager()
    
    for i in range(num_runs):
        start_memory = memory_manager.get_current_usage()
        start_time = time.perf_counter()
        
        if hasattr(model, 'forward'):
            output, metadata = model.forward(input_data, use_cache=False)
        else:
            output = model(input_data)
            metadata = {}
        
        mx.eval(output)  # Force synchronization
        end_time = time.perf_counter()
        end_memory = memory_manager.get_current_usage()
        
        times.append(end_time - start_time)
        memory_usage.append(end_memory - start_memory)
    
    # Calculate statistics
    times_ms = [t * 1000 for t in times]
    
    return {
        'avg_latency_ms': np.mean(times_ms),
        'p50_latency_ms': np.percentile(times_ms, 50),
        'p90_latency_ms': np.percentile(times_ms, 90),
        'p99_latency_ms': np.percentile(times_ms, 99),
        'min_latency_ms': np.min(times_ms),
        'max_latency_ms': np.max(times_ms),
        'std_latency_ms': np.std(times_ms),
        'avg_memory_mb': np.mean(memory_usage) / (1024 * 1024),
        'throughput_tokens_per_sec': (input_data.shape[0] * input_data.shape[1]) / np.mean(times)
    }


def compare_implementations():
    """Compare standard and M4 Pro optimized implementations."""
    print("=" * 80)
    print("BLT M4 Pro Optimization Benchmark")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Hardware: Apple Silicon M4 Pro")
    print("=" * 80)
    
    # Test configurations
    test_configs = [
        {"batch_size": 1, "seq_len": 128, "name": "Small (1x128)"},
        {"batch_size": 4, "seq_len": 512, "name": "Medium (4x512)"},
        {"batch_size": 16, "seq_len": 1024, "name": "Large (16x1024)"},
        {"batch_size": 32, "seq_len": 2048, "name": "XLarge (32x2048)"}
    ]
    
    # Model configurations
    model_configs = ["small", "medium", "large"]
    
    results = {}
    
    for model_size in model_configs:
        print(f"\n\nTesting {model_size.upper()} model configuration...")
        print("-" * 60)
        
        # Create models
        print("Creating standard BLT model...")
        standard_model = create_blt_model(model_size=model_size)
        
        print("Creating M4 Pro optimized BLT model...")
        optimized_model = create_m4_pro_optimized_blt(model_size=model_size)
        
        # Run benchmarks for each test configuration
        for config in test_configs:
            print(f"\n  Testing {config['name']}...")
            
            # Create test input
            test_input = mx.random.randint(
                0, 256, 
                (config['batch_size'], config['seq_len'])
            )
            
            # Benchmark standard model
            print("    Benchmarking standard implementation...")
            standard_results = benchmark_model(standard_model, test_input)
            
            # Benchmark optimized model
            print("    Benchmarking M4 Pro optimized implementation...")
            optimized_results = benchmark_model(optimized_model, test_input)
            
            # Calculate speedup
            speedup = standard_results['avg_latency_ms'] / optimized_results['avg_latency_ms']
            memory_reduction = 1 - (optimized_results['avg_memory_mb'] / standard_results['avg_memory_mb'])
            
            # Store results
            key = f"{model_size}_{config['name']}"
            results[key] = {
                'config': config,
                'standard': standard_results,
                'optimized': optimized_results,
                'speedup': speedup,
                'memory_reduction_percent': memory_reduction * 100
            }
            
            # Print summary
            print(f"    Results:")
            print(f"      Standard:  {standard_results['avg_latency_ms']:.2f}ms")
            print(f"      Optimized: {optimized_results['avg_latency_ms']:.2f}ms")
            print(f"      Speedup:   {speedup:.2f}x")
            print(f"      Memory:    {memory_reduction*100:.1f}% reduction")
    
    # Print overall summary
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    avg_speedups = []
    avg_memory_reductions = []
    
    for model_size in model_configs:
        size_speedups = []
        size_memory = []
        
        for config in test_configs:
            key = f"{model_size}_{config['name']}"
            if key in results:
                size_speedups.append(results[key]['speedup'])
                size_memory.append(results[key]['memory_reduction_percent'])
        
        avg_speedup = np.mean(size_speedups)
        avg_memory = np.mean(size_memory)
        
        avg_speedups.append(avg_speedup)
        avg_memory_reductions.append(avg_memory)
        
        print(f"\n{model_size.upper()} Model:")
        print(f"  Average Speedup: {avg_speedup:.2f}x")
        print(f"  Average Memory Reduction: {avg_memory:.1f}%")
    
    print(f"\n\nOVERALL:")
    print(f"  Average Speedup: {np.mean(avg_speedups):.2f}x")
    print(f"  Average Memory Reduction: {np.mean(avg_memory_reductions):.1f}%")
    
    # Save detailed results
    output_file = f"blt_m4_pro_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n\nDetailed results saved to: {output_file}")
    
    return results


def test_specific_optimizations():
    """Test specific M4 Pro optimizations."""
    print("\n\n" + "=" * 80)
    print("TESTING SPECIFIC M4 PRO OPTIMIZATIONS")
    print("=" * 80)
    
    # Create optimized model
    model = create_m4_pro_optimized_blt(model_size="medium")
    
    # Test cache effectiveness
    print("\n1. Testing Cache Effectiveness...")
    test_input = mx.random.randint(0, 256, (8, 512))
    
    # First run without cache
    start = time.perf_counter()
    _, metadata1 = model.forward(test_input, use_cache=False)
    time1 = time.perf_counter() - start
    
    # Second run with cache
    start = time.perf_counter()
    _, metadata2 = model.forward(test_input, use_cache=True)
    time2 = time.perf_counter() - start
    
    # Third run should hit cache
    start = time.perf_counter()
    _, metadata3 = model.forward(test_input, use_cache=True)
    time3 = time.perf_counter() - start
    
    print(f"  No cache:    {time1*1000:.2f}ms")
    print(f"  Cold cache:  {time2*1000:.2f}ms")
    print(f"  Warm cache:  {time3*1000:.2f}ms")
    print(f"  Cache speedup: {time1/time3:.2f}x")
    
    # Test unified memory optimization
    print("\n2. Testing Unified Memory Optimization...")
    memory_manager = M4ProMemoryManager()
    
    # Large batch to test memory efficiency
    large_input = mx.random.randint(0, 256, (64, 1024))
    
    start_memory = memory_manager.get_current_usage()
    output, metadata = model.forward(large_input)
    end_memory = memory_manager.get_current_usage()
    
    memory_used_mb = (end_memory - start_memory) / (1024 * 1024)
    print(f"  Memory used: {memory_used_mb:.2f}MB")
    print(f"  Memory efficiency: {metadata.get('memory_usage_mb', 0):.2f}MB reported")
    
    # Test Neural Engine utilization
    print("\n3. Testing Neural Engine Optimization...")
    # This would interface with actual Neural Engine metrics
    print(f"  GPU utilization: {metadata.get('gpu_utilization', 0)*100:.1f}%")
    print(f"  Optimization active: {metadata.get('optimization_active', False)}")
    
    # Test bandwidth optimization
    print("\n4. Testing Memory Bandwidth Optimization...")
    bandwidth_results = model.benchmark((32, 2048), num_runs=20)
    
    print(f"  Throughput: {bandwidth_results['throughput_mbps']:.2f} MB/s")
    print(f"  Bandwidth utilization: {bandwidth_results['memory_bandwidth_utilization']*100:.1f}%")
    print(f"  P99 latency: {bandwidth_results['p99_latency_ms']:.2f}ms")


if __name__ == "__main__":
    # Ensure we're on GPU
    mx.set_default_device(mx.gpu)
    
    print("Starting BLT M4 Pro optimization benchmark...")
    print(f"Default device: {mx.default_device()}")
    
    # Run comparison benchmark
    results = compare_implementations()
    
    # Test specific optimizations
    test_specific_optimizations()
    
    print("\n\nBenchmark complete!")