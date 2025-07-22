"""Unified Memory Benchmarking Tools for Apple Silicon.

This module provides comprehensive benchmarking utilities to measure
memory performance and optimize configurations for different model sizes.
"""

import time
import gc
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import logging

from .memory_manager import UnifiedMemoryManager, get_memory_stats

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    test_name: str
    duration_ms: float
    memory_used_gb: float
    throughput_gb_per_sec: float
    operations_per_sec: float
    peak_memory_gb: float
    efficiency_score: float
    metadata: Dict[str, Any]


class MemoryBenchmarkSuite:
    """Comprehensive memory benchmark suite for Apple Silicon."""
    
    def __init__(self, memory_manager: Optional[UnifiedMemoryManager] = None):
        """Initialize benchmark suite.
        
        Args:
            memory_manager: Optional memory manager instance
        """
        self.memory_manager = memory_manager or UnifiedMemoryManager()
        self.results: List[BenchmarkResult] = []
        
        # Detect device capabilities
        self.device = self._detect_optimal_device()
        logger.info(f"Benchmarking on device: {self.device}")
    
    def _detect_optimal_device(self) -> str:
        """Detect the optimal device for benchmarking."""
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _measure_time_and_memory(self, func: Callable, *args, **kwargs) -> Tuple[float, float, float]:
        """Measure execution time and memory usage.
        
        Args:
            func: Function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (duration_ms, memory_used_gb, peak_memory_gb)
        """
        # Cleanup before measurement
        self.memory_manager.cleanup_memory()
        
        # Initial memory
        initial_stats = get_memory_stats()
        initial_memory = initial_stats.used_gb
        peak_memory = initial_memory
        
        # Measure execution time
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            
            # Force completion for async operations
            if hasattr(result, 'device') and result.device.type in ['cuda', 'mps']:
                if result.device.type == 'cuda':
                    torch.cuda.synchronize()
                elif result.device.type == 'mps':
                    torch.mps.synchronize()
        except Exception as e:
            logger.error(f"Benchmark function failed: {e}")
            return 0.0, 0.0, 0.0
        
        end_time = time.perf_counter()
        
        # Final memory
        final_stats = get_memory_stats()
        final_memory = final_stats.used_gb
        peak_memory = max(peak_memory, final_memory)
        
        duration_ms = (end_time - start_time) * 1000
        memory_used_gb = final_memory - initial_memory
        
        return duration_ms, memory_used_gb, peak_memory
    
    def benchmark_memory_allocation(self, sizes: List[Tuple[int, ...]], dtypes: List[torch.dtype] = None) -> Dict[str, Any]:
        """Benchmark memory allocation performance.
        
        Args:
            sizes: List of tensor shapes to test
            dtypes: List of data types to test
            
        Returns:
            Benchmark results dictionary
        """
        if dtypes is None:
            dtypes = [torch.float32, torch.float16, torch.int8]
        
        logger.info("Benchmarking memory allocation...")
        results = {}
        
        for dtype in dtypes:
            dtype_results = []
            
            for size in sizes:
                def allocate_tensor():
                    tensor = torch.empty(size, dtype=dtype, device=self.device)
                    return tensor
                
                duration_ms, memory_used_gb, peak_memory_gb = self._measure_time_and_memory(allocate_tensor)
                
                # Calculate metrics
                num_elements = np.prod(size)
                bytes_per_element = torch.tensor([], dtype=dtype).element_size()
                total_bytes = num_elements * bytes_per_element
                throughput_gb_per_sec = (total_bytes / (1024**3)) / (duration_ms / 1000) if duration_ms > 0 else 0
                
                result = BenchmarkResult(
                    test_name=f"allocation_{dtype}_{size}",
                    duration_ms=duration_ms,
                    memory_used_gb=memory_used_gb,
                    throughput_gb_per_sec=throughput_gb_per_sec,
                    operations_per_sec=1000 / duration_ms if duration_ms > 0 else 0,
                    peak_memory_gb=peak_memory_gb,
                    efficiency_score=throughput_gb_per_sec / memory_used_gb if memory_used_gb > 0 else 0,
                    metadata={
                        "size": size,
                        "dtype": str(dtype),
                        "num_elements": num_elements,
                        "total_bytes": total_bytes
                    }
                )
                
                dtype_results.append(result)
                self.results.append(result)
                
                # Cleanup
                self.memory_manager.cleanup_memory()
            
            results[str(dtype)] = dtype_results
        
        return results
    
    def benchmark_matrix_operations(self, sizes: List[Tuple[int, int, int]]) -> Dict[str, Any]:
        """Benchmark matrix operation performance.
        
        Args:
            sizes: List of (M, K, N) dimensions for matrix multiplication
            
        Returns:
            Benchmark results dictionary
        """
        logger.info("Benchmarking matrix operations...")
        results = {}
        
        operations = {
            "matmul": lambda a, b: torch.matmul(a, b),
            "addmm": lambda a, b: torch.addmm(torch.zeros(a.shape[0], b.shape[1], device=self.device), a, b),
            "bmm": lambda a, b: torch.bmm(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)
        }
        
        for op_name, op_func in operations.items():
            op_results = []
            
            for m, k, n in sizes:
                def run_operation():
                    a = torch.randn(m, k, device=self.device, dtype=torch.float16)
                    b = torch.randn(k, n, device=self.device, dtype=torch.float16)
                    return op_func(a, b)
                
                duration_ms, memory_used_gb, peak_memory_gb = self._measure_time_and_memory(run_operation)
                
                # Calculate FLOPS
                flops = 2 * m * k * n  # Multiply-add operations
                gflops_per_sec = (flops / 1e9) / (duration_ms / 1000) if duration_ms > 0 else 0
                
                result = BenchmarkResult(
                    test_name=f"{op_name}_{m}x{k}x{n}",
                    duration_ms=duration_ms,
                    memory_used_gb=memory_used_gb,
                    throughput_gb_per_sec=0,  # Not applicable for compute
                    operations_per_sec=gflops_per_sec,
                    peak_memory_gb=peak_memory_gb,
                    efficiency_score=gflops_per_sec / memory_used_gb if memory_used_gb > 0 else 0,
                    metadata={
                        "operation": op_name,
                        "dimensions": (m, k, n),
                        "flops": flops,
                        "gflops_per_sec": gflops_per_sec
                    }
                )
                
                op_results.append(result)
                self.results.append(result)
                
                # Cleanup
                self.memory_manager.cleanup_memory()
            
            results[op_name] = op_results
        
        return results
    
    def benchmark_attention_patterns(self, sequence_lengths: List[int], hidden_sizes: List[int]) -> Dict[str, Any]:
        """Benchmark attention mechanism patterns.
        
        Args:
            sequence_lengths: List of sequence lengths to test
            hidden_sizes: List of hidden dimensions to test
            
        Returns:
            Benchmark results dictionary
        """
        logger.info("Benchmarking attention patterns...")
        results = {}
        
        for seq_len in sequence_lengths:
            for hidden_size in hidden_sizes:
                # Standard attention
                def standard_attention():
                    q = torch.randn(1, seq_len, hidden_size, device=self.device, dtype=torch.float16)
                    k = torch.randn(1, seq_len, hidden_size, device=self.device, dtype=torch.float16)
                    v = torch.randn(1, seq_len, hidden_size, device=self.device, dtype=torch.float16)
                    
                    # Scaled dot-product attention
                    scores = torch.matmul(q, k.transpose(-2, -1)) / (hidden_size ** 0.5)
                    attn_weights = F.softmax(scores, dim=-1)
                    output = torch.matmul(attn_weights, v)
                    return output
                
                duration_ms, memory_used_gb, peak_memory_gb = self._measure_time_and_memory(standard_attention)
                
                # Calculate attention-specific metrics
                attention_ops = seq_len * seq_len * hidden_size * 2  # QK^T + attention * V
                ops_per_sec = attention_ops / (duration_ms / 1000) if duration_ms > 0 else 0
                
                result = BenchmarkResult(
                    test_name=f"attention_seq{seq_len}_hidden{hidden_size}",
                    duration_ms=duration_ms,
                    memory_used_gb=memory_used_gb,
                    throughput_gb_per_sec=0,
                    operations_per_sec=ops_per_sec / 1e6,  # Millions of ops per sec
                    peak_memory_gb=peak_memory_gb,
                    efficiency_score=ops_per_sec / memory_used_gb if memory_used_gb > 0 else 0,
                    metadata={
                        "sequence_length": seq_len,
                        "hidden_size": hidden_size,
                        "attention_ops": attention_ops,
                        "memory_complexity": f"O({seq_len}²)"
                    }
                )
                
                results[f"seq{seq_len}_hidden{hidden_size}"] = result
                self.results.append(result)
                
                # Cleanup
                self.memory_manager.cleanup_memory()
        
        return results
    
    def benchmark_model_sizes(self, parameter_counts: List[int]) -> Dict[str, Any]:
        """Benchmark different model sizes for memory requirements.
        
        Args:
            parameter_counts: List of parameter counts to test
            
        Returns:
            Benchmark results dictionary
        """
        logger.info("Benchmarking model size configurations...")
        results = {}
        
        for param_count in parameter_counts:
            model_results = {}
            
            # Test different precisions
            precisions = ["fp32", "fp16", "int8"]
            
            for precision in precisions:
                # Estimate and test model memory
                estimated_memory = self.memory_manager.estimate_model_memory(
                    param_count, precision
                )
                
                # Check if model fits in memory
                fits_in_memory = self.memory_manager.check_available_memory(estimated_memory)
                
                config = {
                    "parameter_count": param_count,
                    "precision": precision,
                    "estimated_memory_gb": estimated_memory,
                    "fits_in_memory": fits_in_memory,
                    "memory_efficiency": param_count / (estimated_memory * 1e9) if estimated_memory > 0 else 0
                }
                
                model_results[precision] = config
            
            # Get optimization recommendations
            optimization = self.memory_manager.optimize_for_model_size(param_count)
            model_results["optimization"] = optimization
            
            results[f"{param_count // 1e9:.0f}B"] = model_results
        
        return results
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run a comprehensive benchmark suite.
        
        Returns:
            Complete benchmark results
        """
        logger.info("Starting comprehensive memory benchmark...")
        
        comprehensive_results = {
            "system_info": {
                "device": self.device,
                "memory_stats": asdict(get_memory_stats()),
                "timestamp": time.time()
            },
            "benchmarks": {}
        }
        
        # Memory allocation benchmark
        allocation_sizes = [
            (1024, 1024),      # 1M elements
            (2048, 2048),      # 4M elements
            (4096, 4096),      # 16M elements
            (8192, 8192),      # 64M elements
        ]
        comprehensive_results["benchmarks"]["allocation"] = self.benchmark_memory_allocation(allocation_sizes)
        
        # Matrix operations benchmark
        matrix_sizes = [
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096)
        ]
        comprehensive_results["benchmarks"]["matrix_ops"] = self.benchmark_matrix_operations(matrix_sizes)
        
        # Attention patterns benchmark
        sequence_lengths = [128, 512, 1024, 2048]
        hidden_sizes = [512, 768, 1024]
        comprehensive_results["benchmarks"]["attention"] = self.benchmark_attention_patterns(
            sequence_lengths, hidden_sizes
        )
        
        # Model size analysis
        parameter_counts = [1e9, 7e9, 13e9, 30e9]  # 1B, 7B, 13B, 30B
        comprehensive_results["benchmarks"]["model_sizes"] = self.benchmark_model_sizes(parameter_counts)
        
        # Performance summary
        comprehensive_results["summary"] = self._generate_summary()
        
        logger.info("Comprehensive benchmark completed")
        return comprehensive_results
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate performance summary from benchmark results."""
        if not self.results:
            return {}
        
        # Calculate aggregated metrics
        total_duration = sum(r.duration_ms for r in self.results)
        avg_duration = total_duration / len(self.results)
        max_throughput = max(r.throughput_gb_per_sec for r in self.results if r.throughput_gb_per_sec > 0)
        avg_efficiency = sum(r.efficiency_score for r in self.results) / len(self.results)
        
        # Find best performing operations
        best_throughput = max(self.results, key=lambda x: x.throughput_gb_per_sec)
        best_efficiency = max(self.results, key=lambda x: x.efficiency_score)
        
        return {
            "total_tests": len(self.results),
            "total_duration_ms": total_duration,
            "average_duration_ms": avg_duration,
            "max_throughput_gb_per_sec": max_throughput,
            "average_efficiency_score": avg_efficiency,
            "best_throughput_test": best_throughput.test_name,
            "best_efficiency_test": best_efficiency.test_name,
            "device_performance_rating": self._calculate_performance_rating()
        }
    
    def _calculate_performance_rating(self) -> str:
        """Calculate overall device performance rating."""
        if not self.results:
            return "Unknown"
        
        # Simple heuristic based on average efficiency
        avg_efficiency = sum(r.efficiency_score for r in self.results) / len(self.results)
        
        if avg_efficiency > 100:
            return "Excellent"
        elif avg_efficiency > 50:
            return "Good"
        elif avg_efficiency > 20:
            return "Fair"
        else:
            return "Poor"
    
    def save_results(self, filepath: str):
        """Save benchmark results to file.
        
        Args:
            filepath: Path to save results
        """
        results_data = {
            "results": [asdict(r) for r in self.results],
            "summary": self._generate_summary(),
            "system_info": {
                "device": self.device,
                "memory_stats": asdict(get_memory_stats())
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Benchmark results saved to: {filepath}")
    
    def generate_report(self) -> str:
        """Generate a human-readable benchmark report."""
        if not self.results:
            return "No benchmark results available."
        
        report = ["Apple Silicon Memory Benchmark Report", "=" * 50, ""]
        
        # System info
        stats = get_memory_stats()
        report.extend([
            "SYSTEM INFORMATION:",
            f"  Device: {self.device}",
            f"  Total Memory: {stats.total_gb:.1f} GB",
            f"  Available Memory: {stats.available_gb:.1f} GB",
            f"  Memory Utilization: {stats.utilization_percent:.1f}%",
            ""
        ])
        
        # Summary
        summary = self._generate_summary()
        report.extend([
            "BENCHMARK SUMMARY:",
            f"  Total Tests: {summary.get('total_tests', 0)}",
            f"  Average Duration: {summary.get('average_duration_ms', 0):.2f} ms",
            f"  Max Throughput: {summary.get('max_throughput_gb_per_sec', 0):.2f} GB/s",
            f"  Performance Rating: {summary.get('device_performance_rating', 'Unknown')}",
            ""
        ])
        
        # Top performing tests
        if self.results:
            fastest_test = min(self.results, key=lambda x: x.duration_ms)
            highest_throughput = max(self.results, key=lambda x: x.throughput_gb_per_sec)
            
            report.extend([
                "TOP PERFORMERS:",
                f"  Fastest Test: {fastest_test.test_name} ({fastest_test.duration_ms:.2f} ms)",
                f"  Highest Throughput: {highest_throughput.test_name} ({highest_throughput.throughput_gb_per_sec:.2f} GB/s)",
                ""
            ])
        
        # Recommendations
        report.extend([
            "RECOMMENDATIONS:",
            f"  • Use {self.device} for optimal performance",
            f"  • Current memory utilization: {stats.utilization_percent:.1f}%"
        ])
        
        if stats.utilization_percent > 80:
            report.append("  • Consider reducing batch size or model size")
        elif stats.utilization_percent < 50:
            report.append("  • Memory headroom available for larger models")
        
        return '\n'.join(report)


def run_quick_benchmark() -> Dict[str, Any]:
    """Run a quick benchmark for basic performance assessment."""
    suite = MemoryBenchmarkSuite()
    
    # Quick tests
    quick_sizes = [(1024, 1024), (2048, 2048)]
    matrix_sizes = [(512, 512, 512), (1024, 1024, 1024)]
    
    results = {}
    results["allocation"] = suite.benchmark_memory_allocation(quick_sizes)
    results["matrix_ops"] = suite.benchmark_matrix_operations(matrix_sizes)
    results["summary"] = suite._generate_summary()
    
    return results


def main():
    """Main entry point for standalone benchmarking."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Apple Silicon Memory Benchmark")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    suite = MemoryBenchmarkSuite()
    
    if args.quick:
        results = run_quick_benchmark()
    else:
        results = suite.run_comprehensive_benchmark()
    
    # Print report
    print(suite.generate_report())
    
    # Save results if requested
    if args.output:
        suite.save_results(args.output)
    else:
        # Default filename
        timestamp = int(time.time())
        suite.save_results(f"benchmark_results_{timestamp}.json")


if __name__ == "__main__":
    main()