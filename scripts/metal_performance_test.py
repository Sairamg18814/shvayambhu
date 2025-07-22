#!/usr/bin/env python3
"""Metal Performance Testing Harness for Shvayambhu.

This script provides comprehensive testing of Metal Performance Shaders
and Apple Silicon optimization for the Shvayambhu LLM project.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.hardware.benchmarking import MemoryBenchmarkSuite, BenchmarkResult
from utils.hardware.memory_manager import UnifiedMemoryManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class MetalPerformanceTestSuite:
    """Comprehensive Metal performance testing suite."""
    
    def __init__(self, device: str = "auto"):
        """Initialize test suite.
        
        Args:
            device: Target device ("auto", "mps", "cpu")
        """
        self.device = self._select_device(device)
        self.memory_manager = UnifiedMemoryManager()
        self.benchmark_suite = MemoryBenchmarkSuite(self.memory_manager)
        self.test_results = {}
        
        logger.info(f"Metal Performance Test Suite initialized on {self.device}")
    
    def _select_device(self, device: str) -> torch.device:
        """Select optimal device for testing."""
        if device == "auto":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def test_basic_operations(self) -> Dict[str, Any]:
        """Test basic Metal operations."""
        logger.info("Testing basic Metal operations...")
        
        results = {}
        
        # Test tensor creation
        sizes = [(1024, 1024), (2048, 2048), (4096, 4096)]
        dtypes = [torch.float32, torch.float16, torch.int8]
        
        for size in sizes:
            for dtype in dtypes:
                test_name = f"tensor_creation_{size[0]}x{size[1]}_{dtype}"
                
                start_time = time.perf_counter()
                
                # Create tensor
                tensor = torch.randn(size, dtype=dtype, device=self.device)
                
                # Synchronize to ensure completion
                if self.device.type == "mps":
                    torch.mps.synchronize()
                elif self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                
                results[test_name] = {
                    "duration_ms": (end_time - start_time) * 1000,
                    "size": size,
                    "dtype": str(dtype),
                    "memory_mb": tensor.numel() * tensor.element_size() / (1024**2)
                }
                
                # Cleanup
                del tensor
                if self.device.type == "mps":
                    torch.mps.empty_cache()
        
        self.test_results["basic_operations"] = results
        return results
    
    def test_matrix_operations(self) -> Dict[str, Any]:
        """Test matrix multiplication performance."""
        logger.info("Testing matrix operations...")
        
        results = {}
        
        # Test different matrix sizes
        sizes = [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]
        dtypes = [torch.float32, torch.float16]
        
        for m, n in sizes:
            k = n  # Square matrices
            
            for dtype in dtypes:
                test_name = f"matmul_{m}x{k}x{n}_{dtype}"
                
                # Create test matrices
                a = torch.randn(m, k, dtype=dtype, device=self.device)
                b = torch.randn(k, n, dtype=dtype, device=self.device)
                
                # Warmup
                for _ in range(5):
                    torch.matmul(a, b)
                
                if self.device.type == "mps":
                    torch.mps.synchronize()
                elif self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                # Benchmark
                start_time = time.perf_counter()
                
                num_iterations = 10
                for _ in range(num_iterations):
                    result = torch.matmul(a, b)
                
                if self.device.type == "mps":
                    torch.mps.synchronize()
                elif self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                
                # Calculate metrics
                total_time = end_time - start_time
                avg_time_ms = (total_time / num_iterations) * 1000
                
                # FLOPS calculation
                flops = 2 * m * k * n  # Multiply-add operations
                gflops_per_sec = (flops / 1e9) / (total_time / num_iterations)
                
                results[test_name] = {
                    "duration_ms": avg_time_ms,
                    "gflops_per_sec": gflops_per_sec,
                    "matrix_size": (m, k, n),
                    "dtype": str(dtype),
                    "iterations": num_iterations
                }
                
                # Cleanup
                del a, b, result
                if self.device.type == "mps":
                    torch.mps.empty_cache()
        
        self.test_results["matrix_operations"] = results
        return results
    
    def test_attention_mechanism(self) -> Dict[str, Any]:
        """Test attention mechanism performance."""
        logger.info("Testing attention mechanisms...")
        
        results = {}
        
        # Different configurations
        configs = [
            {"seq_len": 128, "hidden_size": 512, "num_heads": 8},
            {"seq_len": 512, "hidden_size": 768, "num_heads": 12},
            {"seq_len": 1024, "hidden_size": 1024, "num_heads": 16},
            {"seq_len": 2048, "hidden_size": 1024, "num_heads": 16}
        ]
        
        for config in configs:
            seq_len = config["seq_len"]
            hidden_size = config["hidden_size"]
            num_heads = config["num_heads"]
            head_dim = hidden_size // num_heads
            
            test_name = f"attention_seq{seq_len}_h{hidden_size}_heads{num_heads}"
            
            # Create attention inputs
            batch_size = 1
            q = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device=self.device)
            k = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device=self.device)
            v = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device=self.device)
            
            # Reshape for multi-head attention
            q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            
            # Warmup
            for _ in range(3):
                scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
                attn_weights = F.softmax(scores, dim=-1)
                output = torch.matmul(attn_weights, v)
            
            if self.device.type == "mps":
                torch.mps.synchronize()
            elif self.device.type == "cuda":
                torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.perf_counter()
            
            num_iterations = 5
            for _ in range(num_iterations):
                # Scaled dot-product attention
                scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
                attn_weights = F.softmax(scores, dim=-1)
                output = torch.matmul(attn_weights, v)
            
            if self.device.type == "mps":
                torch.mps.synchronize()
            elif self.device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            # Calculate metrics
            total_time = end_time - start_time
            avg_time_ms = (total_time / num_iterations) * 1000
            
            # Memory complexity
            attention_memory = batch_size * num_heads * seq_len * seq_len * 2  # fp16
            
            results[test_name] = {
                "duration_ms": avg_time_ms,
                "sequence_length": seq_len,
                "hidden_size": hidden_size,
                "num_heads": num_heads,
                "attention_memory_mb": attention_memory / (1024**2),
                "iterations": num_iterations
            }
            
            # Cleanup
            del q, k, v, scores, attn_weights, output
            if self.device.type == "mps":
                torch.mps.empty_cache()
        
        self.test_results["attention_mechanisms"] = results
        return results
    
    def test_quantization_performance(self) -> Dict[str, Any]:
        """Test quantization performance impact."""
        logger.info("Testing quantization performance...")
        
        results = {}
        
        # Create a simple model for testing
        class TestModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.linear1 = nn.Linear(input_size, hidden_size)
                self.linear2 = nn.Linear(hidden_size, hidden_size)
                self.linear3 = nn.Linear(hidden_size, output_size)
                self.activation = nn.GELU()
            
            def forward(self, x):
                x = self.activation(self.linear1(x))
                x = self.activation(self.linear2(x))
                return self.linear3(x)
        
        # Test configurations
        model_config = {"input_size": 1024, "hidden_size": 4096, "output_size": 1024}
        input_shape = (32, model_config["input_size"])  # Batch size 32
        
        dtypes = [torch.float32, torch.float16]
        
        for dtype in dtypes:
            test_name = f"quantization_{dtype}"
            
            # Create model and input
            model = TestModel(**model_config).to(device=self.device, dtype=dtype)
            input_tensor = torch.randn(input_shape, dtype=dtype, device=self.device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    output = model(input_tensor)
            
            if self.device.type == "mps":
                torch.mps.synchronize()
            elif self.device.type == "cuda":
                torch.cuda.synchronize()
            
            # Benchmark inference
            start_time = time.perf_counter()
            
            num_iterations = 20
            with torch.no_grad():
                for _ in range(num_iterations):
                    output = model(input_tensor)
            
            if self.device.type == "mps":
                torch.mps.synchronize()
            elif self.device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            # Calculate model size
            param_count = sum(p.numel() for p in model.parameters())
            model_size_mb = param_count * torch.tensor([], dtype=dtype).element_size() / (1024**2)
            
            # Calculate metrics
            total_time = end_time - start_time
            avg_time_ms = (total_time / num_iterations) * 1000
            throughput = num_iterations / total_time
            
            results[test_name] = {
                "duration_ms": avg_time_ms,
                "throughput_inferences_per_sec": throughput,
                "model_size_mb": model_size_mb,
                "parameter_count": param_count,
                "dtype": str(dtype),
                "iterations": num_iterations
            }
            
            # Cleanup
            del model, input_tensor, output
            if self.device.type == "mps":
                torch.mps.empty_cache()
        
        self.test_results["quantization_performance"] = results
        return results
    
    def test_memory_bandwidth(self) -> Dict[str, Any]:
        """Test memory bandwidth utilization."""
        logger.info("Testing memory bandwidth...")
        
        results = {}
        
        # Different tensor sizes for bandwidth testing
        sizes = [
            (1024, 1024),     # 1M elements
            (2048, 2048),     # 4M elements  
            (4096, 4096),     # 16M elements
            (8192, 8192),     # 64M elements
        ]
        
        dtypes = [torch.float32, torch.float16]
        
        for dtype in dtypes:
            for size in sizes:
                test_name = f"bandwidth_{size[0]}x{size[1]}_{dtype}"
                
                # Create source tensor
                src_tensor = torch.randn(size, dtype=dtype, device=self.device)
                
                # Warmup
                for _ in range(3):
                    dst_tensor = src_tensor.clone()
                
                if self.device.type == "mps":
                    torch.mps.synchronize()
                elif self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                # Benchmark memory copy
                start_time = time.perf_counter()
                
                num_iterations = 50
                for _ in range(num_iterations):
                    dst_tensor = src_tensor.clone()
                
                if self.device.type == "mps":
                    torch.mps.synchronize()
                elif self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                
                # Calculate bandwidth
                total_time = end_time - start_time
                bytes_per_copy = src_tensor.numel() * src_tensor.element_size()
                total_bytes = bytes_per_copy * num_iterations
                bandwidth_gb_per_sec = (total_bytes / (1024**3)) / total_time
                
                results[test_name] = {
                    "bandwidth_gb_per_sec": bandwidth_gb_per_sec,
                    "tensor_size": size,
                    "dtype": str(dtype),
                    "bytes_per_copy": bytes_per_copy,
                    "iterations": num_iterations
                }
                
                # Cleanup
                del src_tensor, dst_tensor
                if self.device.type == "mps":
                    torch.mps.empty_cache()
        
        self.test_results["memory_bandwidth"] = results
        return results
    
    def test_blt_specific_operations(self) -> Dict[str, Any]:
        """Test BLT-specific operations for Shvayambhu."""
        logger.info("Testing BLT-specific operations...")
        
        results = {}
        
        # Simulate byte-level operations
        byte_sequences = [256, 512, 1024, 2048, 4096]  # Different sequence lengths
        
        for seq_len in byte_sequences:
            test_name = f"blt_byte_processing_{seq_len}"
            
            # Create byte sequence (as uint8 tensor)
            byte_tensor = torch.randint(0, 256, (seq_len,), dtype=torch.uint8, device=self.device)
            
            # Simulate entropy calculation
            def calculate_entropy():
                # Convert to float for calculations
                float_tensor = byte_tensor.float()
                
                # Simple entropy approximation using histogram
                histogram = torch.zeros(256, device=self.device)
                for i in range(256):
                    histogram[i] = (byte_tensor == i).float().sum()
                
                # Normalize
                probs = histogram / seq_len
                probs = probs[probs > 0]  # Remove zeros
                
                # Calculate entropy
                entropy = -(probs * torch.log2(probs)).sum()
                return entropy
            
            # Warmup
            for _ in range(3):
                entropy = calculate_entropy()
            
            if self.device.type == "mps":
                torch.mps.synchronize()
            elif self.device.type == "cuda":
                torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.perf_counter()
            
            num_iterations = 100
            for _ in range(num_iterations):
                entropy = calculate_entropy()
            
            if self.device.type == "mps":
                torch.mps.synchronize()
            elif self.device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            # Calculate metrics
            total_time = end_time - start_time
            avg_time_ms = (total_time / num_iterations) * 1000
            bytes_per_sec = (seq_len * num_iterations) / total_time
            
            results[test_name] = {
                "duration_ms": avg_time_ms,
                "bytes_per_sec": bytes_per_sec,
                "sequence_length": seq_len,
                "iterations": num_iterations
            }
            
            # Cleanup
            del byte_tensor
            if self.device.type == "mps":
                torch.mps.empty_cache()
        
        self.test_results["blt_operations"] = results
        return results
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all performance tests."""
        logger.info("Starting comprehensive Metal performance test...")
        
        # Run all test suites
        self.test_basic_operations()
        self.test_matrix_operations() 
        self.test_attention_mechanism()
        self.test_quantization_performance()
        self.test_memory_bandwidth()
        self.test_blt_specific_operations()
        
        # Add system information
        self.test_results["system_info"] = {
            "device": str(self.device),
            "pytorch_version": torch.__version__,
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
            "timestamp": time.time()
        }
        
        # Generate summary
        self.test_results["summary"] = self._generate_summary()
        
        logger.info("Comprehensive test completed")
        return self.test_results
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate performance summary."""
        summary = {}
        
        # Aggregate metrics from different test categories
        for category, tests in self.test_results.items():
            if category in ["system_info", "summary"]:
                continue
            
            if isinstance(tests, dict):
                durations = []
                for test_name, test_data in tests.items():
                    if isinstance(test_data, dict) and "duration_ms" in test_data:
                        durations.append(test_data["duration_ms"])
                
                if durations:
                    summary[f"{category}_avg_duration_ms"] = sum(durations) / len(durations)
                    summary[f"{category}_max_duration_ms"] = max(durations)
                    summary[f"{category}_min_duration_ms"] = min(durations)
        
        # Overall performance rating
        if "matrix_operations" in self.test_results:
            matrix_results = self.test_results["matrix_operations"]
            gflops_values = [test["gflops_per_sec"] for test in matrix_results.values() 
                           if "gflops_per_sec" in test]
            
            if gflops_values:
                max_gflops = max(gflops_values)
                if max_gflops > 1000:
                    summary["performance_rating"] = "Excellent"
                elif max_gflops > 500:
                    summary["performance_rating"] = "Good"
                elif max_gflops > 100:
                    summary["performance_rating"] = "Fair"
                else:
                    summary["performance_rating"] = "Poor"
                
                summary["max_gflops"] = max_gflops
        
        return summary
    
    def save_results(self, filepath: str):
        """Save test results to file."""
        with open(filepath, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        logger.info(f"Test results saved to: {filepath}")
    
    def generate_report(self) -> str:
        """Generate human-readable performance report."""
        if not self.test_results:
            return "No test results available."
        
        report = ["Metal Performance Test Report", "=" * 40, ""]
        
        # System info
        if "system_info" in self.test_results:
            sys_info = self.test_results["system_info"]
            report.extend([
                "SYSTEM INFORMATION:",
                f"  Device: {sys_info.get('device', 'Unknown')}",
                f"  PyTorch Version: {sys_info.get('pytorch_version', 'Unknown')}",
                f"  MPS Available: {sys_info.get('mps_available', False)}",
                f"  CUDA Available: {sys_info.get('cuda_available', False)}",
                ""
            ])
        
        # Summary
        if "summary" in self.test_results:
            summary = self.test_results["summary"]
            report.extend([
                "PERFORMANCE SUMMARY:",
                f"  Performance Rating: {summary.get('performance_rating', 'Unknown')}",
                f"  Max GFLOPS: {summary.get('max_gflops', 0):.1f}",
                ""
            ])
        
        # Category summaries
        for category in ["basic_operations", "matrix_operations", "attention_mechanisms", 
                        "quantization_performance", "memory_bandwidth", "blt_operations"]:
            if category in self.test_results:
                report.append(f"{category.upper().replace('_', ' ')}:")
                
                tests = self.test_results[category]
                if isinstance(tests, dict):
                    for test_name, test_data in tests.items():
                        if isinstance(test_data, dict) and "duration_ms" in test_data:
                            duration = test_data["duration_ms"]
                            report.append(f"  {test_name}: {duration:.2f} ms")
                
                report.append("")
        
        return '\n'.join(report)


def main():
    """Main entry point for Metal performance testing."""
    parser = argparse.ArgumentParser(description="Metal Performance Test Suite")
    parser.add_argument("--device", default="auto", 
                       choices=["auto", "mps", "cuda", "cpu"],
                       help="Target device for testing")
    parser.add_argument("--output", type=str, default="metal_performance_results.json",
                       help="Output file for results")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick test suite only")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create test suite
    test_suite = MetalPerformanceTestSuite(device=args.device)
    
    # Run tests
    if args.quick:
        # Quick test - only basic operations
        test_suite.test_basic_operations()
        test_suite.test_matrix_operations()
    else:
        # Full comprehensive test
        test_suite.run_comprehensive_test()
    
    # Generate and print report
    report = test_suite.generate_report()
    print(report)
    
    # Save results
    test_suite.save_results(args.output)
    
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()