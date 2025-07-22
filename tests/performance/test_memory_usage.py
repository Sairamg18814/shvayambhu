"""Memory usage profiling for BLT components.

This module profiles memory usage patterns across the BLT pipeline,
identifying potential memory leaks and optimization opportunities.
"""

import gc
import pytest
import torch
import numpy as np
import tracemalloc
import psutil
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from contextlib import contextmanager

from shvayambhu.core.blt.pipeline import BLTPipeline
from shvayambhu.core.blt.encoder import LocalEncoder
from shvayambhu.core.blt.transformer import LatentTransformer
from shvayambhu.core.blt.decoder import LocalDecoder
from shvayambhu.core.blt.patching import BLTInputProcessor


@dataclass
class MemorySnapshot:
    """Container for memory usage snapshot."""
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    available_mb: float  # Available system memory
    percent: float  # Percentage of system memory used
    gpu_mb: Optional[float] = None  # GPU memory if available
    
    def __repr__(self):
        return (
            f"RSS: {self.rss_mb:.2f}MB, VMS: {self.vms_mb:.2f}MB, "
            f"Available: {self.available_mb:.2f}MB, Used: {self.percent:.1f}%"
        )


@contextmanager
def memory_tracker():
    """Context manager for tracking memory usage."""
    tracemalloc.start()
    gc.collect()
    
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024
    
    yield
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    end_memory = process.memory_info().rss / 1024 / 1024
    
    print(f"Memory used: {end_memory - start_memory:.2f}MB")
    print(f"Peak tracked: {peak / 1024 / 1024:.2f}MB")


class MemoryProfiler:
    """Comprehensive memory profiler for BLT components."""
    
    def __init__(self):
        self.device = torch.device("cpu")
        self.process = psutil.Process(os.getpid())
        self.snapshots: List[MemorySnapshot] = []
    
    def take_snapshot(self, label: str = "") -> MemorySnapshot:
        """Take a memory snapshot."""
        gc.collect()  # Force garbage collection
        
        mem_info = self.process.memory_info()
        sys_mem = psutil.virtual_memory()
        
        snapshot = MemorySnapshot(
            rss_mb=mem_info.rss / 1024 / 1024,
            vms_mb=mem_info.vms / 1024 / 1024,
            available_mb=sys_mem.available / 1024 / 1024,
            percent=self.process.memory_percent()
        )
        
        self.snapshots.append(snapshot)
        if label:
            print(f"{label}: {snapshot}")
        
        return snapshot
    
    def profile_component_memory(self, component_name: str, component, test_input):
        """Profile memory usage of a specific component."""
        print(f"\n=== Profiling {component_name} ===")
        
        # Initial snapshot
        self.take_snapshot("Before initialization")
        
        # Run component multiple times
        for i in range(5):
            self.take_snapshot(f"Before run {i+1}")
            
            with memory_tracker():
                output = component(test_input)
                
            self.take_snapshot(f"After run {i+1}")
            
            # Force cleanup
            del output
            gc.collect()
        
        # Check for memory leaks
        initial_memory = self.snapshots[1].rss_mb
        final_memory = self.snapshots[-1].rss_mb
        
        print(f"\nMemory leak check:")
        print(f"Initial: {initial_memory:.2f}MB")
        print(f"Final: {final_memory:.2f}MB")
        print(f"Difference: {final_memory - initial_memory:.2f}MB")
        
        if final_memory - initial_memory > 10:  # More than 10MB growth
            print("WARNING: Potential memory leak detected!")
    
    def profile_pipeline_memory(self):
        """Profile memory usage of complete pipeline."""
        print("\n=== Profiling Complete Pipeline ===")
        
        config = {
            "vocab_size": 256,
            "hidden_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "patch_embedding_dim": 768,
            "max_patch_size": 32,
            "device": self.device
        }
        
        # Take initial snapshot
        self.take_snapshot("Before pipeline creation")
        
        # Create pipeline
        pipeline = BLTPipeline(config)
        self.take_snapshot("After pipeline creation")
        
        # Test with different text sizes
        text_sizes = [100, 1000, 10000, 50000]
        
        for size in text_sizes:
            text = "a" * size
            byte_seq = text.encode('utf-8')
            
            self.take_snapshot(f"Before processing {size} bytes")
            
            with memory_tracker():
                output = pipeline.process_bytes(byte_seq, mode='inference')
            
            self.take_snapshot(f"After processing {size} bytes")
            
            del output
            gc.collect()
    
    def profile_batch_memory_scaling(self):
        """Profile how memory scales with batch size."""
        print("\n=== Profiling Batch Memory Scaling ===")
        
        pipeline = BLTPipeline({
            "vocab_size": 256,
            "hidden_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "patch_embedding_dim": 768,
            "max_patch_size": 32,
            "device": self.device
        })
        
        base_text = "Test text for batch processing " * 10
        batch_sizes = [1, 2, 4, 8, 16, 32]
        
        memory_usage = []
        
        for batch_size in batch_sizes:
            texts = [base_text] * batch_size
            byte_sequences = [text.encode('utf-8') for text in texts]
            
            # Clear memory
            gc.collect()
            before = self.take_snapshot()
            
            # Process batch
            with memory_tracker():
                outputs = pipeline.process_batch(byte_sequences, mode='inference')
            
            after = self.take_snapshot()
            memory_used = after.rss_mb - before.rss_mb
            memory_usage.append((batch_size, memory_used))
            
            print(f"Batch size {batch_size}: {memory_used:.2f}MB "
                  f"({memory_used/batch_size:.2f}MB per item)")
            
            del outputs
            gc.collect()
        
        # Check if memory scales linearly
        if len(memory_usage) > 1:
            # Simple linear regression
            x = np.array([m[0] for m in memory_usage])
            y = np.array([m[1] for m in memory_usage])
            
            # Fit line
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            
            print(f"\nMemory scaling: {m:.2f}MB per item + {c:.2f}MB base")
    
    def profile_cache_memory(self):
        """Profile memory usage of caching mechanisms."""
        print("\n=== Profiling Cache Memory ===")
        
        from shvayambhu.core.blt.entropy import EntropyCalculator
        
        calc = EntropyCalculator(window_size=256)
        
        # Generate test data
        test_sequences = [
            b"repeated" * 100,  # Will be cached
            b"unique" + bytes(range(100)),  # Won't be cached effectively
        ]
        
        for i, seq in enumerate(test_sequences):
            self.take_snapshot(f"Before sequence {i+1}")
            
            # Calculate entropy multiple times
            for _ in range(10):
                entropy = calc.calculate_entropy(seq)
            
            self.take_snapshot(f"After sequence {i+1}")
            
            print(f"Cache size: {len(calc.cache)} entries")
    
    def detect_memory_leaks(self):
        """Run leak detection tests."""
        print("\n=== Memory Leak Detection ===")
        
        pipeline = BLTPipeline({
            "vocab_size": 256,
            "hidden_dim": 768,
            "num_layers": 6,
            "num_heads": 8,
            "patch_embedding_dim": 512,
            "max_patch_size": 32,
            "device": self.device
        })
        
        test_text = "Memory leak detection test " * 50
        byte_seq = test_text.encode('utf-8')
        
        # Take baseline
        gc.collect()
        baseline = self.take_snapshot("Baseline")
        
        # Run many iterations
        num_iterations = 100
        for i in range(num_iterations):
            output = pipeline.process_bytes(byte_seq, mode='inference')
            del output
            
            if (i + 1) % 20 == 0:
                gc.collect()
                snapshot = self.take_snapshot(f"After {i+1} iterations")
                
                # Check for steady growth
                growth = snapshot.rss_mb - baseline.rss_mb
                growth_per_iter = growth / (i + 1)
                
                if growth_per_iter > 0.1:  # More than 0.1MB per iteration
                    print(f"WARNING: Memory growing at {growth_per_iter:.3f}MB/iteration")
        
        # Final check
        gc.collect()
        final = self.take_snapshot("Final")
        total_growth = final.rss_mb - baseline.rss_mb
        
        print(f"\nTotal memory growth: {total_growth:.2f}MB over {num_iterations} iterations")
        print(f"Average per iteration: {total_growth/num_iterations:.3f}MB")
        
        if total_growth > 10:
            print("FAIL: Significant memory leak detected!")
        else:
            print("PASS: No significant memory leak detected")
    
    def profile_gradient_memory(self):
        """Profile memory usage during gradient computation."""
        print("\n=== Profiling Gradient Memory ===")
        
        pipeline = BLTPipeline({
            "vocab_size": 256,
            "hidden_dim": 768,
            "num_layers": 6,
            "num_heads": 8,
            "patch_embedding_dim": 512,
            "max_patch_size": 32,
            "device": self.device
        })
        
        pipeline.train()
        test_text = "Gradient memory test " * 20
        byte_seq = test_text.encode('utf-8')
        
        # Without gradients
        self.take_snapshot("Before no-grad forward")
        with torch.no_grad():
            output = pipeline.process_bytes(byte_seq, mode='inference')
        self.take_snapshot("After no-grad forward")
        del output
        gc.collect()
        
        # With gradients
        self.take_snapshot("Before grad forward")
        output = pipeline.process_bytes(byte_seq, mode='training')
        self.take_snapshot("After grad forward")
        
        # Backward pass
        loss = output['loss']
        self.take_snapshot("Before backward")
        loss.backward()
        self.take_snapshot("After backward")
        
        # Clear gradients
        pipeline.zero_grad()
        gc.collect()
        self.take_snapshot("After clearing gradients")
    
    def generate_memory_report(self):
        """Generate a memory usage report."""
        print("\n=== Memory Usage Report ===")
        
        if not self.snapshots:
            print("No snapshots taken")
            return
        
        # Find peak memory usage
        peak_snapshot = max(self.snapshots, key=lambda s: s.rss_mb)
        print(f"Peak memory: {peak_snapshot}")
        
        # Calculate average
        avg_rss = np.mean([s.rss_mb for s in self.snapshots])
        print(f"Average RSS: {avg_rss:.2f}MB")
        
        # Memory growth
        if len(self.snapshots) > 1:
            growth = self.snapshots[-1].rss_mb - self.snapshots[0].rss_mb
            print(f"Total growth: {growth:.2f}MB")


class TestMemoryUsage:
    """Pytest test cases for memory usage."""
    
    @pytest.fixture
    def profiler(self):
        """Create memory profiler."""
        return MemoryProfiler()
    
    def test_pipeline_memory_limits(self, profiler):
        """Test that pipeline stays within memory limits."""
        config = {
            "vocab_size": 256,
            "hidden_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "patch_embedding_dim": 768,
            "max_patch_size": 32,
            "device": torch.device("cpu")
        }
        
        # Create pipeline
        before = profiler.take_snapshot()
        pipeline = BLTPipeline(config)
        after = profiler.take_snapshot()
        
        # Model should fit in reasonable memory
        model_size = after.rss_mb - before.rss_mb
        assert model_size < 500  # Less than 500MB for model
        
        # Process large text
        large_text = "a" * 10000
        before = profiler.take_snapshot()
        output = pipeline.process_bytes(large_text.encode('utf-8'), mode='inference')
        after = profiler.take_snapshot()
        
        # Processing should use reasonable memory
        processing_memory = after.rss_mb - before.rss_mb
        assert processing_memory < 100  # Less than 100MB for processing
    
    def test_no_memory_leaks(self, profiler):
        """Test for memory leaks in repeated processing."""
        pipeline = BLTPipeline({
            "vocab_size": 256,
            "hidden_dim": 512,
            "num_layers": 4,
            "num_heads": 8,
            "patch_embedding_dim": 512,
            "max_patch_size": 32,
            "device": torch.device("cpu")
        })
        
        test_text = "Memory leak test"
        byte_seq = test_text.encode('utf-8')
        
        # Baseline
        gc.collect()
        baseline = profiler.take_snapshot()
        
        # Run many times
        for _ in range(50):
            output = pipeline.process_bytes(byte_seq, mode='inference')
            del output
        
        # Check final memory
        gc.collect()
        final = profiler.take_snapshot()
        
        # Should not grow significantly
        growth = final.rss_mb - baseline.rss_mb
        assert growth < 5  # Less than 5MB growth
    
    def test_batch_memory_efficiency(self, profiler):
        """Test that batching is memory efficient."""
        pipeline = BLTPipeline({
            "vocab_size": 256,
            "hidden_dim": 512,
            "num_layers": 4,
            "num_heads": 8,
            "patch_embedding_dim": 512,
            "max_patch_size": 32,
            "device": torch.device("cpu")
        })
        
        text = "Batch memory test"
        
        # Single item processing
        gc.collect()
        before_single = profiler.take_snapshot()
        for _ in range(8):
            output = pipeline.process_bytes(text.encode('utf-8'), mode='inference')
            del output
        after_single = profiler.take_snapshot()
        single_memory = after_single.rss_mb - before_single.rss_mb
        
        # Batch processing
        gc.collect()
        before_batch = profiler.take_snapshot()
        byte_sequences = [text.encode('utf-8')] * 8
        outputs = pipeline.process_batch(byte_sequences, mode='inference')
        after_batch = profiler.take_snapshot()
        batch_memory = after_batch.rss_mb - before_batch.rss_mb
        
        # Batch should be more memory efficient
        assert batch_memory < single_memory * 1.5  # At most 50% more than single


def run_memory_profiling():
    """Run complete memory profiling suite."""
    profiler = MemoryProfiler()
    
    print("Running Memory Profiling Suite...")
    print("=" * 60)
    
    profiler.profile_pipeline_memory()
    profiler.profile_batch_memory_scaling()
    profiler.profile_cache_memory()
    profiler.detect_memory_leaks()
    profiler.profile_gradient_memory()
    profiler.generate_memory_report()
    
    print("\n" + "=" * 60)
    print("Memory profiling completed!")


if __name__ == "__main__":
    run_memory_profiling()