"""Memory profiling tools for Shvayambhu.

This module provides comprehensive memory profiling and optimization
recommendations for all components.
"""

import gc
import sys
import time
import tracemalloc
import psutil
import os
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
import threading
from contextlib import contextmanager


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""
    timestamp: float
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    available_mb: float
    percent: float
    gpu_mb: Optional[float] = None
    label: str = ""
    traceback: Optional[List[str]] = None
    
    def delta(self, other: 'MemorySnapshot') -> 'MemoryDelta':
        """Calculate delta between snapshots."""
        return MemoryDelta(
            time_delta=self.timestamp - other.timestamp,
            rss_delta_mb=self.rss_mb - other.rss_mb,
            vms_delta_mb=self.vms_mb - other.vms_mb,
            gpu_delta_mb=(self.gpu_mb - other.gpu_mb) if self.gpu_mb and other.gpu_mb else None
        )


@dataclass
class MemoryDelta:
    """Memory change between snapshots."""
    time_delta: float
    rss_delta_mb: float
    vms_delta_mb: float
    gpu_delta_mb: Optional[float] = None
    
    @property
    def rss_rate_mb_per_sec(self) -> float:
        return self.rss_delta_mb / max(self.time_delta, 0.001)


@dataclass
class MemoryProfile:
    """Complete memory profile for a component or operation."""
    component_name: str
    snapshots: List[MemorySnapshot] = field(default_factory=list)
    peak_rss_mb: float = 0.0
    peak_vms_mb: float = 0.0
    total_allocated_mb: float = 0.0
    allocation_count: int = 0
    gc_collections: Dict[int, int] = field(default_factory=dict)
    
    def add_snapshot(self, snapshot: MemorySnapshot):
        """Add a snapshot and update peaks."""
        self.snapshots.append(snapshot)
        self.peak_rss_mb = max(self.peak_rss_mb, snapshot.rss_mb)
        self.peak_vms_mb = max(self.peak_vms_mb, snapshot.vms_mb)


class MemoryProfiler:
    """Comprehensive memory profiler."""
    
    def __init__(self, enable_gpu: bool = True):
        self.process = psutil.Process(os.getpid())
        self.profiles: Dict[str, MemoryProfile] = {}
        self.enable_gpu = enable_gpu and torch.cuda.is_available()
        self.is_profiling = False
        self.baseline_snapshot: Optional[MemorySnapshot] = None
        
        # Component tracking
        self.component_stack: List[str] = []
        self.allocation_tracking: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        
        # Thresholds for warnings
        self.warning_thresholds = {
            'leak_rate_mb_per_sec': 1.0,
            'peak_memory_mb': 1000.0,
            'allocation_size_mb': 100.0
        }
    
    def take_snapshot(self, label: str = "") -> MemorySnapshot:
        """Take a memory snapshot."""
        gc.collect()  # Force garbage collection
        
        mem_info = self.process.memory_info()
        sys_mem = psutil.virtual_memory()
        
        # GPU memory if available
        gpu_mb = None
        if self.enable_gpu:
            gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
        
        # Get traceback if tracing
        traceback = None
        if tracemalloc.is_tracing():
            trace = tracemalloc.get_traced_memory()
            if trace[0] > 0:
                traceback = tracemalloc.get_traceback_limit()
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            rss_mb=mem_info.rss / 1024 / 1024,
            vms_mb=mem_info.vms / 1024 / 1024,
            available_mb=sys_mem.available / 1024 / 1024,
            percent=self.process.memory_percent(),
            gpu_mb=gpu_mb,
            label=label,
            traceback=traceback
        )
        
        return snapshot
    
    @contextmanager
    def profile_component(self, component_name: str):
        """Profile memory usage of a component."""
        # Start profiling
        start_snapshot = self.take_snapshot(f"{component_name}_start")
        
        # Initialize profile if needed
        if component_name not in self.profiles:
            self.profiles[component_name] = MemoryProfile(component_name)
        
        profile = self.profiles[component_name]
        profile.add_snapshot(start_snapshot)
        
        # Track component
        self.component_stack.append(component_name)
        
        # Record GC state
        gc_before = [gc.get_count(i) for i in range(3)]
        
        try:
            yield profile
        finally:
            # End profiling
            end_snapshot = self.take_snapshot(f"{component_name}_end")
            profile.add_snapshot(end_snapshot)
            
            # Record GC collections
            gc_after = [gc.get_count(i) for i in range(3)]
            for i in range(3):
                profile.gc_collections[i] = gc_after[i] - gc_before[i]
            
            # Check for issues
            delta = end_snapshot.delta(start_snapshot)
            self._check_memory_issues(component_name, delta)
            
            # Remove from stack
            self.component_stack.pop()
    
    def _check_memory_issues(self, component_name: str, delta: MemoryDelta):
        """Check for memory issues and log warnings."""
        # Check for memory leaks
        if delta.rss_rate_mb_per_sec > self.warning_thresholds['leak_rate_mb_per_sec']:
            print(f"WARNING: Potential memory leak in {component_name}: "
                  f"{delta.rss_rate_mb_per_sec:.2f} MB/sec")
        
        # Check peak memory
        profile = self.profiles[component_name]
        if profile.peak_rss_mb > self.warning_thresholds['peak_memory_mb']:
            print(f"WARNING: High peak memory in {component_name}: "
                  f"{profile.peak_rss_mb:.2f} MB")
    
    @contextmanager
    def trace_allocations(self):
        """Trace memory allocations."""
        tracemalloc.start()
        
        try:
            yield
        finally:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
            print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Tuple[Any, MemoryProfile]:
        """Profile a function's memory usage."""
        func_name = func.__name__
        
        with self.profile_component(func_name) as profile:
            result = func(*args, **kwargs)
        
        return result, profile
    
    def analyze_component(self, component_name: str) -> Dict[str, Any]:
        """Analyze memory usage for a component."""
        if component_name not in self.profiles:
            return {"error": f"No profile for component: {component_name}"}
        
        profile = self.profiles[component_name]
        
        if len(profile.snapshots) < 2:
            return {"error": "Insufficient snapshots for analysis"}
        
        # Calculate statistics
        rss_values = [s.rss_mb for s in profile.snapshots]
        
        # Memory growth
        start_rss = profile.snapshots[0].rss_mb
        end_rss = profile.snapshots[-1].rss_mb
        growth_mb = end_rss - start_rss
        
        # Time analysis
        duration = profile.snapshots[-1].timestamp - profile.snapshots[0].timestamp
        
        analysis = {
            "component": component_name,
            "duration_sec": duration,
            "memory_growth_mb": growth_mb,
            "growth_rate_mb_per_sec": growth_mb / max(duration, 0.001),
            "peak_rss_mb": profile.peak_rss_mb,
            "avg_rss_mb": np.mean(rss_values),
            "std_rss_mb": np.std(rss_values),
            "gc_collections": profile.gc_collections,
            "num_snapshots": len(profile.snapshots)
        }
        
        # Detect patterns
        analysis["patterns"] = self._detect_memory_patterns(profile)
        
        return analysis
    
    def _detect_memory_patterns(self, profile: MemoryProfile) -> List[str]:
        """Detect memory usage patterns."""
        patterns = []
        
        if len(profile.snapshots) < 3:
            return patterns
        
        rss_values = [s.rss_mb for s in profile.snapshots]
        
        # Check for monotonic growth (leak)
        if all(rss_values[i] <= rss_values[i+1] for i in range(len(rss_values)-1)):
            patterns.append("monotonic_growth")
        
        # Check for sawtooth (allocation/deallocation cycles)
        peaks = []
        for i in range(1, len(rss_values)-1):
            if rss_values[i] > rss_values[i-1] and rss_values[i] > rss_values[i+1]:
                peaks.append(i)
        
        if len(peaks) > 2:
            patterns.append("sawtooth")
        
        # Check for steady state
        if len(rss_values) > 10:
            recent = rss_values[-10:]
            if np.std(recent) < 0.1 * np.mean(recent):
                patterns.append("steady_state")
        
        return patterns
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory report."""
        report = {
            "timestamp": time.time(),
            "system_info": self._get_system_info(),
            "component_analyses": {},
            "recommendations": []
        }
        
        # Analyze each component
        for component_name in self.profiles:
            report["component_analyses"][component_name] = self.analyze_component(component_name)
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(report["component_analyses"])
        
        return report
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system memory information."""
        vm = psutil.virtual_memory()
        
        info = {
            "total_memory_mb": vm.total / 1024 / 1024,
            "available_memory_mb": vm.available / 1024 / 1024,
            "memory_percent": vm.percent,
            "python_version": sys.version,
            "process_memory_mb": self.process.memory_info().rss / 1024 / 1024
        }
        
        if self.enable_gpu:
            info["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            info["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        
        return info
    
    def _generate_recommendations(self, analyses: Dict[str, Dict]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        for component, analysis in analyses.items():
            if "error" in analysis:
                continue
            
            # High memory growth
            if analysis["growth_rate_mb_per_sec"] > 0.5:
                recommendations.append(
                    f"{component}: High memory growth rate. "
                    "Consider checking for memory leaks or implementing caching."
                )
            
            # High peak memory
            if analysis["peak_rss_mb"] > 500:
                recommendations.append(
                    f"{component}: High peak memory usage. "
                    "Consider batch processing or memory-efficient algorithms."
                )
            
            # Sawtooth pattern
            if "sawtooth" in analysis.get("patterns", []):
                recommendations.append(
                    f"{component}: Sawtooth memory pattern detected. "
                    "Consider object pooling or reusing allocations."
                )
            
            # Monotonic growth
            if "monotonic_growth" in analysis.get("patterns", []):
                recommendations.append(
                    f"{component}: Monotonic memory growth detected. "
                    "Likely memory leak - check for circular references."
                )
        
        return recommendations
    
    def plot_memory_usage(
        self,
        component_name: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """Plot memory usage over time."""
        if component_name:
            components = [component_name] if component_name in self.profiles else []
        else:
            components = list(self.profiles.keys())
        
        if not components:
            print("No components to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        for comp in components:
            profile = self.profiles[comp]
            if len(profile.snapshots) < 2:
                continue
            
            times = [(s.timestamp - profile.snapshots[0].timestamp) for s in profile.snapshots]
            rss_values = [s.rss_mb for s in profile.snapshots]
            
            ax1.plot(times, rss_values, label=f"{comp} RSS", marker='o')
            
            # Plot memory deltas
            if len(profile.snapshots) > 1:
                deltas = []
                delta_times = []
                for i in range(1, len(profile.snapshots)):
                    delta = profile.snapshots[i].delta(profile.snapshots[i-1])
                    deltas.append(delta.rss_delta_mb)
                    delta_times.append(times[i])
                
                ax2.bar(delta_times, deltas, width=0.1, label=f"{comp} Delta")
        
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Memory (MB)")
        ax1.set_title("Memory Usage Over Time")
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Memory Delta (MB)")
        ax2.set_title("Memory Changes")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def save_report(self, filepath: str):
        """Save memory report to file."""
        report = self.generate_report()
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def reset(self):
        """Reset profiler state."""
        self.profiles.clear()
        self.component_stack.clear()
        self.allocation_tracking.clear()


class ComponentMemoryTracker:
    """Track memory usage for specific components."""
    
    def __init__(self, profiler: MemoryProfiler):
        self.profiler = profiler
        self.component_memory: Dict[str, float] = {}
        self.lock = threading.Lock()
    
    def track_component(self, component_name: str, component_obj: Any):
        """Track memory usage of a component."""
        with self.lock:
            # Calculate object size
            size_mb = self._calculate_object_size(component_obj) / 1024 / 1024
            self.component_memory[component_name] = size_mb
    
    def _calculate_object_size(self, obj: Any) -> int:
        """Calculate approximate size of an object."""
        if isinstance(obj, torch.nn.Module):
            return sum(p.numel() * p.element_size() for p in obj.parameters())
        elif isinstance(obj, torch.Tensor):
            return obj.numel() * obj.element_size()
        elif isinstance(obj, np.ndarray):
            return obj.nbytes
        else:
            return sys.getsizeof(obj)
    
    def get_component_sizes(self) -> Dict[str, float]:
        """Get all component sizes."""
        with self.lock:
            return self.component_memory.copy()


def profile_memory_usage(func: Callable) -> Callable:
    """Decorator to profile function memory usage."""
    def wrapper(*args, **kwargs):
        profiler = MemoryProfiler()
        result, profile = profiler.profile_function(func, *args, **kwargs)
        
        # Print summary
        analysis = profiler.analyze_component(func.__name__)
        print(f"\nMemory Profile for {func.__name__}:")
        print(f"  Duration: {analysis['duration_sec']:.2f}s")
        print(f"  Memory growth: {analysis['memory_growth_mb']:.2f}MB")
        print(f"  Peak memory: {analysis['peak_rss_mb']:.2f}MB")
        
        return result
    
    return wrapper