"""Memory management utilities for Apple Silicon optimization."""

import os
import gc
import psutil
import torch
from contextlib import contextmanager
from typing import Dict, Any, Optional, NamedTuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class MemoryStats(NamedTuple):
    """Memory usage statistics."""
    total_memory: int
    available_memory: int
    used_memory: int
    percent_used: float
    allocated_tensors: int
    reserved_memory: int
    

@dataclass
class MemoryConfig:
    """Memory configuration for M4 Pro optimization."""
    max_memory_gb: int = 48
    warning_threshold: float = 0.8  # Warn at 80% usage
    critical_threshold: float = 0.9  # Critical at 90% usage
    cleanup_threshold: float = 0.85  # Trigger cleanup at 85%
    device: str = "mps"  # Metal Performance Shaders
    

class MemoryManager:
    """Manages memory allocation and optimization for Apple Silicon."""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self._peak_memory = 0
        self._allocation_history = []
        
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        vm = psutil.virtual_memory()
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            tensor_count = len([obj for obj in gc.get_objects() if torch.is_tensor(obj)])
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS doesn't have detailed memory tracking yet
            allocated = 0
            reserved = 0
            tensor_count = len([obj for obj in gc.get_objects() if torch.is_tensor(obj)])
        else:
            allocated = 0
            reserved = 0
            tensor_count = 0
            
        return MemoryStats(
            total_memory=vm.total,
            available_memory=vm.available,
            used_memory=vm.used,
            percent_used=vm.percent,
            allocated_tensors=tensor_count,
            reserved_memory=reserved
        )
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage and provide warnings."""
        stats = self.get_memory_stats()
        percent_used = stats.percent_used / 100.0
        
        status = "normal"
        if percent_used > self.config.critical_threshold:
            status = "critical"
            logger.critical(f"Memory usage critical: {stats.percent_used:.1f}%")
        elif percent_used > self.config.warning_threshold:
            status = "warning"
            logger.warning(f"Memory usage high: {stats.percent_used:.1f}%")
            
        return {
            "status": status,
            "percent_used": stats.percent_used,
            "available_gb": stats.available_memory / (1024**3),
            "total_gb": stats.total_memory / (1024**3),
            "tensor_count": stats.allocated_tensors
        }
    
    @contextmanager
    def memory_context(self, name: str = "operation"):
        """Context manager for tracking memory usage."""
        initial_stats = self.get_memory_stats()
        
        try:
            yield
        finally:
            final_stats = self.get_memory_stats()
            memory_delta = final_stats.used_memory - initial_stats.used_memory
            
            if memory_delta > 0:
                logger.info(f"{name} allocated {memory_delta / (1024**2):.1f} MB")
                
            # Update peak memory
            self._peak_memory = max(self._peak_memory, final_stats.used_memory)
            
            # Check if cleanup needed
            if final_stats.percent_used / 100.0 > self.config.cleanup_threshold:
                self.cleanup_memory()
    
    def cleanup_memory(self):
        """Perform memory cleanup."""
        logger.info("Performing memory cleanup...")
        
        # Clear Python garbage
        gc.collect()
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        # For MPS, we rely on automatic memory management
        # but can still trigger garbage collection
        
        stats_after = self.get_memory_stats()
        logger.info(f"Memory after cleanup: {stats_after.percent_used:.1f}%")
    
    def optimize_for_inference(self):
        """Optimize memory settings for inference."""
        # Disable gradient computation
        torch.set_grad_enabled(False)
        
        # Set memory allocator settings
        if hasattr(torch.backends, 'mps'):
            # MPS-specific optimizations
            os.environ['PYTORCH_MPS_AGGRESSIVE_CACHE_CLEANUP'] = '1'
            
        logger.info("Memory optimized for inference")
    
    def get_recommended_batch_size(self, model_size_gb: float) -> int:
        """Get recommended batch size based on available memory."""
        stats = self.get_memory_stats()
        available_gb = stats.available_memory / (1024**3)
        
        # Conservative estimate: use 70% of available memory
        usable_memory = available_gb * 0.7
        
        # Rough estimate: each sample needs ~2x model size in memory
        batch_size = max(1, int(usable_memory / (model_size_gb * 2)))
        
        return batch_size
    
    def monitor_memory_leaks(self) -> Dict[str, Any]:
        """Monitor for potential memory leaks."""
        current_stats = self.get_memory_stats()
        self._allocation_history.append({
            "timestamp": psutil.Process().create_time(),
            "memory_used": current_stats.used_memory,
            "tensor_count": current_stats.allocated_tensors
        })
        
        # Keep only last 100 measurements
        if len(self._allocation_history) > 100:
            self._allocation_history.pop(0)
            
        # Check for consistent memory growth
        if len(self._allocation_history) >= 10:
            recent = self._allocation_history[-10:]
            memory_trend = [h["memory_used"] for h in recent]
            
            # Simple leak detection: consistent growth
            is_growing = all(memory_trend[i] <= memory_trend[i+1] 
                           for i in range(len(memory_trend)-1))
            
            growth_rate = (memory_trend[-1] - memory_trend[0]) / len(memory_trend)
            
            return {
                "potential_leak": is_growing and growth_rate > 1024*1024,  # 1MB/measurement
                "growth_rate_mb": growth_rate / (1024*1024),
                "measurements": len(self._allocation_history)
            }
            
        return {"potential_leak": False, "measurements": len(self._allocation_history)}


# Global memory manager instance
_memory_manager = None


def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


def reset_memory_manager():
    """Reset global memory manager."""
    global _memory_manager
    _memory_manager = None
