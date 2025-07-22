"""Batched processing for BLT pipeline.

This module implements efficient batched processing for BLT components
to maximize throughput and minimize latency.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass, field
import threading
from queue import Queue, Empty
import time
from concurrent.futures import ThreadPoolExecutor, Future
import multiprocessing as mp
from collections import defaultdict

from .patching import DynamicPatcher, BLTInputProcessor
from .patch_optimizer import PatchOptimizer
from .patch_cache import HierarchicalCache
from ..utils.profiling.memory_profiler import MemoryProfiler


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_batch_size: int = 32
    max_sequence_length: int = 4096
    dynamic_batching: bool = True
    padding_strategy: str = "longest"  # longest, max_length, bucket
    bucket_sizes: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048, 4096])
    prefetch_factor: int = 2
    num_workers: int = 4
    timeout: float = 30.0
    enable_profiling: bool = False
    memory_limit_mb: int = 1000


@dataclass
class BatchStats:
    """Statistics for batch processing."""
    num_batches: int = 0
    total_sequences: int = 0
    total_bytes: int = 0
    processing_time: float = 0.0
    queue_time: float = 0.0
    padding_overhead: float = 0.0
    cache_hits: int = 0
    memory_peak_mb: float = 0.0
    
    @property
    def throughput_bytes_per_sec(self) -> float:
        return self.total_bytes / max(self.processing_time, 0.001)
    
    @property
    def throughput_sequences_per_sec(self) -> float:
        return self.total_sequences / max(self.processing_time, 0.001)
    
    @property
    def avg_batch_size(self) -> float:
        return self.total_sequences / max(self.num_batches, 1)


class BatchProcessor:
    """Efficient batched processing for BLT."""
    
    def __init__(
        self,
        config: Optional[BatchConfig] = None,
        patcher: Optional[DynamicPatcher] = None,
        optimizer: Optional[PatchOptimizer] = None,
        cache: Optional[HierarchicalCache] = None
    ):
        self.config = config or BatchConfig()
        self.patcher = patcher or DynamicPatcher()
        self.optimizer = optimizer or PatchOptimizer()
        self.cache = cache or HierarchicalCache()
        
        # Processing state
        self.input_queue = Queue(maxsize=self.config.max_batch_size * 2)
        self.output_queue = Queue()
        self.batch_queue = Queue(maxsize=self.config.prefetch_factor)
        
        # Worker threads
        self.workers = []
        self.executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
        self.running = False
        
        # Statistics
        self.stats = BatchStats()
        self.profiler = MemoryProfiler() if self.config.enable_profiling else None
        
        # Bucket allocator for dynamic batching
        self.bucket_allocator = BucketAllocator(self.config.bucket_sizes)
    
    def start(self):
        """Start batch processing workers."""
        self.running = True
        
        # Start batch formation thread
        batch_thread = threading.Thread(target=self._batch_formation_worker)
        batch_thread.start()
        self.workers.append(batch_thread)
        
        # Start processing workers
        for i in range(self.config.num_workers):
            worker = threading.Thread(target=self._processing_worker, args=(i,))
            worker.start()
            self.workers.append(worker)
    
    def stop(self):
        """Stop batch processing workers."""
        self.running = False
        
        # Send sentinel values
        for _ in range(len(self.workers)):
            self.input_queue.put(None)
            self.batch_queue.put(None)
        
        # Wait for workers
        for worker in self.workers:
            worker.join()
        
        self.executor.shutdown()
    
    def process_sequences(
        self,
        sequences: List[bytes],
        content_types: Optional[List[str]] = None,
        return_futures: bool = False
    ) -> Union[List[Dict[str, Any]], List[Future]]:
        """Process multiple sequences in batches."""
        if not self.running:
            self.start()
        
        futures = []
        
        # Submit sequences
        for i, seq in enumerate(sequences):
            content_type = content_types[i] if content_types else None
            future = self.executor.submit(
                self._submit_sequence, seq, content_type
            )
            futures.append(future)
        
        if return_futures:
            return futures
        else:
            # Wait for results
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=self.config.timeout)
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})
            return results
    
    def _submit_sequence(
        self,
        sequence: bytes,
        content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Submit a sequence for processing."""
        # Create request
        request = {
            "sequence": sequence,
            "content_type": content_type,
            "timestamp": time.time(),
            "result_future": Future()
        }
        
        # Add to queue
        self.input_queue.put(request)
        
        # Wait for result
        return request["result_future"].result(timeout=self.config.timeout)
    
    def _batch_formation_worker(self):
        """Worker thread for forming batches."""
        pending_requests = []
        
        while self.running:
            try:
                # Get new request
                request = self.input_queue.get(timeout=0.1)
                if request is None:
                    break
                
                pending_requests.append(request)
                
                # Check if should form batch
                if self._should_form_batch(pending_requests):
                    batch = self._form_batch(pending_requests)
                    self.batch_queue.put(batch)
                    pending_requests = []
                
            except Empty:
                # Timeout - check if should flush pending
                if pending_requests and self._should_flush(pending_requests):
                    batch = self._form_batch(pending_requests)
                    self.batch_queue.put(batch)
                    pending_requests = []
        
        # Flush remaining
        if pending_requests:
            batch = self._form_batch(pending_requests)
            self.batch_queue.put(batch)
    
    def _should_form_batch(self, requests: List[Dict]) -> bool:
        """Check if should form batch from pending requests."""
        if len(requests) >= self.config.max_batch_size:
            return True
        
        if self.config.dynamic_batching:
            # Check total size
            total_size = sum(len(r["sequence"]) for r in requests)
            if total_size > self.config.max_sequence_length:
                return True
        
        return False
    
    def _should_flush(self, requests: List[Dict]) -> bool:
        """Check if should flush pending requests."""
        if not requests:
            return False
        
        # Check oldest request age
        oldest_age = time.time() - requests[0]["timestamp"]
        return oldest_age > 0.1  # 100ms timeout
    
    def _form_batch(self, requests: List[Dict]) -> Dict[str, Any]:
        """Form a batch from requests."""
        batch_start = time.time()
        
        # Group by content type if available
        groups = defaultdict(list)
        for req in requests:
            content_type = req.get("content_type", "unknown")
            groups[content_type].append(req)
        
        # Create batch data
        batch_sequences = []
        batch_requests = []
        
        for content_type, group_requests in groups.items():
            # Sort by length for better padding
            if self.config.padding_strategy == "bucket":
                group_requests.sort(key=lambda r: len(r["sequence"]))
            
            for req in group_requests:
                batch_sequences.append(req["sequence"])
                batch_requests.append(req)
        
        # Allocate to buckets if using bucket strategy
        if self.config.padding_strategy == "bucket":
            buckets = self.bucket_allocator.allocate(batch_sequences)
            batch_data = []
            
            for bucket_size, bucket_sequences in buckets.items():
                if bucket_sequences:
                    padded = self._pad_sequences(bucket_sequences, bucket_size)
                    batch_data.append({
                        "sequences": padded,
                        "bucket_size": bucket_size,
                        "num_sequences": len(bucket_sequences)
                    })
        else:
            # Simple padding
            max_length = max(len(seq) for seq in batch_sequences)
            if self.config.padding_strategy == "max_length":
                max_length = min(max_length, self.config.max_sequence_length)
            
            padded = self._pad_sequences(batch_sequences, max_length)
            batch_data = [{
                "sequences": padded,
                "max_length": max_length,
                "num_sequences": len(batch_sequences)
            }]
        
        return {
            "batch_data": batch_data,
            "requests": batch_requests,
            "formation_time": time.time() - batch_start
        }
    
    def _pad_sequences(
        self,
        sequences: List[bytes],
        target_length: int
    ) -> np.ndarray:
        """Pad sequences to target length."""
        padded = np.zeros((len(sequences), target_length), dtype=np.uint8)
        
        for i, seq in enumerate(sequences):
            length = min(len(seq), target_length)
            padded[i, :length] = np.frombuffer(seq[:length], dtype=np.uint8)
        
        return padded
    
    def _processing_worker(self, worker_id: int):
        """Worker thread for processing batches."""
        while self.running:
            try:
                batch = self.batch_queue.get(timeout=0.1)
                if batch is None:
                    break
                
                # Process batch
                results = self._process_batch(batch, worker_id)
                
                # Return results
                for req, result in zip(batch["requests"], results):
                    req["result_future"].set_result(result)
                
            except Empty:
                continue
            except Exception as e:
                # Set error for all requests in batch
                if "requests" in batch:
                    for req in batch["requests"]:
                        req["result_future"].set_exception(e)
    
    def _process_batch(
        self,
        batch: Dict[str, Any],
        worker_id: int
    ) -> List[Dict[str, Any]]:
        """Process a batch of sequences."""
        process_start = time.time()
        results = []
        
        if self.profiler:
            with self.profiler.profile_component(f"batch_worker_{worker_id}"):
                results = self._process_batch_internal(batch)
        else:
            results = self._process_batch_internal(batch)
        
        # Update statistics
        self.stats.num_batches += 1
        self.stats.processing_time += time.time() - process_start
        
        return results
    
    def _process_batch_internal(
        self,
        batch: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Internal batch processing logic."""
        all_results = []
        
        for batch_data in batch["batch_data"]:
            sequences = batch_data["sequences"]
            
            # Process each sequence
            batch_results = []
            for i in range(batch_data["num_sequences"]):
                seq_array = sequences[i]
                
                # Find actual length (before padding)
                actual_length = np.where(seq_array == 0)[0]
                if len(actual_length) > 0:
                    actual_length = actual_length[0]
                else:
                    actual_length = len(seq_array)
                
                # Extract actual sequence
                byte_seq = seq_array[:actual_length].tobytes()
                
                # Check cache
                cached_result = self.cache.get(byte_seq)
                if cached_result is not None:
                    self.stats.cache_hits += 1
                    batch_results.append({
                        "patches": cached_result,
                        "cached": True
                    })
                    continue
                
                # Create patches
                if self.optimizer:
                    boundaries, opt_info = self.optimizer.optimize_patch_sizes(byte_seq)
                    patches = [byte_seq[s:e] for s, e in boundaries]
                else:
                    patches = self.patcher.create_patches(byte_seq)
                    boundaries = self.patcher.get_patch_boundaries()
                
                # Cache result
                self.cache.put(byte_seq, patches)
                
                batch_results.append({
                    "patches": patches,
                    "boundaries": boundaries,
                    "cached": False
                })
            
            all_results.extend(batch_results)
        
        # Update stats
        self.stats.total_sequences += len(all_results)
        self.stats.total_bytes += sum(
            sum(len(p) for p in r["patches"]) 
            for r in all_results
        )
        
        return all_results
    
    def get_stats(self) -> BatchStats:
        """Get batch processing statistics."""
        return self.stats
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = BatchStats()


class BucketAllocator:
    """Allocate sequences to buckets for efficient batching."""
    
    def __init__(self, bucket_sizes: List[int]):
        self.bucket_sizes = sorted(bucket_sizes)
    
    def allocate(
        self,
        sequences: List[bytes]
    ) -> Dict[int, List[bytes]]:
        """Allocate sequences to buckets."""
        buckets = {size: [] for size in self.bucket_sizes}
        
        for seq in sequences:
            # Find appropriate bucket
            seq_len = len(seq)
            bucket_size = None
            
            for size in self.bucket_sizes:
                if seq_len <= size:
                    bucket_size = size
                    break
            
            if bucket_size is None:
                # Use largest bucket
                bucket_size = self.bucket_sizes[-1]
            
            buckets[bucket_size].append(seq)
        
        return buckets


class StreamingBatchProcessor:
    """Streaming batch processor for continuous input."""
    
    def __init__(
        self,
        batch_processor: BatchProcessor,
        window_size: int = 1000,
        overlap: int = 100
    ):
        self.batch_processor = batch_processor
        self.window_size = window_size
        self.overlap = overlap
        self.buffer = bytearray()
        self.processed_offset = 0
    
    def process_stream(
        self,
        data: bytes,
        flush: bool = False
    ) -> List[Dict[str, Any]]:
        """Process streaming data."""
        self.buffer.extend(data)
        results = []
        
        # Process complete windows
        while len(self.buffer) >= self.window_size:
            window = bytes(self.buffer[:self.window_size])
            
            # Process window
            result = self.batch_processor.process_sequences([window])[0]
            result["offset"] = self.processed_offset
            results.append(result)
            
            # Slide window
            slide_amount = self.window_size - self.overlap
            self.buffer = self.buffer[slide_amount:]
            self.processed_offset += slide_amount
        
        # Process remaining if flush
        if flush and self.buffer:
            remaining = bytes(self.buffer)
            result = self.batch_processor.process_sequences([remaining])[0]
            result["offset"] = self.processed_offset
            results.append(result)
            
            self.buffer.clear()
            self.processed_offset += len(remaining)
        
        return results


def create_optimized_batch_processor(
    config: Dict[str, Any]
) -> BatchProcessor:
    """Create optimized batch processor with configuration."""
    batch_config = BatchConfig(
        max_batch_size=config.get("max_batch_size", 32),
        max_sequence_length=config.get("max_sequence_length", 4096),
        dynamic_batching=config.get("dynamic_batching", True),
        padding_strategy=config.get("padding_strategy", "bucket"),
        num_workers=config.get("num_workers", mp.cpu_count()),
        enable_profiling=config.get("enable_profiling", False)
    )
    
    # Create components
    patcher = DynamicPatcher(
        min_patch_size=config.get("min_patch_size", 4),
        max_patch_size=config.get("max_patch_size", 32)
    )
    
    optimizer = PatchOptimizer()
    
    cache = HierarchicalCache(
        l1_size_mb=config.get("l1_cache_mb", 10),
        l2_size_mb=config.get("l2_cache_mb", 100)
    )
    
    processor = BatchProcessor(
        config=batch_config,
        patcher=patcher,
        optimizer=optimizer,
        cache=cache
    )
    
    return processor