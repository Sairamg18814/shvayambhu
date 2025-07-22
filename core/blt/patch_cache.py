"""Patch caching mechanism for BLT.

This module implements an efficient caching system for frequently used patches
to improve encoding/decoding performance.
"""

import hashlib
import pickle
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
import numpy as np
import torch
import threading
from pathlib import Path
import lmdb
import json


@dataclass
class CacheEntry:
    """Single cache entry."""
    patch_bytes: bytes
    embedding: torch.Tensor
    frequency: int = 1
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0
    content_type: str = "unknown"
    
    def __post_init__(self):
        self.size_bytes = len(self.patch_bytes) + self.embedding.element_size() * self.embedding.numel()


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_accesses: int = 0
    size_bytes: int = 0
    num_entries: int = 0
    
    @property
    def hit_rate(self) -> float:
        return self.hits / max(self.total_accesses, 1)
    
    @property
    def miss_rate(self) -> float:
        return self.misses / max(self.total_accesses, 1)
    
    def __repr__(self):
        return (
            f"CacheStats(hit_rate={self.hit_rate:.3f}, "
            f"entries={self.num_entries}, "
            f"size={self.size_bytes/1024/1024:.2f}MB)"
        )


class LRUCache:
    """Thread-safe LRU cache for patches."""
    
    def __init__(self, max_size_mb: int = 100):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()
        self.lock = threading.RLock()
    
    def _hash_patch(self, patch_bytes: bytes) -> str:
        """Create hash key for patch."""
        return hashlib.sha256(patch_bytes).hexdigest()[:16]
    
    def get(self, patch_bytes: bytes) -> Optional[torch.Tensor]:
        """Get embedding from cache."""
        key = self._hash_patch(patch_bytes)
        
        with self.lock:
            self.stats.total_accesses += 1
            
            if key in self.cache:
                # Move to end (most recently used)
                entry = self.cache.pop(key)
                entry.frequency += 1
                entry.last_access = time.time()
                self.cache[key] = entry
                
                self.stats.hits += 1
                return entry.embedding.clone()
            else:
                self.stats.misses += 1
                return None
    
    def put(
        self,
        patch_bytes: bytes,
        embedding: torch.Tensor,
        content_type: str = "unknown"
    ):
        """Put embedding in cache."""
        key = self._hash_patch(patch_bytes)
        
        with self.lock:
            # Check if already exists
            if key in self.cache:
                entry = self.cache.pop(key)
                entry.embedding = embedding
                entry.frequency += 1
                entry.last_access = time.time()
                self.cache[key] = entry
                return
            
            # Create new entry
            entry = CacheEntry(
                patch_bytes=patch_bytes,
                embedding=embedding,
                content_type=content_type
            )
            
            # Evict if necessary
            while (self.stats.size_bytes + entry.size_bytes > self.max_size_bytes and 
                   len(self.cache) > 0):
                self._evict_lru()
            
            # Add to cache
            self.cache[key] = entry
            self.stats.size_bytes += entry.size_bytes
            self.stats.num_entries = len(self.cache)
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self.cache:
            return
        
        # Remove oldest item (first in OrderedDict)
        key, entry = self.cache.popitem(last=False)
        self.stats.size_bytes -= entry.size_bytes
        self.stats.evictions += 1
        self.stats.num_entries = len(self.cache)
    
    def clear(self):
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.stats = CacheStats()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.stats


class FrequencyCache:
    """Frequency-based cache that keeps most frequent patches."""
    
    def __init__(self, max_size_mb: int = 100, min_frequency: int = 3):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.min_frequency = min_frequency
        self.cache: Dict[str, CacheEntry] = {}
        self.frequency_map: defaultdict[str, int] = defaultdict(int)
        self.stats = CacheStats()
        self.lock = threading.RLock()
    
    def _hash_patch(self, patch_bytes: bytes) -> str:
        """Create hash key for patch."""
        return hashlib.sha256(patch_bytes).hexdigest()[:16]
    
    def get(self, patch_bytes: bytes) -> Optional[torch.Tensor]:
        """Get embedding from cache."""
        key = self._hash_patch(patch_bytes)
        
        with self.lock:
            self.stats.total_accesses += 1
            self.frequency_map[key] += 1
            
            if key in self.cache:
                entry = self.cache[key]
                entry.frequency += 1
                entry.last_access = time.time()
                self.stats.hits += 1
                return entry.embedding.clone()
            else:
                self.stats.misses += 1
                
                # Check if should be cached
                if self.frequency_map[key] >= self.min_frequency:
                    return None  # Will be cached on next put()
                
                return None
    
    def put(
        self,
        patch_bytes: bytes,
        embedding: torch.Tensor,
        content_type: str = "unknown"
    ):
        """Put embedding in cache if frequency threshold met."""
        key = self._hash_patch(patch_bytes)
        
        with self.lock:
            # Only cache if frequency threshold met
            if self.frequency_map[key] < self.min_frequency:
                return
            
            # Check if already cached
            if key in self.cache:
                self.cache[key].embedding = embedding
                self.cache[key].frequency = self.frequency_map[key]
                return
            
            # Create new entry
            entry = CacheEntry(
                patch_bytes=patch_bytes,
                embedding=embedding,
                frequency=self.frequency_map[key],
                content_type=content_type
            )
            
            # Evict if necessary
            while (self.stats.size_bytes + entry.size_bytes > self.max_size_bytes and 
                   len(self.cache) > 0):
                self._evict_least_frequent()
            
            # Add to cache
            self.cache[key] = entry
            self.stats.size_bytes += entry.size_bytes
            self.stats.num_entries = len(self.cache)
    
    def _evict_least_frequent(self):
        """Evict least frequently used entry."""
        if not self.cache:
            return
        
        # Find entry with lowest frequency
        min_key = min(self.cache.keys(), key=lambda k: self.cache[k].frequency)
        entry = self.cache.pop(min_key)
        
        self.stats.size_bytes -= entry.size_bytes
        self.stats.evictions += 1
        self.stats.num_entries = len(self.cache)


class PersistentPatchCache:
    """Persistent patch cache using LMDB."""
    
    def __init__(
        self,
        cache_dir: str,
        max_size_gb: int = 1,
        readonly: bool = False
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.readonly = readonly
        
        # Open LMDB environment
        self.env = lmdb.open(
            str(self.cache_dir),
            map_size=self.max_size_bytes,
            readonly=readonly,
            lock=not readonly
        )
        
        # In-memory LRU for fast access
        self.memory_cache = LRUCache(max_size_mb=100)
        self.stats = CacheStats()
    
    def get(self, patch_bytes: bytes) -> Optional[torch.Tensor]:
        """Get embedding from cache."""
        # Check memory cache first
        embedding = self.memory_cache.get(patch_bytes)
        if embedding is not None:
            return embedding
        
        # Check persistent cache
        key = hashlib.sha256(patch_bytes).digest()
        
        with self.env.begin() as txn:
            value = txn.get(key)
            
        if value is not None:
            # Deserialize
            entry_dict = pickle.loads(value)
            embedding = torch.from_numpy(entry_dict['embedding'])
            
            # Add to memory cache
            self.memory_cache.put(patch_bytes, embedding)
            
            self.stats.hits += 1
            return embedding
        else:
            self.stats.misses += 1
            return None
    
    def put(
        self,
        patch_bytes: bytes,
        embedding: torch.Tensor,
        content_type: str = "unknown"
    ):
        """Put embedding in cache."""
        if self.readonly:
            return
        
        # Add to memory cache
        self.memory_cache.put(patch_bytes, embedding, content_type)
        
        # Persist to disk
        key = hashlib.sha256(patch_bytes).digest()
        entry_dict = {
            'embedding': embedding.cpu().numpy(),
            'content_type': content_type,
            'timestamp': time.time()
        }
        value = pickle.dumps(entry_dict)
        
        with self.env.begin(write=True) as txn:
            txn.put(key, value)
    
    def close(self):
        """Close cache."""
        self.env.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.env.begin() as txn:
            db_stats = txn.stat()
        
        return {
            'memory_cache': self.memory_cache.get_stats(),
            'persistent_entries': db_stats['entries'],
            'persistent_size_mb': db_stats['psize'] * db_stats['leaf_pages'] / 1024 / 1024
        }


class HierarchicalCache:
    """Hierarchical cache with multiple levels."""
    
    def __init__(
        self,
        l1_size_mb: int = 10,
        l2_size_mb: int = 100,
        persistent_dir: Optional[str] = None
    ):
        # L1: Small, fast frequency cache
        self.l1_cache = FrequencyCache(max_size_mb=l1_size_mb, min_frequency=5)
        
        # L2: Larger LRU cache
        self.l2_cache = LRUCache(max_size_mb=l2_size_mb)
        
        # L3: Optional persistent cache
        self.l3_cache = None
        if persistent_dir:
            self.l3_cache = PersistentPatchCache(persistent_dir)
        
        self.stats = {
            'l1_hits': 0,
            'l2_hits': 0,
            'l3_hits': 0,
            'misses': 0
        }
    
    def get(self, patch_bytes: bytes) -> Optional[torch.Tensor]:
        """Get embedding from cache hierarchy."""
        # Check L1
        embedding = self.l1_cache.get(patch_bytes)
        if embedding is not None:
            self.stats['l1_hits'] += 1
            return embedding
        
        # Check L2
        embedding = self.l2_cache.get(patch_bytes)
        if embedding is not None:
            self.stats['l2_hits'] += 1
            # Promote to L1 if frequent
            return embedding
        
        # Check L3
        if self.l3_cache:
            embedding = self.l3_cache.get(patch_bytes)
            if embedding is not None:
                self.stats['l3_hits'] += 1
                # Promote to L2
                self.l2_cache.put(patch_bytes, embedding)
                return embedding
        
        self.stats['misses'] += 1
        return None
    
    def put(
        self,
        patch_bytes: bytes,
        embedding: torch.Tensor,
        content_type: str = "unknown"
    ):
        """Put embedding in appropriate cache level."""
        # Always add to L2
        self.l2_cache.put(patch_bytes, embedding, content_type)
        
        # L1 will cache based on frequency
        self.l1_cache.put(patch_bytes, embedding, content_type)
        
        # Persist if available
        if self.l3_cache:
            self.l3_cache.put(patch_bytes, embedding, content_type)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get hierarchical cache statistics."""
        total = sum(self.stats.values())
        
        return {
            'l1_stats': self.l1_cache.get_stats(),
            'l2_stats': self.l2_cache.get_stats(),
            'l3_stats': self.l3_cache.get_stats() if self.l3_cache else None,
            'hierarchy_stats': {
                'l1_hit_rate': self.stats['l1_hits'] / max(total, 1),
                'l2_hit_rate': self.stats['l2_hits'] / max(total, 1),
                'l3_hit_rate': self.stats['l3_hits'] / max(total, 1),
                'miss_rate': self.stats['misses'] / max(total, 1)
            }
        }


class ContentAwareCache:
    """Cache that considers content type for optimization."""
    
    def __init__(self, max_size_mb: int = 100):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        
        # Separate caches for different content types
        self.caches = {
            'text': LRUCache(max_size_mb=max_size_mb // 4),
            'code': FrequencyCache(max_size_mb=max_size_mb // 4),
            'binary': LRUCache(max_size_mb=max_size_mb // 4),
            'mixed': LRUCache(max_size_mb=max_size_mb // 4)
        }
        
        self.content_detector = ContentTypeDetector()
    
    def get(
        self,
        patch_bytes: bytes,
        content_type: Optional[str] = None
    ) -> Optional[torch.Tensor]:
        """Get embedding from appropriate cache."""
        if content_type is None:
            content_type = self.content_detector.detect(patch_bytes)
        
        cache = self.caches.get(content_type, self.caches['mixed'])
        return cache.get(patch_bytes)
    
    def put(
        self,
        patch_bytes: bytes,
        embedding: torch.Tensor,
        content_type: Optional[str] = None
    ):
        """Put embedding in appropriate cache."""
        if content_type is None:
            content_type = self.content_detector.detect(patch_bytes)
        
        cache = self.caches.get(content_type, self.caches['mixed'])
        cache.put(patch_bytes, embedding, content_type)


class ContentTypeDetector:
    """Simple content type detector for patches."""
    
    def detect(self, patch_bytes: bytes) -> str:
        """Detect content type of patch."""
        try:
            text = patch_bytes.decode('utf-8')
            
            # Check for code patterns
            if any(pattern in text for pattern in ['def ', 'class ', 'function', '{}', '()']):
                return 'code'
            
            # Check for high entropy (binary-like)
            unique_bytes = len(set(patch_bytes))
            if unique_bytes / len(patch_bytes) > 0.8:
                return 'binary'
            
            return 'text'
            
        except UnicodeDecodeError:
            return 'binary'


def create_patch_cache(
    cache_type: str = "hierarchical",
    **kwargs
) -> Union[LRUCache, FrequencyCache, HierarchicalCache, ContentAwareCache]:
    """Factory function to create appropriate cache."""
    if cache_type == "lru":
        return LRUCache(**kwargs)
    elif cache_type == "frequency":
        return FrequencyCache(**kwargs)
    elif cache_type == "hierarchical":
        return HierarchicalCache(**kwargs)
    elif cache_type == "content_aware":
        return ContentAwareCache(**kwargs)
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")