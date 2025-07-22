"""Advanced memory-augmented networks for external knowledge storage and retrieval.

This module provides comprehensive memory augmentation capabilities including external
memory stores, retrieval mechanisms, update protocols, and compression strategies
for enhanced knowledge persistence and accessibility.
"""

import os
import time
import json
import pickle
import sqlite3
import hashlib
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import tempfile
import gzip
import lzma

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory storage."""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    WORKING = "working"
    PROCEDURAL = "procedural"
    CACHE = "cache"


class RetrievalStrategy(Enum):
    """Memory retrieval strategies."""
    EXACT_MATCH = "exact_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    TEMPORAL_PROXIMITY = "temporal_proximity"
    FREQUENCY_BASED = "frequency_based"
    RELEVANCE_SCORING = "relevance_scoring"
    ADAPTIVE_HYBRID = "adaptive_hybrid"


class CompressionAlgorithm(Enum):
    """Memory compression algorithms."""
    NONE = "none"
    GZIP = "gzip"
    LZMA = "lzma"
    HUFFMAN = "huffman"
    ADAPTIVE = "adaptive"


@dataclass
class MemoryEntry:
    """Individual memory entry with metadata."""
    entry_id: str
    content: Any
    memory_type: MemoryType
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    importance_score: float = 0.5
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    compression: CompressionAlgorithm = CompressionAlgorithm.NONE
    compressed_size: int = 0
    
    def __post_init__(self):
        if not self.entry_id:
            content_str = str(self.content)
            self.entry_id = hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def calculate_age(self) -> float:
        """Calculate age of memory entry in seconds."""
        return time.time() - self.timestamp
    
    def calculate_recency_score(self) -> float:
        """Calculate recency score (0-1, higher = more recent)."""
        age = self.calculate_age()
        # Exponential decay with half-life of 1 day
        half_life = 86400  # 24 hours in seconds
        return 2 ** (-age / half_life)


@dataclass
class RetrievalResult:
    """Result of memory retrieval operation."""
    entries: List[MemoryEntry] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    total_results: int = 0
    retrieval_time: float = 0.0
    strategy_used: RetrievalStrategy = RetrievalStrategy.EXACT_MATCH
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryCompressor:
    """Handle memory compression and decompression."""
    
    def __init__(self):
        """Initialize memory compressor."""
        self.compression_stats = defaultdict(int)
        
    def compress(self, data: Any, algorithm: CompressionAlgorithm = CompressionAlgorithm.ADAPTIVE) -> Tuple[bytes, CompressionAlgorithm, int]:
        """Compress data using specified algorithm.
        
        Args:
            data: Data to compress
            algorithm: Compression algorithm to use
            
        Returns:
            Tuple of (compressed_data, algorithm_used, compression_ratio)
        """
        # Serialize data
        if isinstance(data, (str, bytes)):
            serialized = data.encode() if isinstance(data, str) else data
        else:
            serialized = pickle.dumps(data)
        
        original_size = len(serialized)
        
        if algorithm == CompressionAlgorithm.ADAPTIVE:
            algorithm = self._select_best_algorithm(serialized)
        
        if algorithm == CompressionAlgorithm.NONE:
            return serialized, algorithm, original_size
        elif algorithm == CompressionAlgorithm.GZIP:
            compressed = gzip.compress(serialized)
        elif algorithm == CompressionAlgorithm.LZMA:
            compressed = lzma.compress(serialized)
        else:
            # Fallback to gzip
            compressed = gzip.compress(serialized)
            algorithm = CompressionAlgorithm.GZIP
        
        compression_ratio = len(compressed) / original_size if original_size > 0 else 1.0
        self.compression_stats[algorithm.value] += 1
        
        return compressed, algorithm, int(compression_ratio * 100)
    
    def decompress(self, compressed_data: bytes, algorithm: CompressionAlgorithm) -> Any:
        """Decompress data using specified algorithm.
        
        Args:
            compressed_data: Compressed data
            algorithm: Algorithm used for compression
            
        Returns:
            Decompressed data
        """
        if algorithm == CompressionAlgorithm.NONE:
            decompressed = compressed_data
        elif algorithm == CompressionAlgorithm.GZIP:
            decompressed = gzip.decompress(compressed_data)
        elif algorithm == CompressionAlgorithm.LZMA:
            decompressed = lzma.decompress(compressed_data)
        else:
            raise ValueError(f"Unsupported compression algorithm: {algorithm}")
        
        # Try to deserialize if it's pickled data
        try:
            return pickle.loads(decompressed)
        except (pickle.UnpicklingError, UnicodeDecodeError):
            # Return as bytes if not pickled
            try:
                return decompressed.decode('utf-8')
            except UnicodeDecodeError:
                return decompressed
    
    def _select_best_algorithm(self, data: bytes) -> CompressionAlgorithm:
        """Select the best compression algorithm for given data."""
        # For small data, compression overhead isn't worth it
        if len(data) < 1024:
            return CompressionAlgorithm.NONE
        
        # For text-like data, LZMA often works better
        try:
            data.decode('utf-8')
            return CompressionAlgorithm.LZMA
        except UnicodeDecodeError:
            # For binary data, GZIP is usually faster
            return CompressionAlgorithm.GZIP


class MemoryIndex:
    """Indexing system for efficient memory retrieval."""
    
    def __init__(self):
        """Initialize memory index."""
        self.content_index = defaultdict(set)  # content hash -> entry_ids
        self.temporal_index = defaultdict(set)  # time bucket -> entry_ids
        self.type_index = defaultdict(set)  # memory_type -> entry_ids
        self.metadata_index = defaultdict(lambda: defaultdict(set))  # key -> value -> entry_ids
        self.embedding_index = {}  # entry_id -> embedding
        
    def add_entry(self, entry: MemoryEntry):
        """Add entry to all relevant indices."""
        # Content hash index
        content_hash = hashlib.sha256(str(entry.content).encode()).hexdigest()[:16]
        self.content_index[content_hash].add(entry.entry_id)
        
        # Temporal index (bucket by hour)
        time_bucket = int(entry.timestamp // 3600)  # Hour buckets
        self.temporal_index[time_bucket].add(entry.entry_id)
        
        # Type index
        self.type_index[entry.memory_type].add(entry.entry_id)
        
        # Metadata index
        for key, value in entry.metadata.items():
            self.metadata_index[key][str(value)].add(entry.entry_id)
        
        # Embedding index
        if entry.embedding is not None:
            self.embedding_index[entry.entry_id] = entry.embedding
    
    def remove_entry(self, entry: MemoryEntry):
        """Remove entry from all indices."""
        # Content hash index
        content_hash = hashlib.sha256(str(entry.content).encode()).hexdigest()[:16]
        self.content_index[content_hash].discard(entry.entry_id)
        
        # Temporal index
        time_bucket = int(entry.timestamp // 3600)
        self.temporal_index[time_bucket].discard(entry.entry_id)
        
        # Type index
        self.type_index[entry.memory_type].discard(entry.entry_id)
        
        # Metadata index
        for key, value in entry.metadata.items():
            self.metadata_index[key][str(value)].discard(entry.entry_id)
        
        # Embedding index
        self.embedding_index.pop(entry.entry_id, None)
    
    def find_by_content_hash(self, content: str) -> Set[str]:
        """Find entries by content hash."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return self.content_index.get(content_hash, set())
    
    def find_by_time_range(self, start_time: float, end_time: float) -> Set[str]:
        """Find entries within time range."""
        start_bucket = int(start_time // 3600)
        end_bucket = int(end_time // 3600)
        
        result_ids = set()
        for bucket in range(start_bucket, end_bucket + 1):
            result_ids.update(self.temporal_index.get(bucket, set()))
        
        return result_ids
    
    def find_by_type(self, memory_type: MemoryType) -> Set[str]:
        """Find entries by memory type."""
        return self.type_index.get(memory_type, set())
    
    def find_by_metadata(self, key: str, value: str) -> Set[str]:
        """Find entries by metadata key-value pair."""
        return self.metadata_index.get(key, {}).get(str(value), set())
    
    def find_similar_embeddings(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find entries with similar embeddings."""
        if not self.embedding_index:
            return []
        
        similarities = []
        for entry_id, embedding in self.embedding_index.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities.append((entry_id, similarity))
        
        # Sort by similarity (descending) and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return np.dot(a, b) / (norm_a * norm_b)


class ExternalMemoryStore:
    """Core external memory storage system."""
    
    def __init__(self, storage_path: str = None, max_memory_size: int = 1000000):
        """Initialize external memory store.
        
        Args:
            storage_path: Path to persistent storage directory
            max_memory_size: Maximum number of memory entries
        """
        self.storage_path = storage_path or tempfile.mkdtemp(prefix="memory_store_")
        self.max_memory_size = max_memory_size
        
        # Initialize storage
        os.makedirs(self.storage_path, exist_ok=True)
        self.db_path = os.path.join(self.storage_path, "memory.db")
        
        # Initialize components
        self.memory_index = MemoryIndex()
        self.compressor = MemoryCompressor()
        self.memory_cache = {}  # entry_id -> MemoryEntry (in-memory cache)
        self.access_queue = deque()  # LRU tracking
        
        # Threading
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Statistics
        self.stats = {
            'total_entries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'compressions': 0,
            'decompressions': 0,
            'retrievals': 0,
            'updates': 0
        }
        
        # Initialize database
        self._initialize_database()
        self._load_existing_entries()
    
    def _initialize_database(self):
        """Initialize SQLite database for persistent storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_entries (
                    entry_id TEXT PRIMARY KEY,
                    content BLOB,
                    memory_type TEXT,
                    timestamp REAL,
                    access_count INTEGER,
                    last_accessed REAL,
                    importance_score REAL,
                    embedding BLOB,
                    metadata TEXT,
                    compression_algorithm TEXT,
                    compressed_size INTEGER
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_entries(memory_type)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON memory_entries(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_importance ON memory_entries(importance_score)
            """)
            
            conn.commit()
    
    def _load_existing_entries(self):
        """Load existing entries from database into index."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT entry_id, memory_type, timestamp, embedding, metadata
                FROM memory_entries
            """)
            
            for row in cursor.fetchall():
                entry_id, memory_type, timestamp, embedding_blob, metadata_json = row
                
                # Create minimal entry for indexing
                entry = MemoryEntry(
                    entry_id=entry_id,
                    content="",  # Content loaded on demand
                    memory_type=MemoryType(memory_type),
                    timestamp=timestamp,
                    metadata=json.loads(metadata_json) if metadata_json else {}
                )
                
                # Deserialize embedding if present
                if embedding_blob:
                    entry.embedding = pickle.loads(embedding_blob)
                
                # Add to index
                self.memory_index.add_entry(entry)
                self.stats['total_entries'] += 1
    
    def store(self, content: Any, memory_type: MemoryType = MemoryType.SEMANTIC,
              importance_score: float = 0.5, metadata: Dict[str, Any] = None,
              embedding: np.ndarray = None) -> str:
        """Store content in external memory.
        
        Args:
            content: Content to store
            memory_type: Type of memory
            importance_score: Importance score (0-1)
            metadata: Additional metadata
            embedding: Optional vector embedding
            
        Returns:
            Entry ID of stored content
        """
        with self.lock:
            # Create memory entry
            entry = MemoryEntry(
                entry_id="",
                content=content,
                memory_type=memory_type,
                importance_score=importance_score,
                metadata=metadata or {},
                embedding=embedding
            )
            
            # Compress content for storage
            compressed_content, compression_algo, compression_ratio = self.compressor.compress(content)
            entry.compression = compression_algo
            entry.compressed_size = compression_ratio
            
            # Store in database
            self._store_entry_to_db(entry, compressed_content)
            
            # Add to index
            self.memory_index.add_entry(entry)
            
            # Add to cache (decompress content for cache)
            self.memory_cache[entry.entry_id] = entry
            self.access_queue.append(entry.entry_id)
            
            # Manage cache size
            self._manage_cache_size()
            
            self.stats['total_entries'] += 1
            self.stats['compressions'] += 1
            
            logger.info(f"Stored memory entry {entry.entry_id} with {compression_algo.value} compression")
            
            return entry.entry_id
    
    def _store_entry_to_db(self, entry: MemoryEntry, compressed_content: bytes):
        """Store entry to database."""
        with sqlite3.connect(self.db_path) as conn:
            # Serialize embedding
            embedding_blob = pickle.dumps(entry.embedding) if entry.embedding is not None else None
            
            conn.execute("""
                INSERT OR REPLACE INTO memory_entries 
                (entry_id, content, memory_type, timestamp, access_count, last_accessed,
                 importance_score, embedding, metadata, compression_algorithm, compressed_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.entry_id,
                compressed_content,
                entry.memory_type.value,
                entry.timestamp,
                entry.access_count,
                entry.last_accessed,
                entry.importance_score,
                embedding_blob,
                json.dumps(entry.metadata),
                entry.compression.value,
                entry.compressed_size
            ))
            conn.commit()
    
    def retrieve(self, query: Union[str, np.ndarray], strategy: RetrievalStrategy = RetrievalStrategy.ADAPTIVE_HYBRID,
                 memory_types: List[MemoryType] = None, limit: int = 10,
                 min_similarity: float = 0.1) -> RetrievalResult:
        """Retrieve memories based on query.
        
        Args:
            query: Query string or embedding vector
            strategy: Retrieval strategy to use
            memory_types: Filter by memory types
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            
        Returns:
            RetrievalResult with matching entries
        """
        start_time = time.time()
        
        with self.lock:
            # Determine strategy if adaptive
            if strategy == RetrievalStrategy.ADAPTIVE_HYBRID:
                strategy = self._select_retrieval_strategy(query)
            
            # Get candidate entry IDs based on strategy
            candidate_ids = self._get_candidate_ids(query, strategy, memory_types)
            
            # Load and score candidates
            scored_entries = []
            for entry_id in candidate_ids:
                entry = self._load_entry(entry_id)
                if entry:
                    score = self._calculate_similarity_score(query, entry, strategy)
                    if score >= min_similarity:
                        scored_entries.append((entry, score))
            
            # Sort by score and limit results
            scored_entries.sort(key=lambda x: x[1], reverse=True)
            scored_entries = scored_entries[:limit]
            
            # Extract entries and scores
            entries = [entry for entry, score in scored_entries]
            scores = [score for entry, score in scored_entries]
            
            # Update access statistics
            for entry in entries:
                entry.update_access()
                self._update_entry_access(entry)
            
            retrieval_time = time.time() - start_time
            self.stats['retrievals'] += 1
            
            result = RetrievalResult(
                entries=entries,
                scores=scores,
                total_results=len(entries),
                retrieval_time=retrieval_time,
                strategy_used=strategy,
                metadata={
                    'candidates_considered': len(candidate_ids),
                    'memory_types_filtered': [t.value for t in memory_types] if memory_types else None
                }
            )
            
            logger.info(f"Retrieved {len(entries)} entries using {strategy.value} in {retrieval_time:.3f}s")
            
            return result
    
    def _select_retrieval_strategy(self, query: Union[str, np.ndarray]) -> RetrievalStrategy:
        """Select optimal retrieval strategy based on query type and history."""
        if isinstance(query, np.ndarray):
            return RetrievalStrategy.SEMANTIC_SIMILARITY
        elif isinstance(query, str):
            # For exact matches, try exact match first
            if len(query) < 50 and query.replace(' ', '').isalnum():
                return RetrievalStrategy.EXACT_MATCH
            else:
                return RetrievalStrategy.RELEVANCE_SCORING
        else:
            return RetrievalStrategy.RELEVANCE_SCORING
    
    def _get_candidate_ids(self, query: Union[str, np.ndarray], strategy: RetrievalStrategy,
                          memory_types: List[MemoryType] = None) -> Set[str]:
        """Get candidate entry IDs based on retrieval strategy."""
        candidate_ids = set()
        
        if strategy == RetrievalStrategy.EXACT_MATCH and isinstance(query, str):
            candidate_ids = self.memory_index.find_by_content_hash(query)
        
        elif strategy == RetrievalStrategy.SEMANTIC_SIMILARITY and isinstance(query, np.ndarray):
            similar_entries = self.memory_index.find_similar_embeddings(query, top_k=100)
            candidate_ids = {entry_id for entry_id, similarity in similar_entries}
        
        elif strategy == RetrievalStrategy.TEMPORAL_PROXIMITY:
            # Get entries from the last 24 hours
            current_time = time.time()
            time_window = 86400  # 24 hours
            candidate_ids = self.memory_index.find_by_time_range(
                current_time - time_window, current_time
            )
        
        elif strategy == RetrievalStrategy.FREQUENCY_BASED:
            # Get all entries (will be sorted by access frequency)
            candidate_ids = set(self.memory_cache.keys())
            # Also get from database if cache is incomplete
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT entry_id FROM memory_entries ORDER BY access_count DESC LIMIT 1000")
                candidate_ids.update(row[0] for row in cursor.fetchall())
        
        else:  # RELEVANCE_SCORING or fallback
            # Combine multiple strategies
            if isinstance(query, str):
                # Content-based candidates
                candidate_ids.update(self.memory_index.find_by_content_hash(query))
                
                # Metadata-based candidates (if query contains recognizable patterns)
                words = query.lower().split()
                for word in words:
                    for key in self.memory_index.metadata_index:
                        for value in self.memory_index.metadata_index[key]:
                            if word in value.lower():
                                candidate_ids.update(self.memory_index.metadata_index[key][value])
            
            # Add temporal candidates
            current_time = time.time()
            candidate_ids.update(self.memory_index.find_by_time_range(
                current_time - 7200, current_time  # Last 2 hours
            ))
        
        # Filter by memory types if specified
        if memory_types:
            type_filtered_ids = set()
            for memory_type in memory_types:
                type_filtered_ids.update(self.memory_index.find_by_type(memory_type))
            candidate_ids &= type_filtered_ids
        
        return candidate_ids
    
    def _calculate_similarity_score(self, query: Union[str, np.ndarray], entry: MemoryEntry,
                                   strategy: RetrievalStrategy) -> float:
        """Calculate similarity score between query and entry."""
        if strategy == RetrievalStrategy.EXACT_MATCH:
            return 1.0 if str(query) == str(entry.content) else 0.0
        
        elif strategy == RetrievalStrategy.SEMANTIC_SIMILARITY and isinstance(query, np.ndarray):
            if entry.embedding is not None:
                return self.memory_index._cosine_similarity(query, entry.embedding)
            return 0.0
        
        elif strategy == RetrievalStrategy.TEMPORAL_PROXIMITY:
            return entry.calculate_recency_score()
        
        elif strategy == RetrievalStrategy.FREQUENCY_BASED:
            # Normalize access count (simple approach)
            max_access = max(1, max(e.access_count for e in self.memory_cache.values())) if self.memory_cache else 1
            return entry.access_count / max_access
        
        else:  # RELEVANCE_SCORING
            # Multi-factor scoring
            scores = []
            
            # Content similarity (simple word overlap)
            if isinstance(query, str):
                query_words = set(query.lower().split())
                content_words = set(str(entry.content).lower().split())
                if query_words and content_words:
                    jaccard = len(query_words & content_words) / len(query_words | content_words)
                    scores.append(jaccard * 0.4)
            
            # Recency score
            scores.append(entry.calculate_recency_score() * 0.2)
            
            # Importance score
            scores.append(entry.importance_score * 0.2)
            
            # Access frequency (normalized)
            if self.memory_cache:
                max_access = max(1, max(e.access_count for e in self.memory_cache.values()))
                scores.append((entry.access_count / max_access) * 0.2)
            
            return sum(scores) if scores else 0.0
    
    def _load_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Load entry from cache or database."""
        # Check cache first
        if entry_id in self.memory_cache:
            self.stats['cache_hits'] += 1
            return self.memory_cache[entry_id]
        
        # Load from database
        self.stats['cache_misses'] += 1
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT entry_id, content, memory_type, timestamp, access_count,
                       last_accessed, importance_score, embedding, metadata,
                       compression_algorithm, compressed_size
                FROM memory_entries WHERE entry_id = ?
            """, (entry_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Unpack row
            (entry_id, compressed_content, memory_type, timestamp, access_count,
             last_accessed, importance_score, embedding_blob, metadata_json,
             compression_algorithm, compressed_size) = row
            
            # Decompress content
            compression_algo = CompressionAlgorithm(compression_algorithm)
            content = self.compressor.decompress(compressed_content, compression_algo)
            self.stats['decompressions'] += 1
            
            # Deserialize embedding
            embedding = pickle.loads(embedding_blob) if embedding_blob else None
            
            # Create entry
            entry = MemoryEntry(
                entry_id=entry_id,
                content=content,
                memory_type=MemoryType(memory_type),
                timestamp=timestamp,
                access_count=access_count,
                last_accessed=last_accessed,
                importance_score=importance_score,
                embedding=embedding,
                metadata=json.loads(metadata_json) if metadata_json else {},
                compression=compression_algo,
                compressed_size=compressed_size
            )
            
            # Add to cache
            self.memory_cache[entry_id] = entry
            self.access_queue.append(entry_id)
            self._manage_cache_size()
            
            return entry
    
    def _update_entry_access(self, entry: MemoryEntry):
        """Update entry access statistics in database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE memory_entries 
                SET access_count = ?, last_accessed = ?
                WHERE entry_id = ?
            """, (entry.access_count, entry.last_accessed, entry.entry_id))
            conn.commit()
    
    def _manage_cache_size(self):
        """Manage in-memory cache size using LRU eviction."""
        cache_limit = min(10000, self.max_memory_size // 10)  # 10% of max size
        
        while len(self.memory_cache) > cache_limit and self.access_queue:
            # Remove least recently used entry
            lru_entry_id = self.access_queue.popleft()
            if lru_entry_id in self.memory_cache:
                del self.memory_cache[lru_entry_id]
    
    def update(self, entry_id: str, content: Any = None, importance_score: float = None,
               metadata: Dict[str, Any] = None, embedding: np.ndarray = None) -> bool:
        """Update existing memory entry.
        
        Args:
            entry_id: ID of entry to update
            content: New content (optional)
            importance_score: New importance score (optional)
            metadata: New metadata (optional)
            embedding: New embedding (optional)
            
        Returns:
            True if update successful, False otherwise
        """
        with self.lock:
            entry = self._load_entry(entry_id)
            if not entry:
                return False
            
            # Update fields
            if content is not None:
                entry.content = content
            if importance_score is not None:
                entry.importance_score = importance_score
            if metadata is not None:
                entry.metadata.update(metadata)
            if embedding is not None:
                entry.embedding = embedding
            
            # Recompress and store
            compressed_content, compression_algo, compression_ratio = self.compressor.compress(entry.content)
            entry.compression = compression_algo
            entry.compressed_size = compression_ratio
            
            self._store_entry_to_db(entry, compressed_content)
            
            # Update index
            self.memory_index.remove_entry(entry)
            self.memory_index.add_entry(entry)
            
            # Update cache
            self.memory_cache[entry_id] = entry
            
            self.stats['updates'] += 1
            
            logger.info(f"Updated memory entry {entry_id}")
            
            return True
    
    def delete(self, entry_id: str) -> bool:
        """Delete memory entry.
        
        Args:
            entry_id: ID of entry to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        with self.lock:
            entry = self._load_entry(entry_id)
            if not entry:
                return False
            
            # Remove from database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM memory_entries WHERE entry_id = ?", (entry_id,))
                conn.commit()
            
            # Remove from index
            self.memory_index.remove_entry(entry)
            
            # Remove from cache
            self.memory_cache.pop(entry_id, None)
            
            # Remove from access queue
            if entry_id in self.access_queue:
                temp_queue = deque()
                while self.access_queue:
                    item = self.access_queue.popleft()
                    if item != entry_id:
                        temp_queue.append(item)
                self.access_queue = temp_queue
            
            self.stats['total_entries'] -= 1
            
            logger.info(f"Deleted memory entry {entry_id}")
            
            return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        with self.lock:
            # Calculate cache statistics
            cache_hit_rate = (self.stats['cache_hits'] / 
                            max(1, self.stats['cache_hits'] + self.stats['cache_misses']))
            
            # Calculate storage statistics
            storage_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            return {
                'total_entries': self.stats['total_entries'],
                'cache_entries': len(self.memory_cache),
                'cache_hit_rate': cache_hit_rate,
                'total_retrievals': self.stats['retrievals'],
                'total_updates': self.stats['updates'],
                'compressions_performed': self.stats['compressions'],
                'decompressions_performed': self.stats['decompressions'],
                'storage_size_bytes': storage_size,
                'compression_stats': dict(self.compressor.compression_stats),
                'memory_types_distribution': self._get_memory_type_distribution(),
                'average_importance_score': self._get_average_importance_score()
            }
    
    def _get_memory_type_distribution(self) -> Dict[str, int]:
        """Get distribution of memory types."""
        distribution = {}
        for memory_type, entry_ids in self.memory_index.type_index.items():
            distribution[memory_type.value] = len(entry_ids)
        return distribution
    
    def _get_average_importance_score(self) -> float:
        """Calculate average importance score across all entries."""
        if not self.memory_cache:
            return 0.0
        
        total_score = sum(entry.importance_score for entry in self.memory_cache.values())
        return total_score / len(self.memory_cache)
    
    def cleanup_old_entries(self, max_age_days: int = 30, min_importance: float = 0.1):
        """Clean up old or low-importance entries.
        
        Args:
            max_age_days: Maximum age in days before entries are candidates for cleanup
            min_importance: Minimum importance score to keep entries
        """
        current_time = time.time()
        age_threshold = max_age_days * 86400  # Convert days to seconds
        
        entries_to_delete = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT entry_id, timestamp, importance_score, access_count
                FROM memory_entries
                WHERE timestamp < ? AND importance_score < ?
                ORDER BY access_count ASC, importance_score ASC
            """, (current_time - age_threshold, min_importance))
            
            for row in cursor.fetchall():
                entry_id, timestamp, importance_score, access_count = row
                
                # Additional criteria for deletion
                age_days = (current_time - timestamp) / 86400
                
                # Delete if old and rarely accessed
                if age_days > max_age_days and access_count < 5:
                    entries_to_delete.append(entry_id)
        
        # Delete entries
        deleted_count = 0
        for entry_id in entries_to_delete:
            if self.delete(entry_id):
                deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old memory entries")
        
        return deleted_count
    
    def close(self):
        """Close memory store and clean up resources."""
        with self.lock:
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            # Clear caches
            self.memory_cache.clear()
            self.access_queue.clear()
            
            logger.info("External memory store closed")


class MemoryAugmentedSystem:
    """High-level memory-augmented system orchestrating multiple memory stores."""
    
    def __init__(self, storage_base_path: str = None):
        """Initialize memory-augmented system.
        
        Args:
            storage_base_path: Base path for all memory stores
        """
        self.storage_base_path = storage_base_path or tempfile.mkdtemp(prefix="memory_system_")
        os.makedirs(self.storage_base_path, exist_ok=True)
        
        # Initialize memory stores for different types
        self.memory_stores = {
            MemoryType.EPISODIC: ExternalMemoryStore(
                os.path.join(self.storage_base_path, "episodic"), 100000
            ),
            MemoryType.SEMANTIC: ExternalMemoryStore(
                os.path.join(self.storage_base_path, "semantic"), 500000
            ),
            MemoryType.WORKING: ExternalMemoryStore(
                os.path.join(self.storage_base_path, "working"), 10000
            ),
            MemoryType.PROCEDURAL: ExternalMemoryStore(
                os.path.join(self.storage_base_path, "procedural"), 50000
            ),
            MemoryType.CACHE: ExternalMemoryStore(
                os.path.join(self.storage_base_path, "cache"), 20000
            )
        }
        
        self.global_stats = {
            'queries_processed': 0,
            'memories_stored': 0,
            'cross_store_retrievals': 0
        }
    
    def store_memory(self, content: Any, memory_type: MemoryType = MemoryType.SEMANTIC,
                     importance_score: float = 0.5, metadata: Dict[str, Any] = None,
                     embedding: np.ndarray = None) -> str:
        """Store memory in appropriate store.
        
        Args:
            content: Content to store
            memory_type: Type of memory
            importance_score: Importance score
            metadata: Additional metadata
            embedding: Vector embedding
            
        Returns:
            Entry ID
        """
        store = self.memory_stores[memory_type]
        entry_id = store.store(content, memory_type, importance_score, metadata, embedding)
        
        self.global_stats['memories_stored'] += 1
        
        return entry_id
    
    def retrieve_memories(self, query: Union[str, np.ndarray],
                         memory_types: List[MemoryType] = None,
                         strategy: RetrievalStrategy = RetrievalStrategy.ADAPTIVE_HYBRID,
                         limit: int = 10) -> RetrievalResult:
        """Retrieve memories across multiple stores.
        
        Args:
            query: Query string or embedding
            memory_types: Types of memory to search
            strategy: Retrieval strategy
            limit: Maximum results
            
        Returns:
            Combined retrieval results
        """
        if memory_types is None:
            memory_types = list(MemoryType)
        
        all_entries = []
        all_scores = []
        total_retrieval_time = 0.0
        
        # Retrieve from each specified memory store
        for memory_type in memory_types:
            if memory_type in self.memory_stores:
                store = self.memory_stores[memory_type]
                result = store.retrieve(query, strategy, [memory_type], limit)
                
                all_entries.extend(result.entries)
                all_scores.extend(result.scores)
                total_retrieval_time += result.retrieval_time
        
        # Sort combined results by score
        combined = list(zip(all_entries, all_scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        
        # Limit results
        combined = combined[:limit]
        final_entries = [entry for entry, score in combined]
        final_scores = [score for entry, score in combined]
        
        self.global_stats['queries_processed'] += 1
        self.global_stats['cross_store_retrievals'] += 1
        
        return RetrievalResult(
            entries=final_entries,
            scores=final_scores,
            total_results=len(final_entries),
            retrieval_time=total_retrieval_time,
            strategy_used=strategy,
            metadata={
                'stores_queried': [t.value for t in memory_types],
                'total_candidates': len(all_entries)
            }
        )
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        store_stats = {}
        total_entries = 0
        total_storage_size = 0
        
        for memory_type, store in self.memory_stores.items():
            stats = store.get_statistics()
            store_stats[memory_type.value] = stats
            total_entries += stats['total_entries']
            total_storage_size += stats['storage_size_bytes']
        
        return {
            'global_stats': self.global_stats,
            'total_entries_across_stores': total_entries,
            'total_storage_size_bytes': total_storage_size,
            'store_statistics': store_stats,
            'active_stores': len(self.memory_stores)
        }
    
    def cleanup_all_stores(self, max_age_days: int = 30, min_importance: float = 0.1):
        """Clean up all memory stores."""
        total_deleted = 0
        
        for memory_type, store in self.memory_stores.items():
            deleted = store.cleanup_old_entries(max_age_days, min_importance)
            total_deleted += deleted
            logger.info(f"Cleaned {deleted} entries from {memory_type.value} store")
        
        return total_deleted
    
    def close(self):
        """Close all memory stores."""
        for store in self.memory_stores.values():
            store.close()
        
        logger.info("Memory-augmented system closed")


# Convenience functions for easy usage
def create_memory_system(storage_path: str = None) -> MemoryAugmentedSystem:
    """Create a new memory-augmented system.
    
    Args:
        storage_path: Path for persistent storage
        
    Returns:
        MemoryAugmentedSystem instance
    """
    return MemoryAugmentedSystem(storage_path)


def store_knowledge(system: MemoryAugmentedSystem, content: str, 
                   importance: float = 0.5, metadata: Dict[str, Any] = None) -> str:
    """Store knowledge in semantic memory.
    
    Args:
        system: Memory system
        content: Knowledge content
        importance: Importance score
        metadata: Additional metadata
        
    Returns:
        Entry ID
    """
    return system.store_memory(
        content=content,
        memory_type=MemoryType.SEMANTIC,
        importance_score=importance,
        metadata=metadata
    )


def retrieve_knowledge(system: MemoryAugmentedSystem, query: str,
                      limit: int = 5) -> List[str]:
    """Retrieve knowledge from memory system.
    
    Args:
        system: Memory system
        query: Query string
        limit: Maximum results
        
    Returns:
        List of content strings
    """
    result = system.retrieve_memories(
        query=query,
        memory_types=[MemoryType.SEMANTIC, MemoryType.EPISODIC],
        limit=limit
    )
    
    return [str(entry.content) for entry in result.entries]


# Additional classes for transformer integration
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MemoryBank(nn.Module):
    """External memory bank for storing and retrieving information."""
    
    def __init__(
        self,
        memory_size: int = 1024,
        memory_dim: int = 768,
        num_heads: int = 8,
    ):
        super().__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        
        # Initialize memory bank
        self.memory = nn.Parameter(
            torch.randn(memory_size, memory_dim) * 0.02,
            requires_grad=True
        )
        
        # Memory addressing
        self.query_proj = nn.Linear(memory_dim, memory_dim)
        self.key_proj = nn.Linear(memory_dim, memory_dim)
        self.value_proj = nn.Linear(memory_dim, memory_dim)
        
    def forward(
        self,
        query: torch.Tensor,
        update: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Read from and optionally update memory."""
        batch_size, seq_len, _ = query.shape
        
        # Project query and memory
        q = self.query_proj(query)  # [B, L, D]
        k = self.key_proj(self.memory)  # [M, D]
        v = self.value_proj(self.memory)  # [M, D]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.memory_dim)
        attn_weights = F.softmax(scores, dim=-1)  # [B, L, M]
        
        # Read from memory
        memory_output = torch.matmul(attn_weights, v)  # [B, L, D]
        
        # Optional memory update (simplified)
        if update and self.training:
            # Use gradient-based updates during training
            pass
        
        return memory_output, {
            "attention_weights": attn_weights,
            "memory_usage": attn_weights.max(dim=-1)[0].mean().item()
        }


class MemoryAugmentedNetwork(nn.Module):
    """Network augmented with external memory for enhanced capabilities."""
    
    def __init__(
        self,
        hidden_dim: int = 768,
        memory_size: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_bank = MemoryBank(memory_size, hidden_dim, num_heads)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        use_memory: bool = True,
        update_memory: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Process hidden states with optional memory augmentation."""
        if not use_memory:
            return hidden_states, {}
        
        # Read from memory
        memory_output, memory_info = self.memory_bank(
            hidden_states,
            update=update_memory
        )
        
        # Gate and combine
        gate_values = self.gate(torch.cat([hidden_states, memory_output], dim=-1))
        combined = gate_values * hidden_states + (1 - gate_values) * memory_output
        
        # Project output
        output = self.output_proj(combined)
        output = self.dropout(output)
        
        return output, memory_info
    
    def reset_memory(self):
        """Reset memory to initial state."""
        nn.init.normal_(self.memory_bank.memory, mean=0, std=0.02)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        memory_norm = self.memory_bank.memory.norm(dim=-1)
        return {
            "memory_size": self.memory_bank.memory_size,
            "memory_dim": self.memory_bank.memory_dim,
            "avg_norm": memory_norm.mean().item(),
            "max_norm": memory_norm.max().item(),
            "min_norm": memory_norm.min().item(),
        }