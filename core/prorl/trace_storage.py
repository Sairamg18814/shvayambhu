"""Reasoning trace storage system for ProRL.

This module provides persistent storage, retrieval, and analysis of reasoning
traces with support for indexing, querying, and statistical analysis.
"""

import sqlite3
import json
import pickle
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import uuid
import hashlib
import threading
from contextlib import contextmanager
import warnings

from .reasoning_engine import ReasoningStep, ReasoningTrace, ReasoningStrategy, StepType
from .reward_model import ProcessReward, RewardType
from .backtracking import BacktrackDecision, BacktrackResult


@dataclass
class TraceMetadata:
    """Metadata for stored reasoning traces."""
    trace_id: str
    query_hash: str
    strategy: str
    total_steps: int
    success: bool
    confidence: float
    total_reward: float
    duration_seconds: float
    created_at: datetime
    file_path: Optional[str] = None
    compressed_size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    user_rating: Optional[int] = None
    notes: str = ""


@dataclass
class QueryPattern:
    """A pattern for querying stored traces."""
    strategy: Optional[ReasoningStrategy] = None
    min_confidence: float = 0.0
    max_confidence: float = 1.0
    min_steps: int = 0
    max_steps: int = 10000
    min_reward: float = -1.0
    max_reward: float = 1.0
    success_only: bool = False
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    query_keywords: List[str] = field(default_factory=list)


@dataclass
class StorageConfig:
    """Configuration for trace storage system."""
    storage_directory: str = "./reasoning_traces"
    database_name: str = "traces.db"
    compress_traces: bool = True
    max_traces_per_file: int = 100
    auto_cleanup_days: int = 30
    enable_indexing: bool = True
    cache_size_mb: int = 100
    backup_interval_hours: int = 24


class TraceStorageManager:
    """Manages persistent storage of reasoning traces."""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.storage_path = Path(config.storage_directory)
        self.db_path = self.storage_path / config.database_name
        self._lock = threading.Lock()
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # In-memory cache
        self._trace_cache: Dict[str, ReasoningTrace] = {}
        self._metadata_cache: Dict[str, TraceMetadata] = {}
        self._cache_size_bytes = 0
        self._max_cache_bytes = config.cache_size_mb * 1024 * 1024
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Traces metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    query_hash TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    total_steps INTEGER NOT NULL,
                    success BOOLEAN NOT NULL,
                    confidence REAL NOT NULL,
                    total_reward REAL NOT NULL,
                    duration_seconds REAL NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    file_path TEXT,
                    compressed_size_bytes INTEGER,
                    tags TEXT,
                    user_rating INTEGER,
                    notes TEXT
                )
            """)
            
            # Steps index table for detailed queries
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trace_steps (
                    trace_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    step_number INTEGER NOT NULL,
                    step_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reward_score REAL,
                    content_hash TEXT,
                    FOREIGN KEY (trace_id) REFERENCES traces (trace_id)
                )
            """)
            
            # Backtracking events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtrack_events (
                    trace_id TEXT NOT NULL,
                    target_step_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    improvement REAL,
                    timestamp TIMESTAMP NOT NULL,
                    FOREIGN KEY (trace_id) REFERENCES traces (trace_id)
                )
            """)
            
            # Create indexes for performance
            if self.config.enable_indexing:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_traces_strategy ON traces (strategy)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_traces_confidence ON traces (confidence)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_traces_created_at ON traces (created_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_traces_query_hash ON traces (query_hash)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_steps_trace_id ON trace_steps (trace_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_steps_type ON trace_steps (step_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_backtrack_trace_id ON backtrack_events (trace_id)")
            
            conn.commit()
    
    def store_trace(
        self,
        trace: ReasoningTrace,
        rewards: Optional[List[ProcessReward]] = None,
        backtrack_events: Optional[List[Tuple[BacktrackDecision, BacktrackResult]]] = None,
        tags: Optional[List[str]] = None,
        user_rating: Optional[int] = None,
        notes: str = ""
    ) -> bool:
        """Store a reasoning trace with metadata."""
        try:
            with self._lock:
                # Create metadata
                query_hash = self._hash_query(trace.query)
                duration = (trace.end_time or time.time()) - trace.start_time
                
                metadata = TraceMetadata(
                    trace_id=trace.trace_id,
                    query_hash=query_hash,
                    strategy=trace.strategy.value,
                    total_steps=trace.total_steps,
                    success=trace.confidence > 0.5,
                    confidence=trace.confidence,
                    total_reward=trace.total_reward,
                    duration_seconds=duration,
                    created_at=datetime.now(),
                    tags=tags or [],
                    user_rating=user_rating,
                    notes=notes
                )
                
                # Store trace data to file
                file_path = self._store_trace_file(trace, rewards, backtrack_events)
                metadata.file_path = str(file_path)
                metadata.compressed_size_bytes = file_path.stat().st_size
                
                # Store metadata in database
                self._store_metadata(metadata, trace)
                
                # Store step details
                self._store_step_details(trace, rewards)
                
                # Store backtracking events
                if backtrack_events:
                    self._store_backtrack_events(trace.trace_id, backtrack_events)
                
                # Update cache
                self._update_cache(trace.trace_id, trace, metadata)
                
                return True
                
        except Exception as e:
            warnings.warn(f"Failed to store trace {trace.trace_id}: {str(e)}")
            return False
    
    def retrieve_trace(self, trace_id: str) -> Optional[ReasoningTrace]:
        """Retrieve a reasoning trace by ID."""
        # Check cache first
        if trace_id in self._trace_cache:
            return self._trace_cache[trace_id]
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT file_path FROM traces WHERE trace_id = ?",
                    (trace_id,)
                )
                result = cursor.fetchone()
                
                if not result:
                    return None
                
                file_path = Path(result[0])
                if not file_path.exists():
                    warnings.warn(f"Trace file not found: {file_path}")
                    return None
                
                # Load trace from file
                trace = self._load_trace_file(file_path)
                
                # Update cache
                if trace:
                    self._trace_cache[trace_id] = trace
                    self._manage_cache_size()
                
                return trace
                
        except Exception as e:
            warnings.warn(f"Failed to retrieve trace {trace_id}: {str(e)}")
            return None
    
    def query_traces(self, pattern: QueryPattern, limit: int = 100) -> List[TraceMetadata]:
        """Query traces based on pattern."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build query
                query_parts = ["SELECT * FROM traces WHERE 1=1"]
                params = []
                
                if pattern.strategy:
                    query_parts.append("AND strategy = ?")
                    params.append(pattern.strategy.value)
                
                if pattern.min_confidence > 0:
                    query_parts.append("AND confidence >= ?")
                    params.append(pattern.min_confidence)
                
                if pattern.max_confidence < 1:
                    query_parts.append("AND confidence <= ?")
                    params.append(pattern.max_confidence)
                
                if pattern.min_steps > 0:
                    query_parts.append("AND total_steps >= ?")
                    params.append(pattern.min_steps)
                
                if pattern.max_steps < 10000:
                    query_parts.append("AND total_steps <= ?")
                    params.append(pattern.max_steps)
                
                if pattern.min_reward > -1:
                    query_parts.append("AND total_reward >= ?")
                    params.append(pattern.min_reward)
                
                if pattern.max_reward < 1:
                    query_parts.append("AND total_reward <= ?")
                    params.append(pattern.max_reward)
                
                if pattern.success_only:
                    query_parts.append("AND success = 1")
                
                if pattern.date_from:
                    query_parts.append("AND created_at >= ?")
                    params.append(pattern.date_from)
                
                if pattern.date_to:
                    query_parts.append("AND created_at <= ?")
                    params.append(pattern.date_to)
                
                if pattern.tags:
                    for tag in pattern.tags:
                        query_parts.append("AND tags LIKE ?")
                        params.append(f"%{tag}%")
                
                if pattern.query_keywords:
                    for keyword in pattern.query_keywords:
                        query_parts.append("AND query_text LIKE ?")
                        params.append(f"%{keyword}%")
                
                query_parts.append("ORDER BY created_at DESC")
                query_parts.append("LIMIT ?")
                params.append(limit)
                
                query = " ".join(query_parts)
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                # Convert to metadata objects
                metadata_list = []
                for row in results:
                    metadata = TraceMetadata(
                        trace_id=row[0],
                        query_hash=row[1],
                        strategy=row[3],
                        total_steps=row[4],
                        success=bool(row[5]),
                        confidence=row[6],
                        total_reward=row[7],
                        duration_seconds=row[8],
                        created_at=datetime.fromisoformat(row[9]),
                        file_path=row[10],
                        compressed_size_bytes=row[11] or 0,
                        tags=json.loads(row[12]) if row[12] else [],
                        user_rating=row[13],
                        notes=row[14] or ""
                    )
                    metadata_list.append(metadata)
                
                return metadata_list
                
        except Exception as e:
            warnings.warn(f"Failed to query traces: {str(e)}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Basic counts
                cursor.execute("SELECT COUNT(*) FROM traces")
                total_traces = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM traces WHERE success = 1")
                successful_traces = cursor.fetchone()[0]
                
                # Strategy distribution
                cursor.execute("""
                    SELECT strategy, COUNT(*) FROM traces 
                    GROUP BY strategy ORDER BY COUNT(*) DESC
                """)
                strategy_dist = dict(cursor.fetchall())
                
                # Confidence statistics
                cursor.execute("""
                    SELECT AVG(confidence), MIN(confidence), MAX(confidence) 
                    FROM traces
                """)
                confidence_stats = cursor.fetchone()
                
                # Reward statistics
                cursor.execute("""
                    SELECT AVG(total_reward), MIN(total_reward), MAX(total_reward) 
                    FROM traces
                """)
                reward_stats = cursor.fetchone()
                
                # Step statistics
                cursor.execute("""
                    SELECT AVG(total_steps), MIN(total_steps), MAX(total_steps) 
                    FROM traces
                """)
                step_stats = cursor.fetchone()
                
                # Duration statistics
                cursor.execute("""
                    SELECT AVG(duration_seconds), MIN(duration_seconds), MAX(duration_seconds) 
                    FROM traces
                """)
                duration_stats = cursor.fetchone()
                
                # Storage statistics
                cursor.execute("SELECT SUM(compressed_size_bytes) FROM traces")
                total_storage = cursor.fetchone()[0] or 0
                
                # Recent activity
                cursor.execute("""
                    SELECT COUNT(*) FROM traces 
                    WHERE created_at >= datetime('now', '-7 days')
                """)
                recent_traces = cursor.fetchone()[0]
                
                return {
                    "total_traces": total_traces,
                    "successful_traces": successful_traces,
                    "success_rate": successful_traces / total_traces if total_traces > 0 else 0,
                    "strategy_distribution": strategy_dist,
                    "confidence_stats": {
                        "mean": confidence_stats[0] or 0,
                        "min": confidence_stats[1] or 0,
                        "max": confidence_stats[2] or 0
                    },
                    "reward_stats": {
                        "mean": reward_stats[0] or 0,
                        "min": reward_stats[1] or 0,
                        "max": reward_stats[2] or 0
                    },
                    "step_stats": {
                        "mean": step_stats[0] or 0,
                        "min": step_stats[1] or 0,
                        "max": step_stats[2] or 0
                    },
                    "duration_stats": {
                        "mean": duration_stats[0] or 0,
                        "min": duration_stats[1] or 0,
                        "max": duration_stats[2] or 0
                    },
                    "storage_stats": {
                        "total_size_bytes": total_storage,
                        "total_size_mb": total_storage / (1024 * 1024),
                        "cache_size_bytes": self._cache_size_bytes,
                        "cache_entries": len(self._trace_cache)
                    },
                    "recent_activity": {
                        "traces_last_7_days": recent_traces
                    }
                }
                
        except Exception as e:
            warnings.warn(f"Failed to get statistics: {str(e)}")
            return {}
    
    def delete_trace(self, trace_id: str) -> bool:
        """Delete a trace and its associated files."""
        try:
            with self._lock:
                # Get file path before deletion
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT file_path FROM traces WHERE trace_id = ?",
                        (trace_id,)
                    )
                    result = cursor.fetchone()
                    
                    if result and result[0]:
                        file_path = Path(result[0])
                        if file_path.exists():
                            file_path.unlink()
                    
                    # Delete from database
                    cursor.execute("DELETE FROM traces WHERE trace_id = ?", (trace_id,))
                    cursor.execute("DELETE FROM trace_steps WHERE trace_id = ?", (trace_id,))
                    cursor.execute("DELETE FROM backtrack_events WHERE trace_id = ?", (trace_id,))
                    
                    conn.commit()
                
                # Remove from cache
                self._trace_cache.pop(trace_id, None)
                self._metadata_cache.pop(trace_id, None)
                
                return True
                
        except Exception as e:
            warnings.warn(f"Failed to delete trace {trace_id}: {str(e)}")
            return False
    
    def cleanup_old_traces(self, days: Optional[int] = None) -> int:
        """Clean up traces older than specified days."""
        days = days or self.config.auto_cleanup_days
        cutoff_date = datetime.now() - timedelta(days=days)
        
        try:
            with self._lock:
                # Find old traces
                pattern = QueryPattern(date_to=cutoff_date)
                old_traces = self.query_traces(pattern, limit=10000)
                
                deleted_count = 0
                for metadata in old_traces:
                    if self.delete_trace(metadata.trace_id):
                        deleted_count += 1
                
                return deleted_count
                
        except Exception as e:
            warnings.warn(f"Failed to cleanup old traces: {str(e)}")
            return 0
    
    def export_traces(
        self,
        pattern: QueryPattern,
        export_path: str,
        format: str = "json"
    ) -> bool:
        """Export traces matching pattern to file."""
        try:
            traces_metadata = self.query_traces(pattern, limit=10000)
            export_data = []
            
            for metadata in traces_metadata:
                trace = self.retrieve_trace(metadata.trace_id)
                if trace:
                    trace_data = {
                        "metadata": asdict(metadata),
                        "trace": {
                            "trace_id": trace.trace_id,
                            "query": trace.query,
                            "strategy": trace.strategy.value,
                            "start_time": trace.start_time,
                            "end_time": trace.end_time,
                            "final_answer": trace.final_answer,
                            "confidence": trace.confidence,
                            "total_reward": trace.total_reward,
                            "statistics": {
                                "total_steps": trace.total_steps,
                                "successful_steps": trace.successful_steps,
                                "backtrack_count": trace.backtrack_count,
                                "max_depth": trace.max_depth
                            },
                            "steps": [
                                {
                                    "step_id": step.step_id,
                                    "step_number": step.step_number,
                                    "step_type": step.step_type.value,
                                    "content": step.content,
                                    "confidence": step.confidence,
                                    "parent_step_id": step.parent_step_id
                                }
                                for step in trace.steps
                            ]
                        }
                    }
                    export_data.append(trace_data)
            
            # Write to file
            export_file = Path(export_path)
            
            if format.lower() == "json":
                with open(export_file, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            elif format.lower() == "pickle":
                with open(export_file, 'wb') as f:
                    pickle.dump(export_data, f)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to export traces: {str(e)}")
            return False
    
    def backup_database(self, backup_path: Optional[str] = None) -> bool:
        """Create a backup of the database."""
        try:
            if not backup_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.storage_path / f"backup_{timestamp}.db"
            
            backup_path = Path(backup_path)
            
            # Copy database file
            import shutil
            shutil.copy2(self.db_path, backup_path)
            
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to backup database: {str(e)}")
            return False
    
    def _hash_query(self, query: str) -> str:
        """Create hash of query for deduplication."""
        return hashlib.md5(query.encode()).hexdigest()
    
    def _store_trace_file(
        self,
        trace: ReasoningTrace,
        rewards: Optional[List[ProcessReward]],
        backtrack_events: Optional[List[Tuple[BacktrackDecision, BacktrackResult]]]
    ) -> Path:
        """Store trace data to compressed file."""
        # Create file path
        timestamp = datetime.now().strftime("%Y%m%d")
        file_name = f"{timestamp}_{trace.trace_id}.pkl.gz"
        file_path = self.storage_path / "traces" / file_name
        
        # Create directory if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        trace_data = {
            "trace": trace,
            "rewards": rewards,
            "backtrack_events": backtrack_events,
            "stored_at": datetime.now().isoformat()
        }
        
        # Store with compression
        if self.config.compress_traces:
            with gzip.open(file_path, 'wb') as f:
                pickle.dump(trace_data, f)
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(trace_data, f)
        
        return file_path
    
    def _load_trace_file(self, file_path: Path) -> Optional[ReasoningTrace]:
        """Load trace from file."""
        try:
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rb') as f:
                    trace_data = pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    trace_data = pickle.load(f)
            
            return trace_data.get("trace")
            
        except Exception as e:
            warnings.warn(f"Failed to load trace file {file_path}: {str(e)}")
            return None
    
    def _store_metadata(self, metadata: TraceMetadata, trace: ReasoningTrace):
        """Store trace metadata in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO traces (
                    trace_id, query_hash, query_text, strategy, total_steps,
                    success, confidence, total_reward, duration_seconds,
                    created_at, file_path, compressed_size_bytes, tags,
                    user_rating, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.trace_id,
                metadata.query_hash,
                trace.query,
                metadata.strategy,
                metadata.total_steps,
                metadata.success,
                metadata.confidence,
                metadata.total_reward,
                metadata.duration_seconds,
                metadata.created_at.isoformat(),
                metadata.file_path,
                metadata.compressed_size_bytes,
                json.dumps(metadata.tags),
                metadata.user_rating,
                metadata.notes
            ))
            conn.commit()
    
    def _store_step_details(self, trace: ReasoningTrace, rewards: Optional[List[ProcessReward]]):
        """Store detailed step information."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            reward_map = {}
            if rewards:
                reward_map = {r.step_id: r.total_reward for r in rewards}
            
            for step in trace.steps:
                content_hash = hashlib.md5(step.content.encode()).hexdigest()
                reward_score = reward_map.get(step.step_id, 0.0)
                
                cursor.execute("""
                    INSERT INTO trace_steps (
                        trace_id, step_id, step_number, step_type,
                        confidence, reward_score, content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    trace.trace_id,
                    step.step_id,
                    step.step_number,
                    step.step_type.value,
                    step.confidence,
                    reward_score,
                    content_hash
                ))
            
            conn.commit()
    
    def _store_backtrack_events(
        self,
        trace_id: str,
        backtrack_events: List[Tuple[BacktrackDecision, BacktrackResult]]
    ):
        """Store backtracking events."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for decision, result in backtrack_events:
                cursor.execute("""
                    INSERT INTO backtrack_events (
                        trace_id, target_step_id, reason, strategy,
                        success, improvement, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    trace_id,
                    decision.target_step_id,
                    decision.reason.value,
                    decision.strategy.value,
                    result.success,
                    result.improvement_achieved,
                    datetime.now().isoformat()
                ))
            
            conn.commit()
    
    def _update_cache(self, trace_id: str, trace: ReasoningTrace, metadata: TraceMetadata):
        """Update in-memory cache."""
        # Estimate size
        trace_size = len(pickle.dumps(trace))
        
        # Check if we need to make room
        if self._cache_size_bytes + trace_size > self._max_cache_bytes:
            self._evict_cache_entries()
        
        # Add to cache
        self._trace_cache[trace_id] = trace
        self._metadata_cache[trace_id] = metadata
        self._cache_size_bytes += trace_size
    
    def _manage_cache_size(self):
        """Manage cache size by evicting old entries."""
        if self._cache_size_bytes > self._max_cache_bytes:
            self._evict_cache_entries()
    
    def _evict_cache_entries(self):
        """Evict cache entries to make room."""
        # Simple LRU-style eviction (remove oldest entries)
        # In practice, would implement proper LRU
        if len(self._trace_cache) > 10:
            # Remove oldest entries
            traces_to_remove = list(self._trace_cache.keys())[:5]
            for trace_id in traces_to_remove:
                if trace_id in self._trace_cache:
                    trace_size = len(pickle.dumps(self._trace_cache[trace_id]))
                    del self._trace_cache[trace_id]
                    self._cache_size_bytes -= trace_size
                
                self._metadata_cache.pop(trace_id, None)
    
    @contextmanager
    def batch_operation(self):
        """Context manager for batch operations."""
        try:
            # Disable auto-commit for batch operations
            yield
        finally:
            pass
    
    def get_similar_traces(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.5
    ) -> List[Tuple[TraceMetadata, float]]:
        """Find traces with similar queries."""
        query_hash = self._hash_query(query)
        
        # Simple similarity based on query hash and keywords
        # In practice, would use embeddings for semantic similarity
        query_words = set(query.lower().split())
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM traces")
                results = cursor.fetchall()
                
                similar_traces = []
                
                for row in results:
                    trace_query = row[2]  # query_text
                    trace_words = set(trace_query.lower().split())
                    
                    # Calculate Jaccard similarity
                    intersection = query_words & trace_words
                    union = query_words | trace_words
                    similarity = len(intersection) / len(union) if union else 0
                    
                    if similarity >= min_similarity:
                        metadata = TraceMetadata(
                            trace_id=row[0],
                            query_hash=row[1],
                            strategy=row[3],
                            total_steps=row[4],
                            success=bool(row[5]),
                            confidence=row[6],
                            total_reward=row[7],
                            duration_seconds=row[8],
                            created_at=datetime.fromisoformat(row[9]),
                            file_path=row[10],
                            compressed_size_bytes=row[11] or 0,
                            tags=json.loads(row[12]) if row[12] else [],
                            user_rating=row[13],
                            notes=row[14] or ""
                        )
                        
                        similar_traces.append((metadata, similarity))
                
                # Sort by similarity and limit
                similar_traces.sort(key=lambda x: x[1], reverse=True)
                return similar_traces[:limit]
                
        except Exception as e:
            warnings.warn(f"Failed to find similar traces: {str(e)}")
            return []