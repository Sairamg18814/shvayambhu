"""LoRA checkpoint system for SEAL architecture.

This module provides comprehensive checkpointing capabilities for LoRA adapters,
including versioning, rollback, and efficient storage.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import time
import json
import pickle
import hashlib
import shutil
import sqlite3
from collections import defaultdict, deque
import numpy as np
import warnings

from .lora_adapter import LoRALinear, LoRAEmbedding, LoRAConfig
from .memory_efficient_lora import MemoryEfficientLoRA


@dataclass
class CheckpointMetadata:
    """Metadata for a LoRA checkpoint."""
    checkpoint_id: str
    timestamp: float
    description: str
    
    # Model state
    model_hash: str
    adapter_count: int
    total_parameters: int
    
    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    validation_scores: Dict[str, float] = field(default_factory=dict)
    
    # Training context
    training_step: int = 0
    epoch: int = 0
    learning_rate: float = 0.0
    
    # Checkpoint details
    file_size_mb: float = 0.0
    compression_ratio: float = 1.0
    storage_format: str = "torch"  # "torch", "safetensors", "onnx"
    
    # Dependencies
    parent_checkpoint: Optional[str] = None
    base_model_version: str = ""
    
    # Tags and annotations
    tags: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management."""
    # Storage settings
    checkpoint_dir: Path = Path("checkpoints/lora")
    max_checkpoints: int = 100
    auto_cleanup: bool = True
    
    # Compression settings
    enable_compression: bool = True
    compression_level: int = 6  # 1-9, higher = better compression
    use_delta_compression: bool = True  # Store only differences
    
    # Storage formats
    default_format: str = "torch"  # "torch", "safetensors", "pickle"
    enable_safetensors: bool = True
    
    # Validation settings
    validate_on_save: bool = True
    validate_on_load: bool = True
    checksum_verification: bool = True
    
    # Auto-checkpoint settings
    auto_checkpoint_interval: int = 1000  # steps
    auto_checkpoint_on_improvement: bool = True
    performance_threshold: float = 0.01  # Minimum improvement for auto-checkpoint
    
    # Backup settings
    enable_remote_backup: bool = False
    backup_location: Optional[str] = None
    backup_interval_hours: int = 24


class LoRACheckpointManager:
    """Manages checkpoints for LoRA adapters with versioning and rollback."""
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.checkpoint_dir = config.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Database for metadata
        self.db_path = self.checkpoint_dir / "metadata.db"
        self._init_database()
        
        # In-memory state
        self.checkpoint_cache = {}
        self.performance_history = deque(maxlen=1000)
        self.last_auto_checkpoint = 0
        
        # Delta compression state
        self.baseline_states = {}
        
    def _init_database(self):
        """Initialize SQLite database for checkpoint metadata."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    description TEXT,
                    model_hash TEXT,
                    adapter_count INTEGER,
                    total_parameters INTEGER,
                    training_step INTEGER,
                    epoch INTEGER,
                    learning_rate REAL,
                    file_size_mb REAL,
                    compression_ratio REAL,
                    storage_format TEXT,
                    parent_checkpoint TEXT,
                    base_model_version TEXT,
                    tags TEXT,
                    notes TEXT,
                    metadata_json TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    checkpoint_id TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    FOREIGN KEY (checkpoint_id) REFERENCES checkpoints (checkpoint_id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS validation_scores (
                    checkpoint_id TEXT,
                    score_name TEXT,
                    score_value REAL,
                    FOREIGN KEY (checkpoint_id) REFERENCES checkpoints (checkpoint_id)
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON checkpoints(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_training_step ON checkpoints(training_step)
            """)
            
            conn.commit()
    
    def save_checkpoint(
        self,
        adapters: Dict[str, Union[LoRALinear, LoRAEmbedding, MemoryEfficientLoRA]],
        metadata: CheckpointMetadata,
        model_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save a checkpoint of LoRA adapters."""
        # Generate checkpoint ID if not provided
        if not metadata.checkpoint_id:
            metadata.checkpoint_id = self._generate_checkpoint_id()
        
        checkpoint_path = self.checkpoint_dir / f"{metadata.checkpoint_id}.pt"
        
        # Prepare checkpoint data
        checkpoint_data = {
            "metadata": asdict(metadata),
            "adapters": {},
            "model_state": model_state or {},
            "config": self.config.__dict__,
            "timestamp": time.time()
        }
        
        # Save adapter states
        total_params = 0
        for name, adapter in adapters.items():
            if hasattr(adapter, 'state_dict'):
                adapter_state = adapter.state_dict()
            else:
                # Handle custom adapter types
                adapter_state = self._extract_adapter_state(adapter)
            
            checkpoint_data["adapters"][name] = adapter_state
            total_params += sum(p.numel() for p in adapter_state.values() if isinstance(p, torch.Tensor))
        
        metadata.total_parameters = total_params
        
        # Apply compression if enabled
        if self.config.enable_compression:
            checkpoint_data = self._compress_checkpoint(checkpoint_data, metadata.checkpoint_id)
        
        # Save to disk
        if self.config.default_format == "torch":
            torch.save(checkpoint_data, checkpoint_path)
        elif self.config.default_format == "safetensors" and self.config.enable_safetensors:
            self._save_as_safetensors(checkpoint_data, checkpoint_path)
        else:
            with open(checkpoint_path.with_suffix(".pkl"), "wb") as f:
                pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Calculate file size
        metadata.file_size_mb = checkpoint_path.stat().st_size / 1024 / 1024
        
        # Validate checkpoint if enabled
        if self.config.validate_on_save:
            if not self._validate_checkpoint(checkpoint_path):
                warnings.warn(f"Checkpoint validation failed for {metadata.checkpoint_id}")
        
        # Save metadata to database
        self._save_metadata_to_db(metadata)
        
        # Auto-cleanup old checkpoints
        if self.config.auto_cleanup:
            self._cleanup_old_checkpoints()
        
        print(f"Saved checkpoint: {metadata.checkpoint_id} ({metadata.file_size_mb:.2f} MB)")
        return metadata.checkpoint_id
    
    def load_checkpoint(
        self,
        checkpoint_id: str,
        device: Optional[torch.device] = None
    ) -> Tuple[Dict[str, Any], CheckpointMetadata]:
        """Load a checkpoint by ID."""
        # Try different file extensions
        possible_paths = [
            self.checkpoint_dir / f"{checkpoint_id}.pt",
            self.checkpoint_dir / f"{checkpoint_id}.pkl",
            self.checkpoint_dir / f"{checkpoint_id}.safetensors"
        ]
        
        checkpoint_path = None
        for path in possible_paths:
            if path.exists():
                checkpoint_path = path
                break
        
        if checkpoint_path is None:
            raise FileNotFoundError(f"Checkpoint {checkpoint_id} not found")
        
        # Load checkpoint data
        if checkpoint_path.suffix == ".pt":
            checkpoint_data = torch.load(checkpoint_path, map_location=device)
        elif checkpoint_path.suffix == ".safetensors":
            checkpoint_data = self._load_from_safetensors(checkpoint_path)
        else:
            with open(checkpoint_path, "rb") as f:
                checkpoint_data = pickle.load(f)
        
        # Validate checkpoint if enabled
        if self.config.validate_on_load:
            if not self._validate_checkpoint_data(checkpoint_data):
                raise RuntimeError(f"Checkpoint validation failed for {checkpoint_id}")
        
        # Decompress if needed
        if self.config.enable_compression and "compressed" in checkpoint_data:
            checkpoint_data = self._decompress_checkpoint(checkpoint_data)
        
        # Extract metadata
        metadata = CheckpointMetadata(**checkpoint_data["metadata"])
        
        return checkpoint_data, metadata
    
    def restore_adapters(
        self,
        checkpoint_id: str,
        adapter_classes: Dict[str, type],
        device: Optional[torch.device] = None
    ) -> Dict[str, Union[LoRALinear, LoRAEmbedding]]:
        """Restore LoRA adapters from checkpoint."""
        checkpoint_data, metadata = self.load_checkpoint(checkpoint_id, device)
        
        restored_adapters = {}
        for name, adapter_state in checkpoint_data["adapters"].items():
            if name not in adapter_classes:
                warnings.warn(f"No adapter class provided for {name}, skipping")
                continue
            
            # Create adapter instance
            adapter_class = adapter_classes[name]
            adapter = self._create_adapter_from_state(adapter_class, adapter_state)
            
            if device:
                adapter = adapter.to(device)
            
            restored_adapters[name] = adapter
        
        return restored_adapters
    
    def list_checkpoints(
        self,
        limit: Optional[int] = None,
        tags: Optional[List[str]] = None,
        min_timestamp: Optional[float] = None,
        max_timestamp: Optional[float] = None
    ) -> List[CheckpointMetadata]:
        """List available checkpoints with optional filtering."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM checkpoints WHERE 1=1"
            params = []
            
            if min_timestamp:
                query += " AND timestamp >= ?"
                params.append(min_timestamp)
            
            if max_timestamp:
                query += " AND timestamp <= ?"
                params.append(max_timestamp)
            
            if tags:
                # Simple tag filtering (could be improved with proper tag indexing)
                tag_conditions = []
                for tag in tags:
                    tag_conditions.append("tags LIKE ?")
                    params.append(f"%{tag}%")
                
                if tag_conditions:
                    query += " AND (" + " OR ".join(tag_conditions) + ")"
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            checkpoints = []
            for row in rows:
                # Convert row to dictionary
                checkpoint_dict = dict(zip([col[0] for col in cursor.description], row))
                
                # Parse JSON fields
                if checkpoint_dict['tags']:
                    checkpoint_dict['tags'] = json.loads(checkpoint_dict['tags'])
                else:
                    checkpoint_dict['tags'] = []
                
                if checkpoint_dict['metadata_json']:
                    extra_metadata = json.loads(checkpoint_dict['metadata_json'])
                    checkpoint_dict.update(extra_metadata)
                
                # Remove non-CheckpointMetadata fields
                for key in ['metadata_json']:
                    checkpoint_dict.pop(key, None)
                
                checkpoints.append(CheckpointMetadata(**checkpoint_dict))
            
            return checkpoints
    
    def delete_checkpoint(self, checkpoint_id: str):
        """Delete a checkpoint and its metadata."""
        # Remove files
        possible_paths = [
            self.checkpoint_dir / f"{checkpoint_id}.pt",
            self.checkpoint_dir / f"{checkpoint_id}.pkl",
            self.checkpoint_dir / f"{checkpoint_id}.safetensors"
        ]
        
        for path in possible_paths:
            if path.exists():
                path.unlink()
        
        # Remove from database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,))
            cursor.execute("DELETE FROM performance_metrics WHERE checkpoint_id = ?", (checkpoint_id,))
            cursor.execute("DELETE FROM validation_scores WHERE checkpoint_id = ?", (checkpoint_id,))
            conn.commit()
        
        print(f"Deleted checkpoint: {checkpoint_id}")
    
    def rollback_to_checkpoint(
        self,
        checkpoint_id: str,
        adapter_targets: Dict[str, Union[LoRALinear, LoRAEmbedding]]
    ) -> bool:
        """Rollback adapters to a previous checkpoint state."""
        try:
            # Load checkpoint
            checkpoint_data, metadata = self.load_checkpoint(checkpoint_id)
            
            # Restore adapter states
            for name, adapter in adapter_targets.items():
                if name in checkpoint_data["adapters"]:
                    adapter_state = checkpoint_data["adapters"][name]
                    adapter.load_state_dict(adapter_state)
                else:
                    warnings.warn(f"Adapter {name} not found in checkpoint {checkpoint_id}")
            
            print(f"Successfully rolled back to checkpoint: {checkpoint_id}")
            return True
            
        except Exception as e:
            warnings.warn(f"Rollback failed: {str(e)}")
            return False
    
    def create_checkpoint_branch(
        self,
        base_checkpoint_id: str,
        branch_name: str,
        adapters: Dict[str, Union[LoRALinear, LoRAEmbedding]]
    ) -> str:
        """Create a new checkpoint branch from an existing checkpoint."""
        # Load base checkpoint
        base_data, base_metadata = self.load_checkpoint(base_checkpoint_id)
        
        # Create new metadata
        new_metadata = CheckpointMetadata(
            checkpoint_id=f"{base_checkpoint_id}_branch_{branch_name}",
            timestamp=time.time(),
            description=f"Branch '{branch_name}' from {base_checkpoint_id}",
            model_hash=self._compute_model_hash(adapters),
            adapter_count=len(adapters),
            total_parameters=0,
            parent_checkpoint=base_checkpoint_id,
            tags=["branch", branch_name]
        )
        
        # Save new checkpoint
        return self.save_checkpoint(adapters, new_metadata)
    
    def merge_checkpoints(
        self,
        checkpoint_ids: List[str],
        merge_strategy: str = "average",
        weights: Optional[List[float]] = None
    ) -> str:
        """Merge multiple checkpoints into a new checkpoint."""
        if len(checkpoint_ids) < 2:
            raise ValueError("Need at least 2 checkpoints to merge")
        
        if weights and len(weights) != len(checkpoint_ids):
            raise ValueError("Number of weights must match number of checkpoints")
        
        if not weights:
            weights = [1.0 / len(checkpoint_ids)] * len(checkpoint_ids)
        
        # Load all checkpoints
        checkpoints = []
        for checkpoint_id in checkpoint_ids:
            data, metadata = self.load_checkpoint(checkpoint_id)
            checkpoints.append((data, metadata))
        
        # Merge adapter states
        merged_adapters = {}
        base_adapters = checkpoints[0][0]["adapters"]
        
        for adapter_name in base_adapters.keys():
            # Check if adapter exists in all checkpoints
            if not all(adapter_name in cp[0]["adapters"] for cp, _ in checkpoints):
                warnings.warn(f"Adapter {adapter_name} not found in all checkpoints, skipping")
                continue
            
            # Merge based on strategy
            if merge_strategy == "average":
                merged_state = self._average_adapter_states(
                    [cp[0]["adapters"][adapter_name] for cp, _ in checkpoints],
                    weights
                )
            elif merge_strategy == "weighted_sum":
                merged_state = self._weighted_sum_adapter_states(
                    [cp[0]["adapters"][adapter_name] for cp, _ in checkpoints],
                    weights
                )
            else:
                raise ValueError(f"Unknown merge strategy: {merge_strategy}")
            
            merged_adapters[adapter_name] = merged_state
        
        # Create merged checkpoint metadata
        merge_id = f"merge_{'_'.join(checkpoint_ids[:3])}_{int(time.time())}"
        merged_metadata = CheckpointMetadata(
            checkpoint_id=merge_id,
            timestamp=time.time(),
            description=f"Merged from {len(checkpoint_ids)} checkpoints",
            model_hash="",  # Will be computed during save
            adapter_count=len(merged_adapters),
            total_parameters=0,  # Will be computed during save
            tags=["merged", merge_strategy]
        )
        
        # Create temporary adapters for saving
        temp_adapters = {}
        for name, state in merged_adapters.items():
            # Create a minimal adapter wrapper for the state
            temp_adapters[name] = type('TempAdapter', (), {
                'state_dict': lambda: state
            })()
        
        return self.save_checkpoint(temp_adapters, merged_metadata)
    
    def auto_checkpoint(
        self,
        adapters: Dict[str, Union[LoRALinear, LoRAEmbedding]],
        current_step: int,
        performance_metrics: Dict[str, float]
    ) -> Optional[str]:
        """Automatically create checkpoint based on configured criteria."""
        should_checkpoint = False
        reason = ""
        
        # Check step interval
        if (current_step - self.last_auto_checkpoint) >= self.config.auto_checkpoint_interval:
            should_checkpoint = True
            reason = f"step_interval_{self.config.auto_checkpoint_interval}"
        
        # Check performance improvement
        if (self.config.auto_checkpoint_on_improvement and 
            self.performance_history):
            
            # Get best previous performance
            best_score = max(h.get('primary_metric', 0) for h in self.performance_history)
            current_score = performance_metrics.get('primary_metric', 0)
            
            if current_score > best_score + self.config.performance_threshold:
                should_checkpoint = True
                reason = f"performance_improvement_{current_score:.4f}"
        
        if should_checkpoint:
            metadata = CheckpointMetadata(
                checkpoint_id="",  # Will be generated
                timestamp=time.time(),
                description=f"Auto-checkpoint: {reason}",
                model_hash="",
                adapter_count=len(adapters),
                total_parameters=0,
                performance_metrics=performance_metrics,
                training_step=current_step,
                tags=["auto", reason]
            )
            
            checkpoint_id = self.save_checkpoint(adapters, metadata)
            self.last_auto_checkpoint = current_step
            
            # Update performance history
            self.performance_history.append(performance_metrics)
            
            return checkpoint_id
        
        return None
    
    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get statistics about checkpoint storage and usage."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Basic stats
            cursor.execute("SELECT COUNT(*) FROM checkpoints")
            total_checkpoints = cursor.fetchone()[0]
            
            cursor.execute("SELECT SUM(file_size_mb) FROM checkpoints")
            total_size_mb = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT AVG(compression_ratio) FROM checkpoints WHERE compression_ratio > 0")
            avg_compression = cursor.fetchone()[0] or 1.0
            
            # Storage format distribution
            cursor.execute("SELECT storage_format, COUNT(*) FROM checkpoints GROUP BY storage_format")
            format_distribution = dict(cursor.fetchall())
            
            # Tag distribution
            cursor.execute("SELECT tags FROM checkpoints WHERE tags IS NOT NULL")
            all_tags = []
            for (tags_json,) in cursor.fetchall():
                if tags_json:
                    all_tags.extend(json.loads(tags_json))
            
            tag_counts = defaultdict(int)
            for tag in all_tags:
                tag_counts[tag] += 1
            
            return {
                "total_checkpoints": total_checkpoints,
                "total_size_mb": total_size_mb,
                "average_compression_ratio": avg_compression,
                "format_distribution": format_distribution,
                "tag_distribution": dict(tag_counts),
                "checkpoint_directory": str(self.checkpoint_dir),
                "database_size_mb": self.db_path.stat().st_size / 1024 / 1024
            }
    
    # Helper methods
    
    def _generate_checkpoint_id(self) -> str:
        """Generate unique checkpoint ID."""
        timestamp = int(time.time() * 1000)  # milliseconds
        random_suffix = hashlib.md5(str(timestamp).encode()).hexdigest()[:8]
        return f"checkpoint_{timestamp}_{random_suffix}"
    
    def _compute_model_hash(self, adapters: Dict[str, Any]) -> str:
        """Compute hash of adapter states for verification."""
        state_strings = []
        for name in sorted(adapters.keys()):
            adapter = adapters[name]
            if hasattr(adapter, 'state_dict'):
                state_dict = adapter.state_dict()
                # Create string representation of state
                param_strings = []
                for key in sorted(state_dict.keys()):
                    param = state_dict[key]
                    if isinstance(param, torch.Tensor):
                        param_strings.append(f"{key}:{param.sum().item():.6f}")
                state_strings.append(f"{name}:{':'.join(param_strings)}")
        
        combined_string = "|".join(state_strings)
        return hashlib.sha256(combined_string.encode()).hexdigest()[:16]
    
    def _extract_adapter_state(self, adapter: Any) -> Dict[str, torch.Tensor]:
        """Extract state from custom adapter types."""
        state = {}
        for attr_name in dir(adapter):
            attr = getattr(adapter, attr_name)
            if isinstance(attr, torch.Tensor) and attr.requires_grad:
                state[attr_name] = attr.detach().clone()
        return state
    
    def _compress_checkpoint(self, data: Dict[str, Any], checkpoint_id: str) -> Dict[str, Any]:
        """Apply compression to checkpoint data."""
        import gzip
        
        # Delta compression if enabled
        if self.config.use_delta_compression and checkpoint_id in self.baseline_states:
            data = self._apply_delta_compression(data, checkpoint_id)
        else:
            # Store as new baseline
            self.baseline_states[checkpoint_id] = data.copy()
        
        # Serialize and compress
        serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = gzip.compress(serialized, compresslevel=self.config.compression_level)
        
        return {
            "compressed": True,
            "original_size": len(serialized),
            "compressed_size": len(compressed),
            "data": compressed
        }
    
    def _decompress_checkpoint(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress checkpoint data."""
        import gzip
        
        compressed_data = data["data"]
        decompressed = gzip.decompress(compressed_data)
        return pickle.loads(decompressed)
    
    def _save_metadata_to_db(self, metadata: CheckpointMetadata):
        """Save checkpoint metadata to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Main checkpoint record
            cursor.execute("""
                INSERT OR REPLACE INTO checkpoints 
                (checkpoint_id, timestamp, description, model_hash, adapter_count,
                 total_parameters, training_step, epoch, learning_rate, file_size_mb,
                 compression_ratio, storage_format, parent_checkpoint, base_model_version,
                 tags, notes, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.checkpoint_id, metadata.timestamp, metadata.description,
                metadata.model_hash, metadata.adapter_count, metadata.total_parameters,
                metadata.training_step, metadata.epoch, metadata.learning_rate,
                metadata.file_size_mb, metadata.compression_ratio, metadata.storage_format,
                metadata.parent_checkpoint, metadata.base_model_version,
                json.dumps(metadata.tags), metadata.notes,
                json.dumps({"performance_metrics": metadata.performance_metrics,
                           "validation_scores": metadata.validation_scores})
            ))
            
            # Performance metrics
            for metric_name, value in metadata.performance_metrics.items():
                cursor.execute("""
                    INSERT INTO performance_metrics (checkpoint_id, metric_name, metric_value)
                    VALUES (?, ?, ?)
                """, (metadata.checkpoint_id, metric_name, value))
            
            # Validation scores
            for score_name, value in metadata.validation_scores.items():
                cursor.execute("""
                    INSERT INTO validation_scores (checkpoint_id, score_name, score_value)
                    VALUES (?, ?, ?)
                """, (metadata.checkpoint_id, score_name, value))
            
            conn.commit()
    
    def _validate_checkpoint(self, checkpoint_path: Path) -> bool:
        """Validate checkpoint file integrity."""
        try:
            # Check file exists and is readable
            if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
                return False
            
            # Try to load the checkpoint
            if checkpoint_path.suffix == ".pt":
                data = torch.load(checkpoint_path, map_location="cpu")
            else:
                with open(checkpoint_path, "rb") as f:
                    data = pickle.load(f)
            
            # Verify required keys
            required_keys = ["metadata", "adapters", "timestamp"]
            return all(key in data for key in required_keys)
            
        except Exception:
            return False
    
    def _validate_checkpoint_data(self, data: Dict[str, Any]) -> bool:
        """Validate checkpoint data structure."""
        required_keys = ["metadata", "adapters", "timestamp"]
        return all(key in data for key in required_keys)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to stay within limits."""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) > self.config.max_checkpoints:
            # Sort by timestamp and remove oldest
            checkpoints_to_remove = sorted(
                checkpoints, key=lambda x: x.timestamp
            )[:-self.config.max_checkpoints]
            
            for checkpoint in checkpoints_to_remove:
                # Don't remove checkpoints with specific tags
                if not any(tag in ["important", "baseline", "release"] for tag in checkpoint.tags):
                    self.delete_checkpoint(checkpoint.checkpoint_id)
    
    def _average_adapter_states(
        self,
        states: List[Dict[str, torch.Tensor]],
        weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """Average multiple adapter states with weights."""
        if not states:
            return {}
        
        merged_state = {}
        for key in states[0].keys():
            tensors = [state[key] for state in states if key in state]
            weighted_tensors = [t * w for t, w in zip(tensors, weights)]
            merged_state[key] = torch.stack(weighted_tensors).sum(dim=0)
        
        return merged_state
    
    def _weighted_sum_adapter_states(
        self,
        states: List[Dict[str, torch.Tensor]],
        weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """Weighted sum of multiple adapter states."""
        return self._average_adapter_states(states, weights)  # Same implementation
    
    def _create_adapter_from_state(
        self,
        adapter_class: type,
        state_dict: Dict[str, torch.Tensor]
    ) -> Union[LoRALinear, LoRAEmbedding]:
        """Create adapter instance from state dict."""
        # This would need to be customized based on actual adapter constructors
        # For now, assume we can infer dimensions from state dict
        
        if 'lora_A' in state_dict and 'lora_B' in state_dict:
            rank, in_features = state_dict['lora_A'].shape
            out_features, _ = state_dict['lora_B'].shape
            
            adapter = adapter_class(in_features, out_features, rank)
            adapter.load_state_dict(state_dict)
            return adapter
        
        raise ValueError("Cannot determine adapter dimensions from state dict")
    
    def _save_as_safetensors(self, data: Dict[str, Any], path: Path):
        """Save checkpoint using safetensors format."""
        try:
            import safetensors.torch
            # Implementation would depend on safetensors API
            # For now, fall back to torch.save
            torch.save(data, path.with_suffix(".pt"))
        except ImportError:
            warnings.warn("safetensors not available, falling back to torch.save")
            torch.save(data, path.with_suffix(".pt"))
    
    def _load_from_safetensors(self, path: Path) -> Dict[str, Any]:
        """Load checkpoint from safetensors format."""
        try:
            import safetensors.torch
            # Implementation would depend on safetensors API
            # For now, fall back to torch.load
            return torch.load(path.with_suffix(".pt"))
        except ImportError:
            warnings.warn("safetensors not available, falling back to torch.load")
            return torch.load(path.with_suffix(".pt"))
    
    def _apply_delta_compression(self, data: Dict[str, Any], checkpoint_id: str) -> Dict[str, Any]:
        """Apply delta compression against baseline."""
        # Simplified delta compression - store only differences
        baseline = self.baseline_states.get(checkpoint_id, {})
        
        # This would compute parameter differences and store only deltas
        # Implementation would depend on specific compression strategy
        return data  # Placeholder implementation