"""Rollback mechanism for SEAL architecture.

This module provides comprehensive rollback capabilities for self-editing
operations, allowing safe recovery from failed or problematic edits.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import time
import json
from pathlib import Path
import warnings
from collections import deque
import hashlib

from .parameter_diff import ParameterDiff
from .edit_validation import ValidationResult


@dataclass
class Checkpoint:
    """A model checkpoint for rollback purposes."""
    checkpoint_id: str
    timestamp: float
    description: str
    state_dict: Dict[str, torch.Tensor]
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_metrics: Optional[Dict[str, float]] = None
    edit_sequence: List[str] = field(default_factory=list)  # IDs of applied edits
    
    def __post_init__(self):
        if not self.checkpoint_id:
            self.checkpoint_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique checkpoint ID."""
        content = f"{self.timestamp}_{self.description}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class EditRecord:
    """Record of an applied edit for rollback purposes."""
    edit_id: str
    timestamp: float
    parameter_name: str
    diff: ParameterDiff
    validation_result: Optional[ValidationResult] = None
    performance_before: Optional[Dict[str, float]] = None
    performance_after: Optional[Dict[str, float]] = None
    rollback_diff: Optional[ParameterDiff] = None  # Inverse diff for rollback
    
    def __post_init__(self):
        if not self.edit_id:
            self.edit_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique edit ID."""
        content = f"{self.timestamp}_{self.parameter_name}_{self.diff.magnitude}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class RollbackConfig:
    """Configuration for rollback manager."""
    max_checkpoints: int = 10
    max_edit_history: int = 100
    auto_checkpoint_interval: float = 3600.0  # 1 hour in seconds
    enable_performance_monitoring: bool = True
    performance_degradation_threshold: float = 0.05  # 5% degradation
    enable_automatic_rollback: bool = False
    rollback_validation_threshold: float = 0.7


class RollbackManager:
    """Manages checkpoints and rollback operations for SEAL."""
    
    def __init__(self, config: RollbackConfig):
        self.config = config
        self.checkpoints: deque = deque(maxlen=config.max_checkpoints)
        self.edit_history: deque = deque(maxlen=config.max_edit_history)
        self.last_checkpoint_time = time.time()
        self.baseline_performance: Optional[Dict[str, float]] = None
        
        # Current state tracking
        self.current_model_state: Optional[Dict[str, torch.Tensor]] = None
        self.applied_edits: List[str] = []  # List of edit IDs in order
    
    def create_checkpoint(
        self,
        model: nn.Module,
        description: str = "manual",
        validation_metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """Create a checkpoint of the current model state."""
        checkpoint = Checkpoint(
            checkpoint_id="",  # Will be auto-generated
            timestamp=time.time(),
            description=description,
            state_dict=self._deep_copy_state_dict(model.state_dict()),
            validation_metrics=validation_metrics,
            edit_sequence=self.applied_edits.copy()
        )
        
        self.checkpoints.append(checkpoint)
        self.last_checkpoint_time = checkpoint.timestamp
        
        print(f"Created checkpoint {checkpoint.checkpoint_id}: {description}")
        return checkpoint.checkpoint_id
    
    def _deep_copy_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Create a deep copy of the state dict."""
        return {key: tensor.clone().detach() for key, tensor in state_dict.items()}
    
    def record_edit(
        self,
        parameter_name: str,
        diff: ParameterDiff,
        validation_result: Optional[ValidationResult] = None,
        performance_before: Optional[Dict[str, float]] = None,
        performance_after: Optional[Dict[str, float]] = None
    ) -> str:
        """Record an applied edit for potential rollback."""
        # Compute inverse diff for rollback
        rollback_diff = self._compute_inverse_diff(diff)
        
        edit_record = EditRecord(
            edit_id="",  # Will be auto-generated
            timestamp=time.time(),
            parameter_name=parameter_name,
            diff=diff,
            validation_result=validation_result,
            performance_before=performance_before,
            performance_after=performance_after,
            rollback_diff=rollback_diff
        )
        
        self.edit_history.append(edit_record)
        self.applied_edits.append(edit_record.edit_id)
        
        # Check if automatic checkpoint is needed
        if (edit_record.timestamp - self.last_checkpoint_time > 
            self.config.auto_checkpoint_interval):
            # Note: Would need model reference for auto-checkpoint
            pass
        
        # Check for automatic rollback if enabled
        if (self.config.enable_automatic_rollback and 
            self._should_trigger_automatic_rollback(edit_record)):
            warnings.warn(f"Edit {edit_record.edit_id} triggered automatic rollback conditions")
        
        return edit_record.edit_id
    
    def _compute_inverse_diff(self, diff: ParameterDiff) -> ParameterDiff:
        """Compute the inverse of a parameter diff for rollback."""
        inverse_diff = ParameterDiff(
            name=diff.name,
            shape=diff.shape,
            diff_type=diff.diff_type,
            magnitude=diff.magnitude,
            timestamp=time.time(),
            source=f"inverse_of_{diff.source if diff.source else 'unknown'}"
        )
        
        if diff.diff_type == 'full' and diff.full_diff is not None:
            inverse_diff.full_diff = -diff.full_diff
        
        elif diff.diff_type == 'lora' and diff.lora_A is not None and diff.lora_B is not None:
            # For LoRA, the inverse is just negative LoRA matrices
            inverse_diff.lora_A = diff.lora_A.clone()
            inverse_diff.lora_B = -diff.lora_B.clone()
            inverse_diff.lora_rank = diff.lora_rank
        
        elif diff.diff_type == 'sparse' and diff.sparse_values is not None:
            inverse_diff.sparse_indices = diff.sparse_indices.clone()
            inverse_diff.sparse_values = -diff.sparse_values.clone()
            inverse_diff.sparsity_ratio = diff.sparsity_ratio
        
        return inverse_diff
    
    def _should_trigger_automatic_rollback(self, edit_record: EditRecord) -> bool:
        """Check if an edit should trigger automatic rollback."""
        if not self.config.enable_automatic_rollback:
            return False
        
        # Check validation result
        if (edit_record.validation_result and 
            edit_record.validation_result.confidence < self.config.rollback_validation_threshold):
            return True
        
        # Check performance degradation
        if (edit_record.performance_before and edit_record.performance_after and
            self.config.enable_performance_monitoring):
            
            for metric_name in edit_record.performance_before:
                if metric_name in edit_record.performance_after:
                    before = edit_record.performance_before[metric_name]
                    after = edit_record.performance_after[metric_name]
                    
                    # Assume higher is better for most metrics
                    degradation = (before - after) / before if before != 0 else 0
                    
                    if degradation > self.config.performance_degradation_threshold:
                        return True
        
        return False
    
    def rollback_to_checkpoint(
        self,
        model: nn.Module,
        checkpoint_id: str
    ) -> bool:
        """Rollback model to a specific checkpoint."""
        # Find the checkpoint
        target_checkpoint = None
        for checkpoint in self.checkpoints:
            if checkpoint.checkpoint_id == checkpoint_id:
                target_checkpoint = checkpoint
                break
        
        if target_checkpoint is None:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        try:
            # Load the checkpoint state
            model.load_state_dict(target_checkpoint.state_dict)
            
            # Update applied edits to match checkpoint
            self.applied_edits = target_checkpoint.edit_sequence.copy()
            
            # Remove edit history that came after this checkpoint
            self._clean_edit_history_after_timestamp(target_checkpoint.timestamp)
            
            print(f"Successfully rolled back to checkpoint {checkpoint_id}")
            return True
        
        except Exception as e:
            print(f"Failed to rollback to checkpoint {checkpoint_id}: {str(e)}")
            return False
    
    def rollback_last_n_edits(
        self,
        model: nn.Module,
        n: int = 1
    ) -> bool:
        """Rollback the last n edits."""
        if n <= 0:
            return True
        
        if len(self.edit_history) < n:
            warnings.warn(f"Only {len(self.edit_history)} edits available, rolling back all")
            n = len(self.edit_history)
        
        # Get the last n edits in reverse order
        edits_to_rollback = list(self.edit_history)[-n:]
        edits_to_rollback.reverse()
        
        success = True
        rolled_back_count = 0
        
        for edit_record in edits_to_rollback:
            if self._rollback_single_edit(model, edit_record):
                rolled_back_count += 1
                # Remove from applied edits
                if edit_record.edit_id in self.applied_edits:
                    self.applied_edits.remove(edit_record.edit_id)
            else:
                success = False
                break
        
        # Remove rolled back edits from history
        for _ in range(rolled_back_count):
            if self.edit_history:
                self.edit_history.pop()
        
        print(f"Rolled back {rolled_back_count}/{n} edits")
        return success
    
    def _rollback_single_edit(
        self,
        model: nn.Module,
        edit_record: EditRecord
    ) -> bool:
        """Rollback a single edit using its inverse diff."""
        if edit_record.rollback_diff is None:
            warnings.warn(f"No rollback diff available for edit {edit_record.edit_id}")
            return False
        
        try:
            # Find the target parameter
            target_param = None
            for name, param in model.named_parameters():
                if name == edit_record.parameter_name:
                    target_param = param
                    break
            
            if target_param is None:
                warnings.warn(f"Parameter {edit_record.parameter_name} not found in model")
                return False
            
            # Apply the inverse diff
            with torch.no_grad():
                if edit_record.rollback_diff.diff_type == 'full':
                    target_param.add_(edit_record.rollback_diff.full_diff)
                
                elif edit_record.rollback_diff.diff_type == 'lora':
                    lora_change = torch.mm(
                        edit_record.rollback_diff.lora_A,
                        edit_record.rollback_diff.lora_B
                    )
                    target_param.add_(lora_change)
                
                elif edit_record.rollback_diff.diff_type == 'sparse':
                    if edit_record.rollback_diff.sparse_indices.dim() == 2:
                        target_param[
                            edit_record.rollback_diff.sparse_indices[:, 0],
                            edit_record.rollback_diff.sparse_indices[:, 1]
                        ] += edit_record.rollback_diff.sparse_values
                    else:
                        target_param.view(-1)[
                            edit_record.rollback_diff.sparse_indices
                        ] += edit_record.rollback_diff.sparse_values
            
            return True
        
        except Exception as e:
            warnings.warn(f"Failed to rollback edit {edit_record.edit_id}: {str(e)}")
            return False
    
    def _clean_edit_history_after_timestamp(self, timestamp: float):
        """Remove edit history entries after a given timestamp."""
        # Convert deque to list for easier manipulation
        edit_list = list(self.edit_history)
        
        # Filter out edits after the timestamp
        filtered_edits = [edit for edit in edit_list if edit.timestamp <= timestamp]
        
        # Rebuild the deque
        self.edit_history.clear()
        self.edit_history.extend(filtered_edits)
    
    def get_rollback_candidates(self) -> List[Dict[str, Any]]:
        """Get list of possible rollback targets (checkpoints and recent edits)."""
        candidates = []
        
        # Add checkpoints
        for checkpoint in self.checkpoints:
            candidates.append({
                "type": "checkpoint",
                "id": checkpoint.checkpoint_id,
                "timestamp": checkpoint.timestamp,
                "description": checkpoint.description,
                "validation_metrics": checkpoint.validation_metrics
            })
        
        # Add recent edits (last 10)
        recent_edits = list(self.edit_history)[-10:]
        for edit in recent_edits:
            candidates.append({
                "type": "edit",
                "id": edit.edit_id,
                "timestamp": edit.timestamp,
                "parameter": edit.parameter_name,
                "magnitude": edit.diff.magnitude,
                "validation_confidence": (edit.validation_result.confidence 
                                        if edit.validation_result else None)
            })
        
        # Sort by timestamp (most recent first)
        candidates.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return candidates
    
    def validate_rollback_feasibility(
        self,
        target_checkpoint_id: Optional[str] = None,
        target_edit_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate if a rollback operation is feasible."""
        result = {
            "feasible": True,
            "warnings": [],
            "errors": [],
            "estimated_edits_lost": 0
        }
        
        if target_checkpoint_id:
            # Find checkpoint
            target_checkpoint = None
            for checkpoint in self.checkpoints:
                if checkpoint.checkpoint_id == target_checkpoint_id:
                    target_checkpoint = checkpoint
                    break
            
            if target_checkpoint is None:
                result["feasible"] = False
                result["errors"].append(f"Checkpoint {target_checkpoint_id} not found")
                return result
            
            # Calculate edits that would be lost
            edits_after_checkpoint = [
                edit for edit in self.edit_history 
                if edit.timestamp > target_checkpoint.timestamp
            ]
            result["estimated_edits_lost"] = len(edits_after_checkpoint)
            
            if result["estimated_edits_lost"] > 0:
                result["warnings"].append(
                    f"Rollback would lose {result['estimated_edits_lost']} edits"
                )
        
        elif target_edit_id:
            # Find edit and check if rollback diff is available
            target_edit = None
            for edit in self.edit_history:
                if edit.edit_id == target_edit_id:
                    target_edit = edit
                    break
            
            if target_edit is None:
                result["feasible"] = False
                result["errors"].append(f"Edit {target_edit_id} not found")
                return result
            
            if target_edit.rollback_diff is None:
                result["feasible"] = False
                result["errors"].append("No rollback diff available for this edit")
        
        return result
    
    def get_rollback_statistics(self) -> Dict[str, Any]:
        """Get statistics about rollback operations and edit history."""
        stats = {
            "total_checkpoints": len(self.checkpoints),
            "total_edits": len(self.edit_history),
            "applied_edits": len(self.applied_edits),
            "last_checkpoint_age": time.time() - self.last_checkpoint_time,
            "edit_success_rate": 0.0,
            "avg_edit_magnitude": 0.0
        }
        
        if self.edit_history:
            successful_edits = sum(
                1 for edit in self.edit_history 
                if edit.validation_result and edit.validation_result.is_valid
            )
            stats["edit_success_rate"] = successful_edits / len(self.edit_history)
            
            stats["avg_edit_magnitude"] = sum(
                edit.diff.magnitude for edit in self.edit_history
            ) / len(self.edit_history)
        
        return stats
    
    def cleanup_old_data(self, max_age_hours: float = 24.0):
        """Clean up old checkpoints and edit history."""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        # Remove old checkpoints (keep at least 2)
        checkpoints_to_keep = []
        for checkpoint in self.checkpoints:
            if (checkpoint.timestamp > cutoff_time or 
                len(checkpoints_to_keep) < 2):
                checkpoints_to_keep.append(checkpoint)
        
        self.checkpoints.clear()
        self.checkpoints.extend(checkpoints_to_keep)
        
        # Edit history is automatically managed by deque maxlen
        print(f"Cleanup completed. Kept {len(checkpoints_to_keep)} checkpoints")
    
    def save_rollback_state(self, filepath: Path):
        """Save rollback manager state to disk."""
        state = {
            "config": {
                "max_checkpoints": self.config.max_checkpoints,
                "max_edit_history": self.config.max_edit_history,
                "auto_checkpoint_interval": self.config.auto_checkpoint_interval
            },
            "applied_edits": self.applied_edits,
            "last_checkpoint_time": self.last_checkpoint_time,
            "baseline_performance": self.baseline_performance
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_rollback_state(self, filepath: Path):
        """Load rollback manager state from disk."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.applied_edits = state.get("applied_edits", [])
        self.last_checkpoint_time = state.get("last_checkpoint_time", time.time())
        self.baseline_performance = state.get("baseline_performance")