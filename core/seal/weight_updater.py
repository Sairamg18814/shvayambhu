"""Weight update module for SEAL.

This module applies self-edit instructions to model weights using
gradient-free optimization techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
from dataclasses import dataclass
import copy
import logging

from .edit_format import EditInstruction, EditType, EditScope

logger = logging.getLogger(__name__)


@dataclass
class UpdateConfig:
    """Configuration for weight updates."""
    learning_rate: float = 0.01
    momentum: float = 0.9
    max_norm: float = 1.0  # Gradient clipping
    regularization: float = 0.001
    update_steps: int = 10
    validation_frequency: int = 5
    rollback_on_failure: bool = True
    

class WeightUpdate:
    """Represents a weight update operation."""
    
    def __init__(
        self,
        parameter_name: str,
        update_tensor: torch.Tensor,
        edit_instruction: EditInstruction
    ):
        self.parameter_name = parameter_name
        self.update_tensor = update_tensor
        self.edit_instruction = edit_instruction
        self.applied = False
        self.rollback_state = None
        
    def apply(self, model: nn.Module) -> bool:
        """Apply update to model."""
        try:
            # Find parameter
            param = self._find_parameter(model, self.parameter_name)
            if param is None:
                logger.warning(f"Parameter {self.parameter_name} not found")
                return False
            
            # Save rollback state
            self.rollback_state = param.data.clone()
            
            # Apply update
            with torch.no_grad():
                param.data += self.update_tensor * self.edit_instruction.strength
            
            self.applied = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply update: {e}")
            return False
    
    def rollback(self, model: nn.Module) -> bool:
        """Rollback update."""
        if not self.applied or self.rollback_state is None:
            return False
        
        try:
            param = self._find_parameter(model, self.parameter_name)
            if param is not None:
                with torch.no_grad():
                    param.data.copy_(self.rollback_state)
                self.applied = False
                return True
        except Exception as e:
            logger.error(f"Failed to rollback: {e}")
        
        return False
    
    def _find_parameter(self, model: nn.Module, name: str) -> Optional[nn.Parameter]:
        """Find parameter by name in model."""
        parts = name.split('.')
        module = model
        
        for part in parts[:-1]:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        
        if hasattr(module, parts[-1]):
            return getattr(module, parts[-1])
        return None


class WeightUpdater:
    """Applies edit instructions to model weights."""
    
    def __init__(self, config: Optional[UpdateConfig] = None):
        self.config = config or UpdateConfig()
        self.update_history: List[WeightUpdate] = []
        self.performance_tracker = PerformanceTracker()
        
    def apply_edit(
        self,
        model: nn.Module,
        instruction: EditInstruction,
        validation_fn: Optional[Callable] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Apply a single edit instruction to model.
        
        Args:
            model: Model to update
            instruction: Edit instruction
            validation_fn: Optional validation function
            
        Returns:
            Tuple of (success, metrics)
        """
        logger.info(f"Applying edit: {instruction.description}")
        
        # Generate weight updates
        updates = self._generate_updates(model, instruction)
        
        if not updates:
            logger.warning("No updates generated")
            return False, {"error": "No updates generated"}
        
        # Apply updates
        success_count = 0
        for update in updates:
            if update.apply(model):
                success_count += 1
                self.update_history.append(update)
        
        # Validate if function provided
        validation_passed = True
        validation_metrics = {}
        
        if validation_fn and success_count > 0:
            validation_passed, validation_metrics = self._validate_updates(
                model, validation_fn, updates
            )
            
            # Rollback if validation failed and configured to do so
            if not validation_passed and self.config.rollback_on_failure:
                logger.warning("Validation failed, rolling back updates")
                for update in reversed(updates):
                    update.rollback(model)
                success_count = 0
        
        return success_count > 0 and validation_passed, {
            "updates_applied": success_count,
            "total_updates": len(updates),
            "validation_passed": validation_passed,
            "validation_metrics": validation_metrics
        }
    
    def apply_batch(
        self,
        model: nn.Module,
        instructions: List[EditInstruction],
        validation_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Apply a batch of edit instructions."""
        results = []
        
        for instruction in instructions:
            success, metrics = self.apply_edit(model, instruction, validation_fn)
            results.append({
                "instruction": instruction.description,
                "success": success,
                "metrics": metrics
            })
        
        # Aggregate results
        total_success = sum(r["success"] for r in results)
        
        return {
            "total_instructions": len(instructions),
            "successful_edits": total_success,
            "success_rate": total_success / len(instructions) if instructions else 0,
            "individual_results": results
        }
    
    def _generate_updates(
        self,
        model: nn.Module,
        instruction: EditInstruction
    ) -> List[WeightUpdate]:
        """Generate weight updates from instruction."""
        updates = []
        
        # Get target parameters based on scope
        target_params = self._get_target_parameters(model, instruction.scope)
        
        # Generate updates based on edit type
        if instruction.edit_type == EditType.KNOWLEDGE_UPDATE:
            updates = self._generate_knowledge_updates(target_params, instruction)
        elif instruction.edit_type == EditType.BEHAVIOR_MODIFICATION:
            updates = self._generate_behavior_updates(target_params, instruction)
        elif instruction.edit_type == EditType.SKILL_ENHANCEMENT:
            updates = self._generate_skill_updates(target_params, instruction)
        elif instruction.edit_type == EditType.BIAS_CORRECTION:
            updates = self._generate_bias_updates(target_params, instruction)
        else:
            # Default update strategy
            updates = self._generate_default_updates(target_params, instruction)
        
        return updates
    
    def _get_target_parameters(
        self,
        model: nn.Module,
        scope: EditScope
    ) -> Dict[str, nn.Parameter]:
        """Get parameters matching the scope."""
        target_params = {}
        
        for name, param in model.named_parameters():
            # Check if parameter matches scope
            if scope.global_edit:
                target_params[name] = param
            else:
                # Check layers
                if scope.layers:
                    layer_match = any(
                        f"layer{l}" in name or f"layers.{l}" in name
                        for l in scope.layers
                    )
                    if not layer_match:
                        continue
                
                # Check modules
                if scope.modules:
                    module_match = any(m in name for m in scope.modules)
                    if not module_match:
                        continue
                
                # Check parameter names
                if scope.parameters:
                    param_match = any(p in name for p in scope.parameters)
                    if not param_match:
                        continue
                
                target_params[name] = param
        
        return target_params
    
    def _generate_knowledge_updates(
        self,
        params: Dict[str, nn.Parameter],
        instruction: EditInstruction
    ) -> List[WeightUpdate]:
        """Generate updates for knowledge modification."""
        updates = []
        
        # Focus on embedding and output layers
        for name, param in params.items():
            if any(layer in name for layer in ["embed", "output", "lm_head"]):
                # Generate perturbation based on instruction
                update_tensor = self._compute_knowledge_perturbation(
                    param, instruction
                )
                
                updates.append(WeightUpdate(name, update_tensor, instruction))
        
        return updates
    
    def _generate_behavior_updates(
        self,
        params: Dict[str, nn.Parameter],
        instruction: EditInstruction
    ) -> List[WeightUpdate]:
        """Generate updates for behavior modification."""
        updates = []
        
        # Focus on attention and MLP layers
        for name, param in params.items():
            if any(layer in name for layer in ["attention", "mlp", "ffn"]):
                update_tensor = self._compute_behavior_perturbation(
                    param, instruction
                )
                
                updates.append(WeightUpdate(name, update_tensor, instruction))
        
        return updates
    
    def _generate_skill_updates(
        self,
        params: Dict[str, nn.Parameter],
        instruction: EditInstruction
    ) -> List[WeightUpdate]:
        """Generate updates for skill enhancement."""
        updates = []
        
        # Apply smaller updates across all parameters
        for name, param in params.items():
            update_tensor = self._compute_skill_perturbation(
                param, instruction
            )
            
            updates.append(WeightUpdate(name, update_tensor, instruction))
        
        return updates
    
    def _generate_bias_updates(
        self,
        params: Dict[str, nn.Parameter],
        instruction: EditInstruction
    ) -> List[WeightUpdate]:
        """Generate updates for bias correction."""
        updates = []
        
        # Focus on bias parameters and normalization layers
        for name, param in params.items():
            if "bias" in name or "norm" in name:
                update_tensor = self._compute_bias_perturbation(
                    param, instruction
                )
                
                updates.append(WeightUpdate(name, update_tensor, instruction))
        
        return updates
    
    def _generate_default_updates(
        self,
        params: Dict[str, nn.Parameter],
        instruction: EditInstruction
    ) -> List[WeightUpdate]:
        """Generate default updates."""
        updates = []
        
        for name, param in params.items():
            # Simple random perturbation
            update_tensor = torch.randn_like(param) * self.config.learning_rate
            updates.append(WeightUpdate(name, update_tensor, instruction))
        
        return updates
    
    def _compute_knowledge_perturbation(
        self,
        param: nn.Parameter,
        instruction: EditInstruction
    ) -> torch.Tensor:
        """Compute perturbation for knowledge update."""
        # This is a simplified version - in practice, you'd use
        # more sophisticated methods like influence functions
        
        # Random direction with magnitude based on confidence
        direction = torch.randn_like(param)
        direction = F.normalize(direction.flatten(), dim=0).reshape(param.shape)
        
        magnitude = self.config.learning_rate * instruction.confidence
        
        return direction * magnitude
    
    def _compute_behavior_perturbation(
        self,
        param: nn.Parameter,
        instruction: EditInstruction
    ) -> torch.Tensor:
        """Compute perturbation for behavior modification."""
        # Add structured noise to encourage exploration
        base_noise = torch.randn_like(param)
        
        # Add low-rank structure
        if param.dim() == 2:
            u = torch.randn(param.size(0), 1)
            v = torch.randn(1, param.size(1))
            low_rank = torch.matmul(u, v)
            base_noise = base_noise + 0.5 * low_rank
        
        return base_noise * self.config.learning_rate * instruction.strength
    
    def _compute_skill_perturbation(
        self,
        param: nn.Parameter,
        instruction: EditInstruction
    ) -> torch.Tensor:
        """Compute perturbation for skill enhancement."""
        # Smaller, more conservative updates
        noise = torch.randn_like(param)
        
        # Apply regularization to prevent catastrophic forgetting
        regularization = -self.config.regularization * param
        
        return (noise + regularization) * self.config.learning_rate * 0.1
    
    def _compute_bias_perturbation(
        self,
        param: nn.Parameter,
        instruction: EditInstruction
    ) -> torch.Tensor:
        """Compute perturbation for bias correction."""
        # Move biases toward zero to reduce bias
        return -param * self.config.learning_rate * instruction.strength
    
    def _validate_updates(
        self,
        model: nn.Module,
        validation_fn: Callable,
        updates: List[WeightUpdate]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate model after updates."""
        try:
            metrics = validation_fn(model)
            
            # Check if key metrics improved or at least didn't degrade
            passed = True
            if "accuracy" in metrics:
                passed &= metrics["accuracy"] >= 0.9 * metrics.get("baseline_accuracy", 0)
            if "perplexity" in metrics:
                passed &= metrics["perplexity"] <= 1.1 * metrics.get("baseline_perplexity", float('inf'))
            
            return passed, metrics
            
        except Exception as e:
            logger.error(f"Validation failed with error: {e}")
            return False, {"error": str(e)}
    
    def get_update_summary(self) -> Dict[str, Any]:
        """Get summary of all updates applied."""
        by_type = {}
        for update in self.update_history:
            edit_type = update.edit_instruction.edit_type.value
            by_type[edit_type] = by_type.get(edit_type, 0) + 1
        
        return {
            "total_updates": len(self.update_history),
            "updates_by_type": by_type,
            "parameters_affected": len(set(u.parameter_name for u in self.update_history))
        }


class PerformanceTracker:
    """Track model performance across updates."""
    
    def __init__(self):
        self.metrics_history = []
        
    def record(
        self,
        step: int,
        metrics: Dict[str, float],
        edit_description: str
    ):
        """Record performance metrics."""
        self.metrics_history.append({
            "step": step,
            "metrics": metrics,
            "edit": edit_description,
            "timestamp": torch.cuda.Event(enable_timing=True)
        })
    
    def get_trend(self, metric_name: str) -> List[float]:
        """Get trend for specific metric."""
        return [
            entry["metrics"].get(metric_name, 0)
            for entry in self.metrics_history
        ]
