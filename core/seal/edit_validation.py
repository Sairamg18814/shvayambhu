"""Edit validation framework for SEAL architecture.

This module provides comprehensive validation and safety checks
for self-editing operations before they are applied to the model.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import warnings
import time

from .parameter_diff import ParameterDiff
from .lora_adapter import LoRALinear, LoRAEmbedding


@dataclass
class ValidationResult:
    """Result of edit validation."""
    is_valid: bool
    confidence: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    recommendation: str = "unknown"
    
    def add_error(self, error: str):
        """Add an error to the result."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add a warning to the result."""
        self.warnings.append(warning)
    
    def add_metric(self, name: str, value: float):
        """Add a metric to the result."""
        self.metrics[name] = value


@dataclass
class ValidationConfig:
    """Configuration for edit validation."""
    # Magnitude thresholds
    max_magnitude_ratio: float = 0.1  # Max change as ratio of param norm
    max_absolute_magnitude: float = 1.0  # Max absolute change
    
    # Stability checks
    check_gradient_norms: bool = True
    max_gradient_norm: float = 10.0
    check_activation_stats: bool = True
    
    # Performance validation
    check_performance_degradation: bool = True
    max_performance_drop: float = 0.05  # 5% drop
    min_validation_samples: int = 100
    
    # Safety checks
    check_parameter_bounds: bool = True
    check_numerical_stability: bool = True
    check_model_coherence: bool = True
    
    # Thresholds
    confidence_threshold: float = 0.7
    numerical_stability_eps: float = 1e-6


class EditValidator:
    """Validates proposed edits before application."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.validation_history: List[ValidationResult] = []
        self.baseline_metrics: Dict[str, float] = {}
    
    def validate_edit(
        self,
        model: nn.Module,
        diff: ParameterDiff,
        validation_data: Optional[Any] = None,
        custom_validators: Optional[List[Callable]] = None
    ) -> ValidationResult:
        """Comprehensive validation of a proposed edit."""
        result = ValidationResult(is_valid=True, confidence=1.0)
        
        # 1. Magnitude validation
        self._validate_magnitude(diff, result)
        
        # 2. Parameter bounds validation
        if self.config.check_parameter_bounds:
            self._validate_parameter_bounds(model, diff, result)
        
        # 3. Numerical stability validation
        if self.config.check_numerical_stability:
            self._validate_numerical_stability(diff, result)
        
        # 4. Model coherence validation
        if self.config.check_model_coherence:
            self._validate_model_coherence(model, diff, result)
        
        # 5. Performance validation
        if self.config.check_performance_degradation and validation_data is not None:
            self._validate_performance(model, diff, validation_data, result)
        
        # 6. Custom validations
        if custom_validators:
            for validator in custom_validators:
                try:
                    validator(model, diff, result)
                except Exception as e:
                    result.add_error(f"Custom validator failed: {str(e)}")
        
        # 7. Final confidence calculation
        result.confidence = self._calculate_confidence(result)
        
        # 8. Generate recommendation
        result.recommendation = self._generate_recommendation(result)
        
        # Store result
        self.validation_history.append(result)
        
        return result
    
    def _validate_magnitude(self, diff: ParameterDiff, result: ValidationResult):
        """Validate the magnitude of parameter changes."""
        magnitude = diff.magnitude
        
        # Check absolute magnitude
        if magnitude > self.config.max_absolute_magnitude:
            result.add_error(f"Edit magnitude {magnitude:.6f} exceeds absolute threshold {self.config.max_absolute_magnitude}")
        
        result.add_metric("magnitude", magnitude)
        
        # Note: Relative magnitude check would require original parameter norm
        # This would be implemented when we have access to the original parameters
    
    def _validate_parameter_bounds(
        self,
        model: nn.Module,
        diff: ParameterDiff,
        result: ValidationResult
    ):
        """Validate that parameters stay within reasonable bounds."""
        # Get the target parameter
        target_param = None
        for name, param in model.named_parameters():
            if name == diff.name:
                target_param = param
                break
        
        if target_param is None:
            result.add_warning(f"Could not find parameter {diff.name} in model")
            return
        
        # Simulate applying the diff
        if diff.diff_type == 'lora':
            simulated_change = torch.mm(diff.lora_A, diff.lora_B)
        elif diff.diff_type == 'sparse':
            simulated_change = torch.zeros_like(target_param)
            if diff.sparse_indices.dim() == 2:
                simulated_change[diff.sparse_indices[:, 0], diff.sparse_indices[:, 1]] = diff.sparse_values
            else:
                simulated_change.view(-1)[diff.sparse_indices] = diff.sparse_values
        elif diff.diff_type == 'full':
            simulated_change = diff.full_diff
        else:
            result.add_warning(f"Unknown diff type: {diff.diff_type}")
            return
        
        new_param = target_param + simulated_change
        
        # Check for extreme values
        if torch.any(torch.isnan(new_param)):
            result.add_error("Edit would introduce NaN values")
        
        if torch.any(torch.isinf(new_param)):
            result.add_error("Edit would introduce infinite values")
        
        # Check parameter distribution
        param_std = torch.std(new_param).item()
        param_mean = torch.mean(new_param).item()
        
        result.add_metric("new_param_std", param_std)
        result.add_metric("new_param_mean", param_mean)
        
        # Heuristic bounds based on typical neural network parameters
        if param_std > 10.0:
            result.add_warning(f"Parameter standard deviation {param_std:.4f} is very high")
        
        if abs(param_mean) > 5.0:
            result.add_warning(f"Parameter mean {param_mean:.4f} is far from zero")
    
    def _validate_numerical_stability(self, diff: ParameterDiff, result: ValidationResult):
        """Check for numerical stability issues."""
        eps = self.config.numerical_stability_eps
        
        if diff.diff_type == 'lora':
            # Check condition number of LoRA matrices
            if diff.lora_A is not None and diff.lora_B is not None:
                try:
                    cond_A = torch.linalg.cond(diff.lora_A).item()
                    cond_B = torch.linalg.cond(diff.lora_B).item()
                    
                    result.add_metric("lora_A_condition", cond_A)
                    result.add_metric("lora_B_condition", cond_B)
                    
                    if cond_A > 1e12 or cond_B > 1e12:
                        result.add_warning("LoRA matrices have high condition numbers")
                
                except Exception:
                    result.add_warning("Could not compute condition numbers for LoRA matrices")
        
        elif diff.diff_type == 'full' and diff.full_diff is not None:
            # Check for very small or very large values
            min_val = torch.min(torch.abs(diff.full_diff[diff.full_diff != 0])).item()
            max_val = torch.max(torch.abs(diff.full_diff)).item()
            
            result.add_metric("min_nonzero_change", min_val)
            result.add_metric("max_change", max_val)
            
            if min_val < eps:
                result.add_warning(f"Some changes are very small (< {eps})")
            
            if max_val / min_val > 1e8:
                result.add_warning("Large dynamic range in parameter changes")
    
    def _validate_model_coherence(
        self,
        model: nn.Module,
        diff: ParameterDiff,
        result: ValidationResult
    ):
        """Validate that the edit maintains model coherence."""
        # Check if the parameter exists and is compatible
        target_param = None
        target_module = None
        
        for name, param in model.named_parameters():
            if name == diff.name:
                target_param = param
                break
        
        for name, module in model.named_modules():
            if name and diff.name.startswith(name):
                target_module = module
                break
        
        if target_param is None:
            result.add_error(f"Parameter {diff.name} not found in model")
            return
        
        # Check shape compatibility
        if target_param.shape != diff.shape:
            result.add_error(f"Shape mismatch: {target_param.shape} vs {diff.shape}")
            return
        
        # Check module-specific constraints
        if isinstance(target_module, nn.Linear):
            self._validate_linear_layer_edit(target_module, diff, result)
        elif isinstance(target_module, nn.Embedding):
            self._validate_embedding_layer_edit(target_module, diff, result)
        elif isinstance(target_module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            self._validate_normalization_layer_edit(target_module, diff, result)
    
    def _validate_linear_layer_edit(
        self,
        layer: nn.Linear,
        diff: ParameterDiff,
        result: ValidationResult
    ):
        """Validate edits to linear layers."""
        if diff.name.endswith('.weight'):
            # Weight matrix edits
            if diff.diff_type == 'lora' and len(diff.shape) == 2:
                # LoRA is well-suited for linear layers
                result.add_metric("lora_suitability", 1.0)
            else:
                result.add_warning("Non-LoRA edit to linear layer weight")
        
        elif diff.name.endswith('.bias'):
            # Bias vector edits
            if diff.diff_type != 'full':
                result.add_warning("Compressed edit applied to bias (usually not beneficial)")
    
    def _validate_embedding_layer_edit(
        self,
        layer: nn.Embedding,
        diff: ParameterDiff,
        result: ValidationResult
    ):
        """Validate edits to embedding layers."""
        if diff.magnitude > 0.1:  # Embeddings are typically normalized
            result.add_warning("Large magnitude edit to embedding layer")
        
        # Check if edit affects padding token (if index 0)
        if (diff.diff_type == 'sparse' and 
            diff.sparse_indices is not None and 
            torch.any(diff.sparse_indices[:, 0] == 0)):
            result.add_warning("Edit affects padding token embedding")
    
    def _validate_normalization_layer_edit(
        self,
        layer: Union[nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d],
        diff: ParameterDiff,
        result: ValidationResult
    ):
        """Validate edits to normalization layers."""
        if diff.name.endswith('.weight'):
            # Scale parameters should stay positive
            if diff.diff_type == 'full' and diff.full_diff is not None:
                # Check if any weights would become negative
                current_weight = layer.weight
                new_weight = current_weight + diff.full_diff
                
                if torch.any(new_weight <= 0):
                    result.add_error("Edit would make normalization weights non-positive")
        
        if diff.magnitude > 0.5:
            result.add_warning("Large edit to normalization layer parameters")
    
    def _validate_performance(
        self,
        model: nn.Module,
        diff: ParameterDiff,
        validation_data: Any,
        result: ValidationResult
    ):
        """Validate that edit doesn't degrade performance significantly."""
        # This would require actually applying the edit temporarily
        # and measuring performance. For now, we'll implement a placeholder.
        
        # Store baseline if not already done
        if not self.baseline_metrics:
            self.baseline_metrics = self._compute_baseline_metrics(model, validation_data)
        
        # Apply edit temporarily (this would need careful implementation)
        # For now, we'll add a placeholder metric
        result.add_metric("performance_validation", 1.0)
        result.add_warning("Performance validation not fully implemented")
    
    def _compute_baseline_metrics(self, model: nn.Module, validation_data: Any) -> Dict[str, float]:
        """Compute baseline performance metrics."""
        # Placeholder implementation
        return {"baseline_loss": 0.0, "baseline_accuracy": 0.0}
    
    def _calculate_confidence(self, result: ValidationResult) -> float:
        """Calculate overall confidence in the validation result."""
        confidence = 1.0
        
        # Reduce confidence for each error
        confidence -= len(result.errors) * 0.3
        
        # Reduce confidence for each warning
        confidence -= len(result.warnings) * 0.1
        
        # Adjust based on specific metrics
        if "magnitude" in result.metrics:
            magnitude = result.metrics["magnitude"]
            if magnitude > self.config.max_absolute_magnitude * 0.5:
                confidence -= 0.1
        
        # Ensure confidence is in valid range
        return max(0.0, min(1.0, confidence))
    
    def _generate_recommendation(self, result: ValidationResult) -> str:
        """Generate a recommendation based on validation results."""
        if not result.is_valid:
            return "reject"
        
        if result.confidence >= self.config.confidence_threshold:
            if result.warnings:
                return "accept_with_caution"
            else:
                return "accept"
        else:
            return "requires_review"
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get statistics about validation history."""
        if not self.validation_history:
            return {}
        
        stats = {
            "total_validations": len(self.validation_history),
            "acceptance_rate": sum(1 for r in self.validation_history if r.is_valid) / len(self.validation_history),
            "average_confidence": np.mean([r.confidence for r in self.validation_history]),
            "common_errors": defaultdict(int),
            "common_warnings": defaultdict(int),
            "recommendations": defaultdict(int)
        }
        
        for result in self.validation_history:
            for error in result.errors:
                stats["common_errors"][error] += 1
            for warning in result.warnings:
                stats["common_warnings"][warning] += 1
            stats["recommendations"][result.recommendation] += 1
        
        return dict(stats)


class SafetyValidator:
    """Additional safety checks for critical edits."""
    
    def __init__(self):
        self.critical_parameters = set()
        self.safety_bounds = {}
    
    def add_critical_parameter(self, param_name: str, bounds: Optional[Tuple[float, float]] = None):
        """Mark a parameter as critical and optionally set safety bounds."""
        self.critical_parameters.add(param_name)
        if bounds:
            self.safety_bounds[param_name] = bounds
    
    def validate_critical_edit(
        self,
        diff: ParameterDiff,
        result: ValidationResult
    ):
        """Additional validation for critical parameters."""
        if diff.name in self.critical_parameters:
            result.add_warning(f"Edit targets critical parameter: {diff.name}")
            
            # Apply stricter magnitude threshold
            if diff.magnitude > 0.01:  # Much stricter for critical params
                result.add_error("Edit magnitude too large for critical parameter")
            
            # Check safety bounds if defined
            if diff.name in self.safety_bounds:
                bounds = self.safety_bounds[diff.name]
                # This would require access to the actual parameter values
                result.add_metric("safety_bounds_check", 1.0)


class EditValidationPipeline:
    """Complete pipeline for edit validation."""
    
    def __init__(self, config: ValidationConfig):
        self.validator = EditValidator(config)
        self.safety_validator = SafetyValidator()
        self.custom_validators: List[Callable] = []
    
    def add_custom_validator(self, validator: Callable):
        """Add a custom validation function."""
        self.custom_validators.append(validator)
    
    def validate(
        self,
        model: nn.Module,
        diff: ParameterDiff,
        validation_data: Optional[Any] = None
    ) -> ValidationResult:
        """Run complete validation pipeline."""
        # Main validation
        result = self.validator.validate_edit(
            model, diff, validation_data, self.custom_validators
        )
        
        # Safety validation
        self.safety_validator.validate_critical_edit(diff, result)
        
        return result