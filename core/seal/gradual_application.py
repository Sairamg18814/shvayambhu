"""Gradual LoRA application system for SEAL architecture.

This module provides safe, incremental application of LoRA adapters
with rollback capabilities and monitoring.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import time
import json
import numpy as np
from collections import defaultdict, deque
import threading
import warnings
from pathlib import Path
import sqlite3

from .lora_adapter import LoRALinear, LoRAEmbedding, LoRAConfig
from .memory_efficient_lora import MemoryEfficientLoRA
from .performance_analyzer import PerformanceAnalyzer, PerformanceSnapshot, PerformanceConfig
from .rollback_manager import RollbackManager, CheckpointConfig


@dataclass
class GradualApplicationConfig:
    """Configuration for gradual LoRA application."""
    # Application strategy
    strategy: str = "linear"  # "linear", "exponential", "staged", "adaptive"
    
    # Timing settings
    total_duration_minutes: float = 60.0  # Total time for full application
    application_steps: int = 10  # Number of discrete steps
    step_interval_minutes: float = 6.0  # Time between steps
    
    # Safety settings
    enable_monitoring: bool = True
    performance_threshold: float = 0.05  # 5% degradation threshold
    enable_auto_rollback: bool = True
    rollback_threshold: float = 0.10  # 10% degradation triggers rollback
    
    # Validation settings
    validation_interval_steps: int = 1  # Validate every N steps
    min_validation_samples: int = 50
    validation_timeout_minutes: float = 5.0
    
    # Application parameters
    min_application_ratio: float = 0.0
    max_application_ratio: float = 1.0
    warmup_steps: int = 2  # Initial steps with smaller increments
    
    # Monitoring settings
    monitor_metrics: List[str] = field(default_factory=lambda: [
        "loss", "accuracy", "latency", "memory_usage"
    ])
    
    # Recovery settings
    enable_partial_rollback: bool = True
    partial_rollback_ratio: float = 0.5  # Roll back to 50% if issues detected


@dataclass
class ApplicationStep:
    """A single step in gradual application."""
    step_number: int
    timestamp: float
    application_ratio: float
    
    # Performance before/after
    performance_before: Optional[PerformanceSnapshot] = None
    performance_after: Optional[PerformanceSnapshot] = None
    
    # Validation results
    validation_passed: bool = True
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)
    
    # Step details
    applied_adapters: List[str] = field(default_factory=list)
    step_duration_seconds: float = 0.0
    
    # Decision
    continue_application: bool = True
    rollback_triggered: bool = False
    notes: str = ""


@dataclass
class ApplicationResult:
    """Result of gradual application process."""
    success: bool
    final_application_ratio: float
    total_steps_completed: int
    total_duration_minutes: float
    
    # Performance summary
    initial_performance: Optional[PerformanceSnapshot] = None
    final_performance: Optional[PerformanceSnapshot] = None
    performance_improvement: Dict[str, float] = field(default_factory=dict)
    
    # Step history
    application_steps: List[ApplicationStep] = field(default_factory=list)
    
    # Issues encountered
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    rollbacks_performed: int = 0
    
    # Final state
    applied_adapters: Dict[str, float] = field(default_factory=dict)  # adapter -> application ratio
    recommendation: str = "continue"  # "continue", "rollback", "investigate"
    
    # Statistics
    average_step_duration_seconds: float = 0.0
    performance_stability_score: float = 1.0


class GradualApplicationManager:
    """Manages gradual application of LoRA adapters."""
    
    def __init__(
        self,
        model: nn.Module,
        config: GradualApplicationConfig,
        performance_analyzer: Optional[PerformanceAnalyzer] = None,
        rollback_manager: Optional[RollbackManager] = None
    ):
        self.model = model
        self.config = config
        
        # Setup performance monitoring
        if performance_analyzer is None:
            perf_config = PerformanceConfig(core_metrics=config.monitor_metrics)
            self.performance_analyzer = PerformanceAnalyzer(perf_config)
        else:
            self.performance_analyzer = performance_analyzer
        
        # Setup rollback capability
        if rollback_manager is None:
            rollback_config = CheckpointConfig(checkpoint_dir=Path("gradual_app_checkpoints"))
            self.rollback_manager = RollbackManager(rollback_config)
        else:
            self.rollback_manager = rollback_manager
        
        # Application state
        self.current_step = 0
        self.application_history: List[ApplicationStep] = []
        self.current_adapters: Dict[str, Union[LoRALinear, LoRAEmbedding]] = {}
        self.application_ratios: Dict[str, float] = {}
        
        # Safety monitoring
        self.baseline_performance: Optional[PerformanceSnapshot] = None
        self.performance_history: deque = deque(maxlen=100)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Application strategies
        self.strategies = {
            "linear": self._linear_strategy,
            "exponential": self._exponential_strategy,
            "staged": self._staged_strategy,
            "adaptive": self._adaptive_strategy
        }
    
    def apply_adapters_gradually(
        self,
        adapters: Dict[str, Union[LoRALinear, LoRAEmbedding]],
        validation_fn: Optional[Callable[[nn.Module], Dict[str, float]]] = None
    ) -> ApplicationResult:
        """Apply LoRA adapters gradually with monitoring and rollback."""
        start_time = time.time()
        
        # Initialize
        self.current_adapters = adapters.copy()
        self.application_ratios = {name: 0.0 for name in adapters.keys()}
        self.current_step = 0
        self.application_history = []
        
        # Create initial checkpoint
        initial_checkpoint = self.rollback_manager.create_checkpoint(
            self.model, "gradual_application_start"
        )
        
        # Capture baseline performance
        self.baseline_performance = self.performance_analyzer.capture_performance_snapshot(
            self.model, context={"phase": "baseline"}
        )
        
        # Get application schedule
        application_schedule = self._generate_application_schedule()
        
        print(f"Starting gradual application with {len(application_schedule)} steps")
        
        # Execute application steps
        result = ApplicationResult(
            success=True,
            final_application_ratio=0.0,
            total_steps_completed=0,
            total_duration_minutes=0.0,
            initial_performance=self.baseline_performance
        )
        
        try:
            for step_number, target_ratio in enumerate(application_schedule, 1):
                step_start_time = time.time()
                
                print(f"Step {step_number}/{len(application_schedule)}: "
                      f"Applying to {target_ratio:.2%}")
                
                # Create application step
                step = ApplicationStep(
                    step_number=step_number,
                    timestamp=step_start_time,
                    application_ratio=target_ratio
                )
                
                # Capture performance before step
                if self.config.enable_monitoring:
                    step.performance_before = self.performance_analyzer.capture_performance_snapshot(
                        self.model, context={"phase": "before_step", "step": step_number}
                    )
                
                # Apply the step
                applied_adapters = self._apply_step(target_ratio)
                step.applied_adapters = applied_adapters
                
                # Capture performance after step
                if self.config.enable_monitoring:
                    step.performance_after = self.performance_analyzer.capture_performance_snapshot(
                        self.model, context={"phase": "after_step", "step": step_number}
                    )
                
                # Validate step
                if validation_fn and step_number % self.config.validation_interval_steps == 0:
                    step.validation_metrics = validation_fn(self.model)
                    step.validation_passed = self._validate_step(step)
                
                # Check for issues
                step.continue_application = self._evaluate_step_safety(step)
                
                # Record step timing
                step.step_duration_seconds = time.time() - step_start_time
                
                # Store step
                self.application_history.append(step)
                result.application_steps.append(step)
                result.total_steps_completed = step_number
                
                # Check if we should continue
                if not step.continue_application:
                    if step.rollback_triggered:
                        result.success = False
                        result.errors.append(f"Application stopped at step {step_number} due to safety concerns")
                        
                        # Perform rollback
                        self._perform_rollback(initial_checkpoint)
                        result.rollbacks_performed += 1
                    
                    break
                
                # Wait before next step (except for last step)
                if step_number < len(application_schedule):
                    time.sleep(self.config.step_interval_minutes * 60)
        
        except Exception as e:
            result.success = False
            result.errors.append(f"Application failed with error: {str(e)}")
            
            # Emergency rollback
            self._perform_rollback(initial_checkpoint)
            result.rollbacks_performed += 1
        
        # Finalize result
        end_time = time.time()
        result.total_duration_minutes = (end_time - start_time) / 60
        result.final_application_ratio = max(self.application_ratios.values()) if self.application_ratios else 0.0
        result.applied_adapters = self.application_ratios.copy()
        
        # Final performance capture
        result.final_performance = self.performance_analyzer.capture_performance_snapshot(
            self.model, context={"phase": "final"}
        )
        
        # Calculate performance improvement
        if result.initial_performance and result.final_performance:
            result.performance_improvement = self._calculate_performance_improvement(
                result.initial_performance, result.final_performance
            )
        
        # Generate recommendation
        result.recommendation = self._generate_recommendation(result)
        
        # Calculate statistics
        if result.application_steps:
            result.average_step_duration_seconds = np.mean([
                step.step_duration_seconds for step in result.application_steps
            ])
            result.performance_stability_score = self._calculate_stability_score(result.application_steps)
        
        print(f"Gradual application completed: {result.success}, "
              f"final ratio: {result.final_application_ratio:.2%}")
        
        return result
    
    def _generate_application_schedule(self) -> List[float]:
        """Generate application schedule based on strategy."""
        strategy_fn = self.strategies.get(self.config.strategy, self._linear_strategy)
        return strategy_fn()
    
    def _linear_strategy(self) -> List[float]:
        """Linear application strategy."""
        steps = self.config.application_steps
        min_ratio = self.config.min_application_ratio
        max_ratio = self.config.max_application_ratio
        
        # Linear interpolation
        ratios = np.linspace(min_ratio, max_ratio, steps)
        
        # Apply warmup if configured
        if self.config.warmup_steps > 0:
            warmup_ratios = np.linspace(min_ratio, ratios[self.config.warmup_steps], 
                                       self.config.warmup_steps + 1)[:-1]
            ratios[:self.config.warmup_steps] = warmup_ratios
        
        return ratios.tolist()
    
    def _exponential_strategy(self) -> List[float]:
        """Exponential application strategy (slow start, fast finish)."""
        steps = self.config.application_steps
        min_ratio = self.config.min_application_ratio
        max_ratio = self.config.max_application_ratio
        
        # Exponential curve
        x = np.linspace(0, 1, steps)
        ratios = min_ratio + (max_ratio - min_ratio) * (np.exp(2 * x) - 1) / (np.exp(2) - 1)
        
        return ratios.tolist()
    
    def _staged_strategy(self) -> List[float]:
        """Staged application strategy (discrete jumps)."""
        steps = self.config.application_steps
        min_ratio = self.config.min_application_ratio
        max_ratio = self.config.max_application_ratio
        
        # Create stages with holds
        stage_size = max(1, steps // 4)  # 4 main stages
        ratios = []
        
        for stage in range(4):
            stage_ratio = min_ratio + (max_ratio - min_ratio) * (stage + 1) / 4
            
            # Hold at this ratio for multiple steps
            for _ in range(stage_size):
                ratios.append(stage_ratio)
                if len(ratios) >= steps:
                    break
            
            if len(ratios) >= steps:
                break
        
        # Ensure we have exactly the right number of steps
        while len(ratios) < steps:
            ratios.append(max_ratio)
        
        return ratios[:steps]
    
    def _adaptive_strategy(self) -> List[float]:
        """Adaptive strategy (adjusts based on observed performance)."""
        # Start with linear strategy, then adapt during execution
        return self._linear_strategy()
    
    def _apply_step(self, target_ratio: float) -> List[str]:
        """Apply a single step of the gradual application."""
        applied_adapters = []
        
        with self.lock:
            for adapter_name, adapter in self.current_adapters.items():
                # Update application ratio
                self.application_ratios[adapter_name] = target_ratio
                
                # Apply the ratio to the adapter
                if hasattr(adapter, 'set_application_ratio'):
                    adapter.set_application_ratio(target_ratio)
                else:
                    # For standard LoRA adapters, scale the parameters
                    self._scale_adapter_parameters(adapter, target_ratio)
                
                applied_adapters.append(adapter_name)
        
        return applied_adapters
    
    def _scale_adapter_parameters(self, adapter: Union[LoRALinear, LoRAEmbedding], ratio: float):
        """Scale adapter parameters by the given ratio."""
        if hasattr(adapter, '_original_scaling'):
            # Restore original scaling first
            adapter.scaling = adapter._original_scaling
        else:
            # Store original scaling
            adapter._original_scaling = getattr(adapter, 'scaling', 1.0)
        
        # Apply ratio scaling
        adapter.scaling = adapter._original_scaling * ratio
    
    def _validate_step(self, step: ApplicationStep) -> bool:
        """Validate a single application step."""
        if not step.validation_metrics:
            return True
        
        # Check each validation metric
        for metric_name, value in step.validation_metrics.items():
            if metric_name in self.config.monitor_metrics:
                # Compare with baseline
                if self.baseline_performance and metric_name in self.baseline_performance.metrics:
                    baseline_value = self.baseline_performance.metrics[metric_name]
                    
                    # Calculate relative change
                    if baseline_value != 0:
                        relative_change = (value - baseline_value) / baseline_value
                        
                        # Check for significant degradation
                        if relative_change < -self.config.performance_threshold:
                            step.validation_errors.append(
                                f"Metric {metric_name} degraded by {abs(relative_change):.2%}"
                            )
                            return False
        
        return True
    
    def _evaluate_step_safety(self, step: ApplicationStep) -> bool:
        """Evaluate if it's safe to continue after this step."""
        # Check validation results
        if not step.validation_passed:
            step.rollback_triggered = True
            step.notes = "Failed validation checks"
            return False
        
        # Check performance degradation
        if (step.performance_before and step.performance_after and 
            self.config.enable_monitoring):
            
            degradation = self._calculate_performance_degradation(
                step.performance_before, step.performance_after
            )
            
            if degradation > self.config.rollback_threshold:
                step.rollback_triggered = True
                step.notes = f"Performance degradation {degradation:.2%} exceeds threshold"
                return False
        
        # Check for adaptive strategy adjustments
        if self.config.strategy == "adaptive":
            # Adjust future steps based on current performance
            self._adapt_remaining_schedule(step)
        
        return True
    
    def _calculate_performance_degradation(
        self,
        before: PerformanceSnapshot,
        after: PerformanceSnapshot
    ) -> float:
        """Calculate performance degradation between snapshots."""
        degradations = []
        
        for metric in self.config.monitor_metrics:
            if metric in before.metrics and metric in after.metrics:
                before_value = before.metrics[metric]
                after_value = after.metrics[metric]
                
                if before_value != 0:
                    # Assume lower values are better for loss-like metrics
                    # and higher values are better for accuracy-like metrics
                    if metric in ["loss", "latency", "memory_usage"]:
                        # Lower is better
                        degradation = (after_value - before_value) / before_value
                    else:
                        # Higher is better
                        degradation = (before_value - after_value) / before_value
                    
                    degradations.append(max(0, degradation))  # Only consider degradations
        
        return max(degradations) if degradations else 0.0
    
    def _adapt_remaining_schedule(self, step: ApplicationStep):
        """Adapt remaining application schedule based on current performance."""
        # This is a placeholder for adaptive strategy implementation
        # In practice, this would analyze the step performance and adjust
        # the remaining application schedule accordingly
        pass
    
    def _perform_rollback(self, checkpoint_id: str) -> bool:
        """Perform rollback to a previous checkpoint."""
        try:
            success = self.rollback_manager.rollback_to_checkpoint(self.model, checkpoint_id)
            
            if success:
                # Reset application state
                with self.lock:
                    self.application_ratios = {name: 0.0 for name in self.current_adapters.keys()}
                
                print(f"Successfully rolled back to checkpoint {checkpoint_id}")
            else:
                print(f"Failed to rollback to checkpoint {checkpoint_id}")
            
            return success
            
        except Exception as e:
            print(f"Rollback failed: {str(e)}")
            return False
    
    def _calculate_performance_improvement(
        self,
        initial: PerformanceSnapshot,
        final: PerformanceSnapshot
    ) -> Dict[str, float]:
        """Calculate performance improvement between snapshots."""
        improvements = {}
        
        for metric in self.config.monitor_metrics:
            if metric in initial.metrics and metric in final.metrics:
                initial_value = initial.metrics[metric]
                final_value = final.metrics[metric]
                
                if initial_value != 0:
                    relative_change = (final_value - initial_value) / initial_value
                    improvements[metric] = relative_change
        
        return improvements
    
    def _generate_recommendation(self, result: ApplicationResult) -> str:
        """Generate recommendation based on application result."""
        if not result.success:
            return "investigate"
        
        if result.final_application_ratio >= 0.9:  # 90% or more applied
            return "continue"
        
        # Check performance improvement
        if result.performance_improvement:
            avg_improvement = np.mean(list(result.performance_improvement.values()))
            if avg_improvement > 0.02:  # 2% improvement
                return "continue"
            elif avg_improvement < -0.05:  # 5% degradation
                return "rollback"
        
        return "investigate"
    
    def _calculate_stability_score(self, steps: List[ApplicationStep]) -> float:
        """Calculate performance stability score across steps."""
        if len(steps) < 2:
            return 1.0
        
        # Look at performance variance across steps
        performance_changes = []
        
        for i in range(1, len(steps)):
            current_step = steps[i]
            previous_step = steps[i-1]
            
            if (current_step.performance_after and previous_step.performance_after):
                for metric in self.config.monitor_metrics:
                    if (metric in current_step.performance_after.metrics and
                        metric in previous_step.performance_after.metrics):
                        
                        current_value = current_step.performance_after.metrics[metric]
                        previous_value = previous_step.performance_after.metrics[metric]
                        
                        if previous_value != 0:
                            change = abs((current_value - previous_value) / previous_value)
                            performance_changes.append(change)
        
        if not performance_changes:
            return 1.0
        
        # Stability score: 1 - normalized variance
        variance = np.var(performance_changes)
        stability_score = max(0, 1 - variance * 10)  # Scale variance
        
        return stability_score
    
    def pause_application(self):
        """Pause the gradual application process."""
        # This would require modifying the main application loop
        # to check for pause conditions
        pass
    
    def resume_application(self):
        """Resume a paused application process."""
        # Implementation would depend on how pause is implemented
        pass
    
    def get_application_status(self) -> Dict[str, Any]:
        """Get current status of gradual application."""
        with self.lock:
            return {
                "current_step": self.current_step,
                "total_steps": self.config.application_steps,
                "current_ratios": self.application_ratios.copy(),
                "application_history_length": len(self.application_history),
                "baseline_captured": self.baseline_performance is not None,
                "performance_history_length": len(self.performance_history)
            }


# Utility functions

def create_gradual_application_plan(
    adapters: Dict[str, Union[LoRALinear, LoRAEmbedding]],
    target_performance_improvement: float = 0.05,
    max_risk_tolerance: float = 0.10,
    total_duration_hours: float = 1.0
) -> GradualApplicationConfig:
    """Create a gradual application plan based on requirements."""
    # Determine number of steps based on risk tolerance
    # Higher risk tolerance = fewer steps, lower = more steps
    min_steps = 5
    max_steps = 20
    
    risk_factor = max_risk_tolerance
    steps = int(min_steps + (max_steps - min_steps) * (1 - risk_factor))
    
    # Determine strategy based on adapter complexity
    adapter_count = len(adapters)
    if adapter_count > 10:
        strategy = "staged"  # Many adapters, use staged approach
    elif target_performance_improvement > 0.1:
        strategy = "exponential"  # High improvement expected, start slow
    else:
        strategy = "linear"  # Standard linear approach
    
    config = GradualApplicationConfig(
        strategy=strategy,
        total_duration_minutes=total_duration_hours * 60,
        application_steps=steps,
        step_interval_minutes=(total_duration_hours * 60) / steps,
        performance_threshold=max_risk_tolerance / 2,
        rollback_threshold=max_risk_tolerance,
        enable_auto_rollback=True,
        enable_monitoring=True
    )
    
    return config


def simulate_gradual_application(
    config: GradualApplicationConfig,
    performance_fn: Callable[[float], Dict[str, float]]
) -> ApplicationResult:
    """Simulate gradual application for testing purposes."""
    # This function simulates the gradual application process
    # without actually modifying a model, useful for testing
    
    start_time = time.time()
    steps = []
    
    # Generate application schedule
    manager = GradualApplicationManager(None, config)  # No model for simulation
    schedule = manager._generate_application_schedule()
    
    for step_number, ratio in enumerate(schedule, 1):
        step = ApplicationStep(
            step_number=step_number,
            timestamp=time.time(),
            application_ratio=ratio
        )
        
        # Simulate performance
        step.validation_metrics = performance_fn(ratio)
        step.validation_passed = True
        step.continue_application = True
        
        steps.append(step)
    
    # Create result
    result = ApplicationResult(
        success=True,
        final_application_ratio=schedule[-1] if schedule else 0.0,
        total_steps_completed=len(steps),
        total_duration_minutes=(time.time() - start_time) / 60,
        application_steps=steps
    )
    
    return result