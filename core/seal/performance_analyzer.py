"""Performance impact analyzer for SEAL architecture.

This module analyzes the impact of self-editing operations on model
performance across various metrics and tasks.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import numpy as np
import time
from collections import defaultdict, deque
import json
from pathlib import Path
import warnings

from .parameter_diff import ParameterDiff
from .edit_validation import ValidationResult


@dataclass
class PerformanceSnapshot:
    """A snapshot of model performance at a point in time."""
    timestamp: float
    metrics: Dict[str, float]
    model_state_hash: str
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Computational metrics
    inference_latency_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    throughput_tokens_per_sec: Optional[float] = None
    
    # Quality metrics
    perplexity: Optional[float] = None
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    
    # Task-specific metrics
    task_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class PerformanceImpact:
    """Analysis of performance impact from an edit."""
    edit_id: str
    parameter_name: str
    edit_magnitude: float
    
    # Before/after snapshots
    before_snapshot: PerformanceSnapshot
    after_snapshot: PerformanceSnapshot
    
    # Impact analysis
    metric_changes: Dict[str, float] = field(default_factory=dict)
    relative_changes: Dict[str, float] = field(default_factory=dict)
    significance_scores: Dict[str, float] = field(default_factory=dict)
    
    # Overall assessment
    overall_impact: float = 0.0  # Positive = improvement, negative = degradation
    impact_category: str = "neutral"  # "positive", "negative", "neutral", "mixed"
    confidence: float = 0.0
    
    # Recommendations
    recommendation: str = "monitor"  # "keep", "rollback", "monitor", "investigate"
    notes: str = ""


@dataclass
class PerformanceConfig:
    """Configuration for performance analysis."""
    # Metrics to track
    core_metrics: List[str] = field(default_factory=lambda: [
        "loss", "perplexity", "accuracy", "inference_latency_ms", "memory_usage_mb"
    ])
    
    # Significance thresholds
    min_significant_change: float = 0.01  # 1% change
    high_impact_threshold: float = 0.05   # 5% change
    critical_threshold: float = 0.10      # 10% change
    
    # Analysis settings
    baseline_window_size: int = 10  # Number of snapshots for baseline
    significance_test_alpha: float = 0.05
    enable_statistical_testing: bool = True
    
    # Performance bounds
    max_acceptable_latency_increase: float = 0.20  # 20% increase
    max_acceptable_memory_increase: float = 0.15   # 15% increase
    min_acceptable_accuracy: float = 0.90          # 90% of baseline


class PerformanceAnalyzer:
    """Analyzes performance impact of edits."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.baseline_snapshots: deque = deque(maxlen=config.baseline_window_size)
        self.performance_history: List[PerformanceSnapshot] = []
        self.impact_history: List[PerformanceImpact] = []
        
        # Metric evaluators
        self.metric_evaluators: Dict[str, Callable] = {}
        self._setup_default_evaluators()
    
    def _setup_default_evaluators(self):
        """Setup default metric evaluation functions."""
        self.metric_evaluators.update({
            "loss": self._evaluate_loss,
            "perplexity": self._evaluate_perplexity,
            "accuracy": self._evaluate_accuracy,
            "inference_latency_ms": self._evaluate_latency,
            "memory_usage_mb": self._evaluate_memory_usage,
            "throughput_tokens_per_sec": self._evaluate_throughput
        })
    
    def register_metric_evaluator(self, metric_name: str, evaluator: Callable):
        """Register a custom metric evaluator."""
        self.metric_evaluators[metric_name] = evaluator
    
    def capture_performance_snapshot(
        self,
        model: nn.Module,
        validation_data: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> PerformanceSnapshot:
        """Capture a comprehensive performance snapshot."""
        timestamp = time.time()
        metrics = {}
        
        # Evaluate all registered metrics
        for metric_name, evaluator in self.metric_evaluators.items():
            try:
                if metric_name in self.config.core_metrics:
                    value = evaluator(model, validation_data)
                    if value is not None:
                        metrics[metric_name] = value
            except Exception as e:
                warnings.warn(f"Failed to evaluate metric {metric_name}: {str(e)}")
        
        # Create model state hash for comparison
        model_state_hash = self._compute_model_hash(model)
        
        snapshot = PerformanceSnapshot(
            timestamp=timestamp,
            metrics=metrics,
            model_state_hash=model_state_hash,
            context=context or {},
            inference_latency_ms=metrics.get("inference_latency_ms"),
            memory_usage_mb=metrics.get("memory_usage_mb"),
            throughput_tokens_per_sec=metrics.get("throughput_tokens_per_sec"),
            perplexity=metrics.get("perplexity"),
            accuracy=metrics.get("accuracy"),
            loss=metrics.get("loss")
        )
        
        # Store snapshot
        self.performance_history.append(snapshot)
        if len(self.baseline_snapshots) < self.config.baseline_window_size:
            self.baseline_snapshots.append(snapshot)
        
        return snapshot
    
    def _compute_model_hash(self, model: nn.Module) -> str:
        """Compute a hash of the model state for comparison."""
        import hashlib
        
        # Simple hash based on parameter norms
        param_norms = []
        for param in model.parameters():
            param_norms.append(torch.norm(param).item())
        
        content = "_".join(f"{norm:.6f}" for norm in param_norms[:10])  # First 10 params
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def analyze_edit_impact(
        self,
        edit_id: str,
        parameter_name: str,
        edit_magnitude: float,
        before_snapshot: PerformanceSnapshot,
        after_snapshot: PerformanceSnapshot
    ) -> PerformanceImpact:
        """Analyze the performance impact of an edit."""
        # Calculate metric changes
        metric_changes = {}
        relative_changes = {}
        significance_scores = {}
        
        for metric_name in self.config.core_metrics:
            if (metric_name in before_snapshot.metrics and 
                metric_name in after_snapshot.metrics):
                
                before_value = before_snapshot.metrics[metric_name]
                after_value = after_snapshot.metrics[metric_name]
                
                # Absolute change
                change = after_value - before_value
                metric_changes[metric_name] = change
                
                # Relative change
                if before_value != 0:
                    relative_change = change / abs(before_value)
                    relative_changes[metric_name] = relative_change
                    
                    # Significance score
                    significance = self._calculate_significance(
                        metric_name, before_value, after_value
                    )
                    significance_scores[metric_name] = significance
        
        # Calculate overall impact
        overall_impact = self._calculate_overall_impact(relative_changes)
        
        # Determine impact category
        impact_category = self._categorize_impact(relative_changes, significance_scores)
        
        # Calculate confidence
        confidence = self._calculate_confidence(significance_scores)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            relative_changes, significance_scores, impact_category
        )
        
        impact = PerformanceImpact(
            edit_id=edit_id,
            parameter_name=parameter_name,
            edit_magnitude=edit_magnitude,
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
            metric_changes=metric_changes,
            relative_changes=relative_changes,
            significance_scores=significance_scores,
            overall_impact=overall_impact,
            impact_category=impact_category,
            confidence=confidence,
            recommendation=recommendation
        )
        
        self.impact_history.append(impact)
        return impact
    
    def _calculate_significance(
        self,
        metric_name: str,
        before_value: float,
        after_value: float
    ) -> float:
        """Calculate significance of a metric change."""
        if not self.baseline_snapshots:
            return 0.0
        
        # Get baseline values for this metric
        baseline_values = []
        for snapshot in self.baseline_snapshots:
            if metric_name in snapshot.metrics:
                baseline_values.append(snapshot.metrics[metric_name])
        
        if len(baseline_values) < 3:
            # Not enough baseline data
            return 0.0
        
        # Calculate baseline statistics
        baseline_mean = np.mean(baseline_values)
        baseline_std = np.std(baseline_values)
        
        if baseline_std == 0:
            return 0.0
        
        # Z-score of the change
        change = after_value - before_value
        z_score = abs(change) / baseline_std
        
        # Convert to significance score (0-1)
        significance = min(z_score / 3.0, 1.0)  # 3-sigma rule
        
        return significance
    
    def _calculate_overall_impact(self, relative_changes: Dict[str, float]) -> float:
        """Calculate overall impact score."""
        if not relative_changes:
            return 0.0
        
        # Weight different metrics
        metric_weights = {
            "loss": -1.0,        # Lower is better
            "perplexity": -1.0,  # Lower is better
            "accuracy": 1.0,     # Higher is better
            "inference_latency_ms": -0.5,  # Lower is better, but less important
            "memory_usage_mb": -0.3,       # Lower is better, but less important
            "throughput_tokens_per_sec": 0.5  # Higher is better
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric_name, change in relative_changes.items():
            weight = metric_weights.get(metric_name, 0.0)
            if weight != 0.0:
                weighted_sum += weight * change
                total_weight += abs(weight)
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def _categorize_impact(
        self,
        relative_changes: Dict[str, float],
        significance_scores: Dict[str, float]
    ) -> str:
        """Categorize the impact of an edit."""
        significant_changes = {
            metric: change for metric, change in relative_changes.items()
            if significance_scores.get(metric, 0) > 0.5
        }
        
        if not significant_changes:
            return "neutral"
        
        positive_count = sum(1 for change in significant_changes.values() if change > 0)
        negative_count = sum(1 for change in significant_changes.values() if change < 0)
        
        if positive_count > 0 and negative_count > 0:
            return "mixed"
        elif positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _calculate_confidence(self, significance_scores: Dict[str, float]) -> float:
        """Calculate confidence in the impact analysis."""
        if not significance_scores:
            return 0.0
        
        # Average significance across metrics
        avg_significance = np.mean(list(significance_scores.values()))
        
        # Adjust for number of metrics
        metric_count_factor = min(len(significance_scores) / 5.0, 1.0)
        
        confidence = avg_significance * metric_count_factor
        return confidence
    
    def _generate_recommendation(
        self,
        relative_changes: Dict[str, float],
        significance_scores: Dict[str, float],
        impact_category: str
    ) -> str:
        """Generate a recommendation based on the impact analysis."""
        # Check for critical degradations
        for metric, change in relative_changes.items():
            significance = significance_scores.get(metric, 0)
            
            if significance > 0.7:  # High significance
                if metric in ["inference_latency_ms", "memory_usage_mb"]:
                    if change > self.config.max_acceptable_latency_increase:
                        return "rollback"
                elif metric in ["accuracy", "loss", "perplexity"]:
                    if abs(change) > self.config.critical_threshold:
                        return "rollback"
        
        # General recommendations
        if impact_category == "negative":
            max_significance = max(significance_scores.values()) if significance_scores else 0
            if max_significance > 0.8:
                return "rollback"
            else:
                return "investigate"
        
        elif impact_category == "positive":
            return "keep"
        
        elif impact_category == "mixed":
            return "investigate"
        
        else:  # neutral
            return "monitor"
    
    def get_performance_trends(
        self,
        metric_names: Optional[List[str]] = None,
        time_window_hours: float = 24.0
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Get performance trends over time."""
        cutoff_time = time.time() - time_window_hours * 3600
        metric_names = metric_names or self.config.core_metrics
        
        trends = defaultdict(list)
        
        for snapshot in self.performance_history:
            if snapshot.timestamp > cutoff_time:
                for metric_name in metric_names:
                    if metric_name in snapshot.metrics:
                        trends[metric_name].append(
                            (snapshot.timestamp, snapshot.metrics[metric_name])
                        )
        
        return dict(trends)
    
    def get_impact_summary(
        self,
        time_window_hours: Optional[float] = None
    ) -> Dict[str, Any]:
        """Get summary of performance impacts."""
        cutoff_time = (time.time() - time_window_hours * 3600 
                      if time_window_hours else 0)
        
        relevant_impacts = [
            impact for impact in self.impact_history
            if impact.before_snapshot.timestamp > cutoff_time
        ]
        
        if not relevant_impacts:
            return {}
        
        summary = {
            "total_edits": len(relevant_impacts),
            "impact_categories": defaultdict(int),
            "recommendations": defaultdict(int),
            "avg_overall_impact": 0.0,
            "avg_confidence": 0.0,
            "most_impacted_parameters": defaultdict(list),
            "best_performing_edits": [],
            "worst_performing_edits": []
        }
        
        total_impact = 0.0
        total_confidence = 0.0
        
        for impact in relevant_impacts:
            summary["impact_categories"][impact.impact_category] += 1
            summary["recommendations"][impact.recommendation] += 1
            total_impact += impact.overall_impact
            total_confidence += impact.confidence
            
            summary["most_impacted_parameters"][impact.parameter_name].append(
                impact.overall_impact
            )
        
        summary["avg_overall_impact"] = total_impact / len(relevant_impacts)
        summary["avg_confidence"] = total_confidence / len(relevant_impacts)
        
        # Sort edits by performance
        sorted_impacts = sorted(relevant_impacts, key=lambda x: x.overall_impact)
        summary["worst_performing_edits"] = [
            {
                "edit_id": impact.edit_id,
                "parameter": impact.parameter_name,
                "impact": impact.overall_impact,
                "category": impact.impact_category
            }
            for impact in sorted_impacts[:5]
        ]
        
        summary["best_performing_edits"] = [
            {
                "edit_id": impact.edit_id,
                "parameter": impact.parameter_name,
                "impact": impact.overall_impact,
                "category": impact.impact_category
            }
            for impact in sorted_impacts[-5:]
        ]
        
        return dict(summary)
    
    # Default metric evaluators
    def _evaluate_loss(self, model: nn.Module, validation_data: Any) -> Optional[float]:
        """Evaluate model loss."""
        if validation_data is None:
            return None
        
        # This would need actual validation data and forward pass
        # Placeholder implementation
        return 1.0
    
    def _evaluate_perplexity(self, model: nn.Module, validation_data: Any) -> Optional[float]:
        """Evaluate model perplexity."""
        # Placeholder - would calculate actual perplexity
        return 10.0
    
    def _evaluate_accuracy(self, model: nn.Module, validation_data: Any) -> Optional[float]:
        """Evaluate model accuracy."""
        # Placeholder - would calculate actual accuracy
        return 0.85
    
    def _evaluate_latency(self, model: nn.Module, validation_data: Any) -> Optional[float]:
        """Evaluate inference latency."""
        # Simple latency measurement
        start_time = time.time()
        
        with torch.no_grad():
            # Dummy forward pass
            dummy_input = torch.randn(1, 100)  # Adjust size as needed
            try:
                _ = model(dummy_input)
                latency_ms = (time.time() - start_time) * 1000
                return latency_ms
            except:
                return None
    
    def _evaluate_memory_usage(self, model: nn.Module, validation_data: Any) -> Optional[float]:
        """Evaluate memory usage."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            # For CPU, this would need psutil or similar
            return None
    
    def _evaluate_throughput(self, model: nn.Module, validation_data: Any) -> Optional[float]:
        """Evaluate throughput."""
        # Placeholder - would measure actual tokens/sec
        return 100.0
    
    def save_analysis_state(self, filepath: Path):
        """Save analyzer state to disk."""
        state = {
            "config": {
                "core_metrics": self.config.core_metrics,
                "min_significant_change": self.config.min_significant_change,
                "high_impact_threshold": self.config.high_impact_threshold
            },
            "performance_history_count": len(self.performance_history),
            "impact_history_count": len(self.impact_history),
            "baseline_snapshots_count": len(self.baseline_snapshots)
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)