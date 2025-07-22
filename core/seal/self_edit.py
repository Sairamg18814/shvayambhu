"""Self-Edit Generation for SEAL Architecture.

This module implements the self-edit generation system that allows
the model to autonomously generate parameter updates for improvement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from pathlib import Path

from .lora_adapter import LoRAAdapter, LoRAConfig
from ..blt.transformer import LatentTransformer

logger = logging.getLogger(__name__)


class EditType(Enum):
    """Types of self-edits that can be generated."""
    PARAMETER_UPDATE = "parameter_update"
    LAYER_SCALING = "layer_scaling"
    ATTENTION_REWEIGHTING = "attention_reweighting"
    ACTIVATION_TUNING = "activation_tuning"
    GRADIENT_MODIFICATION = "gradient_modification"
    LEARNING_RATE_ADAPTATION = "learning_rate_adaptation"


@dataclass
class EditTarget:
    """Target specification for a self-edit."""
    module_name: str
    parameter_name: str
    edit_type: EditType
    target_layers: Optional[List[int]] = None
    target_heads: Optional[List[int]] = None
    confidence: float = 0.0
    importance: float = 0.0


@dataclass
class PerformanceGap:
    """Identified performance gap requiring improvement."""
    task: str
    current_performance: float
    target_performance: float
    confidence: float
    difficulty: float
    sample_data: Optional[List[str]] = None
    error_patterns: Optional[List[str]] = None


@dataclass
class EditCandidate:
    """A candidate self-edit with metadata."""
    edit_id: str
    edit_type: EditType
    target: EditTarget
    magnitude: float
    direction: Tensor  # LoRA-style update direction
    confidence: float
    expected_improvement: float
    safety_score: float
    validation_loss: Optional[float] = None
    performance_metrics: Optional[Dict[str, float]] = None


class PerformanceAnalyzer:
    """Analyzes model performance to identify improvement opportunities."""
    
    def __init__(
        self,
        model: nn.Module,
        evaluation_tasks: List[str],
        min_confidence: float = 0.7
    ):
        self.model = model
        self.evaluation_tasks = evaluation_tasks
        self.min_confidence = min_confidence
        
        # Performance tracking
        self.performance_history = []
        self.error_patterns = {}
        self.improvement_opportunities = []
        
    def analyze_performance(
        self,
        validation_data: List[Dict[str, Any]],
        current_metrics: Dict[str, float]
    ) -> List[PerformanceGap]:
        """Analyze current performance and identify gaps."""
        gaps = []
        
        for task in self.evaluation_tasks:
            task_data = [d for d in validation_data if d.get('task') == task]
            if not task_data:
                continue
            
            # Analyze task-specific performance
            task_metrics = self._evaluate_task_performance(task_data, task)
            current_perf = task_metrics.get('accuracy', 0.0)
            
            # Identify performance targets
            target_perf = self._get_performance_target(task, current_perf)
            
            if target_perf > current_perf + 0.05:  # Significant gap
                gap = PerformanceGap(
                    task=task,
                    current_performance=current_perf,
                    target_performance=target_perf,
                    confidence=task_metrics.get('confidence', 0.0),
                    difficulty=self._estimate_improvement_difficulty(task, current_perf, target_perf),
                    sample_data=[d['text'] for d in task_data[:10]],
                    error_patterns=self._identify_error_patterns(task_data)
                )
                gaps.append(gap)
        
        # Store for future analysis
        self.improvement_opportunities = gaps
        return gaps
    
    def _evaluate_task_performance(
        self,
        task_data: List[Dict[str, Any]],
        task: str
    ) -> Dict[str, float]:
        """Evaluate performance on a specific task."""
        correct = 0
        total = 0
        confidences = []
        
        self.model.eval()
        with torch.no_grad():
            for item in task_data:
                # Get model prediction
                text = item['text']
                target = item.get('target', '')
                
                # Simple accuracy calculation (would be task-specific in practice)
                prediction = self._generate_prediction(text)
                is_correct = self._evaluate_prediction(prediction, target, task)
                
                if is_correct:
                    correct += 1
                total += 1
                
                # Estimate confidence (simplified)
                confidence = self._estimate_prediction_confidence(prediction, text)
                confidences.append(confidence)
        
        return {
            'accuracy': correct / max(total, 1),
            'confidence': np.mean(confidences) if confidences else 0.0,
            'total_samples': total
        }
    
    def _generate_prediction(self, text: str) -> str:
        """Generate prediction for a text input."""
        # Simplified prediction generation
        # In practice, this would use the full BLT pipeline
        try:
            if hasattr(self.model, 'generate'):
                output = self.model.generate(text, max_length=100)
                return output.get('generated_text', '')
            else:
                return text[:50]  # Fallback
        except Exception as e:
            logger.warning(f"Prediction generation failed: {e}")
            return ""
    
    def _evaluate_prediction(self, prediction: str, target: str, task: str) -> bool:
        """Evaluate if prediction is correct for the task."""
        if task == "summarization":
            # Check for key terms overlap
            pred_words = set(prediction.lower().split())
            target_words = set(target.lower().split())
            overlap = len(pred_words & target_words) / max(len(target_words), 1)
            return overlap > 0.3
        elif task == "classification":
            return prediction.strip().lower() == target.strip().lower()
        elif task == "generation":
            # Check for reasonable length and coherence
            return len(prediction) > 10 and not prediction.count(' ') == 0
        else:
            # Default string similarity
            return prediction.lower() in target.lower() or target.lower() in prediction.lower()
    
    def _estimate_prediction_confidence(self, prediction: str, input_text: str) -> float:
        """Estimate confidence in a prediction."""
        # Simplified confidence estimation
        if not prediction:
            return 0.0
        
        # Check for repetition (low confidence indicator)
        words = prediction.split()
        if len(words) > 1 and len(set(words)) / len(words) < 0.5:
            return 0.3
        
        # Check for reasonable length
        if len(prediction) < 5 or len(prediction) > len(input_text) * 3:
            return 0.4
        
        # Default moderate confidence
        return 0.7
    
    def _get_performance_target(self, task: str, current_perf: float) -> float:
        """Get target performance for a task."""
        targets = {
            "summarization": 0.8,
            "classification": 0.9,
            "generation": 0.75,
            "reasoning": 0.85,
            "qa": 0.88
        }
        
        base_target = targets.get(task, 0.8)
        
        # Adaptive target based on current performance
        if current_perf < 0.3:
            return min(base_target, current_perf + 0.2)
        elif current_perf < 0.6:
            return min(base_target, current_perf + 0.15)
        else:
            return min(base_target, current_perf + 0.1)
    
    def _estimate_improvement_difficulty(
        self,
        task: str,
        current_perf: float,
        target_perf: float
    ) -> float:
        """Estimate difficulty of achieving the performance improvement."""
        gap = target_perf - current_perf
        
        # Larger gaps are generally harder
        gap_difficulty = min(gap * 2, 1.0)
        
        # Some tasks are inherently harder to improve
        task_difficulty = {
            "reasoning": 0.9,
            "math": 0.95,
            "summarization": 0.7,
            "classification": 0.5,
            "generation": 0.6
        }.get(task, 0.7)
        
        # High performance improvements are harder
        perf_difficulty = 0.5 + current_perf * 0.5
        
        return min((gap_difficulty + task_difficulty + perf_difficulty) / 3, 1.0)
    
    def _identify_error_patterns(self, task_data: List[Dict[str, Any]]) -> List[str]:
        """Identify common error patterns in failed predictions."""
        patterns = []
        
        # Simple pattern detection (would be more sophisticated in practice)
        error_types = {
            "repetition": 0,
            "truncation": 0,
            "irrelevant": 0,
            "factual_error": 0
        }
        
        for item in task_data[:20]:  # Sample
            prediction = self._generate_prediction(item['text'])
            
            # Check for repetition
            words = prediction.split()
            if len(words) > 1 and len(set(words)) / len(words) < 0.6:
                error_types["repetition"] += 1
            
            # Check for truncation
            if len(prediction) < 10:
                error_types["truncation"] += 1
            
            # Check for irrelevance (simple heuristic)
            input_words = set(item['text'].lower().split())
            pred_words = set(prediction.lower().split())
            if len(input_words & pred_words) / max(len(input_words), 1) < 0.1:
                error_types["irrelevant"] += 1
        
        # Convert to patterns
        total_samples = min(len(task_data), 20)
        for error_type, count in error_types.items():
            if count > total_samples * 0.3:  # 30% threshold
                patterns.append(error_type)
        
        return patterns


class EditGenerator:
    """Generates self-edit candidates based on performance analysis."""
    
    def __init__(
        self,
        model: nn.Module,
        lora_config: LoRAConfig,
        safety_threshold: float = 0.8
    ):
        self.model = model
        self.lora_config = lora_config
        self.safety_threshold = safety_threshold
        
        # Edit generation parameters
        self.edit_magnitude_range = (0.001, 0.1)
        self.max_edits_per_gap = 3
        
        # Gradient-based analysis
        self.gradient_analyzer = GradientAnalyzer(model)
        
    def generate_edit_candidates(
        self,
        performance_gaps: List[PerformanceGap],
        model_state: Dict[str, Tensor]
    ) -> List[EditCandidate]:
        """Generate edit candidates to address performance gaps."""
        all_candidates = []
        
        for gap in performance_gaps:
            candidates = self._generate_candidates_for_gap(gap, model_state)
            all_candidates.extend(candidates)
        
        # Rank candidates by expected improvement and safety
        ranked_candidates = self._rank_candidates(all_candidates)
        
        return ranked_candidates[:20]  # Top 20 candidates
    
    def _generate_candidates_for_gap(
        self,
        gap: PerformanceGap,
        model_state: Dict[str, Tensor]
    ) -> List[EditCandidate]:
        """Generate edit candidates for a specific performance gap."""
        candidates = []
        
        # Analyze gradients for this gap
        gradient_info = self.gradient_analyzer.analyze_gradients_for_gap(gap)
        
        # Generate different types of edits
        for edit_type in [EditType.PARAMETER_UPDATE, EditType.ATTENTION_REWEIGHTING]:
            targets = self._identify_edit_targets(gap, edit_type, gradient_info)
            
            for target in targets:
                candidate = self._create_edit_candidate(gap, edit_type, target, model_state)
                if candidate and candidate.safety_score >= self.safety_threshold:
                    candidates.append(candidate)
        
        return candidates
    
    def _identify_edit_targets(
        self,
        gap: PerformanceGap,
        edit_type: EditType,
        gradient_info: Dict[str, Any]
    ) -> List[EditTarget]:
        """Identify specific targets for edits."""
        targets = []
        
        if edit_type == EditType.PARAMETER_UPDATE:
            # Target high-gradient parameters
            high_grad_modules = gradient_info.get('high_gradient_modules', [])
            for module_name in high_grad_modules[:5]:  # Top 5
                target = EditTarget(
                    module_name=module_name,
                    parameter_name="weight",
                    edit_type=edit_type,
                    confidence=gradient_info.get('confidence', 0.5),
                    importance=gradient_info.get('importance', 0.5)
                )
                targets.append(target)
        
        elif edit_type == EditType.ATTENTION_REWEIGHTING:
            # Target attention layers
            attention_layers = gradient_info.get('attention_layers', [])
            for layer_idx in attention_layers:
                target = EditTarget(
                    module_name=f"transformer.layers.{layer_idx}.attention",
                    parameter_name="out_proj.weight",
                    edit_type=edit_type,
                    target_layers=[layer_idx],
                    confidence=0.6,
                    importance=0.7
                )
                targets.append(target)
        
        return targets
    
    def _create_edit_candidate(
        self,
        gap: PerformanceGap,
        edit_type: EditType,
        target: EditTarget,
        model_state: Dict[str, Tensor]
    ) -> Optional[EditCandidate]:
        """Create a specific edit candidate."""
        try:
            # Get target parameter
            param_key = f"{target.module_name}.{target.parameter_name}"
            if param_key not in model_state:
                return None
            
            param = model_state[param_key]
            
            # Generate edit direction (LoRA-style)
            rank = self.lora_config.rank
            direction = self._generate_edit_direction(param, rank, gap)
            
            # Calculate magnitude
            magnitude = self._calculate_edit_magnitude(gap, target)
            
            # Estimate safety and improvement
            safety_score = self._estimate_safety(direction, magnitude, param)
            expected_improvement = self._estimate_improvement(gap, edit_type, magnitude)
            
            # Create edit ID
            edit_content = f"{target.module_name}:{edit_type.value}:{magnitude:.4f}"
            edit_id = hashlib.md5(edit_content.encode()).hexdigest()[:8]
            
            candidate = EditCandidate(
                edit_id=edit_id,
                edit_type=edit_type,
                target=target,
                magnitude=magnitude,
                direction=direction,
                confidence=target.confidence,
                expected_improvement=expected_improvement,
                safety_score=safety_score
            )
            
            return candidate
            
        except Exception as e:
            logger.warning(f"Failed to create edit candidate: {e}")
            return None
    
    def _generate_edit_direction(
        self,
        param: Tensor,
        rank: int,
        gap: PerformanceGap
    ) -> Tensor:
        """Generate the direction vector for the edit."""
        # Create a random direction scaled by parameter statistics
        param_std = param.std().item()
        param_mean = param.mean().item()
        
        # Generate rank-factorized direction (LoRA style)
        if len(param.shape) >= 2:
            in_features, out_features = param.shape[-2], param.shape[-1]
            
            # Low-rank decomposition
            a = torch.randn(in_features, rank) * param_std * 0.1
            b = torch.randn(rank, out_features) * param_std * 0.1
            direction = torch.matmul(a, b)
            
            # Reshape to match parameter shape
            if len(param.shape) > 2:
                direction = direction.view(param.shape)
        else:
            # For 1D parameters
            direction = torch.randn_like(param) * param_std * 0.1
        
        # Bias direction based on gap characteristics
        if gap.task == "reasoning":
            direction *= 1.2  # Slightly larger updates for reasoning
        elif gap.difficulty > 0.8:
            direction *= 0.8  # Smaller updates for difficult improvements
        
        return direction
    
    def _calculate_edit_magnitude(self, gap: PerformanceGap, target: EditTarget) -> float:
        """Calculate appropriate magnitude for the edit."""
        base_magnitude = 0.01  # Base 1% change
        
        # Scale by performance gap
        gap_scale = min((gap.target_performance - gap.current_performance) * 2, 1.0)
        
        # Scale by confidence
        confidence_scale = target.confidence
        
        # Scale by difficulty (larger changes for harder problems)
        difficulty_scale = 0.5 + gap.difficulty * 0.5
        
        magnitude = base_magnitude * gap_scale * confidence_scale * difficulty_scale
        
        # Clamp to reasonable range
        return max(self.edit_magnitude_range[0], 
                  min(self.edit_magnitude_range[1], magnitude))
    
    def _estimate_safety(self, direction: Tensor, magnitude: float, param: Tensor) -> float:
        """Estimate safety of applying this edit."""
        # Check magnitude relative to parameter scale
        param_norm = param.norm().item()
        direction_norm = direction.norm().item()
        
        if param_norm == 0:
            return 0.0
        
        relative_change = (direction_norm * magnitude) / param_norm
        
        # Safe if change is small relative to parameter
        if relative_change < 0.01:
            return 0.9
        elif relative_change < 0.05:
            return 0.8
        elif relative_change < 0.1:
            return 0.7
        else:
            return 0.5  # Potentially unsafe
    
    def _estimate_improvement(
        self,
        gap: PerformanceGap,
        edit_type: EditType,
        magnitude: float
    ) -> float:
        """Estimate expected improvement from this edit."""
        # Base improvement based on edit type effectiveness
        type_effectiveness = {
            EditType.PARAMETER_UPDATE: 0.7,
            EditType.ATTENTION_REWEIGHTING: 0.6,
            EditType.LAYER_SCALING: 0.5,
            EditType.ACTIVATION_TUNING: 0.4
        }.get(edit_type, 0.5)
        
        # Scale by magnitude
        magnitude_factor = min(magnitude * 10, 1.0)  # Assume linear up to 10% change
        
        # Scale by gap confidence
        confidence_factor = gap.confidence
        
        # Diminishing returns for large current performance
        perf_factor = 1.0 - gap.current_performance * 0.5
        
        improvement = (
            type_effectiveness * magnitude_factor * 
            confidence_factor * perf_factor * 
            (gap.target_performance - gap.current_performance)
        )
        
        return min(improvement, gap.target_performance - gap.current_performance)
    
    def _rank_candidates(self, candidates: List[EditCandidate]) -> List[EditCandidate]:
        """Rank edit candidates by potential value."""
        def score_candidate(candidate: EditCandidate) -> float:
            return (
                candidate.expected_improvement * 0.4 +
                candidate.safety_score * 0.3 +
                candidate.confidence * 0.2 +
                candidate.target.importance * 0.1
            )
        
        return sorted(candidates, key=score_candidate, reverse=True)


class GradientAnalyzer:
    """Analyzes gradients to identify improvement opportunities."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradient_history = {}
        
    def analyze_gradients_for_gap(self, gap: PerformanceGap) -> Dict[str, Any]:
        """Analyze gradients relevant to a performance gap."""
        # Simplified gradient analysis
        # In practice, this would require running backprop on gap-specific data
        
        high_gradient_modules = []
        attention_layers = []
        
        # Identify modules with high gradients (simulated)
        for name, module in self.model.named_modules():
            if "transformer" in name and any(p.requires_grad for p in module.parameters()):
                # Simulate gradient analysis
                if "attention" in name:
                    layer_num = self._extract_layer_number(name)
                    if layer_num is not None:
                        attention_layers.append(layer_num)
                
                # Simulate high gradient detection
                if hash(name + gap.task) % 3 == 0:  # Pseudo-random selection
                    high_gradient_modules.append(name)
        
        return {
            'high_gradient_modules': high_gradient_modules[:10],
            'attention_layers': attention_layers[:5],
            'confidence': 0.7,
            'importance': 0.6
        }
    
    def _extract_layer_number(self, module_name: str) -> Optional[int]:
        """Extract layer number from module name."""
        parts = module_name.split('.')
        for part in parts:
            try:
                return int(part)
            except ValueError:
                continue
        return None


def create_self_edit_system(
    model: nn.Module,
    evaluation_tasks: List[str],
    lora_config: LoRAConfig
) -> Tuple[PerformanceAnalyzer, EditGenerator]:
    """Create a complete self-edit system."""
    
    analyzer = PerformanceAnalyzer(
        model=model,
        evaluation_tasks=evaluation_tasks,
        min_confidence=0.7
    )
    
    generator = EditGenerator(
        model=model,
        lora_config=lora_config,
        safety_threshold=0.8
    )
    
    return analyzer, generator