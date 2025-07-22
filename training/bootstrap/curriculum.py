"""Curriculum Learning Scheduler for Bootstrap Training.

This module implements progressive curriculum learning strategies that
gradually increase training difficulty during bootstrap learning.
"""

import math
import random
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

import torch
from torch import Tensor

from .data_filters import QualityMetrics
from ...core.blt.entropy import calculate_byte_entropy

logger = logging.getLogger(__name__)


class CurriculumStrategy(Enum):
    """Available curriculum learning strategies."""
    LENGTH_BASED = "length_based"
    ENTROPY_BASED = "entropy_based"
    QUALITY_BASED = "quality_based"
    DIFFICULTY_BASED = "difficulty_based"
    COMPETENCY_BASED = "competency_based"
    MIXED = "mixed"


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    strategy: CurriculumStrategy = CurriculumStrategy.MIXED
    
    # General settings
    total_steps: int = 100000
    warmup_steps: int = 5000
    initial_difficulty: float = 0.2
    final_difficulty: float = 1.0
    
    # Length-based curriculum
    min_length: int = 64
    max_length: int = 1024
    length_progression: str = "linear"  # linear, exponential, step
    
    # Entropy-based curriculum
    min_entropy: float = 0.1
    max_entropy: float = 0.9
    entropy_progression: str = "linear"
    
    # Quality-based curriculum
    min_quality: float = 0.5
    max_quality: float = 1.0
    quality_progression: str = "linear"
    
    # Competency-based settings
    competency_window: int = 1000
    competency_threshold: float = 0.8
    adaptation_rate: float = 0.1
    
    # Mixed strategy weights
    strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        "length": 0.3,
        "entropy": 0.3,
        "quality": 0.2,
        "difficulty": 0.2
    })
    
    # Advanced settings
    enable_adaptive_pacing: bool = True
    enable_review_cycles: bool = True
    review_frequency: int = 10000
    difficulty_smoothing: float = 0.1


@dataclass
class DifficultyMetrics:
    """Metrics for assessing training difficulty."""
    length_score: float = 0.0
    entropy_score: float = 0.0
    quality_score: float = 0.0
    overall_difficulty: float = 0.0
    
    # Model performance indicators
    loss: float = float('inf')
    accuracy: float = 0.0
    perplexity: float = float('inf')
    
    # Training dynamics
    gradient_norm: float = 0.0
    learning_rate: float = 0.0
    convergence_rate: float = 0.0


class ProgressionScheduler:
    """Handles different progression schedules."""
    
    @staticmethod
    def linear(step: int, total_steps: int, start: float, end: float) -> float:
        """Linear progression from start to end."""
        progress = min(step / total_steps, 1.0)
        return start + progress * (end - start)
    
    @staticmethod
    def exponential(step: int, total_steps: int, start: float, end: float, rate: float = 2.0) -> float:
        """Exponential progression."""
        progress = min(step / total_steps, 1.0)
        exp_progress = (math.exp(rate * progress) - 1) / (math.exp(rate) - 1)
        return start + exp_progress * (end - start)
    
    @staticmethod
    def logarithmic(step: int, total_steps: int, start: float, end: float) -> float:
        """Logarithmic progression (fast initial growth, then slow)."""
        progress = min(step / total_steps, 1.0)
        if progress == 0:
            return start
        log_progress = math.log(1 + progress) / math.log(2)
        return start + log_progress * (end - start)
    
    @staticmethod
    def step_function(step: int, total_steps: int, start: float, end: float, num_steps: int = 5) -> float:
        """Step function progression."""
        progress = min(step / total_steps, 1.0)
        step_size = 1.0 / num_steps
        step_level = int(progress / step_size)
        step_progress = min(step_level / num_steps, 1.0)
        return start + step_progress * (end - start)
    
    @staticmethod
    def cosine_annealing(step: int, total_steps: int, start: float, end: float) -> float:
        """Cosine annealing schedule."""
        progress = min(step / total_steps, 1.0)
        cos_progress = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        return start + cos_progress * (end - start)


class CurriculumScheduler:
    """Main curriculum learning scheduler."""
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.current_step = 0
        self.scheduler = ProgressionScheduler()
        
        # Performance tracking
        self.performance_history = []
        self.difficulty_history = []
        self.competency_scores = []
        
        # Adaptive pacing
        self.adaptive_multiplier = 1.0
        self.last_review_step = 0
        
        logger.info(f"Initialized curriculum scheduler with strategy: {config.strategy.value}")
    
    def step(self, performance_metrics: Optional[DifficultyMetrics] = None) -> None:
        """Advance the curriculum by one step."""
        self.current_step += 1
        
        # Track performance
        if performance_metrics:
            self.performance_history.append(performance_metrics)
            self._update_competency_scores(performance_metrics)
        
        # Adaptive pacing adjustment
        if self.config.enable_adaptive_pacing and performance_metrics:
            self._adjust_pacing(performance_metrics)
        
        # Review cycles
        if (self.config.enable_review_cycles and 
            self.current_step - self.last_review_step >= self.config.review_frequency):
            self._trigger_review_cycle()
    
    def get_current_difficulty(self) -> float:
        """Get current overall difficulty level."""
        if self.current_step < self.config.warmup_steps:
            # During warmup, start with very low difficulty
            warmup_progress = self.current_step / self.config.warmup_steps
            return self.config.initial_difficulty * warmup_progress
        
        # Adjust step based on adaptive pacing
        effective_step = self.current_step * self.adaptive_multiplier
        effective_total = self.config.total_steps * self.adaptive_multiplier
        
        base_difficulty = self.scheduler.linear(
            int(effective_step - self.config.warmup_steps),
            int(effective_total - self.config.warmup_steps),
            self.config.initial_difficulty,
            self.config.final_difficulty
        )
        
        # Apply smoothing
        if self.difficulty_history:
            smoothed_difficulty = (
                (1 - self.config.difficulty_smoothing) * base_difficulty +
                self.config.difficulty_smoothing * self.difficulty_history[-1]
            )
        else:
            smoothed_difficulty = base_difficulty
        
        self.difficulty_history.append(smoothed_difficulty)
        return smoothed_difficulty
    
    def get_length_range(self) -> Tuple[int, int]:
        """Get current sequence length range."""
        if self.config.strategy in [CurriculumStrategy.LENGTH_BASED, CurriculumStrategy.MIXED]:
            difficulty = self.get_current_difficulty()
            
            if self.config.length_progression == "linear":
                max_len = self.scheduler.linear(
                    self.current_step, self.config.total_steps,
                    self.config.min_length, self.config.max_length
                )
            elif self.config.length_progression == "exponential":
                max_len = self.scheduler.exponential(
                    self.current_step, self.config.total_steps,
                    self.config.min_length, self.config.max_length
                )
            else:  # step
                max_len = self.scheduler.step_function(
                    self.current_step, self.config.total_steps,
                    self.config.min_length, self.config.max_length
                )
            
            # Ensure minimum length is always available
            min_len = min(self.config.min_length, int(max_len * 0.5))
            
            return int(min_len), int(max_len)
        
        return self.config.min_length, self.config.max_length
    
    def get_entropy_range(self) -> Tuple[float, float]:
        """Get current entropy range."""
        if self.config.strategy in [CurriculumStrategy.ENTROPY_BASED, CurriculumStrategy.MIXED]:
            if self.config.entropy_progression == "linear":
                max_entropy = self.scheduler.linear(
                    self.current_step, self.config.total_steps,
                    self.config.min_entropy, self.config.max_entropy
                )
            else:  # logarithmic - start with high entropy, gradually reduce
                max_entropy = self.scheduler.logarithmic(
                    self.current_step, self.config.total_steps,
                    self.config.max_entropy, self.config.min_entropy
                )
            
            min_entropy = min(self.config.min_entropy, max_entropy * 0.5)
            return min_entropy, max_entropy
        
        return self.config.min_entropy, self.config.max_entropy
    
    def get_quality_threshold(self) -> float:
        """Get current quality threshold."""
        if self.config.strategy in [CurriculumStrategy.QUALITY_BASED, CurriculumStrategy.MIXED]:
            if self.config.quality_progression == "linear":
                return self.scheduler.linear(
                    self.current_step, self.config.total_steps,
                    self.config.max_quality, self.config.min_quality  # Start high, go low
                )
            else:  # exponential
                return self.scheduler.exponential(
                    self.current_step, self.config.total_steps,
                    self.config.max_quality, self.config.min_quality
                )
        
        return self.config.min_quality
    
    def should_include_sample(
        self,
        text: str,
        byte_sequence: bytes,
        quality_metrics: Optional[QualityMetrics] = None
    ) -> bool:
        """Determine if a sample should be included at current curriculum level."""
        
        # Always include during warmup
        if self.current_step < self.config.warmup_steps:
            return True
        
        # Calculate sample difficulty
        sample_difficulty = self._calculate_sample_difficulty(
            text, byte_sequence, quality_metrics
        )
        
        # Get current difficulty threshold
        current_difficulty = self.get_current_difficulty()
        
        # Include sample if its difficulty is within acceptable range
        # Allow some samples below current level for stability
        min_acceptable = max(0.0, current_difficulty - 0.2)
        max_acceptable = min(1.0, current_difficulty + 0.1)
        
        return min_acceptable <= sample_difficulty <= max_acceptable
    
    def _calculate_sample_difficulty(
        self,
        text: str,
        byte_sequence: bytes,
        quality_metrics: Optional[QualityMetrics] = None
    ) -> float:
        """Calculate overall difficulty score for a sample."""
        scores = {}
        weights = self.config.strategy_weights
        
        # Length-based difficulty
        length = len(byte_sequence)
        length_difficulty = (length - self.config.min_length) / (
            self.config.max_length - self.config.min_length
        )
        scores["length"] = max(0.0, min(1.0, length_difficulty))
        
        # Entropy-based difficulty
        entropy = calculate_byte_entropy(byte_sequence)
        entropy_difficulty = (entropy - self.config.min_entropy) / (
            self.config.max_entropy - self.config.min_entropy
        )
        scores["entropy"] = max(0.0, min(1.0, entropy_difficulty))
        
        # Quality-based difficulty (inverse - lower quality = higher difficulty)
        if quality_metrics:
            quality_difficulty = 1.0 - quality_metrics.overall_quality
        else:
            quality_difficulty = 0.5  # Default moderate difficulty
        scores["quality"] = quality_difficulty
        
        # Structural difficulty
        difficulty_indicators = [
            len(text.split()) > 100,  # Long documents
            text.count('\n') > 20,    # Many paragraphs
            any(ord(c) > 127 for c in text[:100]),  # Non-ASCII characters
            text.count('"') > 10,     # Quoted content
            text.count('(') + text.count('[') > 10,  # Complex structure
        ]
        structural_difficulty = sum(difficulty_indicators) / len(difficulty_indicators)
        scores["difficulty"] = structural_difficulty
        
        # Weighted combination
        total_score = sum(
            scores.get(key, 0.0) * weight 
            for key, weight in weights.items()
        )
        
        return max(0.0, min(1.0, total_score))
    
    def _update_competency_scores(self, metrics: DifficultyMetrics) -> None:
        """Update competency tracking."""
        # Simple competency based on loss reduction
        if len(self.performance_history) > 1:
            prev_loss = self.performance_history[-2].loss
            current_loss = metrics.loss
            
            if prev_loss > 0:
                improvement = max(0.0, (prev_loss - current_loss) / prev_loss)
                competency = min(1.0, improvement * 10)  # Scale to 0-1
            else:
                competency = 0.5
        else:
            competency = 0.5  # Default
        
        self.competency_scores.append(competency)
        
        # Keep only recent scores
        if len(self.competency_scores) > self.config.competency_window:
            self.competency_scores.pop(0)
    
    def _adjust_pacing(self, metrics: DifficultyMetrics) -> None:
        """Adjust pacing based on performance."""
        if len(self.competency_scores) < 10:  # Need some history
            return
        
        recent_competency = np.mean(self.competency_scores[-10:])
        
        if recent_competency > self.config.competency_threshold:
            # Performing well, can accelerate
            self.adaptive_multiplier = min(2.0, self.adaptive_multiplier + self.config.adaptation_rate)
        elif recent_competency < self.config.competency_threshold * 0.5:
            # Struggling, slow down
            self.adaptive_multiplier = max(0.5, self.adaptive_multiplier - self.config.adaptation_rate)
        
        logger.debug(f"Adjusted pacing multiplier to {self.adaptive_multiplier:.3f} "
                    f"(competency: {recent_competency:.3f})")
    
    def _trigger_review_cycle(self) -> None:
        """Trigger a review cycle for previously seen easier material."""
        self.last_review_step = self.current_step
        
        # Temporarily reduce difficulty for review
        self.adaptive_multiplier *= 0.7
        
        logger.info(f"Triggered review cycle at step {self.current_step}")
    
    def get_curriculum_stats(self) -> Dict[str, Any]:
        """Get comprehensive curriculum statistics."""
        current_difficulty = self.get_current_difficulty()
        min_len, max_len = self.get_length_range()
        min_ent, max_ent = self.get_entropy_range()
        quality_thresh = self.get_quality_threshold()
        
        stats = {
            "current_step": self.current_step,
            "current_difficulty": current_difficulty,
            "adaptive_multiplier": self.adaptive_multiplier,
            "length_range": (min_len, max_len),
            "entropy_range": (min_ent, max_ent),
            "quality_threshold": quality_thresh,
            "competency_scores": {
                "mean": np.mean(self.competency_scores) if self.competency_scores else 0.0,
                "recent": np.mean(self.competency_scores[-10:]) if len(self.competency_scores) >= 10 else 0.0,
                "trend": self._calculate_competency_trend()
            },
            "strategy": self.config.strategy.value,
            "progress": min(self.current_step / self.config.total_steps, 1.0)
        }
        
        return stats
    
    def _calculate_competency_trend(self) -> str:
        """Calculate competency trend."""
        if len(self.competency_scores) < 20:
            return "insufficient_data"
        
        recent = np.mean(self.competency_scores[-10:])
        older = np.mean(self.competency_scores[-20:-10])
        
        if recent > older + 0.1:
            return "improving"
        elif recent < older - 0.1:
            return "declining"
        else:
            return "stable"
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save curriculum state."""
        checkpoint = {
            "current_step": self.current_step,
            "adaptive_multiplier": self.adaptive_multiplier,
            "last_review_step": self.last_review_step,
            "competency_scores": self.competency_scores,
            "difficulty_history": self.difficulty_history,
            "config": self.config.__dict__
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved curriculum checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load curriculum state."""
        checkpoint = torch.load(filepath)
        
        self.current_step = checkpoint["current_step"]
        self.adaptive_multiplier = checkpoint["adaptive_multiplier"]
        self.last_review_step = checkpoint["last_review_step"]
        self.competency_scores = checkpoint["competency_scores"]
        self.difficulty_history = checkpoint["difficulty_history"]
        
        logger.info(f"Loaded curriculum checkpoint from {filepath}")


class CurriculumDataFilter:
    """Filter that applies curriculum learning criteria to data selection."""
    
    def __init__(self, scheduler: CurriculumScheduler):
        self.scheduler = scheduler
        self.stats = {
            "total_seen": 0,
            "total_accepted": 0,
            "difficulty_distribution": [0] * 10  # 10 bins for difficulty
        }
    
    def filter_sample(
        self,
        text: str,
        byte_sequence: bytes,
        quality_metrics: Optional[QualityMetrics] = None
    ) -> bool:
        """Filter sample based on current curriculum level."""
        self.stats["total_seen"] += 1
        
        should_include = self.scheduler.should_include_sample(
            text, byte_sequence, quality_metrics
        )
        
        if should_include:
            self.stats["total_accepted"] += 1
            
            # Track difficulty distribution
            difficulty = self.scheduler._calculate_sample_difficulty(
                text, byte_sequence, quality_metrics
            )
            bin_idx = min(9, int(difficulty * 10))
            self.stats["difficulty_distribution"][bin_idx] += 1
        
        return should_include
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get filtering statistics."""
        acceptance_rate = (
            self.stats["total_accepted"] / max(self.stats["total_seen"], 1)
        )
        
        return {
            "total_seen": self.stats["total_seen"],
            "total_accepted": self.stats["total_accepted"],
            "acceptance_rate": acceptance_rate,
            "difficulty_distribution": self.stats["difficulty_distribution"],
            "curriculum_stats": self.scheduler.get_curriculum_stats()
        }


def create_curriculum_scheduler(
    strategy: str = "mixed",
    total_steps: int = 100000,
    **kwargs
) -> CurriculumScheduler:
    """Create a curriculum scheduler with default configuration."""
    
    config = CurriculumConfig(
        strategy=CurriculumStrategy(strategy),
        total_steps=total_steps,
        **kwargs
    )
    
    return CurriculumScheduler(config)


def create_adaptive_curriculum(
    model_performance_fn: Callable[[], DifficultyMetrics],
    **kwargs
) -> CurriculumScheduler:
    """Create an adaptive curriculum that responds to model performance."""
    
    config = CurriculumConfig(
        strategy=CurriculumStrategy.COMPETENCY_BASED,
        enable_adaptive_pacing=True,
        enable_review_cycles=True,
        **kwargs
    )
    
    scheduler = CurriculumScheduler(config)
    
    # This would typically be integrated into the training loop
    # to automatically adjust based on model performance
    
    return scheduler