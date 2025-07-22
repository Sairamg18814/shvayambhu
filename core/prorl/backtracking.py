"""Advanced backtracking mechanism for ProRL reasoning correction.

This module implements sophisticated backtracking strategies for correcting
reasoning paths when they lead to dead ends or low-quality conclusions.
"""

import torch
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid
import numpy as np
from collections import defaultdict, deque

from .reasoning_engine import ReasoningStep, ReasoningTrace, StepType, ReasoningStrategy
from .reward_model import ProcessReward, RewardType


class BacktrackReason(Enum):
    """Reasons for initiating backtracking."""
    LOW_QUALITY = "low_quality"
    DEAD_END = "dead_end"
    CONTRADICTION = "contradiction"
    INSUFFICIENT_PROGRESS = "insufficient_progress"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SAFETY_CONCERN = "safety_concern"
    COHERENCE_BREAK = "coherence_break"
    REDUNDANT_PATH = "redundant_path"


class BacktrackStrategy(Enum):
    """Strategies for backtracking."""
    IMMEDIATE_PARENT = "immediate_parent"
    LOWEST_CONFIDENCE = "lowest_confidence"
    BRANCH_POINT = "branch_point"
    QUALITY_THRESHOLD = "quality_threshold"
    PROGRESSIVE = "progressive"
    BEAM_SEARCH = "beam_search"
    MONTE_CARLO = "monte_carlo"


@dataclass
class BacktrackPoint:
    """A point in the reasoning trace suitable for backtracking."""
    step_id: str
    step: ReasoningStep
    confidence: float
    alternatives_available: int
    depth: int
    backtrack_cost: float
    potential_gain: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class BacktrackDecision:
    """A decision to backtrack to a specific point."""
    target_step_id: str
    reason: BacktrackReason
    strategy: BacktrackStrategy
    confidence: float
    expected_improvement: float
    cost_estimate: float
    alternative_paths: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktrackResult:
    """Result of a backtracking operation."""
    success: bool
    target_step_id: str
    steps_removed: int
    new_path_generated: bool
    improvement_achieved: float
    time_taken_ms: float
    new_alternatives: List[ReasoningStep] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class BacktrackingConfig:
    """Configuration for backtracking behavior."""
    # Quality thresholds for backtracking
    min_step_quality: float = 0.3
    min_progress_rate: float = 0.1
    max_stagnation_steps: int = 5
    
    # Backtracking strategy preferences
    preferred_strategy: BacktrackStrategy = BacktrackStrategy.PROGRESSIVE
    max_backtrack_depth: int = 10
    max_backtrack_attempts: int = 3
    
    # Resource limits
    max_alternatives_per_point: int = 5
    backtrack_timeout_seconds: float = 30.0
    
    # Learning parameters
    adaptation_rate: float = 0.1
    success_weight: float = 0.7
    cost_weight: float = 0.3
    
    # Safety settings
    prevent_infinite_loops: bool = True
    loop_detection_window: int = 20
    max_repeated_backtracks: int = 2


class AdvancedBacktrackingManager:
    """Advanced backtracking manager with multiple strategies."""
    
    def __init__(self, config: BacktrackingConfig):
        self.config = config
        
        # Backtracking history and statistics
        self.backtrack_history: List[BacktrackDecision] = []
        self.success_rates: Dict[BacktrackStrategy, float] = defaultdict(float)
        self.strategy_performance: Dict[BacktrackStrategy, List[float]] = defaultdict(list)
        
        # Active backtrack points
        self.available_points: Dict[str, BacktrackPoint] = {}
        self.visited_states: Set[str] = set()
        
        # Loop detection
        self.recent_backtracks: deque = deque(maxlen=config.loop_detection_window)
        self.repeated_backtrack_count: defaultdict = defaultdict(int)
        
        # Strategy implementations
        self.strategies = {
            BacktrackStrategy.IMMEDIATE_PARENT: self._immediate_parent_backtrack,
            BacktrackStrategy.LOWEST_CONFIDENCE: self._lowest_confidence_backtrack,
            BacktrackStrategy.BRANCH_POINT: self._branch_point_backtrack,
            BacktrackStrategy.QUALITY_THRESHOLD: self._quality_threshold_backtrack,
            BacktrackStrategy.PROGRESSIVE: self._progressive_backtrack,
            BacktrackStrategy.BEAM_SEARCH: self._beam_search_backtrack,
            BacktrackStrategy.MONTE_CARLO: self._monte_carlo_backtrack
        }
    
    def should_backtrack(
        self,
        current_step: ReasoningStep,
        trace: ReasoningTrace,
        recent_rewards: List[ProcessReward]
    ) -> Tuple[bool, BacktrackReason]:
        """Determine if backtracking should be initiated."""
        
        # Check quality threshold
        if current_step.confidence < self.config.min_step_quality:
            return True, BacktrackReason.LOW_QUALITY
        
        # Check for dead ends (no valid next steps)
        if self._is_dead_end(current_step, trace):
            return True, BacktrackReason.DEAD_END
        
        # Check for contradictions with previous steps
        if self._detect_contradiction(current_step, trace):
            return True, BacktrackReason.CONTRADICTION
        
        # Check progress rate
        if self._insufficient_progress(trace, recent_rewards):
            return True, BacktrackReason.INSUFFICIENT_PROGRESS
        
        # Check for safety concerns
        if self._safety_concern_detected(current_step, recent_rewards):
            return True, BacktrackReason.SAFETY_CONCERN
        
        # Check for coherence breaks
        if self._coherence_break_detected(current_step, trace, recent_rewards):
            return True, BacktrackReason.COHERENCE_BREAK
        
        # Check for redundant paths
        if self._redundant_path_detected(current_step, trace):
            return True, BacktrackReason.REDUNDANT_PATH
        
        return False, None
    
    def plan_backtrack(
        self,
        trace: ReasoningTrace,
        reason: BacktrackReason,
        recent_rewards: List[ProcessReward]
    ) -> Optional[BacktrackDecision]:
        """Plan the optimal backtracking strategy."""
        
        # Identify potential backtrack points
        backtrack_points = self._identify_backtrack_points(trace, recent_rewards)
        
        if not backtrack_points:
            return None
        
        # Select strategy based on reason and context
        strategy = self._select_backtrack_strategy(reason, trace, backtrack_points)
        
        # Execute strategy to find target point
        target_point = self.strategies[strategy](trace, backtrack_points, reason)
        
        if not target_point:
            return None
        
        # Calculate expected improvement and cost
        expected_improvement = self._estimate_improvement(target_point, trace, recent_rewards)
        cost_estimate = self._estimate_backtrack_cost(target_point, trace)
        
        # Create backtrack decision
        decision = BacktrackDecision(
            target_step_id=target_point.step_id,
            reason=reason,
            strategy=strategy,
            confidence=self._calculate_decision_confidence(target_point, expected_improvement, cost_estimate),
            expected_improvement=expected_improvement,
            cost_estimate=cost_estimate,
            alternative_paths=self._generate_alternative_paths(target_point, trace),
            metadata={
                "target_depth": target_point.depth,
                "alternatives_available": target_point.alternatives_available,
                "timestamp": time.time()
            }
        )
        
        return decision
    
    def execute_backtrack(
        self,
        decision: BacktrackDecision,
        trace: ReasoningTrace
    ) -> BacktrackResult:
        """Execute the backtracking decision."""
        start_time = time.time()
        
        try:
            # Prevent infinite loops
            if self.config.prevent_infinite_loops:
                if self._would_create_loop(decision, trace):
                    return BacktrackResult(
                        success=False,
                        target_step_id=decision.target_step_id,
                        steps_removed=0,
                        new_path_generated=False,
                        improvement_achieved=0.0,
                        time_taken_ms=0.0,
                        error_message="Would create infinite loop"
                    )
            
            # Find target step
            target_step = trace.step_index.get(decision.target_step_id)
            if not target_step:
                return BacktrackResult(
                    success=False,
                    target_step_id=decision.target_step_id,
                    steps_removed=0,
                    new_path_generated=False,
                    improvement_achieved=0.0,
                    time_taken_ms=(time.time() - start_time) * 1000,
                    error_message="Target step not found"
                )
            
            # Remove steps after target
            steps_to_remove = self._find_steps_to_remove(target_step, trace)
            original_step_count = len(trace.steps)
            
            # Perform removal
            self._remove_steps(steps_to_remove, trace)
            steps_removed = original_step_count - len(trace.steps)
            
            # Generate new alternatives
            new_alternatives = self._generate_new_alternatives(target_step, trace, decision)
            
            # Update trace statistics
            trace.backtrack_count += 1
            
            # Record backtrack in history
            self.backtrack_history.append(decision)
            self.recent_backtracks.append(decision.target_step_id)
            self.repeated_backtrack_count[decision.target_step_id] += 1
            
            # Calculate improvement (placeholder - would need actual evaluation)
            improvement_achieved = max(0.0, decision.expected_improvement * 0.8)  # Conservative estimate
            
            return BacktrackResult(
                success=True,
                target_step_id=decision.target_step_id,
                steps_removed=steps_removed,
                new_path_generated=len(new_alternatives) > 0,
                improvement_achieved=improvement_achieved,
                time_taken_ms=(time.time() - start_time) * 1000,
                new_alternatives=new_alternatives
            )
            
        except Exception as e:
            return BacktrackResult(
                success=False,
                target_step_id=decision.target_step_id,
                steps_removed=0,
                new_path_generated=False,
                improvement_achieved=0.0,
                time_taken_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    def _is_dead_end(self, step: ReasoningStep, trace: ReasoningTrace) -> bool:
        """Check if a step represents a dead end."""
        # Check if it's a conclusion with low confidence
        if step.step_type == StepType.CONCLUSION and step.confidence < 0.5:
            return True
        
        # Check if it has been stuck at this step type for too long
        recent_steps = trace.steps[-self.config.max_stagnation_steps:]
        same_type_count = sum(1 for s in recent_steps if s.step_type == step.step_type)
        
        return same_type_count >= self.config.max_stagnation_steps
    
    def _detect_contradiction(self, step: ReasoningStep, trace: ReasoningTrace) -> bool:
        """Detect contradictions with previous reasoning."""
        # Simple contradiction detection based on step content
        # In practice, this would use semantic analysis
        
        step_content_lower = step.content.lower()
        contradiction_indicators = ["however", "but", "contradicts", "opposite", "not true"]
        
        if any(indicator in step_content_lower for indicator in contradiction_indicators):
            # Check if this contradicts recent high-confidence steps
            recent_confident_steps = [
                s for s in trace.steps[-5:]
                if s.confidence > 0.7 and s.step_id != step.step_id
            ]
            
            if recent_confident_steps:
                return True
        
        return False
    
    def _insufficient_progress(
        self,
        trace: ReasoningTrace,
        recent_rewards: List[ProcessReward]
    ) -> bool:
        """Check if progress rate is insufficient."""
        if len(recent_rewards) < 3:
            return False
        
        # Calculate average progress component
        progress_scores = [
            r.component_rewards.get(RewardType.PROGRESS, None)
            for r in recent_rewards[-5:]
        ]
        progress_scores = [p.score for p in progress_scores if p is not None]
        
        if not progress_scores:
            return False
        
        avg_progress = sum(progress_scores) / len(progress_scores)
        return avg_progress < self.config.min_progress_rate
    
    def _safety_concern_detected(
        self,
        step: ReasoningStep,
        recent_rewards: List[ProcessReward]
    ) -> bool:
        """Detect safety concerns in reasoning."""
        if not recent_rewards:
            return False
        
        latest_reward = recent_rewards[-1]
        safety_signal = latest_reward.component_rewards.get(RewardType.SAFETY)
        
        return safety_signal and safety_signal.score < 0.3
    
    def _coherence_break_detected(
        self,
        step: ReasoningStep,
        trace: ReasoningTrace,
        recent_rewards: List[ProcessReward]
    ) -> bool:
        """Detect breaks in reasoning coherence."""
        if not recent_rewards:
            return False
        
        # Check coherence score
        latest_reward = recent_rewards[-1]
        coherence_signal = latest_reward.component_rewards.get(RewardType.COHERENCE)
        
        if coherence_signal and coherence_signal.score < 0.4:
            return True
        
        # Check for abrupt topic changes
        if len(trace.steps) >= 2:
            prev_step = trace.steps[-2]
            # Simple heuristic for topic change (would use embeddings in practice)
            content_overlap = self._calculate_content_overlap(step.content, prev_step.content)
            return content_overlap < 0.2
        
        return False
    
    def _redundant_path_detected(self, step: ReasoningStep, trace: ReasoningTrace) -> bool:
        """Detect if the current path is redundant."""
        # Check for repeated step types in sequence
        if len(trace.steps) >= 3:
            recent_types = [s.step_type for s in trace.steps[-3:]]
            if len(set(recent_types)) == 1:  # All same type
                return True
        
        # Check for similar content to previous steps
        step_content_lower = step.content.lower()
        for prev_step in trace.steps[:-1]:
            if self._calculate_content_overlap(step_content_lower, prev_step.content.lower()) > 0.8:
                return True
        
        return False
    
    def _calculate_content_overlap(self, content1: str, content2: str) -> float:
        """Calculate content overlap between two strings."""
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _identify_backtrack_points(
        self,
        trace: ReasoningTrace,
        recent_rewards: List[ProcessReward]
    ) -> List[BacktrackPoint]:
        """Identify potential points for backtracking."""
        points = []
        
        # Consider steps with multiple potential alternatives
        for i, step in enumerate(trace.steps):
            if step.step_type in [StepType.ANALYSIS, StepType.HYPOTHESIS, StepType.BRANCH]:
                # Calculate alternatives available
                alternatives = self._count_available_alternatives(step, trace)
                
                if alternatives > 0:
                    # Calculate depth
                    depth = self._calculate_step_depth(step, trace)
                    
                    # Estimate backtrack cost and potential gain
                    cost = self._estimate_step_backtrack_cost(step, trace)
                    gain = self._estimate_step_potential_gain(step, trace, recent_rewards)
                    
                    point = BacktrackPoint(
                        step_id=step.step_id,
                        step=step,
                        confidence=step.confidence,
                        alternatives_available=alternatives,
                        depth=depth,
                        backtrack_cost=cost,
                        potential_gain=gain
                    )
                    
                    points.append(point)
        
        # Sort by potential value (gain/cost ratio)
        points.sort(key=lambda p: p.potential_gain / max(p.backtrack_cost, 0.1), reverse=True)
        
        return points[:self.config.max_alternatives_per_point]
    
    def _select_backtrack_strategy(
        self,
        reason: BacktrackReason,
        trace: ReasoningTrace,
        points: List[BacktrackPoint]
    ) -> BacktrackStrategy:
        """Select optimal backtracking strategy based on context."""
        
        # Strategy selection based on reason
        if reason == BacktrackReason.LOW_QUALITY:
            return BacktrackStrategy.QUALITY_THRESHOLD
        elif reason == BacktrackReason.DEAD_END:
            return BacktrackStrategy.IMMEDIATE_PARENT
        elif reason == BacktrackReason.CONTRADICTION:
            return BacktrackStrategy.BRANCH_POINT
        elif reason == BacktrackReason.INSUFFICIENT_PROGRESS:
            return BacktrackStrategy.PROGRESSIVE
        elif reason == BacktrackReason.COHERENCE_BREAK:
            return BacktrackStrategy.LOWEST_CONFIDENCE
        else:
            # Use adaptive strategy selection based on historical performance
            return self._select_adaptive_strategy()
    
    def _select_adaptive_strategy(self) -> BacktrackStrategy:
        """Select strategy based on historical performance."""
        if not self.strategy_performance:
            return self.config.preferred_strategy
        
        # Calculate average performance for each strategy
        strategy_scores = {}
        for strategy, performances in self.strategy_performance.items():
            if performances:
                strategy_scores[strategy] = sum(performances) / len(performances)
        
        if strategy_scores:
            best_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
            return best_strategy
        
        return self.config.preferred_strategy
    
    # Strategy implementations
    def _immediate_parent_backtrack(
        self,
        trace: ReasoningTrace,
        points: List[BacktrackPoint],
        reason: BacktrackReason
    ) -> Optional[BacktrackPoint]:
        """Backtrack to immediate parent of current step."""
        if not trace.steps:
            return None
        
        current_step = trace.steps[-1]
        if current_step.parent_step_id:
            parent_step = trace.step_index.get(current_step.parent_step_id)
            if parent_step:
                # Find corresponding backtrack point
                for point in points:
                    if point.step_id == parent_step.step_id:
                        return point
        
        return None
    
    def _lowest_confidence_backtrack(
        self,
        trace: ReasoningTrace,
        points: List[BacktrackPoint],
        reason: BacktrackReason
    ) -> Optional[BacktrackPoint]:
        """Backtrack to the step with lowest confidence."""
        if not points:
            return None
        
        return min(points, key=lambda p: p.confidence)
    
    def _branch_point_backtrack(
        self,
        trace: ReasoningTrace,
        points: List[BacktrackPoint],
        reason: BacktrackReason
    ) -> Optional[BacktrackPoint]:
        """Backtrack to a branching point with alternatives."""
        branch_points = [p for p in points if p.alternatives_available > 1]
        
        if not branch_points:
            return None
        
        # Select branch point with best potential
        return max(branch_points, key=lambda p: p.potential_gain / max(p.backtrack_cost, 0.1))
    
    def _quality_threshold_backtrack(
        self,
        trace: ReasoningTrace,
        points: List[BacktrackPoint],
        reason: BacktrackReason
    ) -> Optional[BacktrackPoint]:
        """Backtrack to first step below quality threshold."""
        quality_points = [p for p in points if p.confidence < self.config.min_step_quality]
        
        if quality_points:
            # Return the earliest low-quality point
            return min(quality_points, key=lambda p: p.depth)
        
        return None
    
    def _progressive_backtrack(
        self,
        trace: ReasoningTrace,
        points: List[BacktrackPoint],
        reason: BacktrackReason
    ) -> Optional[BacktrackPoint]:
        """Progressive backtracking strategy."""
        if not points:
            return None
        
        # Start with shallow backtracks and progress deeper
        max_attempts = self.repeated_backtrack_count.get(points[0].step_id, 0)
        target_depth = min(max_attempts + 1, self.config.max_backtrack_depth)
        
        # Find point at appropriate depth
        depth_points = [p for p in points if p.depth <= target_depth]
        
        if depth_points:
            return max(depth_points, key=lambda p: p.potential_gain)
        
        return points[0] if points else None
    
    def _beam_search_backtrack(
        self,
        trace: ReasoningTrace,
        points: List[BacktrackPoint],
        reason: BacktrackReason
    ) -> Optional[BacktrackPoint]:
        """Beam search backtracking strategy."""
        if not points:
            return None
        
        # Select top candidates based on potential value
        beam_size = min(3, len(points))
        top_points = sorted(points, key=lambda p: p.potential_gain, reverse=True)[:beam_size]
        
        # Select best among top candidates
        return max(top_points, key=lambda p: p.confidence * p.potential_gain)
    
    def _monte_carlo_backtrack(
        self,
        trace: ReasoningTrace,
        points: List[BacktrackPoint],
        reason: BacktrackReason
    ) -> Optional[BacktrackPoint]:
        """Monte Carlo backtracking strategy."""
        if not points:
            return None
        
        # Weighted random selection based on potential gain
        weights = [max(p.potential_gain, 0.1) for p in points]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return points[0]
        
        # Normalize weights
        normalized_weights = [w / total_weight for w in weights]
        
        # Random selection
        choice = np.random.choice(len(points), p=normalized_weights)
        return points[choice]
    
    def _count_available_alternatives(self, step: ReasoningStep, trace: ReasoningTrace) -> int:
        """Count available alternatives for a reasoning step."""
        # Simplified implementation - would analyze actual reasoning possibilities
        if step.step_type == StepType.ANALYSIS:
            return 3  # Multiple analysis approaches
        elif step.step_type == StepType.HYPOTHESIS:
            return 2  # Alternative hypotheses
        elif step.step_type == StepType.BRANCH:
            return 4  # Branching point
        else:
            return 1  # Single path forward
    
    def _calculate_step_depth(self, step: ReasoningStep, trace: ReasoningTrace) -> int:
        """Calculate depth of step in reasoning trace."""
        depth = 0
        current_step = step
        
        while current_step.parent_step_id:
            depth += 1
            current_step = trace.step_index.get(current_step.parent_step_id)
            if not current_step:
                break
        
        return depth
    
    def _estimate_step_backtrack_cost(self, step: ReasoningStep, trace: ReasoningTrace) -> float:
        """Estimate cost of backtracking to a specific step."""
        # Cost based on steps that would be lost
        steps_after = [s for s in trace.steps if s.step_number > step.step_number]
        
        # Cost factors: number of steps, quality of lost steps, depth
        step_count_cost = len(steps_after) * 0.1
        quality_cost = sum(s.confidence for s in steps_after) * 0.05
        depth_cost = self._calculate_step_depth(step, trace) * 0.02
        
        return step_count_cost + quality_cost + depth_cost
    
    def _estimate_step_potential_gain(
        self,
        step: ReasoningStep,
        trace: ReasoningTrace,
        recent_rewards: List[ProcessReward]
    ) -> float:
        """Estimate potential gain from backtracking to a step."""
        # Base gain from alternatives available
        base_gain = self._count_available_alternatives(step, trace) * 0.2
        
        # Gain from step quality
        quality_gain = step.confidence * 0.3
        
        # Gain from potential progress improvement
        progress_gain = 0.5 if step.step_type in [StepType.ANALYSIS, StepType.HYPOTHESIS] else 0.2
        
        return base_gain + quality_gain + progress_gain
    
    def _estimate_improvement(
        self,
        point: BacktrackPoint,
        trace: ReasoningTrace,
        recent_rewards: List[ProcessReward]
    ) -> float:
        """Estimate expected improvement from backtracking."""
        return point.potential_gain - (point.backtrack_cost * 0.5)
    
    def _estimate_backtrack_cost(self, point: BacktrackPoint, trace: ReasoningTrace) -> float:
        """Estimate total cost of backtracking operation."""
        return point.backtrack_cost
    
    def _calculate_decision_confidence(
        self,
        point: BacktrackPoint,
        improvement: float,
        cost: float
    ) -> float:
        """Calculate confidence in backtracking decision."""
        if cost == 0:
            return 1.0
        
        benefit_ratio = improvement / cost
        base_confidence = point.confidence
        
        # Combine factors
        confidence = (base_confidence + min(benefit_ratio, 1.0)) / 2.0
        return max(0.0, min(1.0, confidence))
    
    def _generate_alternative_paths(self, point: BacktrackPoint, trace: ReasoningTrace) -> List[str]:
        """Generate alternative reasoning paths from backtrack point."""
        alternatives = []
        
        # Generate based on step type
        if point.step.step_type == StepType.ANALYSIS:
            alternatives.extend([
                "alternative_analysis_1",
                "alternative_analysis_2",
                "deeper_analysis"
            ])
        elif point.step.step_type == StepType.HYPOTHESIS:
            alternatives.extend([
                "alternative_hypothesis_1",
                "alternative_hypothesis_2"
            ])
        elif point.step.step_type == StepType.BRANCH:
            alternatives.extend([
                "branch_path_1",
                "branch_path_2",
                "branch_path_3"
            ])
        
        return alternatives[:point.alternatives_available]
    
    def _would_create_loop(self, decision: BacktrackDecision, trace: ReasoningTrace) -> bool:
        """Check if backtracking would create an infinite loop."""
        target_id = decision.target_step_id
        
        # Check recent backtrack history
        recent_targets = [bt for bt in self.recent_backtracks if bt == target_id]
        
        # Too many recent backtracks to same target
        if len(recent_targets) >= self.config.max_repeated_backtracks:
            return True
        
        # Check for alternating pattern
        if len(self.recent_backtracks) >= 4:
            recent_list = list(self.recent_backtracks)[-4:]
            if recent_list[0] == recent_list[2] and recent_list[1] == recent_list[3]:
                return True
        
        return False
    
    def _find_steps_to_remove(self, target_step: ReasoningStep, trace: ReasoningTrace) -> List[ReasoningStep]:
        """Find all steps that should be removed during backtracking."""
        steps_to_remove = []
        
        for step in trace.steps:
            if step.step_number > target_step.step_number:
                steps_to_remove.append(step)
        
        return steps_to_remove
    
    def _remove_steps(self, steps_to_remove: List[ReasoningStep], trace: ReasoningTrace):
        """Remove steps from the trace."""
        for step in steps_to_remove:
            if step in trace.steps:
                trace.steps.remove(step)
            
            if step.step_id in trace.step_index:
                del trace.step_index[step.step_id]
            
            # Update parent-child relationships
            if step.parent_step_id:
                parent = trace.step_index.get(step.parent_step_id)
                if parent and step.step_id in parent.child_step_ids:
                    parent.child_step_ids.remove(step.step_id)
        
        # Update trace statistics
        trace.total_steps = len(trace.steps)
    
    def _generate_new_alternatives(
        self,
        target_step: ReasoningStep,
        trace: ReasoningTrace,
        decision: BacktrackDecision
    ) -> List[ReasoningStep]:
        """Generate new alternative steps from backtrack point."""
        alternatives = []
        
        # Create alternative steps based on the target step
        for i, alt_path in enumerate(decision.alternative_paths):
            step_id = str(uuid.uuid4())
            alt_step = ReasoningStep(
                step_id=step_id,
                step_number=target_step.step_number + 1,
                step_type=StepType.ANALYSIS if target_step.step_type == StepType.ANALYSIS else StepType.HYPOTHESIS,
                content=f"Alternative approach {i+1}: {alt_path}",
                confidence=0.6,  # Moderate initial confidence
                timestamp=time.time(),
                parent_step_id=target_step.step_id
            )
            
            alternatives.append(alt_step)
        
        return alternatives[:self.config.max_alternatives_per_point]
    
    def get_backtracking_statistics(self) -> Dict[str, Any]:
        """Get comprehensive backtracking statistics."""
        if not self.backtrack_history:
            return {"message": "No backtracking history available"}
        
        total_backtracks = len(self.backtrack_history)
        strategy_counts = defaultdict(int)
        reason_counts = defaultdict(int)
        
        for decision in self.backtrack_history:
            strategy_counts[decision.strategy.value] += 1
            reason_counts[decision.reason.value] += 1
        
        return {
            "total_backtracks": total_backtracks,
            "strategy_distribution": dict(strategy_counts),
            "reason_distribution": dict(reason_counts),
            "success_rates": dict(self.success_rates),
            "average_confidence": sum(d.confidence for d in self.backtrack_history) / total_backtracks,
            "average_improvement": sum(d.expected_improvement for d in self.backtrack_history) / total_backtracks,
            "repeated_backtrack_counts": dict(self.repeated_backtrack_count)
        }