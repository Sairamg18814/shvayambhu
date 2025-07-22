"""Core reasoning engine for ProRL (Prolonged Reinforcement Learning).

This module implements the main reasoning loop capable of extended multi-step
reasoning with up to 2000+ steps, backtracking, and strategy selection.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid
import json
from collections import defaultdict, deque
import numpy as np
import warnings

from ..blt.pipeline import BLTPipeline


class ReasoningStrategy(Enum):
    """Available reasoning strategies."""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    GRAPH_OF_THOUGHT = "graph_of_thought"
    SELF_CONSISTENCY = "self_consistency"
    METACOGNITIVE = "metacognitive"
    AUTO = "auto"


class StepType(Enum):
    """Types of reasoning steps."""
    ANALYSIS = "analysis"
    HYPOTHESIS = "hypothesis"
    VERIFICATION = "verification"
    CONCLUSION = "conclusion"
    BACKTRACK = "backtrack"
    BRANCH = "branch"
    SYNTHESIS = "synthesis"


@dataclass
class ReasoningStep:
    """A single step in the reasoning process."""
    step_id: str
    step_number: int
    step_type: StepType
    content: str
    confidence: float
    timestamp: float
    
    # Context and relationships
    parent_step_id: Optional[str] = None
    child_step_ids: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Evaluation metrics
    reward_score: float = 0.0
    quality_score: float = 0.0
    relevance_score: float = 0.0
    coherence_score: float = 0.0
    
    # Meta information
    generation_time_ms: float = 0.0
    tokens_generated: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningTrace:
    """Complete trace of a reasoning session."""
    trace_id: str
    query: str
    strategy: ReasoningStrategy
    start_time: float
    end_time: Optional[float] = None
    
    # Reasoning steps
    steps: List[ReasoningStep] = field(default_factory=list)
    step_index: Dict[str, ReasoningStep] = field(default_factory=dict)
    
    # Trace structure
    root_step_ids: List[str] = field(default_factory=list)
    leaf_step_ids: List[str] = field(default_factory=list)
    depth: int = 0
    max_depth: int = 0
    
    # Final result
    final_answer: Optional[str] = None
    confidence: float = 0.0
    total_reward: float = 0.0
    
    # Statistics
    total_steps: int = 0
    successful_steps: int = 0
    backtrack_count: int = 0
    branch_count: int = 0
    
    # Resource usage
    total_generation_time_ms: float = 0.0
    total_tokens_generated: int = 0
    memory_peak_mb: float = 0.0


@dataclass
class ReasoningConfig:
    """Configuration for the reasoning engine."""
    max_steps: int = 2000
    max_depth: int = 50
    max_time_seconds: float = 300.0  # 5 minutes
    
    # Quality thresholds
    min_step_confidence: float = 0.3
    min_step_quality: float = 0.4
    backtrack_threshold: float = 0.2
    
    # Strategy selection
    default_strategy: ReasoningStrategy = ReasoningStrategy.AUTO
    enable_strategy_switching: bool = True
    strategy_switch_threshold: float = 0.1
    
    # Exploration parameters
    exploration_rate: float = 0.1
    temperature: float = 0.7
    top_k: int = 5
    top_p: float = 0.9
    
    # Resource limits
    max_memory_mb: float = 8192.0
    max_generation_tokens: int = 512
    min_generation_tokens: int = 20
    
    # Reward model settings
    reward_window_size: int = 5
    reward_decay: float = 0.95
    quality_weight: float = 0.3
    relevance_weight: float = 0.3
    coherence_weight: float = 0.2
    confidence_weight: float = 0.2


class BaseReasoningLoop:
    """Base reasoning loop implementation."""
    
    def __init__(
        self,
        model: nn.Module,
        config: ReasoningConfig,
        reward_model: Optional[Callable] = None
    ):
        self.model = model
        self.config = config
        self.reward_model = reward_model
        
        # Active reasoning state
        self.current_trace: Optional[ReasoningTrace] = None
        self.step_queue: deque = deque()
        self.active_branches: Dict[str, List[str]] = {}
        
        # Strategy implementations
        self.strategies = {}
        self._register_default_strategies()
        
        # Performance tracking
        self.session_stats = {
            "total_traces": 0,
            "successful_traces": 0,
            "avg_steps_per_trace": 0.0,
            "avg_time_per_trace": 0.0,
            "total_backtracks": 0,
            "strategy_usage": defaultdict(int)
        }
    
    def _register_default_strategies(self):
        """Register default reasoning strategies."""
        self.strategies = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: self._chain_of_thought_step,
            ReasoningStrategy.TREE_OF_THOUGHT: self._tree_of_thought_step,
            ReasoningStrategy.GRAPH_OF_THOUGHT: self._graph_of_thought_step,
            ReasoningStrategy.SELF_CONSISTENCY: self._self_consistency_step,
            ReasoningStrategy.METACOGNITIVE: self._metacognitive_step
        }
    
    def reason(
        self,
        query: str,
        strategy: Optional[ReasoningStrategy] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ReasoningTrace:
        """Execute multi-step reasoning on a query."""
        start_time = time.time()
        
        # Initialize reasoning trace
        trace_id = str(uuid.uuid4())
        strategy = strategy or self.config.default_strategy
        
        if strategy == ReasoningStrategy.AUTO:
            strategy = self._select_strategy(query, context)
        
        self.current_trace = ReasoningTrace(
            trace_id=trace_id,
            query=query,
            strategy=strategy,
            start_time=start_time
        )
        
        # Create initial step
        initial_step = self._create_initial_step(query)
        self._add_step(initial_step)
        self.step_queue.append(initial_step.step_id)
        
        try:
            # Main reasoning loop
            while (self.step_queue and 
                   self.current_trace.total_steps < self.config.max_steps and
                   time.time() - start_time < self.config.max_time_seconds):
                
                # Get next step to process
                current_step_id = self.step_queue.popleft()
                current_step = self.current_trace.step_index[current_step_id]
                
                # Generate next reasoning step
                next_steps = self._generate_next_steps(current_step)
                
                # Evaluate and filter steps
                valid_steps = self._evaluate_steps(next_steps)
                
                if not valid_steps:
                    # Need to backtrack
                    backtrack_step = self._attempt_backtrack(current_step)
                    if backtrack_step:
                        self._add_step(backtrack_step)
                        self.step_queue.append(backtrack_step.step_id)
                    continue
                
                # Add valid steps to trace
                for step in valid_steps:
                    self._add_step(step)
                    
                    # Determine if step should continue reasoning
                    if self._should_continue_from_step(step):
                        self.step_queue.append(step.step_id)
                    elif step.step_type == StepType.CONCLUSION:
                        # Potential final answer
                        self._evaluate_potential_conclusion(step)
                
                # Check for convergence
                if self._has_converged():
                    break
                
                # Adaptive strategy switching
                if (self.config.enable_strategy_switching and
                    self._should_switch_strategy()):
                    new_strategy = self._select_strategy(
                        query, {"current_trace": self.current_trace}
                    )
                    if new_strategy != strategy:
                        strategy = new_strategy
                        self.current_trace.strategy = strategy
            
            # Finalize reasoning
            self._finalize_reasoning()
            
        except Exception as e:
            warnings.warn(f"Reasoning failed: {str(e)}")
            self.current_trace.final_answer = "Reasoning failed due to internal error."
            self.current_trace.confidence = 0.0
        
        finally:
            # Complete trace
            self.current_trace.end_time = time.time()
            self._update_session_stats()
        
        return self.current_trace
    
    def _create_initial_step(self, query: str) -> ReasoningStep:
        """Create the initial reasoning step."""
        step_id = str(uuid.uuid4())
        return ReasoningStep(
            step_id=step_id,
            step_number=0,
            step_type=StepType.ANALYSIS,
            content=f"Let me analyze this query: {query}",
            confidence=0.8,
            timestamp=time.time()
        )
    
    def _generate_next_steps(self, current_step: ReasoningStep) -> List[ReasoningStep]:
        """Generate next reasoning steps based on current step and strategy."""
        strategy_func = self.strategies.get(self.current_trace.strategy)
        if not strategy_func:
            strategy_func = self.strategies[ReasoningStrategy.CHAIN_OF_THOUGHT]
        
        return strategy_func(current_step)
    
    def _chain_of_thought_step(self, current_step: ReasoningStep) -> List[ReasoningStep]:
        """Generate next step using Chain-of-Thought reasoning."""
        if current_step.step_type == StepType.ANALYSIS:
            # Generate hypothesis
            return [self._create_step(
                StepType.HYPOTHESIS,
                "Based on my analysis, I hypothesize that...",
                current_step.step_id
            )]
        elif current_step.step_type == StepType.HYPOTHESIS:
            # Generate verification
            return [self._create_step(
                StepType.VERIFICATION,
                "Let me verify this hypothesis by...",
                current_step.step_id
            )]
        elif current_step.step_type == StepType.VERIFICATION:
            # Generate conclusion
            return [self._create_step(
                StepType.CONCLUSION,
                "Based on my verification, I conclude that...",
                current_step.step_id
            )]
        else:
            return []
    
    def _tree_of_thought_step(self, current_step: ReasoningStep) -> List[ReasoningStep]:
        """Generate multiple candidate steps for Tree-of-Thought exploration."""
        # Generate multiple candidate next steps
        candidates = []
        
        if current_step.step_type == StepType.ANALYSIS:
            # Multiple hypotheses
            for i in range(3):
                candidates.append(self._create_step(
                    StepType.HYPOTHESIS,
                    f"Hypothesis {i+1}: Alternative perspective...",
                    current_step.step_id
                ))
        elif current_step.step_type == StepType.HYPOTHESIS:
            # Different verification approaches
            candidates.append(self._create_step(
                StepType.VERIFICATION,
                "Verify through logical reasoning...",
                current_step.step_id
            ))
            candidates.append(self._create_step(
                StepType.VERIFICATION,
                "Verify through empirical evidence...",
                current_step.step_id
            ))
        
        return candidates
    
    def _graph_of_thought_step(self, current_step: ReasoningStep) -> List[ReasoningStep]:
        """Generate steps that can connect to multiple previous steps."""
        # Graph-based reasoning allows more complex connections
        candidates = []
        
        # Find relevant previous steps to connect to
        relevant_steps = self._find_relevant_steps(current_step)
        
        if relevant_steps:
            synthesis_step = self._create_step(
                StepType.SYNTHESIS,
                "Synthesizing information from multiple sources...",
                current_step.step_id
            )
            synthesis_step.dependencies.extend([s.step_id for s in relevant_steps])
            candidates.append(synthesis_step)
        
        return candidates
    
    def _self_consistency_step(self, current_step: ReasoningStep) -> List[ReasoningStep]:
        """Generate steps for self-consistency checking."""
        if current_step.step_type == StepType.CONCLUSION:
            # Verify conclusion consistency
            return [self._create_step(
                StepType.VERIFICATION,
                "Checking consistency of conclusion with previous steps...",
                current_step.step_id
            )]
        return self._chain_of_thought_step(current_step)
    
    def _metacognitive_step(self, current_step: ReasoningStep) -> List[ReasoningStep]:
        """Generate metacognitive reasoning steps."""
        # Add self-reflection and strategy evaluation
        meta_step = self._create_step(
            StepType.ANALYSIS,
            "Reflecting on my reasoning process so far...",
            current_step.step_id
        )
        meta_step.metadata["metacognitive"] = True
        return [meta_step]
    
    def _create_step(
        self,
        step_type: StepType,
        content: str,
        parent_id: Optional[str] = None
    ) -> ReasoningStep:
        """Create a new reasoning step."""
        step_id = str(uuid.uuid4())
        step_number = self.current_trace.total_steps + 1
        
        step = ReasoningStep(
            step_id=step_id,
            step_number=step_number,
            step_type=step_type,
            content=content,
            confidence=0.5,  # Will be updated by evaluation
            timestamp=time.time(),
            parent_step_id=parent_id
        )
        
        return step
    
    def _add_step(self, step: ReasoningStep):
        """Add a step to the current trace."""
        self.current_trace.steps.append(step)
        self.current_trace.step_index[step.step_id] = step
        self.current_trace.total_steps += 1
        
        # Update parent-child relationships
        if step.parent_step_id:
            parent = self.current_trace.step_index.get(step.parent_step_id)
            if parent:
                parent.child_step_ids.append(step.step_id)
        else:
            self.current_trace.root_step_ids.append(step.step_id)
        
        # Update depth tracking
        depth = self._calculate_step_depth(step)
        self.current_trace.max_depth = max(self.current_trace.max_depth, depth)
    
    def _evaluate_steps(self, steps: List[ReasoningStep]) -> List[ReasoningStep]:
        """Evaluate steps and filter based on quality thresholds."""
        valid_steps = []
        
        for step in steps:
            # Calculate step quality metrics
            quality_score = self._calculate_quality_score(step)
            relevance_score = self._calculate_relevance_score(step)
            coherence_score = self._calculate_coherence_score(step)
            
            # Overall evaluation
            confidence = (
                quality_score * self.config.quality_weight +
                relevance_score * self.config.relevance_weight +
                coherence_score * self.config.coherence_weight +
                step.confidence * self.config.confidence_weight
            )
            
            step.quality_score = quality_score
            step.relevance_score = relevance_score
            step.coherence_score = coherence_score
            step.confidence = confidence
            
            # Filter based on thresholds
            if (confidence >= self.config.min_step_confidence and
                quality_score >= self.config.min_step_quality):
                valid_steps.append(step)
        
        # Sort by confidence and limit if needed
        valid_steps.sort(key=lambda x: x.confidence, reverse=True)
        return valid_steps[:self.config.top_k]
    
    def _calculate_quality_score(self, step: ReasoningStep) -> float:
        """Calculate quality score for a reasoning step."""
        # Placeholder implementation - would use actual quality model
        base_score = 0.7
        
        # Length-based heuristic
        if len(step.content) < 20:
            base_score -= 0.2
        elif len(step.content) > 1000:
            base_score -= 0.1
        
        # Type-based adjustment
        if step.step_type == StepType.CONCLUSION:
            base_score += 0.1
        elif step.step_type == StepType.BACKTRACK:
            base_score -= 0.1
        
        return max(0.0, min(1.0, base_score))
    
    def _calculate_relevance_score(self, step: ReasoningStep) -> float:
        """Calculate relevance score for a reasoning step."""
        # Placeholder implementation
        return 0.8
    
    def _calculate_coherence_score(self, step: ReasoningStep) -> float:
        """Calculate coherence score for a reasoning step."""
        # Placeholder implementation
        return 0.75
    
    def _should_continue_from_step(self, step: ReasoningStep) -> bool:
        """Determine if reasoning should continue from this step."""
        if step.step_type == StepType.CONCLUSION:
            return False  # Conclusions typically end reasoning paths
        
        if step.confidence < self.config.min_step_confidence:
            return False  # Low confidence steps shouldn't continue
        
        return True
    
    def _attempt_backtrack(self, failed_step: ReasoningStep) -> Optional[ReasoningStep]:
        """Attempt to backtrack from a failed reasoning path."""
        if not failed_step.parent_step_id:
            return None  # Can't backtrack from root
        
        self.current_trace.backtrack_count += 1
        
        # Find alternative parent or create backtrack step
        parent = self.current_trace.step_index.get(failed_step.parent_step_id)
        if parent:
            backtrack_step = self._create_step(
                StepType.BACKTRACK,
                f"Previous approach didn't work. Let me try a different angle...",
                parent.step_id
            )
            return backtrack_step
        
        return None
    
    def _has_converged(self) -> bool:
        """Check if reasoning has converged to a solution."""
        if not self.current_trace.steps:
            return False
        
        # Check if we have high-confidence conclusions
        conclusions = [
            step for step in self.current_trace.steps
            if (step.step_type == StepType.CONCLUSION and
                step.confidence > 0.8)
        ]
        
        return len(conclusions) >= 2  # Multiple high-confidence conclusions
    
    def _finalize_reasoning(self):
        """Finalize the reasoning trace and determine final answer."""
        if not self.current_trace.steps:
            self.current_trace.final_answer = "No reasoning steps generated."
            self.current_trace.confidence = 0.0
            return
        
        # Find the best conclusion
        conclusions = [
            step for step in self.current_trace.steps
            if step.step_type == StepType.CONCLUSION
        ]
        
        if conclusions:
            best_conclusion = max(conclusions, key=lambda x: x.confidence)
            self.current_trace.final_answer = best_conclusion.content
            self.current_trace.confidence = best_conclusion.confidence
        else:
            # Use the highest confidence step as answer
            best_step = max(self.current_trace.steps, key=lambda x: x.confidence)
            self.current_trace.final_answer = best_step.content
            self.current_trace.confidence = best_step.confidence
        
        # Calculate total reward
        self.current_trace.total_reward = sum(
            step.reward_score for step in self.current_trace.steps
        )
        
        # Update statistics
        self.current_trace.successful_steps = sum(
            1 for step in self.current_trace.steps
            if step.confidence > self.config.min_step_confidence
        )
    
    def _select_strategy(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ReasoningStrategy:
        """Select optimal reasoning strategy for the query."""
        # Simple heuristic-based strategy selection
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["compare", "analyze", "evaluate"]):
            return ReasoningStrategy.TREE_OF_THOUGHT
        elif any(word in query_lower for word in ["connect", "relate", "synthesize"]):
            return ReasoningStrategy.GRAPH_OF_THOUGHT
        elif any(word in query_lower for word in ["verify", "check", "confirm"]):
            return ReasoningStrategy.SELF_CONSISTENCY
        else:
            return ReasoningStrategy.CHAIN_OF_THOUGHT
    
    def _should_switch_strategy(self) -> bool:
        """Determine if strategy should be switched."""
        if self.current_trace.total_steps < 5:
            return False  # Too early to switch
        
        # Check recent step performance
        recent_steps = self.current_trace.steps[-5:]
        avg_confidence = sum(step.confidence for step in recent_steps) / len(recent_steps)
        
        return avg_confidence < self.config.strategy_switch_threshold
    
    def _find_relevant_steps(self, current_step: ReasoningStep) -> List[ReasoningStep]:
        """Find steps relevant to the current step for graph reasoning."""
        relevant_steps = []
        
        for step in self.current_trace.steps:
            if (step.step_id != current_step.step_id and
                step.confidence > 0.6):
                # Simple relevance based on step type compatibility
                if (current_step.step_type == StepType.SYNTHESIS or
                    step.step_type in [StepType.HYPOTHESIS, StepType.ANALYSIS]):
                    relevant_steps.append(step)
        
        return relevant_steps[:3]  # Limit connections
    
    def _calculate_step_depth(self, step: ReasoningStep) -> int:
        """Calculate the depth of a step in the reasoning tree."""
        depth = 0
        current_step = step
        
        while current_step.parent_step_id:
            depth += 1
            current_step = self.current_trace.step_index.get(current_step.parent_step_id)
            if not current_step:
                break
        
        return depth
    
    def _evaluate_potential_conclusion(self, step: ReasoningStep):
        """Evaluate if a conclusion step should end reasoning."""
        if step.confidence > 0.9:
            # High confidence conclusion - likely to end reasoning
            pass
        elif step.confidence < 0.5:
            # Low confidence - continue reasoning
            self.step_queue.append(step.step_id)
    
    def _update_session_stats(self):
        """Update session statistics."""
        self.session_stats["total_traces"] += 1
        
        if self.current_trace.confidence > 0.5:
            self.session_stats["successful_traces"] += 1
        
        self.session_stats["total_backtracks"] += self.current_trace.backtrack_count
        self.session_stats["strategy_usage"][self.current_trace.strategy] += 1
        
        # Update averages
        total = self.session_stats["total_traces"]
        self.session_stats["avg_steps_per_trace"] = (
            (self.session_stats["avg_steps_per_trace"] * (total - 1) + 
             self.current_trace.total_steps) / total
        )
        
        if self.current_trace.end_time:
            trace_time = self.current_trace.end_time - self.current_trace.start_time
            self.session_stats["avg_time_per_trace"] = (
                (self.session_stats["avg_time_per_trace"] * (total - 1) + 
                 trace_time) / total
            )
    
    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reasoning statistics."""
        return {
            "session_stats": dict(self.session_stats),
            "current_trace_stats": {
                "total_steps": self.current_trace.total_steps if self.current_trace else 0,
                "successful_steps": self.current_trace.successful_steps if self.current_trace else 0,
                "backtrack_count": self.current_trace.backtrack_count if self.current_trace else 0,
                "max_depth": self.current_trace.max_depth if self.current_trace else 0,
                "confidence": self.current_trace.confidence if self.current_trace else 0.0
            } if self.current_trace else {}
        }
    
    def export_trace(self, trace: ReasoningTrace, include_metadata: bool = True) -> Dict[str, Any]:
        """Export reasoning trace for analysis or storage."""
        export_data = {
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
                "branch_count": trace.branch_count,
                "max_depth": trace.max_depth
            }
        }
        
        if include_metadata:
            export_data["steps"] = [
                {
                    "step_id": step.step_id,
                    "step_number": step.step_number,
                    "step_type": step.step_type.value,
                    "content": step.content,
                    "confidence": step.confidence,
                    "parent_step_id": step.parent_step_id,
                    "quality_score": step.quality_score,
                    "relevance_score": step.relevance_score,
                    "coherence_score": step.coherence_score
                }
                for step in trace.steps
            ]
        
        return export_data