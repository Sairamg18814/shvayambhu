"""ProRL (Prolonged Reinforcement Learning) reasoning architecture.

This package provides the complete ProRL reasoning system capable of extended
multi-step reasoning with up to 2000+ steps, advanced reasoning strategies,
and comprehensive evaluation and optimization mechanisms.
"""

# Core reasoning engine
from .reasoning_engine import (
    BaseReasoningLoop,
    ReasoningStrategy,
    ReasoningStep,
    ReasoningTrace,
    ReasoningConfig,
    StepType
)

# Process reward model
from .reward_model import (
    ProcessRewardModel,
    ProcessReward,
    RewardSignal,
    RewardType,
    RewardModelConfig
)

# Advanced backtracking
from .backtracking import (
    AdvancedBacktrackingManager,
    BacktrackDecision,
    BacktrackResult,
    BacktrackPoint,
    BacktrackReason,
    BacktrackStrategy,
    BacktrackingConfig
)

# Reasoning trace storage
from .trace_storage import (
    TraceStorageManager,
    TraceMetadata,
    QueryPattern,
    StorageConfig
)

# Strategy selection
from .strategy_selector import (
    IntelligentStrategySelector,
    StrategyRecommendation,
    StrategyPerformance,
    QueryAnalysis,
    QueryType,
    QueryComplexity,
    SelectorConfig
)

# Timeout handling
from .timeout_handler import (
    TimeoutHandler,
    AdaptiveTimeoutManager,
    TimeoutEvent,
    TimeoutConfig,
    TimeoutReason,
    TimeoutStrategy,
    timeout_protection
)

# Tree-of-Thought reasoning
from .tree_of_thought import (
    TreeOfThoughtReasoner,
    ThoughtNode,
    ToTConfig,
    NodeType as ToTNodeType,
    SelectionStrategy
)

# Graph-of-Thought reasoning
from .graph_reasoning import (
    GraphOfThoughtReasoner,
    ReasoningNode,
    ReasoningEdge,
    GoTConfig,
    NodeType as GoTNodeType,
    EdgeType,
    PropagationStrategy
)

__all__ = [
    # Core reasoning engine
    'BaseReasoningLoop',
    'ReasoningStrategy',
    'ReasoningStep',
    'ReasoningTrace',
    'ReasoningConfig',
    'StepType',
    
    # Process reward model
    'ProcessRewardModel',
    'ProcessReward',
    'RewardSignal',
    'RewardType',
    'RewardModelConfig',
    
    # Advanced backtracking
    'AdvancedBacktrackingManager',
    'BacktrackDecision',
    'BacktrackResult',
    'BacktrackPoint',
    'BacktrackReason',
    'BacktrackStrategy',
    'BacktrackingConfig',
    
    # Reasoning trace storage
    'TraceStorageManager',
    'TraceMetadata',
    'QueryPattern',
    'StorageConfig',
    
    # Strategy selection
    'IntelligentStrategySelector',
    'StrategyRecommendation',
    'StrategyPerformance',
    'QueryAnalysis',
    'QueryType',
    'QueryComplexity',
    'SelectorConfig',
    
    # Timeout handling
    'TimeoutHandler',
    'AdaptiveTimeoutManager',
    'TimeoutEvent',
    'TimeoutConfig',
    'TimeoutReason',
    'TimeoutStrategy',
    'timeout_protection',
    
    # Tree-of-Thought reasoning
    'TreeOfThoughtReasoner',
    'ThoughtNode',
    'ToTConfig',
    'ToTNodeType',
    'SelectionStrategy',
    
    # Graph-of-Thought reasoning
    'GraphOfThoughtReasoner',
    'ReasoningNode',
    'ReasoningEdge',
    'GoTConfig',
    'GoTNodeType',
    'EdgeType',
    'PropagationStrategy'
]


def create_prorl_reasoner(
    strategy: ReasoningStrategy = ReasoningStrategy.AUTO,
    config: Optional[ReasoningConfig] = None,
    enable_storage: bool = True,
    storage_config: Optional[StorageConfig] = None,
    enable_timeout: bool = True,
    timeout_config: Optional[TimeoutConfig] = None
) -> 'ProRLReasoner':
    """Create a complete ProRL reasoning system.
    
    Args:
        strategy: Default reasoning strategy to use
        config: Configuration for the base reasoning engine
        enable_storage: Whether to enable trace storage
        storage_config: Configuration for trace storage
        enable_timeout: Whether to enable timeout handling
        timeout_config: Configuration for timeout handling
    
    Returns:
        Configured ProRL reasoning system
    """
    from typing import Optional
    
    # Default configurations
    if config is None:
        config = ReasoningConfig()
    
    if storage_config is None:
        storage_config = StorageConfig()
    
    if timeout_config is None:
        timeout_config = TimeoutConfig()
    
    # Create storage manager
    storage_manager = None
    if enable_storage:
        storage_manager = TraceStorageManager(storage_config)
    
    # Create timeout handler
    timeout_handler = None
    if enable_timeout:
        timeout_handler = TimeoutHandler(timeout_config)
    
    # Create strategy selector
    selector_config = SelectorConfig()
    strategy_selector = IntelligentStrategySelector(
        selector_config, storage_manager
    )
    
    # Create reward model
    reward_config = RewardModelConfig()
    reward_model = ProcessRewardModel(reward_config)
    
    # Create backtracking manager
    backtrack_config = BacktrackingConfig()
    backtrack_manager = AdvancedBacktrackingManager(backtrack_config)
    
    # Create main reasoning loop
    reasoning_loop = BaseReasoningLoop(
        model=None,  # Model will be set by user
        config=config,
        reward_model=reward_model.evaluate_step
    )
    
    return ProRLReasoner(
        reasoning_loop=reasoning_loop,
        strategy_selector=strategy_selector,
        storage_manager=storage_manager,
        timeout_handler=timeout_handler,
        reward_model=reward_model,
        backtrack_manager=backtrack_manager
    )


class ProRLReasoner:
    """Complete ProRL reasoning system integrating all components."""
    
    def __init__(
        self,
        reasoning_loop: BaseReasoningLoop,
        strategy_selector: IntelligentStrategySelector,
        storage_manager: Optional[TraceStorageManager] = None,
        timeout_handler: Optional[TimeoutHandler] = None,
        reward_model: Optional[ProcessRewardModel] = None,
        backtrack_manager: Optional[AdvancedBacktrackingManager] = None
    ):
        self.reasoning_loop = reasoning_loop
        self.strategy_selector = strategy_selector
        self.storage_manager = storage_manager
        self.timeout_handler = timeout_handler
        self.reward_model = reward_model
        self.backtrack_manager = backtrack_manager
        
        # Specialized reasoners
        self._tot_reasoner: Optional[TreeOfThoughtReasoner] = None
        self._got_reasoner: Optional[GraphOfThoughtReasoner] = None
    
    def reason(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        strategy: Optional[ReasoningStrategy] = None,
        max_time_seconds: Optional[float] = None
    ) -> ReasoningTrace:
        """Execute comprehensive reasoning on a query.
        
        Args:
            query: The question or problem to reason about
            context: Additional context information
            strategy: Specific strategy to use (if None, will auto-select)
            max_time_seconds: Maximum time to spend reasoning
        
        Returns:
            Complete reasoning trace with final answer
        """
        from typing import Dict, Any
        import time
        
        start_time = time.time()
        
        # Select optimal strategy if not specified
        if strategy is None or strategy == ReasoningStrategy.AUTO:
            recommendation = self.strategy_selector.select_strategy(
                query, context, {"max_time_seconds": max_time_seconds}
            )
            strategy = recommendation.primary_strategy
        
        # Setup timeout protection if enabled
        operation_id = f"reasoning_{int(start_time * 1000)}"
        
        try:
            if self.timeout_handler and max_time_seconds:
                with timeout_protection(
                    max_time_seconds,
                    TimeoutStrategy.GRACEFUL_FINISH,
                    operation_id
                ):
                    trace = self._execute_reasoning(query, context, strategy)
            else:
                trace = self._execute_reasoning(query, context, strategy)
            
            # Store trace if storage is enabled
            if self.storage_manager:
                rewards = self._get_step_rewards(trace) if self.reward_model else None
                self.storage_manager.store_trace(trace, rewards)
            
            # Update strategy performance
            success = trace.confidence > 0.5
            final_confidence = trace.confidence
            self.strategy_selector.update_performance(
                query, strategy, trace, success, final_confidence
            )
            
            return trace
            
        except Exception as e:
            # Create fallback trace
            trace = ReasoningTrace(
                trace_id=str(uuid.uuid4()),
                query=query,
                strategy=strategy,
                start_time=start_time,
                end_time=time.time(),
                final_answer=f"Reasoning failed: {str(e)}",
                confidence=0.0
            )
            
            return trace
    
    def _execute_reasoning(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
        strategy: ReasoningStrategy
    ) -> ReasoningTrace:
        """Execute reasoning with the specified strategy."""
        
        if strategy == ReasoningStrategy.TREE_OF_THOUGHT:
            return self._execute_tree_of_thought(query, context)
        elif strategy == ReasoningStrategy.GRAPH_OF_THOUGHT:
            return self._execute_graph_of_thought(query, context)
        else:
            # Use base reasoning loop for other strategies
            return self.reasoning_loop.reason(query, strategy, context)
    
    def _execute_tree_of_thought(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> ReasoningTrace:
        """Execute Tree-of-Thought reasoning."""
        
        if self._tot_reasoner is None:
            self._tot_reasoner = TreeOfThoughtReasoner(ToTConfig())
        
        final_answer, confidence, reasoning_path = self._tot_reasoner.reason(
            query, context
        )
        
        # Convert to ReasoningTrace
        trace = ReasoningTrace(
            trace_id=str(uuid.uuid4()),
            query=query,
            strategy=ReasoningStrategy.TREE_OF_THOUGHT,
            start_time=time.time() - 30,  # Estimate
            end_time=time.time(),
            final_answer=final_answer,
            confidence=confidence,
            total_steps=len(reasoning_path),
            successful_steps=len([n for n in reasoning_path if n.confidence > 0.5])
        )
        
        # Convert ThoughtNodes to ReasoningSteps
        for i, thought_node in enumerate(reasoning_path):
            step = ReasoningStep(
                step_id=thought_node.node_id,
                step_number=i,
                step_type=self._convert_thought_node_type(thought_node.node_type),
                content=thought_node.content,
                confidence=thought_node.confidence,
                timestamp=thought_node.created_at
            )
            trace.steps.append(step)
            trace.step_index[step.step_id] = step
        
        return trace
    
    def _execute_graph_of_thought(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> ReasoningTrace:
        """Execute Graph-of-Thought reasoning."""
        
        if self._got_reasoner is None:
            self._got_reasoner = GraphOfThoughtReasoner(GoTConfig())
        
        final_answer, confidence, reasoning_graph = self._got_reasoner.reason(
            query, context
        )
        
        # Convert to ReasoningTrace
        trace = ReasoningTrace(
            trace_id=str(uuid.uuid4()),
            query=query,
            strategy=ReasoningStrategy.GRAPH_OF_THOUGHT,
            start_time=time.time() - 60,  # Estimate
            end_time=time.time(),
            final_answer=final_answer,
            confidence=confidence,
            total_steps=len(reasoning_graph.get("nodes", [])),
            successful_steps=len([
                n for n in reasoning_graph.get("nodes", [])
                if n.get("confidence", 0) > 0.5
            ])
        )
        
        # Convert ReasoningNodes to ReasoningSteps
        for i, node_data in enumerate(reasoning_graph.get("nodes", [])):
            step = ReasoningStep(
                step_id=node_data["id"],
                step_number=i,
                step_type=self._convert_graph_node_type(node_data["type"]),
                content=node_data["content"],
                confidence=node_data["confidence"],
                timestamp=time.time() - (60 - i)  # Estimate timestamps
            )
            trace.steps.append(step)
            trace.step_index[step.step_id] = step
        
        return trace
    
    def _convert_thought_node_type(self, node_type) -> StepType:
        """Convert ToT NodeType to reasoning StepType."""
        
        # Import here to avoid circular imports
        from .tree_of_thought import NodeType as ToTNodeType
        
        mapping = {
            ToTNodeType.ROOT: StepType.ANALYSIS,
            ToTNodeType.THOUGHT: StepType.HYPOTHESIS,
            ToTNodeType.EVALUATION: StepType.VERIFICATION,
            ToTNodeType.SYNTHESIS: StepType.SYNTHESIS,
            ToTNodeType.CONCLUSION: StepType.CONCLUSION,
            ToTNodeType.PRUNED: StepType.BACKTRACK
        }
        
        return mapping.get(node_type, StepType.ANALYSIS)
    
    def _convert_graph_node_type(self, node_type_str: str) -> StepType:
        """Convert GoT node type string to reasoning StepType."""
        
        mapping = {
            "root": StepType.ANALYSIS,
            "concept": StepType.ANALYSIS,
            "evidence": StepType.VERIFICATION,
            "hypothesis": StepType.HYPOTHESIS,
            "conclusion": StepType.CONCLUSION,
            "synthesis": StepType.SYNTHESIS,
            "question": StepType.ANALYSIS,
            "assumption": StepType.HYPOTHESIS,
            "counterargument": StepType.VERIFICATION
        }
        
        return mapping.get(node_type_str, StepType.ANALYSIS)
    
    def _get_step_rewards(self, trace: ReasoningTrace) -> List[ProcessReward]:
        """Get reward evaluations for all steps in the trace."""
        
        if not self.reward_model:
            return []
        
        rewards = []
        for step in trace.steps:
            try:
                reward = self.reward_model.evaluate_step(step, trace)
                rewards.append(reward)
            except Exception as e:
                warnings.warn(f"Failed to evaluate step {step.step_id}: {str(e)}")
        
        return rewards
    
    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components."""
        
        stats = {
            "reasoning_loop": self.reasoning_loop.get_reasoning_statistics() if self.reasoning_loop else {},
            "strategy_selector": self.strategy_selector.get_selector_statistics() if self.strategy_selector else {},
            "storage": self.storage_manager.get_statistics() if self.storage_manager else {},
            "timeout": self.timeout_handler.get_timeout_statistics() if self.timeout_handler else {},
            "reward_model": self.reward_model.get_reward_statistics() if self.reward_model else {},
            "backtracking": self.backtrack_manager.get_backtracking_statistics() if self.backtrack_manager else {}
        }
        
        if self._tot_reasoner:
            stats["tree_of_thought"] = self._tot_reasoner.get_tree_statistics()
        
        if self._got_reasoner:
            stats["graph_of_thought"] = self._got_reasoner.get_graph_statistics()
        
        return stats
    
    def query_similar_traces(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.5
    ) -> List[Tuple[TraceMetadata, float]]:
        """Find traces similar to the given query."""
        
        if not self.storage_manager:
            return []
        
        return self.storage_manager.get_similar_traces(query, limit, min_similarity)
    
    def export_reasoning_session(
        self,
        trace: ReasoningTrace,
        include_rewards: bool = True,
        format: str = "json"
    ) -> Dict[str, Any]:
        """Export a complete reasoning session for analysis."""
        
        session_data = {
            "trace": self.reasoning_loop.export_trace(trace, include_metadata=True),
            "strategy_used": trace.strategy.value,
            "performance_metrics": {
                "total_steps": trace.total_steps,
                "successful_steps": trace.successful_steps,
                "confidence": trace.confidence,
                "total_reward": trace.total_reward,
                "backtrack_count": trace.backtrack_count,
                "duration_seconds": (trace.end_time or time.time()) - trace.start_time
            }
        }
        
        if include_rewards and self.reward_model:
            step_rewards = self._get_step_rewards(trace)
            session_data["step_rewards"] = [
                {
                    "step_id": reward.step_id,
                    "total_reward": reward.total_reward,
                    "component_rewards": {
                        comp_type.value: signal.score
                        for comp_type, signal in reward.component_rewards.items()
                    }
                }
                for reward in step_rewards
            ]
        
        return session_data


# Convenience functions for quick access to specific reasoning strategies
def chain_of_thought_reasoning(
    query: str,
    model,
    context: Optional[Dict[str, Any]] = None,
    max_steps: int = 50
) -> ReasoningTrace:
    """Execute Chain-of-Thought reasoning."""
    
    config = ReasoningConfig(max_steps=max_steps)
    reasoning_loop = BaseReasoningLoop(model, config)
    
    return reasoning_loop.reason(query, ReasoningStrategy.CHAIN_OF_THOUGHT, context)


def tree_of_thought_reasoning(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    max_depth: int = 15,
    max_children: int = 3
) -> Tuple[str, float, List[ThoughtNode]]:
    """Execute Tree-of-Thought reasoning."""
    
    config = ToTConfig(max_depth=max_depth, max_children_per_node=max_children)
    reasoner = TreeOfThoughtReasoner(config)
    
    return reasoner.reason(query, context)


def graph_of_thought_reasoning(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    max_nodes: int = 200,
    max_iterations: int = 30
) -> Tuple[str, float, Dict[str, Any]]:
    """Execute Graph-of-Thought reasoning."""
    
    config = GoTConfig(max_nodes=max_nodes)
    reasoner = GraphOfThoughtReasoner(config)
    
    return reasoner.reason(query, context, max_iterations)


# Package version
__version__ = "1.0.0"

# Package metadata
__author__ = "Shvayambhu LLM Team"
__email__ = "dev@shvayambhu.ai"
__description__ = "ProRL: Prolonged Reinforcement Learning for advanced reasoning"