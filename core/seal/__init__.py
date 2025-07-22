"""SEAL (Self-Adapting Language Model) architecture.

This package provides the core components for self-editing and adaptation
capabilities in the Shvayambhu LLM.
"""

from .self_edit import (
    SelfEditGenerator,
    PerformanceAnalyzer as SelfEditPerformanceAnalyzer,
    EditProposalGenerator,
    EditValidator as SelfEditValidator,
    EditCandidate
)

from .lora_adapter import (
    LoRALinear,
    LoRAEmbedding,
    LoRAConfig
)

from .parameter_diff import (
    ParameterDiff,
    ParameterDiffSystem,
    DiffConfig
)

from .edit_validation import (
    EditValidator,
    ValidationResult,
    ValidationConfig,
    SafetyValidator,
    EditValidationPipeline
)

from .rollback_manager import (
    RollbackManager,
    Checkpoint,
    EditRecord,
    RollbackConfig
)

from .edit_history import (
    EditHistoryTracker,
    EditEvent,
    EditSession
)

from .performance_analyzer import (
    PerformanceAnalyzer,
    PerformanceSnapshot,
    PerformanceImpact,
    PerformanceConfig
)

from .dynamic_lora import (
    DynamicRankSelector,
    RankOptimizationConfig,
    RankAnalysis
)

from .lora_merging import (
    LoRAMerger,
    MergingConfig,
    MergeResult
)

from .memory_efficient_lora import (
    MemoryEfficientLoRA,
    MemoryEfficientLoRAManager,
    MemoryConfig,
    MemoryStats
)

from .lora_checkpoint import (
    LoRACheckpointManager,
    CheckpointMetadata,
    CheckpointConfig
)

from .ab_testing import (
    ABTestManager,
    ABTest,
    ABTestConfig,
    ABTestVariant,
    ABTestResult,
    TrafficRouter
)

from .gradual_application import (
    GradualApplicationManager,
    GradualApplicationConfig,
    ApplicationStep,
    ApplicationResult
)

from .rl_controller import (
    RLOptimizationController,
    RLConfig,
    RewardSystem,
    RewardSignal,
    ExplorationStrategy,
    StabilityMonitor
)

__all__ = [
    # Self-edit generation
    'SelfEditGenerator',
    'SelfEditPerformanceAnalyzer',
    'EditProposalGenerator',
    'SelfEditValidator',
    'EditCandidate',
    
    # LoRA adapters
    'LoRALinear',
    'LoRAEmbedding',
    'LoRAConfig',
    
    # Parameter diffs
    'ParameterDiff',
    'ParameterDiffSystem',
    'DiffConfig',
    
    # Edit validation
    'EditValidator',
    'ValidationResult',
    'ValidationConfig',
    'SafetyValidator',
    'EditValidationPipeline',
    
    # Rollback management
    'RollbackManager',
    'Checkpoint',
    'EditRecord',
    'RollbackConfig',
    
    # Edit history tracking
    'EditHistoryTracker',
    'EditEvent',
    'EditSession',
    
    # Performance analysis
    'PerformanceAnalyzer',
    'PerformanceSnapshot',
    'PerformanceImpact',
    'PerformanceConfig',
    
    # Dynamic LoRA
    'DynamicRankSelector',
    'RankOptimizationConfig',
    'RankAnalysis',
    
    # LoRA merging
    'LoRAMerger',
    'MergingConfig',
    'MergeResult',
    
    # Memory-efficient LoRA
    'MemoryEfficientLoRA',
    'MemoryEfficientLoRAManager',
    'MemoryConfig',
    'MemoryStats',
    
    # LoRA checkpointing
    'LoRACheckpointManager',
    'CheckpointMetadata',
    'CheckpointConfig',
    
    # A/B testing
    'ABTestManager',
    'ABTest',
    'ABTestConfig',
    'ABTestVariant',
    'ABTestResult',
    'TrafficRouter',
    
    # Gradual application
    'GradualApplicationManager',
    'GradualApplicationConfig',
    'ApplicationStep',
    'ApplicationResult',
    
    # RL optimization controller
    'RLOptimizationController',
    'RLConfig',
    'RewardSystem',
    'RewardSignal',
    'ExplorationStrategy',
    'StabilityMonitor'
]