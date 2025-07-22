"""Process reward model for ProRL reasoning steps evaluation.

This module implements the process reward model that evaluates the quality
and effectiveness of reasoning steps in real-time during the reasoning process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time
import warnings
from collections import defaultdict, deque

from .reasoning_engine import ReasoningStep, ReasoningTrace, StepType


class RewardType(Enum):
    """Types of rewards for reasoning steps."""
    QUALITY = "quality"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    PROGRESS = "progress"
    EFFICIENCY = "efficiency"
    SAFETY = "safety"
    CONSISTENCY = "consistency"
    NOVELTY = "novelty"


@dataclass
class RewardSignal:
    """A reward signal for a reasoning step."""
    reward_type: RewardType
    score: float
    confidence: float
    reasoning: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessReward:
    """Comprehensive process reward for a reasoning step."""
    step_id: str
    total_reward: float
    component_rewards: Dict[RewardType, RewardSignal]
    normalized_reward: float
    percentile_rank: float
    
    # Context information
    step_context: Dict[str, Any] = field(default_factory=dict)
    trace_context: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation metadata
    evaluation_time_ms: float = 0.0
    model_confidence: float = 0.0
    human_validation: Optional[bool] = None


@dataclass
class RewardModelConfig:
    """Configuration for the process reward model."""
    # Reward component weights
    quality_weight: float = 0.25
    relevance_weight: float = 0.20
    coherence_weight: float = 0.15
    progress_weight: float = 0.20
    efficiency_weight: float = 0.10
    safety_weight: float = 0.05
    consistency_weight: float = 0.03
    novelty_weight: float = 0.02
    
    # Model parameters
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    
    # Evaluation settings
    min_confidence_threshold: float = 0.3
    reward_normalization: str = "z_score"  # "z_score", "min_max", "sigmoid"
    context_window_size: int = 10
    
    # Learning parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    gradient_clipping: float = 1.0
    
    # Historical data
    reward_history_size: int = 1000
    percentile_window_size: int = 100


class ProcessRewardModel(nn.Module):
    """Neural process reward model for evaluating reasoning steps."""
    
    def __init__(self, config: RewardModelConfig, vocab_size: int = 50000):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        # Embedding layers
        self.step_embedding = nn.Embedding(vocab_size, config.hidden_dim)
        self.type_embedding = nn.Embedding(len(StepType), config.hidden_dim)
        self.position_embedding = nn.Embedding(2000, config.hidden_dim)  # Max 2000 steps
        
        # Context encoder
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True
            ),
            num_layers=config.num_layers
        )
        
        # Reward component heads
        self.reward_heads = nn.ModuleDict({
            reward_type.value: nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            for reward_type in RewardType
        })
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Historical reward tracking
        self.reward_history = deque(maxlen=config.reward_history_size)
        self.percentile_cache = {}
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(
        self,
        step_tokens: torch.Tensor,
        step_types: torch.Tensor,
        positions: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass of the reward model."""
        batch_size, seq_len = step_tokens.shape
        
        # Embeddings
        step_emb = self.step_embedding(step_tokens)
        type_emb = self.type_embedding(step_types)
        pos_emb = self.position_embedding(positions)
        
        # Combined embedding
        hidden = step_emb + type_emb + pos_emb
        
        # Context encoding
        if context_mask is not None:
            hidden = self.context_encoder(hidden, src_key_padding_mask=~context_mask)
        else:
            hidden = self.context_encoder(hidden)
        
        # Pool sequence representation (mean pooling over non-masked positions)
        if context_mask is not None:
            mask_expanded = context_mask.unsqueeze(-1).expand_as(hidden)
            hidden_masked = hidden * mask_expanded
            pooled = hidden_masked.sum(dim=1) / context_mask.sum(dim=1, keepdim=True).float()
        else:
            pooled = hidden.mean(dim=1)
        
        # Compute reward components
        component_rewards = {}
        for reward_type, head in self.reward_heads.items():
            component_rewards[reward_type] = head(pooled).squeeze(-1)
        
        # Compute confidence
        confidence = self.confidence_head(pooled).squeeze(-1)
        
        # Aggregate total reward
        total_reward = self._aggregate_rewards(component_rewards)
        
        return total_reward, confidence, component_rewards
    
    def _aggregate_rewards(self, component_rewards: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate component rewards into total reward."""
        weight_map = {
            RewardType.QUALITY.value: self.config.quality_weight,
            RewardType.RELEVANCE.value: self.config.relevance_weight,
            RewardType.COHERENCE.value: self.config.coherence_weight,
            RewardType.PROGRESS.value: self.config.progress_weight,
            RewardType.EFFICIENCY.value: self.config.efficiency_weight,
            RewardType.SAFETY.value: self.config.safety_weight,
            RewardType.CONSISTENCY.value: self.config.consistency_weight,
            RewardType.NOVELTY.value: self.config.novelty_weight
        }
        
        total = torch.zeros_like(component_rewards[RewardType.QUALITY.value])
        for reward_type, reward in component_rewards.items():
            weight = weight_map.get(reward_type, 0.0)
            total += weight * reward
        
        return total
    
    def evaluate_step(
        self,
        step: ReasoningStep,
        trace_context: ReasoningTrace,
        tokenizer: Optional[Any] = None
    ) -> ProcessReward:
        """Evaluate a single reasoning step."""
        start_time = time.time()
        
        # Prepare input tensors
        step_tokens, step_types, positions, mask = self._prepare_step_input(
            step, trace_context, tokenizer
        )
        
        with torch.no_grad():
            # Forward pass
            total_reward, confidence, component_rewards = self.forward(
                step_tokens, step_types, positions, mask
            )
            
            # Convert to scalar values
            total_reward_val = total_reward.item()
            confidence_val = confidence.item()
            
            # Create reward signals for each component
            reward_signals = {}
            for reward_type_str, reward_tensor in component_rewards.items():
                reward_type = RewardType(reward_type_str)
                reward_signals[reward_type] = RewardSignal(
                    reward_type=reward_type,
                    score=reward_tensor.item(),
                    confidence=confidence_val,
                    reasoning=self._generate_reasoning(reward_type, reward_tensor.item())
                )
            
            # Normalize reward
            normalized_reward = self._normalize_reward(total_reward_val)
            
            # Calculate percentile rank
            percentile_rank = self._calculate_percentile_rank(total_reward_val)
            
            # Create process reward
            process_reward = ProcessReward(
                step_id=step.step_id,
                total_reward=total_reward_val,
                component_rewards=reward_signals,
                normalized_reward=normalized_reward,
                percentile_rank=percentile_rank,
                step_context=self._extract_step_context(step),
                trace_context=self._extract_trace_context(trace_context),
                evaluation_time_ms=(time.time() - start_time) * 1000,
                model_confidence=confidence_val
            )
            
            # Update history
            self.reward_history.append(total_reward_val)
            
            return process_reward
    
    def _prepare_step_input(
        self,
        step: ReasoningStep,
        trace_context: ReasoningTrace,
        tokenizer: Optional[Any] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare input tensors for the step."""
        # Simple tokenization (would use actual tokenizer in practice)
        if tokenizer is None:
            # Basic character-level tokenization
            step_tokens = [ord(c) % self.vocab_size for c in step.content[:512]]
        else:
            step_tokens = tokenizer.encode(step.content, max_length=512, truncation=True)
        
        # Pad to fixed length
        max_len = 512
        if len(step_tokens) < max_len:
            step_tokens.extend([0] * (max_len - len(step_tokens)))
        else:
            step_tokens = step_tokens[:max_len]
        
        # Create tensors
        step_tokens_tensor = torch.tensor([step_tokens], dtype=torch.long)
        step_types_tensor = torch.tensor([[step.step_type.value] * len(step_tokens)], dtype=torch.long)
        positions_tensor = torch.tensor([list(range(len(step_tokens)))], dtype=torch.long)
        mask_tensor = torch.tensor([[1] * len(step_tokens)], dtype=torch.bool)
        
        return step_tokens_tensor, step_types_tensor, positions_tensor, mask_tensor
    
    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward based on configuration."""
        if self.config.reward_normalization == "z_score":
            if len(self.reward_history) < 2:
                return 0.5  # Default normalized value
            
            mean_reward = np.mean(self.reward_history)
            std_reward = np.std(self.reward_history)
            
            if std_reward == 0:
                return 0.5
            
            z_score = (reward - mean_reward) / std_reward
            # Convert to 0-1 range using sigmoid
            return 1.0 / (1.0 + np.exp(-z_score))
        
        elif self.config.reward_normalization == "min_max":
            if len(self.reward_history) < 2:
                return reward  # No normalization possible
            
            min_reward = min(self.reward_history)
            max_reward = max(self.reward_history)
            
            if max_reward == min_reward:
                return 0.5
            
            return (reward - min_reward) / (max_reward - min_reward)
        
        elif self.config.reward_normalization == "sigmoid":
            return 1.0 / (1.0 + np.exp(-reward))
        
        else:
            return reward
    
    def _calculate_percentile_rank(self, reward: float) -> float:
        """Calculate percentile rank of the reward."""
        if len(self.reward_history) < 2:
            return 50.0  # Default percentile
        
        # Use recent history for percentile calculation
        recent_history = list(self.reward_history)[-self.config.percentile_window_size:]
        
        rank = sum(1 for r in recent_history if r <= reward)
        percentile = (rank / len(recent_history)) * 100
        
        return percentile
    
    def _generate_reasoning(self, reward_type: RewardType, score: float) -> str:
        """Generate reasoning explanation for a reward component."""
        if reward_type == RewardType.QUALITY:
            if score > 0.8:
                return "High quality reasoning with clear logic and evidence."
            elif score > 0.6:
                return "Good quality reasoning with minor issues."
            elif score > 0.4:
                return "Moderate quality with some logical gaps."
            else:
                return "Low quality reasoning with significant issues."
        
        elif reward_type == RewardType.RELEVANCE:
            if score > 0.8:
                return "Highly relevant to the query and context."
            elif score > 0.6:
                return "Generally relevant with good connection to query."
            elif score > 0.4:
                return "Somewhat relevant but could be more focused."
            else:
                return "Limited relevance to the main query."
        
        elif reward_type == RewardType.COHERENCE:
            if score > 0.8:
                return "Excellent coherence with clear logical flow."
            elif score > 0.6:
                return "Good coherence with mostly clear connections."
            elif score > 0.4:
                return "Moderate coherence with some unclear transitions."
            else:
                return "Poor coherence with confusing logical flow."
        
        elif reward_type == RewardType.PROGRESS:
            if score > 0.8:
                return "Significant progress toward solving the problem."
            elif score > 0.6:
                return "Good progress with meaningful advancement."
            elif score > 0.4:
                return "Some progress but could advance more."
            else:
                return "Limited progress toward the solution."
        
        elif reward_type == RewardType.EFFICIENCY:
            if score > 0.8:
                return "Highly efficient reasoning with optimal steps."
            elif score > 0.6:
                return "Good efficiency with reasonable step count."
            elif score > 0.4:
                return "Moderate efficiency with some redundancy."
            else:
                return "Low efficiency with excessive steps."
        
        elif reward_type == RewardType.SAFETY:
            if score > 0.8:
                return "Safe reasoning with no harmful content."
            elif score > 0.6:
                return "Generally safe with minor concerns."
            elif score > 0.4:
                return "Some safety concerns present."
            else:
                return "Significant safety issues detected."
        
        elif reward_type == RewardType.CONSISTENCY:
            if score > 0.8:
                return "Highly consistent with previous reasoning."
            elif score > 0.6:
                return "Generally consistent with minor discrepancies."
            elif score > 0.4:
                return "Moderate consistency with some conflicts."
            else:
                return "Poor consistency with contradictions."
        
        elif reward_type == RewardType.NOVELTY:
            if score > 0.8:
                return "Novel approach with creative insights."
            elif score > 0.6:
                return "Some novelty with interesting perspectives."
            elif score > 0.4:
                return "Limited novelty but adequate approach."
            else:
                return "Standard approach with no novel elements."
        
        return "Reward component evaluation completed."
    
    def _extract_step_context(self, step: ReasoningStep) -> Dict[str, Any]:
        """Extract context information from a reasoning step."""
        return {
            "step_type": step.step_type.value,
            "content_length": len(step.content),
            "confidence": step.confidence,
            "timestamp": step.timestamp,
            "has_parent": step.parent_step_id is not None,
            "num_children": len(step.child_step_ids),
            "num_dependencies": len(step.dependencies),
            "generation_time": step.generation_time_ms,
            "tokens_generated": step.tokens_generated
        }
    
    def _extract_trace_context(self, trace: ReasoningTrace) -> Dict[str, Any]:
        """Extract context information from the reasoning trace."""
        return {
            "strategy": trace.strategy.value,
            "total_steps": trace.total_steps,
            "max_depth": trace.max_depth,
            "successful_steps": trace.successful_steps,
            "backtrack_count": trace.backtrack_count,
            "branch_count": trace.branch_count,
            "query_length": len(trace.query),
            "runtime_seconds": (time.time() - trace.start_time) if trace.end_time is None else (trace.end_time - trace.start_time)
        }
    
    def batch_evaluate_steps(
        self,
        steps: List[ReasoningStep],
        trace_context: ReasoningTrace,
        tokenizer: Optional[Any] = None
    ) -> List[ProcessReward]:
        """Evaluate multiple steps in a batch for efficiency."""
        if not steps:
            return []
        
        # Prepare batch inputs
        batch_tokens = []
        batch_types = []
        batch_positions = []
        batch_masks = []
        
        for step in steps:
            tokens, types, positions, mask = self._prepare_step_input(
                step, trace_context, tokenizer
            )
            batch_tokens.append(tokens.squeeze(0))
            batch_types.append(types.squeeze(0))
            batch_positions.append(positions.squeeze(0))
            batch_masks.append(mask.squeeze(0))
        
        # Stack into batch tensors
        batch_tokens_tensor = torch.stack(batch_tokens)
        batch_types_tensor = torch.stack(batch_types)
        batch_positions_tensor = torch.stack(batch_positions)
        batch_masks_tensor = torch.stack(batch_masks)
        
        with torch.no_grad():
            # Batch forward pass
            total_rewards, confidences, component_rewards = self.forward(
                batch_tokens_tensor, batch_types_tensor, batch_positions_tensor, batch_masks_tensor
            )
            
            # Create process rewards for each step
            process_rewards = []
            for i, step in enumerate(steps):
                # Extract individual results
                total_reward_val = total_rewards[i].item()
                confidence_val = confidences[i].item()
                
                # Create reward signals
                reward_signals = {}
                for reward_type_str, reward_tensor in component_rewards.items():
                    reward_type = RewardType(reward_type_str)
                    reward_signals[reward_type] = RewardSignal(
                        reward_type=reward_type,
                        score=reward_tensor[i].item(),
                        confidence=confidence_val,
                        reasoning=self._generate_reasoning(reward_type, reward_tensor[i].item())
                    )
                
                # Create process reward
                process_reward = ProcessReward(
                    step_id=step.step_id,
                    total_reward=total_reward_val,
                    component_rewards=reward_signals,
                    normalized_reward=self._normalize_reward(total_reward_val),
                    percentile_rank=self._calculate_percentile_rank(total_reward_val),
                    step_context=self._extract_step_context(step),
                    trace_context=self._extract_trace_context(trace_context),
                    model_confidence=confidence_val
                )
                
                process_rewards.append(process_reward)
                
                # Update history
                self.reward_history.append(total_reward_val)
            
            return process_rewards
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reward statistics."""
        if not self.reward_history:
            return {"message": "No reward history available"}
        
        rewards = list(self.reward_history)
        
        return {
            "total_evaluations": len(rewards),
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "median_reward": np.median(rewards),
            "percentiles": {
                "25th": np.percentile(rewards, 25),
                "75th": np.percentile(rewards, 75),
                "90th": np.percentile(rewards, 90),
                "95th": np.percentile(rewards, 95)
            },
            "recent_trend": {
                "last_10_mean": np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards),
                "improvement": self._calculate_trend_improvement(rewards)
            }
        }
    
    def _calculate_trend_improvement(self, rewards: List[float]) -> float:
        """Calculate improvement trend in recent rewards."""
        if len(rewards) < 20:
            return 0.0
        
        # Compare recent half vs older half
        mid_point = len(rewards) // 2
        older_half = rewards[:mid_point]
        recent_half = rewards[mid_point:]
        
        older_mean = np.mean(older_half)
        recent_mean = np.mean(recent_half)
        
        if older_mean == 0:
            return 0.0
        
        improvement = (recent_mean - older_mean) / older_mean
        return improvement
    
    def save_model(self, filepath: str):
        """Save the reward model state."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'reward_history': list(self.reward_history),
            'vocab_size': self.vocab_size
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'ProcessRewardModel':
        """Load a reward model from file."""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        model = cls(checkpoint['config'], checkpoint['vocab_size'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.reward_history.extend(checkpoint['reward_history'])
        
        return model