"""Reinforcement Learning Controller for SEAL architecture.

This module implements RL optimization for self-editing operations using
PPO/REINFORCE algorithms with comprehensive reward systems and performance
monitoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
import numpy as np
import time
from collections import defaultdict, deque
import json
from pathlib import Path
import warnings
import math
from enum import Enum
import logging

from .parameter_diff import ParameterDiff
from .edit_validation import ValidationResult
from .performance_analyzer import PerformanceSnapshot, PerformanceImpact


class ExplorationStrategy(Enum):
    """Exploration strategies for RL controller."""
    EPSILON_GREEDY = "epsilon_greedy"
    BOLTZMANN = "boltzmann"
    UCB = "ucb"
    THOMPSON_SAMPLING = "thompson_sampling"
    ENTROPY_REGULARIZED = "entropy_regularized"


@dataclass
class RLConfig:
    """Configuration for RL controller."""
    # PPO parameters
    ppo_epochs: int = 4
    ppo_clip_range: float = 0.2
    ppo_entropy_coeff: float = 0.01
    ppo_value_loss_coeff: float = 0.5
    ppo_max_grad_norm: float = 0.5
    
    # REINFORCE parameters
    reinforce_baseline: bool = True
    reinforce_gamma: float = 0.99
    
    # Learning parameters
    learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 10000
    update_frequency: int = 100
    
    # Exploration
    exploration_strategy: ExplorationStrategy = ExplorationStrategy.EPSILON_GREEDY
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    temperature: float = 1.0
    ucb_c: float = 2.0
    
    # Reward system
    performance_weight: float = 0.4
    stability_weight: float = 0.3
    efficiency_weight: float = 0.2
    safety_weight: float = 0.1
    
    # Credit assignment
    temporal_discount: float = 0.95
    edit_contribution_window: int = 10
    
    # Hyperparameter optimization
    auto_tune_hyperparams: bool = True
    tune_frequency: int = 1000
    hyperopt_trials: int = 50


@dataclass
class RewardSignal:
    """Reward signal for RL training."""
    timestamp: float
    edit_id: str
    
    # Primary rewards
    performance_reward: float = 0.0
    stability_reward: float = 0.0
    efficiency_reward: float = 0.0
    safety_reward: float = 0.0
    
    # Meta information
    edit_magnitude: float = 0.0
    parameter_count: int = 0
    validation_passed: bool = True
    
    # Context
    task_type: str = ""
    difficulty_level: float = 1.0
    
    @property
    def total_reward(self) -> float:
        """Calculate weighted total reward."""
        return (
            self.performance_reward +
            self.stability_reward +
            self.efficiency_reward +
            self.safety_reward
        )


@dataclass
class RLState:
    """State representation for RL agent."""
    # Model state
    model_parameters: torch.Tensor
    parameter_gradients: Optional[torch.Tensor] = None
    layer_activations: Optional[torch.Tensor] = None
    
    # Performance state
    current_performance: Dict[str, float] = field(default_factory=dict)
    performance_history: List[float] = field(default_factory=list)
    
    # Edit history
    recent_edits: List[str] = field(default_factory=list)
    edit_success_rate: float = 0.0
    
    # Resource state
    memory_usage: float = 0.0
    compute_load: float = 0.0
    
    # Task context
    current_task: str = ""
    task_difficulty: float = 1.0
    
    def to_tensor(self) -> torch.Tensor:
        """Convert state to tensor representation."""
        features = []
        
        # Parameter statistics
        if self.model_parameters is not None:
            param_stats = torch.tensor([
                self.model_parameters.mean().item(),
                self.model_parameters.std().item(),
                self.model_parameters.abs().max().item(),
                self.model_parameters.abs().min().item()
            ])
            features.append(param_stats)
        
        # Performance features
        perf_features = torch.tensor([
            self.current_performance.get('accuracy', 0.0),
            self.current_performance.get('loss', 0.0),
            self.current_performance.get('perplexity', 0.0),
            len(self.performance_history),
            np.mean(self.performance_history[-10:]) if self.performance_history else 0.0
        ])
        features.append(perf_features)
        
        # Edit features
        edit_features = torch.tensor([
            len(self.recent_edits),
            self.edit_success_rate,
            self.memory_usage,
            self.compute_load,
            self.task_difficulty
        ])
        features.append(edit_features)
        
        return torch.cat(features)


class PolicyNetwork(nn.Module):
    """Policy network for RL agent."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.network(state), dim=-1)


class ValueNetwork(nn.Module):
    """Value network for RL agent."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class ExplorationManager:
    """Manages exploration strategies for RL agent."""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.epsilon = config.epsilon_start
        self.action_counts = defaultdict(int)
        self.action_values = defaultdict(float)
        self.total_steps = 0
        
    def select_action(
        self,
        action_probs: torch.Tensor,
        state: RLState,
        available_actions: List[int]
    ) -> int:
        """Select action based on exploration strategy."""
        self.total_steps += 1
        
        if self.config.exploration_strategy == ExplorationStrategy.EPSILON_GREEDY:
            return self._epsilon_greedy(action_probs, available_actions)
        elif self.config.exploration_strategy == ExplorationStrategy.BOLTZMANN:
            return self._boltzmann(action_probs, available_actions)
        elif self.config.exploration_strategy == ExplorationStrategy.UCB:
            return self._ucb(action_probs, available_actions)
        elif self.config.exploration_strategy == ExplorationStrategy.THOMPSON_SAMPLING:
            return self._thompson_sampling(action_probs, available_actions)
        elif self.config.exploration_strategy == ExplorationStrategy.ENTROPY_REGULARIZED:
            return self._entropy_regularized(action_probs, available_actions)
        else:
            # Default to greedy
            return available_actions[torch.argmax(action_probs[available_actions]).item()]
    
    def _epsilon_greedy(self, action_probs: torch.Tensor, available_actions: List[int]) -> int:
        """Epsilon-greedy exploration."""
        if np.random.random() < self.epsilon:
            return np.random.choice(available_actions)
        else:
            return available_actions[torch.argmax(action_probs[available_actions]).item()]
    
    def _boltzmann(self, action_probs: torch.Tensor, available_actions: List[int]) -> int:
        """Boltzmann exploration."""
        available_probs = action_probs[available_actions] / self.config.temperature
        available_probs = F.softmax(available_probs, dim=0)
        return np.random.choice(available_actions, p=available_probs.detach().numpy())
    
    def _ucb(self, action_probs: torch.Tensor, available_actions: List[int]) -> int:
        """Upper Confidence Bound exploration."""
        ucb_values = []
        for action in available_actions:
            count = max(self.action_counts[action], 1)
            confidence = self.config.ucb_c * math.sqrt(math.log(self.total_steps) / count)
            ucb_value = self.action_values[action] + confidence
            ucb_values.append(ucb_value)
        
        return available_actions[np.argmax(ucb_values)]
    
    def _thompson_sampling(self, action_probs: torch.Tensor, available_actions: List[int]) -> int:
        """Thompson sampling exploration."""
        sampled_values = []
        for action in available_actions:
            # Assume beta distribution for action values
            alpha = max(self.action_counts[action], 1)
            beta = max(self.total_steps - self.action_counts[action], 1)
            sampled_value = np.random.beta(alpha, beta)
            sampled_values.append(sampled_value)
        
        return available_actions[np.argmax(sampled_values)]
    
    def _entropy_regularized(self, action_probs: torch.Tensor, available_actions: List[int]) -> int:
        """Entropy-regularized exploration."""
        available_probs = action_probs[available_actions]
        entropy = -torch.sum(available_probs * torch.log(available_probs + 1e-8))
        
        # Add entropy bonus to probabilities
        entropy_bonus = self.config.ppo_entropy_coeff * entropy
        adjusted_probs = available_probs + entropy_bonus
        
        return available_actions[torch.argmax(adjusted_probs).item()]
    
    def update_epsilon(self):
        """Update epsilon for epsilon-greedy."""
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )
    
    def update_action_stats(self, action: int, reward: float):
        """Update action statistics."""
        self.action_counts[action] += 1
        # Running average of action values
        count = self.action_counts[action]
        self.action_values[action] += (reward - self.action_values[action]) / count


class RewardSystem:
    """Comprehensive reward system for edit evaluation."""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.baseline_performance = {}
        self.stability_history = deque(maxlen=100)
        
    def calculate_reward(
        self,
        edit_id: str,
        performance_before: PerformanceSnapshot,
        performance_after: PerformanceSnapshot,
        validation_result: ValidationResult,
        edit_magnitude: float,
        resource_cost: Dict[str, float]
    ) -> RewardSignal:
        """Calculate comprehensive reward signal."""
        
        reward = RewardSignal(
            timestamp=time.time(),
            edit_id=edit_id,
            edit_magnitude=edit_magnitude,
            validation_passed=validation_result.is_valid
        )
        
        # Performance reward
        reward.performance_reward = self._calculate_performance_reward(
            performance_before, performance_after
        ) * self.config.performance_weight
        
        # Stability reward
        reward.stability_reward = self._calculate_stability_reward(
            performance_before, performance_after, validation_result
        ) * self.config.stability_weight
        
        # Efficiency reward
        reward.efficiency_reward = self._calculate_efficiency_reward(
            edit_magnitude, resource_cost, reward.performance_reward
        ) * self.config.efficiency_weight
        
        # Safety reward
        reward.safety_reward = self._calculate_safety_reward(
            validation_result
        ) * self.config.safety_weight
        
        return reward
    
    def _calculate_performance_reward(
        self,
        before: PerformanceSnapshot,
        after: PerformanceSnapshot
    ) -> float:
        """Calculate performance improvement reward."""
        performance_delta = 0.0
        
        # Compare key metrics
        for metric in ['accuracy', 'loss', 'perplexity']:
            before_val = before.metrics.get(metric, 0.0)
            after_val = after.metrics.get(metric, 0.0)
            
            if metric == 'accuracy':
                # Higher is better
                delta = after_val - before_val
            else:
                # Lower is better (loss, perplexity)
                delta = before_val - after_val
            
            performance_delta += delta
        
        # Normalize and apply sigmoid to bound reward
        return torch.sigmoid(torch.tensor(performance_delta)).item() * 2 - 1
    
    def _calculate_stability_reward(
        self,
        before: PerformanceSnapshot,
        after: PerformanceSnapshot,
        validation_result: ValidationResult
    ) -> float:
        """Calculate stability reward."""
        if not validation_result.is_valid:
            return -1.0
        
        # Check for performance variance
        variance_penalty = 0.0
        for metric in ['accuracy', 'loss']:
            before_val = before.metrics.get(metric, 0.0)
            after_val = after.metrics.get(metric, 0.0)
            variance = abs(after_val - before_val)
            variance_penalty += variance
        
        # Penalize high variance
        stability_score = 1.0 - min(variance_penalty, 1.0)
        
        self.stability_history.append(stability_score)
        
        return stability_score
    
    def _calculate_efficiency_reward(
        self,
        edit_magnitude: float,
        resource_cost: Dict[str, float],
        performance_gain: float
    ) -> float:
        """Calculate efficiency reward (performance gain per resource cost)."""
        if performance_gain <= 0:
            return -0.5
        
        # Calculate total resource cost
        total_cost = sum(resource_cost.values()) + edit_magnitude
        
        if total_cost == 0:
            return 1.0
        
        # Efficiency = performance gain per unit cost
        efficiency = performance_gain / total_cost
        
        return min(efficiency, 1.0)
    
    def _calculate_safety_reward(self, validation_result: ValidationResult) -> float:
        """Calculate safety reward based on validation."""
        if not validation_result.is_valid:
            return -1.0
        
        # Reward based on confidence of validation
        confidence = getattr(validation_result, 'confidence', 1.0)
        
        return confidence


class CreditAssignmentSystem:
    """Credit assignment system for tracking edit contributions."""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.edit_history = deque(maxlen=config.edit_contribution_window)
        self.performance_history = deque(maxlen=config.edit_contribution_window)
        
    def assign_credit(
        self,
        current_reward: float,
        edit_sequence: List[str]
    ) -> Dict[str, float]:
        """Assign credit to recent edits for current reward."""
        credits = {}
        
        if not edit_sequence:
            return credits
        
        # Temporal discounting
        for i, edit_id in enumerate(reversed(edit_sequence)):
            discount_factor = self.config.temporal_discount ** i
            credit = current_reward * discount_factor
            credits[edit_id] = credit
        
        return credits
    
    def update_history(self, edit_id: str, performance: float):
        """Update edit and performance history."""
        self.edit_history.append(edit_id)
        self.performance_history.append(performance)
    
    def get_edit_contributions(self) -> Dict[str, float]:
        """Get contribution scores for recent edits."""
        if len(self.edit_history) < 2:
            return {}
        
        contributions = {}
        
        # Calculate performance deltas
        for i in range(1, len(self.performance_history)):
            edit_id = self.edit_history[i-1]
            performance_delta = self.performance_history[i] - self.performance_history[i-1]
            
            if edit_id not in contributions:
                contributions[edit_id] = []
            contributions[edit_id].append(performance_delta)
        
        # Average contributions
        for edit_id in contributions:
            contributions[edit_id] = np.mean(contributions[edit_id])
        
        return contributions


class StabilityMonitor:
    """Monitor system stability during RL training."""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.performance_buffer = deque(maxlen=100)
        self.loss_buffer = deque(maxlen=100)
        self.gradient_norms = deque(maxlen=100)
        self.instability_count = 0
        
    def update(
        self,
        performance: float,
        loss: float,
        gradient_norm: Optional[float] = None
    ):
        """Update stability metrics."""
        self.performance_buffer.append(performance)
        self.loss_buffer.append(loss)
        
        if gradient_norm is not None:
            self.gradient_norms.append(gradient_norm)
    
    def check_stability(self) -> Tuple[bool, Dict[str, float]]:
        """Check if training is stable."""
        if len(self.performance_buffer) < 10:
            return True, {}
        
        metrics = {}
        
        # Performance variance
        perf_variance = np.var(list(self.performance_buffer)[-10:])
        metrics['performance_variance'] = perf_variance
        
        # Loss variance
        loss_variance = np.var(list(self.loss_buffer)[-10:])
        metrics['loss_variance'] = loss_variance
        
        # Gradient norm variance
        if self.gradient_norms:
            grad_variance = np.var(list(self.gradient_norms)[-10:])
            metrics['gradient_variance'] = grad_variance
        
        # Check for instability
        is_stable = (
            perf_variance < 0.1 and
            loss_variance < 1.0 and
            (not self.gradient_norms or grad_variance < 10.0)
        )
        
        if not is_stable:
            self.instability_count += 1
        else:
            self.instability_count = max(0, self.instability_count - 1)
        
        return is_stable, metrics
    
    def is_critically_unstable(self) -> bool:
        """Check if system is critically unstable."""
        return self.instability_count > 10


class HyperparameterOptimizer:
    """Automatic hyperparameter optimization for RL controller."""
    
    def __init__(self, config: RLConfig):
        self.config = config
        self.optimization_history = []
        self.best_config = None
        self.best_performance = float('-inf')
        
    def suggest_hyperparameters(self) -> Dict[str, Any]:
        """Suggest new hyperparameters based on Bayesian optimization."""
        # Simple random search for now (can be replaced with more sophisticated methods)
        suggestions = {
            'learning_rate': np.random.uniform(1e-5, 1e-2),
            'ppo_clip_range': np.random.uniform(0.1, 0.3),
            'ppo_entropy_coeff': np.random.uniform(0.001, 0.1),
            'epsilon_decay': np.random.uniform(0.99, 0.999),
            'temperature': np.random.uniform(0.5, 2.0)
        }
        
        return suggestions
    
    def update_performance(
        self,
        hyperparams: Dict[str, Any],
        performance: float
    ):
        """Update optimization history with performance result."""
        self.optimization_history.append({
            'hyperparams': hyperparams,
            'performance': performance,
            'timestamp': time.time()
        })
        
        if performance > self.best_performance:
            self.best_performance = performance
            self.best_config = hyperparams.copy()
    
    def should_optimize(self, step: int) -> bool:
        """Check if hyperparameter optimization should be triggered."""
        return (
            self.config.auto_tune_hyperparams and
            step > 0 and
            step % self.config.tune_frequency == 0
        )


class PerformanceMetricsCollector:
    """Collect and analyze performance metrics for RL training."""
    
    def __init__(self):
        self.metrics_history = defaultdict(list)
        self.aggregated_metrics = {}
        
    def record_metric(self, name: str, value: float, timestamp: Optional[float] = None):
        """Record a performance metric."""
        if timestamp is None:
            timestamp = time.time()
        
        self.metrics_history[name].append({
            'value': value,
            'timestamp': timestamp
        })
    
    def get_metric_statistics(self, name: str, window_size: int = 100) -> Dict[str, float]:
        """Get statistics for a metric over recent window."""
        if name not in self.metrics_history:
            return {}
        
        recent_values = [
            entry['value'] 
            for entry in self.metrics_history[name][-window_size:]
        ]
        
        if not recent_values:
            return {}
        
        return {
            'mean': np.mean(recent_values),
            'std': np.std(recent_values),
            'min': np.min(recent_values),
            'max': np.max(recent_values),
            'trend': self._calculate_trend(recent_values)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for values."""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.001:
            return 'increasing'
        elif slope < -0.001:
            return 'decreasing'
        else:
            return 'stable'
    
    def export_metrics(self, filepath: Path):
        """Export metrics to file."""
        export_data = {
            'metrics_history': dict(self.metrics_history),
            'aggregated_metrics': self.aggregated_metrics,
            'export_timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)


class RLController:
    """Main RL controller for SEAL architecture optimization."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[RLConfig] = None
    ):
        self.config = config or RLConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Networks
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        self.old_policy_net = PolicyNetwork(state_dim, action_dim)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.config.learning_rate
        )
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(),
            lr=self.config.learning_rate
        )
        
        # Components
        self.exploration_manager = ExplorationManager(self.config)
        self.reward_system = RewardSystem(self.config)
        self.credit_assignment = CreditAssignmentSystem(self.config)
        self.stability_monitor = StabilityMonitor(self.config)
        self.hyperopt = HyperparameterOptimizer(self.config)
        self.metrics_collector = PerformanceMetricsCollector()
        
        # Training state
        self.experience_buffer = []
        self.training_step = 0
        self.episode_count = 0
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def select_action(
        self,
        state: RLState,
        available_actions: Optional[List[int]] = None
    ) -> Tuple[int, float]:
        """Select action using current policy."""
        if available_actions is None:
            available_actions = list(range(self.action_dim))
        
        state_tensor = state.to_tensor().unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor).squeeze(0)
            value = self.value_net(state_tensor).squeeze(0)
        
        # Use exploration manager to select action
        action = self.exploration_manager.select_action(
            action_probs, state, available_actions
        )
        
        action_prob = action_probs[action].item()
        
        return action, action_prob
    
    def store_experience(
        self,
        state: RLState,
        action: int,
        action_prob: float,
        reward: RewardSignal,
        next_state: Optional[RLState] = None,
        done: bool = False
    ):
        """Store experience in replay buffer."""
        experience = {
            'state': state.to_tensor(),
            'action': action,
            'action_prob': action_prob,
            'reward': reward.total_reward,
            'reward_signal': reward,
            'next_state': next_state.to_tensor() if next_state else None,
            'done': done,
            'timestamp': time.time()
        }
        
        self.experience_buffer.append(experience)
        
        # Limit buffer size
        if len(self.experience_buffer) > self.config.buffer_size:
            self.experience_buffer.pop(0)
        
        # Update exploration stats
        self.exploration_manager.update_action_stats(action, reward.total_reward)
        
        # Update credit assignment
        if hasattr(reward, 'edit_id'):
            performance = reward.performance_reward
            self.credit_assignment.update_history(reward.edit_id, performance)
        
        # Record metrics
        self.metrics_collector.record_metric('reward', reward.total_reward)
        self.metrics_collector.record_metric('action_prob', action_prob)
    
    def train_step(self) -> Dict[str, float]:
        """Perform one training step."""
        if len(self.experience_buffer) < self.config.batch_size:
            return {}
        
        # Sample batch
        batch_indices = np.random.choice(
            len(self.experience_buffer),
            size=self.config.batch_size,
            replace=False
        )
        batch = [self.experience_buffer[i] for i in batch_indices]
        
        # Convert to tensors
        states = torch.stack([exp['state'] for exp in batch])
        actions = torch.tensor([exp['action'] for exp in batch], dtype=torch.long)
        old_action_probs = torch.tensor([exp['action_prob'] for exp in batch])
        rewards = torch.tensor([exp['reward'] for exp in batch])
        
        # Calculate returns and advantages
        returns = self._calculate_returns(batch)
        values = self.value_net(states).squeeze(1)
        advantages = returns - values.detach()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        training_metrics = {}
        
        # PPO training
        if self.config.ppo_epochs > 0:
            ppo_metrics = self._train_ppo(
                states, actions, old_action_probs, advantages, returns
            )
            training_metrics.update(ppo_metrics)
        
        # Update exploration
        self.exploration_manager.update_epsilon()
        
        # Monitor stability
        avg_loss = training_metrics.get('policy_loss', 0.0)
        avg_performance = np.mean([exp['reward'] for exp in batch])
        gradient_norm = training_metrics.get('gradient_norm', 0.0)
        
        self.stability_monitor.update(avg_performance, avg_loss, gradient_norm)
        is_stable, stability_metrics = self.stability_monitor.check_stability()
        training_metrics.update(stability_metrics)
        
        # Check for critical instability
        if self.stability_monitor.is_critically_unstable():
            self.logger.warning("Critical instability detected. Reducing learning rate.")
            for param_group in self.policy_optimizer.param_groups:
                param_group['lr'] *= 0.5
            for param_group in self.value_optimizer.param_groups:
                param_group['lr'] *= 0.5
        
        # Hyperparameter optimization
        if self.hyperopt.should_optimize(self.training_step):
            self._optimize_hyperparameters(avg_performance)
        
        self.training_step += 1
        
        # Record training metrics
        for name, value in training_metrics.items():
            self.metrics_collector.record_metric(f'training_{name}', value)
        
        return training_metrics
    
    def _calculate_returns(self, batch: List[Dict]) -> torch.Tensor:
        """Calculate discounted returns."""
        returns = []
        
        for i, exp in enumerate(batch):
            G = 0
            for j in range(i, len(batch)):
                if batch[j]['done']:
                    break
                discount = self.config.reinforce_gamma ** (j - i)
                G += discount * batch[j]['reward']
            returns.append(G)
        
        return torch.tensor(returns)
    
    def _train_ppo(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_action_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> Dict[str, float]:
        """Train using PPO algorithm."""
        metrics = {}
        
        # Copy current policy to old policy
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())
        
        for epoch in range(self.config.ppo_epochs):
            # Policy network forward pass
            new_action_probs = self.policy_net(states)
            new_action_probs = new_action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Calculate probability ratio
            ratio = new_action_probs / (old_action_probs + 1e-8)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1.0 - self.config.ppo_clip_range,
                1.0 + self.config.ppo_clip_range
            ) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Entropy bonus
            action_dist = Categorical(self.policy_net(states))
            entropy = action_dist.entropy().mean()
            policy_loss -= self.config.ppo_entropy_coeff * entropy
            
            # Value network
            values = self.value_net(states).squeeze(1)
            value_loss = F.mse_loss(values, returns)
            
            # Combined loss
            total_loss = policy_loss + self.config.ppo_value_loss_coeff * value_loss
            
            # Backward pass
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            policy_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(),
                self.config.ppo_max_grad_norm
            )
            value_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.value_net.parameters(),
                self.config.ppo_max_grad_norm
            )
            
            self.policy_optimizer.step()
            self.value_optimizer.step()
            
            # Record metrics
            if epoch == self.config.ppo_epochs - 1:  # Last epoch
                metrics.update({
                    'policy_loss': policy_loss.item(),
                    'value_loss': value_loss.item(),
                    'entropy': entropy.item(),
                    'gradient_norm': max(policy_grad_norm, value_grad_norm),
                    'clip_fraction': ((ratio < 1.0 - self.config.ppo_clip_range) | 
                                    (ratio > 1.0 + self.config.ppo_clip_range)).float().mean().item()
                })
        
        return metrics
    
    def _optimize_hyperparameters(self, current_performance: float):
        """Optimize hyperparameters based on current performance."""
        # Get current hyperparameters
        current_hyperparams = {
            'learning_rate': self.policy_optimizer.param_groups[0]['lr'],
            'ppo_clip_range': self.config.ppo_clip_range,
            'ppo_entropy_coeff': self.config.ppo_entropy_coeff,
            'epsilon_decay': self.config.epsilon_decay,
            'temperature': self.config.temperature
        }
        
        # Update optimization history
        self.hyperopt.update_performance(current_hyperparams, current_performance)
        
        # Suggest new hyperparameters
        new_hyperparams = self.hyperopt.suggest_hyperparameters()
        
        # Apply suggestions
        for param_group in self.policy_optimizer.param_groups:
            param_group['lr'] = new_hyperparams['learning_rate']
        for param_group in self.value_optimizer.param_groups:
            param_group['lr'] = new_hyperparams['learning_rate']
        
        self.config.ppo_clip_range = new_hyperparams['ppo_clip_range']
        self.config.ppo_entropy_coeff = new_hyperparams['ppo_entropy_coeff']
        self.config.epsilon_decay = new_hyperparams['epsilon_decay']
        self.config.temperature = new_hyperparams['temperature']
        
        self.logger.info(f"Updated hyperparameters: {new_hyperparams}")
    
    def evaluate_policy(
        self,
        eval_states: List[RLState],
        num_episodes: int = 10
    ) -> Dict[str, float]:
        """Evaluate current policy performance."""
        self.policy_net.eval()
        self.value_net.eval()
        
        total_rewards = []
        episode_lengths = []
        
        with torch.no_grad():
            for episode in range(num_episodes):
                episode_reward = 0
                episode_length = 0
                
                for state in eval_states:
                    action, _ = self.select_action(state, available_actions=None)
                    # Note: In real evaluation, you'd step the environment here
                    # For now, we'll simulate with a dummy reward
                    episode_reward += np.random.normal(0, 0.1)
                    episode_length += 1
                
                total_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
        
        self.policy_net.train()
        self.value_net.train()
        
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'success_rate': np.mean([r > 0 for r in total_rewards])
        }
    
    def save_checkpoint(self, filepath: Path):
        """Save RL controller checkpoint."""
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'config': self.config.__dict__,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'exploration_manager_state': {
                'epsilon': self.exploration_manager.epsilon,
                'action_counts': dict(self.exploration_manager.action_counts),
                'action_values': dict(self.exploration_manager.action_values),
                'total_steps': self.exploration_manager.total_steps
            },
            'hyperopt_state': {
                'optimization_history': self.hyperopt.optimization_history,
                'best_config': self.hyperopt.best_config,
                'best_performance': self.hyperopt.best_performance
            }
        }
        
        torch.save(checkpoint, filepath)
        
        # Save metrics separately
        metrics_path = filepath.parent / f"{filepath.stem}_metrics.json"
        self.metrics_collector.export_metrics(metrics_path)
    
    def load_checkpoint(self, filepath: Path):
        """Load RL controller checkpoint."""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
        
        # Restore exploration manager state
        exp_state = checkpoint['exploration_manager_state']
        self.exploration_manager.epsilon = exp_state['epsilon']
        self.exploration_manager.action_counts = defaultdict(int, exp_state['action_counts'])
        self.exploration_manager.action_values = defaultdict(float, exp_state['action_values'])
        self.exploration_manager.total_steps = exp_state['total_steps']
        
        # Restore hyperopt state
        hyperopt_state = checkpoint['hyperopt_state']
        self.hyperopt.optimization_history = hyperopt_state['optimization_history']
        self.hyperopt.best_config = hyperopt_state['best_config']
        self.hyperopt.best_performance = hyperopt_state['best_performance']
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'exploration_epsilon': self.exploration_manager.epsilon,
            'stability_status': self.stability_monitor.check_stability()[0],
            'best_hyperparams': self.hyperopt.best_config,
            'best_performance': self.hyperopt.best_performance
        }
        
        # Add recent metric statistics
        for metric_name in ['reward', 'action_prob', 'training_policy_loss']:
            stats = self.metrics_collector.get_metric_statistics(metric_name)
            if stats:
                summary[f'{metric_name}_stats'] = stats
        
        return summary


__all__ = [
    'RLController',
    'RLConfig',
    'RLState',
    'RewardSignal',
    'ExplorationStrategy',
    'ExplorationManager',
    'RewardSystem',
    'CreditAssignmentSystem',
    'StabilityMonitor',
    'HyperparameterOptimizer',
    'PerformanceMetricsCollector',
    'PolicyNetwork',
    'ValueNetwork'
]