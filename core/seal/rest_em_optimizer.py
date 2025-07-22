"""ReST-EM (Reinforced Self-Training with Expectation-Maximization) optimizer.

This module implements the core optimization algorithm for SEAL that enables
self-improvement through iterative refinement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
from dataclasses import dataclass, field
import logging
from collections import deque
import copy

from .edit_format import EditInstruction, EditBatch
from .edit_generator import EditGenerator, EditContext
from .weight_updater import WeightUpdater, UpdateConfig

logger = logging.getLogger(__name__)


@dataclass
class ReSTEMConfig:
    """Configuration for ReST-EM optimizer."""
    # E-step parameters
    num_samples: int = 10  # Number of edits to sample
    temperature: float = 1.0  # Sampling temperature
    top_k: int = 50  # Top-k sampling
    
    # M-step parameters
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    
    # Optimization parameters
    em_iterations: int = 10  # Number of EM iterations
    inner_steps: int = 5  # Steps per M-step
    patience: int = 3  # Early stopping patience
    
    # Validation parameters
    validation_split: float = 0.1
    validation_frequency: int = 1
    
    # Memory and efficiency
    accumulation_steps: int = 4
    checkpoint_frequency: int = 5
    

@dataclass
class OptimizationState:
    """State of the optimization process."""
    iteration: int = 0
    best_loss: float = float('inf')
    best_model_state: Optional[Dict[str, torch.Tensor]] = None
    patience_counter: int = 0
    metrics_history: List[Dict[str, float]] = field(default_factory=list)
    edit_history: List[EditBatch] = field(default_factory=list)
    

class ReSTEMOptimizer:
    """Reinforced Self-Training with Expectation-Maximization optimizer."""
    
    def __init__(
        self,
        model: nn.Module,
        edit_generator: EditGenerator,
        weight_updater: WeightUpdater,
        config: Optional[ReSTEMConfig] = None
    ):
        self.model = model
        self.edit_generator = edit_generator
        self.weight_updater = weight_updater
        self.config = config or ReSTEMConfig()
        
        # Optimization state
        self.state = OptimizationState()
        
        # PyTorch optimizer for M-step
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.momentum, 0.999),
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.em_iterations
        )
        
        # Edit buffer for accumulation
        self.edit_buffer = deque(maxlen=100)
        
    def optimize(
        self,
        train_data: List[Dict[str, Any]],
        validation_data: Optional[List[Dict[str, Any]]] = None,
        num_iterations: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run ReST-EM optimization.
        
        Args:
            train_data: Training examples
            validation_data: Validation examples
            num_iterations: Number of EM iterations
            
        Returns:
            Optimization results and metrics
        """
        num_iterations = num_iterations or self.config.em_iterations
        
        logger.info(f"Starting ReST-EM optimization for {num_iterations} iterations")
        
        for iteration in range(num_iterations):
            self.state.iteration = iteration
            
            # E-step: Generate self-edits
            edit_batch = self._e_step(train_data)
            
            # M-step: Update model
            metrics = self._m_step(edit_batch, train_data)
            
            # Validation
            if validation_data and iteration % self.config.validation_frequency == 0:
                val_metrics = self._validate(validation_data)
                metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            
            # Update state
            self._update_state(metrics)
            
            # Early stopping check
            if self._should_stop():
                logger.info(f"Early stopping at iteration {iteration}")
                break
            
            # Checkpoint
            if iteration % self.config.checkpoint_frequency == 0:
                self._save_checkpoint()
        
        return self._get_optimization_summary()
    
    def _e_step(self, train_data: List[Dict[str, Any]]) -> EditBatch:
        """E-step: Generate edit instructions.
        
        This step samples edit instructions based on current model performance.
        """
        logger.debug("E-step: Generating edit instructions")
        
        edit_batch = EditBatch()
        
        # Sample subset of training data
        sample_size = min(self.config.num_samples, len(train_data))
        samples = np.random.choice(train_data, size=sample_size, replace=False)
        
        for sample in samples:
            # Create context from sample
            context = self._create_edit_context(sample)
            
            # Generate edits
            with torch.no_grad():
                edits = self.edit_generator(context)
            
            # Add high-confidence edits to batch
            for edit in edits:
                if edit.confidence > 0.5:  # Confidence threshold
                    edit_batch.add_instruction(edit)
        
        # Store in history
        self.state.edit_history.append(edit_batch)
        
        logger.debug(f"Generated {len(edit_batch.instructions)} edit instructions")
        
        return edit_batch
    
    def _m_step(
        self,
        edit_batch: EditBatch,
        train_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """M-step: Update model parameters.
        
        This step applies the edit instructions and fine-tunes the model.
        """
        logger.debug("M-step: Updating model parameters")
        
        metrics = {"loss": 0.0, "accuracy": 0.0}
        
        # Apply edit instructions
        if edit_batch.instructions:
            update_results = self.weight_updater.apply_batch(
                self.model,
                edit_batch.instructions
            )
            metrics["edits_applied"] = update_results["successful_edits"]
            metrics["edit_success_rate"] = update_results["success_rate"]
        
        # Fine-tune with gradient descent
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for step in range(self.config.inner_steps):
            # Sample mini-batch
            batch_size = min(32, len(train_data))
            batch = np.random.choice(train_data, size=batch_size, replace=False)
            
            # Forward pass
            loss, correct = self._compute_loss(batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.config.accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
                
                # Update weights
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * len(batch)
            total_correct += correct
            total_samples += len(batch)
        
        # Update learning rate
        self.scheduler.step()
        
        metrics["loss"] = total_loss / total_samples
        metrics["accuracy"] = total_correct / total_samples
        metrics["learning_rate"] = self.optimizer.param_groups[0]['lr']
        
        return metrics
    
    def _create_edit_context(self, sample: Dict[str, Any]) -> EditContext:
        """Create edit context from training sample."""
        # Generate model output
        self.model.eval()
        with torch.no_grad():
            model_output = self._generate_output(sample["input"])
        
        return EditContext(
            input_text=sample["input"],
            model_output=model_output,
            expected_output=sample.get("output"),
            feedback=sample.get("feedback"),
            performance_metrics=sample.get("metrics")
        )
    
    def _compute_loss(
        self,
        batch: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, int]:
        """Compute loss for a batch."""
        # This is a placeholder - implement based on your task
        total_loss = torch.tensor(0.0, requires_grad=True)
        correct = 0
        
        for sample in batch:
            # Forward pass
            output = self.model(sample["input"])
            
            # Compute loss (example: cross-entropy)
            if "target" in sample:
                loss = F.cross_entropy(output, sample["target"])
                total_loss = total_loss + loss
                
                # Count correct predictions
                pred = output.argmax(dim=-1)
                correct += (pred == sample["target"]).sum().item()
        
        return total_loss / len(batch), correct
    
    def _generate_output(self, input_text: str) -> str:
        """Generate model output for input."""
        # This is a placeholder - implement based on your model
        return "Generated output"
    
    def _validate(
        self,
        validation_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Validate model on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for sample in validation_data:
                loss, correct = self._compute_loss([sample])
                total_loss += loss.item()
                total_correct += correct
                total_samples += 1
        
        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples
        }
    
    def _update_state(self, metrics: Dict[str, float]):
        """Update optimization state."""
        self.state.metrics_history.append(metrics)
        
        # Check if best model
        current_loss = metrics.get("val_loss", metrics.get("loss", float('inf')))
        
        if current_loss < self.state.best_loss:
            self.state.best_loss = current_loss
            self.state.best_model_state = copy.deepcopy(self.model.state_dict())
            self.state.patience_counter = 0
        else:
            self.state.patience_counter += 1
    
    def _should_stop(self) -> bool:
        """Check early stopping criterion."""
        return self.state.patience_counter >= self.config.patience
    
    def _save_checkpoint(self):
        """Save optimization checkpoint."""
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "optimization_state": self.state,
            "config": self.config
        }
        
        # Save checkpoint (implement actual saving logic)
        logger.info(f"Saved checkpoint at iteration {self.state.iteration}")
    
    def _get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results."""
        metrics_history = self.state.metrics_history
        
        if not metrics_history:
            return {"error": "No optimization metrics recorded"}
        
        # Calculate summary statistics
        final_metrics = metrics_history[-1]
        best_metrics = min(metrics_history, key=lambda m: m.get("loss", float('inf')))
        
        # Count total edits
        total_edits = sum(
            len(batch.instructions)
            for batch in self.state.edit_history
        )
        
        return {
            "iterations_completed": self.state.iteration + 1,
            "final_metrics": final_metrics,
            "best_metrics": best_metrics,
            "total_edits_generated": total_edits,
            "improvement": {
                "loss": metrics_history[0].get("loss", 0) - final_metrics.get("loss", 0),
                "accuracy": final_metrics.get("accuracy", 0) - metrics_history[0].get("accuracy", 0)
            },
            "converged": not self._should_stop()
        }


class TwoLoopSEAL:
    """Two-loop architecture for SEAL: outer RL loop + inner update loop."""
    
    def __init__(
        self,
        model: nn.Module,
        edit_generator: EditGenerator,
        config: Optional[ReSTEMConfig] = None
    ):
        self.model = model
        self.edit_generator = edit_generator
        self.config = config or ReSTEMConfig()
        
        # Inner loop: ReST-EM optimizer
        self.inner_optimizer = ReSTEMOptimizer(
            model,
            edit_generator,
            WeightUpdater(),
            config
        )
        
        # Outer loop: RL policy for edit generation
        self.edit_policy = self._build_edit_policy()
        self.policy_optimizer = torch.optim.Adam(
            self.edit_policy.parameters(),
            lr=config.learning_rate * 0.1  # Slower learning for policy
        )
        
        # Replay buffer for RL
        self.replay_buffer = deque(maxlen=1000)
        
    def _build_edit_policy(self) -> nn.Module:
        """Build RL policy network for edit generation."""
        # This is a placeholder - implement based on your needs
        return nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, len(EditType))  # Output edit type probabilities
        )
    
    def train(
        self,
        train_data: List[Dict[str, Any]],
        num_epochs: int = 10
    ) -> Dict[str, Any]:
        """Train with two-loop architecture."""
        results = []
        
        for epoch in range(num_epochs):
            # Outer loop: Update edit policy
            policy_metrics = self._outer_loop_step(train_data)
            
            # Inner loop: Apply edits and optimize
            inner_metrics = self.inner_optimizer.optimize(
                train_data,
                num_iterations=self.config.em_iterations
            )
            
            results.append({
                "epoch": epoch,
                "policy_metrics": policy_metrics,
                "inner_metrics": inner_metrics
            })
            
            logger.info(f"Epoch {epoch} completed")
        
        return {"training_results": results}
    
    def _outer_loop_step(
        self,
        train_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Outer loop: Update edit generation policy using RL."""
        # This is a simplified version - implement proper RL algorithm
        
        # Sample experiences from replay buffer
        if len(self.replay_buffer) < 32:
            return {"policy_loss": 0.0}
        
        batch = list(np.random.choice(self.replay_buffer, size=32, replace=False))
        
        # Compute policy gradient
        total_loss = 0.0
        for experience in batch:
            state, action, reward = experience
            
            # Forward through policy
            logits = self.edit_policy(state)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Policy gradient loss
            loss = -log_probs[action] * reward
            total_loss += loss
        
        # Update policy
        total_loss /= len(batch)
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        self.policy_optimizer.step()
        
        return {"policy_loss": total_loss.item()}
