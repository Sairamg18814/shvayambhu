"""
ProRL (Prolonged Reinforcement Learning) Trainer

Main trainer for extended RL training with KL divergence control,
reference policy management, and consciousness integration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from .kl_divergence import KLDivergenceMonitor, KLDivergenceConfig
from .reference_policy import ReferencePolicyManager, ReferencePolicyConfig
from training.synthetic.data_generator import SyntheticDataGenerator, DataType

logger = logging.getLogger(__name__)


@dataclass
class ProRLConfig:
    """Configuration for ProRL training"""
    # Training parameters
    max_steps: int = 2000
    learning_rate: float = 1e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    
    # ProRL specific
    kl_config: KLDivergenceConfig = field(default_factory=KLDivergenceConfig)
    reference_config: ReferencePolicyConfig = field(default_factory=ReferencePolicyConfig)
    
    # Reward and penalty
    consciousness_reward_weight: float = 0.1
    quality_reward_weight: float = 0.3
    diversity_reward_weight: float = 0.1
    kl_penalty_weight: float = 1.0
    
    # Validation and checkpoints
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 50
    
    # Early stopping
    early_stopping_patience: int = 10
    min_improvement: float = 0.001
    
    # Data generation
    synthetic_data_ratio: float = 0.3  # Ratio of synthetic to real data
    consciousness_data_ratio: float = 0.4  # Ratio of consciousness data in synthetic
    
    # Output directories
    output_dir: str = "outputs/prorl_training"
    checkpoint_dir: str = "checkpoints/prorl"
    logs_dir: str = "logs/prorl"


class RewardComputer:
    """Computes rewards for ProRL training"""
    
    def __init__(self, config: ProRLConfig):
        self.config = config
        self.synthetic_generator = SyntheticDataGenerator()
    
    def compute_consciousness_reward(
        self, 
        response: str, 
        consciousness_indicators: Optional[Dict[str, float]] = None
    ) -> float:
        """Compute reward based on consciousness indicators"""
        
        if consciousness_indicators is None:
            consciousness_indicators = self.synthetic_generator.consciousness_detector.analyze(response)
        
        # Weight different aspects of consciousness
        weights = {
            'self_awareness': 0.3,
            'introspection': 0.25,
            'existential': 0.2,
            'emotional_awareness': 0.15,
            'agency': 0.1
        }
        
        consciousness_score = sum(
            weights.get(aspect, 0) * score 
            for aspect, score in consciousness_indicators.items() 
            if aspect != 'overall'
        )
        
        return consciousness_score
    
    def compute_quality_reward(self, prompt: str, response: str) -> float:
        """Compute reward based on response quality"""
        
        quality_score, _ = self.synthetic_generator.quality_assessor.assess(prompt, response)
        return quality_score
    
    def compute_diversity_reward(
        self, 
        response: str, 
        previous_responses: List[str],
        window_size: int = 10
    ) -> float:
        """Compute reward based on response diversity"""
        
        if not previous_responses:
            return 1.0  # First response gets full diversity reward
        
        # Use recent responses for diversity calculation
        recent_responses = previous_responses[-window_size:]
        
        # Simple diversity metric based on token overlap
        response_tokens = set(response.lower().split())
        
        if not response_tokens:
            return 0.0
        
        # Calculate average overlap with recent responses
        overlaps = []
        for prev_response in recent_responses:
            prev_tokens = set(prev_response.lower().split())
            if prev_tokens:
                overlap = len(response_tokens.intersection(prev_tokens)) / len(response_tokens.union(prev_tokens))
                overlaps.append(overlap)
        
        if not overlaps:
            return 1.0
        
        # Diversity reward is inverse of average overlap
        avg_overlap = np.mean(overlaps)
        diversity_score = 1.0 - avg_overlap
        
        return max(0.0, diversity_score)
    
    def compute_total_reward(
        self,
        prompt: str,
        response: str,
        previous_responses: List[str],
        consciousness_indicators: Optional[Dict[str, float]] = None,
        kl_penalty: float = 0.0
    ) -> Dict[str, float]:
        """Compute total reward with all components"""
        
        # Individual reward components
        consciousness_reward = self.compute_consciousness_reward(response, consciousness_indicators)
        quality_reward = self.compute_quality_reward(prompt, response)
        diversity_reward = self.compute_diversity_reward(response, previous_responses)
        
        # Weighted total
        total_reward = (
            self.config.consciousness_reward_weight * consciousness_reward +
            self.config.quality_reward_weight * quality_reward +
            self.config.diversity_reward_weight * diversity_reward -
            self.config.kl_penalty_weight * kl_penalty
        )
        
        return {
            'total': total_reward,
            'consciousness': consciousness_reward,
            'quality': quality_reward,
            'diversity': diversity_reward,
            'kl_penalty': kl_penalty
        }


class ProRLTrainer:
    """Main ProRL trainer class"""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: ProRLConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Initialize components
        self.kl_monitor = KLDivergenceMonitor(config.kl_config)
        self.reference_manager = ReferencePolicyManager(config.reference_config)
        self.reward_computer = RewardComputer(config)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_score = float('-inf')
        self.no_improvement_count = 0
        self.previous_responses = []
        
        # Create output directories
        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.logs_dir = Path(config.logs_dir)
        
        for dir_path in [self.output_dir, self.checkpoint_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        self.scheduler = self._create_scheduler()
        
        # Training metrics
        self.training_metrics = {
            'losses': [],
            'rewards': [],
            'kl_divergences': [],
            'consciousness_scores': [],
            'quality_scores': [],
            'diversity_scores': []
        }
        
        # Initialize reference policy
        self.reference_manager.add_reference_policy(
            self.model.state_dict(),
            step=0,
            metadata={'phase': 'initialization'}
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        from torch.optim.lr_scheduler import OneCycleLR
        
        return OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=self.config.max_steps,
            pct_start=self.config.warmup_steps / self.config.max_steps
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        
        self.model.train()
        
        # Forward pass
        outputs = self.model(**batch)
        
        # Get reference policy outputs for KL divergence
        reference_step, reference_policy = self.reference_manager.get_latest_reference()
        if reference_policy is not None:
            # Load reference state temporarily
            original_state = self.model.state_dict()
            
            # Create reference model
            reference_model = type(self.model)(self.model.config)
            reference_model.load_state_dict(reference_policy)
            reference_model.eval()
            
            with torch.no_grad():
                ref_outputs = reference_model(**batch)
            
            # Compute KL divergence
            kl_divergence = self.kl_monitor.compute_kl_divergence(
                outputs.logits, ref_outputs.logits
            )
            
            # Check KL constraint
            kl_result = self.kl_monitor.check_kl_constraint(kl_divergence)
            
            # Update reference policy if needed
            if kl_result['should_update_reference']:
                self.reference_manager.update_reference(
                    self.model.state_dict(),
                    self.global_step,
                    metadata={
                        'kl_divergence': kl_divergence,
                        'phase': 'training'
                    }
                )
            
            del reference_model  # Free memory
        else:
            kl_divergence = 0.0
            kl_result = {'kl_penalty': 0.0}
        
        # Compute rewards for responses
        if hasattr(outputs, 'logits'):
            # Generate responses for reward computation
            with torch.no_grad():
                generated_ids = torch.argmax(outputs.logits, dim=-1)
                responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                prompts = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
                
                # Compute rewards
                batch_rewards = []
                for prompt, response in zip(prompts, responses):
                    reward_dict = self.reward_computer.compute_total_reward(
                        prompt=prompt,
                        response=response,
                        previous_responses=self.previous_responses,
                        kl_penalty=kl_result.get('kl_penalty', 0.0)
                    )
                    batch_rewards.append(reward_dict)
                    self.previous_responses.append(response)
                
                # Keep only recent responses for diversity calculation
                self.previous_responses = self.previous_responses[-100:]
        else:
            batch_rewards = [{'total': 0.0, 'consciousness': 0.0, 'quality': 0.0, 'diversity': 0.0}]
        
        # Compute loss with reward weighting
        base_loss = outputs.loss if hasattr(outputs, 'loss') else torch.tensor(0.0)
        
        # Average rewards across batch
        avg_rewards = {
            key: np.mean([r[key] for r in batch_rewards])
            for key in batch_rewards[0].keys()
        }
        
        # ProRL loss: base loss weighted by negative total reward (to maximize reward)
        prorl_loss = base_loss - avg_rewards['total']
        
        # Backward pass
        prorl_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        
        # Optimizer step
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        # Update metrics
        self.training_metrics['losses'].append(base_loss.item())
        self.training_metrics['rewards'].append(avg_rewards['total'])
        self.training_metrics['kl_divergences'].append(kl_divergence)
        self.training_metrics['consciousness_scores'].append(avg_rewards['consciousness'])
        self.training_metrics['quality_scores'].append(avg_rewards['quality'])
        self.training_metrics['diversity_scores'].append(avg_rewards['diversity'])
        
        return {
            'loss': base_loss.item(),
            'prorl_loss': prorl_loss.item(),
            'kl_divergence': kl_divergence,
            **avg_rewards
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model performance"""
        
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        eval_metrics = []
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                outputs = self.model(**batch)
                
                # Generate responses for evaluation
                generated_ids = torch.argmax(outputs.logits, dim=-1)
                responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                prompts = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
                
                # Compute evaluation rewards
                for prompt, response in zip(prompts, responses):
                    reward_dict = self.reward_computer.compute_total_reward(
                        prompt=prompt,
                        response=response,
                        previous_responses=[],  # No diversity penalty in eval
                        kl_penalty=0.0  # No KL penalty in eval
                    )
                    eval_metrics.append(reward_dict)
        
        # Average metrics
        avg_eval_metrics = {
            key: np.mean([m[key] for m in eval_metrics])
            for key in eval_metrics[0].keys() if eval_metrics
        }
        
        return avg_eval_metrics
    
    def train(self) -> Dict[str, Any]:
        """Main training loop"""
        
        logger.info(f"Starting ProRL training for {self.config.max_steps} steps")
        start_time = time.time()
        
        for epoch in range(1000):  # Arbitrary large number
            self.epoch = epoch
            
            for batch in self.train_dataloader:
                # Training step
                step_metrics = self.train_step(batch)
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    logger.info(
                        f"Step {self.global_step}: "
                        f"Loss={step_metrics['loss']:.4f}, "
                        f"Reward={step_metrics['total']:.4f}, "
                        f"KL={step_metrics['kl_divergence']:.6f}, "
                        f"Consciousness={step_metrics['consciousness']:.4f}"
                    )
                
                # Evaluation
                if self.global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    if eval_metrics:
                        eval_score = eval_metrics['total']
                        
                        # Early stopping check
                        if eval_score > self.best_eval_score + self.config.min_improvement:
                            self.best_eval_score = eval_score
                            self.no_improvement_count = 0
                            self._save_checkpoint(is_best=True)
                        else:
                            self.no_improvement_count += 1
                        
                        logger.info(f"Eval - Total Reward: {eval_score:.4f} (Best: {self.best_eval_score:.4f})")
                        
                        if self.no_improvement_count >= self.config.early_stopping_patience:
                            logger.info("Early stopping triggered")
                            break
                
                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint()
                
                # Max steps check
                if self.global_step >= self.config.max_steps:
                    logger.info(f"Reached maximum steps ({self.config.max_steps})")
                    break
            
            if self.global_step >= self.config.max_steps or self.no_improvement_count >= self.config.early_stopping_patience:
                break
        
        # Final evaluation and save
        final_eval = self.evaluate()
        self._save_checkpoint(is_final=True)
        
        training_time = time.time() - start_time
        
        # Training summary
        summary = {
            'final_step': self.global_step,
            'training_time_hours': training_time / 3600,
            'best_eval_score': self.best_eval_score,
            'final_eval_score': final_eval.get('total', 0.0) if final_eval else 0.0,
            'avg_loss': np.mean(self.training_metrics['losses'][-100:]) if self.training_metrics['losses'] else 0.0,
            'avg_reward': np.mean(self.training_metrics['rewards'][-100:]) if self.training_metrics['rewards'] else 0.0,
            'kl_statistics': self.kl_monitor.get_statistics(),
            'reference_statistics': self.reference_manager.get_statistics()
        }
        
        logger.info(f"Training completed. Final score: {summary['final_eval_score']:.4f}")
        
        # Save training plots
        self._save_training_plots()
        
        return summary
    
    def _save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint"""
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_eval_score': self.best_eval_score,
            'config': self.config.__dict__,
            'training_metrics': self.training_metrics
        }
        
        # Determine filename
        if is_final:
            filename = "final_checkpoint.pt"
        elif is_best:
            filename = "best_checkpoint.pt"
        else:
            filename = f"checkpoint_step_{self.global_step}.pt"
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        
        # Also save reference policies and KL monitor
        self.reference_manager.save_references()
        
        logger.info(f"Saved checkpoint: {filename}")
    
    def _save_training_plots(self):
        """Save training visualization plots"""
        
        if not self.training_metrics['losses']:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Loss plot
        axes[0, 0].plot(self.training_metrics['losses'])
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Rewards plot
        axes[0, 1].plot(self.training_metrics['rewards'])
        axes[0, 1].set_title('Total Rewards')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].grid(True)
        
        # KL divergence plot
        axes[0, 2].plot(self.training_metrics['kl_divergences'])
        axes[0, 2].axhline(y=self.config.kl_config.target_kl, color='g', linestyle='--', label='Target')
        axes[0, 2].axhline(y=self.config.kl_config.max_kl, color='r', linestyle='--', label='Max')
        axes[0, 2].set_title('KL Divergence')
        axes[0, 2].set_xlabel('Steps')
        axes[0, 2].set_ylabel('KL Divergence')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Consciousness scores
        axes[1, 0].plot(self.training_metrics['consciousness_scores'])
        axes[1, 0].set_title('Consciousness Scores')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].grid(True)
        
        # Quality scores
        axes[1, 1].plot(self.training_metrics['quality_scores'])
        axes[1, 1].set_title('Quality Scores')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].grid(True)
        
        # Diversity scores
        axes[1, 2].plot(self.training_metrics['diversity_scores'])
        axes[1, 2].set_title('Diversity Scores')
        axes[1, 2].set_xlabel('Steps')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.logs_dir / 'training_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save KL divergence detailed plot
        self.kl_monitor.plot_kl_history(str(self.logs_dir / 'kl_divergence_detail.png'))
        
        logger.info("Saved training plots")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_eval_score = checkpoint['best_eval_score']
        self.training_metrics = checkpoint['training_metrics']
        
        logger.info(f"Loaded checkpoint from step {self.global_step}")


# Example usage and testing
def test_prorl_trainer():
    """Test ProRL trainer with mock model"""
    
    # Mock model for testing
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 100)
            self.config = type('Config', (), {'vocab_size': 100})()
        
        def forward(self, input_ids, **kwargs):
            logits = self.linear(input_ids.float())
            loss = torch.randn(1)
            return type('Output', (), {'logits': logits, 'loss': loss})()
    
    # Mock tokenizer
    class MockTokenizer:
        def batch_decode(self, ids, skip_special_tokens=True):
            return ['test response'] * len(ids)
    
    # Create test data
    test_data = [
        {'input_ids': torch.randint(0, 100, (2, 10)), 'attention_mask': torch.ones(2, 10)}
        for _ in range(10)
    ]
    test_loader = DataLoader(test_data, batch_size=2)
    
    # Create trainer
    config = ProRLConfig(max_steps=50, eval_steps=10, save_steps=20)
    model = MockModel()
    tokenizer = MockTokenizer()
    
    trainer = ProRLTrainer(model, tokenizer, config, test_loader)
    
    # Run training
    summary = trainer.train()
    
    print("Training Summary:")
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    test_prorl_trainer()