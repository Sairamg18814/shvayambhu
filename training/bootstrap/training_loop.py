"""Main training loop for bootstrap training.

This module implements the core training loop with gradient accumulation,
mixed precision training, and distributed training support.
"""

import os
import time
import json
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import wandb

from .data_loader import BootstrapDataLoader
from .objectives import MultiObjectiveTrainer
from .curriculum import CurriculumScheduler
from .statistics_tracker import StatisticsTracker
from ...core.blt.pipeline import BLTPipeline
from ...core.seal.self_edit import SelfEditGenerator
from ...utils.hardware.memory_manager import MemoryManager


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Model config
    model_config: Dict[str, Any]
    
    # Training hyperparameters
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 100000
    warmup_steps: int = 10000
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    gradient_clip: float = 1.0
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"  # float16 or bfloat16
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints/"
    save_steps: int = 1000
    save_total_limit: int = 5
    resume_from_checkpoint: Optional[str] = None
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 500
    eval_samples: int = 1000
    use_wandb: bool = False
    wandb_project: str = "shvayambhu"
    
    # Distributed training
    distributed: bool = False
    local_rank: int = -1
    world_size: int = 1
    
    # Hardware optimization
    device: str = "auto"
    memory_efficient: bool = True
    gradient_checkpointing: bool = True
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_strategy: str = "progressive"


class TrainingLoop:
    """Main training loop implementation."""
    
    def __init__(
        self,
        config: TrainingConfig,
        model: Optional[BLTPipeline] = None,
        data_loader: Optional[BootstrapDataLoader] = None,
        objectives_trainer: Optional[MultiObjectiveTrainer] = None,
        curriculum_scheduler: Optional[CurriculumScheduler] = None,
        statistics_tracker: Optional[StatisticsTracker] = None
    ):
        self.config = config
        
        # Setup device
        self._setup_device()
        
        # Initialize components
        self.model = model or self._create_model()
        self.data_loader = data_loader or self._create_data_loader()
        self.objectives_trainer = objectives_trainer or MultiObjectiveTrainer(self.model)
        self.curriculum_scheduler = curriculum_scheduler
        self.statistics_tracker = statistics_tracker or StatisticsTracker(config.checkpoint_dir)
        
        # Setup distributed training if needed
        if self.config.distributed:
            self._setup_distributed()
        
        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision setup
        self.scaler = GradScaler() if self.config.use_amp else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        self.early_stopping_counter = 0
        
        # Memory management
        self.memory_manager = MemoryManager()
        
        # Setup logging
        if self.config.use_wandb:
            self._setup_wandb()
    
    def _setup_device(self):
        """Setup training device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.config.device)
        
        print(f"Using device: {self.device}")
    
    def _create_model(self) -> BLTPipeline:
        """Create model from config."""
        model = BLTPipeline(self.config.model_config)
        
        # Enable gradient checkpointing if requested
        if self.config.gradient_checkpointing:
            model.enable_gradient_checkpointing()
        
        return model.to(self.device)
    
    def _create_data_loader(self) -> DataLoader:
        """Create data loader."""
        dataset = BootstrapDataLoader(
            data_path=self.config.model_config.get("data_path", "data/"),
            batch_size=self.config.batch_size,
            max_sequence_length=self.config.model_config.get("max_sequence_length", 2048)
        )
        
        # Create sampler for distributed training
        sampler = None
        if self.config.distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.config.world_size,
                rank=self.config.local_rank
            )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        # Get parameters with weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if any(nd in name for nd in ["bias", "LayerNorm", "layernorm"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        
        # Create optimizer
        if self.config.optimizer == "adamw":
            optimizer = optim.AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.config.optimizer == "adam":
            optimizer = optim.Adam(
                param_groups,
                lr=self.config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.config.optimizer == "sgd":
            optimizer = optim.SGD(
                param_groups,
                lr=self.config.learning_rate,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        return optimizer
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        if self.config.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_steps - self.config.warmup_steps,
                eta_min=self.config.learning_rate * 0.1
            )
        elif self.config.scheduler == "linear":
            scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.config.max_steps - self.config.warmup_steps
            )
        elif self.config.scheduler == "constant":
            scheduler = optim.lr_scheduler.ConstantLR(
                self.optimizer,
                factor=1.0
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")
        
        # Add warmup
        if self.config.warmup_steps > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=self.config.warmup_steps
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[self.config.warmup_steps]
            )
        
        return scheduler
    
    def _setup_distributed(self):
        """Setup distributed training."""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ['WORLD_SIZE'])
            gpu = int(os.environ['LOCAL_RANK'])
        else:
            print('Not using distributed mode')
            self.config.distributed = False
            return
        
        torch.cuda.set_device(gpu)
        self.device = torch.device('cuda', gpu)
        
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        self.config.local_rank = gpu
        self.config.world_size = world_size
        
        # Wrap model with DDP
        self.model = DDP(
            self.model,
            device_ids=[gpu],
            output_device=gpu
        )
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        wandb.init(
            project=self.config.wandb_project,
            config=self.config.__dict__,
            resume="allow"
        )
        wandb.watch(self.model)
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config.max_steps} steps")
        
        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)
        
        # Training loop
        train_iterator = tqdm(
            range(self.global_step, self.config.max_steps),
            desc="Training",
            disable=self.config.local_rank > 0
        )
        
        accumulation_steps = 0
        accumulated_loss = 0.0
        
        while self.global_step < self.config.max_steps:
            self.epoch += 1
            
            # Set epoch for distributed sampler
            if self.config.distributed and hasattr(self.data_loader.sampler, 'set_epoch'):
                self.data_loader.sampler.set_epoch(self.epoch)
            
            for batch_idx, batch in enumerate(self.data_loader):
                # Move batch to device
                batch = self._prepare_batch(batch)
                
                # Get curriculum difficulty if enabled
                if self.config.use_curriculum and self.curriculum_scheduler:
                    difficulty = self.curriculum_scheduler.get_difficulty(self.global_step)
                    batch = self._apply_curriculum_filtering(batch, difficulty)
                
                # Forward pass with mixed precision
                with autocast(enabled=self.config.use_amp, dtype=getattr(torch, self.config.amp_dtype)):
                    outputs = self.objectives_trainer.compute_losses(
                        batch['input_bytes'],
                        global_step=self.global_step
                    )
                    loss = outputs['total_loss'] / self.config.gradient_accumulation_steps
                
                # Backward pass
                if self.config.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                accumulated_loss += loss.item()
                accumulation_steps += 1
                
                # Gradient accumulation
                if accumulation_steps >= self.config.gradient_accumulation_steps:
                    # Gradient clipping
                    if self.config.gradient_clip > 0:
                        if self.config.use_amp:
                            self.scaler.unscale_(self.optimizer)
                        
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip
                        )
                    else:
                        grad_norm = 0.0
                    
                    # Optimizer step
                    if self.config.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Update statistics
                    self.global_step += 1
                    train_iterator.update(1)
                    
                    # Log metrics
                    if self.global_step % self.config.logging_steps == 0:
                        self._log_metrics(
                            accumulated_loss,
                            grad_norm,
                            outputs,
                            batch['input_bytes'].shape[0] * self.config.gradient_accumulation_steps
                        )
                    
                    # Reset accumulation
                    accumulated_loss = 0.0
                    accumulation_steps = 0
                
                # Evaluation
                if self.global_step % self.config.eval_steps == 0:
                    eval_loss = self.evaluate()
                    
                    # Early stopping check
                    if self._check_early_stopping(eval_loss):
                        print("Early stopping triggered")
                        return
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                
                # Check if training is complete
                if self.global_step >= self.config.max_steps:
                    break
                
                # Memory management
                if self.config.memory_efficient and self.global_step % 100 == 0:
                    self.memory_manager.optimize_memory()
        
        # Final evaluation and save
        self.evaluate()
        self.save_checkpoint(final=True)
        
        print("Training completed!")
    
    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare batch for training."""
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
    
    def _apply_curriculum_filtering(
        self,
        batch: Dict[str, torch.Tensor],
        difficulty: float
    ) -> Dict[str, torch.Tensor]:
        """Apply curriculum filtering to batch."""
        # Filter samples based on difficulty
        # This is a simplified version - actual implementation would be more sophisticated
        batch_size = batch['input_bytes'].shape[0]
        keep_ratio = difficulty
        keep_count = max(1, int(batch_size * keep_ratio))
        
        # Random selection for now - could be based on sample difficulty
        indices = torch.randperm(batch_size)[:keep_count]
        
        return {
            key: value[indices] if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
    
    def _log_metrics(
        self,
        loss: float,
        grad_norm: float,
        outputs: Dict[str, Any],
        batch_size: int
    ):
        """Log training metrics."""
        metrics = {
            "train/loss": loss,
            "train/learning_rate": self.optimizer.param_groups[0]['lr'],
            "train/gradient_norm": grad_norm,
            "train/global_step": self.global_step,
            "train/epoch": self.epoch
        }
        
        # Add loss components
        for key, value in outputs.items():
            if key != 'total_loss' and isinstance(value, torch.Tensor):
                metrics[f"train/{key}"] = value.item()
        
        # Add throughput metrics
        if hasattr(self, '_last_log_time'):
            time_delta = time.time() - self._last_log_time
            samples_per_sec = batch_size / time_delta
            metrics["train/samples_per_second"] = samples_per_sec
        
        self._last_log_time = time.time()
        
        # Log to wandb
        if self.config.use_wandb:
            wandb.log(metrics, step=self.global_step)
        
        # Log to statistics tracker
        if self.statistics_tracker:
            self.statistics_tracker.update_training_statistics(
                epoch=self.epoch,
                global_step=self.global_step,
                loss=loss,
                learning_rate=self.optimizer.param_groups[0]['lr'],
                gradient_norm=grad_norm,
                throughput=metrics.get("train/samples_per_second", 0),
                memory_usage=self.memory_manager.get_current_usage(),
                loss_components={k.replace("train/", ""): v for k, v in metrics.items() if k.startswith("train/") and k != "train/loss"}
            )
        
        # Print to console
        if self.config.local_rank <= 0:
            print(f"Step {self.global_step}: loss={loss:.4f}, lr={metrics['train/learning_rate']:.6f}")
    
    def evaluate(self) -> float:
        """Run evaluation."""
        self.model.eval()
        eval_losses = []
        
        with torch.no_grad():
            eval_steps = min(
                self.config.eval_samples // self.config.batch_size,
                len(self.data_loader)
            )
            
            for step, batch in enumerate(self.data_loader):
                if step >= eval_steps:
                    break
                
                batch = self._prepare_batch(batch)
                
                with autocast(enabled=self.config.use_amp, dtype=getattr(torch, self.config.amp_dtype)):
                    outputs = self.objectives_trainer.compute_losses(
                        batch['input_bytes'],
                        global_step=self.global_step
                    )
                    loss = outputs['total_loss']
                
                eval_losses.append(loss.item())
        
        self.model.train()
        
        # Calculate average loss
        avg_eval_loss = np.mean(eval_losses)
        
        # Log evaluation metrics
        if self.config.use_wandb:
            wandb.log({
                "eval/loss": avg_eval_loss,
                "eval/perplexity": np.exp(avg_eval_loss)
            }, step=self.global_step)
        
        # Update statistics
        if self.statistics_tracker:
            self.statistics_tracker.add_validation_metric(
                "loss", avg_eval_loss, self.global_step
            )
        
        print(f"Evaluation at step {self.global_step}: loss={avg_eval_loss:.4f}")
        
        return avg_eval_loss
    
    def _check_early_stopping(self, eval_loss: float) -> bool:
        """Check if early stopping should be triggered."""
        if eval_loss < self.best_eval_loss - self.config.early_stopping_threshold:
            self.best_eval_loss = eval_loss
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        
        return self.early_stopping_counter >= self.config.early_stopping_patience
    
    def save_checkpoint(self, final: bool = False):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if final:
            checkpoint_path = checkpoint_dir / "final_checkpoint.pt"
        else:
            checkpoint_path = checkpoint_dir / f"checkpoint_{self.global_step}.pt"
        
        # Get model state dict (handle DDP)
        if isinstance(self.model, DDP):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        
        checkpoint = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
            "config": self.config.__dict__
        }
        
        if self.config.use_amp and self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Manage checkpoint limit
        if not final and self.config.save_total_limit > 0:
            self._cleanup_checkpoints()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer and scheduler
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load training state
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_eval_loss = checkpoint.get("best_eval_loss", float('inf'))
        
        # Load scaler if using AMP
        if self.config.use_amp and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from step {self.global_step}")
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to maintain limit."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: int(p.stem.split('_')[1])
        )
        
        if len(checkpoints) > self.config.save_total_limit:
            for checkpoint in checkpoints[:-self.config.save_total_limit]:
                checkpoint.unlink()
                print(f"Removed old checkpoint: {checkpoint}")


def train_model(config: TrainingConfig):
    """High-level training function."""
    # Create training loop
    trainer = TrainingLoop(config)
    
    # Run training
    trainer.train()
    
    # Generate final report
    if trainer.statistics_tracker:
        trainer.statistics_tracker.generate_report()
        trainer.statistics_tracker.close()
    
    return trainer.model