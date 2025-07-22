"""Self-Supervised Training Objectives for Bootstrap Learning.

This module implements various self-supervised learning objectives
for training the BLT model without external supervision.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import random
import numpy as np
from dataclasses import dataclass

from ...core.blt.entropy import calculate_byte_entropy


@dataclass
class ObjectiveConfig:
    """Configuration for training objectives."""
    name: str
    weight: float
    enabled: bool = True
    schedule: str = "constant"  # constant, linear, cosine
    warmup_steps: int = 0
    min_weight: float = 0.0
    max_weight: float = 1.0


class MaskedLanguageModeling(nn.Module):
    """Masked Language Modeling objective for byte sequences."""
    
    def __init__(
        self,
        vocab_size: int = 256,
        mask_probability: float = 0.15,
        random_probability: float = 0.1,
        keep_probability: float = 0.1,
        mask_token_id: int = 256  # Special mask token
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.mask_probability = mask_probability
        self.random_probability = random_probability
        self.keep_probability = keep_probability
        self.mask_token_id = mask_token_id
        
        # Prediction head
        self.prediction_head = nn.Linear(768, vocab_size)  # Will be configured based on model
        
    def create_masked_input(
        self,
        input_ids: Tensor,
        mask_probability: Optional[float] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Create masked input for MLM training.
        
        Args:
            input_ids: Original input token IDs (batch_size, seq_len)
            mask_probability: Probability of masking (overrides default)
            
        Returns:
            Tuple of (masked_input, targets, loss_mask)
        """
        batch_size, seq_len = input_ids.shape
        mask_prob = mask_probability or self.mask_probability
        
        # Create random mask
        rand = torch.rand(batch_size, seq_len, device=input_ids.device)
        mask = rand < mask_prob
        
        # Create targets (only for masked positions)
        targets = input_ids.clone()
        targets[~mask] = -100  # Ignore non-masked positions in loss
        
        # Create masked input
        masked_input = input_ids.clone()
        
        # Apply masking strategy
        for i in range(batch_size):
            for j in range(seq_len):
                if mask[i, j]:
                    rand_val = random.random()
                    
                    if rand_val < 0.8:  # 80% of time: replace with mask token
                        masked_input[i, j] = self.mask_token_id
                    elif rand_val < 0.9:  # 10% of time: replace with random token
                        masked_input[i, j] = random.randint(0, self.vocab_size - 1)
                    # 10% of time: keep original token
        
        return masked_input, targets, mask
    
    def forward(
        self,
        hidden_states: Tensor,
        targets: Tensor,
        loss_mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Forward pass for MLM objective.
        
        Args:
            hidden_states: Model hidden states (batch_size, seq_len, hidden_dim)
            targets: Target token IDs (batch_size, seq_len)
            loss_mask: Mask for computing loss
            
        Returns:
            Dictionary with loss and predictions
        """
        # Prediction logits
        prediction_logits = self.prediction_head(hidden_states)
        
        # Compute loss
        if targets is not None:
            loss = F.cross_entropy(
                prediction_logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100
            )
        else:
            loss = torch.tensor(0.0, device=hidden_states.device)
        
        return {
            "loss": loss,
            "logits": prediction_logits,
            "predictions": torch.argmax(prediction_logits, dim=-1)
        }


class NextByteePrediction(nn.Module):
    """Next byte prediction objective for autoregressive training."""
    
    def __init__(self, vocab_size: int = 256):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.prediction_head = nn.Linear(768, vocab_size)
        
    def forward(
        self,
        hidden_states: Tensor,
        input_ids: Tensor
    ) -> Dict[str, Tensor]:
        """Forward pass for next byte prediction.
        
        Args:
            hidden_states: Model hidden states (batch_size, seq_len, hidden_dim)
            input_ids: Input token IDs (batch_size, seq_len)
            
        Returns:
            Dictionary with loss and predictions
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Shift hidden states and targets
        logits = self.prediction_head(hidden_states[:, :-1])  # (batch_size, seq_len-1, vocab_size)
        targets = input_ids[:, 1:]  # (batch_size, seq_len-1)
        
        # Compute loss
        loss = F.cross_entropy(
            logits.contiguous().view(-1, self.vocab_size),
            targets.contiguous().view(-1)
        )
        
        return {
            "loss": loss,
            "logits": logits,
            "predictions": torch.argmax(logits, dim=-1)
        }


class EntropyPrediction(nn.Module):
    """Entropy prediction objective for understanding content complexity."""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        self.entropy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def compute_target_entropy(self, byte_sequence: bytes) -> float:
        """Compute target entropy for a byte sequence."""
        return calculate_byte_entropy(byte_sequence)
    
    def forward(
        self,
        hidden_states: Tensor,
        byte_sequences: List[bytes]
    ) -> Dict[str, Tensor]:
        """Forward pass for entropy prediction.
        
        Args:
            hidden_states: Model hidden states (batch_size, seq_len, hidden_dim)
            byte_sequences: Original byte sequences for each item in batch
            
        Returns:
            Dictionary with loss and predictions
        """
        batch_size = hidden_states.shape[0]
        
        # Pool hidden states (mean pooling)
        pooled_states = hidden_states.mean(dim=1)  # (batch_size, hidden_dim)
        
        # Predict entropy
        predicted_entropy = self.entropy_head(pooled_states).squeeze(-1)  # (batch_size,)
        
        # Compute target entropies
        target_entropies = []
        for byte_seq in byte_sequences:
            entropy = self.compute_target_entropy(byte_seq)
            target_entropies.append(entropy)
        
        target_tensor = torch.tensor(target_entropies, device=hidden_states.device)
        
        # Compute loss
        loss = F.mse_loss(predicted_entropy, target_tensor)
        
        return {
            "loss": loss,
            "predicted_entropy": predicted_entropy,
            "target_entropy": target_tensor
        }


class PatchBoundaryPrediction(nn.Module):
    """Patch boundary prediction for learning optimal segmentation."""
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def create_boundary_targets(
        self,
        patch_boundaries: List[Tuple[int, int]],
        sequence_length: int
    ) -> Tensor:
        """Create boundary targets from patch boundaries.
        
        Args:
            patch_boundaries: List of (start, end) tuples
            sequence_length: Total sequence length
            
        Returns:
            Binary tensor indicating boundary positions
        """
        targets = torch.zeros(sequence_length)
        
        for start, end in patch_boundaries:
            if start > 0:
                targets[start] = 1.0  # Mark start of patch
            if end < sequence_length:
                targets[end] = 1.0  # Mark end of patch
        
        return targets
    
    def forward(
        self,
        hidden_states: Tensor,
        patch_boundaries: List[List[Tuple[int, int]]]
    ) -> Dict[str, Tensor]:
        """Forward pass for boundary prediction.
        
        Args:
            hidden_states: Model hidden states (batch_size, seq_len, hidden_dim)
            patch_boundaries: Patch boundaries for each item in batch
            
        Returns:
            Dictionary with loss and predictions
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Predict boundaries
        boundary_logits = self.boundary_head(hidden_states).squeeze(-1)  # (batch_size, seq_len)
        
        # Create targets
        targets = []
        for boundaries in patch_boundaries:
            boundary_targets = self.create_boundary_targets(boundaries, seq_len)
            targets.append(boundary_targets)
        
        target_tensor = torch.stack(targets).to(hidden_states.device)
        
        # Compute loss
        loss = F.binary_cross_entropy_with_logits(boundary_logits, target_tensor)
        
        return {
            "loss": loss,
            "boundary_logits": boundary_logits,
            "predicted_boundaries": torch.sigmoid(boundary_logits) > 0.5
        }


class ContrastiveLearning(nn.Module):
    """Contrastive learning objective for representation learning."""
    
    def __init__(
        self,
        hidden_dim: int = 768,
        projection_dim: int = 256,
        temperature: float = 0.07
    ):
        super().__init__()
        
        self.temperature = temperature
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        
    def create_augmentations(self, byte_sequence: bytes) -> List[bytes]:
        """Create augmented versions of byte sequence.
        
        Args:
            byte_sequence: Original byte sequence
            
        Returns:
            List of augmented sequences
        """
        augmentations = []
        
        # Original sequence
        augmentations.append(byte_sequence)
        
        # Random crop (if sequence is long enough)
        if len(byte_sequence) > 100:
            start = random.randint(0, len(byte_sequence) - 100)
            end = start + random.randint(50, min(200, len(byte_sequence) - start))
            augmentations.append(byte_sequence[start:end])
        
        # Random byte substitution (small probability)
        if len(byte_sequence) > 10:
            seq_list = list(byte_sequence)
            num_substitutions = max(1, len(seq_list) // 50)
            for _ in range(num_substitutions):
                idx = random.randint(0, len(seq_list) - 1)
                seq_list[idx] = random.randint(0, 255)
            augmentations.append(bytes(seq_list))
        
        return augmentations
    
    def forward(
        self,
        hidden_states: Tensor,
        byte_sequences: List[bytes]
    ) -> Dict[str, Tensor]:
        """Forward pass for contrastive learning.
        
        Args:
            hidden_states: Model hidden states (batch_size, seq_len, hidden_dim)
            byte_sequences: Original byte sequences
            
        Returns:
            Dictionary with loss and representations
        """
        batch_size = hidden_states.shape[0]
        
        # Pool representations
        pooled_states = hidden_states.mean(dim=1)  # (batch_size, hidden_dim)
        
        # Project to contrastive space
        projections = self.projection(pooled_states)  # (batch_size, projection_dim)
        projections = F.normalize(projections, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(projections, projections.T) / self.temperature
        
        # Create positive pairs (adjacent items in batch are considered similar)
        labels = torch.arange(batch_size, device=hidden_states.device)
        
        # Compute contrastive loss (InfoNCE)
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return {
            "loss": loss,
            "projections": projections,
            "similarity_matrix": similarity_matrix
        }


class MultiObjectiveTrainer:
    """Trainer that combines multiple self-supervised objectives."""
    
    def __init__(
        self,
        objectives: List[ObjectiveConfig],
        hidden_dim: int = 768,
        vocab_size: int = 256
    ):
        self.objectives = {obj.name: obj for obj in objectives}
        self.current_step = 0
        
        # Initialize objective modules
        self.objective_modules = nn.ModuleDict()
        
        for obj_config in objectives:
            if obj_config.name == "mlm":
                self.objective_modules[obj_config.name] = MaskedLanguageModeling(vocab_size)
            elif obj_config.name == "next_byte":
                self.objective_modules[obj_config.name] = NextByteePrediction(vocab_size)
            elif obj_config.name == "entropy":
                self.objective_modules[obj_config.name] = EntropyPrediction(hidden_dim)
            elif obj_config.name == "boundary":
                self.objective_modules[obj_config.name] = PatchBoundaryPrediction(hidden_dim)
            elif obj_config.name == "contrastive":
                self.objective_modules[obj_config.name] = ContrastiveLearning(hidden_dim)
        
        # Loss weights history
        self.loss_history = {name: [] for name in self.objectives.keys()}
        
    def get_objective_weight(self, objective_name: str) -> float:
        """Get current weight for an objective based on schedule."""
        if objective_name not in self.objectives:
            return 0.0
        
        obj_config = self.objectives[objective_name]
        
        if not obj_config.enabled:
            return 0.0
        
        # Handle warmup
        if self.current_step < obj_config.warmup_steps:
            warmup_factor = self.current_step / obj_config.warmup_steps
            base_weight = obj_config.weight * warmup_factor
        else:
            base_weight = obj_config.weight
        
        # Apply schedule
        if obj_config.schedule == "constant":
            return base_weight
        elif obj_config.schedule == "linear":
            # Linear decay from max to min weight
            decay_factor = max(0.0, 1.0 - (self.current_step - obj_config.warmup_steps) / 10000)
            return obj_config.min_weight + (base_weight - obj_config.min_weight) * decay_factor
        elif obj_config.schedule == "cosine":
            # Cosine annealing
            import math
            decay_factor = 0.5 * (1 + math.cos(math.pi * (self.current_step - obj_config.warmup_steps) / 10000))
            return obj_config.min_weight + (base_weight - obj_config.min_weight) * decay_factor
        
        return base_weight
    
    def compute_loss(
        self,
        model_outputs: Dict[str, Any],
        batch_data: Dict[str, Any]
    ) -> Dict[str, Tensor]:
        """Compute multi-objective loss.
        
        Args:
            model_outputs: Model outputs containing hidden states
            batch_data: Batch data with inputs and targets
            
        Returns:
            Dictionary with losses and metrics
        """
        hidden_states = model_outputs["hidden_states"]
        total_loss = torch.tensor(0.0, device=hidden_states.device)
        individual_losses = {}
        
        # MLM objective
        if "mlm" in self.objective_modules and "mlm_targets" in batch_data:
            weight = self.get_objective_weight("mlm")
            if weight > 0:
                mlm_output = self.objective_modules["mlm"](
                    hidden_states,
                    batch_data["mlm_targets"],
                    batch_data.get("mlm_mask")
                )
                mlm_loss = mlm_output["loss"] * weight
                total_loss += mlm_loss
                individual_losses["mlm"] = mlm_loss
        
        # Next byte prediction
        if "next_byte" in self.objective_modules and "input_ids" in batch_data:
            weight = self.get_objective_weight("next_byte")
            if weight > 0:
                nb_output = self.objective_modules["next_byte"](
                    hidden_states,
                    batch_data["input_ids"]
                )
                nb_loss = nb_output["loss"] * weight
                total_loss += nb_loss
                individual_losses["next_byte"] = nb_loss
        
        # Entropy prediction
        if "entropy" in self.objective_modules and "byte_sequences" in batch_data:
            weight = self.get_objective_weight("entropy")
            if weight > 0:
                entropy_output = self.objective_modules["entropy"](
                    hidden_states,
                    batch_data["byte_sequences"]
                )
                entropy_loss = entropy_output["loss"] * weight
                total_loss += entropy_loss
                individual_losses["entropy"] = entropy_loss
        
        # Patch boundary prediction
        if "boundary" in self.objective_modules and "patch_boundaries" in batch_data:
            weight = self.get_objective_weight("boundary")
            if weight > 0:
                boundary_output = self.objective_modules["boundary"](
                    hidden_states,
                    batch_data["patch_boundaries"]
                )
                boundary_loss = boundary_output["loss"] * weight
                total_loss += boundary_loss
                individual_losses["boundary"] = boundary_loss
        
        # Contrastive learning
        if "contrastive" in self.objective_modules and "byte_sequences" in batch_data:
            weight = self.get_objective_weight("contrastive")
            if weight > 0:
                contrastive_output = self.objective_modules["contrastive"](
                    hidden_states,
                    batch_data["byte_sequences"]
                )
                contrastive_loss = contrastive_output["loss"] * weight
                total_loss += contrastive_loss
                individual_losses["contrastive"] = contrastive_loss
        
        # Update loss history
        for name, loss in individual_losses.items():
            self.loss_history[name].append(loss.item())
        
        return {
            "total_loss": total_loss,
            "individual_losses": individual_losses,
            "loss_weights": {name: self.get_objective_weight(name) for name in self.objectives.keys()}
        }
    
    def step(self):
        """Increment training step."""
        self.current_step += 1
    
    def get_loss_statistics(self, window_size: int = 100) -> Dict[str, Dict[str, float]]:
        """Get loss statistics over recent window."""
        stats = {}
        
        for name, losses in self.loss_history.items():
            if losses:
                recent_losses = losses[-window_size:]
                stats[name] = {
                    "mean": np.mean(recent_losses),
                    "std": np.std(recent_losses),
                    "min": np.min(recent_losses),
                    "max": np.max(recent_losses),
                    "current": recent_losses[-1] if recent_losses else 0.0
                }
        
        return stats


def create_default_objectives() -> List[ObjectiveConfig]:
    """Create default set of training objectives."""
    return [
        ObjectiveConfig(
            name="next_byte",
            weight=1.0,
            enabled=True,
            schedule="constant"
        ),
        ObjectiveConfig(
            name="mlm",
            weight=0.5,
            enabled=True,
            schedule="cosine",
            warmup_steps=1000
        ),
        ObjectiveConfig(
            name="entropy",
            weight=0.2,
            enabled=True,
            schedule="linear",
            warmup_steps=500
        ),
        ObjectiveConfig(
            name="boundary",
            weight=0.3,
            enabled=True,
            schedule="constant",
            warmup_steps=1000
        ),
        ObjectiveConfig(
            name="contrastive",
            weight=0.1,
            enabled=True,
            schedule="cosine",
            warmup_steps=2000
        )
    ]