"""LoRA (Low-Rank Adaptation) Implementation for SEAL Architecture.

This module provides efficient parameter adaptation through low-rank
decomposition, enabling rapid model self-editing without full retraining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation."""
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.1
    target_modules: List[str] = None
    bias: str = "none"  # "none", "all", "lora_only"
    task_type: str = "CAUSAL_LM"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


class LoRALayer(nn.Module):
    """Base LoRA layer implementation."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize weights
        self.reset_parameters()
        
        # Track if adaptation is enabled
        self.enabled = True
        
    def reset_parameters(self):
        """Initialize LoRA parameters."""
        # Initialize A with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Initialize B to zero (important for LoRA)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through LoRA layer."""
        if not self.enabled:
            return torch.zeros_like(x @ self.lora_A.T @ self.lora_B.T)
        
        # Standard LoRA computation: x @ A^T @ B^T
        result = x @ self.lora_A.T @ self.lora_B.T
        result = self.dropout(result)
        return result * self.scaling
    
    def enable_adaptation(self):
        """Enable LoRA adaptation."""
        self.enabled = True
    
    def disable_adaptation(self):
        """Disable LoRA adaptation."""
        self.enabled = False


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""
    
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        # Add LoRA adaptation
        self.lora = LoRALayer(
            self.in_features,
            self.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass combining base layer and LoRA adaptation."""
        base_output = self.base_layer(x)
        lora_output = self.lora(x)
        return base_output + lora_output
    
    def merge_weights(self):
        """Merge LoRA weights into base layer (destructive operation)."""
        if self.lora.enabled:
            # Compute LoRA weight matrix
            lora_weight = self.lora.lora_B @ self.lora.lora_A * self.lora.scaling
            
            # Add to base weight
            with torch.no_grad():
                self.base_layer.weight.data += lora_weight
            
            # Reset LoRA
            self.lora.reset_parameters()
            
        logger.info("Merged LoRA weights into base layer")
    
    def unmerge_weights(self):
        """Separate merged weights (requires storing original weights)."""
        # This would require storing original weights separately
        logger.warning("Weight unmerging not implemented - requires weight backup")


class LoRAEmbedding(nn.Module):
    """Embedding layer with LoRA adaptation."""
    
    def __init__(
        self,
        base_layer: nn.Embedding,
        rank: int = 16,
        alpha: float = 32.0
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.num_embeddings = base_layer.num_embeddings
        self.embedding_dim = base_layer.embedding_dim
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        # LoRA parameters for embeddings
        self.lora_A = nn.Parameter(torch.zeros(rank, self.num_embeddings))
        self.lora_B = nn.Parameter(torch.zeros(self.embedding_dim, rank))
        self.scaling = alpha / rank
        
        # Initialize
        nn.init.zeros_(self.lora_A)
        nn.init.normal_(self.lora_B)
        
    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass with LoRA adaptation."""
        base_output = self.base_layer(input_ids)
        
        # LoRA adaptation for embeddings
        lora_output = F.embedding(
            input_ids,
            (self.lora_B @ self.lora_A).T,
            self.base_layer.padding_idx,
            self.base_layer.max_norm,
            self.base_layer.norm_type,
            self.base_layer.scale_grad_by_freq,
            self.base_layer.sparse
        )
        
        return base_output + lora_output * self.scaling


class LoRAAdapter:
    """Main LoRA adapter for managing adaptations across a model."""
    
    def __init__(self, config: LoRAConfig):
        self.config = config
        self.adapted_modules: Dict[str, nn.Module] = {}
        self.original_modules: Dict[str, nn.Module] = {}
        self.adaptation_history: List[Dict[str, Any]] = []
        
        # Active adaptations
        self.active_adaptations: Set[str] = set()
        
    def add_adapter(
        self,
        model: nn.Module,
        target_modules: Optional[List[str]] = None
    ) -> None:
        """Add LoRA adapters to target modules in the model."""
        target_modules = target_modules or self.config.target_modules
        
        logger.info(f"Adding LoRA adapters to modules: {target_modules}")
        
        for name, module in model.named_modules():
            if self._should_adapt_module(name, module, target_modules):
                self._adapt_module(name, module, model)
        
        logger.info(f"Added {len(self.adapted_modules)} LoRA adapters")
    
    def _should_adapt_module(
        self,
        name: str,
        module: nn.Module,
        target_modules: List[str]
    ) -> bool:
        """Check if a module should be adapted."""
        # Check if module name contains any target module string
        module_name = name.split('.')[-1]
        return any(target in module_name for target in target_modules)
    
    def _adapt_module(
        self,
        name: str,
        module: nn.Module,
        model: nn.Module
    ) -> None:
        """Adapt a specific module with LoRA."""
        try:
            # Store original module
            self.original_modules[name] = module
            
            # Create LoRA adaptation based on module type
            if isinstance(module, nn.Linear):
                adapted_module = LoRALinear(
                    module,
                    rank=self.config.rank,
                    alpha=self.config.alpha,
                    dropout=self.config.dropout
                )
            elif isinstance(module, nn.Embedding):
                adapted_module = LoRAEmbedding(
                    module,
                    rank=self.config.rank,
                    alpha=self.config.alpha
                )
            else:
                logger.warning(f"Unsupported module type for LoRA: {type(module)}")
                return
            
            # Replace module in model
            self._replace_module(model, name, adapted_module)
            self.adapted_modules[name] = adapted_module
            
            logger.debug(f"Adapted module: {name}")
            
        except Exception as e:
            logger.error(f"Failed to adapt module {name}: {e}")
    
    def _replace_module(
        self,
        model: nn.Module,
        module_name: str,
        new_module: nn.Module
    ) -> None:
        """Replace a module in the model hierarchy."""
        parts = module_name.split('.')
        parent = model
        
        # Navigate to parent module
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        # Replace the target module
        setattr(parent, parts[-1], new_module)
    
    def apply_adaptation(
        self,
        adaptation_data: Dict[str, Tensor],
        adaptation_id: str = None
    ) -> None:
        """Apply a specific adaptation to the model."""
        adaptation_id = adaptation_id or f"adaptation_{len(self.adaptation_history)}"
        
        logger.info(f"Applying adaptation: {adaptation_id}")
        
        applied_modules = []
        
        for module_name, adapted_module in self.adapted_modules.items():
            if module_name in adaptation_data:
                try:
                    # Get adaptation parameters
                    adaptation = adaptation_data[module_name]
                    
                    # Apply to LoRA layer
                    if hasattr(adapted_module, 'lora'):
                        self._apply_lora_adaptation(adapted_module.lora, adaptation)
                        applied_modules.append(module_name)
                    
                except Exception as e:
                    logger.error(f"Failed to apply adaptation to {module_name}: {e}")
        
        # Record adaptation
        self.adaptation_history.append({
            'id': adaptation_id,
            'modules': applied_modules,
            'timestamp': torch.tensor(0).item(),  # Simplified timestamp
            'active': True
        })
        
        self.active_adaptations.add(adaptation_id)
        
        logger.info(f"Applied adaptation to {len(applied_modules)} modules")
    
    def _apply_lora_adaptation(
        self,
        lora_layer: LoRALayer,
        adaptation: Dict[str, Tensor]
    ) -> None:
        """Apply adaptation parameters to a LoRA layer."""
        if 'lora_A' in adaptation:
            lora_layer.lora_A.data = adaptation['lora_A']
        
        if 'lora_B' in adaptation:
            lora_layer.lora_B.data = adaptation['lora_B']
        
        if 'scaling' in adaptation:
            lora_layer.scaling = adaptation['scaling'].item()
    
    def remove_adaptation(self, adaptation_id: str) -> None:
        """Remove a specific adaptation."""
        if adaptation_id not in self.active_adaptations:
            logger.warning(f"Adaptation {adaptation_id} not active")
            return
        
        # Find adaptation in history
        adaptation_record = None
        for record in self.adaptation_history:
            if record['id'] == adaptation_id:
                adaptation_record = record
                break
        
        if not adaptation_record:
            logger.error(f"Adaptation {adaptation_id} not found in history")
            return
        
        # Reset affected modules
        for module_name in adaptation_record['modules']:
            if module_name in self.adapted_modules:
                adapted_module = self.adapted_modules[module_name]
                if hasattr(adapted_module, 'lora'):
                    adapted_module.lora.reset_parameters()
        
        # Mark as inactive
        adaptation_record['active'] = False
        self.active_adaptations.discard(adaptation_id)
        
        logger.info(f"Removed adaptation: {adaptation_id}")
    
    def merge_all_adaptations(self) -> None:
        """Merge all active adaptations into base weights."""
        logger.info("Merging all LoRA adaptations into base weights")
        
        for name, adapted_module in self.adapted_modules.items():
            if hasattr(adapted_module, 'merge_weights'):
                try:
                    adapted_module.merge_weights()
                except Exception as e:
                    logger.error(f"Failed to merge weights for {name}: {e}")
        
        # Clear adaptation history
        self.adaptation_history.clear()
        self.active_adaptations.clear()
        
        logger.info("Completed LoRA weight merging")
    
    def enable_all_adaptations(self) -> None:
        """Enable all LoRA adaptations."""
        for adapted_module in self.adapted_modules.values():
            if hasattr(adapted_module, 'lora'):
                adapted_module.lora.enable_adaptation()
        
        logger.info("Enabled all LoRA adaptations")
    
    def disable_all_adaptations(self) -> None:
        """Disable all LoRA adaptations."""
        for adapted_module in self.adapted_modules.values():
            if hasattr(adapted_module, 'lora'):
                adapted_module.lora.disable_adaptation()
        
        logger.info("Disabled all LoRA adaptations")
    
    def get_adaptation_parameters(self) -> Dict[str, Tensor]:
        """Get current adaptation parameters."""
        parameters = {}
        
        for name, adapted_module in self.adapted_modules.items():
            if hasattr(adapted_module, 'lora'):
                lora = adapted_module.lora
                parameters[name] = {
                    'lora_A': lora.lora_A.clone(),
                    'lora_B': lora.lora_B.clone(),
                    'scaling': torch.tensor(lora.scaling)
                }
        
        return parameters
    
    def save_adaptations(self, filepath: str) -> None:
        """Save adaptation state to file."""
        state = {
            'config': self.config.__dict__,
            'parameters': self.get_adaptation_parameters(),
            'history': self.adaptation_history,
            'active_adaptations': list(self.active_adaptations)
        }
        
        torch.save(state, filepath)
        logger.info(f"Saved LoRA adaptations to {filepath}")
    
    def load_adaptations(self, filepath: str) -> None:
        """Load adaptation state from file."""
        state = torch.load(filepath)
        
        # Apply saved parameters
        if 'parameters' in state:
            self.apply_adaptation(state['parameters'], 'loaded_adaptation')
        
        # Restore history and active adaptations
        if 'history' in state:
            self.adaptation_history = state['history']
        
        if 'active_adaptations' in state:
            self.active_adaptations = set(state['active_adaptations'])
        
        logger.info(f"Loaded LoRA adaptations from {filepath}")
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get all trainable LoRA parameters."""
        params = []
        
        for adapted_module in self.adapted_modules.values():
            if hasattr(adapted_module, 'lora'):
                params.extend(adapted_module.lora.parameters())
        
        return params
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter counts for LoRA adaptations."""
        lora_params = 0
        total_params = 0
        
        for adapted_module in self.adapted_modules.values():
            if hasattr(adapted_module, 'lora'):
                lora_params += sum(p.numel() for p in adapted_module.lora.parameters())
            
            if hasattr(adapted_module, 'base_layer'):
                total_params += sum(p.numel() for p in adapted_module.base_layer.parameters())
        
        return {
            'lora_parameters': lora_params,
            'base_parameters': total_params,
            'adaptation_ratio': lora_params / max(total_params, 1)
        }
    
    def print_adaptation_info(self) -> None:
        """Print information about current adaptations."""
        param_counts = self.get_parameter_count()
        
        print(f"LoRA Adapter Information:")
        print(f"  Adapted modules: {len(self.adapted_modules)}")
        print(f"  LoRA parameters: {param_counts['lora_parameters']:,}")
        print(f"  Base parameters: {param_counts['base_parameters']:,}")
        print(f"  Adaptation ratio: {param_counts['adaptation_ratio']:.4f}")
        print(f"  Active adaptations: {len(self.active_adaptations)}")
        print(f"  Rank: {self.config.rank}")
        print(f"  Alpha: {self.config.alpha}")


def create_lora_adapter(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 32.0,
    target_modules: Optional[List[str]] = None
) -> LoRAAdapter:
    """Create and attach LoRA adapter to a model."""
    
    config = LoRAConfig(
        rank=rank,
        alpha=alpha,
        target_modules=target_modules
    )
    
    adapter = LoRAAdapter(config)
    adapter.add_adapter(model)
    
    logger.info(f"Created LoRA adapter with rank {rank}")
    
    return adapter


def get_lora_parameters_only(model: nn.Module) -> List[nn.Parameter]:
    """Extract only LoRA parameters from a model."""
    lora_params = []
    
    for module in model.modules():
        if hasattr(module, 'lora'):
            lora_params.extend(module.lora.parameters())
    
    return lora_params