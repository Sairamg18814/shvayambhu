"""Edit generation module for SEAL.

This module generates self-edit instructions based on model performance,
feedback, and learning objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging

from .edit_format import EditInstruction, EditType, EditScope, EditBatch

logger = logging.getLogger(__name__)


@dataclass
class EditContext:
    """Context for edit generation."""
    input_text: str
    model_output: str
    expected_output: Optional[str] = None
    feedback: Optional[str] = None
    performance_metrics: Dict[str, float] = None
    error_analysis: Dict[str, Any] = None
    

class EditGenerator(nn.Module):
    """Generates self-edit instructions based on context."""
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        edit_vocab_size: int = 50000,
        max_edit_length: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_edit_length = max_edit_length
        
        # Encoder for context
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Edit type classifier
        self.edit_type_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(EditType))
        )
        
        # Edit decoder
        self.edit_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, edit_vocab_size)
        
        # Embeddings
        self.token_embedding = nn.Embedding(edit_vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_edit_length, hidden_dim)
        
        # Special tokens
        self.start_token_id = 0
        self.end_token_id = 1
        self.pad_token_id = 2
        
    def forward(
        self,
        context: EditContext,
        max_edits: int = 5
    ) -> List[EditInstruction]:
        """Generate edit instructions from context."""
        # Encode context
        context_encoding = self._encode_context(context)
        
        # Classify edit types needed
        edit_types = self._classify_edit_types(context_encoding)
        
        # Generate edit instructions
        instructions = []
        for edit_type in edit_types[:max_edits]:
            instruction = self._generate_edit(
                context_encoding, edit_type
            )
            if instruction:
                instructions.append(instruction)
        
        return instructions
    
    def _encode_context(self, context: EditContext) -> torch.Tensor:
        """Encode the context into a fixed representation."""
        # This is a simplified version - in practice, you'd encode
        # all context fields properly
        
        # For now, return a dummy encoding
        batch_size = 1
        seq_len = 128
        return torch.randn(batch_size, seq_len, self.hidden_dim)
    
    def _classify_edit_types(self, context_encoding: torch.Tensor) -> List[EditType]:
        """Classify which edit types are needed."""
        # Pool the context encoding
        pooled = context_encoding.mean(dim=1)  # [batch, hidden]
        
        # Get edit type logits
        logits = self.edit_type_classifier(pooled)  # [batch, num_edit_types]
        probs = F.softmax(logits, dim=-1)
        
        # Get top edit types
        top_probs, top_indices = torch.topk(probs[0], k=min(5, len(EditType)))
        
        edit_types = []
        for idx, prob in zip(top_indices, top_probs):
            if prob > 0.1:  # Threshold
                edit_types.append(list(EditType)[idx.item()])
        
        return edit_types
    
    def _generate_edit(
        self,
        context_encoding: torch.Tensor,
        edit_type: EditType
    ) -> Optional[EditInstruction]:
        """Generate a single edit instruction."""
        # Start with the start token
        device = context_encoding.device
        generated = torch.tensor([[self.start_token_id]], device=device)
        
        # Generate tokens autoregressively
        for _ in range(self.max_edit_length):
            # Get embeddings
            token_emb = self.token_embedding(generated)
            pos_ids = torch.arange(generated.size(1), device=device).unsqueeze(0)
            pos_emb = self.position_embedding(pos_ids)
            
            # Decode
            decoder_input = token_emb + pos_emb
            output = self.edit_decoder(
                decoder_input,
                context_encoding
            )
            
            # Project to vocabulary
            logits = self.output_projection(output[:, -1, :])  # [batch, vocab]
            
            # Sample next token
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [batch, 1]
            
            # Stop if end token
            if next_token.item() == self.end_token_id:
                break
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)
        
        # Convert to EditInstruction (simplified)
        return EditInstruction(
            edit_type=edit_type,
            description=f"Generated edit for {edit_type.value}",
            target="model_behavior",
            operation="modify",
            value=None,
            scope=EditScope(global_edit=True),
            confidence=0.8
        )


class RuleBasedEditGenerator:
    """Rule-based edit generator for bootstrapping."""
    
    def __init__(self):
        self.error_patterns = {
            "factual_error": self._generate_knowledge_update,
            "inconsistent_behavior": self._generate_behavior_modification,
            "skill_gap": self._generate_skill_enhancement,
            "bias_detected": self._generate_bias_correction
        }
    
    def generate(
        self,
        context: EditContext,
        max_edits: int = 5
    ) -> List[EditInstruction]:
        """Generate edits based on rules."""
        instructions = []
        
        # Analyze context for errors
        errors = self._analyze_errors(context)
        
        # Generate edits for each error type
        for error_type, error_info in errors.items():
            if error_type in self.error_patterns:
                instruction = self.error_patterns[error_type](error_info, context)
                if instruction:
                    instructions.append(instruction)
                    if len(instructions) >= max_edits:
                        break
        
        return instructions
    
    def _analyze_errors(self, context: EditContext) -> Dict[str, Any]:
        """Analyze context for different error types."""
        errors = {}
        
        # Check for factual errors
        if context.expected_output and context.model_output:
            if self._is_factual_error(context.model_output, context.expected_output):
                errors["factual_error"] = {
                    "expected": context.expected_output,
                    "actual": context.model_output
                }
        
        # Check for inconsistent behavior
        if context.feedback and "inconsistent" in context.feedback.lower():
            errors["inconsistent_behavior"] = {
                "feedback": context.feedback
            }
        
        # Check performance metrics
        if context.performance_metrics:
            if context.performance_metrics.get("accuracy", 1.0) < 0.7:
                errors["skill_gap"] = {
                    "metric": "accuracy",
                    "value": context.performance_metrics["accuracy"]
                }
        
        return errors
    
    def _is_factual_error(self, output: str, expected: str) -> bool:
        """Check if output contains factual error."""
        # Simple heuristic - check if key facts differ
        output_lower = output.lower()
        expected_lower = expected.lower()
        
        # Extract numbers, dates, names (simplified)
        import re
        output_facts = set(re.findall(r'\b\d+\b|\b[A-Z][a-z]+\b', output))
        expected_facts = set(re.findall(r'\b\d+\b|\b[A-Z][a-z]+\b', expected))
        
        # If facts differ significantly, it's likely an error
        return len(output_facts ^ expected_facts) > len(output_facts & expected_facts)
    
    def _generate_knowledge_update(
        self,
        error_info: Dict[str, Any],
        context: EditContext
    ) -> EditInstruction:
        """Generate knowledge update edit."""
        return EditInstruction(
            edit_type=EditType.KNOWLEDGE_UPDATE,
            description=f"Update knowledge: expected '{error_info['expected']}' "
                       f"but generated '{error_info['actual']}'",
            target="factual_knowledge",
            operation="update",
            value=error_info['expected'],
            scope=EditScope(global_edit=True),
            strength=0.8,
            confidence=0.9
        )
    
    def _generate_behavior_modification(
        self,
        error_info: Dict[str, Any],
        context: EditContext
    ) -> EditInstruction:
        """Generate behavior modification edit."""
        return EditInstruction(
            edit_type=EditType.BEHAVIOR_MODIFICATION,
            description=f"Modify behavior based on feedback: {error_info['feedback']}",
            target="response_generation",
            operation="modify",
            value=error_info['feedback'],
            scope=EditScope(global_edit=True),
            strength=0.7,
            confidence=0.8
        )
    
    def _generate_skill_enhancement(
        self,
        error_info: Dict[str, Any],
        context: EditContext
    ) -> EditInstruction:
        """Generate skill enhancement edit."""
        return EditInstruction(
            edit_type=EditType.SKILL_ENHANCEMENT,
            description=f"Enhance {error_info['metric']} performance "
                       f"(current: {error_info['value']:.2f})",
            target=error_info['metric'],
            operation="enhance",
            value=None,
            scope=EditScope(global_edit=True),
            strength=0.6,
            confidence=0.7
        )
    
    def _generate_bias_correction(
        self,
        error_info: Dict[str, Any],
        context: EditContext
    ) -> EditInstruction:
        """Generate bias correction edit."""
        return EditInstruction(
            edit_type=EditType.BIAS_CORRECTION,
            description="Correct detected bias in model output",
            target="bias_patterns",
            operation="correct",
            value=None,
            scope=EditScope(global_edit=True),
            strength=0.9,
            confidence=0.85
        )


def create_edit_generator(
    model_type: str = "neural",
    **kwargs
) -> Union[EditGenerator, RuleBasedEditGenerator]:
    """Factory function to create edit generator."""
    if model_type == "neural":
        return EditGenerator(**kwargs)
    elif model_type == "rules":
        return RuleBasedEditGenerator()
    else:
        raise ValueError(f"Unknown generator type: {model_type}")
