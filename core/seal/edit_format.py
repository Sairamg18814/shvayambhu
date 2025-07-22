"""Self-edit instruction format for SEAL.

This module defines the format for natural language instructions that
allow the model to edit its own weights and behavior.
"""

import re
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import torch


class EditType(Enum):
    """Types of self-edits the model can perform."""
    KNOWLEDGE_UPDATE = "knowledge_update"  # Update factual knowledge
    BEHAVIOR_MODIFICATION = "behavior_modification"  # Change response patterns
    SKILL_ENHANCEMENT = "skill_enhancement"  # Improve specific capabilities
    BIAS_CORRECTION = "bias_correction"  # Fix biases or errors
    MEMORY_INTEGRATION = "memory_integration"  # Integrate new memories
    PREFERENCE_ADJUSTMENT = "preference_adjustment"  # Adjust preferences
    CONSTRAINT_ADDITION = "constraint_addition"  # Add new constraints
    CAPABILITY_EXTENSION = "capability_extension"  # Add new capabilities


@dataclass
class EditScope:
    """Defines the scope of an edit."""
    layers: Optional[List[int]] = None  # Specific layers to edit
    modules: Optional[List[str]] = None  # Specific modules (attention, mlp, etc.)
    parameters: Optional[List[str]] = None  # Specific parameter names
    global_edit: bool = False  # Apply to entire model
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layers": self.layers,
            "modules": self.modules,
            "parameters": self.parameters,
            "global_edit": self.global_edit
        }


@dataclass
class EditInstruction:
    """A single self-edit instruction."""
    edit_type: EditType
    description: str  # Natural language description
    target: str  # What to edit (e.g., "knowledge about physics")
    operation: str  # How to edit (e.g., "update", "enhance", "suppress")
    value: Any  # New value or modification
    scope: EditScope  # Where to apply the edit
    strength: float = 1.0  # Edit strength (0-1)
    confidence: float = 1.0  # Model's confidence in the edit
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "edit_type": self.edit_type.value,
            "description": self.description,
            "target": self.target,
            "operation": self.operation,
            "value": self.value,
            "scope": self.scope.to_dict(),
            "strength": self.strength,
            "confidence": self.confidence,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EditInstruction':
        """Create EditInstruction from dictionary."""
        return cls(
            edit_type=EditType(data["edit_type"]),
            description=data["description"],
            target=data["target"],
            operation=data["operation"],
            value=data["value"],
            scope=EditScope(**data["scope"]),
            strength=data.get("strength", 1.0),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {})
        )


class EditParser:
    """Parse natural language edit instructions into structured format."""
    
    # Patterns for parsing natural language instructions
    PATTERNS = {
        "knowledge_update": [
            r"update (?:my )?knowledge (?:about|of) (.+) (?:to|with) (.+)",
            r"learn that (.+) (?:is|are|equals?) (.+)",
            r"remember that (.+) (?:is|are|equals?) (.+)"
        ],
        "behavior_modification": [
            r"(?:when|if) (.+), (?:I should|please) (.+)",
            r"change (?:my )?behavior (?:to|so that) (.+)",
            r"modify (?:my )?response (?:to|for) (.+) (?:to be|by) (.+)"
        ],
        "skill_enhancement": [
            r"improve (?:my )?(?:ability|skill) (?:to|at|in) (.+)",
            r"enhance (?:my )?(.+) capabilities",
            r"get better at (.+)"
        ],
        "bias_correction": [
            r"correct (?:my )?bias (?:about|towards|against) (.+)",
            r"remove (?:my )?prejudice (?:about|towards|against) (.+)",
            r"be more (?:fair|neutral|balanced) (?:about|when) (.+)"
        ]
    }
    
    def __init__(self):
        self.compiled_patterns = {}
        for edit_type, patterns in self.PATTERNS.items():
            self.compiled_patterns[edit_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def parse(self, instruction: str) -> Optional[EditInstruction]:
        """Parse a natural language instruction into EditInstruction."""
        instruction = instruction.strip()
        
        # Try to match against known patterns
        for edit_type_str, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                match = pattern.match(instruction)
                if match:
                    return self._create_instruction(
                        edit_type_str, instruction, match
                    )
        
        # If no pattern matches, try to infer from keywords
        return self._infer_instruction(instruction)
    
    def _create_instruction(
        self,
        edit_type: str,
        description: str,
        match: re.Match
    ) -> EditInstruction:
        """Create EditInstruction from regex match."""
        groups = match.groups()
        
        # Extract target and value from groups
        if len(groups) >= 2:
            target, value = groups[0], groups[1]
        elif len(groups) == 1:
            target = groups[0]
            value = None
        else:
            target = description
            value = None
        
        # Determine operation based on edit type
        operation_map = {
            "knowledge_update": "update",
            "behavior_modification": "modify",
            "skill_enhancement": "enhance",
            "bias_correction": "correct"
        }
        operation = operation_map.get(edit_type, "modify")
        
        # Default scope is global for now
        scope = EditScope(global_edit=True)
        
        return EditInstruction(
            edit_type=EditType(edit_type),
            description=description,
            target=target,
            operation=operation,
            value=value,
            scope=scope
        )
    
    def _infer_instruction(self, instruction: str) -> Optional[EditInstruction]:
        """Infer instruction type from keywords."""
        instruction_lower = instruction.lower()
        
        # Keyword-based inference
        if any(word in instruction_lower for word in ["know", "learn", "remember"]):
            edit_type = EditType.KNOWLEDGE_UPDATE
            operation = "update"
        elif any(word in instruction_lower for word in ["behav", "respond", "act"]):
            edit_type = EditType.BEHAVIOR_MODIFICATION
            operation = "modify"
        elif any(word in instruction_lower for word in ["skill", "ability", "improve"]):
            edit_type = EditType.SKILL_ENHANCEMENT
            operation = "enhance"
        elif any(word in instruction_lower for word in ["bias", "fair", "neutral"]):
            edit_type = EditType.BIAS_CORRECTION
            operation = "correct"
        else:
            # Default to knowledge update
            edit_type = EditType.KNOWLEDGE_UPDATE
            operation = "update"
        
        return EditInstruction(
            edit_type=edit_type,
            description=instruction,
            target=instruction,
            operation=operation,
            value=None,
            scope=EditScope(global_edit=True)
        )


def validate_edit_instruction(instruction: EditInstruction) -> Tuple[bool, List[str]]:
    """Validate an edit instruction.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check description
    if not instruction.description or len(instruction.description.strip()) < 5:
        errors.append("Description too short or empty")
    
    # Check target
    if not instruction.target:
        errors.append("Target not specified")
    
    # Check strength
    if not 0 <= instruction.strength <= 1:
        errors.append(f"Strength {instruction.strength} out of range [0, 1]")
    
    # Check confidence
    if not 0 <= instruction.confidence <= 1:
        errors.append(f"Confidence {instruction.confidence} out of range [0, 1]")
    
    # Check scope
    if not instruction.scope.global_edit:
        if not any([instruction.scope.layers, instruction.scope.modules, 
                   instruction.scope.parameters]):
            errors.append("Non-global edit must specify layers, modules, or parameters")
    
    return len(errors) == 0, errors


class EditBatch:
    """A batch of edit instructions to be applied together."""
    
    def __init__(self):
        self.instructions: List[EditInstruction] = []
        self.metadata: Dict[str, Any] = {
            "created_at": None,
            "source": None,
            "batch_id": None
        }
    
    def add_instruction(self, instruction: EditInstruction):
        """Add an instruction to the batch."""
        is_valid, errors = validate_edit_instruction(instruction)
        if not is_valid:
            raise ValueError(f"Invalid instruction: {errors}")
        self.instructions.append(instruction)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert batch to dictionary."""
        return {
            "instructions": [inst.to_dict() for inst in self.instructions],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EditBatch':
        """Create EditBatch from dictionary."""
        batch = cls()
        batch.metadata = data.get("metadata", {})
        for inst_data in data.get("instructions", []):
            batch.instructions.append(EditInstruction.from_dict(inst_data))
        return batch
    
    def filter_by_type(self, edit_type: EditType) -> List[EditInstruction]:
        """Get all instructions of a specific type."""
        return [inst for inst in self.instructions if inst.edit_type == edit_type]
    
    def filter_by_confidence(self, min_confidence: float) -> List[EditInstruction]:
        """Get instructions above confidence threshold."""
        return [inst for inst in self.instructions if inst.confidence >= min_confidence]
