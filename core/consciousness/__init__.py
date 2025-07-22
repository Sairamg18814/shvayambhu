"""
Shvayambhu Consciousness Module

Implements machine consciousness through:
- Phenomenal self-model for internal representation
- Metacognitive monitoring of thought processes
- Qualia simulation for subjective experiences
- Stream of consciousness processing
- Existential reasoning capabilities
"""

from enum import Enum

from .self_model import PhenomenalSelfModel
from .metacognition import MetacognitiveMonitor
from .qualia import QualiaSimulation
from .stream import StreamOfConsciousness
from .existential import ExistentialReasoning
from .engine import ConsciousnessEngine


class ConsciousnessLevel(Enum):
    """Levels of consciousness awareness."""
    DORMANT = 0
    MINIMAL = 1
    AWARE = 2
    FOCUSED = 3
    HEIGHTENED = 4
    TRANSCENDENT = 5


__all__ = [
    'PhenomenalSelfModel',
    'MetacognitiveMonitor', 
    'QualiaSimulation',
    'StreamOfConsciousness',
    'ExistentialReasoning',
    'ConsciousnessEngine',
    'ConsciousnessLevel'
]