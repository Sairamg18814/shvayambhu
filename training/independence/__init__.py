"""
Independence Training Module

Trains Shvayambhu to surpass its teacher models through:
- Self-improvement techniques
- Meta-learning approaches
- Novel capability development
- Teacher model distillation and transcendence
"""

from .trainer import IndependenceTrainer
from .self_improvement import SelfImprovementEngine
from .meta_learning import MetaLearningSystem
from .capability_expansion import CapabilityExpansion

__all__ = [
    'IndependenceTrainer',
    'SelfImprovementEngine',
    'MetaLearningSystem',
    'CapabilityExpansion'
]