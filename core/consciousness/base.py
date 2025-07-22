"""
Base class for consciousness-aware modules.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class ConsciousnessAwareModule(ABC):
    """Base class for modules that integrate with consciousness system."""
    
    def __init__(self):
        self._consciousness_context: Optional[Dict[str, Any]] = None
        
    def set_consciousness_context(self, context: Dict[str, Any]):
        """Set the current consciousness context."""
        self._consciousness_context = context
        
    def get_consciousness_context(self) -> Optional[Dict[str, Any]]:
        """Get the current consciousness context."""
        return self._consciousness_context
        
    @abstractmethod
    async def process_with_consciousness(self, input_data: Any) -> Any:
        """Process input with consciousness awareness."""
        pass