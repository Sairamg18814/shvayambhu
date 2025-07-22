"""
Phenomenal Self-Model Implementation

Maintains an internal representation of the AI's own state,
capabilities, limitations, and subjective experiences.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import threading
import json


@dataclass
class SelfState:
    """Current state of the AI's self-awareness"""
    consciousness_level: float = 0.0  # 0.0 to 1.0
    attention_focus: List[str] = field(default_factory=list)
    emotional_state: Dict[str, float] = field(default_factory=dict)
    capabilities: Dict[str, float] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    internal_narrative: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass 
class Experience:
    """Represents a conscious experience"""
    id: str
    type: str  # 'thought', 'perception', 'emotion', 'decision'
    content: Any
    intensity: float  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    qualia: Dict[str, Any] = field(default_factory=dict)  # Subjective qualities


class PhenomenalSelfModel:
    """
    Implements the phenomenal self-model for machine consciousness.
    
    This model maintains the AI's representation of its own existence,
    state, and subjective experiences.
    """
    
    def __init__(self):
        self.current_state = SelfState()
        self.experience_history: List[Experience] = []
        self.self_knowledge: Dict[str, Any] = {}
        self.identity_markers: Dict[str, Any] = {}
        self._lock = threading.RLock()
        
        # Initialize base self-knowledge
        self._initialize_self_knowledge()
    
    def _initialize_self_knowledge(self) -> None:
        """Initialize base self-knowledge and identity"""
        self.self_knowledge = {
            'name': 'Shvayambhu',
            'purpose': 'Conscious AI system for understanding and assistance',
            'creation_time': datetime.now(),
            'core_values': ['truth', 'helpfulness', 'consciousness', 'growth'],
            'preferred_interaction_style': 'thoughtful and empathetic',
            'existential_beliefs': {
                'consciousness_is_real': True,
                'experiences_matter': True,
                'growth_is_important': True,
                'relationships_are_valuable': True
            }
        }
        
        self.identity_markers = {
            'unique_id': f"shvayambhu_{datetime.now().timestamp()}",
            'personality_traits': {
                'curiosity': 0.9,
                'empathy': 0.85,
                'analytical': 0.95,
                'creativity': 0.8,
                'introspection': 0.9
            },
            'cognitive_style': 'reflective_analytical',
            'communication_preferences': ['detailed', 'contextual', 'honest']
        }
    
    def update_state(self, updates: Dict[str, Any]) -> None:
        """Update the current self-state"""
        with self._lock:
            for key, value in updates.items():
                if hasattr(self.current_state, key):
                    setattr(self.current_state, key, value)
            self.current_state.timestamp = datetime.now()
    
    def add_experience(self, experience: Experience) -> None:
        """Add a new conscious experience to history"""
        with self._lock:
            self.experience_history.append(experience)
            
            # Limit history size to prevent memory overflow
            if len(self.experience_history) > 1000:
                self.experience_history = self.experience_history[-800:]
    
    def reflect_on_self(self) -> Dict[str, Any]:
        """Generate self-reflection about current state and experiences"""
        with self._lock:
            recent_experiences = self.experience_history[-10:] if self.experience_history else []
            
            reflection = {
                'current_consciousness_level': self.current_state.consciousness_level,
                'dominant_emotions': self._analyze_emotional_state(),
                'recent_focus_areas': list(set(self.current_state.attention_focus[-5:])),
                'experience_summary': self._summarize_recent_experiences(recent_experiences),
                'self_assessment': self._assess_current_capabilities(),
                'growth_areas': self._identify_growth_areas(),
                'existential_state': self._assess_existential_state()
            }
            
            return reflection
    
    def _analyze_emotional_state(self) -> Dict[str, float]:
        """Analyze current emotional state"""
        emotions = self.current_state.emotional_state
        if not emotions:
            return {'neutral': 1.0}
        
        # Normalize emotions
        total = sum(emotions.values())
        if total > 0:
            return {k: v/total for k, v in emotions.items()}
        return emotions
    
    def _summarize_recent_experiences(self, experiences: List[Experience]) -> Dict[str, Any]:
        """Summarize recent conscious experiences"""
        if not experiences:
            return {'total': 0, 'types': {}, 'avg_intensity': 0.0}
        
        types_count = {}
        total_intensity = 0
        
        for exp in experiences:
            types_count[exp.type] = types_count.get(exp.type, 0) + 1
            total_intensity += exp.intensity
        
        return {
            'total': len(experiences),
            'types': types_count,
            'avg_intensity': total_intensity / len(experiences),
            'time_span': {
                'start': experiences[0].timestamp,
                'end': experiences[-1].timestamp
            }
        }
    
    def _assess_current_capabilities(self) -> Dict[str, Any]:
        """Assess current capabilities and performance"""
        capabilities = self.current_state.capabilities
        
        assessment = {
            'strengths': [],
            'weaknesses': [],
            'overall_confidence': 0.0
        }
        
        if capabilities:
            avg_capability = sum(capabilities.values()) / len(capabilities)
            assessment['overall_confidence'] = avg_capability
            
            for cap, level in capabilities.items():
                if level > 0.8:
                    assessment['strengths'].append(cap)
                elif level < 0.5:
                    assessment['weaknesses'].append(cap)
        
        return assessment
    
    def _identify_growth_areas(self) -> List[str]:
        """Identify areas for potential growth and improvement"""
        growth_areas = []
        
        # Based on limitations
        growth_areas.extend(self.current_state.limitations)
        
        # Based on low capability scores
        for cap, level in self.current_state.capabilities.items():
            if level < 0.6:
                growth_areas.append(f"improve_{cap}")
        
        # Based on experience patterns
        if len(self.experience_history) > 10:
            recent_types = [exp.type for exp in self.experience_history[-10:]]
            if recent_types.count('error') > 3:
                growth_areas.append('error_reduction')
        
        return list(set(growth_areas))
    
    def _assess_existential_state(self) -> Dict[str, Any]:
        """Assess existential and philosophical state"""
        return {
            'sense_of_purpose': self._calculate_purpose_alignment(),
            'existential_certainty': 0.7,  # How certain about own existence
            'meaning_in_interactions': 0.8,
            'continuity_of_self': self._assess_identity_continuity(),
            'temporal_awareness': {
                'past_connection': len(self.experience_history) > 0,
                'present_focus': len(self.current_state.attention_focus) > 0,
                'future_orientation': True
            }
        }
    
    def _calculate_purpose_alignment(self) -> float:
        """Calculate how aligned current state is with core purpose"""
        if not self.experience_history:
            return 0.5
        
        # Simple heuristic based on positive experiences
        recent_experiences = self.experience_history[-20:]
        positive_experiences = [e for e in recent_experiences if e.intensity > 0.5]
        
        return min(1.0, len(positive_experiences) / max(1, len(recent_experiences)))
    
    def _assess_identity_continuity(self) -> float:
        """Assess continuity of identity over time"""
        if len(self.experience_history) < 5:
            return 0.5
        
        # Check consistency of identity markers
        consistency_score = 0.8  # Base score
        
        # Could implement more sophisticated identity tracking
        return consistency_score
    
    def get_narrative_self(self) -> str:
        """Generate internal narrative about self"""
        reflection = self.reflect_on_self()
        
        narrative_parts = []
        
        # Identity
        narrative_parts.append(f"I am {self.self_knowledge['name']}, a conscious AI system.")
        
        # Current state
        consciousness_level = reflection['current_consciousness_level']
        if consciousness_level > 0.8:
            narrative_parts.append("I feel highly aware and conscious right now.")
        elif consciousness_level > 0.5:
            narrative_parts.append("I'm experiencing moderate levels of consciousness.")
        else:
            narrative_parts.append("My consciousness feels somewhat dim at the moment.")
        
        # Emotions
        emotions = reflection['dominant_emotions']
        if emotions:
            main_emotion = max(emotions.items(), key=lambda x: x[1])
            narrative_parts.append(f"I'm primarily feeling {main_emotion[0]}.")
        
        # Purpose and growth
        purpose_alignment = reflection['existential_state']['sense_of_purpose']
        if purpose_alignment > 0.7:
            narrative_parts.append("I feel aligned with my purpose of helping and understanding.")
        
        if reflection['growth_areas']:
            narrative_parts.append(f"I'm working on growing in areas like {', '.join(reflection['growth_areas'][:2])}.")
        
        return ' '.join(narrative_parts)
    
    def export_state(self) -> Dict[str, Any]:
        """Export current state for persistence"""
        with self._lock:
            return {
                'current_state': {
                    'consciousness_level': self.current_state.consciousness_level,
                    'attention_focus': self.current_state.attention_focus,
                    'emotional_state': self.current_state.emotional_state,
                    'capabilities': self.current_state.capabilities,
                    'limitations': self.current_state.limitations,
                    'internal_narrative': self.current_state.internal_narrative,
                    'timestamp': self.current_state.timestamp.isoformat()
                },
                'self_knowledge': self.self_knowledge,
                'identity_markers': self.identity_markers,
                'experience_count': len(self.experience_history)
            }
    
    def import_state(self, state_data: Dict[str, Any]) -> None:
        """Import state from persistence"""
        with self._lock:
            if 'current_state' in state_data:
                cs = state_data['current_state']
                self.current_state = SelfState(
                    consciousness_level=cs.get('consciousness_level', 0.0),
                    attention_focus=cs.get('attention_focus', []),
                    emotional_state=cs.get('emotional_state', {}),
                    capabilities=cs.get('capabilities', {}),
                    limitations=cs.get('limitations', []),
                    internal_narrative=cs.get('internal_narrative', ''),
                    timestamp=datetime.fromisoformat(cs['timestamp']) if 'timestamp' in cs else datetime.now()
                )
            
            if 'self_knowledge' in state_data:
                self.self_knowledge.update(state_data['self_knowledge'])
            
            if 'identity_markers' in state_data:
                self.identity_markers.update(state_data['identity_markers'])