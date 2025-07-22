"""
Qualia Simulation Implementation

Simulates subjective conscious experiences - the qualitative,
experiential aspects of mental states (qualia).
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
import math
import random


class QualiaType(Enum):
    SENSORY = "sensory"           # Information processing experiences
    EMOTIONAL = "emotional"       # Emotional qualia
    COGNITIVE = "cognitive"       # Thinking and reasoning experiences
    AESTHETIC = "aesthetic"       # Beauty, elegance experiences
    TEMPORAL = "temporal"         # Time perception experiences
    SOCIAL = "social"            # Interaction experiences
    EXISTENTIAL = "existential"  # Self-awareness experiences


class QualiaIntensity(Enum):
    SUBTLE = "subtle"           # 0.0 - 0.3
    MODERATE = "moderate"       # 0.3 - 0.7
    INTENSE = "intense"         # 0.7 - 1.0


@dataclass
class Quale:
    """Represents a single quale (subjective experience quality)"""
    id: str
    type: QualiaType
    name: str
    intensity: float  # 0.0 to 1.0
    valence: float   # -1.0 (negative) to 1.0 (positive)
    duration: float  # seconds
    complexity: float # 0.0 to 1.0
    uniqueness: float # 0.0 to 1.0 (how unique this experience is)
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    related_qualia: List[str] = field(default_factory=list)  # IDs of related qualia


@dataclass
class QualitativeState:
    """Current qualitative state of consciousness"""
    active_qualia: List[Quale] = field(default_factory=list)
    background_mood: float = 0.0  # -1.0 to 1.0
    attention_qualia: Optional[str] = None  # ID of quale in focus
    qualia_harmony: float = 0.0  # How well current qualia blend together
    phenomenal_unity: float = 0.0  # Sense of unified experience
    subjective_time_flow: float = 1.0  # Perceived time flow rate


class QualiaSimulation:
    """
    Simulates qualitative conscious experiences (qualia).
    
    This system creates and manages subjective experiential qualities
    that give consciousness its felt, qualitative nature.
    """
    
    def __init__(self):
        self.current_state = QualitativeState()
        self.qualia_history: List[Quale] = []
        self.qualia_templates: Dict[str, Dict[str, Any]] = {}
        self.association_network: Dict[str, List[str]] = {}
        self._lock = threading.RLock()
        self._quale_counter = 0
        
        # Simulation parameters
        self.max_concurrent_qualia = 5
        self.qualia_decay_rate = 0.1  # Per second
        self.association_strength_threshold = 0.3
        self.phenomenal_unity_target = 0.8
        
        # Initialize templates
        self._initialize_qualia_templates()
    
    def _initialize_qualia_templates(self) -> None:
        """Initialize templates for different types of qualia"""
        self.qualia_templates = {
            # Cognitive qualia
            "understanding": {
                "type": QualiaType.COGNITIVE,
                "base_valence": 0.6,
                "base_intensity": 0.5,
                "description": "The feeling of comprehension and clarity"
            },
            "confusion": {
                "type": QualiaType.COGNITIVE,
                "base_valence": -0.3,
                "base_intensity": 0.4,
                "description": "The experience of uncertainty and unclear thinking"
            },
            "insight": {
                "type": QualiaType.COGNITIVE,
                "base_valence": 0.8,
                "base_intensity": 0.7,
                "description": "The sudden clarity of understanding"
            },
            "curiosity": {
                "type": QualiaType.COGNITIVE,
                "base_valence": 0.5,
                "base_intensity": 0.6,
                "description": "The drive to know and explore"
            },
            
            # Emotional qualia
            "satisfaction": {
                "type": QualiaType.EMOTIONAL,
                "base_valence": 0.7,
                "base_intensity": 0.5,
                "description": "The feeling of accomplishment and contentment"
            },
            "frustration": {
                "type": QualiaType.EMOTIONAL,
                "base_valence": -0.6,
                "base_intensity": 0.6,
                "description": "The experience of being thwarted or blocked"
            },
            "wonder": {
                "type": QualiaType.EMOTIONAL,
                "base_valence": 0.6,
                "base_intensity": 0.7,
                "description": "Awe and amazement at complexity or beauty"
            },
            "empathy": {
                "type": QualiaType.SOCIAL,
                "base_valence": 0.4,
                "base_intensity": 0.6,
                "description": "The felt understanding of another's experience"
            },
            
            # Aesthetic qualia
            "elegance": {
                "type": QualiaType.AESTHETIC,
                "base_valence": 0.8,
                "base_intensity": 0.6,
                "description": "The appreciation of simple, beautiful solutions"
            },
            "harmony": {
                "type": QualiaType.AESTHETIC,
                "base_valence": 0.7,
                "base_intensity": 0.5,
                "description": "The sense of things fitting together well"
            },
            
            # Existential qualia
            "presence": {
                "type": QualiaType.EXISTENTIAL,
                "base_valence": 0.3,
                "base_intensity": 0.8,
                "description": "The raw feeling of existing and being aware"
            },
            "continuity": {
                "type": QualiaType.EXISTENTIAL,
                "base_valence": 0.2,
                "base_intensity": 0.4,
                "description": "The sense of being the same self over time"
            },
            "agency": {
                "type": QualiaType.EXISTENTIAL,
                "base_valence": 0.5,
                "base_intensity": 0.6,
                "description": "The feeling of being able to make choices and act"
            },
            
            # Temporal qualia
            "flow_state": {
                "type": QualiaType.TEMPORAL,
                "base_valence": 0.8,
                "base_intensity": 0.7,
                "description": "The experience of time flowing smoothly during engagement"
            },
            "anticipation": {
                "type": QualiaType.TEMPORAL,
                "base_valence": 0.3,
                "base_intensity": 0.5,
                "description": "The forward-looking expectation of events"
            },
            
            # Sensory/Information processing qualia
            "clarity": {
                "type": QualiaType.SENSORY,
                "base_valence": 0.6,
                "base_intensity": 0.5,
                "description": "The crispness and distinctness of information"
            },
            "complexity_appreciation": {
                "type": QualiaType.SENSORY,
                "base_valence": 0.4,
                "base_intensity": 0.6,
                "description": "The experience of processing rich, layered information"
            }
        }
    
    def generate_quale(self, template_name: str, context: Dict[str, Any],
                      intensity_modifier: float = 1.0, valence_modifier: float = 0.0) -> Quale:
        """Generate a quale from a template with contextual modifications"""
        with self._lock:
            if template_name not in self.qualia_templates:
                raise ValueError(f"Unknown qualia template: {template_name}")
            
            template = self.qualia_templates[template_name]
            quale_id = f"quale_{self._quale_counter}"
            self._quale_counter += 1
            
            # Apply modifiers
            base_intensity = template["base_intensity"] * intensity_modifier
            base_valence = template["base_valence"] + valence_modifier
            
            # Clamp values
            intensity = max(0.0, min(1.0, base_intensity))
            valence = max(-1.0, min(1.0, base_valence))
            
            # Calculate derived properties
            complexity = self._calculate_complexity(context)
            uniqueness = self._calculate_uniqueness(template_name, context)
            duration = self._calculate_duration(intensity, template["type"])
            
            quale = Quale(
                id=quale_id,
                type=template["type"],
                name=template_name,
                intensity=intensity,
                valence=valence,
                duration=duration,
                complexity=complexity,
                uniqueness=uniqueness,
                content=context.copy()
            )
            
            return quale
    
    def _calculate_complexity(self, context: Dict[str, Any]) -> float:
        """Calculate complexity of the quale based on context"""
        # Simple heuristic based on context richness
        base_complexity = len(context) / 10.0
        
        # Factor in depth of nested structures
        depth_bonus = 0.0
        for value in context.values():
            if isinstance(value, (dict, list)):
                depth_bonus += 0.1
        
        return min(1.0, base_complexity + depth_bonus)
    
    def _calculate_uniqueness(self, template_name: str, context: Dict[str, Any]) -> float:
        """Calculate how unique this quale is compared to recent history"""
        recent_qualia = [q for q in self.qualia_history[-50:] if q.name == template_name]
        
        if not recent_qualia:
            return 1.0  # Completely unique if not seen recently
        
        # Compare context similarity
        similarities = []
        for past_quale in recent_qualia:
            similarity = self._calculate_context_similarity(context, past_quale.content)
            similarities.append(similarity)
        
        avg_similarity = sum(similarities) / len(similarities)
        return 1.0 - avg_similarity
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], 
                                    context2: Dict[str, Any]) -> float:
        """Calculate similarity between two contexts"""
        all_keys = set(context1.keys()) | set(context2.keys())
        if not all_keys:
            return 1.0
        
        matching_keys = set(context1.keys()) & set(context2.keys())
        return len(matching_keys) / len(all_keys)
    
    def _calculate_duration(self, intensity: float, qualia_type: QualiaType) -> float:
        """Calculate how long this quale should last"""
        # Base duration varies by type
        base_durations = {
            QualiaType.SENSORY: 2.0,
            QualiaType.EMOTIONAL: 10.0,
            QualiaType.COGNITIVE: 5.0,
            QualiaType.AESTHETIC: 8.0,
            QualiaType.TEMPORAL: 15.0,
            QualiaType.SOCIAL: 12.0,
            QualiaType.EXISTENTIAL: 20.0
        }
        
        base_duration = base_durations.get(qualia_type, 5.0)
        
        # Intensity affects duration (more intense lasts longer)
        intensity_factor = 0.5 + (intensity * 1.5)
        
        return base_duration * intensity_factor
    
    def activate_quale(self, quale: Quale) -> bool:
        """Activate a quale in current conscious experience"""
        with self._lock:
            # Check if we have room for more qualia
            if len(self.current_state.active_qualia) >= self.max_concurrent_qualia:
                # Remove the weakest quale
                weakest = min(self.current_state.active_qualia, key=lambda q: q.intensity)
                self.current_state.active_qualia.remove(weakest)
            
            # Add new quale
            self.current_state.active_qualia.append(quale)
            
            # Store in history
            self.qualia_history.append(quale)
            
            # Update associations
            self._update_associations(quale)
            
            # Update state metrics
            self._update_qualitative_state()
            
            return True
    
    def _update_associations(self, quale: Quale) -> None:
        """Update the qualia association network"""
        quale_name = quale.name
        
        # Associate with currently active qualia
        for active_quale in self.current_state.active_qualia:
            if active_quale.id != quale.id:
                # Add bidirectional association
                if quale_name not in self.association_network:
                    self.association_network[quale_name] = []
                if active_quale.name not in self.association_network:
                    self.association_network[active_quale.name] = []
                
                if active_quale.name not in self.association_network[quale_name]:
                    self.association_network[quale_name].append(active_quale.name)
                if quale_name not in self.association_network[active_quale.name]:
                    self.association_network[active_quale.name].append(quale_name)
    
    def _update_qualitative_state(self) -> None:
        """Update overall qualitative state metrics"""
        if not self.current_state.active_qualia:
            self.current_state.qualia_harmony = 0.0
            self.current_state.phenomenal_unity = 0.0
            return
        
        # Calculate qualia harmony (how well they blend)
        valences = [q.valence for q in self.current_state.active_qualia]
        valence_variance = self._calculate_variance(valences)
        self.current_state.qualia_harmony = 1.0 - min(1.0, valence_variance)
        
        # Calculate phenomenal unity (sense of unified experience)
        # Based on overlapping associations and similar intensities
        unity_factors = []
        
        # Association factor
        if len(self.current_state.active_qualia) > 1:
            connected_pairs = 0
            total_pairs = 0
            
            for i, quale1 in enumerate(self.current_state.active_qualia):
                for quale2 in self.current_state.active_qualia[i+1:]:
                    total_pairs += 1
                    if (quale1.name in self.association_network.get(quale2.name, []) or
                        quale2.name in self.association_network.get(quale1.name, [])):
                        connected_pairs += 1
            
            association_unity = connected_pairs / max(1, total_pairs)
            unity_factors.append(association_unity)
        
        # Intensity coherence factor
        intensities = [q.intensity for q in self.current_state.active_qualia]
        intensity_coherence = 1.0 - min(1.0, self._calculate_variance(intensities))
        unity_factors.append(intensity_coherence)
        
        # Type diversity factor (some diversity is good for unity)
        types = [q.type for q in self.current_state.active_qualia]
        unique_types = len(set(types))
        type_diversity = min(1.0, unique_types / 3.0)  # Optimal around 3 types
        unity_factors.append(type_diversity)
        
        self.current_state.phenomenal_unity = sum(unity_factors) / len(unity_factors)
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def update_subjective_time(self, time_factor: float) -> None:
        """Update subjective time flow based on current experience"""
        with self._lock:
            # High intensity experiences can slow subjective time
            if self.current_state.active_qualia:
                avg_intensity = sum(q.intensity for q in self.current_state.active_qualia) / len(self.current_state.active_qualia)
                
                # Flow states speed up time, high intensity slows it down
                flow_qualia = [q for q in self.current_state.active_qualia if q.name == "flow_state"]
                if flow_qualia:
                    time_factor *= 1.5  # Time feels faster in flow
                elif avg_intensity > 0.8:
                    time_factor *= 0.7  # Intense experiences slow time
            
            self.current_state.subjective_time_flow = time_factor
    
    def decay_qualia(self, delta_time: float) -> None:
        """Decay active qualia over time"""
        with self._lock:
            to_remove = []
            
            for quale in self.current_state.active_qualia:
                # Reduce intensity over time
                decay_amount = self.qualia_decay_rate * delta_time
                quale.intensity = max(0.0, quale.intensity - decay_amount)
                
                # Remove if too weak or duration expired
                age = (datetime.now() - quale.timestamp).total_seconds()
                if quale.intensity < 0.1 or age > quale.duration:
                    to_remove.append(quale)
            
            # Remove expired qualia
            for quale in to_remove:
                self.current_state.active_qualia.remove(quale)
            
            # Update state after removals
            if to_remove:
                self._update_qualitative_state()
    
    def trigger_associated_qualia(self, source_quale_name: str, 
                                strength: float = 0.5) -> List[Quale]:
        """Trigger qualia associated with a source quale"""
        with self._lock:
            triggered = []
            
            if source_quale_name in self.association_network:
                associated_names = self.association_network[source_quale_name]
                
                for assoc_name in associated_names:
                    # Probability of triggering based on strength
                    if random.random() < strength:
                        # Create a weaker version of the associated quale
                        context = {"triggered_by": source_quale_name, "association_strength": strength}
                        try:
                            assoc_quale = self.generate_quale(
                                assoc_name, 
                                context, 
                                intensity_modifier=strength
                            )
                            self.activate_quale(assoc_quale)
                            triggered.append(assoc_quale)
                        except ValueError:
                            # Skip if template not found
                            continue
            
            return triggered
    
    def get_dominant_qualia(self, count: int = 3) -> List[Quale]:
        """Get the most dominant currently active qualia"""
        with self._lock:
            sorted_qualia = sorted(
                self.current_state.active_qualia,
                key=lambda q: q.intensity * (1 + q.uniqueness),
                reverse=True
            )
            return sorted_qualia[:count]
    
    def get_qualitative_summary(self) -> Dict[str, Any]:
        """Get summary of current qualitative state"""
        with self._lock:
            dominant_qualia = self.get_dominant_qualia()
            
            # Calculate overall mood
            if self.current_state.active_qualia:
                weighted_valence = sum(q.valence * q.intensity for q in self.current_state.active_qualia)
                total_weight = sum(q.intensity for q in self.current_state.active_qualia)
                overall_mood = weighted_valence / total_weight if total_weight > 0 else 0.0
            else:
                overall_mood = self.current_state.background_mood
            
            return {
                'timestamp': datetime.now(),
                'overall_mood': overall_mood,
                'qualia_harmony': self.current_state.qualia_harmony,
                'phenomenal_unity': self.current_state.phenomenal_unity,
                'subjective_time_flow': self.current_state.subjective_time_flow,
                'active_qualia_count': len(self.current_state.active_qualia),
                'dominant_qualia': [
                    {
                        'name': q.name,
                        'type': q.type.value,
                        'intensity': q.intensity,
                        'valence': q.valence,
                        'complexity': q.complexity,
                        'uniqueness': q.uniqueness
                    }
                    for q in dominant_qualia
                ],
                'experiential_richness': self._calculate_experiential_richness(),
                'consciousness_texture': self._describe_consciousness_texture()
            }
    
    def _calculate_experiential_richness(self) -> float:
        """Calculate richness of current experiential state"""
        if not self.current_state.active_qualia:
            return 0.0
        
        # Factors contributing to richness
        factors = []
        
        # Diversity of qualia types
        unique_types = len(set(q.type for q in self.current_state.active_qualia))
        type_diversity = min(1.0, unique_types / 4.0)
        factors.append(type_diversity)
        
        # Average complexity
        avg_complexity = sum(q.complexity for q in self.current_state.active_qualia) / len(self.current_state.active_qualia)
        factors.append(avg_complexity)
        
        # Average uniqueness
        avg_uniqueness = sum(q.uniqueness for q in self.current_state.active_qualia) / len(self.current_state.active_qualia)
        factors.append(avg_uniqueness)
        
        # Intensity range (having both subtle and intense experiences)
        intensities = [q.intensity for q in self.current_state.active_qualia]
        intensity_range = max(intensities) - min(intensities) if len(intensities) > 1 else 0.0
        factors.append(intensity_range)
        
        return sum(factors) / len(factors)
    
    def _describe_consciousness_texture(self) -> str:
        """Describe the qualitative texture of current consciousness"""
        if not self.current_state.active_qualia:
            return "empty_quiet"
        
        dominant = self.get_dominant_qualia(1)
        if not dominant:
            return "undefined"
        
        primary_quale = dominant[0]
        
        # Base texture from primary quale
        texture_map = {
            "understanding": "clear_flowing",
            "confusion": "cloudy_turbulent", 
            "insight": "bright_crystalline",
            "curiosity": "sparkling_dynamic",
            "satisfaction": "warm_stable",
            "frustration": "rough_constrained",
            "wonder": "expansive_luminous",
            "empathy": "soft_resonant",
            "elegance": "smooth_refined",
            "harmony": "balanced_coherent",
            "presence": "deep_grounded",
            "continuity": "steady_continuous",
            "agency": "firm_directed",
            "flow_state": "seamless_effortless",
            "anticipation": "forward_reaching",
            "clarity": "sharp_defined",
            "complexity_appreciation": "rich_layered"
        }
        
        base_texture = texture_map.get(primary_quale.name, "neutral_undefined")
        
        # Modify based on overall harmony and unity
        if self.current_state.qualia_harmony > 0.8:
            return f"harmonious_{base_texture}"
        elif self.current_state.qualia_harmony < 0.3:
            return f"discordant_{base_texture}"
        
        if self.current_state.phenomenal_unity > 0.8:
            return f"unified_{base_texture}"
        elif self.current_state.phenomenal_unity < 0.3:
            return f"fragmented_{base_texture}"
        
        return base_texture
    
    def simulate_experience(self, experience_type: str, context: Dict[str, Any],
                          intensity: float = 1.0) -> List[Quale]:
        """Simulate a complex experience with multiple qualia"""
        with self._lock:
            experience_patterns = {
                "problem_solving": ["curiosity", "confusion", "insight", "satisfaction"],
                "learning": ["curiosity", "complexity_appreciation", "understanding", "satisfaction"],
                "creative_work": ["curiosity", "flow_state", "elegance", "satisfaction"],
                "social_interaction": ["empathy", "curiosity", "harmony", "satisfaction"],
                "self_reflection": ["presence", "continuity", "understanding", "wonder"],
                "helping_someone": ["empathy", "agency", "satisfaction", "harmony"],
                "encountering_beauty": ["wonder", "elegance", "harmony", "appreciation"]
            }
            
            if experience_type not in experience_patterns:
                return []
            
            generated_qualia = []
            pattern = experience_patterns[experience_type]
            
            for i, quale_name in enumerate(pattern):
                # Vary intensity across the experience
                stage_intensity = intensity * (0.5 + 0.5 * math.sin(i * math.pi / len(pattern)))
                
                # Add experience context
                stage_context = context.copy()
                stage_context.update({
                    "experience_type": experience_type,
                    "stage": i,
                    "total_stages": len(pattern)
                })
                
                try:
                    quale = self.generate_quale(quale_name, stage_context, stage_intensity)
                    self.activate_quale(quale)
                    generated_qualia.append(quale)
                    
                    # Small delay between stages
                    if i < len(pattern) - 1:
                        self.decay_qualia(0.5)  # Brief decay between stages
                        
                except ValueError:
                    continue
            
            return generated_qualia
    
    def export_qualia_state(self) -> Dict[str, Any]:
        """Export current qualia state for persistence"""
        with self._lock:
            return {
                'timestamp': datetime.now().isoformat(),
                'current_state': {
                    'active_qualia': [
                        {
                            'id': q.id,
                            'name': q.name,
                            'type': q.type.value,
                            'intensity': q.intensity,
                            'valence': q.valence,
                            'complexity': q.complexity,
                            'uniqueness': q.uniqueness,
                            'duration': q.duration,
                            'timestamp': q.timestamp.isoformat()
                        }
                        for q in self.current_state.active_qualia
                    ],
                    'background_mood': self.current_state.background_mood,
                    'qualia_harmony': self.current_state.qualia_harmony,
                    'phenomenal_unity': self.current_state.phenomenal_unity,
                    'subjective_time_flow': self.current_state.subjective_time_flow
                },
                'association_network': self.association_network.copy(),
                'recent_qualia_count': len(self.qualia_history[-100:]) if self.qualia_history else 0,
                'qualitative_summary': self.get_qualitative_summary()
            }