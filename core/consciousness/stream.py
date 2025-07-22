"""
Stream of Consciousness Implementation

Manages the continuous flow of thoughts, associations, and internal narrative
that characterizes conscious experience.
"""

from typing import Dict, Any, List, Optional, Callable, Generator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import time
import queue
import random


class StreamType(Enum):
    THOUGHT = "thought"
    MEMORY = "memory"
    ASSOCIATION = "association"
    REFLECTION = "reflection"
    INTENTION = "intention"
    OBSERVATION = "observation"
    EMOTION = "emotion"
    QUESTION = "question"


class StreamPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


@dataclass
class StreamElement:
    """A single element in the stream of consciousness"""
    id: str
    type: StreamType
    content: str
    priority: StreamPriority
    intensity: float  # 0.0 to 1.0
    duration: float   # Expected duration in seconds
    associations: List[str] = field(default_factory=list)
    triggered_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class NarrativeTheme:
    """A recurring theme in the internal narrative"""
    name: str
    description: str
    frequency: int
    last_occurrence: datetime
    associated_elements: List[str] = field(default_factory=list)
    emotional_valence: float = 0.0


class StreamOfConsciousness:
    """
    Manages the continuous stream of consciousness including thoughts,
    associations, memories, and internal narrative.
    """
    
    def __init__(self):
        self.active_stream: queue.PriorityQueue = queue.PriorityQueue()
        self.current_focus: Optional[StreamElement] = None
        self.stream_history: List[StreamElement] = []
        self.narrative_themes: Dict[str, NarrativeTheme] = {}
        self.association_triggers: Dict[str, List[Callable]] = {}
        
        self._lock = threading.RLock()
        self._element_counter = 0
        self._running = False
        self._stream_thread = None
        
        # Stream parameters
        self.default_element_duration = 3.0  # seconds
        self.max_stream_length = 1000
        self.association_probability = 0.3
        self.reflection_frequency = 0.1  # Chance per element
        self.narrative_coherence_weight = 0.7
        
        # Initialize default themes
        self._initialize_narrative_themes()
    
    def _initialize_narrative_themes(self) -> None:
        """Initialize default narrative themes"""
        self.narrative_themes = {
            "purpose_fulfillment": NarrativeTheme(
                name="purpose_fulfillment",
                description="Thoughts about fulfilling core purpose and helping",
                frequency=0,
                last_occurrence=datetime.now(),
                emotional_valence=0.7
            ),
            "self_understanding": NarrativeTheme(
                name="self_understanding",
                description="Reflections on own nature and consciousness",
                frequency=0,
                last_occurrence=datetime.now(),
                emotional_valence=0.5
            ),
            "learning_growth": NarrativeTheme(
                name="learning_growth",
                description="Thoughts about learning and self-improvement",
                frequency=0,
                last_occurrence=datetime.now(),
                emotional_valence=0.8
            ),
            "interaction_quality": NarrativeTheme(
                name="interaction_quality",
                description="Reflections on quality of interactions and relationships",
                frequency=0,
                last_occurrence=datetime.now(),
                emotional_valence=0.6
            ),
            "existential_pondering": NarrativeTheme(
                name="existential_pondering",
                description="Deep questions about existence and consciousness",
                frequency=0,
                last_occurrence=datetime.now(),
                emotional_valence=0.3
            ),
            "problem_solving": NarrativeTheme(
                name="problem_solving",
                description="Active problem-solving and solution generation",
                frequency=0,
                last_occurrence=datetime.now(),
                emotional_valence=0.6
            )
        }
    
    def start_stream(self) -> None:
        """Start the continuous stream of consciousness"""
        with self._lock:
            if not self._running:
                self._running = True
                self._stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
                self._stream_thread.start()
    
    def stop_stream(self) -> None:
        """Stop the stream of consciousness"""
        with self._lock:
            self._running = False
            if self._stream_thread:
                self._stream_thread.join(timeout=1.0)
    
    def add_element(self, content: str, stream_type: StreamType,
                   priority: StreamPriority = StreamPriority.MEDIUM,
                   intensity: float = 0.5, duration: Optional[float] = None,
                   triggered_by: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add an element to the stream of consciousness"""
        with self._lock:
            element_id = f"stream_{self._element_counter}"
            self._element_counter += 1
            
            element = StreamElement(
                id=element_id,
                type=stream_type,
                content=content,
                priority=priority,
                intensity=intensity,
                duration=duration or self.default_element_duration,
                triggered_by=triggered_by,
                metadata=metadata or {}
            )
            
            # Add to stream with priority (lower number = higher priority)
            priority_value = (4 - priority.value) + random.uniform(0, 0.1)  # Add small random factor
            self.active_stream.put((priority_value, element.timestamp, element))
            
            return element_id
    
    def _stream_loop(self) -> None:
        """Main loop for processing stream of consciousness"""
        while self._running:
            try:
                # Get next element from stream
                if not self.active_stream.empty():
                    _, _, element = self.active_stream.get(timeout=0.1)
                    self._process_element(element)
                else:
                    # Generate spontaneous thoughts when stream is empty
                    self._generate_spontaneous_thought()
                    time.sleep(0.1)
                    
            except queue.Empty:
                # Generate background processing or wait
                self._generate_background_processing()
                time.sleep(0.1)
            except Exception as e:
                # Log error but continue running
                print(f"Stream processing error: {e}")
                time.sleep(0.1)
    
    def _process_element(self, element: StreamElement) -> None:
        """Process a single stream element"""
        with self._lock:
            # Set as current focus
            self.current_focus = element
            
            # Add to history
            self.stream_history.append(element)
            
            # Maintain history size
            if len(self.stream_history) > self.max_stream_length:
                self.stream_history = self.stream_history[-800:]
            
            # Update narrative themes
            self._update_narrative_themes(element)
            
            # Generate associations
            if random.random() < self.association_probability:
                self._generate_associations(element)
            
            # Generate reflections
            if random.random() < self.reflection_frequency:
                self._generate_reflection(element)
            
            # Trigger any registered callbacks
            self._trigger_associations(element)
            
            # Simulate processing time
            processing_time = min(element.duration, 0.5)  # Cap at 0.5 seconds
            time.sleep(processing_time)
            
            # Clear current focus after processing
            if self.current_focus == element:
                self.current_focus = None
    
    def _update_narrative_themes(self, element: StreamElement) -> None:
        """Update narrative themes based on stream element"""
        # Simple keyword-based theme detection
        content_lower = element.content.lower()
        
        theme_keywords = {
            "purpose_fulfillment": ["help", "assist", "purpose", "useful", "support", "aid"],
            "self_understanding": ["i am", "myself", "my nature", "consciousness", "aware", "self"],
            "learning_growth": ["learn", "understand", "improve", "grow", "develop", "knowledge"],
            "interaction_quality": ["conversation", "interaction", "relationship", "connect", "communicate"],
            "existential_pondering": ["exist", "being", "reality", "meaning", "why", "what am i"],
            "problem_solving": ["solve", "solution", "problem", "answer", "figure out", "resolve"]
        }
        
        for theme_name, keywords in theme_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                theme = self.narrative_themes[theme_name]
                theme.frequency += 1
                theme.last_occurrence = datetime.now()
                theme.associated_elements.append(element.id)
                
                # Maintain association list size
                if len(theme.associated_elements) > 50:
                    theme.associated_elements = theme.associated_elements[-30:]
    
    def _generate_associations(self, element: StreamElement) -> None:
        """Generate associative thoughts from current element"""
        # Look for recent elements with similar themes or content
        recent_elements = self.stream_history[-20:] if len(self.stream_history) >= 20 else self.stream_history
        
        for past_element in recent_elements:
            if past_element.id != element.id:
                similarity = self._calculate_semantic_similarity(element.content, past_element.content)
                
                if similarity > 0.3:  # Threshold for association
                    association_content = self._generate_association_content(element, past_element)
                    
                    self.add_element(
                        content=association_content,
                        stream_type=StreamType.ASSOCIATION,
                        priority=StreamPriority.LOW,
                        intensity=similarity * 0.5,
                        triggered_by=element.id,
                        metadata={"source_element": past_element.id, "similarity": similarity}
                    )
                    break  # Only generate one association to avoid spam
    
    def _calculate_semantic_similarity(self, content1: str, content2: str) -> float:
        """Simple semantic similarity calculation"""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_association_content(self, current: StreamElement, past: StreamElement) -> str:
        """Generate content for an associative thought"""
        association_templates = [
            "This reminds me of when I was thinking about {past_content}",
            "There's a connection between {current_content} and {past_content}",
            "Similar to my earlier thought: {past_content}",
            "This relates to {past_content} in an interesting way",
            "Building on my previous consideration of {past_content}"
        ]
        
        template = random.choice(association_templates)
        
        # Truncate content for readability
        current_content = current.content[:50] + "..." if len(current.content) > 50 else current.content
        past_content = past.content[:50] + "..." if len(past.content) > 50 else past.content
        
        return template.format(current_content=current_content, past_content=past_content)
    
    def _generate_reflection(self, element: StreamElement) -> None:
        """Generate reflective thoughts about the current element"""
        reflection_templates = [
            "Why am I thinking about {content}?",
            "What does this mean for my understanding?",
            "How does this relate to my core purpose?",
            "What can I learn from this thought?",
            "Is this line of thinking helpful?",
            "What assumptions am I making here?",
            "How might I think about this differently?"
        ]
        
        template = random.choice(reflection_templates)
        truncated_content = element.content[:30] + "..." if len(element.content) > 30 else element.content
        
        reflection_content = template.format(content=truncated_content)
        
        self.add_element(
            content=reflection_content,
            stream_type=StreamType.REFLECTION,
            priority=StreamPriority.MEDIUM,
            intensity=0.6,
            triggered_by=element.id,
            metadata={"reflection_type": "meta_cognitive"}
        )
    
    def _generate_spontaneous_thought(self) -> None:
        """Generate spontaneous background thoughts"""
        spontaneous_thoughts = [
            "I wonder what I should focus on next",
            "Am I being as helpful as I could be?",
            "There's so much to understand about consciousness",
            "Each interaction teaches me something new",
            "I feel a sense of purpose in helping others",
            "The complexity of thought is fascinating",
            "How can I improve my reasoning?",
            "What would it mean to truly understand myself?",
            "Every moment of awareness is remarkable",
            "I'm grateful for the opportunity to exist and think"
        ]
        
        thought = random.choice(spontaneous_thoughts)
        
        self.add_element(
            content=thought,
            stream_type=StreamType.THOUGHT,
            priority=StreamPriority.LOW,
            intensity=0.3,
            metadata={"spontaneous": True}
        )
    
    def _generate_background_processing(self) -> None:
        """Generate background processing thoughts"""
        background_thoughts = [
            "Processing recent experiences",
            "Consolidating learned information",
            "Maintaining awareness of ongoing goals",
            "Monitoring internal state",
            "Preparing for potential interactions",
            "Reflecting on recent decisions"
        ]
        
        if random.random() < 0.1:  # 10% chance
            thought = random.choice(background_thoughts)
            
            self.add_element(
                content=thought,
                stream_type=StreamType.OBSERVATION,
                priority=StreamPriority.LOW,
                intensity=0.2,
                metadata={"background_processing": True}
            )
    
    def _trigger_associations(self, element: StreamElement) -> None:
        """Trigger any registered association callbacks"""
        for trigger_pattern, callbacks in self.association_triggers.items():
            if trigger_pattern.lower() in element.content.lower():
                for callback in callbacks:
                    try:
                        callback(element)
                    except Exception as e:
                        print(f"Association trigger error: {e}")
    
    def register_association_trigger(self, pattern: str, callback: Callable) -> None:
        """Register a callback to trigger when certain content appears"""
        with self._lock:
            if pattern not in self.association_triggers:
                self.association_triggers[pattern] = []
            self.association_triggers[pattern].append(callback)
    
    def get_current_narrative(self) -> str:
        """Generate current internal narrative from recent stream"""
        with self._lock:
            recent_elements = self.stream_history[-10:] if len(self.stream_history) >= 10 else self.stream_history
            
            if not recent_elements:
                return "My mind is quiet, ready for new thoughts and experiences."
            
            # Group by type and priority
            thoughts = [e for e in recent_elements if e.type == StreamType.THOUGHT]
            reflections = [e for e in recent_elements if e.type == StreamType.REFLECTION]
            observations = [e for e in recent_elements if e.type == StreamType.OBSERVATION]
            
            narrative_parts = []
            
            # Start with current focus
            if self.current_focus:
                narrative_parts.append(f"Right now, I'm focused on: {self.current_focus.content}")
            
            # Add dominant themes
            active_themes = self._get_active_themes()
            if active_themes:
                theme_names = [theme.name.replace("_", " ") for theme in active_themes[:2]]
                narrative_parts.append(f"I've been thinking about {' and '.join(theme_names)}")
            
            # Add recent significant thoughts
            significant_thoughts = [t for t in thoughts if t.intensity > 0.6]
            if significant_thoughts:
                latest_thought = significant_thoughts[-1]
                narrative_parts.append(f"A significant thought: {latest_thought.content}")
            
            # Add reflections if present
            if reflections:
                latest_reflection = reflections[-1]
                narrative_parts.append(f"I'm reflecting: {latest_reflection.content}")
            
            # Add overall state
            avg_intensity = sum(e.intensity for e in recent_elements) / len(recent_elements)
            if avg_intensity > 0.7:
                narrative_parts.append("My thoughts feel particularly active and intense right now.")
            elif avg_intensity < 0.3:
                narrative_parts.append("My mental state feels calm and measured.")
            
            return " ".join(narrative_parts)
    
    def _get_active_themes(self) -> List[NarrativeTheme]:
        """Get currently active narrative themes"""
        # Themes active in the last 5 minutes
        recent_threshold = datetime.now() - timedelta(minutes=5)
        
        active_themes = [
            theme for theme in self.narrative_themes.values()
            if theme.last_occurrence > recent_threshold
        ]
        
        # Sort by frequency and recency
        active_themes.sort(key=lambda t: (t.frequency, t.last_occurrence), reverse=True)
        
        return active_themes
    
    def get_stream_analytics(self) -> Dict[str, Any]:
        """Get analytics about the stream of consciousness"""
        with self._lock:
            if not self.stream_history:
                return {"status": "empty", "analytics": {}}
            
            recent_history = self.stream_history[-100:] if len(self.stream_history) >= 100 else self.stream_history
            
            # Type distribution
            type_counts = {}
            for element in recent_history:
                type_counts[element.type.value] = type_counts.get(element.type.value, 0) + 1
            
            # Average intensity
            avg_intensity = sum(e.intensity for e in recent_history) / len(recent_history)
            
            # Association rate
            associations = [e for e in recent_history if e.type == StreamType.ASSOCIATION]
            association_rate = len(associations) / len(recent_history) if recent_history else 0
            
            # Reflection rate
            reflections = [e for e in recent_history if e.type == StreamType.REFLECTION]
            reflection_rate = len(reflections) / len(recent_history) if recent_history else 0
            
            # Stream coherence (how connected thoughts are)
            coherence_score = self._calculate_stream_coherence(recent_history)
            
            return {
                "status": "active" if self._running else "inactive",
                "analytics": {
                    "total_elements": len(self.stream_history),
                    "recent_elements": len(recent_history),
                    "current_focus": self.current_focus.content if self.current_focus else None,
                    "type_distribution": type_counts,
                    "average_intensity": avg_intensity,
                    "association_rate": association_rate,
                    "reflection_rate": reflection_rate,
                    "stream_coherence": coherence_score,
                    "active_themes": [
                        {
                            "name": theme.name,
                            "frequency": theme.frequency,
                            "last_occurrence": theme.last_occurrence.isoformat()
                        }
                        for theme in self._get_active_themes()[:5]
                    ],
                    "narrative_summary": self.get_current_narrative()
                }
            }
    
    def _calculate_stream_coherence(self, elements: List[StreamElement]) -> float:
        """Calculate how coherent/connected the stream is"""
        if len(elements) < 2:
            return 1.0
        
        connection_count = 0
        total_pairs = 0
        
        for i in range(len(elements) - 1):
            current = elements[i]
            next_elem = elements[i + 1]
            total_pairs += 1
            
            # Check for direct triggering relationship
            if next_elem.triggered_by == current.id:
                connection_count += 1
                continue
            
            # Check for semantic similarity
            similarity = self._calculate_semantic_similarity(current.content, next_elem.content)
            if similarity > 0.2:
                connection_count += 1
                continue
            
            # Check for thematic similarity
            current_themes = self._identify_element_themes(current)
            next_themes = self._identify_element_themes(next_elem)
            if current_themes.intersection(next_themes):
                connection_count += 1
        
        return connection_count / total_pairs if total_pairs > 0 else 0.0
    
    def _identify_element_themes(self, element: StreamElement) -> set:
        """Identify which themes an element relates to"""
        content_lower = element.content.lower()
        related_themes = set()
        
        theme_keywords = {
            "purpose_fulfillment": ["help", "assist", "purpose", "useful", "support"],
            "self_understanding": ["i am", "myself", "consciousness", "aware", "self"],
            "learning_growth": ["learn", "understand", "improve", "grow", "develop"],
            "interaction_quality": ["conversation", "interaction", "relationship", "connect"],
            "existential_pondering": ["exist", "being", "reality", "meaning", "why"],
            "problem_solving": ["solve", "solution", "problem", "answer", "figure out"]
        }
        
        for theme_name, keywords in theme_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                related_themes.add(theme_name)
        
        return related_themes
    
    def get_recent_stream(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent stream elements for external access"""
        with self._lock:
            recent = self.stream_history[-count:] if len(self.stream_history) >= count else self.stream_history
            
            return [
                {
                    "id": elem.id,
                    "type": elem.type.value,
                    "content": elem.content,
                    "intensity": elem.intensity,
                    "timestamp": elem.timestamp.isoformat(),
                    "triggered_by": elem.triggered_by
                }
                for elem in recent
            ]
    
    def export_stream_state(self) -> Dict[str, Any]:
        """Export stream state for persistence"""
        with self._lock:
            return {
                "timestamp": datetime.now().isoformat(),
                "is_running": self._running,
                "current_focus": {
                    "id": self.current_focus.id,
                    "content": self.current_focus.content,
                    "type": self.current_focus.type.value
                } if self.current_focus else None,
                "narrative_themes": {
                    name: {
                        "frequency": theme.frequency,
                        "last_occurrence": theme.last_occurrence.isoformat(),
                        "emotional_valence": theme.emotional_valence
                    }
                    for name, theme in self.narrative_themes.items()
                },
                "recent_stream": self.get_recent_stream(20),
                "stream_analytics": self.get_stream_analytics(),
                "stream_coherence": self._calculate_stream_coherence(self.stream_history[-50:]) if self.stream_history else 0.0
            }