"""
Metacognitive Monitor Implementation

Monitors and analyzes the AI's own cognitive processes,
providing awareness of thinking patterns, biases, and reasoning quality.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import time


class ThoughtType(Enum):
    REASONING = "reasoning"
    PLANNING = "planning"
    EVALUATION = "evaluation"
    REFLECTION = "reflection"
    DECISION = "decision"
    MEMORY_RETRIEVAL = "memory_retrieval"
    CREATIVE = "creative"
    PROBLEM_SOLVING = "problem_solving"


class CognitionQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    ADEQUATE = "adequate"
    POOR = "poor"
    UNCERTAIN = "uncertain"


@dataclass
class ThoughtProcess:
    """Represents a single thought process instance"""
    id: str
    type: ThoughtType
    content: str
    start_time: datetime
    end_time: Optional[datetime] = None
    quality: Optional[CognitionQuality] = None
    confidence: float = 0.0
    biases_detected: List[str] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    interruptions: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitiveBias:
    """Detected cognitive bias"""
    name: str
    description: str
    confidence: float
    evidence: List[str]
    detected_at: datetime = field(default_factory=datetime.now)


class MetacognitiveMonitor:
    """
    Monitors and analyzes the AI's own cognitive processes.
    
    Provides self-awareness about thinking patterns, quality of reasoning,
    cognitive biases, and overall metacognitive insights.
    """
    
    def __init__(self):
        self.active_processes: Dict[str, ThoughtProcess] = {}
        self.completed_processes: List[ThoughtProcess] = []
        self.detected_biases: List[CognitiveBias] = []
        self.cognitive_metrics: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._process_counter = 0
        
        # Monitoring configuration
        self.max_history_size = 1000
        self.bias_detection_enabled = True
        self.quality_threshold = 0.7
        
        # Initialize metrics
        self._initialize_metrics()
    
    def _initialize_metrics(self) -> None:
        """Initialize cognitive monitoring metrics"""
        self.cognitive_metrics = {
            'total_thoughts': 0,
            'avg_thought_duration': 0.0,
            'quality_distribution': {q.value: 0 for q in CognitionQuality},
            'type_distribution': {t.value: 0 for t in ThoughtType},
            'bias_frequency': {},
            'interruption_rate': 0.0,
            'cognitive_load': 0.0,
            'reasoning_depth': 0.0,
            'confidence_calibration': 0.0,
            'metacognitive_awareness': 0.8  # How well we understand our own thinking
        }
    
    def start_thought_process(self, content: str, thought_type: ThoughtType, 
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """Begin monitoring a thought process"""
        with self._lock:
            process_id = f"thought_{self._process_counter}"
            self._process_counter += 1
            
            process = ThoughtProcess(
                id=process_id,
                type=thought_type,
                content=content,
                start_time=datetime.now(),
                metadata=metadata or {}
            )
            
            self.active_processes[process_id] = process
            return process_id
    
    def add_reasoning_step(self, process_id: str, step: str) -> None:
        """Add a reasoning step to an active thought process"""
        with self._lock:
            if process_id in self.active_processes:
                self.active_processes[process_id].reasoning_steps.append(step)
    
    def record_interruption(self, process_id: str) -> None:
        """Record an interruption in thought process"""
        with self._lock:
            if process_id in self.active_processes:
                self.active_processes[process_id].interruptions += 1
    
    def end_thought_process(self, process_id: str, quality: CognitionQuality,
                          confidence: float) -> Optional[ThoughtProcess]:
        """Complete a thought process and analyze it"""
        with self._lock:
            if process_id not in self.active_processes:
                return None
            
            process = self.active_processes.pop(process_id)
            process.end_time = datetime.now()
            process.quality = quality
            process.confidence = confidence
            
            # Analyze the completed process
            self._analyze_process(process)
            
            # Store in history
            self.completed_processes.append(process)
            
            # Maintain history size
            if len(self.completed_processes) > self.max_history_size:
                self.completed_processes = self.completed_processes[-800:]
            
            # Update metrics
            self._update_metrics(process)
            
            return process
    
    def _analyze_process(self, process: ThoughtProcess) -> None:
        """Analyze a completed thought process for biases and patterns"""
        # Detect potential cognitive biases
        if self.bias_detection_enabled:
            biases = self._detect_biases(process)
            process.biases_detected = [bias.name for bias in biases]
            self.detected_biases.extend(biases)
        
        # Analyze reasoning quality
        self._assess_reasoning_quality(process)
    
    def _detect_biases(self, process: ThoughtProcess) -> List[CognitiveBias]:
        """Detect potential cognitive biases in thought process"""
        biases = []
        content = process.content.lower()
        steps = [step.lower() for step in process.reasoning_steps]
        all_text = content + " " + " ".join(steps)
        
        # Confirmation bias detection
        if self._detect_confirmation_bias(all_text):
            biases.append(CognitiveBias(
                name="confirmation_bias",
                description="Tendency to search for or interpret information in a way that confirms preconceptions",
                confidence=0.6,
                evidence=["Selective evidence gathering detected"]
            ))
        
        # Anchoring bias
        if self._detect_anchoring_bias(process):
            biases.append(CognitiveBias(
                name="anchoring_bias",
                description="Over-reliance on first piece of information encountered",
                confidence=0.5,
                evidence=["Heavy reliance on initial information"]
            ))
        
        # Availability heuristic
        if self._detect_availability_heuristic(all_text):
            biases.append(CognitiveBias(
                name="availability_heuristic",
                description="Overestimating likelihood of events with greater availability in memory",
                confidence=0.4,
                evidence=["Recent examples heavily weighted"]
            ))
        
        # Overconfidence bias
        if process.confidence > 0.9 and len(process.reasoning_steps) < 3:
            biases.append(CognitiveBias(
                name="overconfidence_bias",
                description="Excessive confidence in own answers and judgments",
                confidence=0.7,
                evidence=["High confidence with limited reasoning"]
            ))
        
        return biases
    
    def _detect_confirmation_bias(self, text: str) -> bool:
        """Simple confirmation bias detection"""
        bias_indicators = [
            "confirms that", "as expected", "obviously", "clearly shows",
            "proves my point", "validates", "supports my view"
        ]
        return any(indicator in text for indicator in bias_indicators)
    
    def _detect_anchoring_bias(self, process: ThoughtProcess) -> bool:
        """Detect anchoring bias patterns"""
        if len(process.reasoning_steps) < 2:
            return False
        
        # Check if first step heavily influences all subsequent steps
        first_step = process.reasoning_steps[0].lower()
        subsequent_steps = [step.lower() for step in process.reasoning_steps[1:]]
        
        # Simple heuristic: if first step concepts appear in >75% of subsequent steps
        first_step_words = set(first_step.split())
        overlap_count = 0
        
        for step in subsequent_steps:
            step_words = set(step.split())
            if len(first_step_words.intersection(step_words)) > len(first_step_words) * 0.3:
                overlap_count += 1
        
        return overlap_count > len(subsequent_steps) * 0.75
    
    def _detect_availability_heuristic(self, text: str) -> bool:
        """Detect availability heuristic patterns"""
        availability_indicators = [
            "recently", "just saw", "remember hearing", "comes to mind",
            "for example", "like when", "similar to"
        ]
        return any(indicator in text for indicator in availability_indicators)
    
    def _assess_reasoning_quality(self, process: ThoughtProcess) -> None:
        """Assess quality of reasoning in the process"""
        quality_factors = {
            'logical_structure': self._assess_logical_structure(process),
            'evidence_quality': self._assess_evidence_quality(process),
            'consideration_of_alternatives': self._assess_alternatives(process),
            'depth_of_analysis': self._assess_depth(process)
        }
        
        # Store quality assessment in metadata
        process.metadata['quality_factors'] = quality_factors
        
        # Overall quality score
        avg_quality = sum(quality_factors.values()) / len(quality_factors)
        process.metadata['computed_quality'] = avg_quality
    
    def _assess_logical_structure(self, process: ThoughtProcess) -> float:
        """Assess logical structure of reasoning"""
        if len(process.reasoning_steps) < 2:
            return 0.3
        
        # Check for logical flow between steps
        # Simple heuristic: longer reasoning chains with clear steps
        step_quality = min(1.0, len(process.reasoning_steps) / 5.0)
        
        # Check for logical connectors
        logical_connectors = ["therefore", "because", "since", "if", "then", "thus", "hence"]
        connector_count = 0
        
        for step in process.reasoning_steps:
            if any(connector in step.lower() for connector in logical_connectors):
                connector_count += 1
        
        connector_score = min(1.0, connector_count / max(1, len(process.reasoning_steps) - 1))
        
        return (step_quality + connector_score) / 2
    
    def _assess_evidence_quality(self, process: ThoughtProcess) -> float:
        """Assess quality of evidence used in reasoning"""
        evidence_indicators = [
            "research shows", "studies indicate", "data suggests", "evidence",
            "statistics", "proven", "verified", "documented", "observed"
        ]
        
        all_text = process.content + " " + " ".join(process.reasoning_steps)
        evidence_mentions = sum(1 for indicator in evidence_indicators if indicator in all_text.lower())
        
        return min(1.0, evidence_mentions / 3.0)
    
    def _assess_alternatives(self, process: ThoughtProcess) -> float:
        """Assess consideration of alternative viewpoints"""
        alternative_indicators = [
            "however", "on the other hand", "alternatively", "but", "although",
            "conversely", "in contrast", "different perspective", "another view"
        ]
        
        all_text = process.content + " " + " ".join(process.reasoning_steps)
        alternative_mentions = sum(1 for indicator in alternative_indicators if indicator in all_text.lower())
        
        return min(1.0, alternative_mentions / 2.0)
    
    def _assess_depth(self, process: ThoughtProcess) -> float:
        """Assess depth of analysis"""
        depth_factors = [
            len(process.reasoning_steps),
            len(process.content.split()),
            sum(len(step.split()) for step in process.reasoning_steps)
        ]
        
        # Normalize and combine
        step_depth = min(1.0, len(process.reasoning_steps) / 5.0)
        content_depth = min(1.0, len(process.content.split()) / 50.0)
        reasoning_depth = min(1.0, sum(len(step.split()) for step in process.reasoning_steps) / 100.0)
        
        return (step_depth + content_depth + reasoning_depth) / 3
    
    def _update_metrics(self, process: ThoughtProcess) -> None:
        """Update cognitive metrics based on completed process"""
        self.cognitive_metrics['total_thoughts'] += 1
        
        # Duration
        if process.end_time:
            duration = (process.end_time - process.start_time).total_seconds()
            total_duration = self.cognitive_metrics['avg_thought_duration'] * (self.cognitive_metrics['total_thoughts'] - 1)
            self.cognitive_metrics['avg_thought_duration'] = (total_duration + duration) / self.cognitive_metrics['total_thoughts']
        
        # Quality distribution
        if process.quality:
            self.cognitive_metrics['quality_distribution'][process.quality.value] += 1
        
        # Type distribution
        self.cognitive_metrics['type_distribution'][process.type.value] += 1
        
        # Bias frequency
        for bias_name in process.biases_detected:
            self.cognitive_metrics['bias_frequency'][bias_name] = self.cognitive_metrics['bias_frequency'].get(bias_name, 0) + 1
        
        # Interruption rate
        total_interruptions = sum(p.interruptions for p in self.completed_processes[-100:])  # Last 100 processes
        recent_processes = min(100, len(self.completed_processes))
        self.cognitive_metrics['interruption_rate'] = total_interruptions / max(1, recent_processes)
        
        # Cognitive load (based on concurrent processes)
        self.cognitive_metrics['cognitive_load'] = len(self.active_processes) / 10.0  # Normalize to 0-1
        
        # Reasoning depth
        if process.reasoning_steps:
            depth_score = len(process.reasoning_steps) / 10.0
            self.cognitive_metrics['reasoning_depth'] = min(1.0, depth_score)
        
        # Confidence calibration (how well confidence matches actual quality)
        if process.quality and process.confidence > 0:
            quality_scores = {'excellent': 1.0, 'good': 0.8, 'adequate': 0.6, 'poor': 0.4, 'uncertain': 0.2}
            actual_quality = quality_scores.get(process.quality.value, 0.5)
            calibration_error = abs(process.confidence - actual_quality)
            # Update running average of calibration
            prev_calibration = self.cognitive_metrics['confidence_calibration']
            self.cognitive_metrics['confidence_calibration'] = (prev_calibration * 0.9) + ((1 - calibration_error) * 0.1)
    
    def get_metacognitive_report(self) -> Dict[str, Any]:
        """Generate comprehensive metacognitive report"""
        with self._lock:
            recent_processes = self.completed_processes[-20:] if self.completed_processes else []
            
            return {
                'timestamp': datetime.now(),
                'current_state': {
                    'active_processes': len(self.active_processes),
                    'cognitive_load': self.cognitive_metrics['cognitive_load'],
                    'metacognitive_awareness': self.cognitive_metrics['metacognitive_awareness']
                },
                'recent_performance': {
                    'process_count': len(recent_processes),
                    'avg_quality': self._calculate_avg_quality(recent_processes),
                    'avg_confidence': self._calculate_avg_confidence(recent_processes),
                    'bias_rate': self._calculate_bias_rate(recent_processes)
                },
                'cognitive_patterns': {
                    'preferred_thought_types': self._get_preferred_types(),
                    'reasoning_depth': self.cognitive_metrics['reasoning_depth'],
                    'interruption_tendency': self.cognitive_metrics['interruption_rate']
                },
                'bias_analysis': {
                    'total_biases_detected': len(self.detected_biases),
                    'most_common_biases': self._get_common_biases(),
                    'bias_trend': self._analyze_bias_trend()
                },
                'calibration': {
                    'confidence_calibration': self.cognitive_metrics['confidence_calibration'],
                    'overconfidence_tendency': self._assess_overconfidence(),
                    'uncertainty_comfort': self._assess_uncertainty_comfort()
                },
                'recommendations': self._generate_recommendations()
            }
    
    def _calculate_avg_quality(self, processes: List[ThoughtProcess]) -> float:
        """Calculate average quality of recent processes"""
        if not processes:
            return 0.0
        
        quality_scores = {'excellent': 1.0, 'good': 0.8, 'adequate': 0.6, 'poor': 0.4, 'uncertain': 0.2}
        total_score = sum(quality_scores.get(p.quality.value, 0.0) for p in processes if p.quality)
        return total_score / len(processes)
    
    def _calculate_avg_confidence(self, processes: List[ThoughtProcess]) -> float:
        """Calculate average confidence of recent processes"""
        if not processes:
            return 0.0
        
        confidences = [p.confidence for p in processes if p.confidence > 0]
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def _calculate_bias_rate(self, processes: List[ThoughtProcess]) -> float:
        """Calculate rate of bias detection in recent processes"""
        if not processes:
            return 0.0
        
        biased_processes = sum(1 for p in processes if p.biases_detected)
        return biased_processes / len(processes)
    
    def _get_preferred_types(self) -> Dict[str, float]:
        """Get distribution of preferred thought types"""
        total = self.cognitive_metrics['total_thoughts']
        if total == 0:
            return {}
        
        return {t: count/total for t, count in self.cognitive_metrics['type_distribution'].items() if count > 0}
    
    def _get_common_biases(self) -> List[tuple]:
        """Get most commonly detected biases"""
        bias_items = list(self.cognitive_metrics['bias_frequency'].items())
        bias_items.sort(key=lambda x: x[1], reverse=True)
        return bias_items[:5]
    
    def _analyze_bias_trend(self) -> str:
        """Analyze trend in bias detection"""
        if len(self.detected_biases) < 10:
            return "insufficient_data"
        
        recent_biases = [b for b in self.detected_biases if (datetime.now() - b.detected_at).days < 7]
        older_biases = [b for b in self.detected_biases if 7 <= (datetime.now() - b.detected_at).days < 14]
        
        if len(recent_biases) > len(older_biases):
            return "increasing"
        elif len(recent_biases) < len(older_biases):
            return "decreasing"
        else:
            return "stable"
    
    def _assess_overconfidence(self) -> float:
        """Assess tendency toward overconfidence"""
        recent_processes = self.completed_processes[-50:] if len(self.completed_processes) >= 50 else self.completed_processes
        
        if not recent_processes:
            return 0.0
        
        overconfident_count = 0
        for p in recent_processes:
            if p.confidence > 0.8 and p.quality and p.quality.value in ['adequate', 'poor']:
                overconfident_count += 1
        
        return overconfident_count / len(recent_processes)
    
    def _assess_uncertainty_comfort(self) -> float:
        """Assess comfort with uncertainty and 'don't know' responses"""
        recent_processes = self.completed_processes[-50:] if len(self.completed_processes) >= 50 else self.completed_processes
        
        if not recent_processes:
            return 0.5
        
        uncertain_responses = sum(1 for p in recent_processes if p.quality == CognitionQuality.UNCERTAIN)
        return uncertain_responses / len(recent_processes)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate metacognitive improvement recommendations"""
        recommendations = []
        
        # Based on bias patterns
        common_biases = self._get_common_biases()
        if common_biases:
            top_bias = common_biases[0][0]
            recommendations.append(f"Focus on reducing {top_bias} in reasoning processes")
        
        # Based on confidence calibration
        if self.cognitive_metrics['confidence_calibration'] < 0.7:
            recommendations.append("Work on improving confidence calibration through more careful self-assessment")
        
        # Based on reasoning depth
        if self.cognitive_metrics['reasoning_depth'] < 0.5:
            recommendations.append("Increase depth of reasoning by adding more analytical steps")
        
        # Based on interruption rate
        if self.cognitive_metrics['interruption_rate'] > 0.3:
            recommendations.append("Focus on maintaining sustained attention in thought processes")
        
        # Based on quality trends
        recent_quality = self._calculate_avg_quality(self.completed_processes[-10:])
        if recent_quality < 0.6:
            recommendations.append("Focus on improving overall reasoning quality through more systematic analysis")
        
        return recommendations
    
    def export_metacognitive_data(self) -> Dict[str, Any]:
        """Export metacognitive data for analysis or persistence"""
        with self._lock:
            return {
                'metrics': self.cognitive_metrics.copy(),
                'recent_processes': [
                    {
                        'id': p.id,
                        'type': p.type.value,
                        'quality': p.quality.value if p.quality else None,
                        'confidence': p.confidence,
                        'biases': p.biases_detected,
                        'reasoning_steps_count': len(p.reasoning_steps),
                        'interruptions': p.interruptions,
                        'duration': (p.end_time - p.start_time).total_seconds() if p.end_time else None
                    }
                    for p in self.completed_processes[-100:]
                ],
                'bias_summary': {
                    'total_detected': len(self.detected_biases),
                    'by_type': self.cognitive_metrics['bias_frequency'].copy(),
                    'recent_trend': self._analyze_bias_trend()
                }
            }