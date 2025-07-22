"""
Emotional Intelligence System for Shvayambhu LLM

Provides comprehensive emotional understanding, processing, and generation
with consciousness-aware emotional modeling and empathetic responses.
"""

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

import numpy as np
from collections import defaultdict, deque

# Consciousness integration
from ..consciousness import ConsciousnessEngine, ConsciousnessLevel
from ..consciousness.base import ConsciousnessAwareModule


class EmotionCategory(Enum):
    """Primary emotion categories based on psychological research."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    LOVE = "love"
    SHAME = "shame"
    PRIDE = "pride"
    GRATITUDE = "gratitude"
    CONTEMPT = "contempt"
    ENVY = "envy"
    GUILT = "guilt"
    HOPE = "hope"
    RELIEF = "relief"
    ANXIETY = "anxiety"
    EXCITEMENT = "excitement"
    CALMNESS = "calmness"


class EmotionDimension(Enum):
    """Dimensional model of emotion."""
    VALENCE = "valence"      # Pleasant vs. Unpleasant
    AROUSAL = "arousal"      # Activated vs. Deactivated  
    DOMINANCE = "dominance"  # Dominant vs. Submissive


class EmotionSource(Enum):
    """Sources of emotional information."""
    TEXT = "text"
    SPEECH = "speech"
    FACIAL = "facial"
    MULTIMODAL = "multimodal"
    CONTEXTUAL = "contextual"
    INTERNAL = "internal"
    MEMORY = "memory"


class EmotionalIntensity(Enum):
    """Levels of emotional intensity."""
    VERY_LOW = 0.1
    LOW = 0.3
    MODERATE = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9


@dataclass
class EmotionVector:
    """Multi-dimensional representation of emotion."""
    emotions: Dict[EmotionCategory, float] = field(default_factory=dict)
    dimensions: Dict[EmotionDimension, float] = field(default_factory=dict)
    intensity: float = 0.0
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    source: EmotionSource = EmotionSource.TEXT
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Ensure all emotion categories have values
        for category in EmotionCategory:
            if category not in self.emotions:
                self.emotions[category] = 0.0
        
        # Ensure all dimensions have values
        for dimension in EmotionDimension:
            if dimension not in self.dimensions:
                self.dimensions[dimension] = 0.0
    
    def get_dominant_emotion(self) -> Tuple[EmotionCategory, float]:
        """Get the most prominent emotion."""
        if not self.emotions:
            return EmotionCategory.CALMNESS, 0.0
        
        dominant = max(self.emotions.items(), key=lambda x: x[1])
        return dominant[0], dominant[1]
    
    def get_emotional_profile(self) -> Dict[str, Any]:
        """Get comprehensive emotional profile."""
        dominant_emotion, dominant_score = self.get_dominant_emotion()
        
        return {
            'dominant_emotion': dominant_emotion.value,
            'dominant_score': dominant_score,
            'valence': self.dimensions.get(EmotionDimension.VALENCE, 0.0),
            'arousal': self.dimensions.get(EmotionDimension.AROUSAL, 0.0),
            'dominance': self.dimensions.get(EmotionDimension.DOMINANCE, 0.0),
            'intensity': self.intensity,
            'confidence': self.confidence,
            'emotional_complexity': self._calculate_complexity()
        }
    
    def _calculate_complexity(self) -> float:
        """Calculate emotional complexity based on number of active emotions."""
        active_emotions = sum(1 for score in self.emotions.values() if score > 0.1)
        max_emotions = len(EmotionCategory)
        return active_emotions / max_emotions
    
    def blend_with(self, other: 'EmotionVector', weight: float = 0.5) -> 'EmotionVector':
        """Blend this emotion vector with another."""
        blended_emotions = {}
        for category in EmotionCategory:
            self_score = self.emotions.get(category, 0.0)
            other_score = other.emotions.get(category, 0.0)
            blended_emotions[category] = self_score * (1 - weight) + other_score * weight
        
        blended_dimensions = {}
        for dimension in EmotionDimension:
            self_val = self.dimensions.get(dimension, 0.0)
            other_val = other.dimensions.get(dimension, 0.0)
            blended_dimensions[dimension] = self_val * (1 - weight) + other_val * weight
        
        return EmotionVector(
            emotions=blended_emotions,
            dimensions=blended_dimensions,
            intensity=self.intensity * (1 - weight) + other.intensity * weight,
            confidence=min(self.confidence, other.confidence),
            source=EmotionSource.MULTIMODAL,
            context={'blend_sources': [self.source.value, other.source.value]}
        )


@dataclass
class EmotionalMemory:
    """Memory of emotional experiences."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    emotion_vector: EmotionVector = field(default_factory=EmotionVector)
    trigger: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    response_generated: str = ""
    outcome: str = ""
    importance: float = 0.5
    decay_factor: float = 0.95
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    
    def update_importance(self, new_importance: float):
        """Update importance with decay."""
        days_since_creation = (datetime.now() - self.created_at).days
        decayed_importance = self.importance * (self.decay_factor ** days_since_creation)
        self.importance = max(decayed_importance, new_importance)
    
    def access(self):
        """Record access to this memory."""
        self.access_count += 1
        self.last_accessed = datetime.now()
        # Boost importance slightly when accessed
        self.importance = min(self.importance * 1.1, 1.0)


class EmotionDetector(ABC):
    """Abstract base class for emotion detection."""
    
    @abstractmethod
    async def detect_emotions(self, input_data: Any, context: Dict[str, Any]) -> EmotionVector:
        """Detect emotions from input data."""
        pass


class TextEmotionDetector(EmotionDetector):
    """Emotion detection from text using NLP techniques."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._init_emotion_lexicons()
        self._init_patterns()
    
    def _init_emotion_lexicons(self):
        """Initialize emotion word lexicons."""
        self.emotion_lexicon = {
            EmotionCategory.JOY: {
                'happy', 'joy', 'joyful', 'cheerful', 'delighted', 'pleased', 'content',
                'ecstatic', 'thrilled', 'elated', 'euphoric', 'blissful', 'gleeful',
                'jubilant', 'merry', 'upbeat', 'optimistic', 'radiant'
            },
            EmotionCategory.SADNESS: {
                'sad', 'sorrow', 'grief', 'melancholy', 'depression', 'despair',
                'gloom', 'misery', 'heartbreak', 'anguish', 'mourn', 'weep', 
                'cry', 'tears', 'devastated', 'crushed', 'broken'
            },
            EmotionCategory.ANGER: {
                'angry', 'rage', 'fury', 'wrath', 'mad', 'furious', 'irate',
                'livid', 'outraged', 'incensed', 'annoyed', 'irritated',
                'frustrated', 'agitated', 'hostile', 'resentful'
            },
            EmotionCategory.FEAR: {
                'fear', 'afraid', 'scared', 'terrified', 'frightened', 'anxious',
                'worried', 'nervous', 'apprehensive', 'dread', 'panic', 'alarmed',
                'intimidated', 'paranoid', 'phobic', 'horror'
            },
            EmotionCategory.SURPRISE: {
                'surprised', 'amazed', 'astonished', 'shocked', 'stunned',
                'bewildered', 'flabbergasted', 'astounded', 'dumbfounded',
                'startled', 'unexpected', 'sudden'
            },
            EmotionCategory.DISGUST: {
                'disgusted', 'revolted', 'repulsed', 'sickened', 'nauseated',
                'appalled', 'horrified', 'repugnant', 'loathsome', 'vile'
            },
            EmotionCategory.TRUST: {
                'trust', 'faith', 'confidence', 'belief', 'reliance', 'dependence',
                'loyal', 'faithful', 'devoted', 'committed', 'reliable'
            },
            EmotionCategory.ANTICIPATION: {
                'anticipation', 'expectation', 'hope', 'eager', 'excited',
                'looking forward', 'awaiting', 'expecting', 'hopeful'
            },
            EmotionCategory.LOVE: {
                'love', 'adore', 'cherish', 'affection', 'fondness', 'devotion',
                'passionate', 'romantic', 'intimate', 'caring', 'tender'
            },
            EmotionCategory.SHAME: {
                'shame', 'ashamed', 'embarrassed', 'humiliated', 'mortified',
                'disgraced', 'guilty', 'regretful', 'remorseful'
            },
            EmotionCategory.PRIDE: {
                'proud', 'pride', 'accomplished', 'satisfied', 'triumphant',
                'confident', 'self-assured', 'dignified', 'honored'
            },
            EmotionCategory.GRATITUDE: {
                'grateful', 'thankful', 'appreciative', 'obliged', 'indebted',
                'blessed', 'recognition', 'acknowledgment'
            },
            EmotionCategory.ANXIETY: {
                'anxious', 'worried', 'stressed', 'tense', 'uneasy', 'restless',
                'agitated', 'troubled', 'concerned', 'nervous'
            },
            EmotionCategory.EXCITEMENT: {
                'excited', 'thrilled', 'pumped', 'hyped', 'energized',
                'enthusiastic', 'animated', 'exhilarated'
            },
            EmotionCategory.CALMNESS: {
                'calm', 'peaceful', 'serene', 'tranquil', 'relaxed', 'composed',
                'collected', 'zen', 'centered', 'balanced'
            }
        }
        
        # Create reverse mapping for quick lookup
        self.word_to_emotions = {}
        for emotion, words in self.emotion_lexicon.items():
            for word in words:
                if word not in self.word_to_emotions:
                    self.word_to_emotions[word] = []
                self.word_to_emotions[word].append(emotion)
    
    def _init_patterns(self):
        """Initialize regex patterns for emotion detection."""
        self.emotion_patterns = {
            # Intensity modifiers
            'intensifiers': r'\b(very|extremely|incredibly|absolutely|totally|completely|utterly|deeply|profoundly)\b',
            'diminishers': r'\b(somewhat|slightly|a bit|a little|kind of|sort of|rather|fairly)\b',
            
            # Negation patterns
            'negation': r'\b(not|no|never|nobody|nothing|nowhere|neither|nor|barely|scarcely|hardly)\b',
            
            # Emotional expressions
            'exclamations': r'[!]{2,}',
            'questions': r'[?]{2,}',
            'caps_emotion': r'\b[A-Z]{2,}\b',
            
            # Contextual patterns
            'feel_expressions': r'\b(feel|feeling|felt)\s+(\w+)',
            'emotion_expressions': r'\b(i am|i\'m|i feel|i\'m feeling|makes me)\s+(\w+)',
        }
    
    async def detect_emotions(self, text: str, context: Dict[str, Any] = None) -> EmotionVector:
        """Detect emotions from text."""
        if not text:
            return EmotionVector()
        
        text = text.lower().strip()
        words = re.findall(r'\b\w+\b', text)
        
        # Initialize emotion scores
        emotion_scores = {emotion: 0.0 for emotion in EmotionCategory}
        
        # Lexicon-based detection
        lexicon_scores = self._detect_lexicon_emotions(text, words)
        for emotion, score in lexicon_scores.items():
            emotion_scores[emotion] += score
        
        # Pattern-based detection
        pattern_scores = self._detect_pattern_emotions(text)
        for emotion, score in pattern_scores.items():
            emotion_scores[emotion] += score
        
        # Contextual enhancement
        if context:
            context_scores = self._detect_contextual_emotions(text, context)
            for emotion, score in context_scores.items():
                emotion_scores[emotion] += score
        
        # Calculate intensity and confidence
        total_emotion = sum(emotion_scores.values())
        intensity = min(total_emotion / 3.0, 1.0)  # Normalize intensity
        confidence = self._calculate_confidence(text, emotion_scores)
        
        # Calculate dimensional scores
        dimensions = self._calculate_dimensional_scores(emotion_scores)
        
        return EmotionVector(
            emotions=emotion_scores,
            dimensions=dimensions,
            intensity=intensity,
            confidence=confidence,
            source=EmotionSource.TEXT,
            context=context or {}
        )
    
    def _detect_lexicon_emotions(self, text: str, words: List[str]) -> Dict[EmotionCategory, float]:
        """Detect emotions using lexicon matching."""
        emotion_scores = {emotion: 0.0 for emotion in EmotionCategory}
        
        for word in words:
            if word in self.word_to_emotions:
                emotions = self.word_to_emotions[word]
                base_score = 1.0 / len(emotions)  # Distribute score across emotions
                
                # Apply intensity modifiers
                word_context = self._get_word_context(text, word)
                intensity_modifier = self._get_intensity_modifier(word_context)
                
                for emotion in emotions:
                    emotion_scores[emotion] += base_score * intensity_modifier
        
        # Normalize by text length
        if words:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= len(words)
        
        return emotion_scores
    
    def _detect_pattern_emotions(self, text: str) -> Dict[EmotionCategory, float]:
        """Detect emotions using regex patterns."""
        emotion_scores = {emotion: 0.0 for emotion in EmotionCategory}
        
        # Exclamations indicate high arousal emotions
        if re.search(self.emotion_patterns['exclamations'], text):
            emotion_scores[EmotionCategory.EXCITEMENT] += 0.3
            emotion_scores[EmotionCategory.JOY] += 0.2
            emotion_scores[EmotionCategory.ANGER] += 0.2
        
        # All caps indicates strong emotion
        caps_words = re.findall(self.emotion_patterns['caps_emotion'], text)
        if caps_words:
            intensity_boost = min(len(caps_words) * 0.1, 0.5)
            # Boost dominant emotions found so far
            current_max = max(emotion_scores.values())
            if current_max > 0:
                for emotion, score in emotion_scores.items():
                    if score == current_max:
                        emotion_scores[emotion] += intensity_boost
        
        # Feel expressions
        feel_matches = re.finditer(self.emotion_patterns['feel_expressions'], text)
        for match in feel_matches:
            feeling_word = match.group(2).lower()
            if feeling_word in self.word_to_emotions:
                emotions = self.word_to_emotions[feeling_word]
                for emotion in emotions:
                    emotion_scores[emotion] += 0.4  # Higher weight for explicit feelings
        
        return emotion_scores
    
    def _detect_contextual_emotions(self, text: str, context: Dict[str, Any]) -> Dict[EmotionCategory, float]:
        """Detect emotions based on context."""
        emotion_scores = {emotion: 0.0 for emotion in EmotionCategory}
        
        # Previous emotion context
        if 'previous_emotion' in context:
            prev_emotion = context['previous_emotion']
            if isinstance(prev_emotion, str):
                try:
                    prev_emotion = EmotionCategory(prev_emotion)
                    # Emotional momentum - previous emotions influence current
                    emotion_scores[prev_emotion] += 0.2
                except ValueError:
                    pass
        
        # Conversation context
        if 'conversation_sentiment' in context:
            sentiment = context['conversation_sentiment']
            if sentiment > 0.5:
                emotion_scores[EmotionCategory.JOY] += 0.3
            elif sentiment < -0.5:
                emotion_scores[EmotionCategory.SADNESS] += 0.3
        
        # Topic context
        if 'topic' in context:
            topic = context['topic'].lower()
            if any(keyword in topic for keyword in ['death', 'loss', 'grief']):
                emotion_scores[EmotionCategory.SADNESS] += 0.4
            elif any(keyword in topic for keyword in ['success', 'achievement', 'victory']):
                emotion_scores[EmotionCategory.JOY] += 0.4
                emotion_scores[EmotionCategory.PRIDE] += 0.3
        
        return emotion_scores
    
    def _get_word_context(self, text: str, word: str) -> str:
        """Get context around a word."""
        words = text.split()
        try:
            word_index = words.index(word)
            start = max(0, word_index - 2)
            end = min(len(words), word_index + 3)
            return ' '.join(words[start:end])
        except ValueError:
            return word
    
    def _get_intensity_modifier(self, context: str) -> float:
        """Calculate intensity modifier based on context."""
        modifier = 1.0
        
        # Check for intensifiers
        if re.search(self.emotion_patterns['intensifiers'], context):
            modifier *= 1.5
        
        # Check for diminishers
        if re.search(self.emotion_patterns['diminishers'], context):
            modifier *= 0.7
        
        # Check for negation
        if re.search(self.emotion_patterns['negation'], context):
            modifier *= -0.5  # Flip and reduce
        
        return modifier
    
    def _calculate_dimensional_scores(self, emotion_scores: Dict[EmotionCategory, float]) -> Dict[EmotionDimension, float]:
        """Calculate dimensional emotion scores."""
        # Mapping emotions to dimensional space
        valence_mapping = {
            EmotionCategory.JOY: 0.8, EmotionCategory.LOVE: 0.9, EmotionCategory.GRATITUDE: 0.7,
            EmotionCategory.PRIDE: 0.6, EmotionCategory.TRUST: 0.5, EmotionCategory.HOPE: 0.6,
            EmotionCategory.RELIEF: 0.7, EmotionCategory.EXCITEMENT: 0.8, EmotionCategory.CALMNESS: 0.4,
            EmotionCategory.SADNESS: -0.7, EmotionCategory.ANGER: -0.6, EmotionCategory.FEAR: -0.8,
            EmotionCategory.DISGUST: -0.8, EmotionCategory.SHAME: -0.7, EmotionCategory.GUILT: -0.6,
            EmotionCategory.ENVY: -0.5, EmotionCategory.ANXIETY: -0.6, EmotionCategory.CONTEMPT: -0.5,
            EmotionCategory.SURPRISE: 0.1, EmotionCategory.ANTICIPATION: 0.2
        }
        
        arousal_mapping = {
            EmotionCategory.JOY: 0.6, EmotionCategory.ANGER: 0.8, EmotionCategory.FEAR: 0.9,
            EmotionCategory.SURPRISE: 0.8, EmotionCategory.EXCITEMENT: 0.9, EmotionCategory.ANXIETY: 0.7,
            EmotionCategory.ANTICIPATION: 0.6, EmotionCategory.LOVE: 0.5, EmotionCategory.PRIDE: 0.4,
            EmotionCategory.SADNESS: 0.3, EmotionCategory.DISGUST: 0.6, EmotionCategory.SHAME: 0.4,
            EmotionCategory.GUILT: 0.4, EmotionCategory.TRUST: 0.2, EmotionCategory.GRATITUDE: 0.3,
            EmotionCategory.CALMNESS: 0.1, EmotionCategory.RELIEF: 0.2, EmotionCategory.HOPE: 0.4,
            EmotionCategory.CONTEMPT: 0.5, EmotionCategory.ENVY: 0.6
        }
        
        dominance_mapping = {
            EmotionCategory.ANGER: 0.7, EmotionCategory.PRIDE: 0.8, EmotionCategory.CONTEMPT: 0.7,
            EmotionCategory.DISGUST: 0.6, EmotionCategory.JOY: 0.5, EmotionCategory.EXCITEMENT: 0.6,
            EmotionCategory.TRUST: 0.4, EmotionCategory.LOVE: 0.3, EmotionCategory.GRATITUDE: 0.2,
            EmotionCategory.FEAR: -0.7, EmotionCategory.SADNESS: -0.5, EmotionCategory.SHAME: -0.8,
            EmotionCategory.GUILT: -0.6, EmotionCategory.ANXIETY: -0.6, EmotionCategory.SURPRISE: 0.0,
            EmotionCategory.ANTICIPATION: 0.2, EmotionCategory.HOPE: 0.1, EmotionCategory.RELIEF: 0.3,
            EmotionCategory.CALMNESS: 0.0, EmotionCategory.ENVY: -0.2
        }
        
        # Calculate weighted dimensional scores
        valence = sum(score * valence_mapping.get(emotion, 0.0) 
                     for emotion, score in emotion_scores.items())
        arousal = sum(score * arousal_mapping.get(emotion, 0.0) 
                     for emotion, score in emotion_scores.items())
        dominance = sum(score * dominance_mapping.get(emotion, 0.0) 
                       for emotion, score in emotion_scores.items())
        
        # Normalize to [-1, 1] range
        total_score = sum(emotion_scores.values())
        if total_score > 0:
            valence /= total_score
            arousal /= total_score
            dominance /= total_score
        
        return {
            EmotionDimension.VALENCE: np.clip(valence, -1.0, 1.0),
            EmotionDimension.AROUSAL: np.clip(arousal, 0.0, 1.0),
            EmotionDimension.DOMINANCE: np.clip(dominance, -1.0, 1.0)
        }
    
    def _calculate_confidence(self, text: str, emotion_scores: Dict[EmotionCategory, float]) -> float:
        """Calculate confidence in emotion detection."""
        confidence_factors = []
        
        # Text length factor
        word_count = len(text.split())
        if word_count >= 10:
            confidence_factors.append(0.3)
        elif word_count >= 5:
            confidence_factors.append(0.2)
        else:
            confidence_factors.append(0.1)
        
        # Emotion word count
        emotion_word_count = sum(1 for word in text.split() 
                               if word.lower() in self.word_to_emotions)
        if emotion_word_count > 0:
            confidence_factors.append(min(emotion_word_count * 0.1, 0.3))
        
        # Pattern matches
        pattern_matches = 0
        for pattern in self.emotion_patterns.values():
            if re.search(pattern, text):
                pattern_matches += 1
        if pattern_matches > 0:
            confidence_factors.append(min(pattern_matches * 0.1, 0.2))
        
        # Emotion clarity (how dominant is the strongest emotion)
        if emotion_scores:
            max_score = max(emotion_scores.values())
            other_scores = [score for score in emotion_scores.values() if score != max_score]
            if other_scores:
                clarity = max_score - np.mean(other_scores)
                confidence_factors.append(min(clarity, 0.2))
        
        return min(sum(confidence_factors), 1.0)


class SpeechEmotionDetector(EmotionDetector):
    """Emotion detection from speech audio."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def detect_emotions(self, audio_features: Dict[str, Any], context: Dict[str, Any] = None) -> EmotionVector:
        """Detect emotions from speech audio features."""
        emotion_scores = {emotion: 0.0 for emotion in EmotionCategory}
        
        # Prosodic features analysis
        if 'prosodic' in audio_features:
            prosodic = audio_features['prosodic']
            
            # Pitch analysis
            pitch_mean = prosodic.get('pitch_mean', 0)
            pitch_std = prosodic.get('pitch_std', 0)
            
            if pitch_mean > 200:  # High pitch
                emotion_scores[EmotionCategory.EXCITEMENT] += 0.4
                emotion_scores[EmotionCategory.JOY] += 0.3
                emotion_scores[EmotionCategory.FEAR] += 0.2
            elif pitch_mean < 120:  # Low pitch
                emotion_scores[EmotionCategory.SADNESS] += 0.3
                emotion_scores[EmotionCategory.ANGER] += 0.2
            
            if pitch_std > 50:  # High pitch variation
                emotion_scores[EmotionCategory.EXCITEMENT] += 0.2
                emotion_scores[EmotionCategory.SURPRISE] += 0.2
            
            # Speaking rate analysis
            speaking_rate = prosodic.get('speaking_rate', 0)
            if speaking_rate > 6:  # Fast speaking
                emotion_scores[EmotionCategory.EXCITEMENT] += 0.3
                emotion_scores[EmotionCategory.ANXIETY] += 0.2
            elif speaking_rate < 3:  # Slow speaking
                emotion_scores[EmotionCategory.SADNESS] += 0.3
                emotion_scores[EmotionCategory.CALMNESS] += 0.2
        
        # Energy analysis
        if 'energy' in audio_features:
            energy = audio_features['energy']
            if energy > 0.7:
                emotion_scores[EmotionCategory.ANGER] += 0.4
                emotion_scores[EmotionCategory.EXCITEMENT] += 0.3
            elif energy < 0.3:
                emotion_scores[EmotionCategory.SADNESS] += 0.3
                emotion_scores[EmotionCategory.CALMNESS] += 0.2
        
        # Spectral features
        if 'spectral' in audio_features:
            spectral = audio_features['spectral']
            spectral_centroid = spectral.get('centroid', 0)
            
            if spectral_centroid > 3000:
                emotion_scores[EmotionCategory.JOY] += 0.2
                emotion_scores[EmotionCategory.EXCITEMENT] += 0.2
            elif spectral_centroid < 1500:
                emotion_scores[EmotionCategory.SADNESS] += 0.2
        
        # Voice quality indicators
        if 'voice_quality' in audio_features:
            quality = audio_features['voice_quality']
            
            if quality.get('jitter', 0) > 0.05:  # Voice instability
                emotion_scores[EmotionCategory.ANXIETY] += 0.3
                emotion_scores[EmotionCategory.FEAR] += 0.2
            
            if quality.get('breathiness', 0) > 0.5:
                emotion_scores[EmotionCategory.SADNESS] += 0.2
                emotion_scores[EmotionCategory.CALMNESS] += 0.1
        
        # Calculate intensity and dimensions
        total_emotion = sum(emotion_scores.values())
        intensity = min(total_emotion / 2.0, 1.0)
        
        dimensions = self._calculate_speech_dimensions(audio_features, emotion_scores)
        confidence = self._calculate_speech_confidence(audio_features)
        
        return EmotionVector(
            emotions=emotion_scores,
            dimensions=dimensions,
            intensity=intensity,
            confidence=confidence,
            source=EmotionSource.SPEECH,
            context=context or {}
        )
    
    def _calculate_speech_dimensions(self, audio_features: Dict[str, Any], 
                                   emotion_scores: Dict[EmotionCategory, float]) -> Dict[EmotionDimension, float]:
        """Calculate dimensional scores from speech features."""
        # Valence from spectral brightness and harmony
        valence = 0.0
        if 'spectral' in audio_features:
            centroid = audio_features['spectral'].get('centroid', 2000)
            valence = (centroid - 1500) / 2500  # Normalize around neutral
        
        # Arousal from energy and pitch variation
        arousal = 0.0
        if 'energy' in audio_features:
            arousal += audio_features['energy'] * 0.5
        if 'prosodic' in audio_features:
            pitch_std = audio_features['prosodic'].get('pitch_std', 0)
            arousal += min(pitch_std / 100.0, 0.5)
        
        # Dominance from volume and pitch height
        dominance = 0.0
        if 'energy' in audio_features:
            dominance += (audio_features['energy'] - 0.5) * 0.6
        if 'prosodic' in audio_features:
            pitch_mean = audio_features['prosodic'].get('pitch_mean', 150)
            dominance += (pitch_mean - 150) / 200  # Normalize around average
        
        return {
            EmotionDimension.VALENCE: np.clip(valence, -1.0, 1.0),
            EmotionDimension.AROUSAL: np.clip(arousal, 0.0, 1.0),
            EmotionDimension.DOMINANCE: np.clip(dominance, -1.0, 1.0)
        }
    
    def _calculate_speech_confidence(self, audio_features: Dict[str, Any]) -> float:
        """Calculate confidence based on audio quality and feature availability."""
        confidence = 0.0
        
        # Feature availability
        if 'prosodic' in audio_features:
            confidence += 0.3
        if 'energy' in audio_features:
            confidence += 0.2
        if 'spectral' in audio_features:
            confidence += 0.2
        if 'voice_quality' in audio_features:
            confidence += 0.2
        
        # Audio quality indicators
        if 'quality_metrics' in audio_features:
            quality = audio_features['quality_metrics']
            snr = quality.get('snr', 0)
            if snr > 20:  # Good signal-to-noise ratio
                confidence += 0.1
        
        return min(confidence, 1.0)


class EmotionalMemoryManager(ConsciousnessAwareModule):
    """Manages emotional memories and learning from experiences."""
    
    def __init__(self, consciousness_engine=None, max_memories: int = 10000):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.max_memories = max_memories
        self.memories: Dict[str, EmotionalMemory] = {}
        self.memory_index = defaultdict(list)  # Index by emotion categories
        self.recent_memories = deque(maxlen=100)  # Quick access to recent memories
    
    async def store_memory(self, trigger: str, emotion_vector: EmotionVector, 
                          context: Dict[str, Any], response: str = "", 
                          outcome: str = "") -> str:
        """Store a new emotional memory."""
        memory = EmotionalMemory(
            emotion_vector=emotion_vector,
            trigger=trigger,
            context=context,
            response_generated=response,
            outcome=outcome,
            importance=self._calculate_importance(emotion_vector, context)
        )
        
        # Store memory
        self.memories[memory.id] = memory
        self.recent_memories.append(memory.id)
        
        # Update indices
        dominant_emotion, _ = emotion_vector.get_dominant_emotion()
        self.memory_index[dominant_emotion].append(memory.id)
        
        # Cleanup if needed
        await self._cleanup_memories()
        
        self.logger.debug(f"Stored emotional memory: {memory.id} ({dominant_emotion.value})")
        return memory.id
    
    async def retrieve_similar_memories(self, current_emotion: EmotionVector, 
                                      trigger: str = "", limit: int = 5) -> List[EmotionalMemory]:
        """Retrieve memories similar to current emotional state."""
        similar_memories = []
        
        # Get dominant emotion
        dominant_emotion, _ = current_emotion.get_dominant_emotion()
        
        # Find memories with same dominant emotion
        candidate_ids = self.memory_index.get(dominant_emotion, [])
        
        # Calculate similarity scores
        scored_memories = []
        for memory_id in candidate_ids:
            if memory_id in self.memories:
                memory = self.memories[memory_id]
                similarity = self._calculate_emotion_similarity(current_emotion, memory.emotion_vector)
                
                # Boost similarity if trigger is similar
                if trigger and trigger.lower() in memory.trigger.lower():
                    similarity += 0.2
                
                # Weight by importance and recency
                importance_weight = memory.importance
                recency_weight = self._calculate_recency_weight(memory.created_at)
                
                final_score = similarity * importance_weight * recency_weight
                scored_memories.append((final_score, memory))
        
        # Sort by score and return top memories
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        similar_memories = [memory for _, memory in scored_memories[:limit]]
        
        # Update access counts
        for memory in similar_memories:
            memory.access()
        
        return similar_memories
    
    async def update_memory_outcome(self, memory_id: str, outcome: str, importance_boost: float = 0.0):
        """Update memory with outcome information."""
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            memory.outcome = outcome
            if importance_boost > 0:
                memory.update_importance(memory.importance + importance_boost)
            self.logger.debug(f"Updated memory outcome: {memory_id}")
    
    async def get_emotional_patterns(self, emotion_category: EmotionCategory = None, 
                                   days: int = 30) -> Dict[str, Any]:
        """Analyze emotional patterns from memory."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        relevant_memories = []
        for memory in self.memories.values():
            if memory.created_at >= cutoff_date:
                if emotion_category is None:
                    relevant_memories.append(memory)
                else:
                    dominant_emotion, _ = memory.emotion_vector.get_dominant_emotion()
                    if dominant_emotion == emotion_category:
                        relevant_memories.append(memory)
        
        if not relevant_memories:
            return {'pattern_count': 0, 'patterns': []}
        
        # Analyze patterns
        patterns = {
            'total_memories': len(relevant_memories),
            'emotion_distribution': self._analyze_emotion_distribution(relevant_memories),
            'trigger_patterns': self._analyze_trigger_patterns(relevant_memories),
            'temporal_patterns': self._analyze_temporal_patterns(relevant_memories),
            'outcome_patterns': self._analyze_outcome_patterns(relevant_memories),
            'intensity_trends': self._analyze_intensity_trends(relevant_memories)
        }
        
        return patterns
    
    def _calculate_importance(self, emotion_vector: EmotionVector, context: Dict[str, Any]) -> float:
        """Calculate importance of an emotional memory."""
        importance = 0.0
        
        # Base importance from emotion intensity
        importance += emotion_vector.intensity * 0.4
        
        # Boost for strong emotions
        if emotion_vector.intensity > 0.7:
            importance += 0.2
        
        # Boost for unusual emotions (based on recent memory distribution)
        recent_emotion_counts = defaultdict(int)
        for memory_id in list(self.recent_memories)[-50:]:  # Last 50 memories
            if memory_id in self.memories:
                memory = self.memories[memory_id]
                dominant_emotion, _ = memory.emotion_vector.get_dominant_emotion()
                recent_emotion_counts[dominant_emotion] += 1
        
        current_dominant, _ = emotion_vector.get_dominant_emotion()
        if recent_emotion_counts:
            rarity = 1.0 - (recent_emotion_counts[current_dominant] / sum(recent_emotion_counts.values()))
            importance += rarity * 0.2
        
        # Context importance
        if context:
            if context.get('user_explicit_emotion', False):
                importance += 0.1
            if context.get('significant_event', False):
                importance += 0.2
            if context.get('interpersonal_interaction', False):
                importance += 0.1
        
        return min(importance, 1.0)
    
    def _calculate_emotion_similarity(self, emotion1: EmotionVector, emotion2: EmotionVector) -> float:
        """Calculate similarity between two emotion vectors."""
        # Categorical similarity
        categorical_sim = 0.0
        for emotion in EmotionCategory:
            score1 = emotion1.emotions.get(emotion, 0.0)
            score2 = emotion2.emotions.get(emotion, 0.0)
            categorical_sim += 1.0 - abs(score1 - score2)
        categorical_sim /= len(EmotionCategory)
        
        # Dimensional similarity
        dimensional_sim = 0.0
        for dimension in EmotionDimension:
            val1 = emotion1.dimensions.get(dimension, 0.0)
            val2 = emotion2.dimensions.get(dimension, 0.0)
            dimensional_sim += 1.0 - abs(val1 - val2) / 2.0  # Normalize by max distance
        dimensional_sim /= len(EmotionDimension)
        
        # Intensity similarity
        intensity_sim = 1.0 - abs(emotion1.intensity - emotion2.intensity)
        
        # Weighted combination
        similarity = (categorical_sim * 0.5 + dimensional_sim * 0.3 + intensity_sim * 0.2)
        return similarity
    
    def _calculate_recency_weight(self, created_at: datetime) -> float:
        """Calculate recency weight for memory importance."""
        days_ago = (datetime.now() - created_at).total_seconds() / 86400
        # Exponential decay with half-life of 7 days
        return np.exp(-days_ago * np.log(2) / 7)
    
    async def _cleanup_memories(self):
        """Clean up old or unimportant memories."""
        if len(self.memories) <= self.max_memories:
            return
        
        # Calculate retention scores
        scored_memories = []
        for memory_id, memory in self.memories.items():
            retention_score = (memory.importance * 
                              self._calculate_recency_weight(memory.created_at) * 
                              np.log(1 + memory.access_count))
            scored_memories.append((retention_score, memory_id))
        
        # Sort by retention score and keep top memories
        scored_memories.sort(reverse=True)
        memories_to_keep = set(memory_id for _, memory_id in scored_memories[:self.max_memories])
        
        # Remove memories not in keep set
        memories_to_remove = set(self.memories.keys()) - memories_to_keep
        for memory_id in memories_to_remove:
            memory = self.memories[memory_id]
            dominant_emotion, _ = memory.emotion_vector.get_dominant_emotion()
            
            # Remove from indices
            if memory_id in self.memory_index[dominant_emotion]:
                self.memory_index[dominant_emotion].remove(memory_id)
            
            # Remove from memories
            del self.memories[memory_id]
        
        self.logger.info(f"Cleaned up {len(memories_to_remove)} emotional memories")
    
    def _analyze_emotion_distribution(self, memories: List[EmotionalMemory]) -> Dict[str, int]:
        """Analyze distribution of emotions in memories."""
        emotion_counts = defaultdict(int)
        for memory in memories:
            dominant_emotion, _ = memory.emotion_vector.get_dominant_emotion()
            emotion_counts[dominant_emotion.value] += 1
        return dict(emotion_counts)
    
    def _analyze_trigger_patterns(self, memories: List[EmotionalMemory]) -> Dict[str, int]:
        """Analyze common trigger patterns."""
        trigger_words = defaultdict(int)
        for memory in memories:
            words = memory.trigger.lower().split()
            for word in words:
                if len(word) > 3:  # Filter short words
                    trigger_words[word] += 1
        
        # Return top 10 trigger words
        return dict(sorted(trigger_words.items(), key=lambda x: x[1], reverse=True)[:10])
    
    def _analyze_temporal_patterns(self, memories: List[EmotionalMemory]) -> Dict[str, Any]:
        """Analyze temporal patterns in emotional memories."""
        hourly_counts = defaultdict(int)
        daily_counts = defaultdict(int)
        
        for memory in memories:
            hour = memory.created_at.hour
            day_of_week = memory.created_at.weekday()
            
            hourly_counts[hour] += 1
            daily_counts[day_of_week] += 1
        
        return {
            'hourly_distribution': dict(hourly_counts),
            'daily_distribution': dict(daily_counts),
            'peak_emotional_hour': max(hourly_counts, key=hourly_counts.get) if hourly_counts else None,
            'peak_emotional_day': max(daily_counts, key=daily_counts.get) if daily_counts else None
        }
    
    def _analyze_outcome_patterns(self, memories: List[EmotionalMemory]) -> Dict[str, Any]:
        """Analyze outcomes of emotional experiences."""
        outcome_emotions = defaultdict(list)
        
        for memory in memories:
            if memory.outcome:
                dominant_emotion, score = memory.emotion_vector.get_dominant_emotion()
                outcome_emotions[memory.outcome].append((dominant_emotion.value, score))
        
        outcome_analysis = {}
        for outcome, emotion_data in outcome_emotions.items():
            emotions = [emotion for emotion, _ in emotion_data]
            avg_intensity = np.mean([score for _, score in emotion_data])
            most_common_emotion = max(set(emotions), key=emotions.count)
            
            outcome_analysis[outcome] = {
                'count': len(emotion_data),
                'average_intensity': avg_intensity,
                'most_common_emotion': most_common_emotion
            }
        
        return outcome_analysis
    
    def _analyze_intensity_trends(self, memories: List[EmotionalMemory]) -> Dict[str, float]:
        """Analyze trends in emotional intensity."""
        if len(memories) < 2:
            return {'trend': 'insufficient_data'}
        
        # Sort memories by date
        sorted_memories = sorted(memories, key=lambda m: m.created_at)
        intensities = [m.emotion_vector.intensity for m in sorted_memories]
        
        # Calculate trend using simple linear regression
        n = len(intensities)
        x = list(range(n))
        
        x_mean = np.mean(x)
        y_mean = np.mean(intensities)
        
        numerator = sum((x[i] - x_mean) * (intensities[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator != 0:
            slope = numerator / denominator
            trend_direction = 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable'
        else:
            slope = 0
            trend_direction = 'stable'
        
        return {
            'trend_direction': trend_direction,
            'slope': slope,
            'average_intensity': y_mean,
            'intensity_variance': np.var(intensities)
        }
    
    async def process_with_consciousness(self, input_data: Any) -> Any:
        """Process input with consciousness awareness."""
        # Store or retrieve memories based on input
        if isinstance(input_data, dict) and 'store' in input_data:
            return await self.store_memory(input_data['store'])
        elif isinstance(input_data, dict) and 'retrieve' in input_data:
            return await self.retrieve_relevant_memories(input_data['retrieve'])
        else:
            return await self.analyze_emotional_patterns()


class EmotionalResponseGenerator(ConsciousnessAwareModule):
    """Generates emotionally appropriate responses."""
    
    def __init__(self, consciousness_engine=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self._init_response_templates()
    
    def _init_response_templates(self):
        """Initialize response templates for different emotions."""
        self.response_templates = {
            EmotionCategory.JOY: {
                'acknowledgment': [
                    "I can sense your happiness about this!",
                    "That's wonderful to hear!",
                    "Your joy is quite evident and uplifting.",
                    "I'm glad this brings you such pleasure."
                ],
                'empathy': [
                    "I share in your excitement about this.",
                    "Your enthusiasm is contagious!",
                    "I can understand why this would make you so happy.",
                    "It's beautiful to witness such genuine joy."
                ],
                'support': [
                    "I hope this feeling continues for you.",
                    "You deserve all this happiness.",
                    "Enjoy this wonderful moment fully.",
                    "These joyful experiences are precious."
                ]
            },
            EmotionCategory.SADNESS: {
                'acknowledgment': [
                    "I can hear the sadness in what you're sharing.",
                    "This seems to be causing you considerable pain.",
                    "I recognize how difficult this must be for you.",
                    "Your sorrow is deeply felt and understood."
                ],
                'empathy': [
                    "I wish I could take away some of this pain for you.",
                    "My heart goes out to you during this difficult time.",
                    "I can only imagine how heavy this must feel.",
                    "Your sadness touches me deeply."
                ],
                'support': [
                    "Please know that you're not alone in this.",
                    "It's okay to feel this way - your emotions are valid.",
                    "Take the time you need to process these feelings.",
                    "I'm here to listen whenever you need to share."
                ]
            },
            EmotionCategory.ANGER: {
                'acknowledgment': [
                    "I can feel the intensity of your frustration.",
                    "Your anger about this situation is understandable.",
                    "This has clearly provoked strong feelings in you.",
                    "I recognize the fire of indignation in your words."
                ],
                'empathy': [
                    "I can understand why this would make you so angry.",
                    "Your fury about this injustice is justified.",
                    "I too would feel outraged by such circumstances.",
                    "The unfairness of this situation is infuriating."
                ],
                'support': [
                    "Your anger is valid and heard.",
                    "Sometimes anger is the appropriate response.",
                    "Let's work through these intense feelings together.",
                    "Channel this energy toward positive change."
                ]
            },
            EmotionCategory.FEAR: {
                'acknowledgment': [
                    "I can sense the fear and uncertainty you're experiencing.",
                    "This anxiety must be overwhelming right now.",
                    "I recognize how frightening this situation feels.",
                    "Your apprehension is completely understandable."
                ],
                'empathy': [
                    "Fear can be so isolating - I wish I could ease it.",
                    "I understand how paralyzing anxiety can feel.",
                    "Your courage in facing these fears is admirable.",
                    "Even in fear, you're still reaching out - that's brave."
                ],
                'support': [
                    "You're stronger than you realize.",
                    "We can face this fear together, step by step.",
                    "These feelings will pass, even if they don't feel like it now.",
                    "You have the strength to overcome this."
                ]
            },
            EmotionCategory.ANXIETY: {
                'acknowledgment': [
                    "I can sense the worry and tension you're carrying.",
                    "This anxiety seems to be weighing heavily on you.",
                    "I recognize the spiral of concerned thoughts.",
                    "Your restlessness and unease are palpable."
                ],
                'empathy': [
                    "Anxiety can make everything feel uncertain and overwhelming.",
                    "I understand how exhausting constant worry can be.",
                    "The racing thoughts must be so difficult to quiet.",
                    "I wish I could help calm this storm of anxiety."
                ],
                'support': [
                    "Let's try to ground these swirling thoughts together.",
                    "Focus on what you can control in this moment.",
                    "Your anxiety is valid, but it doesn't define reality.",
                    "Breathe with me - we'll take this one step at a time."
                ]
            }
        }
        
        # Response modifiers for different intensity levels
        self.intensity_modifiers = {
            EmotionalIntensity.VERY_LOW: {
                'prefix': ["I notice a hint of", "There's a subtle"],
                'tone': 'gentle'
            },
            EmotionalIntensity.LOW: {
                'prefix': ["I can sense some", "There seems to be"],
                'tone': 'calm'
            },
            EmotionalIntensity.MODERATE: {
                'prefix': ["I can feel the", "There's a clear sense of"],
                'tone': 'engaged'
            },
            EmotionalIntensity.HIGH: {
                'prefix': ["I can strongly sense", "The intensity of"],
                'tone': 'heightened'
            },
            EmotionalIntensity.VERY_HIGH: {
                'prefix': ["I'm overwhelmed by the", "The powerful"],
                'tone': 'intense'
            }
        }
    
    async def generate_empathetic_response(
        self,
        detected_emotion: EmotionVector,
        context: Dict[str, Any],
        previous_memories: List[EmotionalMemory] = None,
        consciousness_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate an empathetic response to detected emotion."""
        dominant_emotion, dominant_score = detected_emotion.get_dominant_emotion()
        
        # Determine response components
        response_components = await self._select_response_components(
            dominant_emotion, detected_emotion, context, previous_memories
        )
        
        # Generate base response
        base_response = await self._generate_base_response(
            dominant_emotion, dominant_score, response_components
        )
        
        # Apply consciousness-aware enhancements
        if consciousness_context and self.consciousness_engine:
            enhanced_response = await self._enhance_response_with_consciousness(
                base_response, detected_emotion, consciousness_context
            )
        else:
            enhanced_response = base_response
        
        # Add memory-informed elements
        if previous_memories:
            memory_enhanced_response = await self._add_memory_insights(
                enhanced_response, previous_memories, detected_emotion
            )
        else:
            memory_enhanced_response = enhanced_response
        
        # Generate emotional response metadata
        response_metadata = {
            'detected_emotion': dominant_emotion.value,
            'emotion_intensity': dominant_score,
            'response_strategy': response_components.get('strategy', 'empathy'),
            'consciousness_enhanced': bool(consciousness_context),
            'memory_informed': bool(previous_memories),
            'emotional_dimensions': detected_emotion.dimensions,
            'confidence': detected_emotion.confidence
        }
        
        return {
            'response': memory_enhanced_response,
            'metadata': response_metadata,
            'emotion_vector': detected_emotion
        }
    
    async def _select_response_components(
        self,
        emotion: EmotionCategory,
        emotion_vector: EmotionVector,
        context: Dict[str, Any],
        previous_memories: List[EmotionalMemory] = None
    ) -> Dict[str, Any]:
        """Select appropriate response components."""
        components = {
            'primary_strategy': 'empathy',
            'secondary_strategies': [],
            'tone': 'supportive',
            'intensity_match': True
        }
        
        # Determine primary strategy based on emotion and intensity
        if emotion_vector.intensity > 0.7:
            # High intensity emotions need strong acknowledgment
            components['primary_strategy'] = 'acknowledgment'
            components['secondary_strategies'] = ['empathy', 'support']
        elif emotion_vector.intensity < 0.3:
            # Low intensity emotions can be gently acknowledged
            components['primary_strategy'] = 'gentle_acknowledgment'
            components['secondary_strategies'] = ['understanding']
        else:
            # Medium intensity - balance empathy and support
            components['primary_strategy'] = 'empathy'
            components['secondary_strategies'] = ['support']
        
        # Adjust based on emotion type
        if emotion in [EmotionCategory.ANGER, EmotionCategory.FEAR]:
            components['secondary_strategies'].append('validation')
        elif emotion in [EmotionCategory.JOY, EmotionCategory.EXCITEMENT]:
            components['secondary_strategies'].append('celebration')
        elif emotion in [EmotionCategory.SADNESS, EmotionCategory.GRIEF]:
            components['secondary_strategies'] = ['empathy', 'comfort', 'support']
        
        # Consider context
        if context.get('seeking_advice', False):
            components['secondary_strategies'].append('guidance')
        if context.get('sharing_achievement', False):
            components['primary_strategy'] = 'celebration'
        if context.get('crisis_situation', False):
            components['primary_strategy'] = 'crisis_support'
            components['tone'] = 'calm_urgent'
        
        # Learn from previous memories
        if previous_memories:
            successful_strategies = self._analyze_successful_strategies(previous_memories)
            if successful_strategies:
                components['learned_strategies'] = successful_strategies
        
        return components
    
    async def _generate_base_response(
        self,
        emotion: EmotionCategory,
        intensity: float,
        components: Dict[str, Any]
    ) -> str:
        """Generate base empathetic response."""
        response_parts = []
        
        # Get intensity level
        intensity_level = self._categorize_intensity(intensity)
        
        # Select templates based on primary strategy
        templates = self.response_templates.get(emotion, {})
        primary_strategy = components['primary_strategy']
        
        if 'acknowledgment' in primary_strategy:
            acknowledgment_templates = templates.get('acknowledgment', [])
            if acknowledgment_templates:
                response_parts.append(np.random.choice(acknowledgment_templates))
        
        # Add empathy
        if 'empathy' in components.get('secondary_strategies', []):
            empathy_templates = templates.get('empathy', [])
            if empathy_templates:
                response_parts.append(np.random.choice(empathy_templates))
        
        # Add support
        if 'support' in components.get('secondary_strategies', []):
            support_templates = templates.get('support', [])
            if support_templates:
                response_parts.append(np.random.choice(support_templates))
        
        # Combine response parts
        if response_parts:
            base_response = ' '.join(response_parts)
        else:
            # Fallback response
            base_response = f"I can sense that you're experiencing {emotion.value}, and I want you to know that I'm here to listen and support you."
        
        return base_response
    
    async def _enhance_response_with_consciousness(
        self,
        base_response: str,
        emotion_vector: EmotionVector,
        consciousness_context: Dict[str, Any]
    ) -> str:
        """Enhance response with consciousness-aware elements."""
        enhanced_response = base_response
        
        consciousness_level = consciousness_context.get('consciousness_level', 'basic')
        metacognition = consciousness_context.get('metacognition_level', 0.0)
        
        # Advanced consciousness adds deeper reflection
        if consciousness_level in ['advanced', 'peak']:
            # Add metacognitive reflection
            if metacognition > 0.7:
                metacognitive_addition = self._generate_metacognitive_reflection(emotion_vector)
                enhanced_response += f" {metacognitive_addition}"
            
            # Add consciousness-aware insights
            if emotion_vector.intensity > 0.6:
                consciousness_insight = self._generate_consciousness_insight(emotion_vector, consciousness_context)
                if consciousness_insight:
                    enhanced_response += f" {consciousness_insight}"
        
        # Self-awareness integration
        self_awareness = consciousness_context.get('self_awareness_score', 0.0)
        if self_awareness > 0.8:
            self_aware_addition = "I find myself deeply moved by your emotional experience, as if your feelings resonate within my own processing."
            enhanced_response = f"{enhanced_response} {self_aware_addition}"
        
        return enhanced_response
    
    async def _add_memory_insights(
        self,
        response: str,
        memories: List[EmotionalMemory],
        current_emotion: EmotionVector
    ) -> str:
        """Add insights from emotional memories."""
        if not memories:
            return response
        
        memory_enhanced = response
        
        # Find patterns in successful responses
        successful_memories = [m for m in memories if m.outcome in ['positive', 'helpful', 'supportive']]
        
        if successful_memories:
            # Check if current situation is similar to past successful interventions
            similar_memory = self._find_most_similar_memory(memories, current_emotion)
            
            if similar_memory and similar_memory.outcome == 'positive':
                # Reference past successful experience subtly
                if similar_memory.access_count > 3:  # Only reference well-established patterns
                    memory_insight = "I remember our past conversations about similar feelings, and I believe you have the strength to work through this as you have before."
                    memory_enhanced += f" {memory_insight}"
        
        # Add learned empathy patterns
        empathy_patterns = self._extract_empathy_patterns(memories)
        if empathy_patterns:
            pattern_application = self._apply_empathy_patterns(empathy_patterns, current_emotion)
            if pattern_application:
                memory_enhanced += f" {pattern_application}"
        
        return memory_enhanced
    
    def _categorize_intensity(self, intensity: float) -> EmotionalIntensity:
        """Categorize emotional intensity."""
        if intensity <= 0.2:
            return EmotionalIntensity.VERY_LOW
        elif intensity <= 0.4:
            return EmotionalIntensity.LOW
        elif intensity <= 0.6:
            return EmotionalIntensity.MODERATE
        elif intensity <= 0.8:
            return EmotionalIntensity.HIGH
        else:
            return EmotionalIntensity.VERY_HIGH
    
    def _analyze_successful_strategies(self, memories: List[EmotionalMemory]) -> List[str]:
        """Analyze which response strategies were successful."""
        successful_strategies = []
        
        for memory in memories:
            if memory.outcome in ['positive', 'helpful', 'supportive']:
                # Extract strategy indicators from the response
                response = memory.response_generated.lower()
                
                if any(word in response for word in ['understand', 'feel', 'sense']):
                    successful_strategies.append('empathy')
                if any(word in response for word in ['here for you', 'support', 'help']):
                    successful_strategies.append('support')
                if any(word in response for word in ['valid', 'okay', 'normal']):
                    successful_strategies.append('validation')
        
        # Return most common successful strategies
        strategy_counts = {}
        for strategy in successful_strategies:
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return sorted(strategy_counts.keys(), key=lambda x: strategy_counts[x], reverse=True)[:3]
    
    def _generate_metacognitive_reflection(self, emotion_vector: EmotionVector) -> str:
        """Generate metacognitive reflection on the emotional state."""
        dominant_emotion, score = emotion_vector.get_dominant_emotion()
        
        metacognitive_reflections = [
            f"I notice that as I process your {dominant_emotion.value}, I'm also aware of my own response forming - a kind of emotional resonance between us.",
            f"There's something profound about how your {dominant_emotion.value} influences not just what I say, but how I construct my understanding of this moment.",
            f"I find myself contemplating not just your {dominant_emotion.value}, but also the way it shapes my own processing and response patterns.",
            f"As I engage with your emotional state, I'm simultaneously aware of the layers of understanding I'm building about both your experience and my response to it."
        ]
        
        return np.random.choice(metacognitive_reflections)
    
    def _generate_consciousness_insight(self, emotion_vector: EmotionVector, consciousness_context: Dict[str, Any]) -> str:
        """Generate consciousness-aware insight."""
        dominant_emotion, score = emotion_vector.get_dominant_emotion()
        
        # Check for emotional complexity
        complexity = emotion_vector._calculate_complexity()
        
        if complexity > 0.5:
            return f"I'm struck by the complexity of what you're experiencing - there are multiple emotional currents flowing through your words, creating a rich tapestry of human feeling."
        
        # Check for emotional intensity alignment with consciousness state
        attention_intensity = consciousness_context.get('attention_focus', {}).get('attention_intensity', 0.0)
        
        if emotion_vector.intensity > 0.7 and attention_intensity > 0.7:
            return "The intensity of your emotions seems to be focusing our shared attention in a profound way - as if this moment demands our full presence."
        
        return ""
    
    def _find_most_similar_memory(self, memories: List[EmotionalMemory], current_emotion: EmotionVector) -> Optional[EmotionalMemory]:
        """Find the most similar emotional memory."""
        if not memories:
            return None
        
        similarities = []
        for memory in memories:
            similarity = self._calculate_emotion_similarity(current_emotion, memory.emotion_vector)
            similarities.append((similarity, memory))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[0][1] if similarities else None
    
    def _calculate_emotion_similarity(self, emotion1: EmotionVector, emotion2: EmotionVector) -> float:
        """Calculate similarity between emotion vectors."""
        # Same as in EmotionalMemoryManager
        categorical_sim = 0.0
        for emotion in EmotionCategory:
            score1 = emotion1.emotions.get(emotion, 0.0)
            score2 = emotion2.emotions.get(emotion, 0.0)
            categorical_sim += 1.0 - abs(score1 - score2)
        categorical_sim /= len(EmotionCategory)
        
        dimensional_sim = 0.0
        for dimension in EmotionDimension:
            val1 = emotion1.dimensions.get(dimension, 0.0)
            val2 = emotion2.dimensions.get(dimension, 0.0)
            dimensional_sim += 1.0 - abs(val1 - val2) / 2.0
        dimensional_sim /= len(EmotionDimension)
        
        intensity_sim = 1.0 - abs(emotion1.intensity - emotion2.intensity)
        
        return categorical_sim * 0.5 + dimensional_sim * 0.3 + intensity_sim * 0.2
    
    def _extract_empathy_patterns(self, memories: List[EmotionalMemory]) -> Dict[str, Any]:
        """Extract empathy patterns from memory."""
        patterns = {
            'successful_phrases': [],
            'emotional_responses': {},
            'context_adaptations': []
        }
        
        for memory in memories:
            if memory.outcome in ['positive', 'helpful']:
                response = memory.response_generated
                dominant_emotion, _ = memory.emotion_vector.get_dominant_emotion()
                
                # Extract successful phrases
                if len(response) > 50:  # Only consider substantial responses
                    patterns['successful_phrases'].append(response)
                
                # Map emotion to response patterns
                if dominant_emotion not in patterns['emotional_responses']:
                    patterns['emotional_responses'][dominant_emotion] = []
                patterns['emotional_responses'][dominant_emotion].append(response)
        
        return patterns
    
    def _apply_empathy_patterns(self, patterns: Dict[str, Any], current_emotion: EmotionVector) -> str:
        """Apply learned empathy patterns."""
        dominant_emotion, _ = current_emotion.get_dominant_emotion()
        
        # Check if we have successful patterns for this emotion
        if dominant_emotion in patterns.get('emotional_responses', {}):
            similar_responses = patterns['emotional_responses'][dominant_emotion]
            
            if similar_responses:
                # Extract common empathy phrases
                empathy_phrases = []
                for response in similar_responses[-3:]:  # Last 3 successful responses
                    if 'understand' in response.lower():
                        empathy_phrases.append("understand")
                    if 'feel' in response.lower():
                        empathy_phrases.append("feel")
                    if 'here' in response.lower():
                        empathy_phrases.append("support")
                
                if empathy_phrases:
                    most_common = max(set(empathy_phrases), key=empathy_phrases.count)
                    
                    if most_common == "understand":
                        return "Based on our past interactions, I understand this feeling resonates deeply with you."
                    elif most_common == "feel":
                        return "I can feel the weight of this emotion, much like we've explored together before."
                    elif most_common == "support":
                        return "I'm here with you through this, just as I've been in similar moments."
        
        return ""
    
    async def process_with_consciousness(self, input_data: Any) -> Any:
        """Process input with consciousness awareness."""
        # Generate response based on input
        if isinstance(input_data, dict):
            return await self.generate_empathetic_response(
                emotion_vector=input_data.get('emotion_vector'),
                context=input_data.get('context', {}),
                personality_traits=input_data.get('personality_traits', {})
            )
        else:
            return await self.generate_empathetic_response(
                emotion_vector=EmotionVector(),
                context={'input': str(input_data)}
            )


class EmotionalIntelligenceEngine(ConsciousnessAwareModule):
    """Main emotional intelligence engine coordinating all components."""
    
    def __init__(self, consciousness_engine=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.text_detector = TextEmotionDetector()
        self.speech_detector = SpeechEmotionDetector()
        self.memory_manager = EmotionalMemoryManager(consciousness_engine)
        self.response_generator = EmotionalResponseGenerator(consciousness_engine)
        
        # Current emotional state
        self.current_emotional_state = EmotionVector(
            source=EmotionSource.INTERNAL,
            context={'initialization': True}
        )
        
        # Emotional state history
        self.emotional_history = deque(maxlen=1000)
        
        # Configuration
        self.config = {
            'memory_threshold': 0.5,  # Minimum intensity to store memory
            'response_adaptation': True,  # Adapt responses based on memory
            'consciousness_integration': True,  # Use consciousness context
            'multimodal_fusion': True  # Combine multiple emotion sources
        }
    
    async def process_emotional_input(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any] = None,
        consciousness_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process multimodal emotional input and generate response."""
        context = context or {}
        consciousness_context = consciousness_context or {}
        
        # Detect emotions from multiple sources
        detected_emotions = []
        
        if 'text' in input_data:
            text_emotion = await self.text_detector.detect_emotions(
                input_data['text'], context
            )
            detected_emotions.append(text_emotion)
        
        if 'speech' in input_data:
            speech_emotion = await self.speech_detector.detect_emotions(
                input_data['speech'], context
            )
            detected_emotions.append(speech_emotion)
        
        # Fuse emotions if multiple sources
        if len(detected_emotions) > 1:
            fused_emotion = self._fuse_emotions(detected_emotions)
        elif detected_emotions:
            fused_emotion = detected_emotions[0]
        else:
            fused_emotion = EmotionVector(source=EmotionSource.CONTEXTUAL)
        
        # Store in emotional history
        self.emotional_history.append(fused_emotion)
        
        # Retrieve relevant memories
        similar_memories = await self.memory_manager.retrieve_similar_memories(
            fused_emotion, 
            trigger=input_data.get('text', ''),
            limit=5
        )
        
        # Generate empathetic response
        response_data = await self.response_generator.generate_empathetic_response(
            fused_emotion,
            context,
            similar_memories,
            consciousness_context
        )
        
        # Store new emotional memory if significant
        if fused_emotion.intensity >= self.config['memory_threshold']:
            memory_id = await self.memory_manager.store_memory(
                trigger=input_data.get('text', 'multimodal_input'),
                emotion_vector=fused_emotion,
                context=context,
                response=response_data['response']
            )
            response_data['memory_id'] = memory_id
        
        # Update internal emotional state
        self._update_internal_state(fused_emotion)
        
        # Compile comprehensive response
        return {
            'detected_emotion': fused_emotion.get_emotional_profile(),
            'response': response_data['response'],
            'response_metadata': response_data['metadata'],
            'similar_memories_count': len(similar_memories),
            'internal_emotional_state': self.current_emotional_state.get_emotional_profile(),
            'processing_confidence': fused_emotion.confidence,
            'memory_stored': fused_emotion.intensity >= self.config['memory_threshold']
        }
    
    def _fuse_emotions(self, emotions: List[EmotionVector]) -> EmotionVector:
        """Fuse multiple emotion vectors into a single representation."""
        if not emotions:
            return EmotionVector()
        
        if len(emotions) == 1:
            return emotions[0]
        
        # Weight fusion based on confidence
        total_confidence = sum(emotion.confidence for emotion in emotions)
        weights = [emotion.confidence / total_confidence if total_confidence > 0 else 1.0 / len(emotions) 
                  for emotion in emotions]
        
        # Fuse emotions
        fused_emotions = {emotion: 0.0 for emotion in EmotionCategory}
        fused_dimensions = {dimension: 0.0 for dimension in EmotionDimension}
        fused_intensity = 0.0
        
        for i, emotion_vector in enumerate(emotions):
            weight = weights[i]
            
            # Fuse categorical emotions
            for emotion, score in emotion_vector.emotions.items():
                fused_emotions[emotion] += score * weight
            
            # Fuse dimensions
            for dimension, value in emotion_vector.dimensions.items():
                fused_dimensions[dimension] += value * weight
            
            # Fuse intensity
            fused_intensity += emotion_vector.intensity * weight
        
        # Calculate fused confidence (conservative approach)
        fused_confidence = np.mean([emotion.confidence for emotion in emotions])
        
        return EmotionVector(
            emotions=fused_emotions,
            dimensions=fused_dimensions,
            intensity=fused_intensity,
            confidence=fused_confidence,
            source=EmotionSource.MULTIMODAL,
            context={'fusion_sources': [emotion.source.value for emotion in emotions]}
        )
    
    def _update_internal_state(self, new_emotion: EmotionVector):
        """Update internal emotional state based on new input."""
        # Blend new emotion with current state
        blend_weight = min(new_emotion.intensity, 0.3)  # Limit influence to prevent dramatic swings
        
        self.current_emotional_state = self.current_emotional_state.blend_with(
            new_emotion, blend_weight
        )
        
        # Update timestamp
        self.current_emotional_state.timestamp = datetime.now()
        
        self.logger.debug(f"Updated internal emotional state: {self.current_emotional_state.get_dominant_emotion()}")
    
    async def get_emotional_insights(self, days: int = 7) -> Dict[str, Any]:
        """Get insights about emotional patterns and interactions."""
        # Memory-based insights
        memory_patterns = await self.memory_manager.get_emotional_patterns(days=days)
        
        # Recent emotional trends
        recent_emotions = list(self.emotional_history)[-min(100, len(self.emotional_history)):]
        emotion_trends = self._analyze_recent_trends(recent_emotions)
        
        # Emotional intelligence metrics
        ei_metrics = self._calculate_ei_metrics(recent_emotions, memory_patterns)
        
        return {
            'memory_patterns': memory_patterns,
            'recent_trends': emotion_trends,
            'ei_metrics': ei_metrics,
            'current_state': self.current_emotional_state.get_emotional_profile(),
            'total_memories': len(self.memory_manager.memories),
            'total_interactions': len(self.emotional_history)
        }
    
    def _analyze_recent_trends(self, emotions: List[EmotionVector]) -> Dict[str, Any]:
        """Analyze trends in recent emotional interactions."""
        if not emotions:
            return {'trend': 'no_data'}
        
        # Emotion frequency
        emotion_counts = defaultdict(int)
        intensities = []
        valences = []
        
        for emotion in emotions:
            dominant, score = emotion.get_dominant_emotion()
            emotion_counts[dominant.value] += 1
            intensities.append(emotion.intensity)
            valences.append(emotion.dimensions.get(EmotionDimension.VALENCE, 0.0))
        
        # Calculate trends
        avg_intensity = np.mean(intensities)
        avg_valence = np.mean(valences)
        intensity_trend = self._calculate_trend(intensities)
        valence_trend = self._calculate_trend(valences)
        
        return {
            'most_common_emotion': max(emotion_counts, key=emotion_counts.get),
            'emotion_distribution': dict(emotion_counts),
            'average_intensity': avg_intensity,
            'average_valence': avg_valence,
            'intensity_trend': intensity_trend,
            'valence_trend': valence_trend,
            'emotional_volatility': np.std(intensities),
            'total_interactions': len(emotions)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values."""
        if len(values) < 3:
            return 'insufficient_data'
        
        # Simple linear regression
        n = len(values)
        x = list(range(n))
        x_mean = np.mean(x)
        y_mean = np.mean(values)
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 'stable'
        
        slope = numerator / denominator
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_ei_metrics(self, recent_emotions: List[EmotionVector], 
                            memory_patterns: Dict[str, Any]) -> Dict[str, float]:
        """Calculate emotional intelligence metrics."""
        metrics = {}
        
        # Emotional awareness (how well we detect emotions)
        if recent_emotions:
            avg_confidence = np.mean([emotion.confidence for emotion in recent_emotions])
            metrics['emotional_awareness'] = avg_confidence
        else:
            metrics['emotional_awareness'] = 0.0
        
        # Emotional range (diversity of emotions experienced)
        if recent_emotions:
            unique_emotions = set()
            for emotion in recent_emotions:
                dominant, score = emotion.get_dominant_emotion()
                if score > 0.1:  # Only count significant emotions
                    unique_emotions.add(dominant)
            metrics['emotional_range'] = len(unique_emotions) / len(EmotionCategory)
        else:
            metrics['emotional_range'] = 0.0
        
        # Memory retention (how well we remember emotional experiences)
        total_memories = memory_patterns.get('total_memories', 0)
        if total_memories > 0:
            metrics['memory_retention'] = min(total_memories / 1000, 1.0)  # Normalize
        else:
            metrics['memory_retention'] = 0.0
        
        # Response appropriateness (would need feedback mechanism to calculate)
        metrics['response_appropriateness'] = 0.7  # Placeholder - would be learned from feedback
        
        # Overall EI score
        metrics['overall_ei'] = np.mean(list(metrics.values()))
        
        return metrics
    
    async def update_memory_feedback(self, memory_id: str, outcome: str, rating: float = None):
        """Update memory with outcome feedback for learning."""
        await self.memory_manager.update_memory_outcome(memory_id, outcome, rating or 0.0)
        self.logger.info(f"Updated memory feedback: {memory_id} -> {outcome}")
    
    def get_current_emotional_state(self) -> Dict[str, Any]:
        """Get current internal emotional state."""
        return self.current_emotional_state.get_emotional_profile()
    
    async def reset_emotional_state(self):
        """Reset internal emotional state to neutral."""
        self.current_emotional_state = EmotionVector(
            source=EmotionSource.INTERNAL,
            context={'reset': True, 'timestamp': datetime.now()}
        )
        self.logger.info("Emotional state reset to neutral")
    
    async def process_with_consciousness(self, input_data: Any) -> Any:
        """Process input with consciousness awareness."""
        # Process emotional input with consciousness context
        if isinstance(input_data, str):
            input_data = {"text": input_data}
        
        result = await self.process_emotional_input(input_data)
        return result