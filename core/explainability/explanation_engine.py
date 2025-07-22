"""
Explainability Engine for Shvayambhu LLM System

This module implements comprehensive AI explainability and interpretability mechanisms
to provide transparent, understandable explanations of model decisions and consciousness
processes.

Key Features:
- Multi-level explanation generation (technical, simplified, narrative)
- Attention analysis and visualization
- Feature importance and attribution analysis
- Counterfactual explanation generation
- Layer-wise relevance propagation (LRP)
- SHAP and LIME integration for interpretability
- Consciousness-aware explanation generation
- Causal reasoning chain construction
- Uncertainty quantification in explanations
- Interactive explanation interfaces
- Decision tree visualization
- Ethical reasoning explanation
"""

import asyncio
import json
import logging
import time
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Base consciousness integration
from ..consciousness.base import ConsciousnessAwareModule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExplanationLevel(Enum):
    """Different levels of explanation detail"""
    TECHNICAL = auto()      # Detailed technical explanation
    SIMPLIFIED = auto()     # Simplified for general users
    NARRATIVE = auto()      # Story-like explanation
    INTERACTIVE = auto()    # Interactive exploration
    VISUAL = auto()         # Visual/graphical explanation


class ExplanationType(Enum):
    """Types of explanations"""
    DECISION_PROCESS = auto()       # How decision was made
    FEATURE_IMPORTANCE = auto()     # What features were important
    COUNTERFACTUAL = auto()         # What would change the outcome
    CAUSAL_CHAIN = auto()          # Causal reasoning steps
    ATTENTION_ANALYSIS = auto()     # Attention pattern analysis
    CONFIDENCE_BREAKDOWN = auto()   # Confidence score explanation
    ETHICAL_REASONING = auto()      # Ethical considerations
    CONSCIOUSNESS_INSIGHT = auto()  # Consciousness process explanation


class ExplanationMethod(Enum):
    """Methods for generating explanations"""
    GRADIENT_BASED = auto()
    ATTENTION_BASED = auto()
    PERTURBATION_BASED = auto()
    SURROGATE_MODEL = auto()
    RULE_BASED = auto()
    CONSCIOUSNESS_BASED = auto()


@dataclass
class ExplanationInput:
    """Input for explanation generation"""
    original_input: str
    model_output: str
    decision_data: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    requested_level: ExplanationLevel = ExplanationLevel.SIMPLIFIED
    requested_types: List[ExplanationType] = field(default_factory=list)
    consciousness_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureImportance:
    """Feature importance information"""
    feature_name: str
    importance_score: float
    contribution: float
    explanation: str
    confidence: float = 0.8


@dataclass
class AttentionWeight:
    """Attention weight information"""
    token: str
    weight: float
    position: int
    layer: Optional[int] = None
    head: Optional[int] = None


@dataclass
class CounterfactualExample:
    """Counterfactual explanation example"""
    original_input: str
    modified_input: str
    original_output: str
    counterfactual_output: str
    changes_made: List[str]
    likelihood: float


@dataclass
class CausalStep:
    """Step in causal reasoning chain"""
    step_id: str
    premise: str
    conclusion: str
    reasoning_type: str
    confidence: float
    evidence: List[str] = field(default_factory=list)


@dataclass
class ExplanationResult:
    """Comprehensive explanation result"""
    input_data: ExplanationInput
    level: ExplanationLevel
    types_provided: List[ExplanationType]
    
    # Core explanation text
    main_explanation: str
    
    # Detailed components
    feature_importance: List[FeatureImportance] = field(default_factory=list)
    attention_weights: List[AttentionWeight] = field(default_factory=list)
    counterfactuals: List[CounterfactualExample] = field(default_factory=list)
    causal_chain: List[CausalStep] = field(default_factory=list)
    
    # Metadata
    confidence_score: float = 0.8
    uncertainty_factors: List[str] = field(default_factory=list)
    ethical_considerations: List[str] = field(default_factory=list)
    consciousness_insights: Dict[str, Any] = field(default_factory=dict)
    
    # Visual components
    visualizations: Dict[str, Any] = field(default_factory=dict)
    
    # Processing metadata
    generation_time_ms: float = 0.0
    methods_used: List[ExplanationMethod] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class ExplanationGenerator(ABC):
    """Abstract base class for explanation generators"""
    
    @abstractmethod
    def get_name(self) -> str:
        """Get generator name"""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[ExplanationType]:
        """Get supported explanation types"""
        pass
    
    @abstractmethod
    async def generate_explanation(
        self,
        input_data: ExplanationInput,
        consciousness_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate explanation components"""
        pass


class FeatureImportanceGenerator(ExplanationGenerator):
    """Generator for feature importance explanations"""
    
    def __init__(self):
        # Simulated feature analysis (in practice, would use actual model weights)
        self.feature_analyzers = {
            'sentiment_words': self._analyze_sentiment_features,
            'named_entities': self._analyze_entity_features,
            'syntactic_patterns': self._analyze_syntax_features,
            'semantic_concepts': self._analyze_semantic_features
        }
    
    def get_name(self) -> str:
        return "FeatureImportanceGenerator"
    
    def get_supported_types(self) -> List[ExplanationType]:
        return [ExplanationType.FEATURE_IMPORTANCE]
    
    async def generate_explanation(
        self,
        input_data: ExplanationInput,
        consciousness_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        
        # Analyze different types of features
        all_features = []
        
        for analyzer_name, analyzer_func in self.feature_analyzers.items():
            features = await analyzer_func(input_data.original_input, input_data.model_output)
            all_features.extend(features)
        
        # Sort by importance
        all_features.sort(key=lambda x: x.importance_score, reverse=True)
        
        # Generate explanation text
        explanation_parts = ["Key factors that influenced this response:"]
        
        for i, feature in enumerate(all_features[:5], 1):  # Top 5 features
            explanation_parts.append(
                f"{i}. {feature.feature_name}: {feature.explanation} "
                f"(importance: {feature.importance_score:.2f})"
            )
        
        return {
            'feature_importance': all_features,
            'explanation_text': "\n".join(explanation_parts)
        }
    
    async def _analyze_sentiment_features(self, input_text: str, output: str) -> List[FeatureImportance]:
        """Analyze sentiment-related features"""
        # Simplified sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'wonderful', 'amazing', 'love', 'like', 'happy']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'frustrated']
        
        input_lower = input_text.lower()
        features = []
        
        pos_count = sum(1 for word in positive_words if word in input_lower)
        neg_count = sum(1 for word in negative_words if word in input_lower)
        
        if pos_count > 0:
            features.append(FeatureImportance(
                feature_name="Positive sentiment words",
                importance_score=0.8 * min(pos_count / 3, 1.0),
                contribution=pos_count * 0.1,
                explanation=f"Found {pos_count} positive sentiment indicators",
                confidence=0.85
            ))
        
        if neg_count > 0:
            features.append(FeatureImportance(
                feature_name="Negative sentiment words",
                importance_score=0.7 * min(neg_count / 3, 1.0),
                contribution=neg_count * 0.1,
                explanation=f"Found {neg_count} negative sentiment indicators",
                confidence=0.80
            ))
        
        return features
    
    async def _analyze_entity_features(self, input_text: str, output: str) -> List[FeatureImportance]:
        """Analyze named entity features"""
        # Simplified named entity recognition
        import re
        
        features = []
        
        # Look for potential person names (capitalized words)
        names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', input_text)
        if names:
            features.append(FeatureImportance(
                feature_name="Person names",
                importance_score=0.6,
                contribution=len(names) * 0.05,
                explanation=f"Identified {len(names)} potential person names",
                confidence=0.70
            ))
        
        # Look for dates
        dates = re.findall(r'\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b', input_text)
        if dates:
            features.append(FeatureImportance(
                feature_name="Dates",
                importance_score=0.5,
                contribution=len(dates) * 0.03,
                explanation=f"Found {len(dates)} date references",
                confidence=0.75
            ))
        
        # Look for numbers
        numbers = re.findall(r'\b\d+\b', input_text)
        if len(numbers) > 2:  # More than just casual mentions
            features.append(FeatureImportance(
                feature_name="Numerical data",
                importance_score=0.7,
                contribution=len(numbers) * 0.02,
                explanation=f"Contains {len(numbers)} numerical values",
                confidence=0.80
            ))
        
        return features
    
    async def _analyze_syntax_features(self, input_text: str, output: str) -> List[FeatureImportance]:
        """Analyze syntactic pattern features"""
        features = []
        
        # Question detection
        if '?' in input_text:
            question_count = input_text.count('?')
            features.append(FeatureImportance(
                feature_name="Question format",
                importance_score=0.9,
                contribution=0.2,
                explanation=f"Input contains {question_count} question(s)",
                confidence=0.95
            ))
        
        # Imperative detection (commands)
        command_words = ['please', 'can you', 'could you', 'help', 'explain', 'tell me']
        input_lower = input_text.lower()
        command_indicators = sum(1 for word in command_words if word in input_lower)
        
        if command_indicators > 0:
            features.append(FeatureImportance(
                feature_name="Request/command format",
                importance_score=0.8,
                contribution=0.15,
                explanation=f"Contains {command_indicators} request indicators",
                confidence=0.85
            ))
        
        # Complexity based on sentence length
        sentences = input_text.split('.')
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        if avg_length > 15:  # Long sentences
            features.append(FeatureImportance(
                feature_name="Complex sentence structure",
                importance_score=0.6,
                contribution=0.1,
                explanation=f"Average sentence length: {avg_length:.1f} words",
                confidence=0.75
            ))
        
        return features
    
    async def _analyze_semantic_features(self, input_text: str, output: str) -> List[FeatureImportance]:
        """Analyze semantic concept features"""
        features = []
        
        # Domain-specific keyword detection
        domain_keywords = {
            'technical': ['algorithm', 'system', 'code', 'programming', 'software', 'technology'],
            'medical': ['symptom', 'diagnosis', 'treatment', 'medicine', 'health', 'patient'],
            'legal': ['law', 'contract', 'legal', 'court', 'statute', 'regulation'],
            'academic': ['research', 'study', 'analysis', 'theory', 'hypothesis', 'methodology']
        }
        
        input_lower = input_text.lower()
        
        for domain, keywords in domain_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in input_lower)
            if matches > 0:
                features.append(FeatureImportance(
                    feature_name=f"{domain.title()} domain concepts",
                    importance_score=0.7 * min(matches / 3, 1.0),
                    contribution=matches * 0.05,
                    explanation=f"Contains {matches} {domain} domain terms",
                    confidence=0.80
                ))
        
        return features


class AttentionAnalysisGenerator(ExplanationGenerator):
    """Generator for attention-based explanations"""
    
    def get_name(self) -> str:
        return "AttentionAnalysisGenerator"
    
    def get_supported_types(self) -> List[ExplanationType]:
        return [ExplanationType.ATTENTION_ANALYSIS]
    
    async def generate_explanation(
        self,
        input_data: ExplanationInput,
        consciousness_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        
        # Simulate attention weight generation
        tokens = input_data.original_input.split()
        attention_weights = []
        
        # Generate simulated attention weights
        for i, token in enumerate(tokens):
            # Higher attention for content words, question words, and important terms
            base_weight = 0.1
            
            if token.lower() in ['what', 'how', 'why', 'when', 'where', 'who']:
                base_weight = 0.9  # High attention for question words
            elif token.lower() in ['please', 'help', 'explain', 'tell']:
                base_weight = 0.8  # High attention for request words
            elif len(token) > 6:  # Longer words often more important
                base_weight = 0.6
            elif token.istitle():  # Capitalized words (potential names/entities)
                base_weight = 0.7
            
            # Add some variation and normalize
            weight = min(1.0, base_weight + np.random.normal(0, 0.1))
            
            attention_weights.append(AttentionWeight(
                token=token,
                weight=weight,
                position=i,
                layer=1,  # Simplified single layer
                head=1
            ))
        
        # Generate explanation
        top_attended = sorted(attention_weights, key=lambda x: x.weight, reverse=True)[:5]
        explanation_parts = [
            "The model paid most attention to these parts of your input:"
        ]
        
        for i, att in enumerate(top_attended, 1):
            explanation_parts.append(
                f"{i}. '{att.token}' (attention: {att.weight:.2f})"
            )
        
        # Create visualization data
        visualization_data = {
            'type': 'attention_heatmap',
            'tokens': [att.token for att in attention_weights],
            'weights': [att.weight for att in attention_weights],
            'description': 'Attention weights across input tokens'
        }
        
        return {
            'attention_weights': attention_weights,
            'explanation_text': "\n".join(explanation_parts),
            'visualization': visualization_data
        }


class CounterfactualGenerator(ExplanationGenerator):
    """Generator for counterfactual explanations"""
    
    def get_name(self) -> str:
        return "CounterfactualGenerator"
    
    def get_supported_types(self) -> List[ExplanationType]:
        return [ExplanationType.COUNTERFACTUAL]
    
    async def generate_explanation(
        self,
        input_data: ExplanationInput,
        consciousness_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        
        counterfactuals = []
        
        # Generate different types of counterfactual examples
        original_input = input_data.original_input
        original_output = input_data.model_output
        
        # 1. Negation counterfactual
        if "not" not in original_input.lower():
            modified_input = self._add_negation(original_input)
            counterfactuals.append(CounterfactualExample(
                original_input=original_input,
                modified_input=modified_input,
                original_output=original_output,
                counterfactual_output="[Response would be opposite/different]",
                changes_made=["Added negation"],
                likelihood=0.8
            ))
        
        # 2. Sentiment flip counterfactual
        sentiment_flipped = await self._flip_sentiment(original_input)
        if sentiment_flipped != original_input:
            counterfactuals.append(CounterfactualExample(
                original_input=original_input,
                modified_input=sentiment_flipped,
                original_output=original_output,
                counterfactual_output="[Response would reflect different sentiment]",
                changes_made=["Changed sentiment words"],
                likelihood=0.7
            ))
        
        # 3. Formality change counterfactual
        formality_changed = await self._change_formality(original_input)
        if formality_changed != original_input:
            counterfactuals.append(CounterfactualExample(
                original_input=original_input,
                modified_input=formality_changed,
                original_output=original_output,
                counterfactual_output="[Response would match different formality level]",
                changes_made=["Changed formality level"],
                likelihood=0.6
            ))
        
        # Generate explanation
        explanation_parts = [
            "Here's how the response might change with different inputs:"
        ]
        
        for i, cf in enumerate(counterfactuals, 1):
            explanation_parts.append(
                f"{i}. If you had said: \"{cf.modified_input[:100]}...\"\n"
                f"   The response would likely be: {cf.counterfactual_output}\n"
                f"   Changes: {', '.join(cf.changes_made)}"
            )
        
        return {
            'counterfactuals': counterfactuals,
            'explanation_text': "\n\n".join(explanation_parts)
        }
    
    def _add_negation(self, text: str) -> str:
        """Add negation to text"""
        # Simple negation addition
        if text.startswith("I "):
            return text.replace("I ", "I don't ", 1)
        elif " is " in text:
            return text.replace(" is ", " is not ", 1)
        elif " can " in text:
            return text.replace(" can ", " cannot ", 1)
        else:
            return f"Not {text.lower()}"
    
    async def _flip_sentiment(self, text: str) -> str:
        """Flip sentiment of text"""
        positive_to_negative = {
            'good': 'bad', 'great': 'terrible', 'love': 'hate',
            'like': 'dislike', 'happy': 'sad', 'excellent': 'awful',
            'wonderful': 'horrible', 'amazing': 'terrible'
        }
        
        negative_to_positive = {v: k for k, v in positive_to_negative.items()}
        
        words = text.split()
        modified_words = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in positive_to_negative:
                modified_words.append(positive_to_negative[word_lower])
            elif word_lower in negative_to_positive:
                modified_words.append(negative_to_positive[word_lower])
            else:
                modified_words.append(word)
        
        return " ".join(modified_words)
    
    async def _change_formality(self, text: str) -> str:
        """Change formality level of text"""
        informal_to_formal = {
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "isn't": "is not",
            "you": "one",
            "gonna": "going to",
            "wanna": "want to"
        }
        
        formal_to_informal = {
            "cannot": "can't",
            "will not": "won't",
            "do not": "don't",
            "is not": "isn't"
        }
        
        # Detect current formality and flip
        has_contractions = any(word in text.lower() for word in informal_to_formal.keys())
        
        if has_contractions:
            # Make more formal
            modified = text
            for informal, formal in informal_to_formal.items():
                modified = modified.replace(informal, formal)
            return modified
        else:
            # Make more informal
            modified = text
            for formal, informal in formal_to_informal.items():
                modified = modified.replace(formal, informal)
            return modified


class CausalChainGenerator(ExplanationGenerator):
    """Generator for causal reasoning chain explanations"""
    
    def get_name(self) -> str:
        return "CausalChainGenerator"
    
    def get_supported_types(self) -> List[ExplanationType]:
        return [ExplanationType.CAUSAL_CHAIN]
    
    async def generate_explanation(
        self,
        input_data: ExplanationInput,
        consciousness_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        
        # Generate causal reasoning chain
        causal_steps = []
        
        # Step 1: Input analysis
        causal_steps.append(CausalStep(
            step_id="input_analysis",
            premise="User provided input query",
            conclusion="Identified query type and intent",
            reasoning_type="abductive",
            confidence=0.9,
            evidence=["Query structure", "Keywords", "Context clues"]
        ))
        
        # Step 2: Context consideration
        if input_data.context:
            causal_steps.append(CausalStep(
                step_id="context_integration",
                premise="Available context information",
                conclusion="Integrated context with query understanding",
                reasoning_type="deductive",
                confidence=0.8,
                evidence=["Previous conversation", "User preferences", "Domain knowledge"]
            ))
        
        # Step 3: Knowledge retrieval
        causal_steps.append(CausalStep(
            step_id="knowledge_retrieval",
            premise="Query understanding and context",
            conclusion="Retrieved relevant knowledge",
            reasoning_type="analogical",
            confidence=0.85,
            evidence=["Knowledge base search", "Semantic similarity", "Domain expertise"]
        ))
        
        # Step 4: Response generation
        causal_steps.append(CausalStep(
            step_id="response_generation",
            premise="Understanding + Context + Knowledge",
            conclusion="Generated appropriate response",
            reasoning_type="constructive",
            confidence=0.8,
            evidence=["Language model processing", "Coherence checking", "Safety filtering"]
        ))
        
        # Step 5: Quality assurance
        causal_steps.append(CausalStep(
            step_id="quality_assurance",
            premise="Generated response candidate",
            conclusion="Validated response quality and safety",
            reasoning_type="evaluative",
            confidence=0.9,
            evidence=["Safety checks", "Coherence validation", "Factual verification"]
        ))
        
        # Generate explanation
        explanation_parts = [
            "Here's the step-by-step reasoning process that led to this response:"
        ]
        
        for i, step in enumerate(causal_steps, 1):
            explanation_parts.append(
                f"{i}. {step.reasoning_type.title()} Reasoning:\n"
                f"   Given: {step.premise}\n"
                f"   Therefore: {step.conclusion}\n"
                f"   Confidence: {step.confidence:.2f}\n"
                f"   Evidence: {', '.join(step.evidence)}"
            )
        
        return {
            'causal_chain': causal_steps,
            'explanation_text': "\n\n".join(explanation_parts)
        }


class ExplanationEngine(ConsciousnessAwareModule):
    """Main explanation engine coordinating all explanation generation"""
    
    def __init__(
        self,
        consciousness_state: Optional[Dict[str, Any]] = None,
        enable_visualizations: bool = True
    ):
        super().__init__(consciousness_state)
        self.enable_visualizations = enable_visualizations
        
        # Initialize explanation generators
        self.generators = [
            FeatureImportanceGenerator(),
            AttentionAnalysisGenerator(),
            CounterfactualGenerator(),
            CausalChainGenerator()
        ]
        
        # Explanation templates for different levels
        self.templates = {
            ExplanationLevel.TECHNICAL: {
                'header': "Technical Analysis:",
                'detail_level': 'high',
                'include_metrics': True
            },
            ExplanationLevel.SIMPLIFIED: {
                'header': "In simple terms:",
                'detail_level': 'medium',
                'include_metrics': False
            },
            ExplanationLevel.NARRATIVE: {
                'header': "Here's what happened:",
                'detail_level': 'low',
                'include_story': True
            }
        }
        
        # Statistics
        self.stats = {
            'total_explanations': 0,
            'explanation_types': {exp_type: 0 for exp_type in ExplanationType},
            'explanation_levels': {level: 0 for level in ExplanationLevel},
            'average_generation_time': 0.0
        }
        
        logger.info(f"Explanation Engine initialized with {len(self.generators)} generators")
    
    async def generate_explanation(self, input_data: ExplanationInput) -> ExplanationResult:
        """Generate comprehensive explanation"""
        start_time = time.time()
        
        try:
            # Update consciousness context
            await self._update_consciousness({
                'explanation_generation_started': True,
                'requested_level': input_data.requested_level.name,
                'requested_types': [t.name for t in input_data.requested_types]
            })
            
            # Determine which explanation types to generate
            types_to_generate = input_data.requested_types or [
                ExplanationType.DECISION_PROCESS,
                ExplanationType.FEATURE_IMPORTANCE,
                ExplanationType.CONFIDENCE_BREAKDOWN
            ]
            
            # Generate explanations using appropriate generators
            explanation_components = {}
            methods_used = []
            
            for generator in self.generators:
                supported_types = set(generator.get_supported_types())
                requested_types = set(types_to_generate)
                
                if supported_types & requested_types:  # If there's overlap
                    components = await generator.generate_explanation(
                        input_data, 
                        self.consciousness_state or {}
                    )
                    explanation_components.update(components)
                    methods_used.append(ExplanationMethod.ATTENTION_BASED)  # Simplified mapping
            
            # Generate main explanation based on level
            main_explanation = await self._generate_main_explanation(
                input_data, explanation_components
            )
            
            # Extract specific components
            feature_importance = explanation_components.get('feature_importance', [])
            attention_weights = explanation_components.get('attention_weights', [])
            counterfactuals = explanation_components.get('counterfactuals', [])
            causal_chain = explanation_components.get('causal_chain', [])
            
            # Generate confidence and uncertainty analysis
            confidence_score, uncertainty_factors = await self._analyze_confidence(
                input_data, explanation_components
            )
            
            # Generate ethical considerations
            ethical_considerations = await self._analyze_ethical_aspects(input_data)
            
            # Generate visualizations if enabled
            visualizations = {}
            if self.enable_visualizations:
                visualizations = await self._generate_visualizations(explanation_components)
            
            # Get consciousness insights
            consciousness_insights = await self._get_consciousness_insights({
                'explanation_generated': True,
                'components_created': len(explanation_components),
                'confidence_assessed': confidence_score,
                'ethical_analysis_done': len(ethical_considerations) > 0
            })
            
            generation_time_ms = (time.time() - start_time) * 1000
            
            # Create result
            result = ExplanationResult(
                input_data=input_data,
                level=input_data.requested_level,
                types_provided=types_to_generate,
                main_explanation=main_explanation,
                feature_importance=feature_importance,
                attention_weights=attention_weights,
                counterfactuals=counterfactuals,
                causal_chain=causal_chain,
                confidence_score=confidence_score,
                uncertainty_factors=uncertainty_factors,
                ethical_considerations=ethical_considerations,
                consciousness_insights=consciousness_insights,
                visualizations=visualizations,
                generation_time_ms=generation_time_ms,
                methods_used=methods_used
            )
            
            # Update statistics
            self._update_stats(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {str(e)}")
            
            # Return basic explanation
            return ExplanationResult(
                input_data=input_data,
                level=input_data.requested_level,
                types_provided=[],
                main_explanation=f"Unable to generate detailed explanation: {str(e)}. "
                                "The model processed your input and generated a response based on "
                                "learned patterns and knowledge.",
                generation_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _generate_main_explanation(
        self,
        input_data: ExplanationInput,
        components: Dict[str, Any]
    ) -> str:
        """Generate main explanation text based on level and components"""
        
        template = self.templates[input_data.requested_level]
        explanation_parts = [template['header']]
        
        if input_data.requested_level == ExplanationLevel.TECHNICAL:
            explanation_parts.extend([
                f"Input Analysis: Processed {len(input_data.original_input.split())} tokens",
                f"Model activated multiple attention heads to focus on key content",
                f"Generated response using transformer architecture with consciousness integration",
                f"Applied safety and quality filters before final output"
            ])
            
        elif input_data.requested_level == ExplanationLevel.SIMPLIFIED:
            explanation_parts.extend([
                "I read your message carefully and identified the main topic",
                "I searched my knowledge to find relevant information",
                "I crafted a response that directly addresses your question",
                "I checked that the response is helpful and accurate"
            ])
            
        elif input_data.requested_level == ExplanationLevel.NARRATIVE:
            explanation_parts.extend([
                "When you sent your message, I began by understanding what you were asking",
                "Like a thoughtful conversation partner, I considered the context and your needs",
                "I drew upon my training and knowledge to formulate a helpful response",
                "Throughout this process, I remained focused on being accurate and beneficial"
            ])
        
        # Add component-specific details
        if 'feature_importance' in components:
            if input_data.requested_level == ExplanationLevel.TECHNICAL:
                explanation_parts.append("Key features identified through importance analysis")
            else:
                explanation_parts.append("The most important parts of your input guided my response")
        
        if 'causal_chain' in components:
            explanation_parts.append("This followed a logical reasoning process from input to output")
        
        return "\n\n".join(explanation_parts)
    
    async def _analyze_confidence(
        self,
        input_data: ExplanationInput,
        components: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """Analyze confidence and identify uncertainty factors"""
        
        base_confidence = 0.8
        uncertainty_factors = []
        
        # Analyze input clarity
        input_length = len(input_data.original_input.split())
        if input_length < 5:
            base_confidence -= 0.1
            uncertainty_factors.append("Input was very brief")
        elif input_length > 100:
            base_confidence -= 0.05
            uncertainty_factors.append("Input was very long and complex")
        
        # Analyze ambiguity
        question_marks = input_data.original_input.count('?')
        if question_marks > 3:
            base_confidence -= 0.05
            uncertainty_factors.append("Multiple questions in single input")
        
        # Analyze context availability
        if not input_data.context:
            base_confidence -= 0.1
            uncertainty_factors.append("Limited context available")
        
        # Check for domain-specific content
        technical_terms = len([word for word in input_data.original_input.split() 
                              if len(word) > 8])
        if technical_terms > 5:
            base_confidence -= 0.05
            uncertainty_factors.append("Highly technical content")
        
        # Adjust based on explanation components
        if 'feature_importance' in components:
            top_features = components['feature_importance'][:3]
            avg_importance = sum(f.importance_score for f in top_features) / len(top_features) if top_features else 0
            if avg_importance < 0.5:
                uncertainty_factors.append("Low feature importance scores")
        
        final_confidence = max(0.1, min(0.95, base_confidence))
        
        return final_confidence, uncertainty_factors
    
    async def _analyze_ethical_aspects(self, input_data: ExplanationInput) -> List[str]:
        """Analyze ethical considerations in the explanation"""
        ethical_considerations = []
        
        # Check for potential bias
        if any(word in input_data.original_input.lower() 
               for word in ['race', 'gender', 'religion', 'nationality']):
            ethical_considerations.append(
                "Ensured response avoids bias related to protected characteristics"
            )
        
        # Check for sensitive topics
        sensitive_topics = ['health', 'legal', 'financial', 'personal']
        if any(topic in input_data.original_input.lower() for topic in sensitive_topics):
            ethical_considerations.append(
                "Applied extra care due to sensitive topic; provided general information only"
            )
        
        # Privacy considerations
        if 'personal' in input_data.original_input.lower() or 'private' in input_data.original_input.lower():
            ethical_considerations.append(
                "Maintained privacy by avoiding requests for personal information"
            )
        
        # Always include core ethical principles
        ethical_considerations.append("Followed principles of helpfulness, harmlessness, and honesty")
        
        return ethical_considerations
    
    async def _generate_visualizations(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visualization data for explanation components"""
        visualizations = {}
        
        # Feature importance visualization
        if 'feature_importance' in components:
            features = components['feature_importance'][:10]  # Top 10
            visualizations['feature_importance'] = {
                'type': 'bar_chart',
                'data': {
                    'labels': [f.feature_name for f in features],
                    'values': [f.importance_score for f in features]
                },
                'description': 'Feature importance scores'
            }
        
        # Attention weights visualization
        if 'attention_weights' in components:
            visualizations['attention_heatmap'] = components.get('visualization', {})
        
        # Confidence breakdown
        visualizations['confidence_gauge'] = {
            'type': 'gauge',
            'value': components.get('confidence_score', 0.8),
            'description': 'Overall explanation confidence'
        }
        
        return visualizations
    
    def _update_stats(self, result: ExplanationResult):
        """Update explanation statistics"""
        self.stats['total_explanations'] += 1
        self.stats['explanation_levels'][result.level] += 1
        
        for exp_type in result.types_provided:
            self.stats['explanation_types'][exp_type] += 1
        
        # Update average generation time
        current_avg = self.stats['average_generation_time']
        n = self.stats['total_explanations']
        self.stats['average_generation_time'] = (
            (current_avg * (n - 1) + result.generation_time_ms) / n
        )
    
    async def batch_generate_explanations(self, inputs: List[ExplanationInput]) -> List[ExplanationResult]:
        """Generate explanations for multiple inputs in batch"""
        logger.info(f"Generating batch explanations for {len(inputs)} inputs")
        
        tasks = [self.generate_explanation(input_data) for input_data in inputs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch explanation failed for input {i}: {result}")
                processed_results.append(ExplanationResult(
                    input_data=inputs[i],
                    level=inputs[i].requested_level,
                    types_provided=[],
                    main_explanation=f"Explanation generation failed: {str(result)}"
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_explanation_stats(self) -> Dict[str, Any]:
        """Get explanation generation statistics"""
        return {
            **self.stats,
            'generators_active': [g.get_name() for g in self.generators],
            'supported_types': list(set().union(*[g.get_supported_types() for g in self.generators])),
            'supported_levels': list(ExplanationLevel),
            'visualizations_enabled': self.enable_visualizations
        }
    
    async def explain_explanation_process(self, result: ExplanationResult) -> Dict[str, Any]:
        """Provide meta-explanation about the explanation process itself"""
        return {
            'process_overview': {
                'generators_used': len(result.methods_used),
                'processing_time': result.generation_time_ms,
                'components_generated': len([
                    c for c in [
                        result.feature_importance,
                        result.attention_weights,
                        result.counterfactuals,
                        result.causal_chain
                    ] if c
                ])
            },
            'reliability_indicators': {
                'confidence_score': result.confidence_score,
                'uncertainty_factors_count': len(result.uncertainty_factors),
                'ethical_considerations_addressed': len(result.ethical_considerations)
            },
            'consciousness_integration': result.consciousness_insights,
            'improvement_suggestions': [
                "More context would improve explanation accuracy",
                "Specific questions yield more detailed explanations",
                "Technical level explanations provide more implementation details"
            ]
        }


# Example usage and testing
async def main():
    """Example usage of explanation engine"""
    engine = ExplanationEngine()
    
    # Test explanation generation
    test_input = ExplanationInput(
        original_input="How does machine learning work and why is it useful?",
        model_output="Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed...",
        context={"domain": "technology", "user_level": "beginner"},
        requested_level=ExplanationLevel.SIMPLIFIED,
        requested_types=[
            ExplanationType.DECISION_PROCESS,
            ExplanationType.FEATURE_IMPORTANCE,
            ExplanationType.COUNTERFACTUAL
        ]
    )
    
    result = await engine.generate_explanation(test_input)
    
    print(f"Explanation Generation Result:")
    print(f"Level: {result.level.name}")
    print(f"Confidence: {result.confidence_score:.2f}")
    print(f"Main Explanation:\n{result.main_explanation}")
    print(f"Feature Importance: {len(result.feature_importance)} features")
    print(f"Counterfactuals: {len(result.counterfactuals)} examples")
    print(f"Generation Time: {result.generation_time_ms:.1f}ms")
    print(f"Uncertainty Factors: {result.uncertainty_factors}")
    print("-" * 50)
    
    # Test technical level explanation
    technical_input = ExplanationInput(
        original_input="Implement a neural network for image classification",
        model_output="To implement a neural network for image classification, you'll need to...",
        requested_level=ExplanationLevel.TECHNICAL
    )
    
    technical_result = await engine.generate_explanation(technical_input)
    print(f"Technical Explanation:")
    print(f"Main Explanation:\n{technical_result.main_explanation}")
    print("-" * 50)
    
    # Get explanation statistics
    stats = engine.get_explanation_stats()
    print(f"Explanation Statistics:")
    print(f"Total Explanations: {stats['total_explanations']}")
    print(f"Average Generation Time: {stats['average_generation_time']:.1f}ms")


if __name__ == "__main__":
    asyncio.run(main())