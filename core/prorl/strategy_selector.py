"""Intelligent reasoning strategy selector for ProRL.

This module provides sophisticated strategy selection based on query analysis,
context, performance history, and adaptive learning from past reasoning sessions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import re
import numpy as np
import time
from collections import defaultdict, Counter
import warnings

from .reasoning_engine import ReasoningStrategy, ReasoningTrace, StepType
from .trace_storage import TraceStorageManager, QueryPattern, TraceMetadata


class QueryType(Enum):
    """Types of queries that may require different reasoning strategies."""
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    FACTUAL = "factual"
    CREATIVE = "creative"
    LOGICAL = "logical"
    MATHEMATICAL = "mathematical"
    OPEN_ENDED = "open_ended"
    STEP_BY_STEP = "step_by_step"
    SYNTHESIS = "synthesis"
    EVALUATION = "evaluation"


class QueryComplexity(Enum):
    """Complexity levels for queries."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class QueryAnalysis:
    """Analysis of a query for strategy selection."""
    query_type: QueryType
    complexity: QueryComplexity
    estimated_steps: int
    confidence: float
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    requires_backtracking: bool = False
    requires_branching: bool = False
    domain: Optional[str] = None
    language_indicators: Dict[str, float] = field(default_factory=dict)


@dataclass
class StrategyPerformance:
    """Performance metrics for a reasoning strategy."""
    strategy: ReasoningStrategy
    success_rate: float
    avg_confidence: float
    avg_steps: float
    avg_time: float
    avg_reward: float
    total_uses: int
    recent_success_rate: float
    domain_performance: Dict[str, float] = field(default_factory=dict)
    complexity_performance: Dict[QueryComplexity, float] = field(default_factory=dict)


@dataclass
class StrategyRecommendation:
    """Recommendation for reasoning strategy selection."""
    primary_strategy: ReasoningStrategy
    fallback_strategies: List[ReasoningStrategy]
    confidence: float
    reasoning: str
    estimated_steps: int
    estimated_time: float
    risk_level: str  # "low", "medium", "high"
    adaptations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectorConfig:
    """Configuration for strategy selector."""
    # Learning parameters
    adaptation_rate: float = 0.1
    performance_window_size: int = 100
    min_samples_for_learning: int = 10
    
    # Strategy preferences
    conservative_mode: bool = False
    prefer_fast_strategies: bool = False
    max_estimated_time: float = 300.0  # 5 minutes
    
    # Query analysis
    enable_query_analysis: bool = True
    use_similarity_matching: bool = True
    similarity_threshold: float = 0.7
    
    # Performance tracking
    track_domain_performance: bool = True
    track_complexity_performance: bool = True
    update_frequency: int = 10  # Update metrics every N uses


class IntelligentStrategySelector:
    """Intelligent selector for optimal reasoning strategies."""
    
    def __init__(
        self,
        config: SelectorConfig,
        storage_manager: Optional[TraceStorageManager] = None
    ):
        self.config = config
        self.storage_manager = storage_manager
        
        # Performance tracking
        self.strategy_performance: Dict[ReasoningStrategy, StrategyPerformance] = {}
        self.query_type_preferences: Dict[QueryType, List[ReasoningStrategy]] = {}
        self.complexity_preferences: Dict[QueryComplexity, List[ReasoningStrategy]] = {}
        
        # Query analysis patterns
        self.query_patterns = self._initialize_query_patterns()
        self.domain_keywords = self._initialize_domain_keywords()
        
        # Adaptive learning
        self.recent_selections: List[Tuple[str, ReasoningStrategy, float]] = []
        self.learning_history: Dict[str, List[float]] = defaultdict(list)
        
        # Initialize default performance metrics
        self._initialize_default_performance()
        
        # Load historical data if available
        if self.storage_manager:
            self._load_historical_performance()
    
    def select_strategy(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> StrategyRecommendation:
        """Select optimal reasoning strategy for the given query."""
        
        # Analyze query
        query_analysis = self._analyze_query(query, context)
        
        # Find similar historical queries
        similar_traces = self._find_similar_queries(query) if self.config.use_similarity_matching else []
        
        # Calculate strategy scores
        strategy_scores = self._calculate_strategy_scores(
            query_analysis, similar_traces, context, constraints
        )
        
        # Select primary and fallback strategies
        sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
        
        primary_strategy = sorted_strategies[0][0]
        fallback_strategies = [s[0] for s in sorted_strategies[1:4]]  # Top 3 alternatives
        
        # Calculate recommendation confidence
        confidence = self._calculate_recommendation_confidence(
            strategy_scores, query_analysis, similar_traces
        )
        
        # Generate reasoning explanation
        reasoning = self._generate_reasoning_explanation(
            primary_strategy, query_analysis, strategy_scores, similar_traces
        )
        
        # Estimate execution parameters
        estimated_steps = self._estimate_steps(primary_strategy, query_analysis)
        estimated_time = self._estimate_time(primary_strategy, query_analysis)
        risk_level = self._assess_risk_level(primary_strategy, query_analysis, confidence)
        
        # Generate strategy adaptations
        adaptations = self._generate_adaptations(primary_strategy, query_analysis, context)
        
        recommendation = StrategyRecommendation(
            primary_strategy=primary_strategy,
            fallback_strategies=fallback_strategies,
            confidence=confidence,
            reasoning=reasoning,
            estimated_steps=estimated_steps,
            estimated_time=estimated_time,
            risk_level=risk_level,
            adaptations=adaptations
        )
        
        # Record selection for learning
        self._record_selection(query, recommendation)
        
        return recommendation
    
    def update_performance(
        self,
        query: str,
        strategy: ReasoningStrategy,
        trace: ReasoningTrace,
        success: bool,
        final_confidence: float
    ):
        """Update strategy performance based on execution results."""
        
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = StrategyPerformance(
                strategy=strategy,
                success_rate=0.0,
                avg_confidence=0.0,
                avg_steps=0.0,
                avg_time=0.0,
                avg_reward=0.0,
                total_uses=0,
                recent_success_rate=0.0
            )
        
        perf = self.strategy_performance[strategy]
        
        # Update basic metrics
        old_total = perf.total_uses
        new_total = old_total + 1
        
        # Update success rate
        perf.success_rate = (perf.success_rate * old_total + (1.0 if success else 0.0)) / new_total
        
        # Update confidence
        perf.avg_confidence = (perf.avg_confidence * old_total + final_confidence) / new_total
        
        # Update steps
        perf.avg_steps = (perf.avg_steps * old_total + trace.total_steps) / new_total
        
        # Update time
        if trace.end_time:
            duration = trace.end_time - trace.start_time
            perf.avg_time = (perf.avg_time * old_total + duration) / new_total
        
        # Update reward
        perf.avg_reward = (perf.avg_reward * old_total + trace.total_reward) / new_total
        
        perf.total_uses = new_total
        
        # Update recent success rate (last 20 uses)
        recent_window = 20
        if new_total <= recent_window:
            perf.recent_success_rate = perf.success_rate
        else:
            # Approximate recent performance
            perf.recent_success_rate = perf.recent_success_rate * 0.95 + (0.05 if success else 0.0)
        
        # Update domain performance
        if self.config.track_domain_performance:
            query_analysis = self._analyze_query(query)
            if query_analysis.domain:
                domain = query_analysis.domain
                if domain not in perf.domain_performance:
                    perf.domain_performance[domain] = 0.0
                
                # Update domain-specific performance
                domain_success = 1.0 if success else 0.0
                perf.domain_performance[domain] = (
                    perf.domain_performance[domain] * 0.9 + domain_success * 0.1
                )
        
        # Update complexity performance
        if self.config.track_complexity_performance:
            query_analysis = self._analyze_query(query)
            complexity = query_analysis.complexity
            if complexity not in perf.complexity_performance:
                perf.complexity_performance[complexity] = 0.0
            
            complexity_success = 1.0 if success else 0.0
            perf.complexity_performance[complexity] = (
                perf.complexity_performance[complexity] * 0.9 + complexity_success * 0.1
            )
        
        # Adaptive learning updates
        self._update_adaptive_preferences(query, strategy, success, final_confidence)
    
    def _analyze_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryAnalysis:
        """Analyze query to determine characteristics for strategy selection."""
        
        query_lower = query.lower()
        words = query_lower.split()
        
        # Determine query type
        query_type = self._classify_query_type(query_lower, words)
        
        # Assess complexity
        complexity = self._assess_complexity(query, words, context)
        
        # Extract keywords and entities
        keywords = self._extract_keywords(query_lower, words)
        entities = self._extract_entities(query)
        concepts = self._extract_concepts(query, context)
        
        # Determine domain
        domain = self._identify_domain(query_lower, keywords)
        
        # Assess requirements
        requires_backtracking = self._requires_backtracking(query_lower, query_type, complexity)
        requires_branching = self._requires_branching(query_lower, query_type)
        
        # Estimate steps
        estimated_steps = self._estimate_query_steps(query_type, complexity, len(words))
        
        # Language indicators
        language_indicators = self._analyze_language_patterns(query_lower)
        
        return QueryAnalysis(
            query_type=query_type,
            complexity=complexity,
            estimated_steps=estimated_steps,
            confidence=0.8,  # Default confidence
            keywords=keywords,
            entities=entities,
            concepts=concepts,
            requires_backtracking=requires_backtracking,
            requires_branching=requires_branching,
            domain=domain,
            language_indicators=language_indicators
        )
    
    def _classify_query_type(self, query_lower: str, words: List[str]) -> QueryType:
        """Classify the type of query."""
        
        # Check for comparative indicators
        comparative_patterns = [
            "compare", "contrast", "difference", "similar", "versus", "vs",
            "better", "worse", "advantages", "disadvantages"
        ]
        if any(pattern in query_lower for pattern in comparative_patterns):
            return QueryType.COMPARATIVE
        
        # Check for analytical indicators
        analytical_patterns = [
            "analyze", "analysis", "examine", "investigate", "study",
            "evaluate", "assess", "review", "breakdown"
        ]
        if any(pattern in query_lower for pattern in analytical_patterns):
            return QueryType.ANALYTICAL
        
        # Check for factual indicators
        factual_patterns = [
            "what is", "who is", "when", "where", "which", "define",
            "definition", "fact", "information"
        ]
        if any(pattern in query_lower for pattern in factual_patterns):
            return QueryType.FACTUAL
        
        # Check for mathematical indicators
        math_patterns = [
            "calculate", "solve", "equation", "formula", "mathematics",
            "compute", "derive", "proof", "theorem"
        ]
        if any(pattern in query_lower for pattern in math_patterns):
            return QueryType.MATHEMATICAL
        
        # Check for logical reasoning
        logical_patterns = [
            "if", "then", "because", "therefore", "logic", "reasoning",
            "conclude", "infer", "deduce", "premise"
        ]
        if any(pattern in query_lower for pattern in logical_patterns):
            return QueryType.LOGICAL
        
        # Check for step-by-step
        step_patterns = [
            "step", "process", "procedure", "method", "how to",
            "instructions", "guide", "tutorial"
        ]
        if any(pattern in query_lower for pattern in step_patterns):
            return QueryType.STEP_BY_STEP
        
        # Check for creative indicators
        creative_patterns = [
            "create", "design", "imagine", "brainstorm", "generate",
            "innovate", "invent", "compose"
        ]
        if any(pattern in query_lower for pattern in creative_patterns):
            return QueryType.CREATIVE
        
        # Check for synthesis
        synthesis_patterns = [
            "synthesize", "combine", "integrate", "merge", "unify",
            "bring together", "consolidate"
        ]
        if any(pattern in query_lower for pattern in synthesis_patterns):
            return QueryType.SYNTHESIS
        
        # Check for evaluation
        evaluation_patterns = [
            "evaluate", "judge", "rate", "rank", "score", "critique",
            "opinion", "recommendation"
        ]
        if any(pattern in query_lower for pattern in evaluation_patterns):
            return QueryType.EVALUATION
        
        # Default to open-ended
        return QueryType.OPEN_ENDED
    
    def _assess_complexity(
        self,
        query: str,
        words: List[str],
        context: Optional[Dict[str, Any]]
    ) -> QueryComplexity:
        """Assess the complexity of the query."""
        
        complexity_score = 0
        
        # Length-based scoring
        if len(words) > 50:
            complexity_score += 3
        elif len(words) > 25:
            complexity_score += 2
        elif len(words) > 10:
            complexity_score += 1
        
        # Complex concept indicators
        complex_indicators = [
            "multiple", "various", "several", "complex", "complicated",
            "intricate", "elaborate", "sophisticated", "nuanced"
        ]
        complexity_score += sum(1 for indicator in complex_indicators if indicator in query.lower())
        
        # Multi-part questions
        if "and" in query.lower():
            complexity_score += query.lower().count("and")
        
        # Question marks (multiple questions)
        complexity_score += query.count("?") - 1 if query.count("?") > 1 else 0
        
        # Context complexity
        if context and len(context) > 5:
            complexity_score += 1
        
        # Map score to complexity level
        if complexity_score >= 6:
            return QueryComplexity.VERY_COMPLEX
        elif complexity_score >= 4:
            return QueryComplexity.COMPLEX
        elif complexity_score >= 2:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    def _extract_keywords(self, query_lower: str, words: List[str]) -> List[str]:
        """Extract important keywords from the query."""
        # Simple keyword extraction (would use NLP in practice)
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "are", "was", "were",
            "be", "been", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "must"
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords[:10]  # Limit to top 10
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from the query."""
        # Simple entity extraction (would use NER in practice)
        # Look for capitalized words that might be entities
        entities = []
        words = query.split()
        
        for word in words:
            if word[0].isupper() and len(word) > 2:
                # Remove punctuation
                clean_word = re.sub(r'[^\w]', '', word)
                if clean_word:
                    entities.append(clean_word)
        
        return entities
    
    def _extract_concepts(self, query: str, context: Optional[Dict[str, Any]]) -> List[str]:
        """Extract key concepts from the query."""
        # Simple concept extraction
        concepts = []
        
        # Domain-specific concept patterns
        concept_patterns = {
            "machine learning": ["algorithm", "model", "training", "neural", "deep"],
            "physics": ["force", "energy", "motion", "wave", "particle"],
            "biology": ["cell", "organism", "evolution", "gene", "protein"],
            "mathematics": ["equation", "function", "variable", "theorem", "proof"]
        }
        
        query_lower = query.lower()
        for concept, keywords in concept_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                concepts.append(concept)
        
        return concepts
    
    def _identify_domain(self, query_lower: str, keywords: List[str]) -> Optional[str]:
        """Identify the domain/field of the query."""
        
        for domain, domain_keywords in self.domain_keywords.items():
            if any(keyword in query_lower for keyword in domain_keywords):
                return domain
        
        return None
    
    def _requires_backtracking(
        self,
        query_lower: str,
        query_type: QueryType,
        complexity: QueryComplexity
    ) -> bool:
        """Determine if the query likely requires backtracking."""
        
        backtrack_indicators = [
            "complex", "difficult", "challenging", "multiple approaches",
            "various methods", "alternative", "reconsider"
        ]
        
        if any(indicator in query_lower for indicator in backtrack_indicators):
            return True
        
        # Complex analytical queries often need backtracking
        if query_type == QueryType.ANALYTICAL and complexity in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX]:
            return True
        
        # Mathematical proofs often need backtracking
        if query_type == QueryType.MATHEMATICAL and "proof" in query_lower:
            return True
        
        return False
    
    def _requires_branching(self, query_lower: str, query_type: QueryType) -> bool:
        """Determine if the query likely requires branching."""
        
        branch_indicators = [
            "multiple", "various", "different", "alternative", "options",
            "possibilities", "scenarios", "cases"
        ]
        
        if any(indicator in query_lower for indicator in branch_indicators):
            return True
        
        # Comparative and evaluation queries often need branching
        if query_type in [QueryType.COMPARATIVE, QueryType.EVALUATION]:
            return True
        
        return False
    
    def _estimate_query_steps(
        self,
        query_type: QueryType,
        complexity: QueryComplexity,
        word_count: int
    ) -> int:
        """Estimate the number of reasoning steps needed."""
        
        base_steps = {
            QueryType.FACTUAL: 3,
            QueryType.ANALYTICAL: 8,
            QueryType.COMPARATIVE: 6,
            QueryType.LOGICAL: 5,
            QueryType.MATHEMATICAL: 7,
            QueryType.CREATIVE: 10,
            QueryType.STEP_BY_STEP: 12,
            QueryType.SYNTHESIS: 9,
            QueryType.EVALUATION: 7,
            QueryType.OPEN_ENDED: 6
        }
        
        complexity_multiplier = {
            QueryComplexity.SIMPLE: 0.5,
            QueryComplexity.MODERATE: 1.0,
            QueryComplexity.COMPLEX: 1.5,
            QueryComplexity.VERY_COMPLEX: 2.0
        }
        
        base = base_steps.get(query_type, 6)
        multiplier = complexity_multiplier.get(complexity, 1.0)
        
        # Word count adjustment
        word_adjustment = min(word_count // 10, 5)
        
        estimated = int(base * multiplier + word_adjustment)
        return max(3, min(estimated, 50))  # Reasonable bounds
    
    def _analyze_language_patterns(self, query_lower: str) -> Dict[str, float]:
        """Analyze language patterns that might influence strategy selection."""
        
        patterns = {
            "uncertainty": 0.0,
            "complexity": 0.0,
            "specificity": 0.0,
            "formality": 0.0
        }
        
        # Uncertainty indicators
        uncertainty_words = ["maybe", "perhaps", "possibly", "might", "could", "uncertain"]
        patterns["uncertainty"] = sum(1 for word in uncertainty_words if word in query_lower) / 10
        
        # Complexity indicators
        complexity_words = ["complex", "complicated", "intricate", "sophisticated", "nuanced"]
        patterns["complexity"] = sum(1 for word in complexity_words if word in query_lower) / 5
        
        # Specificity indicators
        specific_words = ["exactly", "precisely", "specifically", "particular", "exact"]
        patterns["specificity"] = sum(1 for word in specific_words if word in query_lower) / 5
        
        # Formality indicators
        formal_words = ["furthermore", "therefore", "consequently", "moreover", "nonetheless"]
        patterns["formality"] = sum(1 for word in formal_words if word in query_lower) / 5
        
        return patterns
    
    def _calculate_strategy_scores(
        self,
        query_analysis: QueryAnalysis,
        similar_traces: List[Tuple[TraceMetadata, float]],
        context: Optional[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]]
    ) -> Dict[ReasoningStrategy, float]:
        """Calculate scores for each reasoning strategy."""
        
        scores = {}
        
        for strategy in ReasoningStrategy:
            if strategy == ReasoningStrategy.AUTO:
                continue  # Skip AUTO strategy
            
            score = 0.0
            
            # Base strategy fitness for query type
            score += self._get_type_fitness(strategy, query_analysis.query_type)
            
            # Complexity fitness
            score += self._get_complexity_fitness(strategy, query_analysis.complexity)
            
            # Historical performance
            if strategy in self.strategy_performance:
                perf = self.strategy_performance[strategy]
                score += perf.success_rate * 0.3
                score += perf.recent_success_rate * 0.2
                
                # Domain-specific performance
                if query_analysis.domain and query_analysis.domain in perf.domain_performance:
                    score += perf.domain_performance[query_analysis.domain] * 0.2
                
                # Complexity-specific performance
                if query_analysis.complexity in perf.complexity_performance:
                    score += perf.complexity_performance[query_analysis.complexity] * 0.15
            
            # Similar trace performance
            if similar_traces:
                strategy_success_in_similar = sum(
                    1 for trace_meta, _ in similar_traces
                    if trace_meta.strategy == strategy.value and trace_meta.success
                )
                if strategy_success_in_similar > 0:
                    similarity_score = strategy_success_in_similar / len(similar_traces)
                    score += similarity_score * 0.25
            
            # Special requirements
            if query_analysis.requires_backtracking:
                if strategy in [ReasoningStrategy.TREE_OF_THOUGHT, ReasoningStrategy.GRAPH_OF_THOUGHT]:
                    score += 0.2
                elif strategy == ReasoningStrategy.CHAIN_OF_THOUGHT:
                    score -= 0.1
            
            if query_analysis.requires_branching:
                if strategy == ReasoningStrategy.TREE_OF_THOUGHT:
                    score += 0.3
                elif strategy == ReasoningStrategy.GRAPH_OF_THOUGHT:
                    score += 0.25
            
            # Constraint adjustments
            if constraints:
                if constraints.get("prefer_fast", False):
                    if strategy == ReasoningStrategy.CHAIN_OF_THOUGHT:
                        score += 0.2
                    elif strategy in [ReasoningStrategy.TREE_OF_THOUGHT, ReasoningStrategy.GRAPH_OF_THOUGHT]:
                        score -= 0.15
                
                max_time = constraints.get("max_time_seconds", self.config.max_estimated_time)
                estimated_time = self._estimate_time(strategy, query_analysis)
                if estimated_time > max_time:
                    score -= 0.3
            
            # Conservative mode adjustments
            if self.config.conservative_mode:
                if strategy == ReasoningStrategy.CHAIN_OF_THOUGHT:
                    score += 0.1
                elif strategy in [ReasoningStrategy.GRAPH_OF_THOUGHT, ReasoningStrategy.METACOGNITIVE]:
                    score -= 0.1
            
            scores[strategy] = max(0.0, min(1.0, score))
        
        return scores
    
    def _get_type_fitness(self, strategy: ReasoningStrategy, query_type: QueryType) -> float:
        """Get fitness score for strategy-query type combination."""
        
        # Strategy-type fitness matrix
        fitness_matrix = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: {
                QueryType.FACTUAL: 0.8,
                QueryType.LOGICAL: 0.9,
                QueryType.STEP_BY_STEP: 0.9,
                QueryType.MATHEMATICAL: 0.8,
                QueryType.ANALYTICAL: 0.7,
                QueryType.COMPARATIVE: 0.6,
                QueryType.CREATIVE: 0.5,
                QueryType.SYNTHESIS: 0.6,
                QueryType.EVALUATION: 0.7,
                QueryType.OPEN_ENDED: 0.6
            },
            ReasoningStrategy.TREE_OF_THOUGHT: {
                QueryType.COMPARATIVE: 0.9,
                QueryType.EVALUATION: 0.9,
                QueryType.ANALYTICAL: 0.8,
                QueryType.CREATIVE: 0.8,
                QueryType.SYNTHESIS: 0.8,
                QueryType.LOGICAL: 0.7,
                QueryType.MATHEMATICAL: 0.7,
                QueryType.OPEN_ENDED: 0.8,
                QueryType.FACTUAL: 0.5,
                QueryType.STEP_BY_STEP: 0.6
            },
            ReasoningStrategy.GRAPH_OF_THOUGHT: {
                QueryType.SYNTHESIS: 0.9,
                QueryType.ANALYTICAL: 0.9,
                QueryType.COMPARATIVE: 0.8,
                QueryType.CREATIVE: 0.8,
                QueryType.EVALUATION: 0.8,
                QueryType.OPEN_ENDED: 0.8,
                QueryType.LOGICAL: 0.7,
                QueryType.MATHEMATICAL: 0.6,
                QueryType.FACTUAL: 0.5,
                QueryType.STEP_BY_STEP: 0.6
            },
            ReasoningStrategy.SELF_CONSISTENCY: {
                QueryType.FACTUAL: 0.9,
                QueryType.LOGICAL: 0.8,
                QueryType.MATHEMATICAL: 0.8,
                QueryType.EVALUATION: 0.7,
                QueryType.ANALYTICAL: 0.7,
                QueryType.COMPARATIVE: 0.6,
                QueryType.STEP_BY_STEP: 0.6,
                QueryType.SYNTHESIS: 0.5,
                QueryType.CREATIVE: 0.4,
                QueryType.OPEN_ENDED: 0.5
            },
            ReasoningStrategy.METACOGNITIVE: {
                QueryType.ANALYTICAL: 0.8,
                QueryType.EVALUATION: 0.8,
                QueryType.COMPARATIVE: 0.7,
                QueryType.SYNTHESIS: 0.7,
                QueryType.LOGICAL: 0.7,
                QueryType.CREATIVE: 0.6,
                QueryType.MATHEMATICAL: 0.6,
                QueryType.OPEN_ENDED: 0.6,
                QueryType.FACTUAL: 0.4,
                QueryType.STEP_BY_STEP: 0.5
            }
        }
        
        return fitness_matrix.get(strategy, {}).get(query_type, 0.5)
    
    def _get_complexity_fitness(self, strategy: ReasoningStrategy, complexity: QueryComplexity) -> float:
        """Get fitness score for strategy-complexity combination."""
        
        complexity_fitness = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: {
                QueryComplexity.SIMPLE: 0.9,
                QueryComplexity.MODERATE: 0.8,
                QueryComplexity.COMPLEX: 0.6,
                QueryComplexity.VERY_COMPLEX: 0.4
            },
            ReasoningStrategy.TREE_OF_THOUGHT: {
                QueryComplexity.SIMPLE: 0.6,
                QueryComplexity.MODERATE: 0.8,
                QueryComplexity.COMPLEX: 0.9,
                QueryComplexity.VERY_COMPLEX: 0.8
            },
            ReasoningStrategy.GRAPH_OF_THOUGHT: {
                QueryComplexity.SIMPLE: 0.5,
                QueryComplexity.MODERATE: 0.7,
                QueryComplexity.COMPLEX: 0.9,
                QueryComplexity.VERY_COMPLEX: 0.9
            },
            ReasoningStrategy.SELF_CONSISTENCY: {
                QueryComplexity.SIMPLE: 0.8,
                QueryComplexity.MODERATE: 0.7,
                QueryComplexity.COMPLEX: 0.6,
                QueryComplexity.VERY_COMPLEX: 0.5
            },
            ReasoningStrategy.METACOGNITIVE: {
                QueryComplexity.SIMPLE: 0.4,
                QueryComplexity.MODERATE: 0.6,
                QueryComplexity.COMPLEX: 0.8,
                QueryComplexity.VERY_COMPLEX: 0.9
            }
        }
        
        return complexity_fitness.get(strategy, {}).get(complexity, 0.5)
    
    def _find_similar_queries(self, query: str) -> List[Tuple[TraceMetadata, float]]:
        """Find similar queries from historical data."""
        if not self.storage_manager:
            return []
        
        try:
            return self.storage_manager.get_similar_traces(
                query, limit=10, min_similarity=self.config.similarity_threshold
            )
        except Exception as e:
            warnings.warn(f"Failed to find similar queries: {str(e)}")
            return []
    
    def _calculate_recommendation_confidence(
        self,
        strategy_scores: Dict[ReasoningStrategy, float],
        query_analysis: QueryAnalysis,
        similar_traces: List[Tuple[TraceMetadata, float]]
    ) -> float:
        """Calculate confidence in the strategy recommendation."""
        
        if not strategy_scores:
            return 0.5
        
        sorted_scores = sorted(strategy_scores.values(), reverse=True)
        
        # Gap between top two strategies
        if len(sorted_scores) >= 2:
            score_gap = sorted_scores[0] - sorted_scores[1]
            confidence = 0.5 + (score_gap * 0.5)
        else:
            confidence = sorted_scores[0]
        
        # Adjust based on query analysis confidence
        confidence = (confidence + query_analysis.confidence) / 2
        
        # Boost confidence if we have similar successful traces
        if similar_traces:
            successful_similar = sum(1 for trace_meta, _ in similar_traces if trace_meta.success)
            if successful_similar > 0:
                similarity_boost = (successful_similar / len(similar_traces)) * 0.2
                confidence = min(1.0, confidence + similarity_boost)
        
        return confidence
    
    def _generate_reasoning_explanation(
        self,
        strategy: ReasoningStrategy,
        query_analysis: QueryAnalysis,
        strategy_scores: Dict[ReasoningStrategy, float],
        similar_traces: List[Tuple[TraceMetadata, float]]
    ) -> str:
        """Generate explanation for strategy selection."""
        
        explanations = []
        
        # Base strategy explanation
        strategy_explanations = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: "Sequential step-by-step reasoning",
            ReasoningStrategy.TREE_OF_THOUGHT: "Multi-path exploration with branching",
            ReasoningStrategy.GRAPH_OF_THOUGHT: "Complex interconnected reasoning",
            ReasoningStrategy.SELF_CONSISTENCY: "Verification through consistency checking",
            ReasoningStrategy.METACOGNITIVE: "Self-reflective reasoning approach"
        }
        
        explanations.append(f"Selected {strategy_explanations.get(strategy, strategy.value)}")
        
        # Query type reasoning
        explanations.append(f"for {query_analysis.query_type.value} query")
        
        # Complexity reasoning
        if query_analysis.complexity in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX]:
            explanations.append(f"with {query_analysis.complexity.value} complexity")
        
        # Special requirements
        if query_analysis.requires_branching:
            explanations.append("requiring multiple path exploration")
        
        if query_analysis.requires_backtracking:
            explanations.append("potentially needing backtracking")
        
        # Historical performance
        if strategy in self.strategy_performance:
            perf = self.strategy_performance[strategy]
            if perf.recent_success_rate > 0.7:
                explanations.append(f"with strong recent performance ({perf.recent_success_rate:.1%})")
        
        # Similar traces
        if similar_traces:
            explanations.append(f"based on {len(similar_traces)} similar successful cases")
        
        return "; ".join(explanations) + "."
    
    def _estimate_steps(self, strategy: ReasoningStrategy, query_analysis: QueryAnalysis) -> int:
        """Estimate number of steps for strategy-query combination."""
        
        base_estimate = query_analysis.estimated_steps
        
        # Strategy multipliers
        strategy_multipliers = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: 1.0,
            ReasoningStrategy.TREE_OF_THOUGHT: 1.5,
            ReasoningStrategy.GRAPH_OF_THOUGHT: 2.0,
            ReasoningStrategy.SELF_CONSISTENCY: 1.3,
            ReasoningStrategy.METACOGNITIVE: 1.8
        }
        
        multiplier = strategy_multipliers.get(strategy, 1.0)
        estimated = int(base_estimate * multiplier)
        
        return max(3, min(estimated, 100))  # Reasonable bounds
    
    def _estimate_time(self, strategy: ReasoningStrategy, query_analysis: QueryAnalysis) -> float:
        """Estimate execution time for strategy-query combination."""
        
        # Base time per step (seconds)
        base_time_per_step = 2.0
        
        # Strategy time multipliers
        strategy_time_multipliers = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: 1.0,
            ReasoningStrategy.TREE_OF_THOUGHT: 1.8,
            ReasoningStrategy.GRAPH_OF_THOUGHT: 2.5,
            ReasoningStrategy.SELF_CONSISTENCY: 1.5,
            ReasoningStrategy.METACOGNITIVE: 2.2
        }
        
        steps = self._estimate_steps(strategy, query_analysis)
        multiplier = strategy_time_multipliers.get(strategy, 1.0)
        
        estimated_time = steps * base_time_per_step * multiplier
        
        # Historical adjustment
        if strategy in self.strategy_performance:
            historical_time = self.strategy_performance[strategy].avg_time
            if historical_time > 0:
                estimated_time = (estimated_time + historical_time) / 2
        
        return max(5.0, min(estimated_time, 600.0))  # 5 seconds to 10 minutes
    
    def _assess_risk_level(
        self,
        strategy: ReasoningStrategy,
        query_analysis: QueryAnalysis,
        confidence: float
    ) -> str:
        """Assess risk level of using the selected strategy."""
        
        risk_score = 0
        
        # Low confidence increases risk
        if confidence < 0.6:
            risk_score += 2
        elif confidence < 0.8:
            risk_score += 1
        
        # Complex queries increase risk
        if query_analysis.complexity == QueryComplexity.VERY_COMPLEX:
            risk_score += 2
        elif query_analysis.complexity == QueryComplexity.COMPLEX:
            risk_score += 1
        
        # Strategy-specific risks
        if strategy in [ReasoningStrategy.GRAPH_OF_THOUGHT, ReasoningStrategy.METACOGNITIVE]:
            risk_score += 1  # More complex strategies
        
        # Historical performance
        if strategy in self.strategy_performance:
            perf = self.strategy_performance[strategy]
            if perf.recent_success_rate < 0.5:
                risk_score += 2
            elif perf.recent_success_rate < 0.7:
                risk_score += 1
        else:
            risk_score += 1  # Unknown performance
        
        # Map score to risk level
        if risk_score >= 4:
            return "high"
        elif risk_score >= 2:
            return "medium"
        else:
            return "low"
    
    def _generate_adaptations(
        self,
        strategy: ReasoningStrategy,
        query_analysis: QueryAnalysis,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate strategy-specific adaptations."""
        
        adaptations = {}
        
        # Complexity-based adaptations
        if query_analysis.complexity == QueryComplexity.VERY_COMPLEX:
            adaptations["max_depth"] = 20
            adaptations["enable_backtracking"] = True
            adaptations["patience_multiplier"] = 1.5
        elif query_analysis.complexity == QueryComplexity.SIMPLE:
            adaptations["max_depth"] = 5
            adaptations["early_termination"] = True
            adaptations["patience_multiplier"] = 0.8
        
        # Strategy-specific adaptations
        if strategy == ReasoningStrategy.TREE_OF_THOUGHT:
            adaptations["branch_factor"] = 3 if query_analysis.requires_branching else 2
            adaptations["pruning_threshold"] = 0.3
        
        elif strategy == ReasoningStrategy.GRAPH_OF_THOUGHT:
            adaptations["max_connections"] = 5
            adaptations["synthesis_weight"] = 0.3
        
        elif strategy == ReasoningStrategy.SELF_CONSISTENCY:
            adaptations["verification_rounds"] = 2
            adaptations["consistency_threshold"] = 0.8
        
        # Domain-specific adaptations
        if query_analysis.domain:
            domain_adaptations = {
                "mathematics": {"precision_mode": True, "verification_enabled": True},
                "science": {"evidence_weight": 0.8, "citation_required": True},
                "creative": {"exploration_bonus": 0.2, "novelty_weight": 0.3}
            }
            
            if query_analysis.domain in domain_adaptations:
                adaptations.update(domain_adaptations[query_analysis.domain])
        
        return adaptations
    
    def _record_selection(self, query: str, recommendation: StrategyRecommendation):
        """Record strategy selection for learning."""
        
        self.recent_selections.append((
            query,
            recommendation.primary_strategy,
            recommendation.confidence
        ))
        
        # Keep only recent selections
        if len(self.recent_selections) > 100:
            self.recent_selections = self.recent_selections[-100:]
    
    def _update_adaptive_preferences(
        self,
        query: str,
        strategy: ReasoningStrategy,
        success: bool,
        confidence: float
    ):
        """Update adaptive learning preferences."""
        
        query_analysis = self._analyze_query(query)
        
        # Update query type preferences
        if query_analysis.query_type not in self.query_type_preferences:
            self.query_type_preferences[query_analysis.query_type] = []
        
        preferences = self.query_type_preferences[query_analysis.query_type]
        
        if success and confidence > 0.7:
            # Boost preference for successful strategy
            if strategy not in preferences:
                preferences.append(strategy)
            else:
                # Move to front
                preferences.remove(strategy)
                preferences.insert(0, strategy)
        elif not success:
            # Reduce preference for failed strategy
            if strategy in preferences:
                preferences.remove(strategy)
                preferences.append(strategy)  # Move to back
        
        # Limit preference list size
        self.query_type_preferences[query_analysis.query_type] = preferences[:5]
    
    def _initialize_query_patterns(self) -> Dict[str, List[str]]:
        """Initialize query pattern recognition."""
        
        return {
            "comparison": ["compare", "contrast", "versus", "vs", "difference", "similar"],
            "analysis": ["analyze", "examine", "study", "investigate", "breakdown"],
            "evaluation": ["evaluate", "assess", "judge", "rate", "rank", "critique"],
            "creation": ["create", "design", "generate", "compose", "build"],
            "explanation": ["explain", "describe", "clarify", "elaborate", "detail"],
            "problem_solving": ["solve", "resolve", "fix", "address", "tackle"]
        }
    
    def _initialize_domain_keywords(self) -> Dict[str, List[str]]:
        """Initialize domain keyword mappings."""
        
        return {
            "mathematics": ["equation", "formula", "calculate", "solve", "proof", "theorem"],
            "science": ["experiment", "hypothesis", "theory", "research", "study", "analysis"],
            "technology": ["algorithm", "software", "system", "network", "database", "code"],
            "business": ["strategy", "market", "revenue", "profit", "customer", "sales"],
            "philosophy": ["ethics", "morality", "logic", "reasoning", "argument", "belief"],
            "history": ["historical", "past", "ancient", "medieval", "modern", "timeline"],
            "literature": ["novel", "poem", "author", "character", "plot", "theme"],
            "art": ["painting", "sculpture", "design", "artistic", "creative", "aesthetic"]
        }
    
    def _initialize_default_performance(self):
        """Initialize default performance metrics for strategies."""
        
        default_metrics = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: StrategyPerformance(
                strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
                success_rate=0.75,
                avg_confidence=0.7,
                avg_steps=8.0,
                avg_time=15.0,
                avg_reward=0.6,
                total_uses=0,
                recent_success_rate=0.75
            ),
            ReasoningStrategy.TREE_OF_THOUGHT: StrategyPerformance(
                strategy=ReasoningStrategy.TREE_OF_THOUGHT,
                success_rate=0.65,
                avg_confidence=0.75,
                avg_steps=15.0,
                avg_time=30.0,
                avg_reward=0.7,
                total_uses=0,
                recent_success_rate=0.65
            ),
            ReasoningStrategy.GRAPH_OF_THOUGHT: StrategyPerformance(
                strategy=ReasoningStrategy.GRAPH_OF_THOUGHT,
                success_rate=0.60,
                avg_confidence=0.8,
                avg_steps=25.0,
                avg_time=45.0,
                avg_reward=0.75,
                total_uses=0,
                recent_success_rate=0.60
            ),
            ReasoningStrategy.SELF_CONSISTENCY: StrategyPerformance(
                strategy=ReasoningStrategy.SELF_CONSISTENCY,
                success_rate=0.80,
                avg_confidence=0.85,
                avg_steps=10.0,
                avg_time=20.0,
                avg_reward=0.65,
                total_uses=0,
                recent_success_rate=0.80
            ),
            ReasoningStrategy.METACOGNITIVE: StrategyPerformance(
                strategy=ReasoningStrategy.METACOGNITIVE,
                success_rate=0.55,
                avg_confidence=0.75,
                avg_steps=20.0,
                avg_time=40.0,
                avg_reward=0.8,
                total_uses=0,
                recent_success_rate=0.55
            )
        }
        
        self.strategy_performance.update(default_metrics)
    
    def _load_historical_performance(self):
        """Load historical performance data from storage."""
        
        try:
            # Get recent successful traces for each strategy
            for strategy in ReasoningStrategy:
                if strategy == ReasoningStrategy.AUTO:
                    continue
                
                pattern = QueryPattern(
                    strategy=strategy,
                    success_only=True,
                    date_from=datetime.now() - timedelta(days=30)
                )
                
                traces = self.storage_manager.query_traces(pattern, limit=100)
                
                if traces:
                    # Calculate performance metrics
                    total_traces = len(traces)
                    avg_confidence = sum(t.confidence for t in traces) / total_traces
                    avg_steps = sum(t.total_steps for t in traces) / total_traces
                    avg_time = sum(t.duration_seconds for t in traces) / total_traces
                    avg_reward = sum(t.total_reward for t in traces) / total_traces
                    
                    # Update performance
                    if strategy in self.strategy_performance:
                        perf = self.strategy_performance[strategy]
                        perf.avg_confidence = avg_confidence
                        perf.avg_steps = avg_steps
                        perf.avg_time = avg_time
                        perf.avg_reward = avg_reward
                        perf.total_uses = total_traces
        
        except Exception as e:
            warnings.warn(f"Failed to load historical performance: {str(e)}")
    
    def get_selector_statistics(self) -> Dict[str, Any]:
        """Get comprehensive selector statistics."""
        
        return {
            "strategy_performance": {
                strategy.value: {
                    "success_rate": perf.success_rate,
                    "recent_success_rate": perf.recent_success_rate,
                    "avg_confidence": perf.avg_confidence,
                    "avg_steps": perf.avg_steps,
                    "avg_time": perf.avg_time,
                    "total_uses": perf.total_uses
                }
                for strategy, perf in self.strategy_performance.items()
            },
            "query_type_preferences": {
                qtype.value: [s.value for s in strategies]
                for qtype, strategies in self.query_type_preferences.items()
            },
            "recent_selections": len(self.recent_selections),
            "learning_history_size": sum(len(history) for history in self.learning_history.values())
        }