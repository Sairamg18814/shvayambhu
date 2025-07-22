"""
Domain-Specific Reasoning Engine for Shvayambhu LLM System

This module implements advanced domain-specific reasoning capabilities that adapt
reasoning strategies, knowledge sources, and cognitive processes based on the
specific domain of the input. Integrates consciousness awareness for enhanced
reasoning quality and self-reflection.

Key Features:
- Multi-domain reasoning (medical, legal, scientific, technical, creative, etc.)
- Dynamic reasoning strategy selection
- Domain-specific knowledge base integration
- Consciousness-guided reasoning processes
- Multi-paradigm reasoning support (deductive, inductive, abductive, analogical)
- Explainable reasoning with domain-appropriate explanations
- Adaptive confidence scoring based on domain expertise
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Base consciousness integration
from ..consciousness.base import ConsciousnessAwareModule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReasoningDomain(Enum):
    """Supported reasoning domains"""
    MEDICAL = auto()
    LEGAL = auto()
    SCIENTIFIC = auto()
    TECHNICAL = auto()
    CREATIVE = auto()
    BUSINESS = auto()
    EDUCATIONAL = auto()
    PHILOSOPHICAL = auto()
    MATHEMATICAL = auto()
    ETHICAL = auto()
    GENERAL = auto()


class ReasoningParadigm(Enum):
    """Different reasoning paradigms"""
    DEDUCTIVE = auto()      # From general principles to specific conclusions
    INDUCTIVE = auto()      # From specific observations to general principles
    ABDUCTIVE = auto()      # Best explanation for observations
    ANALOGICAL = auto()     # Reasoning by similarity
    CAUSAL = auto()         # Cause-effect reasoning
    PROBABILISTIC = auto()  # Probabilistic inference
    DIALECTICAL = auto()    # Thesis-antithesis-synthesis
    COUNTERFACTUAL = auto() # What-if reasoning


class ConfidenceLevel(Enum):
    """Confidence levels for reasoning outputs"""
    VERY_LOW = (0.0, 0.2)
    LOW = (0.2, 0.4)
    MEDIUM = (0.4, 0.6)
    HIGH = (0.6, 0.8)
    VERY_HIGH = (0.8, 1.0)

    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val


@dataclass
class ReasoningInput:
    """Input data for reasoning process"""
    query: str
    context: Dict[str, Any] = field(default_factory=dict)
    domain_hint: Optional[ReasoningDomain] = None
    paradigm_preference: Optional[List[ReasoningParadigm]] = None
    required_confidence: float = 0.7
    time_limit: Optional[float] = None
    consciousness_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningStep:
    """Individual step in reasoning process"""
    step_id: str
    paradigm: ReasoningParadigm
    description: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    confidence: float
    duration_ms: float
    consciousness_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningOutput:
    """Output from reasoning process"""
    conclusion: str
    domain: ReasoningDomain
    paradigms_used: List[ReasoningParadigm]
    confidence: float
    confidence_level: ConfidenceLevel
    explanation: str
    reasoning_steps: List[ReasoningStep]
    alternative_conclusions: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    uncertainty_factors: List[str] = field(default_factory=list)
    consciousness_insights: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class DomainKnowledgeBase:
    """Knowledge base for specific domains"""
    
    def __init__(self, domain: ReasoningDomain):
        self.domain = domain
        self.facts: Dict[str, Any] = {}
        self.rules: Dict[str, Callable] = {}
        self.heuristics: List[str] = []
        self.paradigm_weights: Dict[ReasoningParadigm, float] = {}
        self._load_domain_knowledge()
    
    def _load_domain_knowledge(self):
        """Load domain-specific knowledge"""
        if self.domain == ReasoningDomain.MEDICAL:
            self.paradigm_weights = {
                ReasoningParadigm.DEDUCTIVE: 0.8,
                ReasoningParadigm.INDUCTIVE: 0.7,
                ReasoningParadigm.ABDUCTIVE: 0.9,
                ReasoningParadigm.CAUSAL: 0.8,
                ReasoningParadigm.PROBABILISTIC: 0.7
            }
            self.heuristics = [
                "Consider differential diagnosis",
                "Evaluate symptom combinations",
                "Account for patient history",
                "Consider contraindications"
            ]
        elif self.domain == ReasoningDomain.LEGAL:
            self.paradigm_weights = {
                ReasoningParadigm.DEDUCTIVE: 0.9,
                ReasoningParadigm.ANALOGICAL: 0.8,
                ReasoningParadigm.DIALECTICAL: 0.7,
                ReasoningParadigm.COUNTERFACTUAL: 0.6
            }
            self.heuristics = [
                "Apply relevant statutes and precedents",
                "Consider burden of proof",
                "Evaluate evidence admissibility",
                "Account for jurisdictional differences"
            ]
        elif self.domain == ReasoningDomain.SCIENTIFIC:
            self.paradigm_weights = {
                ReasoningParadigm.INDUCTIVE: 0.8,
                ReasoningParadigm.DEDUCTIVE: 0.7,
                ReasoningParadigm.ABDUCTIVE: 0.8,
                ReasoningParadigm.CAUSAL: 0.9,
                ReasoningParadigm.PROBABILISTIC: 0.8
            }
            self.heuristics = [
                "Formulate testable hypotheses",
                "Consider experimental design",
                "Account for confounding variables",
                "Evaluate statistical significance"
            ]
        elif self.domain == ReasoningDomain.CREATIVE:
            self.paradigm_weights = {
                ReasoningParadigm.ANALOGICAL: 0.9,
                ReasoningParadigm.ABDUCTIVE: 0.7,
                ReasoningParadigm.COUNTERFACTUAL: 0.8,
                ReasoningParadigm.DIALECTICAL: 0.6
            }
            self.heuristics = [
                "Explore unconventional associations",
                "Challenge existing assumptions",
                "Synthesize diverse perspectives",
                "Embrace ambiguity and paradox"
            ]
        # Add more domain configurations as needed


class ReasoningStrategy(ABC):
    """Abstract base class for reasoning strategies"""
    
    @abstractmethod
    def can_handle(self, input_data: ReasoningInput) -> bool:
        """Check if this strategy can handle the input"""
        pass
    
    @abstractmethod
    async def reason(
        self,
        input_data: ReasoningInput,
        knowledge_base: DomainKnowledgeBase,
        consciousness_context: Dict[str, Any]
    ) -> ReasoningStep:
        """Execute reasoning step"""
        pass


class DeductiveReasoningStrategy(ReasoningStrategy):
    """Deductive reasoning: From general to specific"""
    
    def can_handle(self, input_data: ReasoningInput) -> bool:
        return True  # Deductive reasoning is always applicable
    
    async def reason(
        self,
        input_data: ReasoningInput,
        knowledge_base: DomainKnowledgeBase,
        consciousness_context: Dict[str, Any]
    ) -> ReasoningStep:
        start_time = time.time()
        
        # Apply domain-specific rules and general principles
        premises = self._extract_premises(input_data.query, knowledge_base)
        conclusion = self._apply_deductive_rules(premises, knowledge_base)
        confidence = self._calculate_deductive_confidence(premises, conclusion)
        
        duration_ms = (time.time() - start_time) * 1000
        
        return ReasoningStep(
            step_id=f"deductive_{int(time.time() * 1000)}",
            paradigm=ReasoningParadigm.DEDUCTIVE,
            description="Applied deductive reasoning from general principles",
            input_data={"premises": premises},
            output_data={"conclusion": conclusion, "logical_validity": confidence > 0.8},
            confidence=confidence,
            duration_ms=duration_ms,
            consciousness_state=consciousness_context
        )
    
    def _extract_premises(self, query: str, knowledge_base: DomainKnowledgeBase) -> List[str]:
        """Extract relevant premises from query and knowledge base"""
        # Simplified implementation - in reality, this would use NLP and knowledge retrieval
        return ["Premise extracted from domain knowledge", "Context-derived premise"]
    
    def _apply_deductive_rules(self, premises: List[str], knowledge_base: DomainKnowledgeBase) -> str:
        """Apply deductive logical rules"""
        return f"Conclusion derived from {len(premises)} premises using domain rules"
    
    def _calculate_deductive_confidence(self, premises: List[str], conclusion: str) -> float:
        """Calculate confidence in deductive conclusion"""
        return min(0.95, 0.7 + 0.05 * len(premises))  # Simplified confidence calculation


class InductiveReasoningStrategy(ReasoningStrategy):
    """Inductive reasoning: From specific to general"""
    
    def can_handle(self, input_data: ReasoningInput) -> bool:
        return "pattern" in input_data.query.lower() or "examples" in input_data.query.lower()
    
    async def reason(
        self,
        input_data: ReasoningInput,
        knowledge_base: DomainKnowledgeBase,
        consciousness_context: Dict[str, Any]
    ) -> ReasoningStep:
        start_time = time.time()
        
        # Identify patterns and generalize
        examples = self._extract_examples(input_data.query, input_data.context)
        pattern = self._identify_pattern(examples, knowledge_base)
        generalization = self._create_generalization(pattern, knowledge_base)
        confidence = self._calculate_inductive_confidence(examples, pattern)
        
        duration_ms = (time.time() - start_time) * 1000
        
        return ReasoningStep(
            step_id=f"inductive_{int(time.time() * 1000)}",
            paradigm=ReasoningParadigm.INDUCTIVE,
            description="Applied inductive reasoning to identify patterns",
            input_data={"examples": examples},
            output_data={"pattern": pattern, "generalization": generalization},
            confidence=confidence,
            duration_ms=duration_ms,
            consciousness_state=consciousness_context
        )
    
    def _extract_examples(self, query: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract examples from input"""
        return [{"example": "sample data point"}]  # Simplified
    
    def _identify_pattern(self, examples: List[Dict[str, Any]], knowledge_base: DomainKnowledgeBase) -> Dict[str, Any]:
        """Identify patterns in examples"""
        return {"pattern_type": "sequence", "strength": 0.8}
    
    def _create_generalization(self, pattern: Dict[str, Any], knowledge_base: DomainKnowledgeBase) -> str:
        """Create generalization from pattern"""
        return f"General rule based on {pattern['pattern_type']} pattern"
    
    def _calculate_inductive_confidence(self, examples: List[Dict[str, Any]], pattern: Dict[str, Any]) -> float:
        """Calculate confidence in inductive conclusion"""
        return min(0.85, 0.5 + 0.1 * len(examples) + 0.3 * pattern.get("strength", 0))


class AbductiveReasoningStrategy(ReasoningStrategy):
    """Abductive reasoning: Best explanation for observations"""
    
    def can_handle(self, input_data: ReasoningInput) -> bool:
        return "explain" in input_data.query.lower() or "why" in input_data.query.lower()
    
    async def reason(
        self,
        input_data: ReasoningInput,
        knowledge_base: DomainKnowledgeBase,
        consciousness_context: Dict[str, Any]
    ) -> ReasoningStep:
        start_time = time.time()
        
        # Generate and evaluate explanations
        observations = self._extract_observations(input_data.query, input_data.context)
        hypotheses = await self._generate_hypotheses(observations, knowledge_base)
        best_explanation = self._select_best_explanation(hypotheses, knowledge_base)
        confidence = self._calculate_abductive_confidence(observations, best_explanation)
        
        duration_ms = (time.time() - start_time) * 1000
        
        return ReasoningStep(
            step_id=f"abductive_{int(time.time() * 1000)}",
            paradigm=ReasoningParadigm.ABDUCTIVE,
            description="Applied abductive reasoning to find best explanation",
            input_data={"observations": observations},
            output_data={"best_explanation": best_explanation, "alternative_hypotheses": hypotheses},
            confidence=confidence,
            duration_ms=duration_ms,
            consciousness_state=consciousness_context
        )
    
    def _extract_observations(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Extract observations from input"""
        return ["Observation from context", "Derived observation"]
    
    async def _generate_hypotheses(self, observations: List[str], knowledge_base: DomainKnowledgeBase) -> List[Dict[str, Any]]:
        """Generate possible explanations"""
        return [
            {"hypothesis": "Primary explanation", "plausibility": 0.8},
            {"hypothesis": "Alternative explanation", "plausibility": 0.6}
        ]
    
    def _select_best_explanation(self, hypotheses: List[Dict[str, Any]], knowledge_base: DomainKnowledgeBase) -> Dict[str, Any]:
        """Select most plausible explanation"""
        return max(hypotheses, key=lambda h: h["plausibility"])
    
    def _calculate_abductive_confidence(self, observations: List[str], explanation: Dict[str, Any]) -> float:
        """Calculate confidence in abductive conclusion"""
        return explanation["plausibility"] * (0.7 + 0.05 * len(observations))


class DomainReasoningEngine(ConsciousnessAwareModule):
    """Main domain-specific reasoning engine"""
    
    def __init__(
        self,
        consciousness_state: Optional[Dict[str, Any]] = None,
        max_reasoning_time: float = 30.0,
        max_steps: int = 10
    ):
        super().__init__(consciousness_state)
        self.max_reasoning_time = max_reasoning_time
        self.max_steps = max_steps
        
        # Initialize knowledge bases for each domain
        self.knowledge_bases = {
            domain: DomainKnowledgeBase(domain) for domain in ReasoningDomain
        }
        
        # Initialize reasoning strategies
        self.strategies = [
            DeductiveReasoningStrategy(),
            InductiveReasoningStrategy(),
            AbductiveReasoningStrategy()
            # Add more strategies as needed
        ]
        
        # Domain detection patterns (simplified)
        self.domain_patterns = {
            ReasoningDomain.MEDICAL: ["symptom", "diagnosis", "treatment", "patient", "disease"],
            ReasoningDomain.LEGAL: ["law", "statute", "contract", "liability", "court"],
            ReasoningDomain.SCIENTIFIC: ["hypothesis", "experiment", "data", "theory", "research"],
            ReasoningDomain.TECHNICAL: ["algorithm", "system", "implementation", "code", "architecture"],
            ReasoningDomain.CREATIVE: ["creative", "art", "design", "innovative", "imagination"],
            ReasoningDomain.BUSINESS: ["strategy", "market", "profit", "business", "competition"],
            ReasoningDomain.MATHEMATICAL: ["equation", "proof", "theorem", "calculation", "formula"],
            ReasoningDomain.PHILOSOPHICAL: ["ethics", "meaning", "existence", "consciousness", "morality"]
        }
        
        self.executor = ThreadPoolExecutor(max_workers=4)
        logger.info("Domain Reasoning Engine initialized")
    
    def _detect_domain(self, query: str, context: Dict[str, Any], hint: Optional[ReasoningDomain] = None) -> ReasoningDomain:
        """Detect the most appropriate domain for reasoning"""
        if hint:
            return hint
        
        query_lower = query.lower()
        domain_scores = {}
        
        for domain, patterns in self.domain_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores.key, key=domain_scores.get)
        
        return ReasoningDomain.GENERAL
    
    def _select_strategies(
        self,
        input_data: ReasoningInput,
        domain: ReasoningDomain,
        knowledge_base: DomainKnowledgeBase
    ) -> List[ReasoningStrategy]:
        """Select appropriate reasoning strategies"""
        if input_data.paradigm_preference:
            # Filter strategies based on preference
            strategy_map = {
                ReasoningParadigm.DEDUCTIVE: DeductiveReasoningStrategy,
                ReasoningParadigm.INDUCTIVE: InductiveReasoningStrategy,
                ReasoningParadigm.ABDUCTIVE: AbductiveReasoningStrategy
            }
            return [strategy for strategy in self.strategies 
                    if any(isinstance(strategy, strategy_map[p]) for p in input_data.paradigm_preference 
                          if p in strategy_map)]
        
        # Use domain weights to select strategies
        suitable_strategies = [s for s in self.strategies if s.can_handle(input_data)]
        return suitable_strategies
    
    def _determine_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Determine confidence level from numerical confidence"""
        for level in ConfidenceLevel:
            if level.min_val <= confidence <= level.max_val:
                return level
        return ConfidenceLevel.MEDIUM
    
    async def reason(self, input_data: ReasoningInput) -> ReasoningOutput:
        """Execute domain-specific reasoning process"""
        start_time = time.time()
        
        try:
            # Update consciousness context
            await self._update_consciousness({
                'reasoning_input': input_data.query,
                'domain_detection': 'starting',
                'reasoning_strategies': 'initializing'
            })
            
            # Detect domain
            domain = self._detect_domain(input_data.query, input_data.context, input_data.domain_hint)
            knowledge_base = self.knowledge_bases[domain]
            
            logger.info(f"Reasoning for domain: {domain.name}")
            
            # Select reasoning strategies
            strategies = self._select_strategies(input_data, domain, knowledge_base)
            
            if not strategies:
                strategies = [self.strategies[0]]  # Fallback to deductive reasoning
            
            # Execute reasoning steps
            reasoning_steps = []
            consciousness_context = {
                'domain': domain.name,
                'strategies_selected': [s.__class__.__name__ for s in strategies],
                'step_count': 0
            }
            
            for i, strategy in enumerate(strategies):
                if len(reasoning_steps) >= self.max_steps:
                    break
                
                if time.time() - start_time > self.max_reasoning_time:
                    break
                
                consciousness_context['current_strategy'] = strategy.__class__.__name__
                consciousness_context['step_count'] = i + 1
                
                step = await strategy.reason(input_data, knowledge_base, consciousness_context)
                reasoning_steps.append(step)
                
                # Update consciousness with step result
                await self._update_consciousness({
                    'reasoning_step_completed': step.step_id,
                    'step_confidence': step.confidence,
                    'paradigm_used': step.paradigm.name
                })
            
            # Synthesize final conclusion
            conclusion, final_confidence = self._synthesize_conclusion(reasoning_steps, knowledge_base)
            confidence_level = self._determine_confidence_level(final_confidence)
            
            # Generate explanation
            explanation = self._generate_explanation(reasoning_steps, domain, knowledge_base)
            
            # Generate alternatives
            alternatives = self._generate_alternatives(reasoning_steps, knowledge_base)
            
            # Identify uncertainty factors
            uncertainty_factors = self._identify_uncertainty_factors(reasoning_steps, final_confidence)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Final consciousness update
            consciousness_insights = await self._get_consciousness_insights({
                'conclusion': conclusion,
                'confidence': final_confidence,
                'paradigms_used': [step.paradigm.name for step in reasoning_steps],
                'processing_time_ms': processing_time_ms
            })
            
            return ReasoningOutput(
                conclusion=conclusion,
                domain=domain,
                paradigms_used=[step.paradigm for step in reasoning_steps],
                confidence=final_confidence,
                confidence_level=confidence_level,
                explanation=explanation,
                reasoning_steps=reasoning_steps,
                alternative_conclusions=alternatives,
                sources=knowledge_base.heuristics,
                uncertainty_factors=uncertainty_factors,
                consciousness_insights=consciousness_insights,
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            logger.error(f"Reasoning failed: {str(e)}")
            # Return error result
            return ReasoningOutput(
                conclusion=f"Reasoning failed: {str(e)}",
                domain=ReasoningDomain.GENERAL,
                paradigms_used=[],
                confidence=0.0,
                confidence_level=ConfidenceLevel.VERY_LOW,
                explanation="An error occurred during reasoning",
                reasoning_steps=[],
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    def _synthesize_conclusion(
        self,
        reasoning_steps: List[ReasoningStep],
        knowledge_base: DomainKnowledgeBase
    ) -> Tuple[str, float]:
        """Synthesize final conclusion from reasoning steps"""
        if not reasoning_steps:
            return "No conclusion could be reached", 0.0
        
        # Weight conclusions by confidence and domain-specific paradigm weights
        weighted_conclusions = []
        total_weight = 0.0
        
        for step in reasoning_steps:
            paradigm_weight = knowledge_base.paradigm_weights.get(step.paradigm, 0.5)
            weight = step.confidence * paradigm_weight
            weighted_conclusions.append((step.output_data.get("conclusion", ""), weight))
            total_weight += weight
        
        if total_weight == 0:
            return reasoning_steps[-1].output_data.get("conclusion", "Unable to determine"), 0.1
        
        # For simplicity, return the highest weighted conclusion
        best_conclusion = max(weighted_conclusions, key=lambda x: x[1])
        final_confidence = min(0.95, total_weight / len(reasoning_steps))
        
        return best_conclusion[0], final_confidence
    
    def _generate_explanation(
        self,
        reasoning_steps: List[ReasoningStep],
        domain: ReasoningDomain,
        knowledge_base: DomainKnowledgeBase
    ) -> str:
        """Generate human-readable explanation of reasoning process"""
        explanation_parts = [
            f"Applied {domain.name.lower()} domain reasoning using {len(reasoning_steps)} steps:"
        ]
        
        for i, step in enumerate(reasoning_steps, 1):
            explanation_parts.append(
                f"{i}. {step.paradigm.name.title()} reasoning: {step.description} "
                f"(confidence: {step.confidence:.2f})"
            )
        
        if knowledge_base.heuristics:
            explanation_parts.append("Considered domain-specific heuristics:")
            explanation_parts.extend(f"- {h}" for h in knowledge_base.heuristics[:3])
        
        return "\n".join(explanation_parts)
    
    def _generate_alternatives(
        self,
        reasoning_steps: List[ReasoningStep],
        knowledge_base: DomainKnowledgeBase
    ) -> List[Dict[str, Any]]:
        """Generate alternative conclusions"""
        alternatives = []
        
        for step in reasoning_steps:
            if "alternative_hypotheses" in step.output_data:
                for alt in step.output_data["alternative_hypotheses"]:
                    alternatives.append({
                        "conclusion": alt.get("hypothesis", "Alternative explanation"),
                        "confidence": alt.get("plausibility", 0.5),
                        "reasoning": f"From {step.paradigm.name.lower()} reasoning"
                    })
        
        return alternatives[:3]  # Limit to top 3 alternatives
    
    def _identify_uncertainty_factors(
        self,
        reasoning_steps: List[ReasoningStep],
        final_confidence: float
    ) -> List[str]:
        """Identify factors that contribute to uncertainty"""
        factors = []
        
        if final_confidence < 0.6:
            factors.append("Low overall confidence in reasoning chain")
        
        low_confidence_steps = [s for s in reasoning_steps if s.confidence < 0.5]
        if low_confidence_steps:
            factors.append(f"{len(low_confidence_steps)} reasoning steps had low confidence")
        
        if len(reasoning_steps) < 2:
            factors.append("Limited reasoning perspectives applied")
        
        avg_duration = sum(s.duration_ms for s in reasoning_steps) / len(reasoning_steps)
        if avg_duration < 50:
            factors.append("Rapid reasoning may have missed nuances")
        
        return factors
    
    async def batch_reason(self, inputs: List[ReasoningInput]) -> List[ReasoningOutput]:
        """Process multiple reasoning requests in batch"""
        logger.info(f"Processing batch of {len(inputs)} reasoning requests")
        
        # Use asyncio to process requests concurrently
        tasks = [self.reason(input_data) for input_data in inputs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch reasoning failed for input {i}: {result}")
                processed_results.append(ReasoningOutput(
                    conclusion=f"Processing failed: {str(result)}",
                    domain=ReasoningDomain.GENERAL,
                    paradigms_used=[],
                    confidence=0.0,
                    confidence_level=ConfidenceLevel.VERY_LOW,
                    explanation="Batch processing error",
                    reasoning_steps=[]
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_domain_capabilities(self) -> Dict[ReasoningDomain, Dict[str, Any]]:
        """Get information about domain-specific capabilities"""
        capabilities = {}
        
        for domain, kb in self.knowledge_bases.items():
            capabilities[domain] = {
                "supported_paradigms": list(kb.paradigm_weights.keys()),
                "paradigm_weights": kb.paradigm_weights,
                "heuristics_count": len(kb.heuristics),
                "sample_heuristics": kb.heuristics[:3]
            }
        
        return capabilities
    
    async def explain_reasoning_process(self, output: ReasoningOutput) -> Dict[str, Any]:
        """Provide detailed explanation of reasoning process"""
        return {
            "domain_analysis": {
                "detected_domain": output.domain.name,
                "domain_confidence": "High" if len([p for p in self.domain_patterns[output.domain] 
                                                 if p in output.consciousness_insights.get('reasoning_input', '').lower()]) > 1 else "Medium"
            },
            "paradigm_analysis": {
                "paradigms_used": [p.name for p in output.paradigms_used],
                "paradigm_effectiveness": {p.name: step.confidence for p, step in zip(output.paradigms_used, output.reasoning_steps)}
            },
            "confidence_breakdown": {
                "overall_confidence": output.confidence,
                "confidence_level": output.confidence_level.name,
                "step_confidences": [step.confidence for step in output.reasoning_steps],
                "uncertainty_factors": output.uncertainty_factors
            },
            "consciousness_insights": output.consciousness_insights,
            "performance_metrics": {
                "processing_time_ms": output.processing_time_ms,
                "steps_count": len(output.reasoning_steps),
                "average_step_time": sum(s.duration_ms for s in output.reasoning_steps) / len(output.reasoning_steps) if output.reasoning_steps else 0
            }
        }


# Example usage and testing
async def main():
    """Example usage of domain reasoning engine"""
    engine = DomainReasoningEngine()
    
    # Medical reasoning example
    medical_input = ReasoningInput(
        query="Patient presents with chest pain, shortness of breath, and elevated troponin levels. What is the most likely diagnosis?",
        context={"patient_age": 65, "symptoms_duration": "2 hours"},
        domain_hint=ReasoningDomain.MEDICAL,
        required_confidence=0.7
    )
    
    result = await engine.reason(medical_input)
    print(f"Medical Reasoning Result:")
    print(f"Domain: {result.domain.name}")
    print(f"Conclusion: {result.conclusion}")
    print(f"Confidence: {result.confidence:.2f} ({result.confidence_level.name})")
    print(f"Explanation: {result.explanation}")
    print(f"Processing time: {result.processing_time_ms:.1f}ms")
    print("-" * 50)
    
    # Legal reasoning example
    legal_input = ReasoningInput(
        query="Is a contract void if one party was under duress when signing?",
        context={"contract_type": "employment", "duress_type": "economic"},
        domain_hint=ReasoningDomain.LEGAL
    )
    
    result = await engine.reason(legal_input)
    print(f"Legal Reasoning Result:")
    print(f"Domain: {result.domain.name}")
    print(f"Conclusion: {result.conclusion}")
    print(f"Confidence: {result.confidence:.2f} ({result.confidence_level.name})")
    print(f"Processing time: {result.processing_time_ms:.1f}ms")
    
    # Get domain capabilities
    capabilities = engine.get_domain_capabilities()
    print(f"\nSupported domains: {list(capabilities.keys())}")


if __name__ == "__main__":
    asyncio.run(main())