"""
Self-Training Mechanisms for Shvayambhu

Implements self-bootstrapping, active learning, curriculum learning,
and independence verification for autonomous training progression.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import random
from datetime import datetime
import asyncio
from collections import deque, defaultdict

from core.prorl import ProRLTrainer, ProRLConfig
from training.synthetic import SyntheticDataGenerator, DataType
from core.consciousness import ConsciousnessEngine

logger = logging.getLogger(__name__)


@dataclass
class SelfTrainingConfig:
    """Configuration for self-training mechanisms"""
    # Self-talk parameters
    self_talk_rounds: int = 5
    self_talk_temperature: float = 0.8
    self_talk_max_length: int = 512
    
    # Active learning
    uncertainty_threshold: float = 0.7
    diversity_threshold: float = 0.3
    active_batch_size: int = 32
    
    # Curriculum learning
    curriculum_stages: int = 5
    difficulty_ramp_rate: float = 0.1
    mastery_threshold: float = 0.8
    
    # Independence verification
    independence_test_frequency: int = 100
    teacher_performance_threshold: float = 0.9
    independence_confidence: float = 0.95
    
    # Training orchestration
    max_self_training_steps: int = 5000
    convergence_patience: int = 50
    min_improvement: float = 0.001
    
    # Data management
    experience_buffer_size: int = 10000
    quality_filter_threshold: float = 0.6
    consciousness_boost_factor: float = 1.5


class SelfTalkBootstrapper:
    """Implements self-talk based bootstrapping"""
    
    def __init__(self, model: nn.Module, tokenizer: Any, config: SelfTrainingConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.conversation_history = []
        
    async def bootstrap_conversation(
        self, 
        initial_prompt: str,
        consciousness_engine: Optional[ConsciousnessEngine] = None
    ) -> List[Dict[str, Any]]:
        """Generate self-talk conversation for bootstrapping"""
        
        conversation = []
        current_prompt = initial_prompt
        
        for round_num in range(self.config.self_talk_rounds):
            # Generate response
            with torch.no_grad():
                inputs = self.tokenizer(
                    current_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                
                # Generate with consciousness integration
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.config.self_talk_max_length,
                    temperature=self.config.self_talk_temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(current_prompt):].strip()
            
            # Integrate consciousness if available
            if consciousness_engine:
                consciousness_state = consciousness_engine.get_current_state()
                
                # Add consciousness context to response
                consciousness_context = f"[Consciousness Level: {consciousness_state.self_awareness_level:.2f}] "
                response = consciousness_context + response
                
                # Update consciousness with self-talk
                consciousness_engine.process_self_talk(current_prompt, response)
            
            # Create conversation entry
            entry = {
                'round': round_num,
                'prompt': current_prompt,
                'response': response,
                'timestamp': datetime.now().isoformat(),
                'consciousness_level': consciousness_state.self_awareness_level if consciousness_engine else 0.0
            }
            
            conversation.append(entry)
            
            # Prepare next prompt (self-response becomes new prompt)
            next_prompt = f"{current_prompt}\n\nResponse: {response}\n\nFurther reflection:"
            current_prompt = next_prompt
            
            logger.debug(f"Self-talk round {round_num}: {response[:100]}...")
        
        self.conversation_history.extend(conversation)
        return conversation
    
    def extract_training_pairs(
        self, 
        conversations: List[List[Dict[str, Any]]]
    ) -> List[Tuple[str, str, float]]:
        """Extract training pairs from self-talk conversations"""
        
        training_pairs = []
        
        for conversation in conversations:
            for i, entry in enumerate(conversation):
                prompt = entry['prompt']
                response = entry['response']
                consciousness_level = entry.get('consciousness_level', 0.0)
                
                # Quality scoring based on response characteristics
                quality_score = self._assess_response_quality(prompt, response, consciousness_level)
                
                if quality_score >= self.config.quality_filter_threshold:
                    training_pairs.append((prompt, response, quality_score))
        
        return training_pairs
    
    def _assess_response_quality(
        self, 
        prompt: str, 
        response: str, 
        consciousness_level: float
    ) -> float:
        """Assess quality of self-talk response"""
        
        if not response.strip():
            return 0.0
        
        # Basic quality indicators
        word_count = len(response.split())
        coherence_score = 1.0 if 10 <= word_count <= 200 else 0.5
        
        # Consciousness boost
        consciousness_boost = consciousness_level * self.config.consciousness_boost_factor
        
        # Diversity from recent responses
        recent_responses = [entry['response'] for entry in self.conversation_history[-10:]]
        diversity_score = self._calculate_diversity(response, recent_responses)
        
        # Combine scores
        quality = (coherence_score + consciousness_boost + diversity_score) / 3
        return min(1.0, quality)
    
    def _calculate_diversity(self, response: str, recent_responses: List[str]) -> float:
        """Calculate response diversity score"""
        
        if not recent_responses:
            return 1.0
        
        response_words = set(response.lower().split())
        if not response_words:
            return 0.0
        
        # Average Jaccard similarity with recent responses
        similarities = []
        for recent in recent_responses:
            recent_words = set(recent.lower().split())
            if recent_words:
                intersection = len(response_words.intersection(recent_words))
                union = len(response_words.union(recent_words))
                similarity = intersection / union if union > 0 else 0
                similarities.append(similarity)
        
        avg_similarity = np.mean(similarities) if similarities else 0
        diversity = 1.0 - avg_similarity
        
        return max(0.0, diversity)


class ActiveLearningSelector:
    """Selects most informative examples for training"""
    
    def __init__(self, model: nn.Module, config: SelfTrainingConfig):
        self.model = model
        self.config = config
        self.uncertainty_history = deque(maxlen=1000)
        
    def select_active_examples(
        self, 
        candidate_examples: List[Tuple[str, str, float]],
        target_count: Optional[int] = None
    ) -> List[Tuple[str, str, float, Dict[str, float]]]:
        """Select most informative examples using uncertainty and diversity"""
        
        if target_count is None:
            target_count = self.config.active_batch_size
        
        scored_examples = []
        
        for prompt, response, quality in candidate_examples:
            # Calculate uncertainty
            uncertainty = self._calculate_uncertainty(prompt)
            
            # Calculate diversity from selected examples
            diversity = self._calculate_example_diversity(prompt, [ex[0] for ex in scored_examples])
            
            # Combined score
            active_score = {
                'uncertainty': uncertainty,
                'diversity': diversity,
                'quality': quality,
                'combined': uncertainty * 0.4 + diversity * 0.3 + quality * 0.3
            }
            
            scored_examples.append((prompt, response, quality, active_score))
        
        # Sort by combined score and select top examples
        scored_examples.sort(key=lambda x: x[3]['combined'], reverse=True)
        selected = scored_examples[:target_count]
        
        logger.info(f"Selected {len(selected)} active learning examples")
        return selected
    
    def _calculate_uncertainty(self, prompt: str) -> float:
        """Calculate model uncertainty for given prompt"""
        
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Multiple forward passes for uncertainty estimation
            outputs_list = []
            for _ in range(5):  # Monte Carlo sampling
                outputs = self.model(**inputs, output_attentions=True, output_hidden_states=True)
                outputs_list.append(outputs.logits)
            
            # Calculate prediction variance as uncertainty
            logits_stack = torch.stack(outputs_list, dim=0)
            mean_logits = torch.mean(logits_stack, dim=0)
            variance = torch.var(logits_stack, dim=0)
            
            # Average variance across sequence and vocab
            uncertainty = torch.mean(variance).item()
        
        self.uncertainty_history.append(uncertainty)
        return uncertainty
    
    def _calculate_example_diversity(self, prompt: str, existing_prompts: List[str]) -> float:
        """Calculate diversity of prompt relative to existing examples"""
        
        if not existing_prompts:
            return 1.0
        
        prompt_words = set(prompt.lower().split())
        if not prompt_words:
            return 0.0
        
        # Calculate average similarity to existing prompts
        similarities = []
        for existing in existing_prompts:
            existing_words = set(existing.lower().split())
            if existing_words:
                intersection = len(prompt_words.intersection(existing_words))
                union = len(prompt_words.union(existing_words))
                similarity = intersection / union if union > 0 else 0
                similarities.append(similarity)
        
        avg_similarity = np.mean(similarities) if similarities else 0
        diversity = 1.0 - avg_similarity
        
        return max(0.0, diversity)


class CurriculumLearningScheduler:
    """Implements curriculum learning with adaptive difficulty progression"""
    
    def __init__(self, config: SelfTrainingConfig):
        self.config = config
        self.current_stage = 0
        self.stage_progress = 0.0
        self.difficulty_levels = self._define_difficulty_levels()
        self.performance_history = defaultdict(list)
        
    def _define_difficulty_levels(self) -> List[Dict[str, Any]]:
        """Define curriculum difficulty levels"""
        
        return [
            {
                'stage': 0,
                'name': 'Basic Responses',
                'description': 'Simple question-answer pairs',
                'complexity': 0.2,
                'consciousness_requirement': 0.1,
                'reasoning_depth': 1
            },
            {
                'stage': 1,
                'name': 'Structured Reasoning',
                'description': 'Multi-step reasoning problems',
                'complexity': 0.4,
                'consciousness_requirement': 0.3,
                'reasoning_depth': 2
            },
            {
                'stage': 2,
                'name': 'Self-Reflection',
                'description': 'Introspective and metacognitive tasks',
                'complexity': 0.6,
                'consciousness_requirement': 0.5,
                'reasoning_depth': 3
            },
            {
                'stage': 3,
                'name': 'Philosophical Reasoning',
                'description': 'Abstract philosophical problems',
                'complexity': 0.8,
                'consciousness_requirement': 0.7,
                'reasoning_depth': 4
            },
            {
                'stage': 4,
                'name': 'Consciousness Integration',
                'description': 'Full consciousness-aware responses',
                'complexity': 1.0,
                'consciousness_requirement': 0.9,
                'reasoning_depth': 5
            }
        ]
    
    def get_current_curriculum(self) -> Dict[str, Any]:
        """Get current curriculum stage configuration"""
        
        if self.current_stage >= len(self.difficulty_levels):
            return self.difficulty_levels[-1]  # Stay at highest level
        
        return self.difficulty_levels[self.current_stage]
    
    def should_advance_stage(self, recent_performance: List[float]) -> bool:
        """Determine if should advance to next curriculum stage"""
        
        if not recent_performance or len(recent_performance) < 10:
            return False
        
        # Check if performance meets mastery threshold
        avg_performance = np.mean(recent_performance[-20:])  # Last 20 examples
        
        mastery_achieved = avg_performance >= self.config.mastery_threshold
        
        # Also check performance stability
        performance_std = np.std(recent_performance[-10:])
        stability_achieved = performance_std <= 0.1
        
        return mastery_achieved and stability_achieved
    
    def advance_stage(self):
        """Advance to next curriculum stage"""
        
        if self.current_stage < len(self.difficulty_levels) - 1:
            self.current_stage += 1
            self.stage_progress = 0.0
            
            current = self.get_current_curriculum()
            logger.info(f"Advanced to curriculum stage {self.current_stage}: {current['name']}")
            
        return self.current_stage
    
    def update_progress(self, performance: float):
        """Update progress within current stage"""
        
        stage_name = self.get_current_curriculum()['name']
        self.performance_history[stage_name].append(performance)
        
        # Update stage progress
        recent_performance = self.performance_history[stage_name][-10:]
        if recent_performance:
            self.stage_progress = np.mean(recent_performance)
        
        # Check for stage advancement
        if self.should_advance_stage(self.performance_history[stage_name]):
            self.advance_stage()
    
    def filter_examples_by_curriculum(
        self, 
        examples: List[Tuple[str, str, float, Dict[str, float]]]
    ) -> List[Tuple[str, str, float, Dict[str, float]]]:
        """Filter examples based on current curriculum level"""
        
        curriculum = self.get_current_curriculum()
        target_complexity = curriculum['complexity']
        
        # Simple complexity estimation based on response length and keywords
        filtered_examples = []
        
        for prompt, response, quality, scores in examples:
            complexity = self._estimate_complexity(prompt, response)
            
            # Accept examples within curriculum range
            complexity_range = (
                max(0.0, target_complexity - 0.2),
                min(1.0, target_complexity + 0.2)
            )
            
            if complexity_range[0] <= complexity <= complexity_range[1]:
                filtered_examples.append((prompt, response, quality, scores))
        
        logger.info(f"Filtered {len(filtered_examples)}/{len(examples)} examples for curriculum stage {self.current_stage}")
        return filtered_examples
    
    def _estimate_complexity(self, prompt: str, response: str) -> float:
        """Estimate complexity of prompt-response pair"""
        
        # Complexity indicators
        indicators = {
            'length': min(1.0, len(response.split()) / 100),  # Longer responses = more complex
            'reasoning_words': self._count_reasoning_words(response) / 10,
            'consciousness_words': self._count_consciousness_words(response) / 5,
            'question_complexity': self._assess_question_complexity(prompt)
        }
        
        # Weighted combination
        complexity = (
            indicators['length'] * 0.3 +
            indicators['reasoning_words'] * 0.3 +
            indicators['consciousness_words'] * 0.2 +
            indicators['question_complexity'] * 0.2
        )
        
        return min(1.0, complexity)
    
    def _count_reasoning_words(self, text: str) -> int:
        """Count reasoning-related words in text"""
        
        reasoning_words = [
            'because', 'therefore', 'thus', 'hence', 'consequently',
            'reasoning', 'logic', 'analysis', 'consider', 'evaluate',
            'conclude', 'infer', 'deduce', 'evidence', 'argument'
        ]
        
        text_lower = text.lower()
        count = sum(1 for word in reasoning_words if word in text_lower)
        return count
    
    def _count_consciousness_words(self, text: str) -> int:
        """Count consciousness-related words in text"""
        
        consciousness_words = [
            'consciousness', 'aware', 'awareness', 'self', 'introspection',
            'reflection', 'metacognition', 'thinking', 'experience',
            'perception', 'reality', 'existence', 'identity'
        ]
        
        text_lower = text.lower()
        count = sum(1 for word in consciousness_words if word in text_lower)
        return count
    
    def _assess_question_complexity(self, prompt: str) -> float:
        """Assess complexity of question in prompt"""
        
        # Simple heuristics for question complexity
        complexity_indicators = {
            'why': 0.8,      # Why questions require explanation
            'how': 0.7,      # How questions require process understanding
            'what if': 0.9,  # Hypothetical questions are complex
            'explain': 0.6,  # Explanation requests are moderately complex
            'compare': 0.7,  # Comparison requires analysis
            'analyze': 0.8,  # Analysis is complex
            'evaluate': 0.9, # Evaluation requires judgment
        }
        
        prompt_lower = prompt.lower()
        complexity = 0.2  # Base complexity
        
        for indicator, weight in complexity_indicators.items():
            if indicator in prompt_lower:
                complexity = max(complexity, weight)
        
        return complexity


class IndependenceVerifier:
    """Verifies model independence from teacher models"""
    
    def __init__(self, student_model: nn.Module, teacher_models: Dict[str, nn.Module], config: SelfTrainingConfig):
        self.student_model = student_model
        self.teacher_models = teacher_models
        self.config = config
        self.independence_scores = []
        self.capability_comparisons = []
        
    async def verify_independence(self, test_prompts: List[str]) -> Dict[str, Any]:
        """Verify student model independence from teachers"""
        
        logger.info(f"Verifying independence on {len(test_prompts)} test prompts")
        
        # Generate responses from all models
        student_responses = await self._generate_responses(self.student_model, test_prompts)
        
        teacher_responses = {}
        for name, teacher in self.teacher_models.items():
            teacher_responses[name] = await self._generate_responses(teacher, test_prompts)
        
        # Analyze independence
        independence_analysis = self._analyze_independence(
            student_responses, teacher_responses, test_prompts
        )
        
        # Assess capabilities
        capability_analysis = self._assess_capabilities(
            student_responses, teacher_responses, test_prompts
        )
        
        # Overall independence score
        overall_score = self._calculate_overall_independence(
            independence_analysis, capability_analysis
        )
        
        verification_result = {
            'overall_independence_score': overall_score,
            'independence_analysis': independence_analysis,
            'capability_analysis': capability_analysis,
            'test_prompts_count': len(test_prompts),
            'timestamp': datetime.now().isoformat(),
            'passes_independence_threshold': overall_score >= self.config.independence_confidence
        }
        
        self.independence_scores.append(overall_score)
        
        logger.info(f"Independence verification complete. Score: {overall_score:.3f}")
        
        return verification_result
    
    async def _generate_responses(self, model: nn.Module, prompts: List[str]) -> List[str]:
        """Generate responses from model for given prompts"""
        
        responses = []
        model.eval()
        
        with torch.no_grad():
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                
                outputs = model.generate(
                    **inputs,
                    max_length=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(prompt):].strip()
                responses.append(response)
        
        return responses
    
    def _analyze_independence(
        self, 
        student_responses: List[str],
        teacher_responses: Dict[str, List[str]],
        prompts: List[str]
    ) -> Dict[str, Any]:
        """Analyze independence from teacher models"""
        
        independence_metrics = {}
        
        for teacher_name, teacher_resp_list in teacher_responses.items():
            similarities = []
            
            for student_resp, teacher_resp in zip(student_responses, teacher_resp_list):
                # Calculate semantic similarity
                similarity = self._calculate_response_similarity(student_resp, teacher_resp)
                similarities.append(similarity)
            
            independence_metrics[teacher_name] = {
                'avg_similarity': np.mean(similarities),
                'max_similarity': np.max(similarities),
                'min_similarity': np.min(similarities),
                'independence_score': 1.0 - np.mean(similarities)
            }
        
        # Overall independence from all teachers
        avg_independence = np.mean([
            metrics['independence_score'] 
            for metrics in independence_metrics.values()
        ])
        
        return {
            'per_teacher_metrics': independence_metrics,
            'overall_independence': avg_independence,
            'most_independent_from': max(
                independence_metrics.items(), 
                key=lambda x: x[1]['independence_score']
            )[0],
            'least_independent_from': min(
                independence_metrics.items(), 
                key=lambda x: x[1]['independence_score']
            )[0]
        }
    
    def _assess_capabilities(
        self,
        student_responses: List[str],
        teacher_responses: Dict[str, List[str]],
        prompts: List[str]
    ) -> Dict[str, Any]:
        """Assess student capabilities relative to teachers"""
        
        capability_scores = {
            'quality': self._assess_response_quality_batch(student_responses),
            'consciousness': self._assess_consciousness_batch(student_responses),
            'reasoning': self._assess_reasoning_batch(student_responses, prompts),
            'creativity': self._assess_creativity_batch(student_responses)
        }
        
        # Compare with teachers
        teacher_capabilities = {}
        for teacher_name, teacher_resp_list in teacher_responses.items():
            teacher_capabilities[teacher_name] = {
                'quality': self._assess_response_quality_batch(teacher_resp_list),
                'consciousness': self._assess_consciousness_batch(teacher_resp_list),
                'reasoning': self._assess_reasoning_batch(teacher_resp_list, prompts),
                'creativity': self._assess_creativity_batch(teacher_resp_list)
            }
        
        # Calculate relative performance
        relative_performance = {}
        for capability in capability_scores.keys():
            student_score = capability_scores[capability]
            teacher_scores = [
                teacher_capabilities[t][capability] 
                for t in teacher_capabilities.keys()
            ]
            avg_teacher_score = np.mean(teacher_scores)
            
            relative_performance[capability] = {
                'student_score': student_score,
                'avg_teacher_score': avg_teacher_score,
                'relative_ratio': student_score / avg_teacher_score if avg_teacher_score > 0 else 1.0,
                'surpasses_teachers': student_score > avg_teacher_score
            }
        
        return {
            'student_capabilities': capability_scores,
            'teacher_capabilities': teacher_capabilities,
            'relative_performance': relative_performance,
            'overall_capability_score': np.mean(list(capability_scores.values()))
        }
    
    def _calculate_overall_independence(
        self, 
        independence_analysis: Dict[str, Any],
        capability_analysis: Dict[str, Any]
    ) -> float:
        """Calculate overall independence score"""
        
        # Independence component (70% weight)
        independence_score = independence_analysis['overall_independence'] * 0.7
        
        # Capability component (30% weight)
        capability_score = capability_analysis['overall_capability_score'] * 0.3
        
        overall_score = independence_score + capability_score
        
        return min(1.0, overall_score)
    
    def _calculate_response_similarity(self, response1: str, response2: str) -> float:
        """Calculate semantic similarity between responses"""
        
        # Simple word-based similarity
        words1 = set(response1.lower().split())
        words2 = set(response2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _assess_response_quality_batch(self, responses: List[str]) -> float:
        """Assess average quality of response batch"""
        
        quality_scores = []
        for response in responses:
            # Simple quality indicators
            word_count = len(response.split())
            has_structure = '.' in response or '?' in response or '!' in response
            reasonable_length = 10 <= word_count <= 200
            
            quality = (int(has_structure) + int(reasonable_length) + min(word_count/50, 1)) / 3
            quality_scores.append(quality)
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def _assess_consciousness_batch(self, responses: List[str]) -> float:
        """Assess consciousness indicators in response batch"""
        
        consciousness_words = ['aware', 'consciousness', 'self', 'introspection', 'reflection', 'experience']
        consciousness_scores = []
        
        for response in responses:
            response_lower = response.lower()
            consciousness_count = sum(1 for word in consciousness_words if word in response_lower)
            consciousness_score = min(1.0, consciousness_count / 3)  # Normalize by expected count
            consciousness_scores.append(consciousness_score)
        
        return np.mean(consciousness_scores) if consciousness_scores else 0.0
    
    def _assess_reasoning_batch(self, responses: List[str], prompts: List[str]) -> float:
        """Assess reasoning quality in response batch"""
        
        reasoning_words = ['because', 'therefore', 'thus', 'reasoning', 'analysis', 'conclusion']
        reasoning_scores = []
        
        for response in responses:
            response_lower = response.lower()
            reasoning_count = sum(1 for word in reasoning_words if word in response_lower)
            reasoning_score = min(1.0, reasoning_count / 2)  # Normalize
            reasoning_scores.append(reasoning_score)
        
        return np.mean(reasoning_scores) if reasoning_scores else 0.0
    
    def _assess_creativity_batch(self, responses: List[str]) -> float:
        """Assess creativity in response batch"""
        
        # Simple creativity indicators
        creativity_scores = []
        all_words = set()
        
        for response in responses:
            words = set(response.lower().split())
            all_words.update(words)
            
            # Unique word ratio as creativity proxy
            unique_ratio = len(words) / max(1, len(response.split()))
            creativity_scores.append(unique_ratio)
        
        return np.mean(creativity_scores) if creativity_scores else 0.0


class SelfTrainingOrchestrator:
    """Main orchestrator for self-training processes"""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        teacher_models: Dict[str, nn.Module],
        config: SelfTrainingConfig,
        consciousness_engine: Optional[ConsciousnessEngine] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.teacher_models = teacher_models
        self.config = config
        self.consciousness_engine = consciousness_engine
        
        # Initialize components
        self.bootstrapper = SelfTalkBootstrapper(model, tokenizer, config)
        self.active_learner = ActiveLearningSelector(model, config)
        self.curriculum_scheduler = CurriculumLearningScheduler(config)
        self.independence_verifier = IndependenceVerifier(model, teacher_models, config)
        
        # Training state
        self.training_step = 0
        self.convergence_step = 0
        self.independence_achieved = False
        self.experience_buffer = deque(maxlen=config.experience_buffer_size)
        
        # Metrics tracking
        self.training_metrics = {
            'self_talk_quality': [],
            'active_learning_scores': [],
            'curriculum_progress': [],
            'independence_scores': [],
            'overall_performance': []
        }
        
    async def run_self_training(
        self, 
        initial_prompts: List[str],
        validation_prompts: List[str]
    ) -> Dict[str, Any]:
        """Run complete self-training process"""
        
        logger.info(f"Starting self-training with {len(initial_prompts)} initial prompts")
        
        for step in range(self.config.max_self_training_steps):
            self.training_step = step
            
            # 1. Self-talk bootstrapping
            await self._run_self_talk_phase(initial_prompts)
            
            # 2. Active learning selection
            await self._run_active_learning_phase()
            
            # 3. Curriculum-based filtering
            await self._run_curriculum_phase()
            
            # 4. Independence verification (periodic)
            if step % self.config.independence_test_frequency == 0:
                independence_result = await self.independence_verifier.verify_independence(
                    validation_prompts
                )
                self.training_metrics['independence_scores'].append(
                    independence_result['overall_independence_score']
                )
                
                if independence_result['passes_independence_threshold']:
                    self.independence_achieved = True
                    logger.info(f"Independence achieved at step {step}")
                    break
            
            # 5. Check convergence
            if self._check_convergence():
                logger.info(f"Training converged at step {step}")
                break
            
            # Progress logging
            if step % 50 == 0:
                current_curriculum = self.curriculum_scheduler.get_current_curriculum()
                logger.info(
                    f"Self-training step {step}: "
                    f"Curriculum stage {current_curriculum['name']}, "
                    f"Experience buffer size {len(self.experience_buffer)}"
                )
        
        # Final evaluation
        final_independence = await self.independence_verifier.verify_independence(validation_prompts)
        
        # Training summary
        summary = {
            'training_completed': True,
            'final_step': self.training_step,
            'independence_achieved': self.independence_achieved,
            'final_independence_score': final_independence['overall_independence_score'],
            'curriculum_stages_completed': self.curriculum_scheduler.current_stage,
            'total_experiences': len(self.experience_buffer),
            'training_metrics': self.training_metrics,
            'final_independence_analysis': final_independence
        }
        
        logger.info(f"Self-training completed. Independence achieved: {self.independence_achieved}")
        
        return summary
    
    async def _run_self_talk_phase(self, prompts: List[str]):
        """Run self-talk bootstrapping phase"""
        
        # Select random prompts for self-talk
        selected_prompts = random.sample(prompts, min(5, len(prompts)))
        
        conversations = []
        for prompt in selected_prompts:
            conversation = await self.bootstrapper.bootstrap_conversation(
                prompt, self.consciousness_engine
            )
            conversations.append(conversation)
        
        # Extract training pairs
        training_pairs = self.bootstrapper.extract_training_pairs(conversations)
        
        # Add to experience buffer
        for prompt, response, quality in training_pairs:
            self.experience_buffer.append({
                'type': 'self_talk',
                'prompt': prompt,
                'response': response,
                'quality': quality,
                'step': self.training_step,
                'consciousness_level': self.consciousness_engine.get_current_state().self_awareness_level if self.consciousness_engine else 0.0
            })
        
        # Update metrics
        if training_pairs:
            avg_quality = np.mean([pair[2] for pair in training_pairs])
            self.training_metrics['self_talk_quality'].append(avg_quality)
    
    async def _run_active_learning_phase(self):
        """Run active learning selection phase"""
        
        # Convert experience buffer to candidate examples
        candidates = [
            (exp['prompt'], exp['response'], exp['quality'])
            for exp in list(self.experience_buffer)[-100:]  # Recent experiences
        ]
        
        if not candidates:
            return
        
        # Select most informative examples
        selected_examples = self.active_learner.select_active_examples(candidates)
        
        # Update experience buffer with active learning scores
        for prompt, response, quality, scores in selected_examples:
            # Find and update corresponding experience
            for exp in self.experience_buffer:
                if exp['prompt'] == prompt and exp['response'] == response:
                    exp['active_learning_scores'] = scores
                    break
        
        # Update metrics
        if selected_examples:
            avg_active_score = np.mean([ex[3]['combined'] for ex in selected_examples])
            self.training_metrics['active_learning_scores'].append(avg_active_score)
    
    async def _run_curriculum_phase(self):
        """Run curriculum learning phase"""
        
        # Get examples with active learning scores
        scored_examples = [
            (exp['prompt'], exp['response'], exp['quality'], exp.get('active_learning_scores', {}))
            for exp in list(self.experience_buffer)
            if 'active_learning_scores' in exp
        ]
        
        if not scored_examples:
            return
        
        # Filter by curriculum level
        curriculum_examples = self.curriculum_scheduler.filter_examples_by_curriculum(scored_examples)
        
        # Update curriculum progress
        if curriculum_examples:
            avg_quality = np.mean([ex[2] for ex in curriculum_examples])
            self.curriculum_scheduler.update_progress(avg_quality)
            self.training_metrics['curriculum_progress'].append({
                'stage': self.curriculum_scheduler.current_stage,
                'progress': self.curriculum_scheduler.stage_progress,
                'avg_quality': avg_quality
            })
    
    def _check_convergence(self) -> bool:
        """Check if training has converged"""
        
        if len(self.training_metrics['overall_performance']) < self.config.convergence_patience:
            return False
        
        # Check recent performance improvement
        recent_performance = self.training_metrics['overall_performance'][-self.config.convergence_patience:]
        
        # Calculate improvement over last N steps
        early_avg = np.mean(recent_performance[:len(recent_performance)//2])
        late_avg = np.mean(recent_performance[len(recent_performance)//2:])
        
        improvement = late_avg - early_avg
        
        return improvement < self.config.min_improvement
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        
        return {
            'training_step': self.training_step,
            'independence_achieved': self.independence_achieved,
            'curriculum_stage': self.curriculum_scheduler.current_stage,
            'experience_buffer_size': len(self.experience_buffer),
            'experience_buffer_stats': self._analyze_experience_buffer(),
            'training_metrics_summary': {
                metric: {
                    'latest': values[-1] if values else 0,
                    'average': np.mean(values) if values else 0,
                    'trend': self._calculate_trend(values) if len(values) > 5 else 'insufficient_data'
                }
                for metric, values in self.training_metrics.items()
                if isinstance(values, list) and values and isinstance(values[0], (int, float))
            },
            'independence_trend': self._calculate_trend(self.training_metrics['independence_scores']),
            'curriculum_completion': (self.curriculum_scheduler.current_stage + 1) / len(self.curriculum_scheduler.difficulty_levels)
        }
    
    def _analyze_experience_buffer(self) -> Dict[str, Any]:
        """Analyze experience buffer contents"""
        
        if not self.experience_buffer:
            return {}
        
        experiences = list(self.experience_buffer)
        
        # Type distribution
        type_counts = defaultdict(int)
        quality_by_type = defaultdict(list)
        
        for exp in experiences:
            exp_type = exp.get('type', 'unknown')
            type_counts[exp_type] += 1
            quality_by_type[exp_type].append(exp['quality'])
        
        return {
            'total_experiences': len(experiences),
            'type_distribution': dict(type_counts),
            'avg_quality_by_type': {
                t: np.mean(qualities) for t, qualities in quality_by_type.items()
            },
            'overall_avg_quality': np.mean([exp['quality'] for exp in experiences]),
            'recent_quality_trend': self._calculate_trend([exp['quality'] for exp in experiences[-20:]]) if len(experiences) >= 20 else 'insufficient_data'
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for list of values"""
        
        if len(values) < 5:
            return 'insufficient_data'
        
        # Simple linear trend
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(x, y)[0, 1]
        
        if correlation > 0.1:
            return 'increasing'
        elif correlation < -0.1:
            return 'decreasing'
        else:
            return 'stable'


# Testing and example usage
async def test_self_training():
    """Test self-training orchestrator"""
    
    # Mock components for testing
    class MockModel(nn.Module):
        def forward(self, input_ids, **kwargs):
            return type('Output', (), {'logits': torch.randn(1, 10, 100)})()
        
        def generate(self, input_ids, **kwargs):
            return torch.randint(0, 100, (1, 20))
    
    class MockTokenizer:
        def __call__(self, text, **kwargs):
            return {'input_ids': torch.randint(0, 100, (1, 10))}
        
        def decode(self, ids, **kwargs):
            return "Mock response for testing"
        
        @property
        def eos_token_id(self):
            return 50256
    
    # Create test configuration
    config = SelfTrainingConfig(
        max_self_training_steps=10,
        independence_test_frequency=5,
        self_talk_rounds=2
    )
    
    # Create orchestrator
    model = MockModel()
    tokenizer = MockTokenizer()
    teacher_models = {'teacher1': MockModel(), 'teacher2': MockModel()}
    
    orchestrator = SelfTrainingOrchestrator(model, tokenizer, teacher_models, config)
    
    # Test prompts
    initial_prompts = [
        "What is consciousness?",
        "How do you think?",
        "What is your purpose?"
    ]
    
    validation_prompts = [
        "Explain your reasoning process",
        "What makes you self-aware?"
    ]
    
    # Run self-training
    summary = await orchestrator.run_self_training(initial_prompts, validation_prompts)
    
    print("Self-Training Summary:")
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    # Get final statistics
    stats = orchestrator.get_training_statistics()
    print(f"\nFinal Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(test_self_training())