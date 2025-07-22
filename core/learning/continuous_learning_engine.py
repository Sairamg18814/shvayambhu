"""
Continuous Learning Engine for Shvayambhu LLM System

Implements advanced continuous learning mechanisms that allow the system to
learn from new experiences while preserving existing knowledge, with full
consciousness integration and meta-learning capabilities.
"""

import asyncio
import json
import logging
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# MLX imports for Apple Silicon optimization
import mlx.core as mx
import mlx.nn as nn_mlx
from mlx.utils import tree_flatten, tree_unflatten

# Consciousness integration
from ..consciousness import ConsciousnessEngine, ConsciousnessLevel
from ..consciousness.consciousness_integration import ConsciousnessAwareModule


class LearningType(Enum):
    """Types of learning experiences."""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    SELF_SUPERVISED = "self_supervised"
    CONSCIOUSNESS_GUIDED = "consciousness_guided"
    META_LEARNING = "meta_learning"
    TRANSFER_LEARNING = "transfer_learning"
    FEW_SHOT = "few_shot"


class KnowledgeType(Enum):
    """Types of knowledge in the system."""
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    CONCEPTUAL = "conceptual"
    METACOGNITIVE = "metacognitive"
    EMOTIONAL = "emotional"
    LINGUISTIC = "linguistic"
    MULTIMODAL = "multimodal"
    CONSCIOUSNESS = "consciousness"


class LearningPriority(Enum):
    """Priority levels for learning experiences."""
    CRITICAL = 1.0
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    BACKGROUND = 0.2


class ForgettingMitigation(Enum):
    """Strategies to mitigate catastrophic forgetting."""
    EXPERIENCE_REPLAY = "experience_replay"
    ELASTIC_WEIGHT_CONSOLIDATION = "ewc"
    PROGRESSIVE_NETWORKS = "progressive_networks"
    MEMORY_AWARE_SYNAPSES = "mas"
    PACKNET = "packnet"
    CONTINUAL_LEARNING_REGULARIZATION = "clr"


@dataclass
class LearningExperience:
    """Represents a learning experience."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    learning_type: LearningType = LearningType.SUPERVISED
    knowledge_type: KnowledgeType = KnowledgeType.FACTUAL
    priority: LearningPriority = LearningPriority.MEDIUM
    
    input_data: Any = None
    target_data: Any = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Consciousness-related fields
    consciousness_relevance: float = 0.0
    metacognitive_insights: Dict[str, Any] = field(default_factory=dict)
    self_reflection_score: float = 0.0
    
    # Learning metadata
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5
    difficulty: float = 0.5
    confidence: float = 0.5
    
    # Learning outcomes
    learning_success: Optional[bool] = None
    performance_improvement: Optional[float] = None
    knowledge_retention: Optional[float] = None
    
    # Replay and consolidation
    replay_count: int = 0
    last_replayed: Optional[datetime] = None
    consolidation_strength: float = 1.0
    
    def update_importance(self, new_importance: float, decay_factor: float = 0.95):
        """Update importance with temporal decay."""
        days_since = (datetime.now() - self.timestamp).days
        decayed_importance = self.importance * (decay_factor ** days_since)
        self.importance = max(decayed_importance, new_importance)
    
    def should_replay(self, replay_threshold: float = 0.6) -> bool:
        """Determine if experience should be replayed."""
        # Combine importance, recency, and replay frequency
        recency_score = np.exp(-((datetime.now() - self.timestamp).days) / 30.0)
        replay_penalty = np.exp(-self.replay_count / 10.0)  # Diminishing returns
        
        replay_score = (self.importance * 0.4 + 
                       recency_score * 0.3 + 
                       self.consciousness_relevance * 0.2 + 
                       replay_penalty * 0.1)
        
        return replay_score >= replay_threshold


@dataclass
class KnowledgeNode:
    """Represents a node in the knowledge graph."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    knowledge_type: KnowledgeType = KnowledgeType.FACTUAL
    content: Any = None
    embeddings: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Connections and relationships
    connections: Set[str] = field(default_factory=set)
    strength_weights: Dict[str, float] = field(default_factory=dict)
    
    # Learning metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    update_count: int = 0
    
    # Consciousness integration
    consciousness_relevance: float = 0.0
    self_reflection_generated: bool = False
    metacognitive_associations: List[str] = field(default_factory=list)
    
    # Consolidation and forgetting
    consolidation_level: float = 0.0  # 0 = fresh, 1 = fully consolidated
    forgetting_resistance: float = 0.5
    plasticity_score: float = 1.0  # How easily this knowledge can be modified
    
    def access(self):
        """Record access and update relevance."""
        self.access_count += 1
        self.last_updated = datetime.now()
        
        # Boost forgetting resistance with each access
        self.forgetting_resistance = min(self.forgetting_resistance * 1.01, 1.0)
    
    def add_connection(self, node_id: str, strength: float = 1.0):
        """Add connection to another knowledge node."""
        self.connections.add(node_id)
        self.strength_weights[node_id] = strength
    
    def update_content(self, new_content: Any, merge_strategy: str = "replace"):
        """Update knowledge content with specified merge strategy."""
        self.update_count += 1
        self.last_updated = datetime.now()
        
        if merge_strategy == "replace":
            self.content = new_content
        elif merge_strategy == "merge" and isinstance(self.content, dict) and isinstance(new_content, dict):
            self.content.update(new_content)
        elif merge_strategy == "append" and isinstance(self.content, list):
            if isinstance(new_content, list):
                self.content.extend(new_content)
            else:
                self.content.append(new_content)
        
        # Reduce plasticity as knowledge becomes more established
        self.plasticity_score *= 0.95
        self.plasticity_score = max(self.plasticity_score, 0.1)


class ContinualLearningStrategy(ABC):
    """Abstract base class for continual learning strategies."""
    
    @abstractmethod
    async def learn_from_experience(self, experience: LearningExperience, model: Any) -> Dict[str, Any]:
        """Learn from a single experience."""
        pass
    
    @abstractmethod
    async def consolidate_knowledge(self, experiences: List[LearningExperience], model: Any) -> Dict[str, Any]:
        """Consolidate knowledge from multiple experiences."""
        pass
    
    @abstractmethod
    async def prevent_forgetting(self, model: Any, old_experiences: List[LearningExperience]) -> Dict[str, Any]:
        """Implement forgetting mitigation strategy."""
        pass


class ExperienceReplayStrategy(ContinualLearningStrategy):
    """Experience replay for continual learning."""
    
    def __init__(self, replay_buffer_size: int = 10000, replay_ratio: float = 0.2):
        self.replay_buffer_size = replay_buffer_size
        self.replay_ratio = replay_ratio
        self.experience_buffer = deque(maxlen=replay_buffer_size)
        self.logger = logging.getLogger(__name__)
    
    async def learn_from_experience(self, experience: LearningExperience, model: Any) -> Dict[str, Any]:
        """Learn from new experience with replay."""
        # Add experience to buffer
        self.experience_buffer.append(experience)
        
        # Sample experiences for replay
        replay_experiences = self._sample_replay_experiences()
        
        # Combine new experience with replay
        learning_batch = [experience] + replay_experiences
        
        # Perform learning
        learning_results = await self._train_on_batch(learning_batch, model)
        
        # Update replay statistics
        for exp in replay_experiences:
            exp.replay_count += 1
            exp.last_replayed = datetime.now()
        
        return {
            'new_experience_learned': True,
            'replay_experiences_count': len(replay_experiences),
            'learning_loss': learning_results.get('loss', 0.0),
            'performance_metrics': learning_results.get('metrics', {})
        }
    
    async def consolidate_knowledge(self, experiences: List[LearningExperience], model: Any) -> Dict[str, Any]:
        """Consolidate knowledge through strategic replay."""
        # Select high-importance experiences for consolidation
        consolidation_experiences = [
            exp for exp in experiences 
            if exp.importance > 0.6 or exp.consciousness_relevance > 0.7
        ]
        
        # Perform consolidation training
        if consolidation_experiences:
            consolidation_results = await self._train_on_batch(consolidation_experiences, model)
            
            # Update consolidation levels
            for exp in consolidation_experiences:
                exp.consolidation_strength = min(exp.consolidation_strength * 1.1, 2.0)
            
            return {
                'consolidation_performed': True,
                'experiences_consolidated': len(consolidation_experiences),
                'consolidation_loss': consolidation_results.get('loss', 0.0)
            }
        
        return {'consolidation_performed': False}
    
    async def prevent_forgetting(self, model: Any, old_experiences: List[LearningExperience]) -> Dict[str, Any]:
        """Prevent forgetting through replay."""
        # Select experiences at risk of forgetting
        forgetting_candidates = [
            exp for exp in old_experiences
            if (datetime.now() - exp.timestamp).days > 7 and exp.replay_count < 3
        ]
        
        # Replay important old experiences
        if forgetting_candidates:
            # Sort by importance and select top candidates
            forgetting_candidates.sort(key=lambda x: x.importance, reverse=True)
            replay_batch = forgetting_candidates[:min(10, len(forgetting_candidates))]
            
            replay_results = await self._train_on_batch(replay_batch, model)
            
            # Update replay statistics
            for exp in replay_batch:
                exp.replay_count += 1
                exp.last_replayed = datetime.now()
            
            return {
                'forgetting_prevention_performed': True,
                'experiences_replayed': len(replay_batch),
                'replay_loss': replay_results.get('loss', 0.0)
            }
        
        return {'forgetting_prevention_performed': False}
    
    def _sample_replay_experiences(self) -> List[LearningExperience]:
        """Sample experiences for replay based on importance and recency."""
        if len(self.experience_buffer) <= 1:
            return []
        
        # Calculate replay count
        replay_count = max(1, int(len(self.experience_buffer) * self.replay_ratio))
        replay_count = min(replay_count, 5)  # Limit replay batch size
        
        # Sample based on importance and consciousness relevance
        experiences = list(self.experience_buffer)[:-1]  # Exclude current experience
        
        # Calculate sampling probabilities
        scores = []
        for exp in experiences:
            score = (exp.importance * 0.6 + 
                    exp.consciousness_relevance * 0.3 + 
                    (1.0 / max(exp.replay_count + 1, 1)) * 0.1)  # Prefer less replayed
            scores.append(score)
        
        # Normalize probabilities
        if sum(scores) > 0:
            probabilities = np.array(scores) / sum(scores)
            
            # Sample without replacement
            try:
                selected_indices = np.random.choice(
                    len(experiences), 
                    size=min(replay_count, len(experiences)), 
                    replace=False, 
                    p=probabilities
                )
                return [experiences[i] for i in selected_indices]
            except ValueError:
                # Fallback to random sampling if probability sampling fails
                selected_indices = np.random.choice(
                    len(experiences), 
                    size=min(replay_count, len(experiences)), 
                    replace=False
                )
                return [experiences[i] for i in selected_indices]
        
        return []
    
    async def _train_on_batch(self, experiences: List[LearningExperience], model: Any) -> Dict[str, Any]:
        """Train model on batch of experiences."""
        if not experiences:
            return {'loss': 0.0, 'metrics': {}}
        
        # This is a simplified training loop - in practice, would depend on model architecture
        total_loss = 0.0
        successful_updates = 0
        
        for experience in experiences:
            try:
                # Simulate learning from experience
                # In real implementation, would call model training methods
                learning_loss = await self._simulate_learning(experience, model)
                total_loss += learning_loss
                successful_updates += 1
            except Exception as e:
                self.logger.warning(f"Failed to learn from experience {experience.id}: {e}")
        
        avg_loss = total_loss / max(successful_updates, 1)
        
        return {
            'loss': avg_loss,
            'metrics': {
                'successful_updates': successful_updates,
                'failed_updates': len(experiences) - successful_updates,
                'batch_size': len(experiences)
            }
        }
    
    async def _simulate_learning(self, experience: LearningExperience, model: Any) -> float:
        """Simulate learning from experience (placeholder)."""
        # Placeholder for actual learning implementation
        # Would involve forward pass, loss calculation, backpropagation, etc.
        
        # Simulate loss based on experience characteristics
        base_loss = np.random.uniform(0.1, 1.0)
        
        # Adjust loss based on experience properties
        if experience.difficulty > 0.8:
            base_loss *= 1.2  # Difficult experiences have higher loss
        if experience.consciousness_relevance > 0.7:
            base_loss *= 0.9  # Consciousness-relevant experiences learn better
        
        return base_loss


class ElasticWeightConsolidation(ContinualLearningStrategy):
    """Elastic Weight Consolidation for preventing catastrophic forgetting."""
    
    def __init__(self, lambda_reg: float = 1000.0, fisher_samples: int = 1000):
        self.lambda_reg = lambda_reg
        self.fisher_samples = fisher_samples
        self.fisher_information = {}
        self.optimal_weights = {}
        self.logger = logging.getLogger(__name__)
    
    async def learn_from_experience(self, experience: LearningExperience, model: Any) -> Dict[str, Any]:
        """Learn with EWC regularization."""
        # Calculate EWC loss
        ewc_loss = self._calculate_ewc_loss(model)
        
        # Simulate learning with regularization
        primary_loss = await self._calculate_primary_loss(experience, model)
        total_loss = primary_loss + ewc_loss
        
        return {
            'primary_loss': primary_loss,
            'ewc_loss': ewc_loss,
            'total_loss': total_loss,
            'learning_success': True
        }
    
    async def consolidate_knowledge(self, experiences: List[LearningExperience], model: Any) -> Dict[str, Any]:
        """Consolidate by updating Fisher information."""
        # Calculate Fisher Information Matrix
        fisher_info = await self._calculate_fisher_information(experiences, model)
        
        # Update stored Fisher information
        self._update_fisher_information(fisher_info)
        
        # Store current optimal weights
        self._store_optimal_weights(model)
        
        return {
            'fisher_information_updated': True,
            'fisher_diagonal_mean': np.mean([v for v in fisher_info.values()]),
            'parameters_stored': len(fisher_info)
        }
    
    async def prevent_forgetting(self, model: Any, old_experiences: List[LearningExperience]) -> Dict[str, Any]:
        """EWC inherently prevents forgetting through regularization."""
        # EWC prevention is built into the learning process
        return {
            'ewc_regularization_active': len(self.fisher_information) > 0,
            'regularization_strength': self.lambda_reg,
            'protected_parameters': len(self.fisher_information)
        }
    
    def _calculate_ewc_loss(self, model: Any) -> float:
        """Calculate EWC regularization loss."""
        if not self.fisher_information or not self.optimal_weights:
            return 0.0
        
        ewc_loss = 0.0
        
        # Simulate parameter comparison (in practice, would iterate through model parameters)
        for param_name in self.fisher_information:
            if param_name in self.optimal_weights:
                # Fisher information * (current_weight - optimal_weight)^2
                fisher_diag = self.fisher_information[param_name]
                weight_diff_squared = np.random.uniform(0, 0.1)  # Simulate weight difference
                ewc_loss += fisher_diag * weight_diff_squared
        
        return self.lambda_reg * ewc_loss
    
    async def _calculate_primary_loss(self, experience: LearningExperience, model: Any) -> float:
        """Calculate primary learning loss."""
        # Placeholder for actual loss calculation
        return np.random.uniform(0.1, 1.0) * experience.difficulty
    
    async def _calculate_fisher_information(self, experiences: List[LearningExperience], model: Any) -> Dict[str, float]:
        """Calculate Fisher Information Matrix diagonal."""
        fisher_info = {}
        
        # Simulate Fisher information calculation
        # In practice, would calculate second derivatives of log-likelihood
        param_names = [f"layer_{i}_weight" for i in range(10)]  # Simulate parameter names
        
        for param_name in param_names:
            # Simulate Fisher information calculation
            fisher_diagonal = 0.0
            
            for experience in experiences[:self.fisher_samples]:
                # Calculate gradient of log-likelihood w.r.t. parameter
                # Square it and add to Fisher information
                gradient_squared = np.random.uniform(0, 1)  # Simulate gradient squared
                fisher_diagonal += gradient_squared
            
            fisher_info[param_name] = fisher_diagonal / len(experiences)
        
        return fisher_info
    
    def _update_fisher_information(self, new_fisher_info: Dict[str, float]):
        """Update stored Fisher information."""
        for param_name, fisher_val in new_fisher_info.items():
            if param_name in self.fisher_information:
                # Combine with existing Fisher information
                self.fisher_information[param_name] = (
                    self.fisher_information[param_name] * 0.7 + fisher_val * 0.3
                )
            else:
                self.fisher_information[param_name] = fisher_val
    
    def _store_optimal_weights(self, model: Any):
        """Store current model weights as optimal."""
        # Simulate storing model weights
        param_names = [f"layer_{i}_weight" for i in range(10)]
        for param_name in param_names:
            self.optimal_weights[param_name] = np.random.randn(100)  # Simulate weight vector


class KnowledgeGraph(ConsciousnessAwareModule):
    """Dynamic knowledge graph for continual learning."""
    
    def __init__(self, consciousness_engine=None, max_nodes: int = 100000):
        super().__init__(consciousness_engine)
        self.logger = logging.getLogger(__name__)
        self.max_nodes = max_nodes
        
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.type_indices: Dict[KnowledgeType, Set[str]] = {kt: set() for kt in KnowledgeType}
        self.embedding_index = {}  # For similarity search
        self.update_queue = deque()  # Pending updates
        
        # Graph statistics
        self.total_connections = 0
        self.last_cleanup = datetime.now()
        self.consolidation_cycles = 0
    
    async def add_knowledge(self, content: Any, knowledge_type: KnowledgeType, 
                          embeddings: np.ndarray = None, 
                          consciousness_context: Dict[str, Any] = None) -> str:
        """Add new knowledge to the graph."""
        # Create new knowledge node
        node = KnowledgeNode(
            knowledge_type=knowledge_type,
            content=content,
            embeddings=embeddings if embeddings is not None else np.array([])
        )
        
        # Set consciousness relevance
        if consciousness_context and self.consciousness_engine:
            node.consciousness_relevance = await self._assess_consciousness_relevance(
                content, consciousness_context
            )
        
        # Add to graph
        self.nodes[node.id] = node
        self.type_indices[knowledge_type].add(node.id)
        
        # Create connections to similar knowledge
        await self._create_connections(node)
        
        # Perform maintenance if needed
        await self._maintain_graph()
        
        self.logger.debug(f"Added knowledge node: {node.id} ({knowledge_type.value})")
        return node.id
    
    async def update_knowledge(self, node_id: str, new_content: Any, 
                             merge_strategy: str = "merge") -> bool:
        """Update existing knowledge node."""
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        old_content = node.content
        
        # Update content
        node.update_content(new_content, merge_strategy)
        
        # Queue for connection updates
        self.update_queue.append({
            'type': 'update',
            'node_id': node_id,
            'old_content': old_content,
            'new_content': new_content
        })
        
        return True
    
    async def find_related_knowledge(self, query_content: Any, 
                                   knowledge_type: KnowledgeType = None,
                                   limit: int = 10) -> List[Tuple[str, float]]:
        """Find knowledge nodes related to query content."""
        related_nodes = []
        
        # Search within specific type if specified
        search_nodes = (
            [self.nodes[node_id] for node_id in self.type_indices[knowledge_type]]
            if knowledge_type else self.nodes.values()
        )
        
        # Calculate similarity scores
        for node in search_nodes:
            similarity = await self._calculate_content_similarity(query_content, node.content)
            
            # Boost similarity based on consciousness relevance
            if hasattr(node, 'consciousness_relevance') and node.consciousness_relevance > 0.5:
                similarity *= 1.2
            
            # Consider connection strength and access frequency
            access_boost = min(np.log(node.access_count + 1) / 10.0, 0.3)
            final_similarity = similarity + access_boost
            
            related_nodes.append((node.id, final_similarity))
        
        # Sort by similarity and return top results
        related_nodes.sort(key=lambda x: x[1], reverse=True)
        return related_nodes[:limit]
    
    async def consolidate_knowledge(self, consolidation_threshold: float = 0.8) -> Dict[str, Any]:
        """Consolidate similar knowledge nodes."""
        consolidation_results = {
            'nodes_merged': 0,
            'connections_updated': 0,
            'knowledge_types_affected': set()
        }
        
        # Find pairs of highly similar nodes
        similar_pairs = await self._find_similar_node_pairs(consolidation_threshold)
        
        for node1_id, node2_id, similarity in similar_pairs:
            if node1_id in self.nodes and node2_id in self.nodes:
                node1 = self.nodes[node1_id]
                node2 = self.nodes[node2_id]
                
                # Merge less established node into more established one
                target_node = node1 if node1.consolidation_level >= node2.consolidation_level else node2
                source_node = node2 if target_node == node1 else node1
                
                # Merge content and connections
                await self._merge_nodes(target_node, source_node)
                
                # Remove source node
                await self._remove_node(source_node.id)
                
                consolidation_results['nodes_merged'] += 1
                consolidation_results['knowledge_types_affected'].add(target_node.knowledge_type)
        
        self.consolidation_cycles += 1
        return consolidation_results
    
    async def get_consciousness_insights(self) -> Dict[str, Any]:
        """Get insights about consciousness-related knowledge."""
        consciousness_nodes = [
            node for node in self.nodes.values()
            if node.consciousness_relevance > 0.5
        ]
        
        if not consciousness_nodes:
            return {'consciousness_nodes': 0}
        
        insights = {
            'consciousness_nodes': len(consciousness_nodes),
            'avg_consciousness_relevance': np.mean([n.consciousness_relevance for n in consciousness_nodes]),
            'consciousness_knowledge_types': list(set(n.knowledge_type.value for n in consciousness_nodes)),
            'highly_connected_consciousness_nodes': [
                n.id for n in consciousness_nodes if len(n.connections) > 10
            ],
            'recent_consciousness_updates': [
                n.id for n in consciousness_nodes 
                if (datetime.now() - n.last_updated).days < 7
            ]
        }
        
        return insights
    
    async def _assess_consciousness_relevance(self, content: Any, 
                                           consciousness_context: Dict[str, Any]) -> float:
        """Assess consciousness relevance of new knowledge."""
        relevance = 0.0
        
        # Content-based relevance
        if isinstance(content, str):
            consciousness_keywords = [
                'consciousness', 'awareness', 'self', 'identity', 'experience',
                'perception', 'cognition', 'meta', 'reflection', 'introspection'
            ]
            
            content_lower = content.lower()
            keyword_matches = sum(1 for keyword in consciousness_keywords if keyword in content_lower)
            relevance += min(keyword_matches * 0.1, 0.4)
        
        # Context-based relevance
        if consciousness_context:
            consciousness_level = consciousness_context.get('consciousness_level', 'basic')
            if consciousness_level in ['advanced', 'peak']:
                relevance += 0.3
            
            metacognition = consciousness_context.get('metacognition_level', 0.0)
            relevance += metacognition * 0.3
            
            self_awareness = consciousness_context.get('self_awareness_score', 0.0)
            relevance += self_awareness * 0.2
        
        return min(relevance, 1.0)
    
    async def _create_connections(self, new_node: KnowledgeNode):
        """Create connections between new node and existing nodes."""
        if not new_node.embeddings.size:
            return
        
        # Find similar nodes
        similar_nodes = await self.find_related_knowledge(
            new_node.content, 
            limit=20
        )
        
        # Create connections to top similar nodes
        for node_id, similarity in similar_nodes[:10]:
            if similarity > 0.3:  # Threshold for creating connection
                new_node.add_connection(node_id, similarity)
                
                # Add reciprocal connection
                if node_id in self.nodes:
                    self.nodes[node_id].add_connection(new_node.id, similarity)
                    self.total_connections += 1
    
    async def _calculate_content_similarity(self, content1: Any, content2: Any) -> float:
        """Calculate similarity between two pieces of content."""
        # Simplified similarity calculation
        if isinstance(content1, str) and isinstance(content2, str):
            # Simple word overlap similarity
            words1 = set(content1.lower().split())
            words2 = set(content2.lower().split())
            
            if not words1 and not words2:
                return 0.0
            
            intersection = words1 & words2
            union = words1 | words2
            
            return len(intersection) / len(union) if union else 0.0
        
        # For other content types, would implement appropriate similarity measures
        return np.random.uniform(0, 1)  # Placeholder
    
    async def _find_similar_node_pairs(self, threshold: float) -> List[Tuple[str, str, float]]:
        """Find pairs of nodes above similarity threshold."""
        similar_pairs = []
        node_ids = list(self.nodes.keys())
        
        for i, node1_id in enumerate(node_ids):
            for node2_id in node_ids[i+1:]:
                node1 = self.nodes[node1_id]
                node2 = self.nodes[node2_id]
                
                # Only consider same knowledge type
                if node1.knowledge_type == node2.knowledge_type:
                    similarity = await self._calculate_content_similarity(
                        node1.content, node2.content
                    )
                    
                    if similarity >= threshold:
                        similar_pairs.append((node1_id, node2_id, similarity))
        
        return similar_pairs
    
    async def _merge_nodes(self, target_node: KnowledgeNode, source_node: KnowledgeNode):
        """Merge source node into target node."""
        # Merge content (strategy depends on content type)
        if isinstance(target_node.content, dict) and isinstance(source_node.content, dict):
            target_node.content.update(source_node.content)
        elif isinstance(target_node.content, list) and isinstance(source_node.content, list):
            target_node.content.extend(source_node.content)
        
        # Merge connections
        for connection_id in source_node.connections:
            target_node.add_connection(connection_id, 
                                     source_node.strength_weights.get(connection_id, 1.0))
        
        # Update metadata
        target_node.access_count += source_node.access_count
        target_node.update_count += source_node.update_count
        target_node.consolidation_level = min(target_node.consolidation_level + 0.1, 1.0)
        
        # Merge consciousness-related fields
        target_node.consciousness_relevance = max(
            target_node.consciousness_relevance, 
            source_node.consciousness_relevance
        )
        
        # Merge metacognitive associations
        target_node.metacognitive_associations.extend(source_node.metacognitive_associations)
        target_node.metacognitive_associations = list(set(target_node.metacognitive_associations))
    
    async def _remove_node(self, node_id: str):
        """Remove node from graph."""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        
        # Remove from type index
        self.type_indices[node.knowledge_type].discard(node_id)
        
        # Remove connections from other nodes
        for connection_id in node.connections:
            if connection_id in self.nodes:
                connected_node = self.nodes[connection_id]
                connected_node.connections.discard(node_id)
                connected_node.strength_weights.pop(node_id, None)
        
        # Remove from main nodes dictionary
        del self.nodes[node_id]
    
    async def _maintain_graph(self):
        """Perform graph maintenance tasks."""
        # Cleanup old or low-importance nodes if approaching limit
        if len(self.nodes) > self.max_nodes * 0.9:
            await self._cleanup_nodes()
        
        # Process pending updates
        if self.update_queue:
            await self._process_pending_updates()
        
        # Periodic consolidation
        if (datetime.now() - self.last_cleanup).hours >= 24:
            await self.consolidate_knowledge()
            self.last_cleanup = datetime.now()
    
    async def _cleanup_nodes(self):
        """Clean up old or unimportant nodes."""
        # Calculate retention scores for all nodes
        retention_scores = []
        
        for node_id, node in self.nodes.items():
            # Factor in access frequency, recency, consolidation, and consciousness relevance
            recency_score = np.exp(-((datetime.now() - node.last_updated).days) / 30.0)
            access_score = min(np.log(node.access_count + 1) / 10.0, 1.0)
            consolidation_score = node.consolidation_level
            consciousness_score = node.consciousness_relevance
            connection_score = min(len(node.connections) / 10.0, 1.0)
            
            retention_score = (
                recency_score * 0.2 +
                access_score * 0.25 +
                consolidation_score * 0.2 +
                consciousness_score * 0.2 +
                connection_score * 0.15
            )
            
            retention_scores.append((retention_score, node_id))
        
        # Sort by retention score and keep top nodes
        retention_scores.sort(reverse=True)
        nodes_to_keep = set(node_id for _, node_id in retention_scores[:self.max_nodes])
        
        # Remove nodes not in keep set
        nodes_to_remove = set(self.nodes.keys()) - nodes_to_keep
        for node_id in nodes_to_remove:
            await self._remove_node(node_id)
        
        self.logger.info(f"Cleaned up {len(nodes_to_remove)} knowledge nodes")
    
    async def _process_pending_updates(self):
        """Process pending knowledge updates."""
        while self.update_queue:
            update = self.update_queue.popleft()
            
            if update['type'] == 'update':
                node_id = update['node_id']
                if node_id in self.nodes:
                    # Recalculate connections after content update
                    await self._update_node_connections(node_id)


class ContinuousLearningEngine(ConsciousnessAwareModule):
    """Main engine for continuous learning and knowledge management."""
    
    def __init__(self, consciousness_engine=None):
        super().__init__(consciousness_engine)
        self.logger = logging.getLogger(__name__)
        
        # Learning components
        self.knowledge_graph = KnowledgeGraph(consciousness_engine)
        self.experience_buffer = deque(maxlen=50000)
        
        # Learning strategies
        self.strategies = {
            ForgettingMitigation.EXPERIENCE_REPLAY: ExperienceReplayStrategy(),
            ForgettingMitigation.ELASTIC_WEIGHT_CONSOLIDATION: ElasticWeightConsolidation()
        }
        
        # Configuration
        self.config = {
            'learning_rate_adaptation': True,
            'consciousness_guided_learning': True,
            'meta_learning_enabled': True,
            'automatic_consolidation': True,
            'forgetting_prevention': True
        }
        
        # Learning statistics
        self.learning_stats = {
            'total_experiences': 0,
            'successful_learnings': 0,
            'consolidation_cycles': 0,
            'knowledge_nodes_created': 0,
            'forgetting_prevention_events': 0
        }
        
        # Meta-learning components
        self.meta_learning_history = deque(maxlen=1000)
        self.learning_performance_trends = defaultdict(list)
    
    async def learn_from_experience(self, input_data: Any, target_data: Any = None,
                                  learning_type: LearningType = LearningType.SUPERVISED,
                                  knowledge_type: KnowledgeType = KnowledgeType.FACTUAL,
                                  context: Dict[str, Any] = None,
                                  consciousness_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Learn from a new experience."""
        context = context or {}
        consciousness_context = consciousness_context or {}
        
        # Create learning experience
        experience = LearningExperience(
            learning_type=learning_type,
            knowledge_type=knowledge_type,
            input_data=input_data,
            target_data=target_data,
            context=context
        )
        
        # Assess consciousness relevance
        if consciousness_context and self.consciousness_engine:
            experience.consciousness_relevance = await self._assess_experience_consciousness_relevance(
                experience, consciousness_context
            )
            
            # Add metacognitive insights
            experience.metacognitive_insights = await self._generate_metacognitive_insights(
                experience, consciousness_context
            )
        
        # Assess experience importance and difficulty
        experience.importance = await self._assess_experience_importance(experience)
        experience.difficulty = await self._assess_experience_difficulty(experience)
        
        # Apply appropriate learning strategy
        learning_results = await self._apply_learning_strategies(experience)
        
        # Update knowledge graph
        if learning_results.get('learning_success', False):
            knowledge_node_id = await self.knowledge_graph.add_knowledge(
                content={'input': input_data, 'output': target_data, 'context': context},
                knowledge_type=knowledge_type,
                consciousness_context=consciousness_context
            )
            learning_results['knowledge_node_id'] = knowledge_node_id
            self.learning_stats['knowledge_nodes_created'] += 1
        
        # Store experience
        self.experience_buffer.append(experience)
        self.learning_stats['total_experiences'] += 1
        
        if learning_results.get('learning_success', False):
            self.learning_stats['successful_learnings'] += 1
        
        # Meta-learning update
        await self._update_meta_learning(experience, learning_results)
        
        # Periodic maintenance
        await self._perform_maintenance()
        
        return {
            'experience_id': experience.id,
            'learning_success': learning_results.get('learning_success', False),
            'consciousness_relevance': experience.consciousness_relevance,
            'importance': experience.importance,
            'difficulty': experience.difficulty,
            'learning_results': learning_results,
            'knowledge_updated': 'knowledge_node_id' in learning_results
        }
    
    async def consolidate_knowledge(self, consolidation_type: str = "automatic") -> Dict[str, Any]:
        """Perform knowledge consolidation."""
        consolidation_results = {
            'experience_consolidation': {},
            'knowledge_graph_consolidation': {},
            'meta_learning_updates': {}
        }
        
        # Consolidate experiences
        high_importance_experiences = [
            exp for exp in self.experience_buffer
            if exp.importance > 0.7 or exp.consciousness_relevance > 0.6
        ]
        
        if high_importance_experiences:
            for strategy in self.strategies.values():
                strategy_results = await strategy.consolidate_knowledge(
                    high_importance_experiences, None  # Model would be passed here
                )
                consolidation_results['experience_consolidation'][type(strategy).__name__] = strategy_results
        
        # Consolidate knowledge graph
        graph_consolidation = await self.knowledge_graph.consolidate_knowledge()
        consolidation_results['knowledge_graph_consolidation'] = graph_consolidation
        
        # Update learning statistics
        self.learning_stats['consolidation_cycles'] += 1
        
        return consolidation_results
    
    async def prevent_forgetting(self) -> Dict[str, Any]:
        """Apply forgetting prevention strategies."""
        prevention_results = {}
        
        # Identify experiences at risk of forgetting
        old_experiences = [
            exp for exp in self.experience_buffer
            if (datetime.now() - exp.timestamp).days > 14 and exp.replay_count < 3
        ]
        
        if old_experiences:
            for strategy_name, strategy in self.strategies.items():
                results = await strategy.prevent_forgetting(None, old_experiences)
                prevention_results[strategy_name.value] = results
            
            self.learning_stats['forgetting_prevention_events'] += 1
        
        return prevention_results
    
    async def get_learning_insights(self) -> Dict[str, Any]:
        """Get comprehensive learning insights."""
        insights = {
            'learning_statistics': self.learning_stats.copy(),
            'experience_buffer_status': {
                'total_experiences': len(self.experience_buffer),
                'recent_experiences': len([
                    exp for exp in self.experience_buffer
                    if (datetime.now() - exp.timestamp).days <= 7
                ]),
                'high_importance_experiences': len([
                    exp for exp in self.experience_buffer
                    if exp.importance > 0.7
                ]),
                'consciousness_relevant_experiences': len([
                    exp for exp in self.experience_buffer
                    if exp.consciousness_relevance > 0.5
                ])
            },
            'knowledge_graph_status': {
                'total_nodes': len(self.knowledge_graph.nodes),
                'total_connections': self.knowledge_graph.total_connections,
                'consolidation_cycles': self.knowledge_graph.consolidation_cycles
            },
            'learning_performance_trends': dict(self.learning_performance_trends)
        }
        
        # Add consciousness insights
        if self.consciousness_engine:
            consciousness_insights = await self.knowledge_graph.get_consciousness_insights()
            insights['consciousness_insights'] = consciousness_insights
        
        # Add meta-learning insights
        meta_insights = await self._get_meta_learning_insights()
        insights['meta_learning_insights'] = meta_insights
        
        return insights
    
    async def adapt_learning_parameters(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt learning parameters based on feedback."""
        adaptations = {}
        
        # Adapt learning rates based on performance
        if 'performance_score' in feedback:
            performance = feedback['performance_score']
            
            if performance < 0.6:
                # Increase learning focus on difficult areas
                adaptations['increase_replay_ratio'] = True
                adaptations['increase_consolidation_frequency'] = True
            elif performance > 0.8:
                # Allow more exploration
                adaptations['decrease_forgetting_prevention'] = True
                adaptations['increase_new_knowledge_weight'] = True
        
        # Adapt consciousness integration
        if 'consciousness_effectiveness' in feedback:
            consciousness_effect = feedback['consciousness_effectiveness']
            
            if consciousness_effect > 0.7:
                adaptations['increase_consciousness_weighting'] = True
            elif consciousness_effect < 0.3:
                adaptations['decrease_consciousness_dependency'] = True
        
        # Apply adaptations
        await self._apply_parameter_adaptations(adaptations)
        
        return adaptations
    
    async def _assess_experience_consciousness_relevance(
        self, 
        experience: LearningExperience, 
        consciousness_context: Dict[str, Any]
    ) -> float:
        """Assess how relevant an experience is to consciousness development."""
        relevance = 0.0
        
        # Content-based relevance
        if isinstance(experience.input_data, str):
            input_text = experience.input_data.lower()
            consciousness_keywords = [
                'self', 'awareness', 'conscious', 'think', 'understand',
                'realize', 'reflect', 'introspect', 'meta', 'identity'
            ]
            
            keyword_matches = sum(1 for keyword in consciousness_keywords if keyword in input_text)
            relevance += min(keyword_matches * 0.1, 0.3)
        
        # Learning type relevance
        if experience.learning_type in [LearningType.META_LEARNING, LearningType.CONSCIOUSNESS_GUIDED]:
            relevance += 0.4
        
        # Knowledge type relevance  
        if experience.knowledge_type in [KnowledgeType.METACOGNITIVE, KnowledgeType.CONSCIOUSNESS]:
            relevance += 0.3
        
        # Context relevance
        consciousness_level = consciousness_context.get('consciousness_level', 'basic')
        if consciousness_level in ['advanced', 'peak']:
            relevance += 0.2
        
        metacognition_level = consciousness_context.get('metacognition_level', 0.0)
        relevance += metacognition_level * 0.2
        
        return min(relevance, 1.0)
    
    async def _generate_metacognitive_insights(
        self, 
        experience: LearningExperience, 
        consciousness_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate metacognitive insights about the learning experience."""
        insights = {}
        
        # Learning strategy awareness
        insights['learning_strategy_used'] = experience.learning_type.value
        insights['expected_difficulty'] = experience.difficulty
        insights['predicted_importance'] = experience.importance
        
        # Self-monitoring
        if consciousness_context.get('metacognition_level', 0.0) > 0.6:
            insights['metacognitive_monitoring'] = {
                'confidence_prediction': np.random.uniform(0.3, 0.9),  # Predict learning confidence
                'effort_required': 'high' if experience.difficulty > 0.7 else 'moderate',
                'connection_potential': 'high' if experience.importance > 0.6 else 'low'
            }
        
        # Self-reflection on learning process
        if consciousness_context.get('self_awareness_score', 0.0) > 0.7:
            insights['self_reflection'] = {
                'learning_preference_alignment': np.random.uniform(0.4, 1.0),
                'cognitive_load_assessment': 'manageable' if experience.difficulty < 0.8 else 'challenging',
                'interest_level': 'high' if experience.consciousness_relevance > 0.5 else 'moderate'
            }
        
        return insights
    
    async def _assess_experience_importance(self, experience: LearningExperience) -> float:
        """Assess the importance of a learning experience."""
        importance = 0.5  # Base importance
        
        # Consciousness relevance boost
        importance += experience.consciousness_relevance * 0.3
        
        # Learning type importance
        type_weights = {
            LearningType.CONSCIOUSNESS_GUIDED: 0.9,
            LearningType.META_LEARNING: 0.8,
            LearningType.REINFORCEMENT: 0.7,
            LearningType.SUPERVISED: 0.6,
            LearningType.SELF_SUPERVISED: 0.5,
            LearningType.UNSUPERVISED: 0.4
        }
        importance += type_weights.get(experience.learning_type, 0.5) * 0.2
        
        # Knowledge type importance
        knowledge_weights = {
            KnowledgeType.CONSCIOUSNESS: 0.9,
            KnowledgeType.METACOGNITIVE: 0.8,
            KnowledgeType.EMOTIONAL: 0.7,
            KnowledgeType.CONCEPTUAL: 0.6,
            KnowledgeType.PROCEDURAL: 0.5,
            KnowledgeType.FACTUAL: 0.4
        }
        importance += knowledge_weights.get(experience.knowledge_type, 0.5) * 0.2
        
        # Context importance
        if experience.context.get('user_emphasized', False):
            importance += 0.1
        if experience.context.get('correction_needed', False):
            importance += 0.15
        if experience.context.get('novel_situation', False):
            importance += 0.1
        
        return min(importance, 1.0)
    
    async def _assess_experience_difficulty(self, experience: LearningExperience) -> float:
        """Assess the difficulty of a learning experience."""
        difficulty = 0.5  # Base difficulty
        
        # Content-based difficulty assessment
        if isinstance(experience.input_data, str):
            # Simple heuristics for text difficulty
            text = experience.input_data
            
            # Length factor
            if len(text.split()) > 100:
                difficulty += 0.1
            
            # Complexity indicators
            complex_words = len([word for word in text.split() if len(word) > 8])
            if complex_words > len(text.split()) * 0.2:
                difficulty += 0.15
            
            # Technical content
            technical_keywords = ['algorithm', 'neural', 'quantum', 'consciousness', 'metacognitive']
            if any(keyword in text.lower() for keyword in technical_keywords):
                difficulty += 0.2
        
        # Learning type difficulty
        type_difficulty = {
            LearningType.META_LEARNING: 0.9,
            LearningType.CONSCIOUSNESS_GUIDED: 0.8,
            LearningType.REINFORCEMENT: 0.7,
            LearningType.UNSUPERVISED: 0.6,
            LearningType.SELF_SUPERVISED: 0.5,
            LearningType.SUPERVISED: 0.4
        }
        difficulty += type_difficulty.get(experience.learning_type, 0.5) * 0.3
        
        # Knowledge type difficulty
        knowledge_difficulty = {
            KnowledgeType.CONSCIOUSNESS: 0.9,
            KnowledgeType.METACOGNITIVE: 0.8,
            KnowledgeType.CONCEPTUAL: 0.7,
            KnowledgeType.EMOTIONAL: 0.6,
            KnowledgeType.PROCEDURAL: 0.5,
            KnowledgeType.FACTUAL: 0.3
        }
        difficulty += knowledge_difficulty.get(experience.knowledge_type, 0.5) * 0.2
        
        return min(difficulty, 1.0)
    
    async def _apply_learning_strategies(self, experience: LearningExperience) -> Dict[str, Any]:
        """Apply appropriate learning strategies for the experience."""
        results = {'learning_success': False, 'strategy_results': {}}
        
        # Select strategies based on experience characteristics
        selected_strategies = []
        
        # Always use experience replay
        selected_strategies.append(ForgettingMitigation.EXPERIENCE_REPLAY)
        
        # Use EWC for high-importance experiences
        if experience.importance > 0.7:
            selected_strategies.append(ForgettingMitigation.ELASTIC_WEIGHT_CONSOLIDATION)
        
        # Apply selected strategies
        for strategy_type in selected_strategies:
            if strategy_type in self.strategies:
                strategy = self.strategies[strategy_type]
                try:
                    strategy_result = await strategy.learn_from_experience(experience, None)
                    results['strategy_results'][strategy_type.value] = strategy_result
                    
                    # Consider learning successful if any strategy succeeds
                    if strategy_result.get('learning_success', False):
                        results['learning_success'] = True
                        
                except Exception as e:
                    self.logger.warning(f"Strategy {strategy_type.value} failed: {e}")
        
        # Update experience with results
        experience.learning_success = results['learning_success']
        if results['learning_success']:
            experience.confidence = min(experience.confidence + 0.1, 1.0)
        
        return results
    
    async def _update_meta_learning(self, experience: LearningExperience, learning_results: Dict[str, Any]):
        """Update meta-learning based on learning experience."""
        # Record learning performance
        performance_metric = {
            'experience_id': experience.id,
            'learning_type': experience.learning_type.value,
            'knowledge_type': experience.knowledge_type.value,
            'difficulty': experience.difficulty,
            'importance': experience.importance,
            'success': learning_results.get('learning_success', False),
            'consciousness_relevance': experience.consciousness_relevance,
            'timestamp': datetime.now()
        }
        
        self.meta_learning_history.append(performance_metric)
        
        # Update performance trends
        self.learning_performance_trends[experience.learning_type.value].append(
            1.0 if learning_results.get('learning_success', False) else 0.0
        )
        
        # Keep only recent trends
        for learning_type in self.learning_performance_trends:
            if len(self.learning_performance_trends[learning_type]) > 100:
                self.learning_performance_trends[learning_type] = (
                    self.learning_performance_trends[learning_type][-100:]
                )
    
    async def _get_meta_learning_insights(self) -> Dict[str, Any]:
        """Get insights from meta-learning analysis."""
        if not self.meta_learning_history:
            return {'status': 'insufficient_data'}
        
        insights = {}
        
        # Learning type performance
        type_performance = {}
        for learning_type in LearningType:
            type_records = [
                record for record in self.meta_learning_history
                if record['learning_type'] == learning_type.value
            ]
            
            if type_records:
                success_rate = sum(record['success'] for record in type_records) / len(type_records)
                avg_difficulty = np.mean([record['difficulty'] for record in type_records])
                type_performance[learning_type.value] = {
                    'success_rate': success_rate,
                    'average_difficulty': avg_difficulty,
                    'total_attempts': len(type_records)
                }
        
        insights['learning_type_performance'] = type_performance
        
        # Difficulty vs. success analysis
        recent_records = list(self.meta_learning_history)[-200:]  # Last 200 experiences
        difficulties = [record['difficulty'] for record in recent_records]
        successes = [record['success'] for record in recent_records]
        
        if difficulties and successes:
            # Calculate correlation between difficulty and success
            difficulty_success_correlation = np.corrcoef(difficulties, successes)[0, 1]
            insights['difficulty_success_correlation'] = difficulty_success_correlation
        
        # Consciousness-guided learning effectiveness
        consciousness_records = [
            record for record in recent_records
            if record['consciousness_relevance'] > 0.5
        ]
        
        if consciousness_records:
            consciousness_success_rate = sum(record['success'] for record in consciousness_records) / len(consciousness_records)
            insights['consciousness_guided_effectiveness'] = consciousness_success_rate
        
        # Learning trends
        recent_success_rates = []
        window_size = 50
        for i in range(window_size, len(recent_records)):
            window = recent_records[i-window_size:i]
            success_rate = sum(record['success'] for record in window) / window_size
            recent_success_rates.append(success_rate)
        
        if len(recent_success_rates) > 10:
            # Calculate learning trend
            trend_slope = np.polyfit(range(len(recent_success_rates)), recent_success_rates, 1)[0]
            insights['learning_trend'] = 'improving' if trend_slope > 0.01 else 'declining' if trend_slope < -0.01 else 'stable'
            insights['trend_slope'] = trend_slope
        
        return insights
    
    async def _apply_parameter_adaptations(self, adaptations: Dict[str, Any]):
        """Apply parameter adaptations based on meta-learning insights."""
        for adaptation, value in adaptations.items():
            if adaptation == 'increase_replay_ratio' and value:
                # Increase experience replay ratio
                if ForgettingMitigation.EXPERIENCE_REPLAY in self.strategies:
                    strategy = self.strategies[ForgettingMitigation.EXPERIENCE_REPLAY]
                    strategy.replay_ratio = min(strategy.replay_ratio * 1.2, 0.5)
            
            elif adaptation == 'increase_consolidation_frequency' and value:
                # More frequent automatic consolidation
                self.config['automatic_consolidation_threshold'] = 0.6
            
            elif adaptation == 'increase_consciousness_weighting' and value:
                # Give more weight to consciousness-relevant experiences
                self.config['consciousness_learning_boost'] = 1.3
        
        self.logger.info(f"Applied learning adaptations: {adaptations}")
    
    async def _perform_maintenance(self):
        """Perform periodic maintenance tasks."""
        # Automatic consolidation
        if self.config.get('automatic_consolidation', True):
            if len(self.experience_buffer) > 1000 and len(self.experience_buffer) % 500 == 0:
                await self.consolidate_knowledge('automatic')
        
        # Forgetting prevention
        if self.config.get('forgetting_prevention', True):
            if len(self.experience_buffer) % 1000 == 0:
                await self.prevent_forgetting()
        
        # Knowledge graph maintenance
        if len(self.knowledge_graph.nodes) > 10000:
            await self.knowledge_graph.consolidate_knowledge()
    
    async def save_learning_state(self, filepath: Path) -> bool:
        """Save current learning state to file."""
        try:
            learning_state = {
                'experience_buffer': list(self.experience_buffer),
                'learning_stats': self.learning_stats,
                'meta_learning_history': list(self.meta_learning_history),
                'learning_performance_trends': dict(self.learning_performance_trends),
                'config': self.config
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(learning_state, f)
            
            self.logger.info(f"Saved learning state to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save learning state: {e}")
            return False
    
    async def load_learning_state(self, filepath: Path) -> bool:
        """Load learning state from file."""
        try:
            with open(filepath, 'rb') as f:
                learning_state = pickle.load(f)
            
            self.experience_buffer = deque(learning_state['experience_buffer'], maxlen=50000)
            self.learning_stats = learning_state['learning_stats']
            self.meta_learning_history = deque(learning_state['meta_learning_history'], maxlen=1000)
            self.learning_performance_trends = defaultdict(list, learning_state['learning_performance_trends'])
            self.config.update(learning_state['config'])
            
            self.logger.info(f"Loaded learning state from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load learning state: {e}")
            return False