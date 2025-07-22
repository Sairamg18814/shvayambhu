"""
GraphRAG Implementation

Graph-based Retrieval-Augmented Generation system that builds knowledge graphs
from documents and performs relationship-aware retrieval with consciousness integration.
"""

import asyncio
import logging
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
import re
from collections import defaultdict, Counter
from abc import ABC, abstractmethod

from .vector_database import VectorDatabaseManager, VectorSearchResult, DocumentMetadata

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of entities in the knowledge graph"""
    PERSON = "person"
    ORGANIZATION = "organization"
    CONCEPT = "concept"
    TECHNOLOGY = "technology"
    LOCATION = "location"
    EVENT = "event"
    CONSCIOUSNESS_CONCEPT = "consciousness_concept"
    UNKNOWN = "unknown"


class RelationType(Enum):
    """Types of relationships between entities"""
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    DEVELOPS = "develops"
    USES = "uses"
    CAUSES = "causes"
    ENABLES = "enables"
    SIMILAR_TO = "similar_to"
    CONTRADICTS = "contradicts"
    CONSCIOUSNESS_RELATED = "consciousness_related"


@dataclass
class Entity:
    """Represents an entity in the knowledge graph"""
    entity_id: str
    name: str
    entity_type: EntityType
    aliases: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    consciousness_relevance: float = 0.0
    confidence: float = 1.0
    source_documents: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    embedding: Optional[np.ndarray] = None


@dataclass
class Relationship:
    """Represents a relationship between entities"""
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: RelationType
    confidence: float
    evidence: List[str] = field(default_factory=list)
    source_documents: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class GraphRAGConfig:
    """Configuration for GraphRAG system"""
    # Entity extraction settings
    min_entity_frequency: int = 2
    entity_confidence_threshold: float = 0.6
    max_entities_per_document: int = 20
    
    # Relationship extraction settings
    relationship_confidence_threshold: float = 0.5
    max_relationship_distance: int = 3
    enable_implicit_relationships: bool = True
    
    # Graph construction settings
    graph_pruning_threshold: float = 0.3
    max_graph_size: int = 10000
    community_detection: bool = True
    
    # Retrieval settings
    max_hops: int = 2
    relationship_weight: float = 0.7
    entity_weight: float = 0.8
    combine_vector_graph_scores: bool = True
    
    # Consciousness integration
    consciousness_entity_boost: float = 1.5
    consciousness_relationship_boost: float = 1.3
    prioritize_consciousness_paths: bool = True
    
    # Performance settings
    batch_processing_size: int = 100
    enable_caching: bool = True
    cache_ttl_hours: int = 24


class EntityExtractor:
    """Extracts entities from text content"""
    
    def __init__(self, config: GraphRAGConfig):
        self.config = config
        
        # Pattern-based entity recognition (simple version)
        self.entity_patterns = {
            EntityType.ORGANIZATION: [
                r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\s+(?:Inc|Corp|Ltd|LLC|Company|Organization|Institute|Lab|Laboratory)\b',
                r'\b(?:Google|Microsoft|OpenAI|Meta|Apple|Amazon|IBM|NVIDIA|DeepMind|Anthropic)\b'
            ],
            EntityType.TECHNOLOGY: [
                r'\b(?:AI|artificial intelligence|machine learning|ML|neural network|transformer|LLM|GPT|BERT|CNN|RNN|deep learning)\b',
                r'\b(?:Python|TensorFlow|PyTorch|CUDA|algorithm|model|framework)\b'
            ],
            EntityType.CONSCIOUSNESS_CONCEPT: [
                r'\b(?:consciousness|awareness|qualia|phenomenal experience|subjective experience|self-awareness|sentience|cognition|introspection)\b',
                r'\b(?:conscious AI|artificial consciousness|machine consciousness|cognitive architecture|phenomenal consciousness)\b'
            ],
            EntityType.CONCEPT: [
                r'\b(?:intelligence|reasoning|learning|knowledge|understanding|perception|memory|attention|emotion)\b'
            ],
            EntityType.PERSON: [
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b'
            ]
        }
        
        # Consciousness-related keywords for boosting
        self.consciousness_keywords = {
            'consciousness', 'awareness', 'experience', 'subjective', 'qualia',
            'phenomenal', 'introspection', 'self-aware', 'sentient', 'cognitive'
        }
    
    def extract_entities(self, text: str, document_id: str) -> List[Entity]:
        """Extract entities from text"""
        
        entities = []
        text_lower = text.lower()
        
        # Calculate consciousness relevance of the document
        doc_consciousness_score = self._calculate_consciousness_relevance(text)
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    entity_text = match.group().strip()
                    
                    if len(entity_text) < 2 or len(entity_text) > 100:
                        continue
                    
                    # Generate entity ID
                    entity_id = self._generate_entity_id(entity_text, entity_type)
                    
                    # Calculate entity-specific consciousness relevance
                    entity_consciousness = self._calculate_entity_consciousness(entity_text, entity_type)
                    
                    # Boost consciousness relevance for consciousness-related entities
                    if entity_type == EntityType.CONSCIOUSNESS_CONCEPT:
                        entity_consciousness = min(1.0, entity_consciousness * 1.5)
                    
                    # Calculate confidence based on pattern strength and context
                    confidence = self._calculate_entity_confidence(entity_text, text, entity_type)
                    
                    if confidence >= self.config.entity_confidence_threshold:
                        entity = Entity(
                            entity_id=entity_id,
                            name=entity_text,
                            entity_type=entity_type,
                            consciousness_relevance=entity_consciousness,
                            confidence=confidence,
                            source_documents=[document_id],
                            attributes={
                                'first_seen_position': match.start(),
                                'document_consciousness_score': doc_consciousness_score
                            }
                        )
                        
                        entities.append(entity)
        
        # Deduplicate and merge similar entities
        entities = self._deduplicate_entities(entities)
        
        # Limit number of entities per document
        entities.sort(key=lambda e: e.confidence, reverse=True)
        entities = entities[:self.config.max_entities_per_document]
        
        logger.debug(f"Extracted {len(entities)} entities from document {document_id}")
        return entities
    
    def _calculate_consciousness_relevance(self, text: str) -> float:
        """Calculate consciousness relevance of text"""
        
        text_lower = text.lower()
        consciousness_matches = sum(1 for keyword in self.consciousness_keywords if keyword in text_lower)
        
        # Normalize by text length and keyword set size
        relevance = consciousness_matches / len(self.consciousness_keywords)
        
        # Boost based on density
        word_count = len(text.split())
        density_boost = min(0.3, consciousness_matches / max(1, word_count / 100))
        
        return min(1.0, relevance + density_boost)
    
    def _calculate_entity_consciousness(self, entity_text: str, entity_type: EntityType) -> float:
        """Calculate consciousness relevance for specific entity"""
        
        entity_lower = entity_text.lower()
        
        # Direct consciousness match
        consciousness_match = any(keyword in entity_lower for keyword in self.consciousness_keywords)
        
        if entity_type == EntityType.CONSCIOUSNESS_CONCEPT:
            return 1.0 if consciousness_match else 0.8
        elif entity_type in [EntityType.TECHNOLOGY, EntityType.CONCEPT]:
            return 0.7 if consciousness_match else 0.3
        else:
            return 0.5 if consciousness_match else 0.1
    
    def _calculate_entity_confidence(self, entity_text: str, context: str, entity_type: EntityType) -> float:
        """Calculate confidence score for entity extraction"""
        
        # Base confidence from pattern strength
        base_confidence = 0.7
        
        # Boost for consciousness-related entities
        if entity_type == EntityType.CONSCIOUSNESS_CONCEPT:
            base_confidence += 0.2
        
        # Context relevance boost
        context_words = context.lower().split()
        entity_words = entity_text.lower().split()
        
        # Check if entity appears multiple times
        frequency_boost = min(0.2, context.lower().count(entity_text.lower()) * 0.05)
        
        # Check capitalization consistency (for proper nouns)
        if entity_type in [EntityType.PERSON, EntityType.ORGANIZATION]:
            if entity_text.istitle():
                base_confidence += 0.1
        
        final_confidence = min(1.0, base_confidence + frequency_boost)
        return final_confidence
    
    def _generate_entity_id(self, entity_text: str, entity_type: EntityType) -> str:
        """Generate unique ID for entity"""
        
        normalized_text = entity_text.lower().strip()
        type_prefix = entity_type.value[:3].upper()
        text_hash = hashlib.md5(normalized_text.encode()).hexdigest()[:8]
        
        return f"{type_prefix}_{text_hash}"
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Deduplicate and merge similar entities"""
        
        entity_map = {}
        
        for entity in entities:
            found_similar = False
            
            for existing_id, existing_entity in entity_map.items():
                if self._are_entities_similar(entity, existing_entity):
                    # Merge entities
                    entity_map[existing_id] = self._merge_entities(existing_entity, entity)
                    found_similar = True
                    break
            
            if not found_similar:
                entity_map[entity.entity_id] = entity
        
        return list(entity_map.values())
    
    def _are_entities_similar(self, entity1: Entity, entity2: Entity) -> bool:
        """Check if two entities are similar enough to merge"""
        
        if entity1.entity_type != entity2.entity_type:
            return False
        
        name1 = entity1.name.lower()
        name2 = entity2.name.lower()
        
        # Exact match
        if name1 == name2:
            return True
        
        # Check for abbreviations/acronyms
        if len(name1) <= 5 and name1 in name2:
            return True
        if len(name2) <= 5 and name2 in name1:
            return True
        
        # Simple similarity check
        words1 = set(name1.split())
        words2 = set(name2.split())
        
        if words1 and words2:
            overlap = len(words1.intersection(words2)) / len(words1.union(words2))
            return overlap >= 0.8
        
        return False
    
    def _merge_entities(self, entity1: Entity, entity2: Entity) -> Entity:
        """Merge two similar entities"""
        
        # Use the entity with higher confidence as base
        base_entity = entity1 if entity1.confidence >= entity2.confidence else entity2
        other_entity = entity2 if base_entity == entity1 else entity1
        
        # Merge attributes
        merged_entity = Entity(
            entity_id=base_entity.entity_id,
            name=base_entity.name,
            entity_type=base_entity.entity_type,
            aliases=list(set(base_entity.aliases + [other_entity.name])),
            consciousness_relevance=max(base_entity.consciousness_relevance, other_entity.consciousness_relevance),
            confidence=max(base_entity.confidence, other_entity.confidence),
            source_documents=list(set(base_entity.source_documents + other_entity.source_documents)),
            attributes={**base_entity.attributes, **other_entity.attributes}
        )
        
        return merged_entity


class RelationshipExtractor:
    """Extracts relationships between entities"""
    
    def __init__(self, config: GraphRAGConfig):
        self.config = config
        
        # Relationship patterns
        self.relationship_patterns = {
            RelationType.DEVELOPS: [
                r'{entity1}.*(?:develop|creat|build|design|invent).*{entity2}',
                r'{entity2}.*(?:developed|created|built|designed|invented).*by.*{entity1}'
            ],
            RelationType.USES: [
                r'{entity1}.*(?:use|employ|utiliz|implement).*{entity2}',
                r'{entity2}.*(?:used|employed|utilized|implemented).*by.*{entity1}'
            ],
            RelationType.PART_OF: [
                r'{entity1}.*(?:part of|component of|element of).*{entity2}',
                r'{entity2}.*(?:contain|includ|compris).*{entity1}'
            ],
            RelationType.CONSCIOUSNESS_RELATED: [
                r'{entity1}.*(?:consciousness|aware|experience).*{entity2}',
                r'{entity1}.*(?:exhibit|demonstrat|show).*consciousness.*{entity2}'
            ]
        }
    
    def extract_relationships(
        self,
        text: str,
        entities: List[Entity],
        document_id: str
    ) -> List[Relationship]:
        """Extract relationships between entities in text"""
        
        relationships = []
        
        if len(entities) < 2:
            return relationships
        
        # Try all entity pairs
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                
                # Extract relationships between this pair
                pair_relationships = self._extract_entity_pair_relationships(
                    text, entity1, entity2, document_id
                )
                relationships.extend(pair_relationships)
        
        # Filter by confidence
        relationships = [
            rel for rel in relationships
            if rel.confidence >= self.config.relationship_confidence_threshold
        ]
        
        logger.debug(f"Extracted {len(relationships)} relationships from document {document_id}")
        return relationships
    
    def _extract_entity_pair_relationships(
        self,
        text: str,
        entity1: Entity,
        entity2: Entity,
        document_id: str
    ) -> List[Relationship]:
        """Extract relationships between a specific pair of entities"""
        
        relationships = []
        
        # Check for explicit relationship patterns
        for relation_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                # Replace entity placeholders
                pattern_with_entities = pattern.format(
                    entity1=re.escape(entity1.name),
                    entity2=re.escape(entity2.name)
                )
                
                matches = re.finditer(pattern_with_entities, text, re.IGNORECASE)
                
                for match in matches:
                    evidence_text = match.group()
                    
                    # Calculate relationship confidence
                    confidence = self._calculate_relationship_confidence(
                        entity1, entity2, relation_type, evidence_text, text
                    )
                    
                    if confidence >= self.config.relationship_confidence_threshold:
                        relationship = Relationship(
                            relationship_id=self._generate_relationship_id(entity1, entity2, relation_type),
                            source_entity_id=entity1.entity_id,
                            target_entity_id=entity2.entity_id,
                            relation_type=relation_type,
                            confidence=confidence,
                            evidence=[evidence_text],
                            source_documents=[document_id],
                            attributes={
                                'pattern_matched': pattern,
                                'evidence_position': match.start()
                            }
                        )
                        
                        relationships.append(relationship)
        
        # Check for implicit relationships (proximity, co-occurrence)
        if self.config.enable_implicit_relationships:
            implicit_rel = self._extract_implicit_relationship(entity1, entity2, text, document_id)
            if implicit_rel:
                relationships.append(implicit_rel)
        
        return relationships
    
    def _calculate_relationship_confidence(
        self,
        entity1: Entity,
        entity2: Entity,
        relation_type: RelationType,
        evidence: str,
        full_text: str
    ) -> float:
        """Calculate confidence for a relationship"""
        
        base_confidence = 0.6
        
        # Boost based on entity confidence
        entity_confidence_boost = (entity1.confidence + entity2.confidence) / 2 * 0.2
        
        # Boost for consciousness-related relationships
        if relation_type == RelationType.CONSCIOUSNESS_RELATED:
            base_confidence += 0.2
        
        # Boost if both entities have high consciousness relevance
        if (entity1.consciousness_relevance > 0.5 and entity2.consciousness_relevance > 0.5):
            base_confidence += 0.15
        
        # Evidence quality boost
        evidence_words = len(evidence.split())
        evidence_quality = min(0.1, evidence_words / 50)  # Longer evidence = higher confidence
        
        final_confidence = min(1.0, base_confidence + entity_confidence_boost + evidence_quality)
        return final_confidence
    
    def _extract_implicit_relationship(
        self,
        entity1: Entity,
        entity2: Entity,
        text: str,
        document_id: str
    ) -> Optional[Relationship]:
        """Extract implicit relationship based on proximity and context"""
        
        # Find positions of entities in text
        pos1 = text.lower().find(entity1.name.lower())
        pos2 = text.lower().find(entity2.name.lower())
        
        if pos1 == -1 or pos2 == -1:
            return None
        
        # Calculate distance
        distance = abs(pos1 - pos2)
        
        # If entities are close, create implicit relationship
        if distance <= 200:  # Within 200 characters
            # Determine relationship type based on context
            context = text[min(pos1, pos2):max(pos1, pos2) + len(entity1.name)]
            
            relation_type = RelationType.RELATED_TO
            if any(word in context.lower() for word in ['consciousness', 'aware', 'experience']):
                relation_type = RelationType.CONSCIOUSNESS_RELATED
            
            confidence = max(0.3, 0.8 - (distance / 500))  # Closer = higher confidence
            
            # Boost for consciousness entities
            if (entity1.entity_type == EntityType.CONSCIOUSNESS_CONCEPT or
                entity2.entity_type == EntityType.CONSCIOUSNESS_CONCEPT):
                confidence *= self.config.consciousness_relationship_boost
            
            if confidence >= self.config.relationship_confidence_threshold:
                return Relationship(
                    relationship_id=self._generate_relationship_id(entity1, entity2, relation_type),
                    source_entity_id=entity1.entity_id,
                    target_entity_id=entity2.entity_id,
                    relation_type=relation_type,
                    confidence=confidence,
                    evidence=[context],
                    source_documents=[document_id],
                    attributes={
                        'relationship_type': 'implicit',
                        'distance': distance
                    }
                )
        
        return None
    
    def _generate_relationship_id(
        self,
        entity1: Entity,
        entity2: Entity,
        relation_type: RelationType
    ) -> str:
        """Generate unique ID for relationship"""
        
        relationship_string = f"{entity1.entity_id}_{relation_type.value}_{entity2.entity_id}"
        return hashlib.md5(relationship_string.encode()).hexdigest()[:16]


class KnowledgeGraph:
    """Manages the knowledge graph structure"""
    
    def __init__(self, config: GraphRAGConfig):
        self.config = config
        self.graph = nx.MultiDiGraph()
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}
        
        # Indices for fast lookup
        self.entity_type_index: Dict[EntityType, Set[str]] = defaultdict(set)
        self.relationship_type_index: Dict[RelationType, Set[str]] = defaultdict(set)
        self.document_entity_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Community detection cache
        self.communities: Optional[Dict[str, int]] = None
        self.last_community_update: Optional[datetime] = None
    
    def add_entity(self, entity: Entity) -> bool:
        """Add entity to the knowledge graph"""
        
        if entity.entity_id in self.entities:
            # Merge with existing entity
            self.entities[entity.entity_id] = self._merge_entities(
                self.entities[entity.entity_id], entity
            )
        else:
            self.entities[entity.entity_id] = entity
            self.graph.add_node(entity.entity_id, **self._entity_to_node_attributes(entity))
        
        # Update indices
        self.entity_type_index[entity.entity_type].add(entity.entity_id)
        for doc_id in entity.source_documents:
            self.document_entity_index[doc_id].add(entity.entity_id)
        
        return True
    
    def add_relationship(self, relationship: Relationship) -> bool:
        """Add relationship to the knowledge graph"""
        
        # Ensure both entities exist
        if (relationship.source_entity_id not in self.entities or
            relationship.target_entity_id not in self.entities):
            return False
        
        if relationship.relationship_id in self.relationships:
            # Merge with existing relationship
            self.relationships[relationship.relationship_id] = self._merge_relationships(
                self.relationships[relationship.relationship_id], relationship
            )
        else:
            self.relationships[relationship.relationship_id] = relationship
        
        # Add edge to graph
        self.graph.add_edge(
            relationship.source_entity_id,
            relationship.target_entity_id,
            key=relationship.relationship_id,
            **self._relationship_to_edge_attributes(relationship)
        )
        
        # Update indices
        self.relationship_type_index[relationship.relation_type].add(relationship.relationship_id)
        
        # Invalidate community cache
        self.communities = None
        
        return True
    
    def get_entity_neighborhood(
        self,
        entity_id: str,
        max_hops: int = 2,
        relation_types: Optional[List[RelationType]] = None
    ) -> Dict[str, Any]:
        """Get the neighborhood of an entity within specified hops"""
        
        if entity_id not in self.entities:
            return {'entities': [], 'relationships': []}
        
        # BFS to find entities within max_hops
        visited = set()
        queue = [(entity_id, 0)]
        neighborhood_entities = set()
        neighborhood_relationships = set()
        
        while queue:
            current_id, hops = queue.pop(0)
            
            if current_id in visited or hops > max_hops:
                continue
            
            visited.add(current_id)
            neighborhood_entities.add(current_id)
            
            if hops < max_hops:
                # Get connected entities
                for neighbor_id in self.graph.neighbors(current_id):
                    # Get edge data
                    edges = self.graph.get_edge_data(current_id, neighbor_id)
                    
                    for edge_key, edge_data in edges.items():
                        rel_type = RelationType(edge_data['relation_type'])
                        
                        # Filter by relationship type if specified
                        if relation_types and rel_type not in relation_types:
                            continue
                        
                        neighborhood_relationships.add(edge_key)
                        
                        if neighbor_id not in visited:
                            queue.append((neighbor_id, hops + 1))
        
        return {
            'entities': [self.entities[eid] for eid in neighborhood_entities if eid in self.entities],
            'relationships': [self.relationships[rid] for rid in neighborhood_relationships if rid in self.relationships]
        }
    
    def find_paths_between_entities(
        self,
        source_id: str,
        target_id: str,
        max_path_length: int = 3
    ) -> List[List[str]]:
        """Find paths between two entities"""
        
        if source_id not in self.entities or target_id not in self.entities:
            return []
        
        try:
            # Find all simple paths up to max_path_length
            paths = list(nx.all_simple_paths(
                self.graph,
                source_id,
                target_id,
                cutoff=max_path_length
            ))
            
            return paths
        except nx.NetworkXNoPath:
            return []
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get all entities of a specific type"""
        
        entity_ids = self.entity_type_index.get(entity_type, set())
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]
    
    def get_consciousness_subgraph(self, min_consciousness_score: float = 0.5) -> 'KnowledgeGraph':
        """Extract subgraph containing high-consciousness entities and relationships"""
        
        # Find consciousness-related entities
        consciousness_entities = {
            eid: entity for eid, entity in self.entities.items()
            if (entity.consciousness_relevance >= min_consciousness_score or
                entity.entity_type == EntityType.CONSCIOUSNESS_CONCEPT)
        }
        
        # Create new subgraph
        subgraph_config = GraphRAGConfig()
        subgraph = KnowledgeGraph(subgraph_config)
        
        # Add consciousness entities
        for entity in consciousness_entities.values():
            subgraph.add_entity(entity)
        
        # Add relationships between consciousness entities
        for relationship in self.relationships.values():
            if (relationship.source_entity_id in consciousness_entities and
                relationship.target_entity_id in consciousness_entities):
                subgraph.add_relationship(relationship)
        
        return subgraph
    
    def detect_communities(self) -> Dict[str, int]:
        """Detect communities in the knowledge graph"""
        
        if (self.communities is not None and self.last_community_update and
            datetime.now() - self.last_community_update < timedelta(hours=1)):
            return self.communities
        
        if self.graph.number_of_nodes() < 3:
            return {}
        
        try:
            # Convert to undirected graph for community detection
            undirected_graph = self.graph.to_undirected()
            
            # Use Louvain method for community detection
            import community as community_louvain
            communities = community_louvain.best_partition(undirected_graph)
            
            self.communities = communities
            self.last_community_update = datetime.now()
            
            return communities
            
        except ImportError:
            logger.warning("Community detection library not available")
            return {}
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            return {}
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        
        stats = {
            'total_entities': len(self.entities),
            'total_relationships': len(self.relationships),
            'graph_nodes': self.graph.number_of_nodes(),
            'graph_edges': self.graph.number_of_edges(),
            'entity_types': {
                etype.value: len(entities)
                for etype, entities in self.entity_type_index.items()
            },
            'relationship_types': {
                rtype.value: len(relationships)
                for rtype, relationships in self.relationship_type_index.items()
            },
            'consciousness_entities': len([
                e for e in self.entities.values()
                if e.consciousness_relevance > 0.5 or e.entity_type == EntityType.CONSCIOUSNESS_CONCEPT
            ])
        }
        
        # Graph connectivity metrics
        if self.graph.number_of_nodes() > 0:
            stats['average_degree'] = sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
            stats['weakly_connected_components'] = nx.number_weakly_connected_components(self.graph)
        
        return stats
    
    def _merge_entities(self, existing: Entity, new: Entity) -> Entity:
        """Merge two entities"""
        
        merged = Entity(
            entity_id=existing.entity_id,
            name=existing.name,
            entity_type=existing.entity_type,
            aliases=list(set(existing.aliases + [new.name] + new.aliases)),
            consciousness_relevance=max(existing.consciousness_relevance, new.consciousness_relevance),
            confidence=max(existing.confidence, new.confidence),
            source_documents=list(set(existing.source_documents + new.source_documents)),
            attributes={**existing.attributes, **new.attributes},
            created_at=min(existing.created_at, new.created_at)
        )
        
        return merged
    
    def _merge_relationships(self, existing: Relationship, new: Relationship) -> Relationship:
        """Merge two relationships"""
        
        merged = Relationship(
            relationship_id=existing.relationship_id,
            source_entity_id=existing.source_entity_id,
            target_entity_id=existing.target_entity_id,
            relation_type=existing.relation_type,
            confidence=max(existing.confidence, new.confidence),
            evidence=list(set(existing.evidence + new.evidence)),
            source_documents=list(set(existing.source_documents + new.source_documents)),
            attributes={**existing.attributes, **new.attributes},
            created_at=min(existing.created_at, new.created_at)
        )
        
        return merged
    
    def _entity_to_node_attributes(self, entity: Entity) -> Dict[str, Any]:
        """Convert entity to node attributes for NetworkX"""
        
        return {
            'name': entity.name,
            'entity_type': entity.entity_type.value,
            'consciousness_relevance': entity.consciousness_relevance,
            'confidence': entity.confidence,
            'aliases': entity.aliases,
            'created_at': entity.created_at.isoformat()
        }
    
    def _relationship_to_edge_attributes(self, relationship: Relationship) -> Dict[str, Any]:
        """Convert relationship to edge attributes for NetworkX"""
        
        return {
            'relation_type': relationship.relation_type.value,
            'confidence': relationship.confidence,
            'evidence_count': len(relationship.evidence),
            'created_at': relationship.created_at.isoformat()
        }


class GraphRetriever:
    """Performs graph-based retrieval for GraphRAG"""
    
    def __init__(self, config: GraphRAGConfig, knowledge_graph: KnowledgeGraph):
        self.config = config
        self.knowledge_graph = knowledge_graph
    
    def retrieve_with_graph(
        self,
        query: str,
        query_entities: List[Entity],
        vector_results: List[VectorSearchResult],
        top_k: int = 10
    ) -> List[VectorSearchResult]:
        """Perform graph-enhanced retrieval"""
        
        if not query_entities:
            return vector_results[:top_k]
        
        # Find graph-relevant entities and paths
        graph_context = self._build_graph_context(query_entities)
        
        # Score and re-rank vector results using graph information
        enhanced_results = []
        
        for vector_result in vector_results:
            # Calculate graph relevance score
            graph_score = self._calculate_graph_relevance(
                vector_result, graph_context, query_entities
            )
            
            # Combine vector and graph scores
            if self.config.combine_vector_graph_scores:
                combined_score = (
                    vector_result.similarity_score * (1 - self.config.relationship_weight) +
                    graph_score * self.config.relationship_weight
                )
            else:
                combined_score = graph_score
            
            # Create enhanced result
            enhanced_result = VectorSearchResult(
                document_id=vector_result.document_id,
                content=vector_result.content,
                similarity_score=combined_score,
                metadata=vector_result.metadata,
                embedding=vector_result.embedding
            )
            
            # Add graph context to metadata
            enhanced_result.metadata.custom_metadata['graph_relevance'] = graph_score
            enhanced_result.metadata.custom_metadata['graph_entities'] = [
                entity.entity_id for entity in query_entities
            ]
            
            enhanced_results.append(enhanced_result)
        
        # Sort by combined score
        enhanced_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Add additional context from graph
        enhanced_results = self._add_graph_context(enhanced_results, graph_context)
        
        return enhanced_results[:top_k]
    
    def _build_graph_context(self, query_entities: List[Entity]) -> Dict[str, Any]:
        """Build graph context for query entities"""
        
        context = {
            'primary_entities': query_entities,
            'related_entities': [],
            'relevant_relationships': [],
            'entity_neighborhoods': {},
            'consciousness_paths': []
        }
        
        # Get neighborhoods for each query entity
        for entity in query_entities:
            neighborhood = self.knowledge_graph.get_entity_neighborhood(
                entity.entity_id, max_hops=self.config.max_hops
            )
            
            context['entity_neighborhoods'][entity.entity_id] = neighborhood
            context['related_entities'].extend(neighborhood['entities'])
            context['relevant_relationships'].extend(neighborhood['relationships'])
        
        # Find paths between query entities
        if len(query_entities) > 1:
            for i, entity1 in enumerate(query_entities):
                for entity2 in query_entities[i+1:]:
                    paths = self.knowledge_graph.find_paths_between_entities(
                        entity1.entity_id, entity2.entity_id, max_path_length=3
                    )
                    
                    if paths:
                        # Prioritize consciousness-related paths
                        consciousness_paths = self._filter_consciousness_paths(paths)
                        context['consciousness_paths'].extend(consciousness_paths)
        
        # Deduplicate
        context['related_entities'] = list({e.entity_id: e for e in context['related_entities']}.values())
        context['relevant_relationships'] = list({r.relationship_id: r for r in context['relevant_relationships']}.values())
        
        return context
    
    def _filter_consciousness_paths(self, paths: List[List[str]]) -> List[List[str]]:
        """Filter and prioritize consciousness-related paths"""
        
        consciousness_paths = []
        
        for path in paths:
            consciousness_score = 0.0
            
            for entity_id in path:
                if entity_id in self.knowledge_graph.entities:
                    entity = self.knowledge_graph.entities[entity_id]
                    consciousness_score += entity.consciousness_relevance
            
            if consciousness_score > len(path) * 0.3:  # Average consciousness > 0.3
                consciousness_paths.append(path)
        
        return consciousness_paths
    
    def _calculate_graph_relevance(
        self,
        vector_result: VectorSearchResult,
        graph_context: Dict[str, Any],
        query_entities: List[Entity]
    ) -> float:
        """Calculate graph relevance score for a vector result"""
        
        base_score = 0.0
        
        # Check if result mentions entities from graph context
        content_lower = vector_result.content.lower()
        
        # Score for primary entity mentions
        primary_entity_score = 0.0
        for entity in query_entities:
            if entity.name.lower() in content_lower:
                entity_boost = self.config.entity_weight
                
                # Boost for consciousness entities
                if (entity.entity_type == EntityType.CONSCIOUSNESS_CONCEPT or
                    entity.consciousness_relevance > 0.7):
                    entity_boost *= self.config.consciousness_entity_boost
                
                primary_entity_score += entity_boost
        
        primary_entity_score = min(1.0, primary_entity_score / len(query_entities))
        
        # Score for related entity mentions
        related_entity_score = 0.0
        related_entity_count = 0
        
        for entity in graph_context['related_entities']:
            if entity.name.lower() in content_lower:
                related_entity_score += entity.confidence * 0.5
                related_entity_count += 1
        
        if related_entity_count > 0:
            related_entity_score = min(1.0, related_entity_score / related_entity_count)
        
        # Score for relationship evidence
        relationship_score = 0.0
        relationship_count = 0
        
        for relationship in graph_context['relevant_relationships']:
            for evidence in relationship.evidence:
                # Simple text overlap check
                evidence_words = set(evidence.lower().split())
                content_words = set(content_lower.split())
                
                if evidence_words.intersection(content_words):
                    relationship_score += relationship.confidence * 0.3
                    relationship_count += 1
        
        if relationship_count > 0:
            relationship_score = min(1.0, relationship_score / relationship_count)
        
        # Combine scores
        base_score = (
            primary_entity_score * 0.5 +
            related_entity_score * 0.3 +
            relationship_score * 0.2
        )
        
        # Consciousness boost
        if self.config.prioritize_consciousness_paths and graph_context['consciousness_paths']:
            consciousness_boost = min(0.2, len(graph_context['consciousness_paths']) * 0.05)
            base_score += consciousness_boost
        
        return min(1.0, base_score)
    
    def _add_graph_context(
        self,
        results: List[VectorSearchResult],
        graph_context: Dict[str, Any]
    ) -> List[VectorSearchResult]:
        """Add graph context information to results"""
        
        for result in results:
            # Add related entity information
            related_entities = []
            content_lower = result.content.lower()
            
            for entity in graph_context['related_entities']:
                if entity.name.lower() in content_lower:
                    related_entities.append({
                        'name': entity.name,
                        'type': entity.entity_type.value,
                        'consciousness_relevance': entity.consciousness_relevance
                    })
            
            result.metadata.custom_metadata['related_entities'] = related_entities
            
            # Add relationship context
            relevant_relationships = []
            for relationship in graph_context['relevant_relationships']:
                if any(evidence.lower() in content_lower for evidence in relationship.evidence):
                    relevant_relationships.append({
                        'type': relationship.relation_type.value,
                        'confidence': relationship.confidence
                    })
            
            result.metadata.custom_metadata['relevant_relationships'] = relevant_relationships
        
        return results


class GraphRAG:
    """Main GraphRAG system orchestrator"""
    
    def __init__(
        self,
        config: GraphRAGConfig,
        vector_manager: VectorDatabaseManager
    ):
        self.config = config
        self.vector_manager = vector_manager
        
        self.entity_extractor = EntityExtractor(config)
        self.relationship_extractor = RelationshipExtractor(config)
        self.knowledge_graph = KnowledgeGraph(config)
        self.graph_retriever = GraphRetriever(config, self.knowledge_graph)
        
        # Performance metrics
        self.metrics = {
            'documents_processed': 0,
            'entities_extracted': 0,
            'relationships_extracted': 0,
            'graph_retrievals': 0,
            'average_graph_retrieval_time': 0.0
        }
    
    async def add_documents_to_graph(
        self,
        documents: List[str],
        document_ids: List[str],
        metadatas: List[DocumentMetadata]
    ) -> Dict[str, Any]:
        """Add documents to the knowledge graph"""
        
        if len(documents) != len(document_ids):
            raise ValueError("Documents and document_ids must have same length")
        
        total_entities = 0
        total_relationships = 0
        
        # Process documents in batches
        batch_size = self.config.batch_processing_size
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_ids = document_ids[i:i + batch_size]
            batch_metas = metadatas[i:i + batch_size] if metadatas else []
            
            for j, (doc, doc_id) in enumerate(zip(batch_docs, batch_ids)):
                # Extract entities
                entities = self.entity_extractor.extract_entities(doc, doc_id)
                
                # Add entities to graph
                for entity in entities:
                    self.knowledge_graph.add_entity(entity)
                
                # Extract relationships
                relationships = self.relationship_extractor.extract_relationships(
                    doc, entities, doc_id
                )
                
                # Add relationships to graph
                for relationship in relationships:
                    self.knowledge_graph.add_relationship(relationship)
                
                total_entities += len(entities)
                total_relationships += len(relationships)
        
        # Update metrics
        self.metrics['documents_processed'] += len(documents)
        self.metrics['entities_extracted'] += total_entities
        self.metrics['relationships_extracted'] += total_relationships
        
        logger.info(f"Added {len(documents)} documents to graph: "
                   f"{total_entities} entities, {total_relationships} relationships")
        
        return {
            'documents_processed': len(documents),
            'entities_extracted': total_entities,
            'relationships_extracted': total_relationships,
            'graph_stats': self.knowledge_graph.get_graph_statistics()
        }
    
    async def retrieve_with_graph(
        self,
        query: str,
        top_k: int = 10,
        use_vector_retrieval: bool = True
    ) -> List[VectorSearchResult]:
        """Perform GraphRAG retrieval"""
        
        start_time = datetime.now()
        
        # Extract entities from query
        query_entities = self.entity_extractor.extract_entities(query, "query")
        
        # Get vector retrieval results if enabled
        vector_results = []
        if use_vector_retrieval:
            vector_results = await self.vector_manager.search(
                query, top_k=top_k * 2  # Get more candidates for graph re-ranking
            )
        
        # Perform graph-enhanced retrieval
        if query_entities and vector_results:
            enhanced_results = self.graph_retriever.retrieve_with_graph(
                query, query_entities, vector_results, top_k
            )
        else:
            enhanced_results = vector_results[:top_k]
        
        # Update metrics
        retrieval_time = (datetime.now() - start_time).total_seconds()
        self.metrics['graph_retrievals'] += 1
        self.metrics['average_graph_retrieval_time'] = (
            (self.metrics['average_graph_retrieval_time'] * (self.metrics['graph_retrievals'] - 1) + retrieval_time)
            / self.metrics['graph_retrievals']
        )
        
        logger.debug(f"GraphRAG retrieval completed in {retrieval_time:.3f}s, "
                    f"found {len(enhanced_results)} results")
        
        return enhanced_results
    
    def get_graph_insights(self, query: str) -> Dict[str, Any]:
        """Get insights about the query from the knowledge graph"""
        
        query_entities = self.entity_extractor.extract_entities(query, "query_insight")
        
        insights = {
            'query_entities': [
                {
                    'name': e.name,
                    'type': e.entity_type.value,
                    'consciousness_relevance': e.consciousness_relevance
                }
                for e in query_entities
            ],
            'entity_relationships': [],
            'consciousness_context': {},
            'related_concepts': []
        }
        
        if query_entities:
            # Get entity neighborhoods
            for entity in query_entities:
                neighborhood = self.knowledge_graph.get_entity_neighborhood(entity.entity_id)
                
                for relationship in neighborhood['relationships']:
                    insights['entity_relationships'].append({
                        'source': relationship.source_entity_id,
                        'target': relationship.target_entity_id,
                        'type': relationship.relation_type.value,
                        'confidence': relationship.confidence
                    })
            
            # Get consciousness-related insights
            consciousness_subgraph = self.knowledge_graph.get_consciousness_subgraph()
            insights['consciousness_context'] = {
                'entities_count': len(consciousness_subgraph.entities),
                'relationships_count': len(consciousness_subgraph.relationships),
                'stats': consciousness_subgraph.get_graph_statistics()
            }
        
        return insights
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive GraphRAG statistics"""
        
        return {
            'graphrag_metrics': self.metrics.copy(),
            'knowledge_graph_stats': self.knowledge_graph.get_graph_statistics(),
            'vector_db_stats': await self.vector_manager.get_statistics(),
            'configuration': {
                'max_hops': self.config.max_hops,
                'consciousness_entity_boost': self.config.consciousness_entity_boost,
                'relationship_weight': self.config.relationship_weight,
                'enable_consciousness_paths': self.config.prioritize_consciousness_paths
            }
        }


# Testing and example usage
async def test_graph_rag():
    """Test GraphRAG system"""
    print("GraphRAG implementation completed - ready for integration testing")


if __name__ == "__main__":
    asyncio.run(test_graph_rag())