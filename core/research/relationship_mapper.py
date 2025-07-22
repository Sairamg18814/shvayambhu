"""Advanced relationship mapping system for entity connections.

This module provides sophisticated relationship extraction and mapping capabilities
to automatically discover and classify connections between entities in text.
"""

import re
import math
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import logging
from itertools import combinations

from .graphrag import Entity, EntityType, Relationship, RelationType
from .entity_extraction import EntityCandidate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RelationshipStrength(Enum):
    """Strength levels for relationships."""
    WEAK = 0.3
    MODERATE = 0.5
    STRONG = 0.7
    VERY_STRONG = 0.9


@dataclass
class RelationshipCandidate:
    """Candidate relationship with extraction metadata."""
    source_entity: Union[Entity, EntityCandidate]
    target_entity: Union[Entity, EntityCandidate]
    relation_type: RelationType
    confidence: float
    evidence: List[str] = field(default_factory=list)
    context: str = ""
    features: Dict[str, Any] = field(default_factory=dict)
    
    def to_relationship(self) -> Relationship:
        """Convert candidate to Relationship object."""
        source_id = (self.source_entity.entity_id 
                    if hasattr(self.source_entity, 'entity_id') 
                    else self.source_entity.to_entity().entity_id)
        target_id = (self.target_entity.entity_id 
                    if hasattr(self.target_entity, 'entity_id') 
                    else self.target_entity.to_entity().entity_id)
        
        return Relationship(
            relationship_id="",
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type=self.relation_type,
            confidence=self.confidence,
            attributes=self.features
        )


class PatternBasedRelationshipExtractor:
    """Extract relationships using linguistic patterns."""
    
    def __init__(self):
        """Initialize pattern-based relationship extractor."""
        self.patterns = self._initialize_patterns()
        self.negation_words = {'not', 'no', 'never', 'none', 'neither', 'nor'}
        
    def _initialize_patterns(self) -> Dict[RelationType, List[str]]:
        """Initialize regex patterns for relationship extraction."""
        return {
            RelationType.WORKS_FOR: [
                r'(\w+(?:\s+\w+)*)\s+(?:works? for|employed by|employee of|staff at)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*),?\s+(?:CEO|president|director|manager|employee|staff|worker)\s+(?:of|at)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:joined|left|quit)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:represents|serves)\s+(\w+(?:\s+\w+)*)',
            ],
            
            RelationType.LOCATED_IN: [
                r'(\w+(?:\s+\w+)*)\s+(?:is located in|located in|situated in|based in|in)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*),\s+(\w+(?:\s+\w+)*)',  # City, State pattern
                r'(\w+(?:\s+\w+)*)\s+(?:headquarters|office|branch)\s+(?:in|at)\s+(\w+(?:\s+\w+)*)',
                r'(?:in|at)\s+(\w+(?:\s+\w+)*),?\s+(\w+(?:\s+\w+)*)',
            ],
            
            RelationType.CREATED_BY: [
                r'(\w+(?:\s+\w+)*)\s+(?:created by|founded by|established by|invented by|developed by)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:created|founded|established|invented|developed)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:is the creator|is the founder|is the inventor)\s+of\s+(\w+(?:\s+\w+)*)',
            ],
            
            RelationType.PART_OF: [
                r'(\w+(?:\s+\w+)*)\s+(?:is part of|belongs to|member of|component of|division of)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:subsidiary|branch|department|unit)\s+of\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:within|inside|part of)\s+(\w+(?:\s+\w+)*)',
            ],
            
            RelationType.OCCURRED_AT: [
                r'(\w+(?:\s+\w+)*)\s+(?:occurred|happened|took place|held)\s+(?:at|in|on)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:at|in|on)\s+(\w+(?:\s+\w+)*)',  # Event at Location
                r'(?:during|at)\s+(\w+(?:\s+\w+)*),?\s+(\w+(?:\s+\w+)*)',
            ],
            
            RelationType.CAUSED_BY: [
                r'(\w+(?:\s+\w+)*)\s+(?:caused by|due to|because of|resulted from)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:leads to|causes|results in|triggers)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:effect|consequence|result)\s+of\s+(\w+(?:\s+\w+)*)',
            ],
            
            RelationType.RELATED_TO: [
                r'(\w+(?:\s+\w+)*)\s+(?:related to|connected to|associated with|linked to)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:and|with|alongside)\s+(\w+(?:\s+\w+)*)',
                r'(?:both|either)\s+(\w+(?:\s+\w+)*)\s+(?:and|or)\s+(\w+(?:\s+\w+)*)',
            ],
            
            RelationType.SUPPORTS: [
                r'(\w+(?:\s+\w+)*)\s+(?:supports|endorses|backs|advocates for)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:agrees with|confirms|validates)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:evidence|proof|support)\s+(?:for|of)\s+(\w+(?:\s+\w+)*)',
            ],
            
            RelationType.CONTRADICTS: [
                r'(\w+(?:\s+\w+)*)\s+(?:contradicts|opposes|disagrees with|conflicts with)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:but|however|nevertheless)\s+(\w+(?:\s+\w+)*)',
                r'(?:unlike|contrary to)\s+(\w+(?:\s+\w+)*),?\s+(\w+(?:\s+\w+)*)',
            ],
            
            RelationType.TEMPORAL_BEFORE: [
                r'(\w+(?:\s+\w+)*)\s+(?:before|prior to|earlier than)\s+(\w+(?:\s+\w+)*)',
                r'(?:after|following)\s+(\w+(?:\s+\w+)*),?\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:then|subsequently|later)\s+(\w+(?:\s+\w+)*)',
            ],
            
            RelationType.TEMPORAL_AFTER: [
                r'(\w+(?:\s+\w+)*)\s+(?:after|following|later than)\s+(\w+(?:\s+\w+)*)',
                r'(?:before|prior to)\s+(\w+(?:\s+\w+)*),?\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:previously|earlier|before)\s+(\w+(?:\s+\w+)*)',
            ],
            
            RelationType.SIMILAR_TO: [
                r'(\w+(?:\s+\w+)*)\s+(?:similar to|like|resembles|comparable to)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:as|just like|same as)\s+(\w+(?:\s+\w+)*)',
                r'(?:both|either)\s+(\w+(?:\s+\w+)*)\s+(?:and|or)\s+(\w+(?:\s+\w+)*)\s+(?:are|have)',
            ],
            
            RelationType.DIFFERENT_FROM: [
                r'(\w+(?:\s+\w+)*)\s+(?:different from|unlike|distinct from|not like)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:vs|versus|compared to)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:rather than|instead of|not)\s+(\w+(?:\s+\w+)*)',
            ],
        }
    
    def extract_relationships(self, entities: List[Union[Entity, EntityCandidate]], 
                            text: str) -> List[RelationshipCandidate]:
        """Extract relationships between entities using patterns.
        
        Args:
            entities: List of entities to find relationships between
            text: Text to search for relationship patterns
            
        Returns:
            List of relationship candidates
        """
        candidates = []
        
        # Create entity lookup for text matching
        entity_lookup = self._create_entity_lookup(entities)
        
        # Extract relationships using patterns
        for relation_type, patterns in self.patterns.items():
            for pattern_str in patterns:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                matches = pattern.finditer(text)
                
                for match in matches:
                    if len(match.groups()) >= 2:
                        entity1_text = match.group(1).strip()
                        entity2_text = match.group(2).strip()
                        
                        # Find corresponding entities
                        entity1 = self._find_best_entity_match(entity1_text, entity_lookup)
                        entity2 = self._find_best_entity_match(entity2_text, entity_lookup)
                        
                        if entity1 and entity2 and entity1 != entity2:
                            # Check for negation in context
                            context = self._get_context(text, match.start(), match.end())
                            is_negated = self._is_negated(context)
                            
                            confidence = self._calculate_pattern_confidence(
                                relation_type, pattern_str, context, is_negated
                            )
                            
                            if confidence > 0.3:
                                candidate = RelationshipCandidate(
                                    source_entity=entity1,
                                    target_entity=entity2,
                                    relation_type=relation_type,
                                    confidence=confidence,
                                    evidence=[match.group(0)],
                                    context=context,
                                    features={
                                        'extraction_method': 'pattern_based',
                                        'pattern_matched': pattern_str,
                                        'is_negated': is_negated,
                                        'match_position': (match.start(), match.end())
                                    }
                                )
                                candidates.append(candidate)
        
        return candidates
    
    def _create_entity_lookup(self, entities: List[Union[Entity, EntityCandidate]]) -> Dict[str, Union[Entity, EntityCandidate]]:
        """Create lookup dictionary for entity text matching."""
        lookup = {}
        
        for entity in entities:
            # Get entity name
            name = entity.name if hasattr(entity, 'name') else entity.text
            lookup[name.lower()] = entity
            
            # Add aliases if available
            if hasattr(entity, 'aliases'):
                for alias in entity.aliases:
                    lookup[alias.lower()] = entity
        
        return lookup
    
    def _find_best_entity_match(self, text: str, entity_lookup: Dict[str, Union[Entity, EntityCandidate]]) -> Optional[Union[Entity, EntityCandidate]]:
        """Find the best matching entity for given text."""
        text_lower = text.lower().strip()
        
        # Exact match
        if text_lower in entity_lookup:
            return entity_lookup[text_lower]
        
        # Partial matches
        best_match = None
        best_score = 0.0
        
        for entity_text, entity in entity_lookup.items():
            score = self._calculate_text_similarity(text_lower, entity_text)
            if score > best_score and score > 0.8:  # High threshold for partial matches
                best_score = score
                best_match = entity
        
        return best_match
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if text1 == text2:
            return 1.0
        
        # Check for substring relationships
        if text1 in text2 or text2 in text1:
            shorter = min(len(text1), len(text2))
            longer = max(len(text1), len(text2))
            return shorter / longer
        
        # Word overlap similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Get context around a match."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]
    
    def _is_negated(self, context: str) -> bool:
        """Check if the relationship is negated in context."""
        words = context.lower().split()
        
        # Look for negation words within 3 words of the relationship
        for i, word in enumerate(words):
            if word in self.negation_words:
                return True
        
        return False
    
    def _calculate_pattern_confidence(self, relation_type: RelationType, pattern: str, 
                                    context: str, is_negated: bool) -> float:
        """Calculate confidence for a pattern match."""
        base_confidence = 0.7
        
        # Negation penalty
        if is_negated:
            base_confidence -= 0.4
        
        # Pattern strength adjustments
        pattern_strengths = {
            RelationType.WORKS_FOR: 0.8,
            RelationType.LOCATED_IN: 0.7,
            RelationType.CREATED_BY: 0.8,
            RelationType.PART_OF: 0.7,
            RelationType.OCCURRED_AT: 0.6,
            RelationType.CAUSED_BY: 0.7,
            RelationType.RELATED_TO: 0.5,  # More generic
            RelationType.SUPPORTS: 0.6,
            RelationType.CONTRADICTS: 0.6,
            RelationType.TEMPORAL_BEFORE: 0.6,
            RelationType.TEMPORAL_AFTER: 0.6,
            RelationType.SIMILAR_TO: 0.5,
            RelationType.DIFFERENT_FROM: 0.6,
        }
        
        strength_multiplier = pattern_strengths.get(relation_type, 0.5)
        confidence = base_confidence * strength_multiplier
        
        # Context quality bonus
        if len(context.split()) > 10:  # Rich context
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))


class StatisticalRelationshipExtractor:
    """Extract relationships using statistical analysis and co-occurrence."""
    
    def __init__(self):
        """Initialize statistical relationship extractor."""
        self.min_cooccurrence = 2
        self.window_size = 50  # Words
        
    def extract_relationships(self, entities: List[Union[Entity, EntityCandidate]], 
                            text: str) -> List[RelationshipCandidate]:
        """Extract relationships using statistical analysis.
        
        Args:
            entities: List of entities to analyze
            text: Text to analyze for relationships
            
        Returns:
            List of relationship candidates
        """
        candidates = []
        
        # Analyze co-occurrence patterns
        cooccurrence_data = self._analyze_cooccurrence(entities, text)
        
        # Extract relationships based on co-occurrence
        for (entity1, entity2), data in cooccurrence_data.items():
            if data['frequency'] >= self.min_cooccurrence:
                # Determine relationship type based on context analysis
                relation_type, confidence = self._infer_relationship_type(
                    entity1, entity2, data['contexts'], text
                )
                
                if confidence > 0.4:
                    candidate = RelationshipCandidate(
                        source_entity=entity1,
                        target_entity=entity2,
                        relation_type=relation_type,
                        confidence=confidence,
                        evidence=data['contexts'],
                        features={
                            'extraction_method': 'statistical',
                            'cooccurrence_frequency': data['frequency'],
                            'average_distance': data['avg_distance'],
                            'contexts_analyzed': len(data['contexts'])
                        }
                    )
                    candidates.append(candidate)
        
        return candidates
    
    def _analyze_cooccurrence(self, entities: List[Union[Entity, EntityCandidate]], 
                            text: str) -> Dict[Tuple, Dict[str, Any]]:
        """Analyze co-occurrence patterns between entities."""
        words = text.split()
        entity_positions = defaultdict(list)
        
        # Find all entity positions in text
        for entity in entities:
            entity_name = entity.name if hasattr(entity, 'name') else entity.text
            entity_words = entity_name.lower().split()
            
            # Find all occurrences of entity in text
            for i in range(len(words) - len(entity_words) + 1):
                if all(words[i + j].lower() == entity_words[j] for j in range(len(entity_words))):
                    entity_positions[entity].append((i, i + len(entity_words) - 1))
        
        # Analyze co-occurrence between entity pairs
        cooccurrence_data = {}
        
        for entity1, entity2 in combinations(entities, 2):
            if entity1 in entity_positions and entity2 in entity_positions:
                distances = []
                contexts = []
                
                for pos1_start, pos1_end in entity_positions[entity1]:
                    for pos2_start, pos2_end in entity_positions[entity2]:
                        # Calculate distance between entities
                        distance = min(
                            abs(pos1_start - pos2_end),
                            abs(pos2_start - pos1_end)
                        )
                        
                        if distance <= self.window_size:
                            distances.append(distance)
                            
                            # Extract context
                            context_start = max(0, min(pos1_start, pos2_start) - 10)
                            context_end = min(len(words), max(pos1_end, pos2_end) + 10)
                            context = ' '.join(words[context_start:context_end])
                            contexts.append(context)
                
                if distances:
                    cooccurrence_data[(entity1, entity2)] = {
                        'frequency': len(distances),
                        'avg_distance': sum(distances) / len(distances),
                        'min_distance': min(distances),
                        'contexts': contexts
                    }
        
        return cooccurrence_data
    
    def _infer_relationship_type(self, entity1: Union[Entity, EntityCandidate], 
                               entity2: Union[Entity, EntityCandidate], 
                               contexts: List[str], text: str) -> Tuple[RelationType, float]:
        """Infer relationship type from context analysis."""
        # Analyze contexts for relationship indicators
        relationship_scores = defaultdict(float)
        
        # Define relationship indicators
        indicators = {
            RelationType.WORKS_FOR: ['works', 'employee', 'staff', 'CEO', 'president', 'manager'],
            RelationType.LOCATED_IN: ['located', 'in', 'at', 'based', 'headquarters'],
            RelationType.CREATED_BY: ['created', 'founded', 'established', 'invented', 'developed'],
            RelationType.PART_OF: ['part', 'member', 'division', 'subsidiary', 'component'],
            RelationType.RELATED_TO: ['related', 'connected', 'associated', 'linked'],
            RelationType.SUPPORTS: ['supports', 'endorses', 'backs', 'agrees'],
            RelationType.CONTRADICTS: ['contradicts', 'opposes', 'disagrees', 'conflicts']
        }
        
        # Score each relationship type based on context
        for context in contexts:
            context_lower = context.lower()
            
            for relation_type, keywords in indicators.items():
                score = sum(1 for keyword in keywords if keyword in context_lower)
                relationship_scores[relation_type] += score / len(keywords)
        
        # Get entity types for type-based relationship inference
        entity1_type = entity1.entity_type if hasattr(entity1, 'entity_type') else EntityType.UNKNOWN
        entity2_type = entity2.entity_type if hasattr(entity2, 'entity_type') else EntityType.UNKNOWN
        
        # Type-based relationship bonuses
        type_bonuses = self._get_type_based_bonuses(entity1_type, entity2_type)
        for relation_type, bonus in type_bonuses.items():
            relationship_scores[relation_type] += bonus
        
        # Select best relationship type
        if relationship_scores:
            best_relation = max(relationship_scores, key=relationship_scores.get)
            confidence = min(1.0, relationship_scores[best_relation] / len(contexts))
            return best_relation, confidence
        else:
            return RelationType.RELATED_TO, 0.3  # Default fallback
    
    def _get_type_based_bonuses(self, type1: EntityType, type2: EntityType) -> Dict[RelationType, float]:
        """Get relationship bonuses based on entity types."""
        bonuses = {}
        
        # Person-Organization relationships
        if type1 == EntityType.PERSON and type2 == EntityType.ORGANIZATION:
            bonuses[RelationType.WORKS_FOR] = 0.3
            bonuses[RelationType.MEMBER_OF] = 0.2
        
        # Organization-Location relationships
        if type1 == EntityType.ORGANIZATION and type2 == EntityType.LOCATION:
            bonuses[RelationType.LOCATED_IN] = 0.3
        
        # Event-Location relationships
        if type1 == EntityType.EVENT and type2 == EntityType.LOCATION:
            bonuses[RelationType.OCCURRED_AT] = 0.3
        
        # Person-Person relationships
        if type1 == EntityType.PERSON and type2 == EntityType.PERSON:
            bonuses[RelationType.RELATED_TO] = 0.2
        
        return bonuses


class AdvancedRelationshipMapper:
    """Advanced relationship mapping system combining multiple extraction methods."""
    
    def __init__(self):
        """Initialize advanced relationship mapper."""
        self.pattern_extractor = PatternBasedRelationshipExtractor()
        self.statistical_extractor = StatisticalRelationshipExtractor()
        
    def extract_relationships(self, entities: List[Union[Entity, EntityCandidate]], 
                            text: str, method: str = "hybrid") -> List[RelationshipCandidate]:
        """Extract relationships using specified method.
        
        Args:
            entities: List of entities to find relationships between
            text: Text to analyze
            method: Extraction method ("pattern", "statistical", "hybrid")
            
        Returns:
            List of relationship candidates
        """
        all_candidates = []
        
        if method in ["pattern", "hybrid"]:
            pattern_candidates = self.pattern_extractor.extract_relationships(entities, text)
            all_candidates.extend(pattern_candidates)
        
        if method in ["statistical", "hybrid"]:
            statistical_candidates = self.statistical_extractor.extract_relationships(entities, text)
            all_candidates.extend(statistical_candidates)
        
        # Merge and deduplicate candidates
        merged_candidates = self._merge_and_deduplicate(all_candidates)
        
        # Rank candidates by confidence
        merged_candidates.sort(key=lambda x: x.confidence, reverse=True)
        
        return merged_candidates
    
    def _merge_and_deduplicate(self, candidates: List[RelationshipCandidate]) -> List[RelationshipCandidate]:
        """Merge and deduplicate relationship candidates."""
        # Group candidates by entity pair and relationship type
        groups = defaultdict(list)
        
        for candidate in candidates:
            # Create a key for grouping
            source_name = (candidate.source_entity.name 
                          if hasattr(candidate.source_entity, 'name') 
                          else candidate.source_entity.text)
            target_name = (candidate.target_entity.name 
                          if hasattr(candidate.target_entity, 'name') 
                          else candidate.target_entity.text)
            
            key = (source_name.lower(), target_name.lower(), candidate.relation_type)
            groups[key].append(candidate)
        
        # Merge groups
        merged_candidates = []
        for group in groups.values():
            if len(group) == 1:
                merged_candidates.append(group[0])
            else:
                merged_candidate = self._merge_candidate_group(group)
                merged_candidates.append(merged_candidate)
        
        return merged_candidates
    
    def _merge_candidate_group(self, group: List[RelationshipCandidate]) -> RelationshipCandidate:
        """Merge a group of similar relationship candidates."""
        # Use the candidate with highest confidence as base
        best_candidate = max(group, key=lambda x: x.confidence)
        
        # Combine evidence and features
        combined_evidence = []
        combined_features = {}
        
        for candidate in group:
            combined_evidence.extend(candidate.evidence)
            combined_features.update(candidate.features)
        
        # Average confidence weighted by extraction method reliability
        method_weights = {
            'pattern_based': 0.7,
            'statistical': 0.5
        }
        
        weighted_confidences = []
        for candidate in group:
            method = candidate.features.get('extraction_method', 'unknown')
            weight = method_weights.get(method, 0.5)
            weighted_confidences.append(candidate.confidence * weight)
        
        avg_confidence = sum(weighted_confidences) / len(weighted_confidences)
        
        # Create merged candidate
        return RelationshipCandidate(
            source_entity=best_candidate.source_entity,
            target_entity=best_candidate.target_entity,
            relation_type=best_candidate.relation_type,
            confidence=avg_confidence,
            evidence=list(set(combined_evidence)),  # Remove duplicates
            context=best_candidate.context,
            features=combined_features
        )
    
    def validate_relationships(self, candidates: List[RelationshipCandidate]) -> List[RelationshipCandidate]:
        """Validate and filter relationship candidates."""
        validated_candidates = []
        
        for candidate in candidates:
            if self._is_valid_relationship(candidate):
                validated_candidates.append(candidate)
        
        return validated_candidates
    
    def _is_valid_relationship(self, candidate: RelationshipCandidate) -> bool:
        """Check if a relationship candidate is valid."""
        # Minimum confidence threshold
        if candidate.confidence < 0.3:
            return False
        
        # Check for self-relationships
        source_name = (candidate.source_entity.name 
                      if hasattr(candidate.source_entity, 'name') 
                      else candidate.source_entity.text)
        target_name = (candidate.target_entity.name 
                      if hasattr(candidate.target_entity, 'name') 
                      else candidate.target_entity.text)
        
        if source_name.lower() == target_name.lower():
            return False
        
        # Type compatibility checks
        source_type = (candidate.source_entity.entity_type 
                      if hasattr(candidate.source_entity, 'entity_type') 
                      else EntityType.UNKNOWN)
        target_type = (candidate.target_entity.entity_type 
                      if hasattr(candidate.target_entity, 'entity_type') 
                      else EntityType.UNKNOWN)
        
        # Check type compatibility for specific relationships
        incompatible_combinations = {
            RelationType.WORKS_FOR: [
                (EntityType.LOCATION, EntityType.PERSON),
                (EntityType.DATE, EntityType.ORGANIZATION)
            ],
            RelationType.LOCATED_IN: [
                (EntityType.PERSON, EntityType.PERSON),
                (EntityType.CONCEPT, EntityType.LOCATION)
            ]
        }
        
        if candidate.relation_type in incompatible_combinations:
            for incompatible_source, incompatible_target in incompatible_combinations[candidate.relation_type]:
                if source_type == incompatible_source and target_type == incompatible_target:
                    return False
        
        return True
    
    def suggest_missing_relationships(self, entities: List[Union[Entity, EntityCandidate]], 
                                    existing_relationships: List[RelationshipCandidate]) -> List[RelationshipCandidate]:
        """Suggest potentially missing relationships based on entity analysis."""
        suggestions = []
        
        # Create sets of existing relationships for quick lookup
        existing_pairs = set()
        for rel in existing_relationships:
            source_name = (rel.source_entity.name 
                          if hasattr(rel.source_entity, 'name') 
                          else rel.source_entity.text)
            target_name = (rel.target_entity.name 
                          if hasattr(rel.target_entity, 'name') 
                          else rel.target_entity.text)
            existing_pairs.add((source_name.lower(), target_name.lower()))
        
        # Suggest relationships based on entity types
        for entity1, entity2 in combinations(entities, 2):
            entity1_name = (entity1.name if hasattr(entity1, 'name') else entity1.text).lower()
            entity2_name = (entity2.name if hasattr(entity2, 'name') else entity2.text).lower()
            
            # Skip if relationship already exists
            if (entity1_name, entity2_name) in existing_pairs or (entity2_name, entity1_name) in existing_pairs:
                continue
            
            # Suggest based on type combinations
            type1 = entity1.entity_type if hasattr(entity1, 'entity_type') else EntityType.UNKNOWN
            type2 = entity2.entity_type if hasattr(entity2, 'entity_type') else EntityType.UNKNOWN
            
            suggested_relations = self._get_suggested_relations_by_type(type1, type2)
            
            for relation_type, confidence in suggested_relations:
                suggestion = RelationshipCandidate(
                    source_entity=entity1,
                    target_entity=entity2,
                    relation_type=relation_type,
                    confidence=confidence,
                    features={
                        'extraction_method': 'suggestion',
                        'suggestion_basis': 'entity_types'
                    }
                )
                suggestions.append(suggestion)
        
        return suggestions
    
    def _get_suggested_relations_by_type(self, type1: EntityType, type2: EntityType) -> List[Tuple[RelationType, float]]:
        """Get suggested relationships based on entity types."""
        suggestions = []
        
        # Person-Organization
        if type1 == EntityType.PERSON and type2 == EntityType.ORGANIZATION:
            suggestions.append((RelationType.WORKS_FOR, 0.4))
            suggestions.append((RelationType.MEMBER_OF, 0.3))
        
        # Organization-Location
        if type1 == EntityType.ORGANIZATION and type2 == EntityType.LOCATION:
            suggestions.append((RelationType.LOCATED_IN, 0.4))
        
        # Event-Location
        if type1 == EntityType.EVENT and type2 == EntityType.LOCATION:
            suggestions.append((RelationType.OCCURRED_AT, 0.4))
        
        # Generic relationships for any type combination
        suggestions.append((RelationType.RELATED_TO, 0.2))
        
        return suggestions