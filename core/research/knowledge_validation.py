"""Comprehensive knowledge validation system for Beyond RAG architecture.

This module provides advanced validation capabilities for knowledge stored in
the GraphRAG system, including consistency checking, quality assessment,
completeness validation, and automated knowledge curation.
"""

import re
import time
import json
import sqlite3
import hashlib
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import logging
import statistics
from datetime import datetime, timedelta

from .graphrag import Entity, EntityType, Relationship, RelationType, Fact, KnowledgeGraphStore
from .entity_extraction import EntityCandidate, HybridEntityExtractor
from .fact_checker import FactVerifier, VerificationResult, Claim
from .memory_aug import MemoryAugmentedSystem, MemoryType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Levels of knowledge validation."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    COMPREHENSIVE = "comprehensive"
    EXHAUSTIVE = "exhaustive"


class ValidationStatus(Enum):
    """Status of validation results."""
    VALID = "valid"
    INVALID = "invalid"
    UNCERTAIN = "uncertain"
    REQUIRES_REVIEW = "requires_review"
    CONFLICTING = "conflicting"


class ConsistencyLevel(Enum):
    """Levels of knowledge consistency."""
    CONSISTENT = "consistent"
    MINOR_INCONSISTENCY = "minor_inconsistency"
    MAJOR_INCONSISTENCY = "major_inconsistency"
    CONTRADICTORY = "contradictory"


@dataclass
class ValidationIssue:
    """Represents a validation issue found in knowledge."""
    issue_id: str
    issue_type: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    affected_entities: List[str] = field(default_factory=list)
    affected_relationships: List[str] = field(default_factory=list)
    affected_facts: List[str] = field(default_factory=list)
    suggested_fix: str = ""
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    report_id: str
    validation_level: ValidationLevel
    overall_status: ValidationStatus
    overall_score: float
    issues: List[ValidationIssue] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    validation_time: float = 0.0
    entities_validated: int = 0
    relationships_validated: int = 0
    facts_validated: int = 0
    timestamp: float = field(default_factory=time.time)


class EntityValidator:
    """Validates entities in the knowledge graph."""
    
    def __init__(self, knowledge_store: KnowledgeGraphStore):
        """Initialize entity validator.
        
        Args:
            knowledge_store: Knowledge graph store to validate
        """
        self.knowledge_store = knowledge_store
        self.entity_extractor = HybridEntityExtractor()
        
    def validate_entity(self, entity: Entity) -> List[ValidationIssue]:
        """Validate a single entity.
        
        Args:
            entity: Entity to validate
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Check entity name validity
        name_issues = self._validate_entity_name(entity)
        issues.extend(name_issues)
        
        # Check entity type consistency
        type_issues = self._validate_entity_type(entity)
        issues.extend(type_issues)
        
        # Check entity attributes
        attribute_issues = self._validate_entity_attributes(entity)
        issues.extend(attribute_issues)
        
        # Check entity completeness
        completeness_issues = self._validate_entity_completeness(entity)
        issues.extend(completeness_issues)
        
        # Check for duplicates
        duplicate_issues = self._check_entity_duplicates(entity)
        issues.extend(duplicate_issues)
        
        return issues
    
    def _validate_entity_name(self, entity: Entity) -> List[ValidationIssue]:
        """Validate entity name."""
        issues = []
        
        # Check if name is empty or too short
        if not entity.name or len(entity.name.strip()) < 2:
            issues.append(ValidationIssue(
                issue_id=f"name_invalid_{entity.entity_id}",
                issue_type="invalid_name",
                severity="high",
                description=f"Entity name is invalid or too short: '{entity.name}'",
                affected_entities=[entity.entity_id],
                suggested_fix="Provide a meaningful name with at least 2 characters",
                confidence=0.95
            ))
        
        # Check for suspicious characters
        suspicious_chars = re.findall(r'[^\w\s\-\.\,\(\)]', entity.name)
        if suspicious_chars:
            issues.append(ValidationIssue(
                issue_id=f"name_suspicious_{entity.entity_id}",
                issue_type="suspicious_characters",
                severity="medium",
                description=f"Entity name contains suspicious characters: {set(suspicious_chars)}",
                affected_entities=[entity.entity_id],
                suggested_fix="Remove or replace suspicious characters",
                confidence=0.8
            ))
        
        # Check for extremely long names
        if len(entity.name) > 200:
            issues.append(ValidationIssue(
                issue_id=f"name_too_long_{entity.entity_id}",
                issue_type="name_too_long",
                severity="medium",
                description=f"Entity name is unusually long ({len(entity.name)} characters)",
                affected_entities=[entity.entity_id],
                suggested_fix="Consider shortening the name or using aliases",
                confidence=0.9
            ))
        
        return issues
    
    def _validate_entity_type(self, entity: Entity) -> List[ValidationIssue]:
        """Validate entity type assignment."""
        issues = []
        
        # Use entity extractor to re-analyze the entity
        extraction_result = self.entity_extractor.extract(entity.name)
        
        if extraction_result.entities:
            predicted_type = extraction_result.entities[0].entity_type
            
            # Check if predicted type matches assigned type
            if predicted_type != entity.entity_type and predicted_type != EntityType.UNKNOWN:
                confidence = extraction_result.entities[0].confidence
                
                if confidence > 0.7:
                    issues.append(ValidationIssue(
                        issue_id=f"type_mismatch_{entity.entity_id}",
                        issue_type="type_mismatch",
                        severity="medium",
                        description=f"Entity type mismatch: assigned '{entity.entity_type.value}', predicted '{predicted_type.value}'",
                        affected_entities=[entity.entity_id],
                        suggested_fix=f"Consider changing type to '{predicted_type.value}'",
                        confidence=confidence
                    ))
        
        return issues
    
    def _validate_entity_attributes(self, entity: Entity) -> List[ValidationIssue]:
        """Validate entity attributes."""
        issues = []
        
        # Check confidence score validity
        if not (0.0 <= entity.confidence <= 1.0):
            issues.append(ValidationIssue(
                issue_id=f"confidence_invalid_{entity.entity_id}",
                issue_type="invalid_confidence",
                severity="high",
                description=f"Entity confidence score is out of range: {entity.confidence}",
                affected_entities=[entity.entity_id],
                suggested_fix="Set confidence to a value between 0.0 and 1.0",
                confidence=1.0
            ))
        
        # Check if confidence is suspiciously low
        if entity.confidence < 0.1:
            issues.append(ValidationIssue(
                issue_id=f"confidence_low_{entity.entity_id}",
                issue_type="low_confidence",
                severity="medium",
                description=f"Entity has very low confidence score: {entity.confidence}",
                affected_entities=[entity.entity_id],
                suggested_fix="Review entity or remove if confidence is too low",
                confidence=0.9
            ))
        
        # Validate aliases
        if entity.aliases:
            for alias in entity.aliases:
                if not alias or len(alias.strip()) < 2:
                    issues.append(ValidationIssue(
                        issue_id=f"alias_invalid_{entity.entity_id}",
                        issue_type="invalid_alias",
                        severity="low",
                        description=f"Entity has invalid alias: '{alias}'",
                        affected_entities=[entity.entity_id],
                        suggested_fix="Remove invalid aliases",
                        confidence=0.95
                    ))
        
        # Check for duplicate aliases
        if entity.aliases and len(entity.aliases) != len(set(entity.aliases)):
            issues.append(ValidationIssue(
                issue_id=f"duplicate_aliases_{entity.entity_id}",
                issue_type="duplicate_aliases",
                severity="low",
                description="Entity has duplicate aliases",
                affected_entities=[entity.entity_id],
                suggested_fix="Remove duplicate aliases",
                confidence=1.0
            ))
        
        return issues
    
    def _validate_entity_completeness(self, entity: Entity) -> List[ValidationIssue]:
        """Validate entity completeness."""
        issues = []
        
        # Check if entity has any relationships
        relationships = self.knowledge_store.get_relationships(entity.entity_id)
        if not relationships:
            issues.append(ValidationIssue(
                issue_id=f"no_relationships_{entity.entity_id}",
                issue_type="isolated_entity",
                severity="medium",
                description="Entity has no relationships (isolated)",
                affected_entities=[entity.entity_id],
                suggested_fix="Add relationships or remove if entity is not relevant",
                confidence=0.8
            ))
        
        # Check if entity appears in any facts
        facts = self._get_facts_mentioning_entity(entity.entity_id)
        if not facts:
            issues.append(ValidationIssue(
                issue_id=f"no_facts_{entity.entity_id}",
                issue_type="no_supporting_facts",
                severity="low",
                description="Entity is not mentioned in any facts",
                affected_entities=[entity.entity_id],
                suggested_fix="Add facts supporting this entity's existence",
                confidence=0.7
            ))
        
        return issues
    
    def _check_entity_duplicates(self, entity: Entity) -> List[ValidationIssue]:
        """Check for potential entity duplicates."""
        issues = []
        
        # Get all entities to check for duplicates
        all_entities = self.knowledge_store.get_all_entities()
        
        potential_duplicates = []
        for other_entity in all_entities:
            if other_entity.entity_id != entity.entity_id:
                similarity = self._calculate_entity_similarity(entity, other_entity)
                if similarity > 0.8:
                    potential_duplicates.append((other_entity, similarity))
        
        if potential_duplicates:
            duplicate_ids = [e.entity_id for e, _ in potential_duplicates]
            issues.append(ValidationIssue(
                issue_id=f"potential_duplicate_{entity.entity_id}",
                issue_type="potential_duplicate",
                severity="medium",
                description=f"Entity may be duplicate of: {[e.name for e, _ in potential_duplicates]}",
                affected_entities=[entity.entity_id] + duplicate_ids,
                suggested_fix="Review and merge duplicate entities if necessary",
                confidence=max(sim for _, sim in potential_duplicates)
            ))
        
        return issues
    
    def _calculate_entity_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """Calculate similarity between two entities."""
        # Name similarity
        name_sim = self._text_similarity(entity1.name.lower(), entity2.name.lower())
        
        # Type similarity
        type_sim = 1.0 if entity1.entity_type == entity2.entity_type else 0.0
        
        # Alias similarity
        alias_sim = 0.0
        if entity1.aliases and entity2.aliases:
            all_aliases_1 = set(alias.lower() for alias in entity1.aliases)
            all_aliases_2 = set(alias.lower() for alias in entity2.aliases)
            if all_aliases_1 & all_aliases_2:  # Common aliases
                alias_sim = 1.0
        
        # Check if one entity's name is in the other's aliases
        if entity1.aliases and entity2.name.lower() in [alias.lower() for alias in entity1.aliases]:
            alias_sim = 1.0
        if entity2.aliases and entity1.name.lower() in [alias.lower() for alias in entity2.aliases]:
            alias_sim = 1.0
        
        # Combined similarity
        return 0.6 * name_sim + 0.3 * type_sim + 0.1 * alias_sim
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity."""
        if text1 == text2:
            return 1.0
        
        # Jaccard similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _get_facts_mentioning_entity(self, entity_id: str) -> List[Fact]:
        """Get facts that mention the entity."""
        # This would need to be implemented based on the fact storage structure
        # For now, return empty list as placeholder
        return []


class RelationshipValidator:
    """Validates relationships in the knowledge graph."""
    
    def __init__(self, knowledge_store: KnowledgeGraphStore):
        """Initialize relationship validator.
        
        Args:
            knowledge_store: Knowledge graph store to validate
        """
        self.knowledge_store = knowledge_store
        
    def validate_relationship(self, relationship: Relationship) -> List[ValidationIssue]:
        """Validate a single relationship.
        
        Args:
            relationship: Relationship to validate
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Check entity existence
        existence_issues = self._validate_entity_existence(relationship)
        issues.extend(existence_issues)
        
        # Check relationship type validity
        type_issues = self._validate_relationship_type(relationship)
        issues.extend(type_issues)
        
        # Check relationship consistency
        consistency_issues = self._validate_relationship_consistency(relationship)
        issues.extend(consistency_issues)
        
        # Check for circular relationships
        circular_issues = self._check_circular_relationships(relationship)
        issues.extend(circular_issues)
        
        # Check for duplicate relationships
        duplicate_issues = self._check_duplicate_relationships(relationship)
        issues.extend(duplicate_issues)
        
        return issues
    
    def _validate_entity_existence(self, relationship: Relationship) -> List[ValidationIssue]:
        """Validate that relationship entities exist."""
        issues = []
        
        source_entity = self.knowledge_store.get_entity(relationship.source_entity_id)
        target_entity = self.knowledge_store.get_entity(relationship.target_entity_id)
        
        if not source_entity:
            issues.append(ValidationIssue(
                issue_id=f"missing_source_{relationship.relationship_id}",
                issue_type="missing_entity",
                severity="critical",
                description=f"Source entity {relationship.source_entity_id} does not exist",
                affected_relationships=[relationship.relationship_id],
                suggested_fix="Remove relationship or create missing entity",
                confidence=1.0
            ))
        
        if not target_entity:
            issues.append(ValidationIssue(
                issue_id=f"missing_target_{relationship.relationship_id}",
                issue_type="missing_entity",
                severity="critical",
                description=f"Target entity {relationship.target_entity_id} does not exist",
                affected_relationships=[relationship.relationship_id],
                suggested_fix="Remove relationship or create missing entity",
                confidence=1.0
            ))
        
        return issues
    
    def _validate_relationship_type(self, relationship: Relationship) -> List[ValidationIssue]:
        """Validate relationship type appropriateness."""
        issues = []
        
        source_entity = self.knowledge_store.get_entity(relationship.source_entity_id)
        target_entity = self.knowledge_store.get_entity(relationship.target_entity_id)
        
        if source_entity and target_entity:
            # Check type compatibility
            incompatible = self._check_type_compatibility(
                source_entity.entity_type,
                target_entity.entity_type,
                relationship.relation_type
            )
            
            if incompatible:
                issues.append(ValidationIssue(
                    issue_id=f"incompatible_types_{relationship.relationship_id}",
                    issue_type="incompatible_entity_types",
                    severity="high",
                    description=f"Relationship type '{relationship.relation_type.value}' incompatible with entity types",
                    affected_relationships=[relationship.relationship_id],
                    affected_entities=[source_entity.entity_id, target_entity.entity_id],
                    suggested_fix="Change relationship type or entity types",
                    confidence=0.9
                ))
        
        return issues
    
    def _validate_relationship_consistency(self, relationship: Relationship) -> List[ValidationIssue]:
        """Validate relationship consistency."""
        issues = []
        
        # Check confidence score
        if not (0.0 <= relationship.confidence <= 1.0):
            issues.append(ValidationIssue(
                issue_id=f"confidence_invalid_{relationship.relationship_id}",
                issue_type="invalid_confidence",
                severity="high",
                description=f"Relationship confidence is out of range: {relationship.confidence}",
                affected_relationships=[relationship.relationship_id],
                suggested_fix="Set confidence to a value between 0.0 and 1.0",
                confidence=1.0
            ))
        
        # Check for suspiciously low confidence
        if relationship.confidence < 0.1:
            issues.append(ValidationIssue(
                issue_id=f"confidence_low_{relationship.relationship_id}",
                issue_type="low_confidence",
                severity="medium",
                description=f"Relationship has very low confidence: {relationship.confidence}",
                affected_relationships=[relationship.relationship_id],
                suggested_fix="Review relationship or remove if confidence is too low",
                confidence=0.9
            ))
        
        return issues
    
    def _check_circular_relationships(self, relationship: Relationship) -> List[ValidationIssue]:
        """Check for circular relationships."""
        issues = []
        
        # Self-reference check
        if relationship.source_entity_id == relationship.target_entity_id:
            issues.append(ValidationIssue(
                issue_id=f"self_reference_{relationship.relationship_id}",
                issue_type="self_reference",
                severity="medium",
                description="Relationship creates self-reference",
                affected_relationships=[relationship.relationship_id],
                suggested_fix="Review if self-reference is intentional",
                confidence=1.0
            ))
        
        return issues
    
    def _check_duplicate_relationships(self, relationship: Relationship) -> List[ValidationIssue]:
        """Check for duplicate relationships."""
        issues = []
        
        # Get all relationships for the source entity
        source_relationships = self.knowledge_store.get_relationships(relationship.source_entity_id)
        
        duplicates = []
        for other_rel in source_relationships:
            if (other_rel.relationship_id != relationship.relationship_id and
                other_rel.target_entity_id == relationship.target_entity_id and
                other_rel.relation_type == relationship.relation_type):
                duplicates.append(other_rel.relationship_id)
        
        if duplicates:
            issues.append(ValidationIssue(
                issue_id=f"duplicate_relationship_{relationship.relationship_id}",
                issue_type="duplicate_relationship",
                severity="medium",
                description=f"Duplicate relationship found: {duplicates}",
                affected_relationships=[relationship.relationship_id] + duplicates,
                suggested_fix="Remove duplicate relationships",
                confidence=1.0
            ))
        
        return issues
    
    def _check_type_compatibility(self, source_type: EntityType, target_type: EntityType,
                                 relation_type: RelationType) -> bool:
        """Check if entity types are compatible with relationship type."""
        # Define incompatible combinations
        incompatible_combinations = {
            RelationType.WORKS_FOR: [
                (EntityType.LOCATION, EntityType.PERSON),
                (EntityType.DATE, EntityType.ORGANIZATION),
                (EntityType.CONCEPT, EntityType.PERSON)
            ],
            RelationType.LOCATED_IN: [
                (EntityType.PERSON, EntityType.PERSON),
                (EntityType.CONCEPT, EntityType.CONCEPT),
                (EntityType.DATE, EntityType.LOCATION)
            ],
            RelationType.OCCURRED_AT: [
                (EntityType.PERSON, EntityType.LOCATION),
                (EntityType.ORGANIZATION, EntityType.LOCATION)
            ]
        }
        
        if relation_type in incompatible_combinations:
            return (source_type, target_type) in incompatible_combinations[relation_type]
        
        return False


class FactValidator:
    """Validates facts in the knowledge graph."""
    
    def __init__(self, knowledge_store: KnowledgeGraphStore, fact_verifier: FactVerifier):
        """Initialize fact validator.
        
        Args:
            knowledge_store: Knowledge graph store
            fact_verifier: Fact verification engine
        """
        self.knowledge_store = knowledge_store
        self.fact_verifier = fact_verifier
        
    def validate_fact(self, fact: Fact) -> List[ValidationIssue]:
        """Validate a single fact.
        
        Args:
            fact: Fact to validate
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Basic validation
        basic_issues = self._validate_fact_basic(fact)
        issues.extend(basic_issues)
        
        # Entity reference validation
        entity_issues = self._validate_fact_entities(fact)
        issues.extend(entity_issues)
        
        # Fact verification
        verification_issues = self._validate_fact_verification(fact)
        issues.extend(verification_issues)
        
        # Source validation
        source_issues = self._validate_fact_sources(fact)
        issues.extend(source_issues)
        
        return issues
    
    def _validate_fact_basic(self, fact: Fact) -> List[ValidationIssue]:
        """Basic fact validation."""
        issues = []
        
        # Check if claim is empty
        if not fact.claim or len(fact.claim.strip()) < 5:
            issues.append(ValidationIssue(
                issue_id=f"empty_claim_{fact.fact_id}",
                issue_type="empty_claim",
                severity="critical",
                description="Fact claim is empty or too short",
                affected_facts=[fact.fact_id],
                suggested_fix="Provide a meaningful claim",
                confidence=1.0
            ))
        
        # Check confidence score
        if not (0.0 <= fact.confidence <= 1.0):
            issues.append(ValidationIssue(
                issue_id=f"confidence_invalid_{fact.fact_id}",
                issue_type="invalid_confidence",
                severity="high",
                description=f"Fact confidence is out of range: {fact.confidence}",
                affected_facts=[fact.fact_id],
                suggested_fix="Set confidence to a value between 0.0 and 1.0",
                confidence=1.0
            ))
        
        return issues
    
    def _validate_fact_entities(self, fact: Fact) -> List[ValidationIssue]:
        """Validate fact entity references."""
        issues = []
        
        # Check if referenced entities exist
        for entity_id in fact.entities:
            entity = self.knowledge_store.get_entity(entity_id)
            if not entity:
                issues.append(ValidationIssue(
                    issue_id=f"missing_entity_{fact.fact_id}_{entity_id}",
                    issue_type="missing_entity_reference",
                    severity="high",
                    description=f"Fact references non-existent entity: {entity_id}",
                    affected_facts=[fact.fact_id],
                    suggested_fix="Remove entity reference or create missing entity",
                    confidence=1.0
                ))
        
        return issues
    
    def _validate_fact_verification(self, fact: Fact) -> List[ValidationIssue]:
        """Validate fact using verification engine."""
        issues = []
        
        try:
            # Create claim from fact
            claim = Claim(
                claim_id="",
                text=fact.claim,
                claim_type=None,  # Will be determined by extractor
                confidence=fact.confidence
            )
            
            # Verify the claim
            verification_report = self.fact_verifier.verify_claim(claim)
            
            # Check verification result
            if verification_report.result == VerificationResult.CONTRADICTED:
                issues.append(ValidationIssue(
                    issue_id=f"contradicted_fact_{fact.fact_id}",
                    issue_type="contradicted_fact",
                    severity="high",
                    description="Fact is contradicted by available evidence",
                    affected_facts=[fact.fact_id],
                    suggested_fix="Review fact or update with correct information",
                    confidence=verification_report.overall_confidence
                ))
            elif verification_report.result == VerificationResult.INSUFFICIENT_EVIDENCE:
                issues.append(ValidationIssue(
                    issue_id=f"insufficient_evidence_{fact.fact_id}",
                    issue_type="insufficient_evidence",
                    severity="medium",
                    description="Fact has insufficient supporting evidence",
                    affected_facts=[fact.fact_id],
                    suggested_fix="Add supporting evidence or lower confidence",
                    confidence=verification_report.overall_confidence
                ))
        
        except Exception as e:
            logger.warning(f"Could not verify fact {fact.fact_id}: {e}")
        
        return issues
    
    def _validate_fact_sources(self, fact: Fact) -> List[ValidationIssue]:
        """Validate fact sources."""
        issues = []
        
        # Check if fact has sources
        if not fact.sources:
            issues.append(ValidationIssue(
                issue_id=f"no_sources_{fact.fact_id}",
                issue_type="no_sources",
                severity="medium",
                description="Fact has no supporting sources",
                affected_facts=[fact.fact_id],
                suggested_fix="Add reliable sources for the fact",
                confidence=0.9
            ))
        
        # Check source quality
        if fact.sources:
            low_quality_sources = []
            for source in fact.sources:
                if self._is_low_quality_source(source):
                    low_quality_sources.append(source)
            
            if low_quality_sources:
                issues.append(ValidationIssue(
                    issue_id=f"low_quality_sources_{fact.fact_id}",
                    issue_type="low_quality_sources",
                    severity="medium",
                    description=f"Fact has low-quality sources: {low_quality_sources}",
                    affected_facts=[fact.fact_id],
                    suggested_fix="Replace with higher-quality sources",
                    confidence=0.8
                ))
        
        return issues
    
    def _is_low_quality_source(self, source: str) -> bool:
        """Check if a source is considered low quality."""
        source_lower = source.lower()
        
        # Low-quality indicators
        low_quality_indicators = [
            'blog', 'personal', 'opinion', 'forum', 'comment',
            'social media', 'reddit', 'twitter', 'facebook'
        ]
        
        return any(indicator in source_lower for indicator in low_quality_indicators)


class KnowledgeValidator:
    """Main knowledge validation system."""
    
    def __init__(self, knowledge_store: KnowledgeGraphStore, fact_verifier: FactVerifier = None,
                 memory_system: MemoryAugmentedSystem = None):
        """Initialize knowledge validator.
        
        Args:
            knowledge_store: Knowledge graph store to validate
            fact_verifier: Optional fact verification engine
            memory_system: Optional memory system for storing validation results
        """
        self.knowledge_store = knowledge_store
        self.fact_verifier = fact_verifier
        self.memory_system = memory_system
        
        # Initialize sub-validators
        self.entity_validator = EntityValidator(knowledge_store)
        self.relationship_validator = RelationshipValidator(knowledge_store)
        if fact_verifier:
            self.fact_validator = FactValidator(knowledge_store, fact_verifier)
        else:
            self.fact_validator = None
            
    def validate_knowledge_graph(self, validation_level: ValidationLevel = ValidationLevel.INTERMEDIATE) -> ValidationReport:
        """Validate the entire knowledge graph.
        
        Args:
            validation_level: Level of validation to perform
            
        Returns:
            Comprehensive validation report
        """
        start_time = time.time()
        
        logger.info(f"Starting knowledge graph validation at {validation_level.value} level")
        
        all_issues = []
        entities_validated = 0
        relationships_validated = 0
        facts_validated = 0
        
        # Validate entities
        if validation_level in [ValidationLevel.BASIC, ValidationLevel.INTERMEDIATE, 
                              ValidationLevel.COMPREHENSIVE, ValidationLevel.EXHAUSTIVE]:
            entities = self.knowledge_store.get_all_entities()
            for entity in entities:
                entity_issues = self.entity_validator.validate_entity(entity)
                all_issues.extend(entity_issues)
                entities_validated += 1
                
                if validation_level == ValidationLevel.BASIC and entities_validated >= 100:
                    break  # Limit for basic validation
        
        # Validate relationships
        if validation_level in [ValidationLevel.INTERMEDIATE, ValidationLevel.COMPREHENSIVE, 
                              ValidationLevel.EXHAUSTIVE]:
            relationships = self.knowledge_store.get_all_relationships()
            for relationship in relationships:
                rel_issues = self.relationship_validator.validate_relationship(relationship)
                all_issues.extend(rel_issues)
                relationships_validated += 1
                
                if validation_level == ValidationLevel.INTERMEDIATE and relationships_validated >= 500:
                    break  # Limit for intermediate validation
        
        # Validate facts
        if validation_level in [ValidationLevel.COMPREHENSIVE, ValidationLevel.EXHAUSTIVE] and self.fact_validator:
            facts = self.knowledge_store.get_all_facts()
            for fact in facts:
                fact_issues = self.fact_validator.validate_fact(fact)
                all_issues.extend(fact_issues)
                facts_validated += 1
                
                if validation_level == ValidationLevel.COMPREHENSIVE and facts_validated >= 1000:
                    break  # Limit for comprehensive validation
        
        # Cross-validation checks for exhaustive level
        if validation_level == ValidationLevel.EXHAUSTIVE:
            cross_issues = self._perform_cross_validation()
            all_issues.extend(cross_issues)
        
        # Calculate overall score and status
        overall_score, overall_status = self._calculate_overall_assessment(all_issues)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_issues)
        
        # Calculate statistics
        statistics = self._calculate_validation_statistics(all_issues, entities_validated, 
                                                         relationships_validated, facts_validated)
        
        validation_time = time.time() - start_time
        
        # Create validation report
        report = ValidationReport(
            report_id=hashlib.sha256(f"{time.time()}_{validation_level.value}".encode()).hexdigest()[:16],
            validation_level=validation_level,
            overall_status=overall_status,
            overall_score=overall_score,
            issues=all_issues,
            statistics=statistics,
            recommendations=recommendations,
            validation_time=validation_time,
            entities_validated=entities_validated,
            relationships_validated=relationships_validated,
            facts_validated=facts_validated
        )
        
        # Store validation report in memory if available
        if self.memory_system:
            self.memory_system.store_memory(
                content=report,
                memory_type=MemoryType.PROCEDURAL,
                importance_score=0.8,
                metadata={
                    'type': 'validation_report',
                    'validation_level': validation_level.value,
                    'overall_status': overall_status.value,
                    'issues_count': len(all_issues)
                }
            )
        
        logger.info(f"Validation completed in {validation_time:.2f}s. Status: {overall_status.value}, Score: {overall_score:.2f}")
        
        return report
    
    def _perform_cross_validation(self) -> List[ValidationIssue]:
        """Perform cross-validation checks between different knowledge components."""
        issues = []
        
        # Check for orphaned entities (entities not referenced by any relationships or facts)
        all_entities = {e.entity_id: e for e in self.knowledge_store.get_all_entities()}
        all_relationships = self.knowledge_store.get_all_relationships()
        all_facts = self.knowledge_store.get_all_facts()
        
        referenced_entities = set()
        
        # Collect entity references from relationships
        for rel in all_relationships:
            referenced_entities.add(rel.source_entity_id)
            referenced_entities.add(rel.target_entity_id)
        
        # Collect entity references from facts
        for fact in all_facts:
            referenced_entities.update(fact.entities)
        
        # Find orphaned entities
        orphaned_entities = set(all_entities.keys()) - referenced_entities
        
        for entity_id in orphaned_entities:
            entity = all_entities[entity_id]
            issues.append(ValidationIssue(
                issue_id=f"orphaned_entity_{entity_id}",
                issue_type="orphaned_entity",
                severity="medium",
                description=f"Entity '{entity.name}' is not referenced by any relationships or facts",
                affected_entities=[entity_id],
                suggested_fix="Add relationships/facts or remove if not needed",
                confidence=0.9
            ))
        
        # Check for consistency between relationships and facts
        # This is a simplified check - could be expanded
        
        return issues
    
    def _calculate_overall_assessment(self, issues: List[ValidationIssue]) -> Tuple[float, ValidationStatus]:
        """Calculate overall validation score and status."""
        if not issues:
            return 1.0, ValidationStatus.VALID
        
        # Weight issues by severity
        severity_weights = {
            'low': 0.1,
            'medium': 0.3,
            'high': 0.7,
            'critical': 1.0
        }
        
        total_weight = sum(severity_weights[issue.severity] for issue in issues)
        max_possible_weight = len(issues) * severity_weights['critical']
        
        # Calculate score (1.0 = perfect, 0.0 = worst)
        score = max(0.0, 1.0 - (total_weight / max_possible_weight))
        
        # Determine status
        if score >= 0.9:
            status = ValidationStatus.VALID
        elif score >= 0.7:
            status = ValidationStatus.UNCERTAIN
        elif score >= 0.5:
            status = ValidationStatus.REQUIRES_REVIEW
        elif any(issue.severity == 'critical' for issue in issues):
            status = ValidationStatus.INVALID
        else:
            status = ValidationStatus.CONFLICTING
        
        return score, status
    
    def _generate_recommendations(self, issues: List[ValidationIssue]) -> List[str]:
        """Generate recommendations based on validation issues."""
        recommendations = []
        
        # Group issues by type
        issue_counts = Counter(issue.issue_type for issue in issues)
        
        # Generate specific recommendations
        if issue_counts.get('missing_entity', 0) > 0:
            recommendations.append("Review and fix missing entity references in relationships and facts")
        
        if issue_counts.get('duplicate_relationship', 0) > 0:
            recommendations.append("Remove duplicate relationships to improve graph clarity")
        
        if issue_counts.get('low_confidence', 0) > 10:
            recommendations.append("Review entities/relationships with low confidence scores")
        
        if issue_counts.get('no_sources', 0) > 0:
            recommendations.append("Add reliable sources to facts lacking documentation")
        
        if issue_counts.get('potential_duplicate', 0) > 0:
            recommendations.append("Merge duplicate entities to consolidate knowledge")
        
        # General recommendations
        critical_issues = [i for i in issues if i.severity == 'critical']
        if critical_issues:
            recommendations.insert(0, f"Address {len(critical_issues)} critical issues immediately")
        
        if len(issues) > 100:
            recommendations.append("Consider implementing automated knowledge curation")
        
        return recommendations
    
    def _calculate_validation_statistics(self, issues: List[ValidationIssue], 
                                       entities_validated: int, relationships_validated: int,
                                       facts_validated: int) -> Dict[str, Any]:
        """Calculate detailed validation statistics."""
        # Group issues by type and severity
        issues_by_type = defaultdict(int)
        issues_by_severity = defaultdict(int)
        
        for issue in issues:
            issues_by_type[issue.issue_type] += 1
            issues_by_severity[issue.severity] += 1
        
        # Calculate confidence statistics
        confidences = [issue.confidence for issue in issues if issue.confidence > 0]
        
        return {
            'total_issues': len(issues),
            'issues_by_type': dict(issues_by_type),
            'issues_by_severity': dict(issues_by_severity),
            'entities_validated': entities_validated,
            'relationships_validated': relationships_validated,
            'facts_validated': facts_validated,
            'average_issue_confidence': statistics.mean(confidences) if confidences else 0.0,
            'most_common_issue_type': max(issues_by_type, key=issues_by_type.get) if issues_by_type else None,
            'validation_coverage': {
                'entities': entities_validated,
                'relationships': relationships_validated,
                'facts': facts_validated
            }
        }
    
    def validate_specific_entity(self, entity_id: str) -> List[ValidationIssue]:
        """Validate a specific entity.
        
        Args:
            entity_id: ID of entity to validate
            
        Returns:
            List of validation issues for the entity
        """
        entity = self.knowledge_store.get_entity(entity_id)
        if not entity:
            return [ValidationIssue(
                issue_id=f"entity_not_found_{entity_id}",
                issue_type="entity_not_found",
                severity="critical",
                description=f"Entity {entity_id} not found",
                confidence=1.0
            )]
        
        return self.entity_validator.validate_entity(entity)
    
    def get_validation_summary(self, report: ValidationReport) -> str:
        """Get a human-readable validation summary.
        
        Args:
            report: Validation report
            
        Returns:
            Human-readable summary string
        """
        summary_lines = [
            f"Knowledge Graph Validation Report",
            f"=====================================",
            f"Overall Status: {report.overall_status.value.upper()}",
            f"Overall Score: {report.overall_score:.2f}/1.00",
            f"Validation Level: {report.validation_level.value}",
            f"Validation Time: {report.validation_time:.2f} seconds",
            f"",
            f"Coverage:",
            f"  - Entities Validated: {report.entities_validated}",
            f"  - Relationships Validated: {report.relationships_validated}",
            f"  - Facts Validated: {report.facts_validated}",
            f"",
            f"Issues Found: {len(report.issues)}",
        ]
        
        if report.statistics.get('issues_by_severity'):
            summary_lines.append("Issues by Severity:")
            for severity, count in report.statistics['issues_by_severity'].items():
                summary_lines.append(f"  - {severity.capitalize()}: {count}")
        
        if report.recommendations:
            summary_lines.extend([
                "",
                "Top Recommendations:",
            ])
            for i, rec in enumerate(report.recommendations[:5], 1):
                summary_lines.append(f"  {i}. {rec}")
        
        return "\n".join(summary_lines)


# Convenience functions
def validate_knowledge_graph(knowledge_store: KnowledgeGraphStore, 
                           validation_level: ValidationLevel = ValidationLevel.INTERMEDIATE,
                           fact_verifier: FactVerifier = None) -> ValidationReport:
    """Convenience function to validate a knowledge graph.
    
    Args:
        knowledge_store: Knowledge graph store to validate
        validation_level: Level of validation to perform
        fact_verifier: Optional fact verification engine
        
    Returns:
        Validation report
    """
    validator = KnowledgeValidator(knowledge_store, fact_verifier)
    return validator.validate_knowledge_graph(validation_level)


def quick_validate(knowledge_store: KnowledgeGraphStore) -> ValidationStatus:
    """Quick validation returning only the overall status.
    
    Args:
        knowledge_store: Knowledge graph store to validate
        
    Returns:
        Overall validation status
    """
    report = validate_knowledge_graph(knowledge_store, ValidationLevel.BASIC)
    return report.overall_status