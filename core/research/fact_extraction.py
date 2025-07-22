"""Advanced fact extraction system for research capabilities.

This module provides comprehensive fact extraction from various text sources,
including structured and unstructured documents, with support for multiple
extraction strategies and confidence scoring.
"""

import re
import json
import time
import sqlite3
import hashlib
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import logging
from urllib.parse import urlparse
import mimetypes

from .graphrag import Entity, EntityType, Relationship, RelationType, Fact, KnowledgeGraphStore
from .entity_extraction import EntityCandidate, HybridEntityExtractor, ExtractionResult
from .fact_checker import ClaimExtractor, Claim, ClaimType
from .memory_aug import MemoryAugmentedSystem, MemoryType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SourceType(Enum):
    """Types of fact sources."""
    TEXT_DOCUMENT = "text_document"
    WEB_PAGE = "web_page"
    RESEARCH_PAPER = "research_paper"
    NEWS_ARTICLE = "news_article"
    BOOK = "book"
    ENCYCLOPEDIA = "encyclopedia"
    DATABASE_ENTRY = "database_entry"
    STRUCTURED_DATA = "structured_data"
    UNKNOWN = "unknown"


class ExtractionStrategy(Enum):
    """Fact extraction strategies."""
    PATTERN_BASED = "pattern_based"
    STATISTICAL = "statistical"
    CONTEXTUAL = "contextual"
    HYBRID = "hybrid"
    CLAIM_BASED = "claim_based"
    ENTITY_RELATIONSHIP = "entity_relationship"


class FactType(Enum):
    """Types of extracted facts."""
    BIOGRAPHICAL = "biographical"
    GEOGRAPHICAL = "geographical"
    HISTORICAL = "historical"
    SCIENTIFIC = "scientific"
    STATISTICAL = "statistical"
    DEFINITIONAL = "definitional"
    RELATIONAL = "relational"
    TEMPORAL = "temporal"
    QUANTITATIVE = "quantitative"
    QUALITATIVE = "qualitative"


@dataclass
class SourceMetadata:
    """Metadata about a fact source."""
    source_id: str
    source_type: SourceType
    title: str = ""
    url: str = ""
    author: str = ""
    publication_date: str = ""
    publisher: str = ""
    language: str = "en"
    reliability_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.source_id:
            content = f"{self.title}_{self.url}_{self.author}_{self.publication_date}"
            self.source_id = hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class ExtractedFact:
    """Represents an extracted fact with metadata."""
    fact_id: str
    statement: str
    fact_type: FactType
    confidence: float
    extraction_strategy: ExtractionStrategy
    source_metadata: SourceMetadata
    entities: List[Entity] = field(default_factory=list)
    supporting_text: str = ""
    context: str = ""
    temporal_info: Dict[str, Any] = field(default_factory=dict)
    quantitative_info: Dict[str, Any] = field(default_factory=dict)
    extracted_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.fact_id:
            content = f"{self.statement}_{self.source_metadata.source_id}_{self.extracted_at}"
            self.fact_id = hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_fact(self) -> Fact:
        """Convert to Fact object for storage in knowledge graph."""
        return Fact(
            fact_id=self.fact_id,
            claim=self.statement,
            entities=[entity.entity_id for entity in self.entities],
            confidence=self.confidence,
            evidence=[self.supporting_text] if self.supporting_text else [],
            sources=[self.source_metadata.url or self.source_metadata.title],
            contradictions=[],
            metadata={
                'fact_type': self.fact_type.value,
                'extraction_strategy': self.extraction_strategy.value,
                'source_type': self.source_metadata.source_type.value,
                'extracted_at': self.extracted_at,
                **self.metadata
            }
        )


@dataclass
class ExtractionReport:
    """Report of fact extraction process."""
    report_id: str
    source_metadata: SourceMetadata
    extraction_strategy: ExtractionStrategy
    facts_extracted: List[ExtractedFact] = field(default_factory=list)
    extraction_time: float = 0.0
    processing_stats: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class PatternBasedExtractor:
    """Extract facts using linguistic patterns."""
    
    def __init__(self):
        """Initialize pattern-based extractor."""
        self.patterns = self._initialize_patterns()
        self.entity_extractor = HybridEntityExtractor()
        
    def _initialize_patterns(self) -> Dict[FactType, List[str]]:
        """Initialize extraction patterns for different fact types."""
        return {
            FactType.BIOGRAPHICAL: [
                r'(\w+(?:\s+\w+)*)\s+(?:was born|born)\s+(?:in|on)\s+(.+)',
                r'(\w+(?:\s+\w+)*)\s+(?:died|passed away)\s+(?:in|on)\s+(.+)',
                r'(\w+(?:\s+\w+)*)\s+(?:is|was)\s+a\s+(.+?)(?:\.|,|\s+who|\s+and)',
                r'(\w+(?:\s+\w+)*)\s+(?:worked|served|acted)\s+as\s+(.+?)(?:\.|,)',
                r'(\w+(?:\s+\w+)*)\s+(?:graduated|studied)\s+(?:from|at)\s+(.+?)(?:\.|,)',
            ],
            
            FactType.GEOGRAPHICAL: [
                r'(\w+(?:\s+\w+)*)\s+is located\s+(?:in|at|on)\s+(.+?)(?:\.|,)',
                r'(\w+(?:\s+\w+)*)\s+(?:is|lies)\s+(?:in|at)\s+(.+?)(?:\.|,)',
                r'the capital of\s+(\w+(?:\s+\w+)*)\s+is\s+(.+?)(?:\.|,)',
                r'(\w+(?:\s+\w+)*)\s+has a population of\s+(.+?)(?:\.|,)',
                r'(\w+(?:\s+\w+)*)\s+covers an area of\s+(.+?)(?:\.|,)',
            ],
            
            FactType.HISTORICAL: [
                r'(?:in|on|during)\s+(\d{4}|\w+\s+\d{1,2},?\s+\d{4}|\w+\s+\d{4}),?\s+(.+?)(?:\.|;)',
                r'(.+?)\s+(?:occurred|happened|took place)\s+(?:in|on|during)\s+(\d{4}|\w+\s+\d{4})(?:\.|,)',
                r'the\s+(.+?)\s+(?:began|started|commenced)\s+(?:in|on)\s+(\d{4}|\w+\s+\d{4})(?:\.|,)',
                r'(.+?)\s+(?:ended|concluded|finished)\s+(?:in|on)\s+(\d{4}|\w+\s+\d{4})(?:\.|,)',
            ],
            
            FactType.SCIENTIFIC: [
                r'(\w+(?:\s+\w+)*)\s+(?:contains|consists of|is composed of)\s+(.+?)(?:\.|,)',
                r'the\s+(.+?)\s+of\s+(\w+(?:\s+\w+)*)\s+is\s+(.+?)(?:\.|,)',
                r'(\w+(?:\s+\w+)*)\s+(?:causes|leads to|results in)\s+(.+?)(?:\.|,)',
                r'(.+?)\s+is caused by\s+(.+?)(?:\.|,)',
                r'(\w+(?:\s+\w+)*)\s+has the formula\s+(.+?)(?:\.|,)',
            ],
            
            FactType.STATISTICAL: [
                r'(\d+(?:\.\d+)?(?:\s*%|\s*percent))\s+of\s+(.+?)(?:\.|,)',
                r'(.+?)\s+(?:increased|decreased|rose|fell)\s+(?:by\s+)?(\d+(?:\.\d+)?(?:\s*%|\s*percent))(?:\.|,)',
                r'the\s+(.+?)\s+rate\s+(?:is|was)\s+(\d+(?:\.\d+)?(?:\s*%|\s*per)?.+?)(?:\.|,)',
                r'(\d+(?:,\d{3})*(?:\.\d+)?)\s+(.+?)\s+(?:were|are)\s+(.+?)(?:\.|,)',
            ],
            
            FactType.DEFINITIONAL: [
                r'(\w+(?:\s+\w+)*)\s+(?:is defined as|means|refers to)\s+(.+?)(?:\.|,)',
                r'(.+?)\s+is\s+(?:a type of|a kind of|an example of)\s+(.+?)(?:\.|,)',
                r'(\w+(?:\s+\w+)*)\s+can be described as\s+(.+?)(?:\.|,)',
                r'by definition,\s+(.+?)\s+(?:is|means)\s+(.+?)(?:\.|,)',
            ],
            
            FactType.TEMPORAL: [
                r'(?:before|after|during|since|until)\s+(.+?),\s+(.+?)(?:\.|;)',
                r'(.+?)\s+(?:preceded|followed|came before|came after)\s+(.+?)(?:\.|,)',
                r'(?:from|between)\s+(\d{4}|\w+\s+\d{4})\s+(?:to|and)\s+(\d{4}|\w+\s+\d{4}),?\s+(.+?)(?:\.|,)',
            ],
            
            FactType.QUANTITATIVE: [
                r'(\w+(?:\s+\w+)*)\s+(?:measures|is|weighs|costs)\s+(\d+(?:\.\d+)?\s*\w+)(?:\.|,)',
                r'the\s+(.+?)\s+of\s+(\w+(?:\s+\w+)*)\s+is\s+(\d+(?:\.\d+)?\s*\w*)(?:\.|,)',
                r'(\w+(?:\s+\w+)*)\s+has\s+(\d+(?:\.\d+)?)\s+(.+?)(?:\.|,)',
            ]
        }
    
    def extract_facts(self, text: str, source_metadata: SourceMetadata) -> List[ExtractedFact]:
        """Extract facts using pattern matching.
        
        Args:
            text: Text to extract facts from
            source_metadata: Metadata about the source
            
        Returns:
            List of extracted facts
        """
        extracted_facts = []
        
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        
        for sentence in sentences:
            # Try each fact type pattern
            for fact_type, patterns in self.patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, sentence, re.IGNORECASE)
                    
                    for match in matches:
                        fact = self._create_fact_from_match(
                            match, sentence, fact_type, source_metadata, text
                        )
                        if fact and fact.confidence > 0.3:
                            extracted_facts.append(fact)
        
        return extracted_facts
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Advanced sentence splitting considering abbreviations
        sentences = re.split(r'(?<!\b(?:Dr|Mr|Mrs|Ms|Prof|vs|etc|Inc|Ltd|Corp|Co)\.)(?<!\b\w\.)(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15 and len(sentence.split()) >= 4:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _create_fact_from_match(self, match, sentence: str, fact_type: FactType,
                               source_metadata: SourceMetadata, full_text: str) -> Optional[ExtractedFact]:
        """Create an ExtractedFact from a pattern match."""
        try:
            # Extract entities from the sentence
            entity_result = self.entity_extractor.extract(sentence)
            
            # Build statement from match groups
            groups = match.groups()
            statement = self._build_statement_from_groups(groups, fact_type)
            
            # Calculate confidence based on various factors
            confidence = self._calculate_extraction_confidence(
                match, sentence, entity_result, fact_type, source_metadata
            )
            
            # Extract temporal and quantitative information
            temporal_info = self._extract_temporal_info(sentence, groups)
            quantitative_info = self._extract_quantitative_info(sentence, groups)
            
            # Get context around the sentence
            context = self._get_sentence_context(sentence, full_text)
            
            return ExtractedFact(
                fact_id="",
                statement=statement,
                fact_type=fact_type,
                confidence=confidence,
                extraction_strategy=ExtractionStrategy.PATTERN_BASED,
                source_metadata=source_metadata,
                entities=entity_result.entities,
                supporting_text=sentence,
                context=context,
                temporal_info=temporal_info,
                quantitative_info=quantitative_info,
                metadata={
                    'pattern_matched': match.re.pattern,
                    'match_groups': groups,
                    'entity_extraction_confidence': entity_result.confidence
                }
            )
            
        except Exception as e:
            logger.warning(f"Error creating fact from match: {e}")
            return None
    
    def _build_statement_from_groups(self, groups: Tuple[str, ...], fact_type: FactType) -> str:
        """Build a coherent statement from regex groups."""
        if not groups:
            return ""
        
        # Different strategies for different fact types
        if fact_type == FactType.BIOGRAPHICAL:
            if len(groups) >= 2:
                return f"{groups[0]} {groups[1]}"
        elif fact_type == FactType.GEOGRAPHICAL:
            if len(groups) >= 2:
                return f"{groups[0]} is located in {groups[1]}"
        elif fact_type == FactType.HISTORICAL:
            if len(groups) >= 2:
                return f"In {groups[0]}, {groups[1]}"
        elif fact_type == FactType.STATISTICAL:
            if len(groups) >= 2:
                return f"{groups[0]} of {groups[1]}"
        else:
            # Generic approach
            return " ".join(str(group) for group in groups if group)
        
        return " ".join(str(group) for group in groups if group)
    
    def _calculate_extraction_confidence(self, match, sentence: str, entity_result: ExtractionResult,
                                       fact_type: FactType, source_metadata: SourceMetadata) -> float:
        """Calculate confidence score for extracted fact."""
        confidence = 0.5  # Base confidence
        
        # Entity extraction quality
        if entity_result.entities:
            avg_entity_confidence = sum(e.confidence for e in entity_result.entities) / len(entity_result.entities)
            confidence += avg_entity_confidence * 0.2
        
        # Pattern specificity bonus
        pattern_bonus = {
            FactType.STATISTICAL: 0.2,
            FactType.QUANTITATIVE: 0.2,
            FactType.TEMPORAL: 0.15,
            FactType.GEOGRAPHICAL: 0.15,
            FactType.BIOGRAPHICAL: 0.1
        }
        confidence += pattern_bonus.get(fact_type, 0.05)
        
        # Source reliability
        confidence += source_metadata.reliability_score * 0.3
        
        # Sentence quality indicators
        if len(sentence.split()) >= 8:  # Reasonable length
            confidence += 0.1
        
        # Presence of numbers/dates (for relevant fact types)
        if fact_type in [FactType.STATISTICAL, FactType.QUANTITATIVE, FactType.TEMPORAL]:
            if re.search(r'\d+', sentence):
                confidence += 0.1
        
        # Proper nouns (likely entities)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', sentence)
        if len(proper_nouns) >= 2:
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _extract_temporal_info(self, sentence: str, groups: Tuple[str, ...]) -> Dict[str, Any]:
        """Extract temporal information from sentence."""
        temporal_info = {}
        
        # Look for years
        years = re.findall(r'\b(19|20)\d{2}\b', sentence)
        if years:
            temporal_info['years'] = years
        
        # Look for dates
        dates = re.findall(r'\b\w+\s+\d{1,2},?\s+\d{4}\b', sentence)
        if dates:
            temporal_info['dates'] = dates
        
        # Look for temporal keywords
        temporal_keywords = re.findall(r'\b(before|after|during|since|until|from|to|between)\b', sentence, re.IGNORECASE)
        if temporal_keywords:
            temporal_info['temporal_relations'] = temporal_keywords
        
        return temporal_info
    
    def _extract_quantitative_info(self, sentence: str, groups: Tuple[str, ...]) -> Dict[str, Any]:
        """Extract quantitative information from sentence."""
        quantitative_info = {}
        
        # Look for numbers with units
        numbers_with_units = re.findall(r'\d+(?:\.\d+)?\s*(?:kg|km|m|cm|mm|tons?|pounds?|%|percent|dollars?|\$)', sentence, re.IGNORECASE)
        if numbers_with_units:
            quantitative_info['measurements'] = numbers_with_units
        
        # Look for percentages
        percentages = re.findall(r'\d+(?:\.\d+)?\s*(?:%|percent)', sentence, re.IGNORECASE)
        if percentages:
            quantitative_info['percentages'] = percentages
        
        # Look for large numbers
        large_numbers = re.findall(r'\d{1,3}(?:,\d{3})+', sentence)
        if large_numbers:
            quantitative_info['large_numbers'] = large_numbers
        
        return quantitative_info
    
    def _get_sentence_context(self, sentence: str, full_text: str, window: int = 100) -> str:
        """Get context around a sentence."""
        start_pos = full_text.find(sentence)
        if start_pos == -1:
            return sentence
        
        context_start = max(0, start_pos - window)
        context_end = min(len(full_text), start_pos + len(sentence) + window)
        
        return full_text[context_start:context_end]


class StatisticalExtractor:
    """Extract facts using statistical analysis."""
    
    def __init__(self):
        """Initialize statistical extractor."""
        self.entity_extractor = HybridEntityExtractor()
        self.claim_extractor = ClaimExtractor()
        
    def extract_facts(self, text: str, source_metadata: SourceMetadata) -> List[ExtractedFact]:
        """Extract facts using statistical analysis.
        
        Args:
            text: Text to extract facts from
            source_metadata: Metadata about the source
            
        Returns:
            List of extracted facts
        """
        extracted_facts = []
        
        # Extract claims first
        claims = self.claim_extractor.extract_claims(text, source_metadata.source_id)
        
        # Convert high-confidence claims to facts
        for claim in claims:
            if claim.confidence > 0.6:
                fact = self._claim_to_fact(claim, source_metadata, text)
                if fact:
                    extracted_facts.append(fact)
        
        # Extract entity-based facts
        entity_facts = self._extract_entity_facts(text, source_metadata)
        extracted_facts.extend(entity_facts)
        
        return extracted_facts
    
    def _claim_to_fact(self, claim: Claim, source_metadata: SourceMetadata, full_text: str) -> Optional[ExtractedFact]:
        """Convert a claim to an extracted fact."""
        # Determine fact type from claim type
        fact_type_mapping = {
            ClaimType.FACTUAL: FactType.DEFINITIONAL,
            ClaimType.NUMERICAL: FactType.QUANTITATIVE,
            ClaimType.TEMPORAL: FactType.TEMPORAL,
            ClaimType.CAUSAL: FactType.SCIENTIFIC,
            ClaimType.COMPARATIVE: FactType.QUALITATIVE,
            ClaimType.RELATIONAL: FactType.RELATIONAL
        }
        
        fact_type = fact_type_mapping.get(claim.claim_type, FactType.QUALITATIVE)
        
        # Get context
        context = self._get_context_for_claim(claim, full_text)
        
        return ExtractedFact(
            fact_id="",
            statement=claim.text,
            fact_type=fact_type,
            confidence=claim.confidence,
            extraction_strategy=ExtractionStrategy.CLAIM_BASED,
            source_metadata=source_metadata,
            entities=claim.entities,
            supporting_text=claim.text,
            context=context,
            metadata={
                'claim_type': claim.claim_type.value if claim.claim_type else None,
                'claim_id': claim.claim_id,
                'predicates': claim.predicates
            }
        )
    
    def _extract_entity_facts(self, text: str, source_metadata: SourceMetadata) -> List[ExtractedFact]:
        """Extract facts based on entity relationships."""
        facts = []
        
        # Extract entities
        entity_result = self.entity_extractor.extract(text)
        
        # Look for relationships between entities
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            # Find entities in this sentence
            sentence_entities = []
            for entity in entity_result.entities:
                if entity.name.lower() in sentence.lower():
                    sentence_entities.append(entity)
            
            # If we have multiple entities, look for relationships
            if len(sentence_entities) >= 2:
                fact = self._create_relationship_fact(sentence, sentence_entities, source_metadata)
                if fact:
                    facts.append(fact)
        
        return facts
    
    def _create_relationship_fact(self, sentence: str, entities: List[Entity], 
                                source_metadata: SourceMetadata) -> Optional[ExtractedFact]:
        """Create a fact from entities found in a sentence."""
        # Simple heuristic to determine fact type
        fact_type = FactType.RELATIONAL
        
        # Look for specific indicators
        if re.search(r'\b(born|died|birth|death)\b', sentence, re.IGNORECASE):
            fact_type = FactType.BIOGRAPHICAL
        elif re.search(r'\b(located|capital|city|country|state)\b', sentence, re.IGNORECASE):
            fact_type = FactType.GEOGRAPHICAL
        elif re.search(r'\b(\d{4}|century|year|before|after)\b', sentence, re.IGNORECASE):
            fact_type = FactType.HISTORICAL
        elif re.search(r'\b(\d+%|percent|ratio|rate)\b', sentence, re.IGNORECASE):
            fact_type = FactType.STATISTICAL
        
        # Calculate confidence based on entity confidence and sentence quality
        entity_confidences = [e.confidence for e in entities]
        avg_confidence = sum(entity_confidences) / len(entity_confidences)
        
        confidence = avg_confidence * 0.7 + source_metadata.reliability_score * 0.3
        
        if confidence > 0.4:
            return ExtractedFact(
                fact_id="",
                statement=sentence,
                fact_type=fact_type,
                confidence=confidence,
                extraction_strategy=ExtractionStrategy.ENTITY_RELATIONSHIP,
                source_metadata=source_metadata,
                entities=entities,
                supporting_text=sentence,
                metadata={
                    'entity_count': len(entities),
                    'entity_types': [e.entity_type.value for e in entities]
                }
            )
        
        return None
    
    def _get_context_for_claim(self, claim: Claim, full_text: str, window: int = 100) -> str:
        """Get context around a claim."""
        if claim.context:
            return claim.context
        
        # Find claim in text and get context
        claim_pos = full_text.find(claim.text)
        if claim_pos == -1:
            return claim.text
        
        context_start = max(0, claim_pos - window)
        context_end = min(len(full_text), claim_pos + len(claim.text) + window)
        
        return full_text[context_start:context_end]


class FactExtractor:
    """Main fact extraction system combining multiple strategies."""
    
    def __init__(self, knowledge_store: KnowledgeGraphStore = None, 
                 memory_system: MemoryAugmentedSystem = None):
        """Initialize fact extractor.
        
        Args:
            knowledge_store: Optional knowledge graph store for fact storage
            memory_system: Optional memory system for caching results
        """
        self.knowledge_store = knowledge_store
        self.memory_system = memory_system
        
        # Initialize extractors
        self.pattern_extractor = PatternBasedExtractor()
        self.statistical_extractor = StatisticalExtractor()
        
        # Statistics
        self.extraction_stats = {
            'total_extractions': 0,
            'facts_extracted': 0,
            'sources_processed': 0,
            'extraction_time': 0.0
        }
        
    def extract_facts_from_text(self, text: str, source_metadata: SourceMetadata = None,
                              strategy: ExtractionStrategy = ExtractionStrategy.HYBRID) -> ExtractionReport:
        """Extract facts from text using specified strategy.
        
        Args:
            text: Text to extract facts from
            source_metadata: Metadata about the source
            strategy: Extraction strategy to use
            
        Returns:
            Extraction report with results
        """
        start_time = time.time()
        
        if not source_metadata:
            source_metadata = SourceMetadata(
                source_id="",
                source_type=SourceType.TEXT_DOCUMENT,
                title="Unknown Source"
            )
        
        logger.info(f"Starting fact extraction from {source_metadata.title} using {strategy.value} strategy")
        
        all_facts = []
        errors = []
        warnings = []
        
        try:
            # Apply extraction strategy
            if strategy == ExtractionStrategy.PATTERN_BASED:
                facts = self.pattern_extractor.extract_facts(text, source_metadata)
                all_facts.extend(facts)
            
            elif strategy == ExtractionStrategy.STATISTICAL:
                facts = self.statistical_extractor.extract_facts(text, source_metadata)
                all_facts.extend(facts)
            
            elif strategy == ExtractionStrategy.HYBRID:
                # Combine multiple strategies
                pattern_facts = self.pattern_extractor.extract_facts(text, source_metadata)
                statistical_facts = self.statistical_extractor.extract_facts(text, source_metadata)
                
                all_facts.extend(pattern_facts)
                all_facts.extend(statistical_facts)
                
                # Deduplicate facts
                all_facts = self._deduplicate_facts(all_facts)
            
            else:
                raise ValueError(f"Unsupported extraction strategy: {strategy}")
            
            # Post-process facts
            all_facts = self._post_process_facts(all_facts, text, source_metadata)
            
            # Store facts in knowledge graph if available
            if self.knowledge_store:
                self._store_facts_in_knowledge_graph(all_facts)
            
        except Exception as e:
            error_msg = f"Error during fact extraction: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
        
        extraction_time = time.time() - start_time
        
        # Update statistics
        self.extraction_stats['total_extractions'] += 1
        self.extraction_stats['facts_extracted'] += len(all_facts)
        self.extraction_stats['sources_processed'] += 1
        self.extraction_stats['extraction_time'] += extraction_time
        
        # Create extraction report
        report = ExtractionReport(
            report_id=hashlib.sha256(f"{source_metadata.source_id}_{time.time()}".encode()).hexdigest()[:16],
            source_metadata=source_metadata,
            extraction_strategy=strategy,
            facts_extracted=all_facts,
            extraction_time=extraction_time,
            processing_stats={
                'text_length': len(text),
                'sentences_processed': len(re.split(r'[.!?]+', text)),
                'entities_found': len(set(entity.entity_id for fact in all_facts for entity in fact.entities)),
                'fact_types_found': list(set(fact.fact_type.value for fact in all_facts))
            },
            errors=errors,
            warnings=warnings
        )
        
        # Store report in memory if available
        if self.memory_system:
            self.memory_system.store_memory(
                content=report,
                memory_type=MemoryType.PROCEDURAL,
                importance_score=0.7,
                metadata={
                    'type': 'fact_extraction_report',
                    'strategy': strategy.value,
                    'facts_count': len(all_facts),
                    'source_type': source_metadata.source_type.value
                }
            )
        
        logger.info(f"Fact extraction completed in {extraction_time:.2f}s. Extracted {len(all_facts)} facts.")
        
        return report
    
    def extract_facts_from_file(self, file_path: str, source_metadata: SourceMetadata = None,
                              strategy: ExtractionStrategy = ExtractionStrategy.HYBRID) -> ExtractionReport:
        """Extract facts from a file.
        
        Args:
            file_path: Path to file to extract facts from
            source_metadata: Metadata about the source
            strategy: Extraction strategy to use
            
        Returns:
            Extraction report with results
        """
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Create source metadata if not provided
            if not source_metadata:
                source_metadata = self._create_source_metadata_from_file(file_path)
            
            return self.extract_facts_from_text(text, source_metadata, strategy)
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            
            # Return empty report with error
            return ExtractionReport(
                report_id="",
                source_metadata=source_metadata or SourceMetadata("", SourceType.UNKNOWN),
                extraction_strategy=strategy,
                errors=[f"Failed to read file: {e}"]
            )
    
    def batch_extract_facts(self, sources: List[Tuple[str, SourceMetadata]], 
                          strategy: ExtractionStrategy = ExtractionStrategy.HYBRID) -> List[ExtractionReport]:
        """Extract facts from multiple sources.
        
        Args:
            sources: List of (text, source_metadata) tuples
            strategy: Extraction strategy to use
            
        Returns:
            List of extraction reports
        """
        reports = []
        
        for i, (text, source_metadata) in enumerate(sources):
            logger.info(f"Processing source {i+1}/{len(sources)}: {source_metadata.title}")
            
            try:
                report = self.extract_facts_from_text(text, source_metadata, strategy)
                reports.append(report)
            except Exception as e:
                logger.error(f"Error processing source {source_metadata.title}: {e}")
                
                error_report = ExtractionReport(
                    report_id="",
                    source_metadata=source_metadata,
                    extraction_strategy=strategy,
                    errors=[f"Processing failed: {e}"]
                )
                reports.append(error_report)
        
        return reports
    
    def _deduplicate_facts(self, facts: List[ExtractedFact]) -> List[ExtractedFact]:
        """Remove duplicate facts."""
        seen_statements = set()
        unique_facts = []
        
        for fact in facts:
            # Normalize statement for comparison
            normalized = re.sub(r'\s+', ' ', fact.statement.lower().strip())
            
            if normalized not in seen_statements:
                unique_facts.append(fact)
                seen_statements.add(normalized)
        
        return unique_facts
    
    def _post_process_facts(self, facts: List[ExtractedFact], text: str, 
                          source_metadata: SourceMetadata) -> List[ExtractedFact]:
        """Post-process extracted facts."""
        processed_facts = []
        
        for fact in facts:
            # Filter out low-quality facts
            if fact.confidence < 0.2:
                continue
            
            # Enhance fact with additional metadata
            fact.metadata['text_length'] = len(text)
            fact.metadata['source_reliability'] = source_metadata.reliability_score
            
            # Validate fact statement
            if len(fact.statement.strip()) < 10:
                continue
            
            processed_facts.append(fact)
        
        return processed_facts
    
    def _store_facts_in_knowledge_graph(self, facts: List[ExtractedFact]):
        """Store extracted facts in knowledge graph."""
        for fact in facts:
            try:
                # Convert to Fact object and store
                kg_fact = fact.to_fact()
                self.knowledge_store.add_fact(kg_fact)
                
                # Also store entities if they don't exist
                for entity in fact.entities:
                    existing_entity = self.knowledge_store.get_entity(entity.entity_id)
                    if not existing_entity:
                        self.knowledge_store.add_entity(entity)
                        
            except Exception as e:
                logger.warning(f"Error storing fact in knowledge graph: {e}")
    
    def _create_source_metadata_from_file(self, file_path: str) -> SourceMetadata:
        """Create source metadata from file path."""
        # Determine source type from file extension
        _, ext = os.path.splitext(file_path)
        source_type = SourceType.TEXT_DOCUMENT
        
        if ext.lower() in ['.html', '.htm']:
            source_type = SourceType.WEB_PAGE
        elif ext.lower() in ['.pdf']:
            source_type = SourceType.RESEARCH_PAPER
        elif ext.lower() in ['.json', '.csv', '.xml']:
            source_type = SourceType.STRUCTURED_DATA
        
        # Extract filename as title
        title = os.path.basename(file_path)
        
        return SourceMetadata(
            source_id="",
            source_type=source_type,
            title=title,
            url=f"file://{file_path}",
            reliability_score=0.7  # Default reliability for files
        )
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get fact extraction statistics."""
        stats = self.extraction_stats.copy()
        
        # Calculate derived statistics
        if stats['total_extractions'] > 0:
            stats['avg_facts_per_extraction'] = stats['facts_extracted'] / stats['total_extractions']
            stats['avg_extraction_time'] = stats['extraction_time'] / stats['total_extractions']
        else:
            stats['avg_facts_per_extraction'] = 0.0
            stats['avg_extraction_time'] = 0.0
        
        return stats


# Convenience functions
def extract_facts_from_text(text: str, title: str = "Unknown Source",
                          strategy: ExtractionStrategy = ExtractionStrategy.HYBRID) -> List[ExtractedFact]:
    """Convenience function to extract facts from text.
    
    Args:
        text: Text to extract facts from
        title: Title of the source
        strategy: Extraction strategy to use
        
    Returns:
        List of extracted facts
    """
    source_metadata = SourceMetadata(
        source_id="",
        source_type=SourceType.TEXT_DOCUMENT,
        title=title
    )
    
    extractor = FactExtractor()
    report = extractor.extract_facts_from_text(text, source_metadata, strategy)
    
    return report.facts_extracted


def extract_facts_from_file(file_path: str, 
                          strategy: ExtractionStrategy = ExtractionStrategy.HYBRID) -> List[ExtractedFact]:
    """Convenience function to extract facts from a file.
    
    Args:
        file_path: Path to file to extract facts from
        strategy: Extraction strategy to use
        
    Returns:
        List of extracted facts
    """
    extractor = FactExtractor()
    report = extractor.extract_facts_from_file(file_path, strategy=strategy)
    
    return report.facts_extracted