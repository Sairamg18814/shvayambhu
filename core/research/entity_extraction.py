"""Advanced entity extraction system for knowledge graph population.

This module provides sophisticated entity extraction capabilities using multiple
approaches including pattern matching, statistical analysis, and contextual
understanding for building comprehensive knowledge graphs.
"""

import re
import string
import math
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import unicodedata
import logging

from .graphrag import Entity, EntityType, Relationship, RelationType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExtractionMethod(Enum):
    """Methods for entity extraction."""
    PATTERN_BASED = "pattern_based"
    STATISTICAL = "statistical"
    CONTEXTUAL = "contextual"
    HYBRID = "hybrid"


@dataclass
class ExtractionResult:
    """Result of entity extraction process."""
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    confidence: float = 0.0
    method_used: ExtractionMethod = ExtractionMethod.HYBRID
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityCandidate:
    """Candidate entity with extraction metadata."""
    text: str
    entity_type: EntityType
    confidence: float
    start_pos: int
    end_pos: int
    context: str = ""
    features: Dict[str, Any] = field(default_factory=dict)
    
    def to_entity(self) -> Entity:
        """Convert candidate to Entity object."""
        return Entity(
            entity_id="",
            name=self.text,
            entity_type=self.entity_type,
            confidence=self.confidence,
            attributes=self.features
        )


class PatternBasedExtractor:
    """Pattern-based entity extraction using regular expressions and rules."""
    
    def __init__(self):
        """Initialize pattern-based extractor with predefined patterns."""
        self.patterns = self._initialize_patterns()
        self.contextual_patterns = self._initialize_contextual_patterns()
        
    def _initialize_patterns(self) -> Dict[EntityType, List[str]]:
        """Initialize regex patterns for different entity types."""
        patterns = {
            EntityType.PERSON: [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
                r'\b[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+\b',  # First M. Last
                r'\b[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+\b',  # First Middle Last
                r'\bDr\. [A-Z][a-z]+ [A-Z][a-z]+\b',  # Dr. First Last
                r'\bProf\. [A-Z][a-z]+ [A-Z][a-z]+\b',  # Prof. First Last
            ],
            
            EntityType.ORGANIZATION: [
                r'\b[A-Z][a-zA-Z&\s]+ (?:Inc|Corp|LLC|Ltd|Company|Corporation|Group|Foundation|Institute)\b',
                r'\b(?:The\s+)?[A-Z][a-zA-Z\s]+ (?:University|College|School)\b',
                r'\b[A-Z][A-Z]+ [A-Z][a-zA-Z\s]+\b',  # Acronym followed by name
                r'\b[A-Z]{2,6}\b',  # Acronyms (2-6 uppercase letters)
            ],
            
            EntityType.LOCATION: [
                r'\b[A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*\b',  # City, State
                r'\b[A-Z][a-z]+ (?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr)\b',
                r'\b(?:Mount|Mt|Lake|River)\s+[A-Z][a-z]+\b',
                r'\b[A-Z][a-z]+ (?:County|Parish|Province|State)\b',
            ],
            
            EntityType.DATE: [
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',
                r'\b\d{4}-\d{2}-\d{2}\b',
                r'\b\d{1,2}-\d{1,2}-\d{4}\b',
            ],
            
            EntityType.NUMBER: [
                r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b',  # Numbers with commas
                r'\b\d+(?:\.\d+)?%\b',  # Percentages
                r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b',  # Currency
                r'\b\d+(?:\.\d+)?\s*(?:million|billion|trillion|thousand)\b',  # Large numbers
            ],
            
            EntityType.EVENT: [
                r'\b[A-Z][a-zA-Z\s]+ (?:Conference|Summit|Meeting|Festival|Championship|War|Battle)\b',
                r'\b(?:The\s+)?[A-Z][a-zA-Z\s]+ (?:Olympics|Games|Awards|Prize)\b',
            ]
        }
        
        # Compile patterns for efficiency
        compiled_patterns = {}
        for entity_type, pattern_list in patterns.items():
            compiled_patterns[entity_type] = [re.compile(pattern, re.IGNORECASE) for pattern in pattern_list]
        
        return compiled_patterns
    
    def _initialize_contextual_patterns(self) -> Dict[EntityType, List[str]]:
        """Initialize contextual patterns that indicate entity types."""
        return {
            EntityType.PERSON: [
                r'(?:said|stated|announced|claimed|argued|believes|thinks)',
                r'(?:CEO|president|director|manager|scientist|researcher|professor|doctor)',
                r'(?:born|died|lived|graduated|studied|worked)',
            ],
            EntityType.ORGANIZATION: [
                r'(?:founded|established|created|launched|acquired|merged)',
                r'(?:headquarters|offices|based|located)',
                r'(?:employees|staff|workers|members)',
            ],
            EntityType.LOCATION: [
                r'(?:located|situated|positioned|found)',
                r'(?:population|residents|inhabitants)',
                r'(?:capital|city|town|village|country|state)',
            ],
        }
    
    def extract(self, text: str) -> List[EntityCandidate]:
        """Extract entities using pattern matching.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            List of entity candidates
        """
        candidates = []
        text_lower = text.lower()
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(text)
                
                for match in matches:
                    candidate_text = match.group().strip()
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Get context (surrounding words)
                    context_start = max(0, start_pos - 50)
                    context_end = min(len(text), end_pos + 50)
                    context = text[context_start:context_end]
                    
                    # Calculate confidence based on pattern strength and context
                    confidence = self._calculate_pattern_confidence(
                        candidate_text, entity_type, context, text_lower
                    )
                    
                    # Filter out low-confidence matches
                    if confidence > 0.3:
                        candidate = EntityCandidate(
                            text=candidate_text,
                            entity_type=entity_type,
                            confidence=confidence,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            context=context,
                            features={
                                'extraction_method': 'pattern_based',
                                'pattern_matched': pattern.pattern
                            }
                        )
                        candidates.append(candidate)
        
        return self._deduplicate_candidates(candidates)
    
    def _calculate_pattern_confidence(self, text: str, entity_type: EntityType, 
                                    context: str, text_lower: str) -> float:
        """Calculate confidence score for a pattern match."""
        confidence = 0.5  # Base confidence
        
        # Length-based adjustments
        if len(text) < 3:
            confidence -= 0.3
        elif len(text) > 20:
            confidence -= 0.1
        
        # Capitalization patterns
        if entity_type in [EntityType.PERSON, EntityType.LOCATION, EntityType.ORGANIZATION]:
            if text.istitle():
                confidence += 0.2
            elif text.isupper() and len(text) <= 6:  # Likely acronym
                confidence += 0.1
            elif text.islower():
                confidence -= 0.2
        
        # Contextual indicators
        contextual_patterns = self.contextual_patterns.get(entity_type, [])
        for pattern_str in contextual_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            if pattern.search(context):
                confidence += 0.15
                break
        
        # Common word penalties
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        if text.lower() in common_words:
            confidence -= 0.5
        
        # Special character handling
        if any(char in text for char in ['@', '#', '$', '%', '^', '&']):
            if entity_type != EntityType.NUMBER:
                confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def _deduplicate_candidates(self, candidates: List[EntityCandidate]) -> List[EntityCandidate]:
        """Remove duplicate and overlapping candidates."""
        # Sort by start position
        candidates.sort(key=lambda x: x.start_pos)
        
        deduplicated = []
        for candidate in candidates:
            # Check for overlaps with existing candidates
            overlap = False
            for existing in deduplicated:
                if (candidate.start_pos < existing.end_pos and 
                    candidate.end_pos > existing.start_pos):
                    # Overlap detected - keep the one with higher confidence
                    if candidate.confidence > existing.confidence:
                        deduplicated.remove(existing)
                    else:
                        overlap = True
                    break
            
            if not overlap:
                deduplicated.append(candidate)
        
        return deduplicated


class StatisticalExtractor:
    """Statistical entity extraction using frequency analysis and scoring."""
    
    def __init__(self):
        """Initialize statistical extractor."""
        self.entity_indicators = self._initialize_indicators()
        self.min_frequency = 2
        self.min_length = 3
        
    def _initialize_indicators(self) -> Dict[EntityType, Dict[str, float]]:
        """Initialize statistical indicators for entity types."""
        return {
            EntityType.PERSON: {
                'title_case_ratio': 0.8,
                'avg_word_length': 5.0,
                'common_prefixes': ['mr', 'mrs', 'ms', 'dr', 'prof'],
                'common_suffixes': ['jr', 'sr', 'iii', 'iv']
            },
            EntityType.ORGANIZATION: {
                'uppercase_ratio': 0.3,
                'acronym_likelihood': 0.6,
                'common_suffixes': ['inc', 'corp', 'llc', 'ltd', 'co']
            },
            EntityType.LOCATION: {
                'title_case_ratio': 0.9,
                'geographic_indicators': ['north', 'south', 'east', 'west', 'central'],
                'location_suffixes': ['city', 'town', 'county', 'state', 'country']
            }
        }
    
    def extract(self, text: str) -> List[EntityCandidate]:
        """Extract entities using statistical analysis.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            List of entity candidates
        """
        candidates = []
        
        # Tokenize and analyze word patterns
        words = self._tokenize_and_clean(text)
        word_frequencies = Counter(words)
        
        # Extract n-grams (1-4 grams)
        ngrams = self._extract_ngrams(words, max_n=4)
        
        # Analyze each n-gram
        for ngram, frequency in ngrams.items():
            if frequency >= self.min_frequency and len(' '.join(ngram)) >= self.min_length:
                ngram_text = ' '.join(ngram)
                
                # Calculate entity type probabilities
                type_scores = self._calculate_type_scores(ngram_text, ngram, frequency)
                
                # Get best entity type
                best_type = max(type_scores, key=type_scores.get)
                confidence = type_scores[best_type]
                
                if confidence > 0.4:
                    # Find positions in text
                    positions = self._find_positions(text, ngram_text)
                    
                    for start_pos, end_pos in positions:
                        context_start = max(0, start_pos - 30)
                        context_end = min(len(text), end_pos + 30)
                        context = text[context_start:context_end]
                        
                        candidate = EntityCandidate(
                            text=ngram_text,
                            entity_type=best_type,
                            confidence=confidence,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            context=context,
                            features={
                                'extraction_method': 'statistical',
                                'frequency': frequency,
                                'type_scores': type_scores
                            }
                        )
                        candidates.append(candidate)
        
        return candidates
    
    def _tokenize_and_clean(self, text: str) -> List[str]:
        """Tokenize text and clean tokens."""
        # Remove punctuation and split
        translator = str.maketrans('', '', string.punctuation.replace('-', ''))
        cleaned_text = text.translate(translator)
        
        words = cleaned_text.split()
        
        # Filter out very short or very long words
        filtered_words = [
            word for word in words 
            if 2 <= len(word) <= 30 and not word.isdigit()
        ]
        
        return filtered_words
    
    def _extract_ngrams(self, words: List[str], max_n: int = 4) -> Dict[Tuple[str, ...], int]:
        """Extract n-grams from word list."""
        ngrams = defaultdict(int)
        
        for n in range(1, max_n + 1):
            for i in range(len(words) - n + 1):
                ngram = tuple(words[i:i + n])
                
                # Filter out n-grams with common stop words in certain positions
                if self._is_valid_ngram(ngram):
                    ngrams[ngram] += 1
        
        return dict(ngrams)
    
    def _is_valid_ngram(self, ngram: Tuple[str, ...]) -> bool:
        """Check if n-gram is valid for entity extraction."""
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Don't allow n-grams that start or end with stop words
        if ngram[0].lower() in stop_words or ngram[-1].lower() in stop_words:
            return False
        
        # Don't allow n-grams that are entirely stop words
        if all(word.lower() in stop_words for word in ngram):
            return False
        
        # Require at least one capitalized word for multi-word n-grams
        if len(ngram) > 1 and not any(word[0].isupper() for word in ngram):
            return False
        
        return True
    
    def _calculate_type_scores(self, text: str, ngram: Tuple[str, ...], frequency: int) -> Dict[EntityType, float]:
        """Calculate entity type probability scores."""
        scores = {}
        
        for entity_type in EntityType:
            score = 0.0
            
            if entity_type == EntityType.PERSON:
                score = self._score_person_likelihood(text, ngram)
            elif entity_type == EntityType.ORGANIZATION:
                score = self._score_organization_likelihood(text, ngram)
            elif entity_type == EntityType.LOCATION:
                score = self._score_location_likelihood(text, ngram)
            elif entity_type == EntityType.CONCEPT:
                score = self._score_concept_likelihood(text, ngram)
            else:
                score = 0.1  # Default low score
            
            # Frequency bonus
            freq_bonus = min(0.2, frequency * 0.05)
            score += freq_bonus
            
            scores[entity_type] = min(1.0, score)
        
        return scores
    
    def _score_person_likelihood(self, text: str, ngram: Tuple[str, ...]) -> float:
        """Calculate likelihood of being a person name."""
        score = 0.0
        
        # Title case bonus
        if all(word.istitle() for word in ngram):
            score += 0.3
        
        # Length patterns
        if len(ngram) == 2:  # First Last
            score += 0.4
        elif len(ngram) == 3:  # First Middle Last
            score += 0.3
        elif len(ngram) == 1:
            score += 0.1
        
        # Common name patterns
        if len(ngram) >= 2:
            # Check for initials (single letters)
            if any(len(word) == 1 and word.isupper() for word in ngram):
                score += 0.2
        
        # Prefix/suffix indicators
        text_lower = text.lower()
        person_indicators = ['mr', 'mrs', 'ms', 'dr', 'prof', 'jr', 'sr']
        if any(indicator in text_lower for indicator in person_indicators):
            score += 0.3
        
        return score
    
    def _score_organization_likelihood(self, text: str, ngram: Tuple[str, ...]) -> float:
        """Calculate likelihood of being an organization."""
        score = 0.0
        
        # Corporate suffixes
        org_suffixes = ['inc', 'corp', 'llc', 'ltd', 'company', 'corporation', 'group']
        text_lower = text.lower()
        if any(suffix in text_lower for suffix in org_suffixes):
            score += 0.5
        
        # Acronym patterns
        if len(ngram) == 1 and ngram[0].isupper() and 2 <= len(ngram[0]) <= 6:
            score += 0.4
        
        # Institution patterns
        institution_words = ['university', 'college', 'school', 'institute', 'foundation']
        if any(word in text_lower for word in institution_words):
            score += 0.4
        
        # Mixed case patterns
        if any('&' in word for word in ngram):
            score += 0.2
        
        return score
    
    def _score_location_likelihood(self, text: str, ngram: Tuple[str, ...]) -> float:
        """Calculate likelihood of being a location."""
        score = 0.0
        
        # Geographic suffixes
        geo_suffixes = ['city', 'town', 'county', 'state', 'country', 'street', 'avenue', 'road']
        text_lower = text.lower()
        if any(suffix in text_lower for suffix in geo_suffixes):
            score += 0.5
        
        # Geographic prefixes
        geo_prefixes = ['mount', 'lake', 'river', 'north', 'south', 'east', 'west']
        if any(prefix in text_lower for prefix in geo_prefixes):
            score += 0.3
        
        # Title case pattern
        if all(word.istitle() for word in ngram):
            score += 0.2
        
        return score
    
    def _score_concept_likelihood(self, text: str, ngram: Tuple[str, ...]) -> float:
        """Calculate likelihood of being a concept."""
        score = 0.1  # Base score for concepts
        
        # Abstract/technical terms
        if len(ngram) >= 2:
            score += 0.2
        
        # Scientific/technical indicators
        technical_indicators = ['theory', 'principle', 'method', 'algorithm', 'system']
        text_lower = text.lower()
        if any(indicator in text_lower for indicator in technical_indicators):
            score += 0.3
        
        return score
    
    def _find_positions(self, text: str, target: str) -> List[Tuple[int, int]]:
        """Find all positions of target string in text."""
        positions = []
        start = 0
        
        while True:
            pos = text.find(target, start)
            if pos == -1:
                break
            positions.append((pos, pos + len(target)))
            start = pos + 1
        
        return positions


class ContextualExtractor:
    """Contextual entity extraction using surrounding text analysis."""
    
    def __init__(self):
        """Initialize contextual extractor."""
        self.context_patterns = self._initialize_context_patterns()
        self.entity_relationships = self._initialize_relationship_patterns()
    
    def _initialize_context_patterns(self) -> Dict[EntityType, List[str]]:
        """Initialize context patterns that strongly indicate entity types."""
        return {
            EntityType.PERSON: [
                r'(?:said|stated|announced|claimed|argued|believes|thinks|mentioned)',
                r'(?:CEO|president|director|manager|scientist|researcher|professor|doctor|author)',
                r'(?:born|died|lived|graduated|studied|worked|employed)',
                r'(?:he|she|his|her|him|her)\s+(?:said|is|was|has|had)',
            ],
            EntityType.ORGANIZATION: [
                r'(?:founded|established|created|launched|acquired|merged|owns)',
                r'(?:headquarters|offices|based|located|operates)',
                r'(?:employees|staff|workers|members|shareholders)',
                r'(?:company|corporation|organization|institution|agency)',
            ],
            EntityType.LOCATION: [
                r'(?:located|situated|positioned|found|based)\s+(?:in|at|on)',
                r'(?:population|residents|inhabitants|citizens)',
                r'(?:capital|city|town|village|country|state|province|region)',
                r'(?:travel|visit|go)\s+to',
            ],
            EntityType.EVENT: [
                r'(?:occurred|happened|took place|held|organized)',
                r'(?:during|at|on)\s+(?:the|this)',
                r'(?:conference|meeting|summit|festival|celebration|ceremony)',
                r'(?:participants|attendees|speakers|organizers)',
            ],
            EntityType.CONCEPT: [
                r'(?:theory|principle|concept|idea|method|approach|technique)',
                r'(?:defined|explained|described|characterized)',
                r'(?:understanding|knowledge|research|study|analysis)',
                r'(?:refers to|means|indicates|suggests)',
            ]
        }
    
    def _initialize_relationship_patterns(self) -> Dict[RelationType, List[str]]:
        """Initialize patterns for relationship extraction."""
        return {
            RelationType.WORKS_FOR: [
                r'(\w+(?:\s+\w+)*)\s+(?:works for|employed by|employee of)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*),?\s+(?:CEO|president|director|manager)\s+of\s+(\w+(?:\s+\w+)*)',
            ],
            RelationType.LOCATED_IN: [
                r'(\w+(?:\s+\w+)*)\s+(?:is located in|is in|located in)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*),\s+(\w+(?:\s+\w+)*)',  # City, State pattern
            ],
            RelationType.CREATED_BY: [
                r'(\w+(?:\s+\w+)*)\s+(?:created by|founded by|established by)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:created|founded|established)\s+(\w+(?:\s+\w+)*)',
            ],
            RelationType.PART_OF: [
                r'(\w+(?:\s+\w+)*)\s+(?:is part of|belongs to|member of)\s+(\w+(?:\s+\w+)*)',
            ],
        }
    
    def extract(self, text: str, existing_entities: List[EntityCandidate] = None) -> List[EntityCandidate]:
        """Extract entities using contextual analysis.
        
        Args:
            text: Input text to extract entities from
            existing_entities: Previously extracted entities to enhance with context
            
        Returns:
            List of entity candidates with contextual enhancements
        """
        if existing_entities is None:
            existing_entities = []
        
        enhanced_entities = []
        
        # Enhance existing entities with contextual information
        for entity in existing_entities:
            enhanced_entity = self._enhance_entity_with_context(entity, text)
            enhanced_entities.append(enhanced_entity)
        
        # Extract new entities based on strong contextual indicators
        new_entities = self._extract_context_driven_entities(text)
        enhanced_entities.extend(new_entities)
        
        return enhanced_entities
    
    def _enhance_entity_with_context(self, entity: EntityCandidate, text: str) -> EntityCandidate:
        """Enhance an entity with contextual information."""
        # Get expanded context
        expanded_context = self._get_expanded_context(entity, text)
        
        # Analyze context for type confirmation
        context_score = self._analyze_context_for_type(expanded_context, entity.entity_type)
        
        # Adjust confidence based on context
        new_confidence = entity.confidence * (0.7 + 0.3 * context_score)
        
        # Add contextual features
        contextual_features = entity.features.copy()
        contextual_features.update({
            'context_score': context_score,
            'expanded_context': expanded_context,
            'context_indicators': self._find_context_indicators(expanded_context, entity.entity_type)
        })
        
        return EntityCandidate(
            text=entity.text,
            entity_type=entity.entity_type,
            confidence=new_confidence,
            start_pos=entity.start_pos,
            end_pos=entity.end_pos,
            context=expanded_context,
            features=contextual_features
        )
    
    def _get_expanded_context(self, entity: EntityCandidate, text: str, window_size: int = 100) -> str:
        """Get expanded context around an entity."""
        start = max(0, entity.start_pos - window_size)
        end = min(len(text), entity.end_pos + window_size)
        return text[start:end]
    
    def _analyze_context_for_type(self, context: str, entity_type: EntityType) -> float:
        """Analyze context to confirm entity type."""
        patterns = self.context_patterns.get(entity_type, [])
        
        matches = 0
        total_patterns = len(patterns)
        
        if total_patterns == 0:
            return 0.5  # Neutral score if no patterns defined
        
        for pattern_str in patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            if pattern.search(context):
                matches += 1
        
        return matches / total_patterns
    
    def _find_context_indicators(self, context: str, entity_type: EntityType) -> List[str]:
        """Find specific context indicators for an entity type."""
        indicators = []
        patterns = self.context_patterns.get(entity_type, [])
        
        for pattern_str in patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            matches = pattern.findall(context)
            indicators.extend(matches)
        
        return indicators
    
    def _extract_context_driven_entities(self, text: str) -> List[EntityCandidate]:
        """Extract entities based on strong contextual indicators."""
        candidates = []
        
        # Look for strong contextual patterns that indicate entities
        for entity_type, patterns in self.context_patterns.items():
            for pattern_str in patterns:
                pattern = re.compile(rf'{pattern_str}\s+([A-Z][a-zA-Z\s]+)', re.IGNORECASE)
                matches = pattern.finditer(text)
                
                for match in matches:
                    entity_text = match.group(1).strip()
                    start_pos = match.start(1)
                    end_pos = match.end(1)
                    
                    # Get context
                    context_start = max(0, start_pos - 50)
                    context_end = min(len(text), end_pos + 50)
                    context = text[context_start:context_end]
                    
                    candidate = EntityCandidate(
                        text=entity_text,
                        entity_type=entity_type,
                        confidence=0.8,  # High confidence for context-driven extraction
                        start_pos=start_pos,
                        end_pos=end_pos,
                        context=context,
                        features={
                            'extraction_method': 'contextual',
                            'pattern_matched': pattern_str
                        }
                    )
                    candidates.append(candidate)
        
        return candidates


class HybridEntityExtractor:
    """Hybrid entity extraction combining multiple methods."""
    
    def __init__(self):
        """Initialize hybrid extractor with all extraction methods."""
        self.pattern_extractor = PatternBasedExtractor()
        self.statistical_extractor = StatisticalExtractor()
        self.contextual_extractor = ContextualExtractor()
        
    def extract(self, text: str, method: ExtractionMethod = ExtractionMethod.HYBRID) -> ExtractionResult:
        """Extract entities using specified method(s).
        
        Args:
            text: Input text to extract entities from
            method: Extraction method to use
            
        Returns:
            ExtractionResult with entities and metadata
        """
        import time
        start_time = time.time()
        
        all_candidates = []
        
        if method in [ExtractionMethod.PATTERN_BASED, ExtractionMethod.HYBRID]:
            pattern_candidates = self.pattern_extractor.extract(text)
            all_candidates.extend(pattern_candidates)
        
        if method in [ExtractionMethod.STATISTICAL, ExtractionMethod.HYBRID]:
            statistical_candidates = self.statistical_extractor.extract(text)
            all_candidates.extend(statistical_candidates)
        
        # Always run contextual enhancement
        enhanced_candidates = self.contextual_extractor.extract(text, all_candidates)
        
        # Merge and deduplicate candidates
        final_candidates = self._merge_and_deduplicate(enhanced_candidates)
        
        # Convert to entities
        entities = [candidate.to_entity() for candidate in final_candidates]
        
        # Extract relationships
        relationships = self._extract_relationships(final_candidates, text)
        
        processing_time = time.time() - start_time
        
        return ExtractionResult(
            entities=entities,
            relationships=relationships,
            confidence=self._calculate_overall_confidence(final_candidates),
            method_used=method,
            processing_time=processing_time,
            metadata={
                'total_candidates': len(all_candidates),
                'final_entities': len(entities),
                'relationships_found': len(relationships),
                'text_length': len(text)
            }
        )
    
    def _merge_and_deduplicate(self, candidates: List[EntityCandidate]) -> List[EntityCandidate]:
        """Merge and deduplicate entity candidates."""
        # Group candidates by text similarity
        groups = []
        for candidate in candidates:
            placed = False
            
            for group in groups:
                # Check if candidate belongs to existing group
                if self._are_similar_entities(candidate, group[0]):
                    group.append(candidate)
                    placed = True
                    break
            
            if not placed:
                groups.append([candidate])
        
        # Merge groups and select best candidate
        merged_candidates = []
        for group in groups:
            if len(group) == 1:
                merged_candidates.append(group[0])
            else:
                merged_candidate = self._merge_candidate_group(group)
                merged_candidates.append(merged_candidate)
        
        return merged_candidates
    
    def _are_similar_entities(self, candidate1: EntityCandidate, candidate2: EntityCandidate) -> bool:
        """Check if two candidates represent the same entity."""
        # Text similarity
        text1 = candidate1.text.lower().strip()
        text2 = candidate2.text.lower().strip()
        
        if text1 == text2:
            return True
        
        # Position overlap
        if (candidate1.start_pos < candidate2.end_pos and 
            candidate1.end_pos > candidate2.start_pos):
            return True
        
        # Substring relationship
        if text1 in text2 or text2 in text1:
            return True
        
        return False
    
    def _merge_candidate_group(self, group: List[EntityCandidate]) -> EntityCandidate:
        """Merge a group of similar candidates into one."""
        # Select the candidate with highest confidence
        best_candidate = max(group, key=lambda x: x.confidence)
        
        # Combine features from all candidates
        merged_features = {}
        for candidate in group:
            merged_features.update(candidate.features)
        
        # Average confidence
        avg_confidence = sum(c.confidence for c in group) / len(group)
        
        # Use the best candidate as base and enhance with merged features
        return EntityCandidate(
            text=best_candidate.text,
            entity_type=best_candidate.entity_type,
            confidence=avg_confidence,
            start_pos=best_candidate.start_pos,
            end_pos=best_candidate.end_pos,
            context=best_candidate.context,
            features=merged_features
        )
    
    def _extract_relationships(self, entities: List[EntityCandidate], text: str) -> List[Relationship]:
        """Extract relationships between entities."""
        relationships = []
        
        # Pattern-based relationship extraction
        relationship_patterns = self.contextual_extractor.entity_relationships
        
        for relation_type, patterns in relationship_patterns.items():
            for pattern_str in patterns:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                matches = pattern.finditer(text)
                
                for match in matches:
                    if len(match.groups()) >= 2:
                        entity1_text = match.group(1).strip()
                        entity2_text = match.group(2).strip()
                        
                        # Find corresponding entities
                        entity1 = self._find_entity_by_text(entities, entity1_text)
                        entity2 = self._find_entity_by_text(entities, entity2_text)
                        
                        if entity1 and entity2:
                            relationship = Relationship(
                                relationship_id="",
                                source_entity_id=entity1.to_entity().entity_id,
                                target_entity_id=entity2.to_entity().entity_id,
                                relation_type=relation_type,
                                confidence=0.7,
                                attributes={
                                    'extraction_method': 'pattern_based',
                                    'pattern_matched': pattern_str
                                }
                            )
                            relationships.append(relationship)
        
        return relationships
    
    def _find_entity_by_text(self, entities: List[EntityCandidate], text: str) -> Optional[EntityCandidate]:
        """Find an entity candidate by text match."""
        text = text.lower().strip()
        
        for entity in entities:
            if entity.text.lower().strip() == text:
                return entity
            
            # Check aliases if available
            if 'aliases' in entity.features:
                for alias in entity.features['aliases']:
                    if alias.lower().strip() == text:
                        return entity
        
        return None
    
    def _calculate_overall_confidence(self, candidates: List[EntityCandidate]) -> float:
        """Calculate overall confidence for the extraction result."""
        if not candidates:
            return 0.0
        
        return sum(c.confidence for c in candidates) / len(candidates)