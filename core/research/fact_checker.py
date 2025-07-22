"""Advanced fact verification engine for research capabilities.

This module provides comprehensive fact-checking capabilities including claim extraction,
evidence evaluation, source verification, and confidence scoring for research and
knowledge validation.
"""

import re
import math
import statistics
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import sqlite3
import json
import time
import logging
import hashlib

from .graphrag import Entity, EntityType, Relationship, RelationType, Fact, KnowledgeGraphStore
from .entity_extraction import EntityCandidate, HybridEntityExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClaimType(Enum):
    """Types of factual claims."""
    FACTUAL = "factual"
    NUMERICAL = "numerical"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    COMPARATIVE = "comparative"
    CATEGORICAL = "categorical"
    RELATIONAL = "relational"
    OPINION = "opinion"
    PREDICTION = "prediction"


class VerificationResult(Enum):
    """Results of fact verification."""
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNVERIFIABLE = "unverifiable"
    CONFLICTING_EVIDENCE = "conflicting_evidence"


class EvidenceType(Enum):
    """Types of evidence."""
    DIRECT = "direct"
    INDIRECT = "indirect"
    STATISTICAL = "statistical"
    EXPERT_OPINION = "expert_opinion"
    PRIMARY_SOURCE = "primary_source"
    SECONDARY_SOURCE = "secondary_source"
    ANECDOTAL = "anecdotal"


@dataclass
class Claim:
    """Represents a factual claim to be verified."""
    claim_id: str
    text: str
    claim_type: ClaimType
    entities: List[Entity] = field(default_factory=list)
    predicates: List[str] = field(default_factory=list)
    confidence: float = 0.0
    source: Optional[str] = None
    context: str = ""
    extracted_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.claim_id:
            self.claim_id = hashlib.sha256(self.text.encode()).hexdigest()[:16]


@dataclass
class Evidence:
    """Represents evidence for or against a claim."""
    evidence_id: str
    text: str
    evidence_type: EvidenceType
    source: str
    reliability_score: float
    relevance_score: float
    supports_claim: bool
    confidence: float = 0.0
    entities: List[Entity] = field(default_factory=list)
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.evidence_id:
            self.evidence_id = hashlib.sha256(f"{self.text}_{self.source}".encode()).hexdigest()[:16]


@dataclass
class VerificationReport:
    """Comprehensive verification report for a claim."""
    claim: Claim
    result: VerificationResult
    overall_confidence: float
    supporting_evidence: List[Evidence] = field(default_factory=list)
    contradicting_evidence: List[Evidence] = field(default_factory=list)
    neutral_evidence: List[Evidence] = field(default_factory=list)
    related_facts: List[Fact] = field(default_factory=list)
    reasoning: str = ""
    verification_time: float = field(default_factory=time.time)
    methodology: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)


class ClaimExtractor:
    """Extract factual claims from text."""
    
    def __init__(self):
        """Initialize claim extractor."""
        self.entity_extractor = HybridEntityExtractor()
        self.claim_patterns = self._initialize_claim_patterns()
        
    def _initialize_claim_patterns(self) -> Dict[ClaimType, List[str]]:
        """Initialize patterns for different types of claims."""
        return {
            ClaimType.FACTUAL: [
                r'(.+) is (.+)',
                r'(.+) has (.+)',
                r'(.+) contains (.+)',
                r'(.+) exists (.+)',
                r'(.+) happened (.+)',
                r'(.+) was (.+)',
                r'(.+) were (.+)',
            ],
            
            ClaimType.NUMERICAL: [
                r'(.+) (?:is|are|was|were|has|have) (\d+(?:\.\d+)?(?:\s*(?:million|billion|trillion|thousand|percent|%))?) (.+)',
                r'(.+) (?:costs?|worth|valued at) \$?(\d+(?:\.\d+)?(?:\s*(?:million|billion|trillion|thousand))?) (.+)',
                r'(\d+(?:\.\d+)?(?:\s*(?:million|billion|trillion|thousand|percent|%))?) (.+) (.+)',
            ],
            
            ClaimType.TEMPORAL: [
                r'(.+) (?:occurred|happened|took place|began|started|ended) (?:in|on|at|during) (.+)',
                r'(?:in|on|at|during) (.+), (.+) (?:occurred|happened|took place)',
                r'(.+) (?:before|after|since|until) (.+)',
                r'(.+) (?:was|were) (?:born|died|created|founded|established) (?:in|on|at) (.+)',
            ],
            
            ClaimType.CAUSAL: [
                r'(.+) (?:causes?|caused|leads? to|results? in|triggers?) (.+)',
                r'(.+) (?:is|was|were) (?:caused by|due to|because of|resulted from) (.+)',
                r'(?:because|since|due to) (.+), (.+)',
                r'(.+) (?:therefore|thus|consequently|as a result) (.+)',
            ],
            
            ClaimType.COMPARATIVE: [
                r'(.+) (?:is|are|was|were) (?:better|worse|faster|slower|larger|smaller|more|less) (?:than) (.+)',
                r'(.+) (?:has|have|had) (?:more|less|fewer|greater|higher|lower) (.+) (?:than) (.+)',
                r'(?:compared to|in comparison to|versus|vs) (.+), (.+)',
            ],
            
            ClaimType.RELATIONAL: [
                r'(.+) (?:works? for|employed by|member of|part of|belongs to) (.+)',
                r'(.+) (?:owns?|controls?|manages?) (.+)',
                r'(.+) (?:is located in|based in|situated in) (.+)',
                r'(.+) (?:married to|related to|connected to|associated with) (.+)',
            ],
        }
    
    def extract_claims(self, text: str, source: Optional[str] = None) -> List[Claim]:
        """Extract claims from text.
        
        Args:
            text: Input text to extract claims from
            source: Optional source identifier
            
        Returns:
            List of extracted claims
        """
        claims = []
        
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        
        for sentence in sentences:
            sentence_claims = self._extract_claims_from_sentence(sentence, source, text)
            claims.extend(sentence_claims)
        
        # Deduplicate claims
        unique_claims = self._deduplicate_claims(claims)
        
        return unique_claims
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (could be enhanced with proper NLP)
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and len(sentence.split()) >= 3:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _extract_claims_from_sentence(self, sentence: str, source: Optional[str], 
                                    full_text: str) -> List[Claim]:
        """Extract claims from a single sentence."""
        claims = []
        
        # Try each claim type pattern
        for claim_type, patterns in self.claim_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                
                for match in matches:
                    # Extract entities from the sentence
                    extraction_result = self.entity_extractor.extract(sentence)
                    
                    claim = Claim(
                        claim_id="",
                        text=sentence.strip(),
                        claim_type=claim_type,
                        entities=extraction_result.entities,
                        predicates=self._extract_predicates(sentence),
                        source=source,
                        context=self._get_sentence_context(sentence, full_text),
                        metadata={
                            'pattern_matched': pattern,
                            'match_groups': match.groups(),
                            'extraction_confidence': extraction_result.confidence
                        }
                    )
                    
                    # Calculate initial confidence
                    claim.confidence = self._calculate_claim_confidence(claim, extraction_result)
                    
                    if claim.confidence > 0.3:  # Threshold for acceptance
                        claims.append(claim)
        
        return claims
    
    def _extract_predicates(self, sentence: str) -> List[str]:
        """Extract predicates (verbs/actions) from sentence."""
        # Simple predicate extraction (could be enhanced with POS tagging)
        predicate_patterns = [
            r'\b(?:is|are|was|were|has|have|had|do|does|did|will|would|can|could|should|might|may)\b',
            r'\b\w+(?:ed|ing|es|s)\b',  # Past/present participles and verb forms
        ]
        
        predicates = []
        for pattern in predicate_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            predicates.extend(matches)
        
        return list(set(predicates))
    
    def _get_sentence_context(self, sentence: str, full_text: str, window: int = 100) -> str:
        """Get context around a sentence."""
        start_pos = full_text.find(sentence)
        if start_pos == -1:
            return sentence
        
        context_start = max(0, start_pos - window)
        context_end = min(len(full_text), start_pos + len(sentence) + window)
        
        return full_text[context_start:context_end]
    
    def _calculate_claim_confidence(self, claim: Claim, extraction_result) -> float:
        """Calculate confidence score for a claim."""
        confidence = 0.5  # Base confidence
        
        # Entity extraction quality
        if extraction_result.entities:
            avg_entity_confidence = sum(e.confidence for e in extraction_result.entities) / len(extraction_result.entities)
            confidence += avg_entity_confidence * 0.3
        
        # Claim type specificity
        type_bonuses = {
            ClaimType.FACTUAL: 0.1,
            ClaimType.NUMERICAL: 0.2,
            ClaimType.TEMPORAL: 0.15,
            ClaimType.CAUSAL: 0.1,
            ClaimType.COMPARATIVE: 0.1,
            ClaimType.RELATIONAL: 0.15
        }
        confidence += type_bonuses.get(claim.claim_type, 0.0)
        
        # Sentence length and complexity
        word_count = len(claim.text.split())
        if 5 <= word_count <= 25:
            confidence += 0.1
        elif word_count > 30:
            confidence -= 0.1
        
        # Presence of specific indicators
        factual_indicators = ['according to', 'research shows', 'studies indicate', 'data suggests']
        if any(indicator in claim.text.lower() for indicator in factual_indicators):
            confidence += 0.1
        
        # Uncertainty indicators (reduce confidence)
        uncertainty_indicators = ['might', 'could', 'possibly', 'perhaps', 'maybe', 'allegedly']
        if any(indicator in claim.text.lower() for indicator in uncertainty_indicators):
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def _deduplicate_claims(self, claims: List[Claim]) -> List[Claim]:
        """Remove duplicate or very similar claims."""
        unique_claims = []
        seen_texts = set()
        
        for claim in claims:
            # Normalize text for comparison
            normalized_text = re.sub(r'\s+', ' ', claim.text.lower().strip())
            
            if normalized_text not in seen_texts:
                unique_claims.append(claim)
                seen_texts.add(normalized_text)
        
        return unique_claims


class EvidenceCollector:
    """Collect and evaluate evidence for claims."""
    
    def __init__(self, knowledge_store: KnowledgeGraphStore):
        """Initialize evidence collector.
        
        Args:
            knowledge_store: Knowledge graph store to search for evidence
        """
        self.knowledge_store = knowledge_store
        self.source_reliability = self._initialize_source_reliability()
        
    def _initialize_source_reliability(self) -> Dict[str, float]:
        """Initialize source reliability scores."""
        return {
            'academic_journal': 0.9,
            'peer_reviewed': 0.85,
            'government_data': 0.8,
            'news_organization': 0.7,
            'encyclopedia': 0.75,
            'official_website': 0.7,
            'book': 0.65,
            'blog': 0.3,
            'social_media': 0.2,
            'unknown': 0.5
        }
    
    def collect_evidence(self, claim: Claim, max_evidence: int = 10) -> List[Evidence]:
        """Collect evidence for a claim.
        
        Args:
            claim: Claim to find evidence for
            max_evidence: Maximum number of evidence items to collect
            
        Returns:
            List of evidence items
        """
        evidence_items = []
        
        # Search knowledge graph for related facts
        kg_evidence = self._search_knowledge_graph(claim)
        evidence_items.extend(kg_evidence)
        
        # Search for related entities and relationships
        entity_evidence = self._search_entity_relationships(claim)
        evidence_items.extend(entity_evidence)
        
        # Rank evidence by relevance and reliability
        ranked_evidence = self._rank_evidence(evidence_items, claim)
        
        return ranked_evidence[:max_evidence]
    
    def _search_knowledge_graph(self, claim: Claim) -> List[Evidence]:
        """Search knowledge graph for evidence."""
        evidence_items = []
        
        # Get all facts from knowledge store
        cursor = self.knowledge_store.conn.execute("""
            SELECT fact_id, claim, confidence, evidence, sources
            FROM facts
            ORDER BY confidence DESC
            LIMIT 1000
        """)
        
        for row in cursor.fetchall():
            fact_id, fact_claim, confidence, evidence_json, sources_json = row
            
            # Calculate relevance to claim
            relevance = self._calculate_text_similarity(claim.text, fact_claim)
            
            if relevance > 0.3:  # Threshold for relevance
                evidence_list = json.loads(evidence_json or '[]')
                sources_list = json.loads(sources_json or '[]')
                
                for i, evidence_text in enumerate(evidence_list):
                    source = sources_list[i] if i < len(sources_list) else "unknown"
                    
                    evidence = Evidence(
                        evidence_id="",
                        text=evidence_text,
                        evidence_type=EvidenceType.DIRECT,
                        source=source,
                        reliability_score=self._calculate_source_reliability(source),
                        relevance_score=relevance,
                        supports_claim=self._determine_support(claim.text, evidence_text),
                        confidence=confidence,
                        metadata={
                            'fact_id': fact_id,
                            'source_type': 'knowledge_graph'
                        }
                    )
                    evidence_items.append(evidence)
        
        return evidence_items
    
    def _search_entity_relationships(self, claim: Claim) -> List[Evidence]:
        """Search for evidence based on entity relationships."""
        evidence_items = []
        
        for entity in claim.entities:
            # Get relationships for this entity
            relationships = self.knowledge_store.get_relationships(entity.entity_id)
            
            for rel in relationships:
                # Get the connected entity
                connected_entity_id = (rel.target_entity_id 
                                     if rel.source_entity_id == entity.entity_id 
                                     else rel.source_entity_id)
                connected_entity = self.knowledge_store.get_entity(connected_entity_id)
                
                if connected_entity:
                    # Create evidence from relationship
                    evidence_text = f"{entity.name} {rel.relation_type.value.replace('_', ' ')} {connected_entity.name}"
                    
                    relevance = self._calculate_text_similarity(claim.text, evidence_text)
                    
                    if relevance > 0.2:
                        evidence = Evidence(
                            evidence_id="",
                            text=evidence_text,
                            evidence_type=EvidenceType.INDIRECT,
                            source="knowledge_graph_relationships",
                            reliability_score=0.7,
                            relevance_score=relevance,
                            supports_claim=self._determine_support(claim.text, evidence_text),
                            confidence=rel.confidence,
                            entities=[entity, connected_entity],
                            metadata={
                                'relationship_id': rel.relationship_id,
                                'relation_type': rel.relation_type.value
                            }
                        )
                        evidence_items.append(evidence)
        
        return evidence_items
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        # Tokenize and clean
        tokens1 = set(re.findall(r'\w+', text1.lower()))
        tokens2 = set(re.findall(r'\w+', text2.lower()))
        
        # Remove stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were'
        }
        
        tokens1 -= stop_words
        tokens2 -= stop_words
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Jaccard similarity
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_source_reliability(self, source: str) -> float:
        """Calculate reliability score for a source."""
        source_lower = source.lower()
        
        # Check for known source types
        for source_type, reliability in self.source_reliability.items():
            if source_type.replace('_', ' ') in source_lower:
                return reliability
        
        # Heuristic-based reliability
        reliability = 0.5  # Default
        
        # Academic indicators
        if any(indicator in source_lower for indicator in ['edu', 'university', 'journal', 'research']):
            reliability += 0.2
        
        # Government indicators
        if any(indicator in source_lower for indicator in ['gov', 'government', 'official']):
            reliability += 0.15
        
        # News indicators
        if any(indicator in source_lower for indicator in ['news', 'times', 'post', 'reuters', 'bbc']):
            reliability += 0.1
        
        # Unreliable indicators
        if any(indicator in source_lower for indicator in ['blog', 'personal', 'opinion', 'social']):
            reliability -= 0.2
        
        return max(0.1, min(1.0, reliability))
    
    def _determine_support(self, claim_text: str, evidence_text: str) -> bool:
        """Determine if evidence supports or contradicts the claim."""
        # Simple heuristic - could be enhanced with NLP
        
        # Look for contradiction indicators
        contradiction_indicators = ['not', 'never', 'no', 'false', 'incorrect', 'wrong', 'contrary']
        
        claim_lower = claim_text.lower()
        evidence_lower = evidence_text.lower()
        
        # Check if evidence contains contradiction indicators relative to claim
        shared_tokens = set(re.findall(r'\w+', claim_lower)) & set(re.findall(r'\w+', evidence_lower))
        
        if shared_tokens and any(indicator in evidence_lower for indicator in contradiction_indicators):
            # If there are shared tokens and contradiction indicators, likely contradicts
            return False
        
        # If high similarity and no contradiction indicators, likely supports
        similarity = self._calculate_text_similarity(claim_text, evidence_text)
        return similarity > 0.3
    
    def _rank_evidence(self, evidence_items: List[Evidence], claim: Claim) -> List[Evidence]:
        """Rank evidence by relevance and reliability."""
        
        def calculate_score(evidence: Evidence) -> float:
            return (
                evidence.relevance_score * 0.4 +
                evidence.reliability_score * 0.3 +
                evidence.confidence * 0.3
            )
        
        return sorted(evidence_items, key=calculate_score, reverse=True)


class FactVerifier:
    """Main fact verification engine."""
    
    def __init__(self, knowledge_store: KnowledgeGraphStore):
        """Initialize fact verifier.
        
        Args:
            knowledge_store: Knowledge graph store for evidence search
        """
        self.knowledge_store = knowledge_store
        self.claim_extractor = ClaimExtractor()
        self.evidence_collector = EvidenceCollector(knowledge_store)
        
    def verify_text(self, text: str, source: Optional[str] = None) -> List[VerificationReport]:
        """Verify all claims in a text.
        
        Args:
            text: Text to verify
            source: Optional source identifier
            
        Returns:
            List of verification reports
        """
        # Extract claims
        claims = self.claim_extractor.extract_claims(text, source)
        
        # Verify each claim
        reports = []
        for claim in claims:
            report = self.verify_claim(claim)
            reports.append(report)
        
        return reports
    
    def verify_claim(self, claim: Claim) -> VerificationReport:
        """Verify a single claim.
        
        Args:
            claim: Claim to verify
            
        Returns:
            Verification report
        """
        start_time = time.time()
        
        # Collect evidence
        evidence_items = self.evidence_collector.collect_evidence(claim)
        
        # Categorize evidence
        supporting_evidence = [e for e in evidence_items if e.supports_claim]
        contradicting_evidence = [e for e in evidence_items if not e.supports_claim]
        
        # Calculate verification result
        result, confidence, reasoning = self._calculate_verification_result(
            claim, supporting_evidence, contradicting_evidence
        )
        
        # Find related facts
        related_facts = self._find_related_facts(claim)
        
        # Create report
        report = VerificationReport(
            claim=claim,
            result=result,
            overall_confidence=confidence,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            related_facts=related_facts,
            reasoning=reasoning,
            verification_time=time.time() - start_time,
            methodology={
                'evidence_sources': len(set(e.source for e in evidence_items)),
                'total_evidence': len(evidence_items),
                'kg_facts_searched': self._count_kg_facts(),
                'entities_analyzed': len(claim.entities)
            },
            limitations=self._identify_limitations(claim, evidence_items)
        )
        
        return report
    
    def _calculate_verification_result(self, claim: Claim, 
                                     supporting_evidence: List[Evidence],
                                     contradicting_evidence: List[Evidence]) -> Tuple[VerificationResult, float, str]:
        """Calculate the verification result for a claim."""
        
        if not supporting_evidence and not contradicting_evidence:
            return VerificationResult.INSUFFICIENT_EVIDENCE, 0.1, "No relevant evidence found in knowledge base."
        
        # Calculate support and contradiction scores
        support_score = 0.0
        contradiction_score = 0.0
        
        if supporting_evidence:
            support_weights = [e.relevance_score * e.reliability_score * e.confidence for e in supporting_evidence]
            support_score = sum(support_weights) / len(support_weights)
        
        if contradicting_evidence:
            contradiction_weights = [e.relevance_score * e.reliability_score * e.confidence for e in contradicting_evidence]
            contradiction_score = sum(contradiction_weights) / len(contradiction_weights)
        
        # Determine result based on scores
        confidence_threshold = 0.6
        
        if support_score > contradiction_score:
            if support_score >= confidence_threshold:
                result = VerificationResult.SUPPORTED
                confidence = support_score
                reasoning = f"Claim is supported by {len(supporting_evidence)} pieces of evidence with high confidence."
            else:
                result = VerificationResult.PARTIALLY_SUPPORTED
                confidence = support_score
                reasoning = f"Claim has some supporting evidence but confidence is moderate."
        
        elif contradiction_score > support_score:
            if contradiction_score >= confidence_threshold:
                result = VerificationResult.CONTRADICTED
                confidence = contradiction_score
                reasoning = f"Claim is contradicted by {len(contradicting_evidence)} pieces of evidence."
            else:
                result = VerificationResult.CONFLICTING_EVIDENCE
                confidence = 0.5
                reasoning = f"Evidence shows conflicting information about this claim."
        
        else:
            # Scores are very close
            result = VerificationResult.CONFLICTING_EVIDENCE
            confidence = 0.5
            reasoning = "Evidence is mixed with equal support and contradiction."
        
        # Special cases
        if len(supporting_evidence) == 0 and len(contradicting_evidence) == 0:
            result = VerificationResult.INSUFFICIENT_EVIDENCE
            confidence = 0.1
        elif claim.claim_type == ClaimType.OPINION:
            result = VerificationResult.UNVERIFIABLE
            confidence = 0.0
            reasoning = "Claim appears to be an opinion and cannot be factually verified."
        
        return result, confidence, reasoning
    
    def _find_related_facts(self, claim: Claim) -> List[Fact]:
        """Find facts related to the claim."""
        related_facts = []
        
        # Search for facts mentioning claim entities
        for entity in claim.entities:
            cursor = self.knowledge_store.conn.execute("""
                SELECT fact_id, claim, entities, confidence, evidence, sources
                FROM facts
                WHERE entities LIKE ?
                ORDER BY confidence DESC
                LIMIT 5
            """, (f'%{entity.entity_id}%',))
            
            for row in cursor.fetchall():
                fact = Fact(
                    fact_id=row[0],
                    claim=row[1],
                    entities=json.loads(row[2] or '[]'),
                    confidence=row[3],
                    evidence=json.loads(row[4] or '[]'),
                    sources=json.loads(row[5] or '[]')
                )
                related_facts.append(fact)
        
        # Deduplicate and sort
        unique_facts = {fact.fact_id: fact for fact in related_facts}.values()
        return sorted(unique_facts, key=lambda x: x.confidence, reverse=True)[:10]
    
    def _count_kg_facts(self) -> int:
        """Count total facts in knowledge graph."""
        cursor = self.knowledge_store.conn.execute("SELECT COUNT(*) FROM facts")
        return cursor.fetchone()[0]
    
    def _identify_limitations(self, claim: Claim, evidence_items: List[Evidence]) -> List[str]:
        """Identify limitations in the verification process."""
        limitations = []
        
        if len(evidence_items) < 3:
            limitations.append("Limited evidence available for comprehensive verification.")
        
        if not claim.entities:
            limitations.append("No entities identified in claim, limiting structured analysis.")
        
        source_diversity = len(set(e.source for e in evidence_items))
        if source_diversity < 2:
            limitations.append("Evidence comes from limited source diversity.")
        
        if claim.claim_type == ClaimType.PREDICTION:
            limitations.append("Claim involves future predictions which cannot be definitively verified.")
        
        if claim.claim_type == ClaimType.NUMERICAL:
            numerical_evidence = [e for e in evidence_items if any(char.isdigit() for char in e.text)]
            if not numerical_evidence:
                limitations.append("Numerical claim lacks quantitative evidence for verification.")
        
        avg_reliability = statistics.mean([e.reliability_score for e in evidence_items]) if evidence_items else 0
        if avg_reliability < 0.6:
            limitations.append("Evidence sources have relatively low reliability scores.")
        
        return limitations
    
    def create_fact_from_verified_claim(self, report: VerificationReport) -> Optional[Fact]:
        """Create a fact from a verified claim if confidence is high enough.
        
        Args:
            report: Verification report
            
        Returns:
            Fact object if claim is sufficiently verified, None otherwise
        """
        if (report.result == VerificationResult.SUPPORTED and 
            report.overall_confidence >= 0.7):
            
            # Extract evidence text
            evidence_texts = [e.text for e in report.supporting_evidence]
            sources = list(set(e.source for e in report.supporting_evidence))
            
            # Extract entity IDs
            entity_ids = [entity.entity_id for entity in report.claim.entities]
            
            fact = Fact(
                fact_id="",
                claim=report.claim.text,
                entities=entity_ids,
                confidence=report.overall_confidence,
                evidence=evidence_texts,
                sources=sources,
                contradictions=[],  # Could be populated from contradicting evidence
                metadata={
                    'verification_report_id': f"report_{int(report.verification_time)}",
                    'claim_type': report.claim.claim_type.value,
                    'verification_confidence': report.overall_confidence,
                    'evidence_count': len(report.supporting_evidence)
                }
            )
            
            return fact
        
        return None
    
    def batch_verify(self, claims: List[Claim], batch_size: int = 10) -> List[VerificationReport]:
        """Verify multiple claims in batches.
        
        Args:
            claims: List of claims to verify
            batch_size: Number of claims to process in each batch
            
        Returns:
            List of verification reports
        """
        all_reports = []
        
        for i in range(0, len(claims), batch_size):
            batch = claims[i:i + batch_size]
            batch_reports = []
            
            for claim in batch:
                report = self.verify_claim(claim)
                batch_reports.append(report)
            
            all_reports.extend(batch_reports)
            
            # Log progress
            logger.info(f"Verified batch {i // batch_size + 1}, {len(all_reports)} claims processed")
        
        return all_reports