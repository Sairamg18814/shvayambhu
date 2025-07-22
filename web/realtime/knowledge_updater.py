"""
Real-Time Knowledge Updater

Updates the knowledge graph with real-time information from web sources,
performs fact verification, and manages temporal knowledge evolution.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import hashlib
from pathlib import Path
import re
import numpy as np
from collections import defaultdict, deque
import networkx as nx
from enum import Enum

from web.connectivity.web_monitor import WebContent, WebMonitor

logger = logging.getLogger(__name__)


class FactStatus(Enum):
    """Status of facts in the knowledge graph"""
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    DISPUTED = "disputed"
    OUTDATED = "outdated"
    CONFLICTING = "conflicting"


@dataclass
class KnowledgeFact:
    """Represents a fact in the knowledge graph"""
    id: str
    subject: str
    predicate: str
    object: str
    confidence: float
    sources: List[str]
    created_at: datetime
    updated_at: datetime
    status: FactStatus
    evidence: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    temporal_validity: Optional[Tuple[datetime, datetime]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeUpdate:
    """Represents an update to the knowledge graph"""
    id: str
    update_type: str  # add, modify, delete, verify, dispute
    target_fact_id: str
    new_data: Dict[str, Any]
    source_url: str
    confidence: float
    timestamp: datetime
    processed: bool = False


@dataclass
class CredibilityScore:
    """Source credibility assessment"""
    source: str
    overall_score: float
    accuracy_history: List[float]
    bias_score: float
    expertise_score: float
    recency_score: float
    update_frequency: float
    last_updated: datetime


class SourceCredibilityAssessor:
    """Assesses credibility of information sources"""
    
    def __init__(self):
        # Pre-defined credibility scores for known sources
        self.known_sources = {
            'arxiv.org': {'base_credibility': 0.95, 'expertise': 0.98, 'bias': 0.1},
            'nature.com': {'base_credibility': 0.95, 'expertise': 0.98, 'bias': 0.15},
            'science.org': {'base_credibility': 0.94, 'expertise': 0.97, 'bias': 0.15},
            'ieee.org': {'base_credibility': 0.93, 'expertise': 0.96, 'bias': 0.12},
            'openai.com': {'base_credibility': 0.85, 'expertise': 0.90, 'bias': 0.25},
            'deepmind.com': {'base_credibility': 0.85, 'expertise': 0.90, 'bias': 0.25},
            'mit.edu': {'base_credibility': 0.92, 'expertise': 0.95, 'bias': 0.18},
            'stanford.edu': {'base_credibility': 0.92, 'expertise': 0.95, 'bias': 0.18},
            'wikipedia.org': {'base_credibility': 0.75, 'expertise': 0.80, 'bias': 0.30},
            'reuters.com': {'base_credibility': 0.80, 'expertise': 0.75, 'bias': 0.25},
            'bbc.co.uk': {'base_credibility': 0.78, 'expertise': 0.70, 'bias': 0.30},
            'cnn.com': {'base_credibility': 0.65, 'expertise': 0.65, 'bias': 0.45},
            'techcrunch.com': {'base_credibility': 0.70, 'expertise': 0.75, 'bias': 0.35}
        }
        
        # Track source performance over time
        self.source_history: Dict[str, CredibilityScore] = {}
    
    def assess_source_credibility(self, source: str, content: Optional[WebContent] = None) -> CredibilityScore:
        """Assess credibility of a source"""
        
        # Get base scores if known source
        base_scores = self.known_sources.get(source, {
            'base_credibility': 0.5,
            'expertise': 0.5,
            'bias': 0.5
        })
        
        # Check existing history
        if source in self.source_history:
            history = self.source_history[source]
            
            # Update recency score
            days_since_update = (datetime.now() - history.last_updated).days
            recency_penalty = min(0.1, days_since_update * 0.01)
            
            # Calculate overall score with history
            overall_score = (
                base_scores['base_credibility'] * 0.4 +
                np.mean(history.accuracy_history[-10:]) * 0.3 +  # Recent accuracy
                (1 - history.bias_score) * 0.2 +  # Lower bias = higher credibility
                history.expertise_score * 0.1 -
                recency_penalty
            )
            
            history.overall_score = max(0.0, min(1.0, overall_score))
            history.last_updated = datetime.now()
            
            return history
        else:
            # Create new credibility score
            overall_score = (
                base_scores['base_credibility'] * 0.6 +
                base_scores['expertise'] * 0.3 +
                (1 - base_scores['bias']) * 0.1
            )
            
            credibility = CredibilityScore(
                source=source,
                overall_score=overall_score,
                accuracy_history=[base_scores['base_credibility']],
                bias_score=base_scores['bias'],
                expertise_score=base_scores['expertise'],
                recency_score=1.0,
                update_frequency=0.0,
                last_updated=datetime.now()
            )
            
            self.source_history[source] = credibility
            return credibility
    
    def update_source_accuracy(self, source: str, accuracy: float):
        """Update source accuracy based on fact verification"""
        
        if source in self.source_history:
            self.source_history[source].accuracy_history.append(accuracy)
            
            # Keep only recent history
            if len(self.source_history[source].accuracy_history) > 50:
                self.source_history[source].accuracy_history = self.source_history[source].accuracy_history[-50:]
            
            # Recalculate overall score
            credibility = self.assess_source_credibility(source)
    
    def get_top_sources(self, min_credibility: float = 0.7) -> List[Tuple[str, float]]:
        """Get sources above credibility threshold"""
        
        top_sources = [
            (source, score.overall_score)
            for source, score in self.source_history.items()
            if score.overall_score >= min_credibility
        ]
        
        top_sources.sort(key=lambda x: x[1], reverse=True)
        return top_sources


class FactExtractor:
    """Extracts factual information from web content"""
    
    def __init__(self):
        # Patterns for extracting facts
        self.fact_patterns = [
            # "X is Y" patterns
            (r'(\w+(?:\s+\w+)*)\s+is\s+(?:a|an|the)?\s*([^.!?]+)', 'is'),
            # "X has Y" patterns  
            (r'(\w+(?:\s+\w+)*)\s+has\s+([^.!?]+)', 'has'),
            # "X was Y" patterns
            (r'(\w+(?:\s+\w+)*)\s+was\s+(?:a|an|the)?\s*([^.!?]+)', 'was'),
            # "X developed Y" patterns
            (r'(\w+(?:\s+\w+)*)\s+developed\s+([^.!?]+)', 'developed'),
            # "X released Y" patterns
            (r'(\w+(?:\s+\w+)*)\s+released\s+([^.!?]+)', 'released'),
            # "X announced Y" patterns
            (r'(\w+(?:\s+\w+)*)\s+announced\s+([^.!?]+)', 'announced')
        ]
        
        # Entity types to focus on
        self.entity_types = {
            'companies': ['OpenAI', 'Google', 'Microsoft', 'Meta', 'DeepMind', 'Anthropic'],
            'technologies': ['AI', 'machine learning', 'neural networks', 'transformer', 'LLM'],
            'products': ['GPT', 'ChatGPT', 'Claude', 'BERT', 'LaMDA', 'PaLM'],
            'research_areas': ['consciousness', 'reasoning', 'alignment', 'safety', 'ethics']
        }
    
    def extract_facts(self, content: WebContent) -> List[KnowledgeFact]:
        """Extract facts from web content"""
        
        facts = []
        text = content.content
        
        for pattern, relation in self.fact_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                subject = match.group(1).strip()
                object_text = match.group(2).strip()
                
                # Filter out very short or very long extractions
                if len(subject) < 2 or len(object_text) < 3:
                    continue
                if len(subject) > 100 or len(object_text) > 200:
                    continue
                
                # Check if subject matches known entities
                relevance_score = self._calculate_entity_relevance(subject, object_text)
                
                if relevance_score < 0.3:  # Skip low-relevance facts
                    continue
                
                # Create fact
                fact_id = self._generate_fact_id(subject, relation, object_text)
                
                fact = KnowledgeFact(
                    id=fact_id,
                    subject=subject,
                    predicate=relation,
                    object=object_text,
                    confidence=min(relevance_score, content.relevance_score),
                    sources=[content.url],
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    status=FactStatus.UNVERIFIED,
                    evidence=[f"Extracted from: {content.title}"],
                    metadata={
                        'source_domain': content.source,
                        'extraction_pattern': pattern,
                        'content_type': content.content_type
                    }
                )
                
                facts.append(fact)
        
        logger.debug(f"Extracted {len(facts)} facts from {content.url}")
        return facts
    
    def _calculate_entity_relevance(self, subject: str, object_text: str) -> float:
        """Calculate relevance of extracted entity"""
        
        relevance = 0.0
        subject_lower = subject.lower()
        object_lower = object_text.lower()
        
        # Check against known entities
        for entity_type, entities in self.entity_types.items():
            for entity in entities:
                if entity.lower() in subject_lower or entity.lower() in object_lower:
                    relevance += 0.3
        
        # Check for AI/tech keywords
        tech_keywords = [
            'ai', 'artificial intelligence', 'machine learning', 'neural', 'algorithm',
            'model', 'research', 'technology', 'innovation', 'breakthrough'
        ]
        
        for keyword in tech_keywords:
            if keyword in subject_lower or keyword in object_lower:
                relevance += 0.2
        
        return min(1.0, relevance)
    
    def _generate_fact_id(self, subject: str, predicate: str, object_text: str) -> str:
        """Generate unique ID for fact"""
        
        fact_string = f"{subject}|{predicate}|{object_text}".lower()
        return hashlib.md5(fact_string.encode()).hexdigest()


class ConflictResolver:
    """Resolves conflicts between contradictory facts"""
    
    def __init__(self, credibility_assessor: SourceCredibilityAssessor):
        self.credibility_assessor = credibility_assessor
    
    def detect_conflicts(self, new_fact: KnowledgeFact, existing_facts: List[KnowledgeFact]) -> List[KnowledgeFact]:
        """Detect conflicts with existing facts"""
        
        conflicts = []
        
        for existing_fact in existing_facts:
            # Same subject and predicate but different object
            if (existing_fact.subject.lower() == new_fact.subject.lower() and
                existing_fact.predicate == new_fact.predicate and
                existing_fact.object.lower() != new_fact.object.lower()):
                
                conflicts.append(existing_fact)
            
            # Temporal conflicts (was vs is)
            elif (existing_fact.subject.lower() == new_fact.subject.lower() and
                  self._are_temporal_conflicts(existing_fact.predicate, new_fact.predicate)):
                
                conflicts.append(existing_fact)
        
        return conflicts
    
    def resolve_conflict(self, new_fact: KnowledgeFact, conflicting_fact: KnowledgeFact) -> Tuple[KnowledgeFact, str]:
        """Resolve conflict between facts"""
        
        # Calculate credibility scores for sources
        new_fact_credibility = self._calculate_fact_credibility(new_fact)
        existing_fact_credibility = self._calculate_fact_credibility(conflicting_fact)
        
        # Consider temporal information
        new_fact_recency = self._calculate_recency_score(new_fact)
        existing_fact_recency = self._calculate_recency_score(conflicting_fact)
        
        # Combined score
        new_fact_score = new_fact_credibility * 0.7 + new_fact_recency * 0.3
        existing_fact_score = existing_fact_credibility * 0.7 + existing_fact_recency * 0.3
        
        resolution_reason = ""
        
        if new_fact_score > existing_fact_score + 0.1:  # Significant difference threshold
            # New fact wins
            winning_fact = new_fact
            winning_fact.contradictions.append(conflicting_fact.id)
            resolution_reason = f"Higher credibility score: {new_fact_score:.3f} vs {existing_fact_score:.3f}"
        else:
            # Existing fact wins, but mark as disputed
            winning_fact = conflicting_fact
            winning_fact.status = FactStatus.DISPUTED
            winning_fact.contradictions.append(new_fact.id)
            resolution_reason = f"Existing fact maintained: {existing_fact_score:.3f} vs {new_fact_score:.3f}"
        
        return winning_fact, resolution_reason
    
    def _are_temporal_conflicts(self, pred1: str, pred2: str) -> bool:
        """Check if predicates represent temporal conflicts"""
        
        temporal_pairs = [
            ('is', 'was'),
            ('has', 'had'),
            ('will', 'did')
        ]
        
        for present, past in temporal_pairs:
            if (pred1 == present and pred2 == past) or (pred1 == past and pred2 == present):
                return True
        
        return False
    
    def _calculate_fact_credibility(self, fact: KnowledgeFact) -> float:
        """Calculate overall credibility of a fact"""
        
        if not fact.sources:
            return fact.confidence
        
        # Average source credibility
        source_scores = []
        for source_url in fact.sources:
            source_domain = self._extract_domain(source_url)
            credibility = self.credibility_assessor.assess_source_credibility(source_domain)
            source_scores.append(credibility.overall_score)
        
        avg_source_credibility = np.mean(source_scores)
        
        # Combine with fact confidence
        combined_credibility = (avg_source_credibility * 0.6 + fact.confidence * 0.4)
        
        return combined_credibility
    
    def _calculate_recency_score(self, fact: KnowledgeFact) -> float:
        """Calculate recency score for fact"""
        
        hours_since_creation = (datetime.now() - fact.created_at).total_seconds() / 3600
        
        # Decay function - more recent facts score higher
        recency_score = max(0.1, 1.0 - (hours_since_creation / 168))  # 1 week half-life
        
        return recency_score
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        
        import re
        match = re.search(r'https?://([^/]+)', url)
        return match.group(1) if match else url


class KnowledgeGraph:
    """Manages the knowledge graph structure"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.facts: Dict[str, KnowledgeFact] = {}
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)  # entity -> fact_ids
        self.predicate_index: Dict[str, Set[str]] = defaultdict(set)  # predicate -> fact_ids
        
    def add_fact(self, fact: KnowledgeFact) -> bool:
        """Add fact to knowledge graph"""
        
        if fact.id in self.facts:
            return False  # Fact already exists
        
        # Add to facts storage
        self.facts[fact.id] = fact
        
        # Add to graph
        self.graph.add_edge(
            fact.subject,
            fact.object,
            key=fact.predicate,
            fact_id=fact.id,
            confidence=fact.confidence,
            created_at=fact.created_at
        )
        
        # Update indices
        self.entity_index[fact.subject.lower()].add(fact.id)
        self.entity_index[fact.object.lower()].add(fact.id)
        self.predicate_index[fact.predicate].add(fact.id)
        
        return True
    
    def update_fact(self, fact_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing fact"""
        
        if fact_id not in self.facts:
            return False
        
        fact = self.facts[fact_id]
        
        # Update fact attributes
        for key, value in updates.items():
            if hasattr(fact, key):
                setattr(fact, key, value)
        
        fact.updated_at = datetime.now()
        
        # Update graph if structural changes
        if 'confidence' in updates:
            # Update edge attributes
            for u, v, data in self.graph.edges(data=True):
                if data.get('fact_id') == fact_id:
                    data['confidence'] = fact.confidence
                    break
        
        return True
    
    def get_facts_by_entity(self, entity: str) -> List[KnowledgeFact]:
        """Get all facts involving an entity"""
        
        entity_lower = entity.lower()
        fact_ids = self.entity_index.get(entity_lower, set())
        
        return [self.facts[fact_id] for fact_id in fact_ids if fact_id in self.facts]
    
    def get_facts_by_predicate(self, predicate: str) -> List[KnowledgeFact]:
        """Get all facts with specific predicate"""
        
        fact_ids = self.predicate_index.get(predicate, set())
        return [self.facts[fact_id] for fact_id in fact_ids if fact_id in self.facts]
    
    def find_related_entities(self, entity: str, max_hops: int = 2) -> List[Tuple[str, int]]:
        """Find entities related to given entity within max hops"""
        
        if entity not in self.graph:
            return []
        
        # BFS to find related entities
        related = []
        visited = set([entity])
        queue = deque([(entity, 0)])
        
        while queue:
            current_entity, hops = queue.popleft()
            
            if hops >= max_hops:
                continue
            
            # Get neighbors
            for neighbor in self.graph.neighbors(current_entity):
                if neighbor not in visited:
                    visited.add(neighbor)
                    related.append((neighbor, hops + 1))
                    queue.append((neighbor, hops + 1))
        
        return related
    
    def get_conflicting_facts(self, fact: KnowledgeFact) -> List[KnowledgeFact]:
        """Find facts that conflict with given fact"""
        
        # Get facts with same subject and predicate
        subject_facts = self.get_facts_by_entity(fact.subject)
        
        conflicts = []
        for existing_fact in subject_facts:
            if (existing_fact.predicate == fact.predicate and
                existing_fact.object.lower() != fact.object.lower()):
                conflicts.append(existing_fact)
        
        return conflicts
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        
        total_facts = len(self.facts)
        total_entities = len(self.entity_index)
        total_predicates = len(self.predicate_index)
        
        # Status distribution
        status_counts = defaultdict(int)
        confidence_scores = []
        
        for fact in self.facts.values():
            status_counts[fact.status.value] += 1
            confidence_scores.append(fact.confidence)
        
        # Graph metrics
        if self.graph.number_of_nodes() > 0:
            avg_degree = sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
            connected_components = nx.number_weakly_connected_components(self.graph)
        else:
            avg_degree = 0
            connected_components = 0
        
        return {
            'total_facts': total_facts,
            'total_entities': total_entities,
            'total_predicates': total_predicates,
            'status_distribution': dict(status_counts),
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'graph_nodes': self.graph.number_of_nodes(),
            'graph_edges': self.graph.number_of_edges(),
            'average_degree': avg_degree,
            'connected_components': connected_components
        }


class RealTimeKnowledgeUpdater:
    """Main orchestrator for real-time knowledge updates"""
    
    def __init__(self, web_monitor: WebMonitor):
        self.web_monitor = web_monitor
        self.knowledge_graph = KnowledgeGraph()
        self.fact_extractor = FactExtractor()
        self.credibility_assessor = SourceCredibilityAssessor()
        self.conflict_resolver = ConflictResolver(self.credibility_assessor)
        
        # Update queues
        self.update_queue: asyncio.Queue = asyncio.Queue()
        self.processing_stats = {
            'updates_processed': 0,
            'facts_added': 0,
            'conflicts_resolved': 0,
            'last_update_time': None,
            'processing_errors': 0
        }
        
        # Subscribe to web monitor updates
        self.web_monitor.add_update_callback(self._handle_web_content_update)
        
        self.is_running = False
    
    async def start_updating(self):
        """Start real-time knowledge updating"""
        
        self.is_running = True
        logger.info("Starting real-time knowledge updates")
        
        # Start processing task
        processing_task = asyncio.create_task(self._process_update_queue())
        
        try:
            await processing_task
        except Exception as e:
            logger.error(f"Error in knowledge updating: {e}")
            self.processing_stats['processing_errors'] += 1
    
    async def stop_updating(self):
        """Stop knowledge updating"""
        
        self.is_running = False
        logger.info("Knowledge updating stopped")
    
    async def _handle_web_content_update(self, content_batch: List[WebContent]):
        """Handle new web content from monitor"""
        
        for content in content_batch:
            # Create update object
            update = KnowledgeUpdate(
                id=f"update_{datetime.now().timestamp()}_{content.id}",
                update_type="extract_facts",
                target_fact_id="",
                new_data={"content": content},
                source_url=content.url,
                confidence=content.relevance_score,
                timestamp=datetime.now()
            )
            
            # Add to processing queue
            await self.update_queue.put(update)
    
    async def _process_update_queue(self):
        """Process knowledge update queue"""
        
        while self.is_running:
            try:
                # Get update from queue (with timeout)
                try:
                    update = await asyncio.wait_for(self.update_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Process update
                await self._process_knowledge_update(update)
                
                # Mark as processed
                update.processed = True
                self.processing_stats['updates_processed'] += 1
                self.processing_stats['last_update_time'] = datetime.now()
                
                # Brief pause to prevent overwhelming
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing update queue: {e}")
                self.processing_stats['processing_errors'] += 1
                await asyncio.sleep(1)
    
    async def _process_knowledge_update(self, update: KnowledgeUpdate):
        """Process individual knowledge update"""
        
        try:
            if update.update_type == "extract_facts":
                content = update.new_data["content"]
                
                # Extract facts from content
                extracted_facts = self.fact_extractor.extract_facts(content)
                
                # Process each extracted fact
                for fact in extracted_facts:
                    await self._process_extracted_fact(fact, content)
            
            elif update.update_type == "verify_fact":
                # Handle fact verification
                await self._verify_fact(update.target_fact_id, update.new_data)
            
            elif update.update_type == "resolve_conflict":
                # Handle conflict resolution
                await self._resolve_fact_conflict(update.target_fact_id, update.new_data)
            
        except Exception as e:
            logger.error(f"Error processing knowledge update {update.id}: {e}")
            self.processing_stats['processing_errors'] += 1
    
    async def _process_extracted_fact(self, fact: KnowledgeFact, source_content: WebContent):
        """Process a newly extracted fact"""
        
        # Check for existing conflicting facts
        conflicting_facts = self.knowledge_graph.get_conflicting_facts(fact)
        
        if conflicting_facts:
            # Resolve conflicts
            for conflicting_fact in conflicting_facts:
                resolved_fact, reason = self.conflict_resolver.resolve_conflict(fact, conflicting_fact)
                
                if resolved_fact.id == fact.id:
                    # New fact wins - add it and update conflicting fact
                    self.knowledge_graph.add_fact(fact)
                    self.knowledge_graph.update_fact(
                        conflicting_fact.id,
                        {'status': FactStatus.CONFLICTING, 'updated_at': datetime.now()}
                    )
                    self.processing_stats['facts_added'] += 1
                else:
                    # Existing fact wins - just update its status
                    self.knowledge_graph.update_fact(
                        conflicting_fact.id,
                        {'status': FactStatus.DISPUTED, 'updated_at': datetime.now()}
                    )
                
                self.processing_stats['conflicts_resolved'] += 1
                logger.debug(f"Resolved conflict: {reason}")
        
        else:
            # No conflicts - add fact directly
            if self.knowledge_graph.add_fact(fact):
                self.processing_stats['facts_added'] += 1
                logger.debug(f"Added new fact: {fact.subject} {fact.predicate} {fact.object}")
        
        # Update source credibility based on fact quality
        source_domain = self._extract_domain_from_url(source_content.url)
        fact_quality = min(fact.confidence, source_content.relevance_score)
        self.credibility_assessor.update_source_accuracy(source_domain, fact_quality)
    
    async def _verify_fact(self, fact_id: str, verification_data: Dict[str, Any]):
        """Verify a fact using additional evidence"""
        
        if fact_id not in self.knowledge_graph.facts:
            return
        
        fact = self.knowledge_graph.facts[fact_id]
        
        # Add verification evidence
        evidence = verification_data.get('evidence', [])
        fact.evidence.extend(evidence)
        
        # Update confidence based on verification
        verification_confidence = verification_data.get('confidence', 0.5)
        fact.confidence = (fact.confidence + verification_confidence) / 2
        
        # Update status
        if fact.confidence >= 0.8:
            fact.status = FactStatus.VERIFIED
        elif fact.confidence >= 0.5:
            fact.status = FactStatus.UNVERIFIED
        else:
            fact.status = FactStatus.DISPUTED
        
        # Update in graph
        self.knowledge_graph.update_fact(fact_id, {
            'confidence': fact.confidence,
            'status': fact.status,
            'evidence': fact.evidence
        })
        
        logger.debug(f"Verified fact {fact_id}: confidence={fact.confidence:.3f}")
    
    async def _resolve_fact_conflict(self, fact_id: str, conflict_data: Dict[str, Any]):
        """Resolve conflicts for a specific fact"""
        
        if fact_id not in self.knowledge_graph.facts:
            return
        
        fact = self.knowledge_graph.facts[fact_id]
        conflicting_fact_ids = conflict_data.get('conflicting_facts', [])
        
        for conflicting_id in conflicting_fact_ids:
            if conflicting_id in self.knowledge_graph.facts:
                conflicting_fact = self.knowledge_graph.facts[conflicting_id]
                
                resolved_fact, reason = self.conflict_resolver.resolve_conflict(fact, conflicting_fact)
                
                # Update the resolved fact in the graph
                self.knowledge_graph.update_fact(resolved_fact.id, {
                    'status': resolved_fact.status,
                    'contradictions': resolved_fact.contradictions,
                    'updated_at': datetime.now()
                })
                
                self.processing_stats['conflicts_resolved'] += 1
                logger.debug(f"Resolved conflict for {fact_id}: {reason}")
    
    def _extract_domain_from_url(self, url: str) -> str:
        """Extract domain from URL"""
        
        import re
        match = re.search(r'https?://([^/]+)', url)
        return match.group(1) if match else url
    
    def get_knowledge_updates_stats(self) -> Dict[str, Any]:
        """Get comprehensive knowledge updating statistics"""
        
        # Combine processing stats with knowledge graph stats
        processing_stats = self.processing_stats.copy()
        if processing_stats['last_update_time']:
            processing_stats['last_update_time'] = processing_stats['last_update_time'].isoformat()
        
        kg_stats = self.knowledge_graph.get_graph_statistics()
        credibility_stats = {
            'top_sources': self.credibility_assessor.get_top_sources(),
            'total_sources_tracked': len(self.credibility_assessor.source_history)
        }
        
        return {
            'is_running': self.is_running,
            'processing_stats': processing_stats,
            'knowledge_graph_stats': kg_stats,
            'credibility_stats': credibility_stats,
            'update_queue_size': self.update_queue.qsize()
        }
    
    async def query_knowledge(self, subject: str, predicate: Optional[str] = None) -> List[KnowledgeFact]:
        """Query knowledge graph for facts"""
        
        if predicate:
            # Get facts by entity and filter by predicate
            entity_facts = self.knowledge_graph.get_facts_by_entity(subject)
            return [fact for fact in entity_facts if fact.predicate == predicate]
        else:
            # Get all facts involving entity
            return self.knowledge_graph.get_facts_by_entity(subject)
    
    async def get_entity_summary(self, entity: str) -> Dict[str, Any]:
        """Get comprehensive summary of an entity"""
        
        facts = self.knowledge_graph.get_facts_by_entity(entity)
        related_entities = self.knowledge_graph.find_related_entities(entity)
        
        # Organize facts by predicate
        facts_by_predicate = defaultdict(list)
        for fact in facts:
            facts_by_predicate[fact.predicate].append(fact)
        
        # Calculate confidence statistics
        confidences = [fact.confidence for fact in facts]
        
        return {
            'entity': entity,
            'total_facts': len(facts),
            'facts_by_predicate': {
                pred: [{'object': f.object, 'confidence': f.confidence, 'status': f.status.value} 
                       for f in fact_list]
                for pred, fact_list in facts_by_predicate.items()
            },
            'related_entities': related_entities,
            'confidence_stats': {
                'average': np.mean(confidences) if confidences else 0,
                'min': np.min(confidences) if confidences else 0,
                'max': np.max(confidences) if confidences else 0
            },
            'last_updated': max([fact.updated_at for fact in facts]).isoformat() if facts else None
        }


# Testing and example usage
async def test_knowledge_updater():
    """Test the real-time knowledge updater"""
    
    # Mock web monitor for testing
    class MockWebMonitor:
        def __init__(self):
            self.callbacks = []
        
        def add_update_callback(self, callback):
            self.callbacks.append(callback)
        
        async def simulate_content_update(self):
            # Simulate web content
            test_content = WebContent(
                id="test_1",
                url="https://openai.com/blog/test",
                title="OpenAI releases new AI model",
                content="OpenAI announced GPT-5, a new language model that is more capable than GPT-4. The model has improved reasoning abilities and can solve complex problems.",
                source="openai.com",
                published_at=datetime.now(),
                retrieved_at=datetime.now(),
                content_type="news",
                relevance_score=0.9,
                keywords=["OpenAI", "GPT-5", "AI", "model"],
                summary="OpenAI releases GPT-5 with improved capabilities",
                language="en"
            )
            
            # Notify callbacks
            for callback in self.callbacks:
                await callback([test_content])
    
    # Create mock web monitor and knowledge updater
    web_monitor = MockWebMonitor()
    updater = RealTimeKnowledgeUpdater(web_monitor)
    
    # Start updating
    update_task = asyncio.create_task(updater.start_updating())
    
    # Simulate content updates
    await web_monitor.simulate_content_update()
    
    # Wait for processing
    await asyncio.sleep(2)
    
    # Query knowledge
    openai_facts = await updater.query_knowledge("OpenAI")
    print(f"Found {len(openai_facts)} facts about OpenAI:")
    for fact in openai_facts:
        print(f"  - {fact.subject} {fact.predicate} {fact.object} (confidence: {fact.confidence:.3f})")
    
    # Get entity summary
    summary = await updater.get_entity_summary("OpenAI")
    print(f"\nOpenAI Summary: {summary}")
    
    # Get statistics
    stats = updater.get_knowledge_updates_stats()
    print(f"\nKnowledge Update Statistics: {stats}")
    
    await updater.stop_updating()


if __name__ == "__main__":
    asyncio.run(test_knowledge_updater())