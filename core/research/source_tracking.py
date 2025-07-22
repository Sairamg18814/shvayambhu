"""Advanced source tracking system for research citations and provenance.

This module provides comprehensive source tracking capabilities including
source registration, provenance tracking, citation management, and
reliability scoring for research and knowledge validation.
"""

import re
import time
import json
import sqlite3
import hashlib
import urllib.parse
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import logging

from .graphrag import KnowledgeGraphStore, Fact
from .fact_extraction import SourceMetadata, SourceType, ExtractedFact
from .memory_aug import MemoryAugmentedSystem, MemoryType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CitationStyle(Enum):
    """Citation formatting styles."""
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    IEEE = "ieee"
    HARVARD = "harvard"
    VANCOUVER = "vancouver"
    CUSTOM = "custom"


class SourceCategory(Enum):
    """Categories of sources for reliability assessment."""
    ACADEMIC_JOURNAL = "academic_journal"
    PEER_REVIEWED = "peer_reviewed"
    GOVERNMENT_OFFICIAL = "government_official"
    NEWS_ORGANIZATION = "news_organization"
    ENCYCLOPEDIA = "encyclopedia"
    BOOK_PUBLISHED = "book_published"
    CONFERENCE_PAPER = "conference_paper"
    THESIS_DISSERTATION = "thesis_dissertation"
    REPORT_TECHNICAL = "report_technical"
    WEBSITE_OFFICIAL = "website_official"
    BLOG_EXPERT = "blog_expert"
    SOCIAL_MEDIA = "social_media"
    WIKI_COLLABORATIVE = "wiki_collaborative"
    UNKNOWN = "unknown"


class AccessibilityStatus(Enum):
    """Accessibility status of sources."""
    PUBLICLY_AVAILABLE = "publicly_available"
    SUBSCRIPTION_REQUIRED = "subscription_required"
    INSTITUTIONAL_ACCESS = "institutional_access"
    ARCHIVE_ONLY = "archive_only"
    UNAVAILABLE = "unavailable"
    BROKEN_LINK = "broken_link"


@dataclass
class SourceRecord:
    """Comprehensive source record with tracking information."""
    source_id: str
    title: str
    url: str = ""
    authors: List[str] = field(default_factory=list)
    publication_date: str = ""
    publisher: str = ""
    doi: str = ""
    isbn: str = ""
    issn: str = ""
    source_type: SourceType = SourceType.UNKNOWN
    source_category: SourceCategory = SourceCategory.UNKNOWN
    language: str = "en"
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    
    # Tracking metadata
    reliability_score: float = 0.5
    accessibility_status: AccessibilityStatus = AccessibilityStatus.PUBLICLY_AVAILABLE
    first_accessed: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    citation_count: int = 0
    
    # Content analysis
    content_hash: str = ""
    content_length: int = 0
    extraction_count: int = 0
    fact_count: int = 0
    
    # Quality metrics
    credibility_indicators: Dict[str, float] = field(default_factory=dict)
    quality_score: float = 0.0
    bias_indicators: Dict[str, float] = field(default_factory=dict)
    
    # Technical metadata
    mime_type: str = ""
    encoding: str = "utf-8"
    last_modified: str = ""
    content_security: Dict[str, Any] = field(default_factory=dict)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.source_id:
            content = f"{self.title}_{self.url}_{self.publication_date}"
            self.source_id = hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def calculate_age_days(self) -> float:
        """Calculate age in days since first access."""
        return (time.time() - self.first_accessed) / 86400
    
    def get_citation_info(self) -> Dict[str, str]:
        """Get citation information in structured format."""
        return {
            'title': self.title,
            'authors': ', '.join(self.authors),
            'publication_date': self.publication_date,
            'publisher': self.publisher,
            'url': self.url,
            'doi': self.doi,
            'isbn': self.isbn,
            'issn': self.issn,
            'access_date': datetime.fromtimestamp(self.last_accessed).strftime('%Y-%m-%d')
        }


@dataclass
class CitationRecord:
    """Record of a citation linking facts to sources."""
    citation_id: str
    fact_id: str
    source_id: str
    page_number: str = ""
    section: str = ""
    quote: str = ""
    paraphrase: str = ""
    citation_type: str = "direct"  # direct, indirect, supporting, contradicting
    confidence: float = 0.0
    context: str = ""
    created_at: float = field(default_factory=time.time)
    created_by: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.citation_id:
            content = f"{self.fact_id}_{self.source_id}_{self.created_at}"
            self.citation_id = hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class ProvenanceTrail:
    """Tracks the provenance of facts through the research process."""
    trail_id: str
    fact_id: str
    source_chain: List[str] = field(default_factory=list)  # source_ids in order
    transformation_steps: List[Dict[str, Any]] = field(default_factory=list)
    confidence_evolution: List[float] = field(default_factory=list)
    validation_steps: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if not self.trail_id:
            content = f"{self.fact_id}_{self.created_at}"
            self.trail_id = hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def add_source(self, source_id: str, transformation_info: Dict[str, Any] = None):
        """Add a source to the provenance trail."""
        self.source_chain.append(source_id)
        if transformation_info:
            self.transformation_steps.append({
                'step': len(self.source_chain),
                'source_id': source_id,
                'timestamp': time.time(),
                **transformation_info
            })
        self.last_updated = time.time()
    
    def add_validation(self, validation_info: Dict[str, Any]):
        """Add validation step to provenance."""
        self.validation_steps.append({
            'timestamp': time.time(),
            **validation_info
        })
        self.last_updated = time.time()


class SourceReliabilityAnalyzer:
    """Analyzes and scores source reliability."""
    
    def __init__(self):
        """Initialize source reliability analyzer."""
        self.domain_scores = self._initialize_domain_scores()
        self.quality_indicators = self._initialize_quality_indicators()
        
    def _initialize_domain_scores(self) -> Dict[str, float]:
        """Initialize domain-based reliability scores."""
        return {
            # Academic and research domains
            'edu': 0.9,
            'ac.uk': 0.9,
            'ac.': 0.85,
            'university': 0.85,
            'scholar.google': 0.9,
            'pubmed': 0.95,
            'arxiv': 0.8,
            'doi.org': 0.9,
            
            # Government domains
            'gov': 0.9,
            'mil': 0.85,
            'europa.eu': 0.85,
            'un.org': 0.85,
            
            # News organizations (major)
            'reuters.com': 0.8,
            'bbc.co.uk': 0.8,
            'nytimes.com': 0.75,
            'washingtonpost.com': 0.75,
            'theguardian.com': 0.75,
            'cnn.com': 0.7,
            
            # Reference sources
            'britannica.com': 0.85,
            'merriam-webster.com': 0.8,
            'oed.com': 0.9,
            
            # Lower reliability sources
            'wikipedia.org': 0.6,
            'blog': 0.3,
            'wordpress': 0.3,
            'medium.com': 0.4,
            'reddit.com': 0.2,
            'twitter.com': 0.2,
            'facebook.com': 0.2,
            'youtube.com': 0.3,
        }
    
    def _initialize_quality_indicators(self) -> Dict[str, Dict[str, float]]:
        """Initialize quality indicators and their weights."""
        return {
            'positive_indicators': {
                'peer_reviewed': 0.3,
                'doi_present': 0.2,
                'multiple_authors': 0.1,
                'recent_publication': 0.1,
                'citation_count': 0.15,
                'institutional_affiliation': 0.15,
                'references_cited': 0.1,
                'methodology_described': 0.1,
                'data_availability': 0.1,
                'conflict_of_interest_declared': 0.05
            },
            'negative_indicators': {
                'no_author': -0.3,
                'anonymous_source': -0.2,
                'no_publication_date': -0.15,
                'broken_links': -0.1,
                'spelling_errors': -0.1,
                'sensational_language': -0.15,
                'no_references': -0.2,
                'conflict_of_interest_undeclared': -0.25,
                'retracted': -1.0,
                'predatory_journal': -0.8
            }
        }
    
    def analyze_source_reliability(self, source: SourceRecord) -> Dict[str, Any]:
        """Analyze source reliability and return detailed assessment.
        
        Args:
            source: Source record to analyze
            
        Returns:
            Dictionary with reliability analysis results
        """
        analysis = {
            'overall_score': 0.5,
            'domain_score': 0.5,
            'content_score': 0.5,
            'metadata_score': 0.5,
            'category_score': 0.5,
            'indicators': {'positive': [], 'negative': []},
            'recommendation': '',
            'confidence': 0.0
        }
        
        # Domain-based scoring
        domain_score = self._calculate_domain_score(source)
        analysis['domain_score'] = domain_score
        
        # Content-based scoring
        content_score = self._calculate_content_score(source)
        analysis['content_score'] = content_score
        
        # Metadata quality scoring
        metadata_score = self._calculate_metadata_score(source)
        analysis['metadata_score'] = metadata_score
        
        # Category-based scoring
        category_score = self._calculate_category_score(source)
        analysis['category_score'] = category_score
        
        # Calculate overall score
        weights = {'domain': 0.3, 'content': 0.25, 'metadata': 0.25, 'category': 0.2}
        overall_score = (
            domain_score * weights['domain'] +
            content_score * weights['content'] +
            metadata_score * weights['metadata'] +
            category_score * weights['category']
        )
        
        analysis['overall_score'] = max(0.0, min(1.0, overall_score))
        
        # Generate recommendation
        analysis['recommendation'] = self._generate_reliability_recommendation(analysis['overall_score'])
        
        # Calculate confidence in the assessment
        analysis['confidence'] = self._calculate_assessment_confidence(source, analysis)
        
        return analysis
    
    def _calculate_domain_score(self, source: SourceRecord) -> float:
        """Calculate reliability score based on domain."""
        if not source.url:
            return 0.3
        
        try:
            parsed_url = urllib.parse.urlparse(source.url)
            domain = parsed_url.netloc.lower()
            
            # Check for exact matches first
            for domain_pattern, score in self.domain_scores.items():
                if domain_pattern in domain:
                    return score
            
            # Default scoring based on TLD
            if domain.endswith('.edu'):
                return 0.8
            elif domain.endswith('.gov'):
                return 0.85
            elif domain.endswith('.org'):
                return 0.6
            elif domain.endswith('.com'):
                return 0.5
            else:
                return 0.4
                
        except Exception:
            return 0.3
    
    def _calculate_content_score(self, source: SourceRecord) -> float:
        """Calculate score based on content quality indicators."""
        score = 0.5
        indicators_found = []
        
        # Positive indicators
        positive_indicators = self.quality_indicators['positive_indicators']
        
        if source.doi:
            score += positive_indicators['doi_present']
            indicators_found.append('doi_present')
        
        if len(source.authors) > 1:
            score += positive_indicators['multiple_authors']
            indicators_found.append('multiple_authors')
        
        if source.publication_date:
            # Check if recent (last 5 years)
            try:
                pub_year = int(source.publication_date[:4])
                current_year = datetime.now().year
                if current_year - pub_year <= 5:
                    score += positive_indicators['recent_publication']
                    indicators_found.append('recent_publication')
            except (ValueError, IndexError):
                pass
        
        if source.abstract:
            score += 0.1  # Has abstract
            indicators_found.append('has_abstract')
        
        if source.keywords:
            score += 0.05  # Has keywords
            indicators_found.append('has_keywords')
        
        # Negative indicators
        negative_indicators = self.quality_indicators['negative_indicators']
        
        if not source.authors:
            score += negative_indicators['no_author']
            indicators_found.append('no_author')
        
        if not source.publication_date:
            score += negative_indicators['no_publication_date']
            indicators_found.append('no_publication_date')
        
        return max(0.0, min(1.0, score))
    
    def _calculate_metadata_score(self, source: SourceRecord) -> float:
        """Calculate score based on metadata completeness."""
        score = 0.0
        total_fields = 8
        
        # Check completeness of key fields
        if source.title:
            score += 1
        if source.authors:
            score += 1
        if source.publication_date:
            score += 1
        if source.publisher:
            score += 1
        if source.url:
            score += 1
        if source.doi or source.isbn or source.issn:
            score += 1
        if source.abstract:
            score += 1
        if source.keywords:
            score += 1
        
        return score / total_fields
    
    def _calculate_category_score(self, source: SourceRecord) -> float:
        """Calculate score based on source category."""
        category_scores = {
            SourceCategory.ACADEMIC_JOURNAL: 0.9,
            SourceCategory.PEER_REVIEWED: 0.85,
            SourceCategory.GOVERNMENT_OFFICIAL: 0.8,
            SourceCategory.NEWS_ORGANIZATION: 0.7,
            SourceCategory.ENCYCLOPEDIA: 0.75,
            SourceCategory.BOOK_PUBLISHED: 0.7,
            SourceCategory.CONFERENCE_PAPER: 0.75,
            SourceCategory.THESIS_DISSERTATION: 0.6,
            SourceCategory.REPORT_TECHNICAL: 0.65,
            SourceCategory.WEBSITE_OFFICIAL: 0.6,
            SourceCategory.BLOG_EXPERT: 0.4,
            SourceCategory.SOCIAL_MEDIA: 0.2,
            SourceCategory.WIKI_COLLABORATIVE: 0.5,
            SourceCategory.UNKNOWN: 0.3
        }
        
        return category_scores.get(source.source_category, 0.3)
    
    def _generate_reliability_recommendation(self, score: float) -> str:
        """Generate reliability recommendation based on score."""
        if score >= 0.8:
            return "Highly reliable source - suitable for critical facts"
        elif score >= 0.6:
            return "Reliable source - good for general information"
        elif score >= 0.4:
            return "Moderately reliable - use with caution, verify with additional sources"
        elif score >= 0.2:
            return "Low reliability - avoid for factual claims"
        else:
            return "Unreliable source - not recommended for factual information"
    
    def _calculate_assessment_confidence(self, source: SourceRecord, analysis: Dict[str, Any]) -> float:
        """Calculate confidence in the reliability assessment."""
        confidence = 0.5
        
        # More confidence if we have complete metadata
        if analysis['metadata_score'] > 0.7:
            confidence += 0.2
        
        # More confidence if domain is well-known
        if analysis['domain_score'] in [0.9, 0.8, 0.85]:
            confidence += 0.2
        
        # More confidence if source has identifiers
        if source.doi or source.isbn:
            confidence += 0.1
        
        # Less confidence for unknown categories
        if source.source_category == SourceCategory.UNKNOWN:
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))


class SourceTracker:
    """Main source tracking and management system."""
    
    def __init__(self, storage_path: str = None, 
                 knowledge_store: KnowledgeGraphStore = None,
                 memory_system: MemoryAugmentedSystem = None):
        """Initialize source tracker.
        
        Args:
            storage_path: Path for persistent storage
            knowledge_store: Knowledge graph store for integration
            memory_system: Memory system for caching
        """
        self.storage_path = storage_path or "source_tracking.db"
        self.knowledge_store = knowledge_store
        self.memory_system = memory_system
        
        # Initialize components
        self.reliability_analyzer = SourceReliabilityAnalyzer()
        
        # Initialize database
        self._initialize_database()
        
        # Statistics
        self.stats = {
            'sources_tracked': 0,
            'citations_created': 0,
            'provenance_trails': 0,
            'reliability_assessments': 0
        }
    
    def _initialize_database(self):
        """Initialize SQLite database for source tracking."""
        with sqlite3.connect(self.storage_path) as conn:
            # Sources table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sources (
                    source_id TEXT PRIMARY KEY,
                    title TEXT,
                    url TEXT,
                    authors TEXT,
                    publication_date TEXT,
                    publisher TEXT,
                    doi TEXT,
                    isbn TEXT,
                    issn TEXT,
                    source_type TEXT,
                    source_category TEXT,
                    language TEXT,
                    abstract TEXT,
                    keywords TEXT,
                    reliability_score REAL,
                    accessibility_status TEXT,
                    first_accessed REAL,
                    last_accessed REAL,
                    access_count INTEGER,
                    citation_count INTEGER,
                    content_hash TEXT,
                    content_length INTEGER,
                    extraction_count INTEGER,
                    fact_count INTEGER,
                    quality_score REAL,
                    metadata TEXT
                )
            """)
            
            # Citations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS citations (
                    citation_id TEXT PRIMARY KEY,
                    fact_id TEXT,
                    source_id TEXT,
                    page_number TEXT,
                    section TEXT,
                    quote TEXT,
                    paraphrase TEXT,
                    citation_type TEXT,
                    confidence REAL,
                    context TEXT,
                    created_at REAL,
                    created_by TEXT,
                    metadata TEXT,
                    FOREIGN KEY (source_id) REFERENCES sources (source_id)
                )
            """)
            
            # Provenance trails table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS provenance_trails (
                    trail_id TEXT PRIMARY KEY,
                    fact_id TEXT,
                    source_chain TEXT,
                    transformation_steps TEXT,
                    confidence_evolution TEXT,
                    validation_steps TEXT,
                    created_at REAL,
                    last_updated REAL
                )
            """)
            
            # Create indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source_url ON sources(url)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source_category ON sources(source_category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_citation_fact ON citations(fact_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_citation_source ON citations(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_provenance_fact ON provenance_trails(fact_id)")
            
            conn.commit()
    
    def register_source(self, source_metadata: SourceMetadata) -> SourceRecord:
        """Register a new source in the tracking system.
        
        Args:
            source_metadata: Source metadata to register
            
        Returns:
            Created source record
        """
        # Create source record
        source_record = SourceRecord(
            source_id=source_metadata.source_id,
            title=source_metadata.title,
            url=source_metadata.url,
            authors=[source_metadata.author] if source_metadata.author else [],
            publication_date=source_metadata.publication_date,
            publisher=source_metadata.publisher,
            source_type=source_metadata.source_type,
            language=source_metadata.language,
            reliability_score=source_metadata.reliability_score,
            metadata=source_metadata.metadata
        )
        
        # Analyze reliability
        reliability_analysis = self.reliability_analyzer.analyze_source_reliability(source_record)
        source_record.reliability_score = reliability_analysis['overall_score']
        source_record.quality_score = reliability_analysis['metadata_score']
        source_record.credibility_indicators = reliability_analysis
        
        # Determine source category
        source_record.source_category = self._determine_source_category(source_record)
        
        # Store in database
        self._store_source_record(source_record)
        
        # Store in memory system if available
        if self.memory_system:
            self.memory_system.store_memory(
                content=source_record,
                memory_type=MemoryType.SEMANTIC,
                importance_score=source_record.reliability_score,
                metadata={
                    'type': 'source_record',
                    'category': source_record.source_category.value,
                    'reliability': source_record.reliability_score
                }
            )
        
        self.stats['sources_tracked'] += 1
        
        logger.info(f"Registered source: {source_record.title} (reliability: {source_record.reliability_score:.2f})")
        
        return source_record
    
    def create_citation(self, fact_id: str, source_id: str, 
                       citation_info: Dict[str, Any] = None) -> CitationRecord:
        """Create a citation linking a fact to a source.
        
        Args:
            fact_id: ID of the fact being cited
            source_id: ID of the source
            citation_info: Additional citation information
            
        Returns:
            Created citation record
        """
        citation_info = citation_info or {}
        
        citation = CitationRecord(
            citation_id="",
            fact_id=fact_id,
            source_id=source_id,
            page_number=citation_info.get('page_number', ''),
            section=citation_info.get('section', ''),
            quote=citation_info.get('quote', ''),
            paraphrase=citation_info.get('paraphrase', ''),
            citation_type=citation_info.get('citation_type', 'direct'),
            confidence=citation_info.get('confidence', 0.8),
            context=citation_info.get('context', ''),
            created_by=citation_info.get('created_by', 'system'),
            metadata=citation_info.get('metadata', {})
        )
        
        # Store citation
        self._store_citation_record(citation)
        
        # Update source citation count
        self._increment_source_citation_count(source_id)
        
        self.stats['citations_created'] += 1
        
        logger.info(f"Created citation {citation.citation_id} linking fact {fact_id} to source {source_id}")
        
        return citation
    
    def create_provenance_trail(self, fact_id: str, initial_source_id: str) -> ProvenanceTrail:
        """Create a provenance trail for a fact.
        
        Args:
            fact_id: ID of the fact
            initial_source_id: ID of the initial source
            
        Returns:
            Created provenance trail
        """
        trail = ProvenanceTrail(
            trail_id="",
            fact_id=fact_id,
            source_chain=[initial_source_id],
            transformation_steps=[{
                'step': 1,
                'source_id': initial_source_id,
                'operation': 'initial_extraction',
                'timestamp': time.time()
            }],
            confidence_evolution=[0.8]  # Initial confidence
        )
        
        # Store provenance trail
        self._store_provenance_trail(trail)
        
        self.stats['provenance_trails'] += 1
        
        logger.info(f"Created provenance trail {trail.trail_id} for fact {fact_id}")
        
        return trail
    
    def get_source_by_id(self, source_id: str) -> Optional[SourceRecord]:
        """Get source record by ID.
        
        Args:
            source_id: Source ID to look up
            
        Returns:
            Source record if found, None otherwise
        """
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM sources WHERE source_id = ?
            """, (source_id,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_source_record(row)
        
        return None
    
    def get_citations_for_fact(self, fact_id: str) -> List[CitationRecord]:
        """Get all citations for a fact.
        
        Args:
            fact_id: Fact ID to get citations for
            
        Returns:
            List of citation records
        """
        citations = []
        
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM citations WHERE fact_id = ?
            """, (fact_id,))
            
            for row in cursor.fetchall():
                citation = self._row_to_citation_record(row)
                citations.append(citation)
        
        return citations
    
    def get_provenance_trail(self, fact_id: str) -> Optional[ProvenanceTrail]:
        """Get provenance trail for a fact.
        
        Args:
            fact_id: Fact ID to get provenance for
            
        Returns:
            Provenance trail if found, None otherwise
        """
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM provenance_trails WHERE fact_id = ?
            """, (fact_id,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_provenance_trail(row)
        
        return None
    
    def search_sources(self, query: str = None, source_type: SourceType = None,
                      min_reliability: float = 0.0, limit: int = 100) -> List[SourceRecord]:
        """Search sources by various criteria.
        
        Args:
            query: Text query to search in title/authors
            source_type: Filter by source type
            min_reliability: Minimum reliability score
            limit: Maximum results to return
            
        Returns:
            List of matching source records
        """
        sources = []
        
        # Build query
        sql_query = "SELECT * FROM sources WHERE reliability_score >= ?"
        params = [min_reliability]
        
        if source_type:
            sql_query += " AND source_type = ?"
            params.append(source_type.value)
        
        if query:
            sql_query += " AND (title LIKE ? OR authors LIKE ?)"
            query_pattern = f"%{query}%"
            params.extend([query_pattern, query_pattern])
        
        sql_query += " ORDER BY reliability_score DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.execute(sql_query, params)
            
            for row in cursor.fetchall():
                source = self._row_to_source_record(row)
                sources.append(source)
        
        return sources
    
    def generate_bibliography(self, fact_ids: List[str], 
                            citation_style: CitationStyle = CitationStyle.APA) -> str:
        """Generate bibliography for a list of facts.
        
        Args:
            fact_ids: List of fact IDs to generate bibliography for
            citation_style: Citation style to use
            
        Returns:
            Formatted bibliography string
        """
        # Get all unique sources used by these facts
        source_ids = set()
        
        for fact_id in fact_ids:
            citations = self.get_citations_for_fact(fact_id)
            for citation in citations:
                source_ids.add(citation.source_id)
        
        # Get source records
        sources = []
        for source_id in source_ids:
            source = self.get_source_by_id(source_id)
            if source:
                sources.append(source)
        
        # Sort alphabetically by title
        sources.sort(key=lambda s: s.title.lower())
        
        # Format bibliography
        bibliography_lines = []
        for source in sources:
            citation_text = self._format_citation(source, citation_style)
            bibliography_lines.append(citation_text)
        
        return "\n".join(bibliography_lines)
    
    def _determine_source_category(self, source: SourceRecord) -> SourceCategory:
        """Determine source category based on source characteristics."""
        url_lower = source.url.lower()
        title_lower = source.title.lower()
        
        # Academic indicators
        if source.doi or 'journal' in title_lower or 'arxiv' in url_lower:
            return SourceCategory.ACADEMIC_JOURNAL
        
        # Government indicators
        if '.gov' in url_lower or 'government' in title_lower:
            return SourceCategory.GOVERNMENT_OFFICIAL
        
        # News indicators
        news_domains = ['reuters', 'bbc', 'cnn', 'nytimes', 'washingtonpost', 'theguardian']
        if any(domain in url_lower for domain in news_domains):
            return SourceCategory.NEWS_ORGANIZATION
        
        # Encyclopedia indicators
        if 'britannica' in url_lower or 'encyclopedia' in title_lower:
            return SourceCategory.ENCYCLOPEDIA
        
        # Book indicators
        if source.isbn or 'book' in title_lower:
            return SourceCategory.BOOK_PUBLISHED
        
        # Wiki indicators
        if 'wikipedia' in url_lower:
            return SourceCategory.WIKI_COLLABORATIVE
        
        # Blog indicators
        if 'blog' in url_lower or 'wordpress' in url_lower or 'medium' in url_lower:
            return SourceCategory.BLOG_EXPERT
        
        # Social media indicators
        social_domains = ['twitter', 'facebook', 'instagram', 'linkedin', 'reddit']
        if any(domain in url_lower for domain in social_domains):
            return SourceCategory.SOCIAL_MEDIA
        
        return SourceCategory.UNKNOWN
    
    def _format_citation(self, source: SourceRecord, style: CitationStyle) -> str:
        """Format citation according to specified style."""
        citation_info = source.get_citation_info()
        
        if style == CitationStyle.APA:
            return self._format_apa_citation(citation_info)
        elif style == CitationStyle.MLA:
            return self._format_mla_citation(citation_info)
        elif style == CitationStyle.CHICAGO:
            return self._format_chicago_citation(citation_info)
        else:
            # Default to APA
            return self._format_apa_citation(citation_info)
    
    def _format_apa_citation(self, info: Dict[str, str]) -> str:
        """Format citation in APA style."""
        parts = []
        
        if info['authors']:
            parts.append(f"{info['authors']}.")
        
        if info['publication_date']:
            year = info['publication_date'][:4] if len(info['publication_date']) >= 4 else info['publication_date']
            parts.append(f"({year}).")
        
        if info['title']:
            parts.append(f"{info['title']}.")
        
        if info['publisher']:
            parts.append(f"{info['publisher']}.")
        
        if info['url']:
            parts.append(f"Retrieved from {info['url']}")
        
        return " ".join(parts)
    
    def _format_mla_citation(self, info: Dict[str, str]) -> str:
        """Format citation in MLA style."""
        parts = []
        
        if info['authors']:
            parts.append(f"{info['authors']}.")
        
        if info['title']:
            parts.append(f'"{info["title"]}."')
        
        if info['publisher']:
            parts.append(f"{info['publisher']},")
        
        if info['publication_date']:
            parts.append(f"{info['publication_date']}.")
        
        if info['url']:
            parts.append(f"Web. {info['access_date']}.")
        
        return " ".join(parts)
    
    def _format_chicago_citation(self, info: Dict[str, str]) -> str:
        """Format citation in Chicago style."""
        parts = []
        
        if info['authors']:
            parts.append(f"{info['authors']}.")
        
        if info['title']:
            parts.append(f'"{info["title"]}."')
        
        if info['publisher']:
            parts.append(f"{info['publisher']},")
        
        if info['publication_date']:
            parts.append(f"{info['publication_date']}.")
        
        if info['url']:
            parts.append(f"Accessed {info['access_date']}. {info['url']}.")
        
        return " ".join(parts)
    
    def _store_source_record(self, source: SourceRecord):
        """Store source record in database."""
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sources VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                    ?, ?, ?, ?, ?, ?
                )
            """, (
                source.source_id, source.title, source.url, json.dumps(source.authors),
                source.publication_date, source.publisher, source.doi, source.isbn,
                source.issn, source.source_type.value, source.source_category.value,
                source.language, source.abstract, json.dumps(source.keywords),
                source.reliability_score, source.accessibility_status.value,
                source.first_accessed, source.last_accessed, source.access_count,
                source.citation_count, source.content_hash, source.content_length,
                source.extraction_count, source.fact_count, source.quality_score,
                json.dumps(source.metadata)
            ))
            conn.commit()
    
    def _store_citation_record(self, citation: CitationRecord):
        """Store citation record in database."""
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO citations VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                citation.citation_id, citation.fact_id, citation.source_id,
                citation.page_number, citation.section, citation.quote,
                citation.paraphrase, citation.citation_type, citation.confidence,
                citation.context, citation.created_at, citation.created_by,
                json.dumps(citation.metadata)
            ))
            conn.commit()
    
    def _store_provenance_trail(self, trail: ProvenanceTrail):
        """Store provenance trail in database."""
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO provenance_trails VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                trail.trail_id, trail.fact_id, json.dumps(trail.source_chain),
                json.dumps(trail.transformation_steps),
                json.dumps(trail.confidence_evolution),
                json.dumps(trail.validation_steps),
                trail.created_at, trail.last_updated
            ))
            conn.commit()
    
    def _increment_source_citation_count(self, source_id: str):
        """Increment citation count for a source."""
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                UPDATE sources SET citation_count = citation_count + 1 
                WHERE source_id = ?
            """, (source_id,))
            conn.commit()
    
    def _row_to_source_record(self, row) -> SourceRecord:
        """Convert database row to SourceRecord."""
        return SourceRecord(
            source_id=row[0],
            title=row[1] or "",
            url=row[2] or "",
            authors=json.loads(row[3] or "[]"),
            publication_date=row[4] or "",
            publisher=row[5] or "",
            doi=row[6] or "",
            isbn=row[7] or "",
            issn=row[8] or "",
            source_type=SourceType(row[9]) if row[9] else SourceType.UNKNOWN,
            source_category=SourceCategory(row[10]) if row[10] else SourceCategory.UNKNOWN,
            language=row[11] or "en",
            abstract=row[12] or "",
            keywords=json.loads(row[13] or "[]"),
            reliability_score=row[14] or 0.5,
            accessibility_status=AccessibilityStatus(row[15]) if row[15] else AccessibilityStatus.PUBLICLY_AVAILABLE,
            first_accessed=row[16] or time.time(),
            last_accessed=row[17] or time.time(),
            access_count=row[18] or 0,
            citation_count=row[19] or 0,
            content_hash=row[20] or "",
            content_length=row[21] or 0,
            extraction_count=row[22] or 0,
            fact_count=row[23] or 0,
            quality_score=row[24] or 0.0,
            metadata=json.loads(row[25] or "{}")
        )
    
    def _row_to_citation_record(self, row) -> CitationRecord:
        """Convert database row to CitationRecord."""
        return CitationRecord(
            citation_id=row[0],
            fact_id=row[1],
            source_id=row[2],
            page_number=row[3] or "",
            section=row[4] or "",
            quote=row[5] or "",
            paraphrase=row[6] or "",
            citation_type=row[7] or "direct",
            confidence=row[8] or 0.0,
            context=row[9] or "",
            created_at=row[10] or time.time(),
            created_by=row[11] or "system",
            metadata=json.loads(row[12] or "{}")
        )
    
    def _row_to_provenance_trail(self, row) -> ProvenanceTrail:
        """Convert database row to ProvenanceTrail."""
        return ProvenanceTrail(
            trail_id=row[0],
            fact_id=row[1],
            source_chain=json.loads(row[2] or "[]"),
            transformation_steps=json.loads(row[3] or "[]"),
            confidence_evolution=json.loads(row[4] or "[]"),
            validation_steps=json.loads(row[5] or "[]"),
            created_at=row[6] or time.time(),
            last_updated=row[7] or time.time()
        )
    
    def get_tracking_statistics(self) -> Dict[str, Any]:
        """Get source tracking statistics."""
        with sqlite3.connect(self.storage_path) as conn:
            # Count sources by category
            cursor = conn.execute("""
                SELECT source_category, COUNT(*), AVG(reliability_score)
                FROM sources GROUP BY source_category
            """)
            category_stats = {row[0]: {'count': row[1], 'avg_reliability': row[2]} 
                            for row in cursor.fetchall()}
            
            # Count total citations
            cursor = conn.execute("SELECT COUNT(*) FROM citations")
            total_citations = cursor.fetchone()[0]
            
            # Count provenance trails
            cursor = conn.execute("SELECT COUNT(*) FROM provenance_trails")
            total_trails = cursor.fetchone()[0]
            
            # Average reliability
            cursor = conn.execute("SELECT AVG(reliability_score) FROM sources")
            avg_reliability = cursor.fetchone()[0] or 0.0
        
        return {
            **self.stats,
            'total_citations_db': total_citations,
            'total_provenance_trails_db': total_trails,
            'average_source_reliability': avg_reliability,
            'sources_by_category': category_stats
        }


# Convenience functions
def create_source_tracker(storage_path: str = None) -> SourceTracker:
    """Create a new source tracker.
    
    Args:
        storage_path: Path for database storage
        
    Returns:
        SourceTracker instance
    """
    return SourceTracker(storage_path)


def track_fact_sources(fact: ExtractedFact, tracker: SourceTracker) -> Tuple[SourceRecord, CitationRecord]:
    """Track sources for an extracted fact.
    
    Args:
        fact: Extracted fact to track sources for
        tracker: Source tracker instance
        
    Returns:
        Tuple of (source_record, citation_record)
    """
    # Register source
    source_record = tracker.register_source(fact.source_metadata)
    
    # Create citation
    citation_record = tracker.create_citation(
        fact.fact_id,
        source_record.source_id,
        {
            'quote': fact.supporting_text,
            'context': fact.context,
            'confidence': fact.confidence,
            'metadata': fact.metadata
        }
    )
    
    return source_record, citation_record