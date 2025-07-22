"""GraphRAG (Graph-based Retrieval Augmented Generation) implementation.

This module provides a comprehensive knowledge graph-based system for storing,
querying, and reasoning over structured knowledge to support research and
fact-checking capabilities.
"""

import sqlite3
import json
import time
import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import pickle
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of entities in the knowledge graph."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    EVENT = "event"
    OBJECT = "object"
    ABSTRACT = "abstract"
    DATE = "date"
    NUMBER = "number"
    UNKNOWN = "unknown"


class RelationType(Enum):
    """Types of relationships between entities."""
    IS_A = "is_a"
    PART_OF = "part_of"
    LOCATED_IN = "located_in"
    OCCURRED_AT = "occurred_at"
    CAUSED_BY = "caused_by"
    RELATED_TO = "related_to"
    CREATED_BY = "created_by"
    WORKS_FOR = "works_for"
    MEMBER_OF = "member_of"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    TEMPORAL_BEFORE = "temporal_before"
    TEMPORAL_AFTER = "temporal_after"
    SIMILAR_TO = "similar_to"
    DIFFERENT_FROM = "different_from"


class ConfidenceLevel(Enum):
    """Confidence levels for facts and relationships."""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9
    CERTAIN = 1.0


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    entity_id: str
    name: str
    entity_type: EntityType
    attributes: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    description: Optional[str] = None
    confidence: float = 1.0
    sources: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if not self.entity_id:
            self.entity_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate a unique entity ID."""
        content = f"{self.name}_{self.entity_type.value}_{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def add_alias(self, alias: str):
        """Add an alias for this entity."""
        if alias not in self.aliases:
            self.aliases.append(alias)
            self.updated_at = time.time()
    
    def add_source(self, source: str):
        """Add a source for this entity."""
        if source not in self.sources:
            self.sources.append(source)
            self.updated_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        data = {
            'entity_id': self.entity_id,
            'name': self.name,
            'entity_type': self.entity_type.value,
            'attributes': self.attributes,
            'aliases': self.aliases,
            'description': self.description,
            'confidence': self.confidence,
            'sources': self.sources,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
        if self.embedding is not None:
            data['embedding'] = self.embedding.tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """Create entity from dictionary representation."""
        embedding = None
        if 'embedding' in data and data['embedding']:
            embedding = np.array(data['embedding'])
        
        return cls(
            entity_id=data['entity_id'],
            name=data['name'],
            entity_type=EntityType(data['entity_type']),
            attributes=data.get('attributes', {}),
            aliases=data.get('aliases', []),
            description=data.get('description'),
            confidence=data.get('confidence', 1.0),
            sources=data.get('sources', []),
            created_at=data.get('created_at', time.time()),
            updated_at=data.get('updated_at', time.time()),
            embedding=embedding
        )


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: RelationType
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    sources: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if not self.relationship_id:
            self.relationship_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate a unique relationship ID."""
        content = f"{self.source_entity_id}_{self.relation_type.value}_{self.target_entity_id}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def add_source(self, source: str):
        """Add a source for this relationship."""
        if source not in self.sources:
            self.sources.append(source)
            self.updated_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary representation."""
        return {
            'relationship_id': self.relationship_id,
            'source_entity_id': self.source_entity_id,
            'target_entity_id': self.target_entity_id,
            'relation_type': self.relation_type.value,
            'attributes': self.attributes,
            'confidence': self.confidence,
            'sources': self.sources,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relationship':
        """Create relationship from dictionary representation."""
        return cls(
            relationship_id=data['relationship_id'],
            source_entity_id=data['source_entity_id'],
            target_entity_id=data['target_entity_id'],
            relation_type=RelationType(data['relation_type']),
            attributes=data.get('attributes', {}),
            confidence=data.get('confidence', 1.0),
            sources=data.get('sources', []),
            created_at=data.get('created_at', time.time()),
            updated_at=data.get('updated_at', time.time())
        )


@dataclass
class Fact:
    """Represents a factual claim with evidence."""
    fact_id: str
    claim: str
    entities: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if not self.fact_id:
            self.fact_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate a unique fact ID."""
        content = f"{self.claim}_{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def add_evidence(self, evidence: str):
        """Add evidence for this fact."""
        if evidence not in self.evidence:
            self.evidence.append(evidence)
            self.updated_at = time.time()
    
    def add_contradiction(self, fact_id: str):
        """Add a contradicting fact."""
        if fact_id not in self.contradictions:
            self.contradictions.append(fact_id)
            self.updated_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert fact to dictionary representation."""
        return {
            'fact_id': self.fact_id,
            'claim': self.claim,
            'entities': self.entities,
            'relationships': self.relationships,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'sources': self.sources,
            'contradictions': self.contradictions,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Fact':
        """Create fact from dictionary representation."""
        return cls(
            fact_id=data['fact_id'],
            claim=data['claim'],
            entities=data.get('entities', []),
            relationships=data.get('relationships', []),
            confidence=data.get('confidence', 1.0),
            evidence=data.get('evidence', []),
            sources=data.get('sources', []),
            contradictions=data.get('contradictions', []),
            created_at=data.get('created_at', time.time()),
            updated_at=data.get('updated_at', time.time())
        )


class KnowledgeGraphStore:
    """SQLite-based knowledge graph storage system."""
    
    def __init__(self, db_path: str = "knowledge_graph.db"):
        """Initialize the knowledge graph store.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the SQLite database with required tables."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=1000")
        self.conn.execute("PRAGMA temp_store=memory")
        
        # Create tables
        self._create_tables()
        self._create_indexes()
    
    def _create_tables(self):
        """Create database tables."""
        
        # Entities table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                attributes TEXT,
                aliases TEXT,
                description TEXT,
                confidence REAL DEFAULT 1.0,
                sources TEXT,
                created_at REAL,
                updated_at REAL,
                embedding BLOB
            )
        """)
        
        # Relationships table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                relationship_id TEXT PRIMARY KEY,
                source_entity_id TEXT NOT NULL,
                target_entity_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                attributes TEXT,
                confidence REAL DEFAULT 1.0,
                sources TEXT,
                created_at REAL,
                updated_at REAL,
                FOREIGN KEY (source_entity_id) REFERENCES entities (entity_id),
                FOREIGN KEY (target_entity_id) REFERENCES entities (entity_id)
            )
        """)
        
        # Facts table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                fact_id TEXT PRIMARY KEY,
                claim TEXT NOT NULL,
                entities TEXT,
                relationships TEXT,
                confidence REAL DEFAULT 1.0,
                evidence TEXT,
                sources TEXT,
                contradictions TEXT,
                created_at REAL,
                updated_at REAL
            )
        """)
        
        # Knowledge versions table for temporal tracking
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_versions (
                version_id TEXT PRIMARY KEY,
                entity_id TEXT,
                relationship_id TEXT,
                fact_id TEXT,
                operation TEXT NOT NULL,
                data TEXT,
                timestamp REAL,
                user_id TEXT
            )
        """)
        
        self.conn.commit()
    
    def _create_indexes(self):
        """Create database indexes for performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)",
            "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)",
            "CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_entity_id)",
            "CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_entity_id)",
            "CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relation_type)",
            "CREATE INDEX IF NOT EXISTS idx_facts_entities ON facts(entities)",
            "CREATE INDEX IF NOT EXISTS idx_versions_timestamp ON knowledge_versions(timestamp)"
        ]
        
        for index_sql in indexes:
            self.conn.execute(index_sql)
        
        self.conn.commit()
    
    def add_entity(self, entity: Entity) -> bool:
        """Add an entity to the knowledge graph.
        
        Args:
            entity: Entity object to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            embedding_blob = None
            if entity.embedding is not None:
                embedding_blob = pickle.dumps(entity.embedding)
            
            self.conn.execute("""
                INSERT OR REPLACE INTO entities 
                (entity_id, name, entity_type, attributes, aliases, description, 
                 confidence, sources, created_at, updated_at, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entity.entity_id,
                entity.name,
                entity.entity_type.value,
                json.dumps(entity.attributes),
                json.dumps(entity.aliases),
                entity.description,
                entity.confidence,
                json.dumps(entity.sources),
                entity.created_at,
                entity.updated_at,
                embedding_blob
            ))
            
            # Record version
            self._record_version(
                entity_id=entity.entity_id,
                operation="INSERT",
                data=json.dumps(entity.to_dict())
            )
            
            self.conn.commit()
            logger.info(f"Added entity: {entity.name} ({entity.entity_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding entity {entity.name}: {str(e)}")
            self.conn.rollback()
            return False
    
    def add_relationship(self, relationship: Relationship) -> bool:
        """Add a relationship to the knowledge graph.
        
        Args:
            relationship: Relationship object to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO relationships 
                (relationship_id, source_entity_id, target_entity_id, relation_type,
                 attributes, confidence, sources, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                relationship.relationship_id,
                relationship.source_entity_id,
                relationship.target_entity_id,
                relationship.relation_type.value,
                json.dumps(relationship.attributes),
                relationship.confidence,
                json.dumps(relationship.sources),
                relationship.created_at,
                relationship.updated_at
            ))
            
            # Record version
            self._record_version(
                relationship_id=relationship.relationship_id,
                operation="INSERT",
                data=json.dumps(relationship.to_dict())
            )
            
            self.conn.commit()
            logger.info(f"Added relationship: {relationship.relationship_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding relationship {relationship.relationship_id}: {str(e)}")
            self.conn.rollback()
            return False
    
    def add_fact(self, fact: Fact) -> bool:
        """Add a fact to the knowledge graph.
        
        Args:
            fact: Fact object to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO facts 
                (fact_id, claim, entities, relationships, confidence,
                 evidence, sources, contradictions, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fact.fact_id,
                fact.claim,
                json.dumps(fact.entities),
                json.dumps(fact.relationships),
                fact.confidence,
                json.dumps(fact.evidence),
                json.dumps(fact.sources),
                json.dumps(fact.contradictions),
                fact.created_at,
                fact.updated_at
            ))
            
            # Record version
            self._record_version(
                fact_id=fact.fact_id,
                operation="INSERT",
                data=json.dumps(fact.to_dict())
            )
            
            self.conn.commit()
            logger.info(f"Added fact: {fact.claim[:100]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error adding fact {fact.fact_id}: {str(e)}")
            self.conn.rollback()
            return False
    
    def _record_version(self, operation: str, data: str, entity_id: str = None, 
                       relationship_id: str = None, fact_id: str = None, user_id: str = None):
        """Record a version entry for change tracking."""
        version_id = str(uuid.uuid4())
        
        self.conn.execute("""
            INSERT INTO knowledge_versions 
            (version_id, entity_id, relationship_id, fact_id, operation, data, timestamp, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            version_id,
            entity_id,
            relationship_id,
            fact_id,
            operation,
            data,
            time.time(),
            user_id
        ))
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by ID.
        
        Args:
            entity_id: ID of the entity to retrieve
            
        Returns:
            Entity object if found, None otherwise
        """
        cursor = self.conn.execute("""
            SELECT entity_id, name, entity_type, attributes, aliases, description,
                   confidence, sources, created_at, updated_at, embedding
            FROM entities WHERE entity_id = ?
        """, (entity_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        # Parse JSON fields
        attributes = json.loads(row[3] or '{}')
        aliases = json.loads(row[4] or '[]')
        sources = json.loads(row[7] or '[]')
        
        # Deserialize embedding
        embedding = None
        if row[10]:
            embedding = pickle.loads(row[10])
        
        return Entity(
            entity_id=row[0],
            name=row[1],
            entity_type=EntityType(row[2]),
            attributes=attributes,
            aliases=aliases,
            description=row[5],
            confidence=row[6],
            sources=sources,
            created_at=row[8],
            updated_at=row[9],
            embedding=embedding
        )
    
    def find_entities(self, name: str = None, entity_type: EntityType = None,
                     limit: int = 100) -> List[Entity]:
        """Find entities matching criteria.
        
        Args:
            name: Entity name to search for (partial match)
            entity_type: Type of entities to search for
            limit: Maximum number of entities to return
            
        Returns:
            List of matching entities
        """
        query = """
            SELECT entity_id, name, entity_type, attributes, aliases, description,
                   confidence, sources, created_at, updated_at, embedding
            FROM entities WHERE 1=1
        """
        params = []
        
        if name:
            query += " AND name LIKE ?"
            params.append(f"%{name}%")
        
        if entity_type:
            query += " AND entity_type = ?"
            params.append(entity_type.value)
        
        query += " ORDER BY confidence DESC, updated_at DESC LIMIT ?"
        params.append(limit)
        
        cursor = self.conn.execute(query, params)
        entities = []
        
        for row in cursor.fetchall():
            attributes = json.loads(row[3] or '{}')
            aliases = json.loads(row[4] or '[]')
            sources = json.loads(row[7] or '[]')
            
            embedding = None
            if row[10]:
                embedding = pickle.loads(row[10])
            
            entity = Entity(
                entity_id=row[0],
                name=row[1],
                entity_type=EntityType(row[2]),
                attributes=attributes,
                aliases=aliases,
                description=row[5],
                confidence=row[6],
                sources=sources,
                created_at=row[8],
                updated_at=row[9],
                embedding=embedding
            )
            entities.append(entity)
        
        return entities
    
    def get_relationships(self, entity_id: str, relation_type: RelationType = None,
                         direction: str = "both") -> List[Relationship]:
        """Get relationships for an entity.
        
        Args:
            entity_id: ID of the entity
            relation_type: Type of relationships to filter by
            direction: "incoming", "outgoing", or "both"
            
        Returns:
            List of relationships
        """
        query_conditions = []
        params = []
        
        if direction in ["outgoing", "both"]:
            query_conditions.append("source_entity_id = ?")
            params.append(entity_id)
        
        if direction in ["incoming", "both"]:
            if query_conditions:
                query_conditions.append("OR target_entity_id = ?")
            else:
                query_conditions.append("target_entity_id = ?")
            params.append(entity_id)
        
        where_clause = "(" + " ".join(query_conditions) + ")"
        
        if relation_type:
            where_clause += " AND relation_type = ?"
            params.append(relation_type.value)
        
        query = f"""
            SELECT relationship_id, source_entity_id, target_entity_id, relation_type,
                   attributes, confidence, sources, created_at, updated_at
            FROM relationships WHERE {where_clause}
            ORDER BY confidence DESC, updated_at DESC
        """
        
        cursor = self.conn.execute(query, params)
        relationships = []
        
        for row in cursor.fetchall():
            attributes = json.loads(row[4] or '{}')
            sources = json.loads(row[6] or '[]')
            
            relationship = Relationship(
                relationship_id=row[0],
                source_entity_id=row[1],
                target_entity_id=row[2],
                relation_type=RelationType(row[3]),
                attributes=attributes,
                confidence=row[5],
                sources=sources,
                created_at=row[7],
                updated_at=row[8]
            )
            relationships.append(relationship)
        
        return relationships
    
    def find_path(self, source_entity_id: str, target_entity_id: str,
                  max_depth: int = 3) -> List[List[str]]:
        """Find paths between two entities using BFS.
        
        Args:
            source_entity_id: Starting entity ID
            target_entity_id: Target entity ID
            max_depth: Maximum path depth
            
        Returns:
            List of paths (each path is a list of entity IDs)
        """
        if source_entity_id == target_entity_id:
            return [[source_entity_id]]
        
        # Build adjacency graph
        adjacency = defaultdict(list)
        cursor = self.conn.execute("""
            SELECT source_entity_id, target_entity_id 
            FROM relationships
        """)
        
        for source, target in cursor.fetchall():
            adjacency[source].append(target)
            adjacency[target].append(source)  # Treat as undirected
        
        # BFS to find paths
        queue = deque([(source_entity_id, [source_entity_id])])
        visited = set()
        paths = []
        
        while queue and len(paths) < 10:  # Limit number of paths
            current_entity, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            if current_entity == target_entity_id:
                paths.append(path)
                continue
            
            if current_entity in visited:
                continue
            
            visited.add(current_entity)
            
            for neighbor in adjacency[current_entity]:
                if neighbor not in path:  # Avoid cycles
                    queue.append((neighbor, path + [neighbor]))
        
        return paths
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics.
        
        Returns:
            Dictionary containing various statistics
        """
        stats = {}
        
        # Entity statistics
        cursor = self.conn.execute("SELECT COUNT(*) FROM entities")
        stats['total_entities'] = cursor.fetchone()[0]
        
        cursor = self.conn.execute("""
            SELECT entity_type, COUNT(*) 
            FROM entities 
            GROUP BY entity_type
        """)
        stats['entities_by_type'] = dict(cursor.fetchall())
        
        # Relationship statistics
        cursor = self.conn.execute("SELECT COUNT(*) FROM relationships")
        stats['total_relationships'] = cursor.fetchone()[0]
        
        cursor = self.conn.execute("""
            SELECT relation_type, COUNT(*) 
            FROM relationships 
            GROUP BY relation_type
        """)
        stats['relationships_by_type'] = dict(cursor.fetchall())
        
        # Fact statistics
        cursor = self.conn.execute("SELECT COUNT(*) FROM facts")
        stats['total_facts'] = cursor.fetchone()[0]
        
        # Average confidence scores
        cursor = self.conn.execute("SELECT AVG(confidence) FROM entities")
        stats['avg_entity_confidence'] = cursor.fetchone()[0] or 0.0
        
        cursor = self.conn.execute("SELECT AVG(confidence) FROM relationships")
        stats['avg_relationship_confidence'] = cursor.fetchone()[0] or 0.0
        
        cursor = self.conn.execute("SELECT AVG(confidence) FROM facts")
        stats['avg_fact_confidence'] = cursor.fetchone()[0] or 0.0
        
        # Most connected entities
        cursor = self.conn.execute("""
            SELECT e.name, e.entity_id, COUNT(r.relationship_id) as connections
            FROM entities e
            LEFT JOIN relationships r ON (e.entity_id = r.source_entity_id OR e.entity_id = r.target_entity_id)
            GROUP BY e.entity_id
            ORDER BY connections DESC
            LIMIT 10
        """)
        stats['most_connected_entities'] = [
            {'name': row[0], 'entity_id': row[1], 'connections': row[2]}
            for row in cursor.fetchall()
        ]
        
        return stats
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None


class GraphRAGSystem:
    """Main GraphRAG system integrating knowledge graph with retrieval and generation."""
    
    def __init__(self, db_path: str = "knowledge_graph.db"):
        """Initialize the GraphRAG system.
        
        Args:
            db_path: Path to the knowledge graph database
        """
        self.knowledge_store = KnowledgeGraphStore(db_path)
        self.entity_cache = {}
        self.relationship_cache = {}
        
    def add_knowledge_from_text(self, text: str, source: str = None) -> Dict[str, Any]:
        """Extract and add knowledge from text to the graph.
        
        Args:
            text: Input text to extract knowledge from
            source: Source identifier for the text
            
        Returns:
            Dictionary with extraction results
        """
        # This would typically use NLP models for entity/relationship extraction
        # For now, we'll implement a simple version
        results = {
            'entities_added': 0,
            'relationships_added': 0,
            'facts_added': 0
        }
        
        # Simple entity extraction (would be replaced with proper NLP)
        entities = self._extract_entities_simple(text)
        for entity in entities:
            if source:
                entity.add_source(source)
            if self.knowledge_store.add_entity(entity):
                results['entities_added'] += 1
        
        # Simple relationship extraction
        relationships = self._extract_relationships_simple(entities, text)
        for relationship in relationships:
            if source:
                relationship.add_source(source)
            if self.knowledge_store.add_relationship(relationship):
                results['relationships_added'] += 1
        
        return results
    
    def _extract_entities_simple(self, text: str) -> List[Entity]:
        """Simple entity extraction (placeholder for proper NLP)."""
        entities = []
        
        # Very basic pattern matching for demonstration
        words = text.split()
        
        for i, word in enumerate(words):
            # Simple heuristics
            if word.isupper() and len(word) > 1:
                entity = Entity(
                    entity_id="",
                    name=word,
                    entity_type=EntityType.ORGANIZATION,
                    confidence=0.7
                )
                entities.append(entity)
            elif word.istitle() and len(word) > 2:
                entity = Entity(
                    entity_id="",
                    name=word,
                    entity_type=EntityType.PERSON,
                    confidence=0.6
                )
                entities.append(entity)
        
        return entities
    
    def _extract_relationships_simple(self, entities: List[Entity], text: str) -> List[Relationship]:
        """Simple relationship extraction (placeholder for proper NLP)."""
        relationships = []
        
        # Very basic relationship extraction
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                relationship = Relationship(
                    relationship_id="",
                    source_entity_id=entities[i].entity_id,
                    target_entity_id=entities[j].entity_id,
                    relation_type=RelationType.RELATED_TO,
                    confidence=0.5
                )
                relationships.append(relationship)
        
        return relationships
    
    def query_knowledge(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Query the knowledge graph for relevant information.
        
        Args:
            query: Natural language query
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary containing query results
        """
        results = {
            'entities': [],
            'relationships': [],
            'facts': [],
            'reasoning_paths': []
        }
        
        # Simple keyword-based search (would be enhanced with semantic search)
        query_words = query.lower().split()
        
        # Search entities
        for word in query_words:
            entities = self.knowledge_store.find_entities(name=word, limit=max_results)
            results['entities'].extend([entity.to_dict() for entity in entities])
        
        # Get relationships for found entities
        entity_ids = [entity['entity_id'] for entity in results['entities']]
        for entity_id in entity_ids[:5]:  # Limit to avoid too many results
            relationships = self.knowledge_store.get_relationships(entity_id)
            results['relationships'].extend([rel.to_dict() for rel in relationships])
        
        # Remove duplicates
        seen_entities = set()
        unique_entities = []
        for entity in results['entities']:
            if entity['entity_id'] not in seen_entities:
                unique_entities.append(entity)
                seen_entities.add(entity['entity_id'])
        results['entities'] = unique_entities
        
        seen_relationships = set()
        unique_relationships = []
        for rel in results['relationships']:
            if rel['relationship_id'] not in seen_relationships:
                unique_relationships.append(rel)
                seen_relationships.add(rel['relationship_id'])
        results['relationships'] = unique_relationships
        
        return results
    
    def validate_fact(self, claim: str) -> Dict[str, Any]:
        """Validate a factual claim against the knowledge graph.
        
        Args:
            claim: Factual claim to validate
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'claim': claim,
            'confidence': 0.0,
            'supporting_evidence': [],
            'contradicting_evidence': [],
            'relevant_entities': [],
            'reasoning': ""
        }
        
        # Extract entities from claim
        entities = self._extract_entities_simple(claim)
        validation['relevant_entities'] = [entity.to_dict() for entity in entities]
        
        # Search for supporting/contradicting evidence in the graph
        # This is a simplified version - would be enhanced with proper reasoning
        
        if entities:
            validation['confidence'] = 0.5  # Neutral confidence if entities exist
            validation['reasoning'] = f"Found {len(entities)} relevant entities in knowledge graph"
        else:
            validation['confidence'] = 0.1
            validation['reasoning'] = "No relevant entities found in knowledge graph"
        
        return validation
    
    def get_entity_neighborhood(self, entity_id: str, depth: int = 2) -> Dict[str, Any]:
        """Get the neighborhood of an entity in the graph.
        
        Args:
            entity_id: ID of the central entity
            depth: Depth of neighborhood to explore
            
        Returns:
            Dictionary containing the entity neighborhood
        """
        neighborhood = {
            'center_entity': None,
            'entities': {},
            'relationships': [],
            'depth': depth
        }
        
        # Get center entity
        center_entity = self.knowledge_store.get_entity(entity_id)
        if not center_entity:
            return neighborhood
        
        neighborhood['center_entity'] = center_entity.to_dict()
        neighborhood['entities'][entity_id] = center_entity.to_dict()
        
        # BFS to explore neighborhood
        visited = set([entity_id])
        current_level = set([entity_id])
        
        for level in range(depth):
            next_level = set()
            
            for current_entity_id in current_level:
                relationships = self.knowledge_store.get_relationships(current_entity_id)
                
                for rel in relationships:
                    neighborhood['relationships'].append(rel.to_dict())
                    
                    # Add connected entities
                    connected_id = (rel.target_entity_id 
                                   if rel.source_entity_id == current_entity_id 
                                   else rel.source_entity_id)
                    
                    if connected_id not in visited:
                        connected_entity = self.knowledge_store.get_entity(connected_id)
                        if connected_entity:
                            neighborhood['entities'][connected_id] = connected_entity.to_dict()
                            next_level.add(connected_id)
                            visited.add(connected_id)
            
            current_level = next_level
            if not current_level:
                break
        
        return neighborhood
    
    def close(self):
        """Close the GraphRAG system."""
        if self.knowledge_store:
            self.knowledge_store.close()