"""Advanced graph querying system for knowledge retrieval.

This module provides sophisticated querying capabilities for the knowledge graph,
including semantic search, path finding, subgraph extraction, and complex queries.
"""

import re
import math
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque, Counter
import heapq
import sqlite3
import json
import logging

from .graphrag import (
    Entity, EntityType, Relationship, RelationType, Fact,
    KnowledgeGraphStore, GraphRAGSystem
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of graph queries."""
    ENTITY_LOOKUP = "entity_lookup"
    RELATIONSHIP_SEARCH = "relationship_search"
    PATH_FINDING = "path_finding"
    NEIGHBORHOOD = "neighborhood"
    SUBGRAPH = "subgraph"
    SEMANTIC_SEARCH = "semantic_search"
    FACTUAL_QUERY = "factual_query"
    AGGREGATION = "aggregation"
    PATTERN_MATCHING = "pattern_matching"


class SortOrder(Enum):
    """Sort order options."""
    ASCENDING = "asc"
    DESCENDING = "desc"


@dataclass
class QueryFilter:
    """Filter criteria for graph queries."""
    entity_types: Optional[List[EntityType]] = None
    relation_types: Optional[List[RelationType]] = None
    confidence_min: Optional[float] = None
    confidence_max: Optional[float] = None
    date_from: Optional[float] = None
    date_to: Optional[float] = None
    attributes: Optional[Dict[str, Any]] = None
    sources: Optional[List[str]] = None


@dataclass
class QueryResult:
    """Result of a graph query."""
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    facts: List[Fact] = field(default_factory=list)
    paths: List[List[str]] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    total_results: int = 0
    query_time: float = 0.0


@dataclass
class PathResult:
    """Result of path finding query."""
    path: List[str]
    entities: List[Entity]
    relationships: List[Relationship]
    total_confidence: float
    path_length: int
    path_weight: float


class SemanticMatcher:
    """Semantic matching for entities and relationships."""
    
    def __init__(self):
        """Initialize semantic matcher."""
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'
        }
    
    def calculate_entity_similarity(self, query: str, entity: Entity) -> float:
        """Calculate semantic similarity between query and entity."""
        # Tokenize and clean query
        query_tokens = self._tokenize_and_clean(query)
        
        # Get entity terms
        entity_terms = self._get_entity_terms(entity)
        
        # Calculate various similarity metrics
        name_similarity = self._calculate_text_similarity(query_tokens, 
                                                        self._tokenize_and_clean(entity.name))
        
        # Check aliases
        alias_similarity = 0.0
        for alias in entity.aliases:
            alias_tokens = self._tokenize_and_clean(alias)
            alias_sim = self._calculate_text_similarity(query_tokens, alias_tokens)
            alias_similarity = max(alias_similarity, alias_sim)
        
        # Description similarity
        description_similarity = 0.0
        if entity.description:
            desc_tokens = self._tokenize_and_clean(entity.description)
            description_similarity = self._calculate_text_similarity(query_tokens, desc_tokens)
        
        # Attributes similarity
        attributes_similarity = 0.0
        for key, value in entity.attributes.items():
            if isinstance(value, str):
                attr_tokens = self._tokenize_and_clean(f"{key} {value}")
                attr_sim = self._calculate_text_similarity(query_tokens, attr_tokens)
                attributes_similarity = max(attributes_similarity, attr_sim)
        
        # Combine similarities with weights
        total_similarity = (
            name_similarity * 0.4 +
            alias_similarity * 0.3 +
            description_similarity * 0.2 +
            attributes_similarity * 0.1
        )
        
        return min(1.0, total_similarity)
    
    def calculate_relationship_similarity(self, query: str, relationship: Relationship) -> float:
        """Calculate semantic similarity between query and relationship."""
        query_tokens = self._tokenize_and_clean(query)
        
        # Relationship type similarity
        rel_type_tokens = self._tokenize_and_clean(relationship.relation_type.value.replace('_', ' '))
        type_similarity = self._calculate_text_similarity(query_tokens, rel_type_tokens)
        
        # Attributes similarity
        attributes_similarity = 0.0
        for key, value in relationship.attributes.items():
            if isinstance(value, str):
                attr_tokens = self._tokenize_and_clean(f"{key} {value}")
                attr_sim = self._calculate_text_similarity(query_tokens, attr_tokens)
                attributes_similarity = max(attributes_similarity, attr_sim)
        
        return (type_similarity * 0.7 + attributes_similarity * 0.3)
    
    def _tokenize_and_clean(self, text: str) -> List[str]:
        """Tokenize and clean text."""
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        
        # Remove stop words and short tokens
        filtered_tokens = [
            token for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return filtered_tokens
    
    def _get_entity_terms(self, entity: Entity) -> List[str]:
        """Get all searchable terms for an entity."""
        terms = []
        
        # Add name
        terms.extend(self._tokenize_and_clean(entity.name))
        
        # Add aliases
        for alias in entity.aliases:
            terms.extend(self._tokenize_and_clean(alias))
        
        # Add description
        if entity.description:
            terms.extend(self._tokenize_and_clean(entity.description))
        
        # Add entity type
        terms.extend(self._tokenize_and_clean(entity.entity_type.value))
        
        return terms
    
    def _calculate_text_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """Calculate similarity between two token lists."""
        if not tokens1 or not tokens2:
            return 0.0
        
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        # Jaccard similarity
        intersection = set1 & set2
        union = set1 | set2
        
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Cosine similarity with term frequency
        tf1 = Counter(tokens1)
        tf2 = Counter(tokens2)
        
        # Calculate dot product
        dot_product = sum(tf1[term] * tf2[term] for term in intersection)
        
        # Calculate magnitudes
        mag1 = math.sqrt(sum(count ** 2 for count in tf1.values()))
        mag2 = math.sqrt(sum(count ** 2 for count in tf2.values()))
        
        cosine = dot_product / (mag1 * mag2) if mag1 * mag2 > 0 else 0.0
        
        # Combine similarities
        return (jaccard * 0.6 + cosine * 0.4)


class PathFinder:
    """Advanced path finding algorithms for the knowledge graph."""
    
    def __init__(self, knowledge_store: KnowledgeGraphStore):
        """Initialize path finder with knowledge store."""
        self.knowledge_store = knowledge_store
        self.adjacency_cache = {}
        self.entity_cache = {}
    
    def find_shortest_paths(self, source_id: str, target_id: str, 
                          max_paths: int = 5, max_depth: int = 4) -> List[PathResult]:
        """Find shortest paths between two entities using Dijkstra's algorithm."""
        # Build adjacency graph with weights
        adjacency = self._build_weighted_adjacency()
        
        if source_id not in adjacency or target_id not in adjacency:
            return []
        
        # Priority queue: (distance, current_node, path, relationships)
        pq = [(0.0, source_id, [source_id], [])]
        visited = set()
        paths_found = []
        
        while pq and len(paths_found) < max_paths:
            distance, current, path, relationships = heapq.heappop(pq)
            
            if len(path) > max_depth:
                continue
            
            if current == target_id:
                # Found a path
                path_result = self._create_path_result(path, relationships, distance)
                paths_found.append(path_result)
                continue
            
            if current in visited:
                continue
            
            visited.add(current)
            
            # Explore neighbors
            for neighbor, edge_data in adjacency[current].items():
                if neighbor not in path:  # Avoid cycles
                    new_distance = distance + edge_data['weight']
                    new_path = path + [neighbor]
                    new_relationships = relationships + [edge_data['relationship']]
                    
                    heapq.heappush(pq, (new_distance, neighbor, new_path, new_relationships))
        
        return sorted(paths_found, key=lambda x: x.path_weight)
    
    def find_paths_with_constraints(self, source_id: str, target_id: str,
                                   relation_types: List[RelationType] = None,
                                   min_confidence: float = 0.0,
                                   max_depth: int = 4) -> List[PathResult]:
        """Find paths with specific constraints."""
        adjacency = self._build_filtered_adjacency(relation_types, min_confidence)
        
        if source_id not in adjacency or target_id not in adjacency:
            return []
        
        # BFS with constraints
        queue = deque([(source_id, [source_id], [], 0.0)])
        visited_paths = set()
        paths_found = []
        
        while queue and len(paths_found) < 10:
            current, path, relationships, total_weight = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            path_key = tuple(path)
            if path_key in visited_paths:
                continue
            visited_paths.add(path_key)
            
            if current == target_id:
                path_result = self._create_path_result(path, relationships, total_weight)
                paths_found.append(path_result)
                continue
            
            # Explore neighbors
            for neighbor, edge_data in adjacency[current].items():
                if neighbor not in path:  # Avoid cycles
                    new_path = path + [neighbor]
                    new_relationships = relationships + [edge_data['relationship']]
                    new_weight = total_weight + edge_data['weight']
                    
                    queue.append((neighbor, new_path, new_relationships, new_weight))
        
        return sorted(paths_found, key=lambda x: x.total_confidence, reverse=True)
    
    def find_multi_hop_relationships(self, entity_id: str, target_types: List[EntityType],
                                   max_hops: int = 3) -> Dict[str, List[PathResult]]:
        """Find entities of specific types within max_hops of source entity."""
        results = defaultdict(list)
        
        # BFS to find entities of target types
        queue = deque([(entity_id, [entity_id], [], 0.0)])
        visited = set()
        
        while queue:
            current, path, relationships, total_weight = queue.popleft()
            
            if len(path) > max_hops:
                continue
            
            if current in visited:
                continue
            visited.add(current)
            
            # Check if current entity matches target types
            current_entity = self._get_entity(current)
            if current_entity and current_entity.entity_type in target_types and current != entity_id:
                path_result = self._create_path_result(path, relationships, total_weight)
                results[current_entity.entity_type.value].append(path_result)
            
            # Explore neighbors
            adjacency = self._build_weighted_adjacency()
            if current in adjacency:
                for neighbor, edge_data in adjacency[current].items():
                    if neighbor not in path:  # Avoid cycles
                        new_path = path + [neighbor]
                        new_relationships = relationships + [edge_data['relationship']]
                        new_weight = total_weight + edge_data['weight']
                        
                        queue.append((neighbor, new_path, new_relationships, new_weight))
        
        # Sort results by confidence
        for entity_type in results:
            results[entity_type].sort(key=lambda x: x.total_confidence, reverse=True)
        
        return dict(results)
    
    def _build_weighted_adjacency(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Build weighted adjacency graph."""
        if 'weighted' in self.adjacency_cache:
            return self.adjacency_cache['weighted']
        
        adjacency = defaultdict(lambda: defaultdict(dict))
        
        # Get all relationships
        cursor = self.knowledge_store.conn.execute("""
            SELECT relationship_id, source_entity_id, target_entity_id, 
                   relation_type, confidence, attributes
            FROM relationships
        """)
        
        for row in cursor.fetchall():
            rel_id, source_id, target_id, rel_type, confidence, attributes_json = row
            
            # Calculate edge weight (lower is better for shortest path)
            weight = 1.0 / (confidence + 0.1)  # Avoid division by zero
            
            relationship_data = {
                'relationship': Relationship(
                    relationship_id=rel_id,
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    relation_type=RelationType(rel_type),
                    confidence=confidence,
                    attributes=json.loads(attributes_json or '{}')
                ),
                'weight': weight,
                'confidence': confidence
            }
            
            # Add both directions (treat as undirected for path finding)
            adjacency[source_id][target_id] = relationship_data
            adjacency[target_id][source_id] = relationship_data
        
        adjacency_dict = {k: dict(v) for k, v in adjacency.items()}
        self.adjacency_cache['weighted'] = adjacency_dict
        return adjacency_dict
    
    def _build_filtered_adjacency(self, relation_types: List[RelationType] = None,
                                min_confidence: float = 0.0) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Build filtered adjacency graph."""
        adjacency = defaultdict(lambda: defaultdict(dict))
        
        # Build query with filters
        query = """
            SELECT relationship_id, source_entity_id, target_entity_id, 
                   relation_type, confidence, attributes
            FROM relationships
            WHERE confidence >= ?
        """
        params = [min_confidence]
        
        if relation_types:
            placeholders = ','.join('?' * len(relation_types))
            query += f" AND relation_type IN ({placeholders})"
            params.extend([rt.value for rt in relation_types])
        
        cursor = self.knowledge_store.conn.execute(query, params)
        
        for row in cursor.fetchall():
            rel_id, source_id, target_id, rel_type, confidence, attributes_json = row
            
            weight = 1.0 / (confidence + 0.1)
            
            relationship_data = {
                'relationship': Relationship(
                    relationship_id=rel_id,
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    relation_type=RelationType(rel_type),
                    confidence=confidence,
                    attributes=json.loads(attributes_json or '{}')
                ),
                'weight': weight,
                'confidence': confidence
            }
            
            adjacency[source_id][target_id] = relationship_data
            adjacency[target_id][source_id] = relationship_data
        
        return {k: dict(v) for k, v in adjacency.items()}
    
    def _create_path_result(self, path: List[str], relationships: List[Relationship],
                          total_weight: float) -> PathResult:
        """Create PathResult from path data."""
        entities = []
        for entity_id in path:
            entity = self._get_entity(entity_id)
            if entity:
                entities.append(entity)
        
        # Calculate total confidence
        if relationships:
            total_confidence = sum(rel.confidence for rel in relationships) / len(relationships)
        else:
            total_confidence = 1.0
        
        return PathResult(
            path=path,
            entities=entities,
            relationships=relationships,
            total_confidence=total_confidence,
            path_length=len(path) - 1,
            path_weight=total_weight
        )
    
    def _get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity from cache or database."""
        if entity_id in self.entity_cache:
            return self.entity_cache[entity_id]
        
        entity = self.knowledge_store.get_entity(entity_id)
        if entity:
            self.entity_cache[entity_id] = entity
        
        return entity


class GraphQueryEngine:
    """Main graph query engine with advanced querying capabilities."""
    
    def __init__(self, knowledge_store: KnowledgeGraphStore):
        """Initialize query engine."""
        self.knowledge_store = knowledge_store
        self.semantic_matcher = SemanticMatcher()
        self.path_finder = PathFinder(knowledge_store)
        
    def execute_query(self, query: str, query_type: QueryType = QueryType.SEMANTIC_SEARCH,
                     filters: QueryFilter = None, limit: int = 100,
                     sort_by: str = "confidence", sort_order: SortOrder = SortOrder.DESCENDING) -> QueryResult:
        """Execute a graph query.
        
        Args:
            query: Natural language query or structured query
            query_type: Type of query to execute
            filters: Optional filters to apply
            limit: Maximum number of results
            sort_by: Field to sort by
            sort_order: Sort order
            
        Returns:
            QueryResult with matching entities, relationships, and metadata
        """
        import time
        start_time = time.time()
        
        result = QueryResult()
        
        try:
            if query_type == QueryType.ENTITY_LOOKUP:
                result = self._entity_lookup(query, filters, limit)
            elif query_type == QueryType.RELATIONSHIP_SEARCH:
                result = self._relationship_search(query, filters, limit)
            elif query_type == QueryType.SEMANTIC_SEARCH:
                result = self._semantic_search(query, filters, limit)
            elif query_type == QueryType.NEIGHBORHOOD:
                result = self._neighborhood_search(query, filters, limit)
            elif query_type == QueryType.PATH_FINDING:
                result = self._path_finding_search(query, filters, limit)
            elif query_type == QueryType.FACTUAL_QUERY:
                result = self._factual_query(query, filters, limit)
            elif query_type == QueryType.AGGREGATION:
                result = self._aggregation_query(query, filters, limit)
            elif query_type == QueryType.PATTERN_MATCHING:
                result = self._pattern_matching_query(query, filters, limit)
            else:
                raise ValueError(f"Unsupported query type: {query_type}")
            
            # Apply sorting
            result = self._sort_results(result, sort_by, sort_order)
            
            # Apply limit
            result = self._limit_results(result, limit)
            
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            result.metadata['error'] = str(e)
        
        result.query_time = time.time() - start_time
        return result
    
    def _entity_lookup(self, query: str, filters: QueryFilter, limit: int) -> QueryResult:
        """Execute entity lookup query."""
        result = QueryResult()
        
        # Parse query for entity name
        entity_name = query.strip()
        
        # Search entities
        entities = self.knowledge_store.find_entities(name=entity_name, limit=limit)
        
        # Apply filters
        if filters:
            entities = self._apply_entity_filters(entities, filters)
        
        # Calculate scores
        scores = {}
        for entity in entities:
            score = self.semantic_matcher.calculate_entity_similarity(query, entity)
            scores[entity.entity_id] = score
        
        result.entities = entities
        result.scores = scores
        result.total_results = len(entities)
        result.metadata = {'query_type': 'entity_lookup', 'original_query': query}
        
        return result
    
    def _relationship_search(self, query: str, filters: QueryFilter, limit: int) -> QueryResult:
        """Execute relationship search query."""
        result = QueryResult()
        
        # Get all relationships
        cursor = self.knowledge_store.conn.execute("""
            SELECT relationship_id, source_entity_id, target_entity_id, 
                   relation_type, attributes, confidence, sources, created_at, updated_at
            FROM relationships
            ORDER BY confidence DESC
            LIMIT ?
        """, (limit,))
        
        relationships = []
        scores = {}
        
        for row in cursor.fetchall():
            relationship = Relationship(
                relationship_id=row[0],
                source_entity_id=row[1],
                target_entity_id=row[2],
                relation_type=RelationType(row[3]),
                attributes=json.loads(row[4] or '{}'),
                confidence=row[5],
                sources=json.loads(row[6] or '[]'),
                created_at=row[7],
                updated_at=row[8]
            )
            
            # Calculate similarity score
            score = self.semantic_matcher.calculate_relationship_similarity(query, relationship)
            if score > 0.1:  # Threshold for relevance
                relationships.append(relationship)
                scores[relationship.relationship_id] = score
        
        # Apply filters
        if filters:
            relationships = self._apply_relationship_filters(relationships, filters)
        
        result.relationships = relationships
        result.scores = scores
        result.total_results = len(relationships)
        result.metadata = {'query_type': 'relationship_search', 'original_query': query}
        
        return result
    
    def _semantic_search(self, query: str, filters: QueryFilter, limit: int) -> QueryResult:
        """Execute semantic search across entities and relationships."""
        result = QueryResult()
        
        # Search entities
        all_entities = self.knowledge_store.find_entities(limit=limit * 2)
        entity_scores = {}
        relevant_entities = []
        
        for entity in all_entities:
            score = self.semantic_matcher.calculate_entity_similarity(query, entity)
            if score > 0.2:  # Threshold for semantic relevance
                relevant_entities.append(entity)
                entity_scores[entity.entity_id] = score
        
        # Search relationships
        cursor = self.knowledge_store.conn.execute("""
            SELECT relationship_id, source_entity_id, target_entity_id, 
                   relation_type, attributes, confidence, sources, created_at, updated_at
            FROM relationships
            ORDER BY confidence DESC
            LIMIT ?
        """, (limit,))
        
        relevant_relationships = []
        relationship_scores = {}
        
        for row in cursor.fetchall():
            relationship = Relationship(
                relationship_id=row[0],
                source_entity_id=row[1],
                target_entity_id=row[2],
                relation_type=RelationType(row[3]),
                attributes=json.loads(row[4] or '{}'),
                confidence=row[5],
                sources=json.loads(row[6] or '[]'),
                created_at=row[7],
                updated_at=row[8]
            )
            
            score = self.semantic_matcher.calculate_relationship_similarity(query, relationship)
            if score > 0.2:
                relevant_relationships.append(relationship)
                relationship_scores[relationship.relationship_id] = score
        
        # Apply filters
        if filters:
            relevant_entities = self._apply_entity_filters(relevant_entities, filters)
            relevant_relationships = self._apply_relationship_filters(relevant_relationships, filters)
        
        # Combine scores
        all_scores = {}
        all_scores.update(entity_scores)
        all_scores.update(relationship_scores)
        
        result.entities = relevant_entities
        result.relationships = relevant_relationships
        result.scores = all_scores
        result.total_results = len(relevant_entities) + len(relevant_relationships)
        result.metadata = {'query_type': 'semantic_search', 'original_query': query}
        
        return result
    
    def _neighborhood_search(self, query: str, filters: QueryFilter, limit: int) -> QueryResult:
        """Execute neighborhood search around entities."""
        result = QueryResult()
        
        # First find the central entity
        entities = self.knowledge_store.find_entities(name=query, limit=5)
        if not entities:
            return result
        
        central_entity = entities[0]  # Take the best match
        
        # Get neighborhood
        neighborhood = self.knowledge_store.query_knowledge(query, max_results=limit)
        
        # Extract entities and relationships from neighborhood
        result.entities = [Entity.from_dict(e) for e in neighborhood['entities']]
        result.relationships = [Relationship.from_dict(r) for r in neighborhood['relationships']]
        
        # Calculate scores based on distance from central entity
        scores = {}
        for entity in result.entities:
            if entity.entity_id == central_entity.entity_id:
                scores[entity.entity_id] = 1.0
            else:
                # Find shortest path to central entity
                paths = self.path_finder.find_shortest_paths(
                    central_entity.entity_id, entity.entity_id, max_paths=1, max_depth=3
                )
                if paths:
                    scores[entity.entity_id] = 1.0 / (paths[0].path_length + 1)
                else:
                    scores[entity.entity_id] = 0.1
        
        result.scores = scores
        result.total_results = len(result.entities) + len(result.relationships)
        result.metadata = {
            'query_type': 'neighborhood',
            'central_entity': central_entity.entity_id,
            'original_query': query
        }
        
        return result
    
    def _path_finding_search(self, query: str, filters: QueryFilter, limit: int) -> QueryResult:
        """Execute path finding query."""
        result = QueryResult()
        
        # Parse query for source and target entities
        # Expected format: "path from X to Y" or "X -> Y"
        path_patterns = [
            r'path from (.+) to (.+)',
            r'(.+) -> (.+)',
            r'(.+) to (.+)',
            r'connect (.+) and (.+)'
        ]
        
        source_name = None
        target_name = None
        
        for pattern in path_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                source_name = match.group(1).strip()
                target_name = match.group(2).strip()
                break
        
        if not source_name or not target_name:
            result.metadata = {'error': 'Could not parse path query. Expected format: "path from X to Y"'}
            return result
        
        # Find source and target entities
        source_entities = self.knowledge_store.find_entities(name=source_name, limit=1)
        target_entities = self.knowledge_store.find_entities(name=target_name, limit=1)
        
        if not source_entities or not target_entities:
            result.metadata = {'error': 'Could not find source or target entities'}
            return result
        
        source_entity = source_entities[0]
        target_entity = target_entities[0]
        
        # Find paths
        paths = self.path_finder.find_shortest_paths(
            source_entity.entity_id, target_entity.entity_id,
            max_paths=limit, max_depth=5
        )
        
        # Extract entities and relationships from paths
        all_entities = {}
        all_relationships = []
        path_data = []
        
        for path_result in paths:
            path_data.append(path_result.path)
            
            for entity in path_result.entities:
                all_entities[entity.entity_id] = entity
            
            all_relationships.extend(path_result.relationships)
        
        result.entities = list(all_entities.values())
        result.relationships = all_relationships
        result.paths = path_data
        result.total_results = len(paths)
        result.metadata = {
            'query_type': 'path_finding',
            'source_entity': source_entity.entity_id,
            'target_entity': target_entity.entity_id,
            'original_query': query
        }
        
        return result
    
    def _factual_query(self, query: str, filters: QueryFilter, limit: int) -> QueryResult:
        """Execute factual query."""
        result = QueryResult()
        
        # Get all facts
        cursor = self.knowledge_store.conn.execute("""
            SELECT fact_id, claim, entities, relationships, confidence,
                   evidence, sources, contradictions, created_at, updated_at
            FROM facts
            ORDER BY confidence DESC
            LIMIT ?
        """, (limit,))
        
        facts = []
        scores = {}
        
        for row in cursor.fetchall():
            fact = Fact(
                fact_id=row[0],
                claim=row[1],
                entities=json.loads(row[2] or '[]'),
                relationships=json.loads(row[3] or '[]'),
                confidence=row[4],
                evidence=json.loads(row[5] or '[]'),
                sources=json.loads(row[6] or '[]'),
                contradictions=json.loads(row[7] or '[]'),
                created_at=row[8],
                updated_at=row[9]
            )
            
            # Calculate relevance score
            score = self.semantic_matcher._calculate_text_similarity(
                self.semantic_matcher._tokenize_and_clean(query),
                self.semantic_matcher._tokenize_and_clean(fact.claim)
            )
            
            if score > 0.1:
                facts.append(fact)
                scores[fact.fact_id] = score
        
        result.facts = facts
        result.scores = scores
        result.total_results = len(facts)
        result.metadata = {'query_type': 'factual_query', 'original_query': query}
        
        return result
    
    def _aggregation_query(self, query: str, filters: QueryFilter, limit: int) -> QueryResult:
        """Execute aggregation query."""
        result = QueryResult()
        
        # Get statistics from knowledge store
        stats = self.knowledge_store.get_statistics()
        
        result.metadata = {
            'query_type': 'aggregation',
            'statistics': stats,
            'original_query': query
        }
        
        return result
    
    def _pattern_matching_query(self, query: str, filters: QueryFilter, limit: int) -> QueryResult:
        """Execute pattern matching query."""
        result = QueryResult()
        
        # Simple pattern matching for entity-relationship-entity patterns
        # Expected format: "(EntityType)-[RelationType]-(EntityType)"
        pattern_match = re.match(r'\((\w+)\)-\[(\w+)\]-\((\w+)\)', query)
        
        if pattern_match:
            source_type = pattern_match.group(1)
            relation_type = pattern_match.group(2)
            target_type = pattern_match.group(3)
            
            # Find matching patterns
            query_sql = """
                SELECT DISTINCT e1.entity_id as source_id, e2.entity_id as target_id,
                       r.relationship_id, r.confidence
                FROM entities e1
                JOIN relationships r ON e1.entity_id = r.source_entity_id
                JOIN entities e2 ON r.target_entity_id = e2.entity_id
                WHERE e1.entity_type = ? AND r.relation_type = ? AND e2.entity_type = ?
                ORDER BY r.confidence DESC
                LIMIT ?
            """
            
            try:
                cursor = self.knowledge_store.conn.execute(
                    query_sql, (source_type, relation_type, target_type, limit)
                )
                
                entity_ids = set()
                relationship_ids = set()
                scores = {}
                
                for row in cursor.fetchall():
                    source_id, target_id, rel_id, confidence = row
                    entity_ids.add(source_id)
                    entity_ids.add(target_id)
                    relationship_ids.add(rel_id)
                    scores[rel_id] = confidence
                
                # Get entities and relationships
                entities = [self.knowledge_store.get_entity(eid) for eid in entity_ids]
                entities = [e for e in entities if e is not None]
                
                relationships = []
                for rel_id in relationship_ids:
                    cursor = self.knowledge_store.conn.execute(
                        "SELECT * FROM relationships WHERE relationship_id = ?", (rel_id,)
                    )
                    row = cursor.fetchone()
                    if row:
                        rel = Relationship(
                            relationship_id=row[0],
                            source_entity_id=row[1],
                            target_entity_id=row[2],
                            relation_type=RelationType(row[3]),
                            attributes=json.loads(row[4] or '{}'),
                            confidence=row[5],
                            sources=json.loads(row[6] or '[]'),
                            created_at=row[7],
                            updated_at=row[8]
                        )
                        relationships.append(rel)
                
                result.entities = entities
                result.relationships = relationships
                result.scores = scores
                result.total_results = len(relationships)
                
            except Exception as e:
                result.metadata['error'] = f"Pattern matching failed: {str(e)}"
        
        else:
            result.metadata['error'] = "Invalid pattern format. Expected: (EntityType)-[RelationType]-(EntityType)"
        
        result.metadata.update({
            'query_type': 'pattern_matching',
            'original_query': query
        })
        
        return result
    
    def _apply_entity_filters(self, entities: List[Entity], filters: QueryFilter) -> List[Entity]:
        """Apply filters to entity list."""
        filtered_entities = []
        
        for entity in entities:
            if self._entity_matches_filters(entity, filters):
                filtered_entities.append(entity)
        
        return filtered_entities
    
    def _apply_relationship_filters(self, relationships: List[Relationship], filters: QueryFilter) -> List[Relationship]:
        """Apply filters to relationship list."""
        filtered_relationships = []
        
        for relationship in relationships:
            if self._relationship_matches_filters(relationship, filters):
                filtered_relationships.append(relationship)
        
        return filtered_relationships
    
    def _entity_matches_filters(self, entity: Entity, filters: QueryFilter) -> bool:
        """Check if entity matches filters."""
        if filters.entity_types and entity.entity_type not in filters.entity_types:
            return False
        
        if filters.confidence_min and entity.confidence < filters.confidence_min:
            return False
        
        if filters.confidence_max and entity.confidence > filters.confidence_max:
            return False
        
        if filters.date_from and entity.created_at < filters.date_from:
            return False
        
        if filters.date_to and entity.created_at > filters.date_to:
            return False
        
        if filters.sources:
            if not any(source in entity.sources for source in filters.sources):
                return False
        
        if filters.attributes:
            for key, value in filters.attributes.items():
                if key not in entity.attributes or entity.attributes[key] != value:
                    return False
        
        return True
    
    def _relationship_matches_filters(self, relationship: Relationship, filters: QueryFilter) -> bool:
        """Check if relationship matches filters."""
        if filters.relation_types and relationship.relation_type not in filters.relation_types:
            return False
        
        if filters.confidence_min and relationship.confidence < filters.confidence_min:
            return False
        
        if filters.confidence_max and relationship.confidence > filters.confidence_max:
            return False
        
        if filters.date_from and relationship.created_at < filters.date_from:
            return False
        
        if filters.date_to and relationship.created_at > filters.date_to:
            return False
        
        if filters.sources:
            if not any(source in relationship.sources for source in filters.sources):
                return False
        
        if filters.attributes:
            for key, value in filters.attributes.items():
                if key not in relationship.attributes or relationship.attributes[key] != value:
                    return False
        
        return True
    
    def _sort_results(self, result: QueryResult, sort_by: str, sort_order: SortOrder) -> QueryResult:
        """Sort query results."""
        reverse = (sort_order == SortOrder.DESCENDING)
        
        if sort_by == "confidence":
            # Sort entities by confidence
            result.entities.sort(key=lambda x: x.confidence, reverse=reverse)
            # Sort relationships by confidence
            result.relationships.sort(key=lambda x: x.confidence, reverse=reverse)
        
        elif sort_by == "score" and result.scores:
            # Sort by relevance scores
            result.entities.sort(
                key=lambda x: result.scores.get(x.entity_id, 0.0), reverse=reverse
            )
            result.relationships.sort(
                key=lambda x: result.scores.get(x.relationship_id, 0.0), reverse=reverse
            )
        
        elif sort_by == "created_at":
            result.entities.sort(key=lambda x: x.created_at, reverse=reverse)
            result.relationships.sort(key=lambda x: x.created_at, reverse=reverse)
        
        elif sort_by == "updated_at":
            result.entities.sort(key=lambda x: x.updated_at, reverse=reverse)
            result.relationships.sort(key=lambda x: x.updated_at, reverse=reverse)
        
        return result
    
    def _limit_results(self, result: QueryResult, limit: int) -> QueryResult:
        """Apply limit to query results."""
        result.entities = result.entities[:limit]
        result.relationships = result.relationships[:limit]
        result.facts = result.facts[:limit]
        result.paths = result.paths[:limit]
        
        return result