"""
RAG (Retrieval-Augmented Generation) System

Advanced RAG implementation with vector databases, corrective RAG,
GraphRAG, and self-reflective RAG capabilities.
"""

from .vector_database import (
    VectorConfig,
    DocumentMetadata,
    VectorSearchResult,
    BaseVectorDatabase,
    ChromaVectorDatabase,
    FAISSVectorDatabase,
    VectorDatabaseManager
)

from .corrective_rag import (
    CRAGConfig,
    QualityAssessment,
    CorrectionResult,
    RetrievalQuality,
    CorrectionAction,
    CorrectiveRAG
)

from .graph_rag import (
    GraphRAGConfig,
    Entity,
    Relationship,
    EntityType,
    RelationType,
    KnowledgeGraph,
    GraphRAG
)

from .self_reflective_rag import (
    SelfReflectiveRAGConfig,
    ReflectionToken,
    ReflectionResult,
    ReflectionType,
    ReflectionLevel,
    SelfReflectiveRAG
)

__all__ = [
    # Vector database
    'VectorConfig',
    'DocumentMetadata', 
    'VectorSearchResult',
    'BaseVectorDatabase',
    'ChromaVectorDatabase',
    'FAISSVectorDatabase',
    'VectorDatabaseManager',
    
    # Corrective RAG
    'CRAGConfig',
    'QualityAssessment',
    'CorrectionResult',
    'RetrievalQuality',
    'CorrectionAction',
    'CorrectiveRAG',
    
    # Graph RAG
    'GraphRAGConfig',
    'Entity',
    'Relationship',
    'EntityType',
    'RelationType',
    'KnowledgeGraph',
    'GraphRAG',
    
    # Self-Reflective RAG
    'SelfReflectiveRAGConfig',
    'ReflectionToken',
    'ReflectionResult',
    'ReflectionType',
    'ReflectionLevel',
    'SelfReflectiveRAG'
]