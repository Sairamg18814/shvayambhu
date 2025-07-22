"""
Vector Database Manager

Comprehensive vector database implementation supporting Chroma and FAISS
with consciousness-aware embeddings and advanced retrieval capabilities.
"""

import asyncio
import numpy as np
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
from collections import defaultdict
from abc import ABC, abstractmethod

# Vector database implementations
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not available")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available")

logger = logging.getLogger(__name__)


@dataclass
class VectorConfig:
    """Configuration for vector database operations"""
    # Database settings
    database_type: str = "chroma"  # "chroma", "faiss", or "hybrid"
    vector_dimension: int = 768  # Default for sentence transformers
    similarity_metric: str = "cosine"  # "cosine", "euclidean", "dot_product"
    
    # Storage settings
    persist_directory: str = "data/vector_db"
    collection_name: str = "shvayambhu_knowledge"
    
    # Performance settings
    batch_size: int = 100
    max_results: int = 10
    similarity_threshold: float = 0.7
    
    # Consciousness-aware settings
    consciousness_weight: float = 1.2  # Boost for consciousness-related content
    consciousness_keywords: List[str] = field(default_factory=lambda: [
        'consciousness', 'awareness', 'experience', 'subjective', 'qualia',
        'phenomenal', 'introspection', 'self-aware', 'sentient', 'cognitive'
    ])
    
    # Index optimization
    enable_indexing: bool = True
    index_refresh_interval: int = 1000  # Refresh after N additions
    
    # Memory management
    max_cache_size: int = 10000
    cleanup_threshold: float = 0.8  # Cleanup when 80% full


@dataclass
class DocumentMetadata:
    """Metadata for documents stored in vector database"""
    document_id: str
    source: str
    title: str
    content_type: str
    timestamp: datetime
    keywords: List[str]
    consciousness_score: float
    relevance_score: float
    chunk_index: int = 0
    total_chunks: int = 1
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorSearchResult:
    """Result from vector similarity search"""
    document_id: str
    content: str
    similarity_score: float
    metadata: DocumentMetadata
    embedding: Optional[np.ndarray] = None


class BaseVectorDatabase(ABC):
    """Abstract base class for vector databases"""
    
    def __init__(self, config: VectorConfig):
        self.config = config
        self.dimension = config.vector_dimension
        self.collection_name = config.collection_name
        
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the vector database"""
        pass
    
    @abstractmethod
    async def add_documents(
        self,
        documents: List[str],
        embeddings: List[np.ndarray],
        metadatas: List[DocumentMetadata]
    ) -> List[str]:
        """Add documents with their embeddings and metadata"""
        pass
    
    @abstractmethod
    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar vectors"""
        pass
    
    @abstractmethod
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the database"""
        pass
    
    @abstractmethod
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        pass


class ChromaVectorDatabase(BaseVectorDatabase):
    """ChromaDB implementation of vector database"""
    
    def __init__(self, config: VectorConfig):
        super().__init__(config)
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is not available")
        
        self.client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None
    
    async def initialize(self) -> bool:
        """Initialize ChromaDB"""
        
        try:
            # Create persistent client
            persist_path = Path(self.config.persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)
            
            settings = Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(persist_path)
            )
            
            self.client = chromadb.Client(settings)
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=None  # We provide embeddings directly
                )
                logger.info(f"Loaded existing ChromaDB collection: {self.collection_name}")
            except Exception:
                # Create new collection
                distance_function = {
                    "cosine": "cosine",
                    "euclidean": "l2",
                    "dot_product": "ip"
                }.get(self.config.similarity_metric, "cosine")
                
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": distance_function},
                    embedding_function=None
                )
                logger.info(f"Created new ChromaDB collection: {self.collection_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            return False
    
    async def add_documents(
        self,
        documents: List[str],
        embeddings: List[np.ndarray],
        metadatas: List[DocumentMetadata]
    ) -> List[str]:
        """Add documents to ChromaDB"""
        
        if not self.collection:
            raise ValueError("Collection not initialized")
        
        if len(documents) != len(embeddings) or len(documents) != len(metadatas):
            raise ValueError("Documents, embeddings, and metadatas must have the same length")
        
        try:
            # Convert embeddings to list format for ChromaDB
            embeddings_list = [emb.tolist() for emb in embeddings]
            
            # Convert metadata to ChromaDB format
            chroma_metadatas = []
            for metadata in metadatas:
                chroma_metadata = {
                    "source": metadata.source,
                    "title": metadata.title,
                    "content_type": metadata.content_type,
                    "timestamp": metadata.timestamp.isoformat(),
                    "keywords": json.dumps(metadata.keywords),
                    "consciousness_score": metadata.consciousness_score,
                    "relevance_score": metadata.relevance_score,
                    "chunk_index": metadata.chunk_index,
                    "total_chunks": metadata.total_chunks,
                }
                # Add custom metadata
                chroma_metadata.update(metadata.custom_metadata)
                chroma_metadatas.append(chroma_metadata)
            
            # Generate IDs
            document_ids = [metadata.document_id for metadata in metadatas]
            
            # Add to collection
            self.collection.add(
                ids=document_ids,
                documents=documents,
                embeddings=embeddings_list,
                metadatas=chroma_metadatas
            )
            
            logger.debug(f"Added {len(documents)} documents to ChromaDB")
            return document_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise
    
    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search ChromaDB for similar vectors"""
        
        if not self.collection:
            raise ValueError("Collection not initialized")
        
        try:
            # Convert embedding to list
            query_embedding_list = query_embedding.tolist()
            
            # Prepare where clause for filtering
            where_clause = {}
            if filters:
                for key, value in filters.items():
                    if key in ["source", "content_type", "title"]:
                        where_clause[key] = value
                    elif key == "consciousness_score_min":
                        where_clause["consciousness_score"] = {"$gte": value}
                    elif key == "relevance_score_min":
                        where_clause["relevance_score"] = {"$gte": value}
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding_list],
                n_results=min(top_k, self.config.max_results),
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances", "embeddings"]
            )
            
            # Convert results to VectorSearchResult format
            search_results = []
            if results["ids"] and results["ids"][0]:  # Check if results exist
                for i in range(len(results["ids"][0])):
                    # Convert distance to similarity score
                    distance = results["distances"][0][i]
                    if self.config.similarity_metric == "cosine":
                        similarity_score = 1.0 - distance
                    elif self.config.similarity_metric == "euclidean":
                        similarity_score = 1.0 / (1.0 + distance)
                    else:  # dot product
                        similarity_score = distance
                    
                    # Parse metadata
                    chroma_metadata = results["metadatas"][0][i]
                    metadata = DocumentMetadata(
                        document_id=results["ids"][0][i],
                        source=chroma_metadata.get("source", ""),
                        title=chroma_metadata.get("title", ""),
                        content_type=chroma_metadata.get("content_type", ""),
                        timestamp=datetime.fromisoformat(chroma_metadata.get("timestamp", datetime.now().isoformat())),
                        keywords=json.loads(chroma_metadata.get("keywords", "[]")),
                        consciousness_score=chroma_metadata.get("consciousness_score", 0.0),
                        relevance_score=chroma_metadata.get("relevance_score", 0.0),
                        chunk_index=chroma_metadata.get("chunk_index", 0),
                        total_chunks=chroma_metadata.get("total_chunks", 1),
                        custom_metadata={k: v for k, v in chroma_metadata.items() 
                                       if k not in ["source", "title", "content_type", "timestamp", 
                                                  "keywords", "consciousness_score", "relevance_score",
                                                  "chunk_index", "total_chunks"]}
                    )
                    
                    # Get embedding if requested
                    embedding = None
                    if results.get("embeddings") and results["embeddings"][0]:
                        embedding = np.array(results["embeddings"][0][i])
                    
                    search_result = VectorSearchResult(
                        document_id=results["ids"][0][i],
                        content=results["documents"][0][i],
                        similarity_score=similarity_score,
                        metadata=metadata,
                        embedding=embedding
                    )
                    
                    search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search ChromaDB: {e}")
            raise
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from ChromaDB"""
        
        if not self.collection:
            raise ValueError("Collection not initialized")
        
        try:
            self.collection.delete(ids=document_ids)
            logger.debug(f"Deleted {len(document_ids)} documents from ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents from ChromaDB: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get ChromaDB collection statistics"""
        
        if not self.collection:
            return {}
        
        try:
            count = self.collection.count()
            
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "database_type": "chromadb"
            }
        except Exception as e:
            logger.error(f"Failed to get ChromaDB stats: {e}")
            return {}


class FAISSVectorDatabase(BaseVectorDatabase):
    """FAISS implementation of vector database"""
    
    def __init__(self, config: VectorConfig):
        super().__init__(config)
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not available")
        
        self.index: Optional[faiss.Index] = None
        self.document_store: Dict[int, Tuple[str, DocumentMetadata]] = {}
        self.id_mapping: Dict[str, int] = {}  # document_id -> faiss_id
        self.next_id = 0
        
    async def initialize(self) -> bool:
        """Initialize FAISS index"""
        
        try:
            # Create index based on similarity metric
            if self.config.similarity_metric == "cosine":
                # Normalize vectors for cosine similarity
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.config.similarity_metric == "euclidean":
                self.index = faiss.IndexFlatL2(self.dimension)
            else:  # dot_product
                self.index = faiss.IndexFlatIP(self.dimension)
            
            # Try to load existing index
            persist_path = Path(self.config.persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)
            
            index_file = persist_path / f"{self.collection_name}.faiss"
            metadata_file = persist_path / f"{self.collection_name}_metadata.json"
            
            if index_file.exists() and metadata_file.exists():
                try:
                    self.index = faiss.read_index(str(index_file))
                    
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        
                    # Restore document store and ID mapping
                    for faiss_id_str, doc_data in metadata.get("document_store", {}).items():
                        faiss_id = int(faiss_id_str)
                        content = doc_data["content"]
                        
                        # Restore DocumentMetadata
                        meta_data = doc_data["metadata"]
                        document_metadata = DocumentMetadata(
                            document_id=meta_data["document_id"],
                            source=meta_data["source"],
                            title=meta_data["title"],
                            content_type=meta_data["content_type"],
                            timestamp=datetime.fromisoformat(meta_data["timestamp"]),
                            keywords=meta_data["keywords"],
                            consciousness_score=meta_data["consciousness_score"],
                            relevance_score=meta_data["relevance_score"],
                            chunk_index=meta_data["chunk_index"],
                            total_chunks=meta_data["total_chunks"],
                            custom_metadata=meta_data.get("custom_metadata", {})
                        )
                        
                        self.document_store[faiss_id] = (content, document_metadata)
                        self.id_mapping[document_metadata.document_id] = faiss_id
                    
                    self.next_id = metadata.get("next_id", 0)
                    logger.info(f"Loaded existing FAISS index with {len(self.document_store)} documents")
                    
                except Exception as e:
                    logger.warning(f"Failed to load existing FAISS index: {e}")
                    # Continue with fresh index
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            return False
    
    async def add_documents(
        self,
        documents: List[str],
        embeddings: List[np.ndarray],
        metadatas: List[DocumentMetadata]
    ) -> List[str]:
        """Add documents to FAISS index"""
        
        if not self.index:
            raise ValueError("Index not initialized")
        
        if len(documents) != len(embeddings) or len(documents) != len(metadatas):
            raise ValueError("Documents, embeddings, and metadatas must have the same length")
        
        try:
            # Prepare embeddings array
            embeddings_array = np.vstack(embeddings).astype('float32')
            
            # Normalize for cosine similarity
            if self.config.similarity_metric == "cosine":
                faiss.normalize_L2(embeddings_array)
            
            # Add to index
            faiss_ids = list(range(self.next_id, self.next_id + len(documents)))
            self.index.add_with_ids(embeddings_array, np.array(faiss_ids, dtype=np.int64))
            
            # Store documents and metadata
            document_ids = []
            for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
                faiss_id = faiss_ids[i]
                self.document_store[faiss_id] = (doc, metadata)
                self.id_mapping[metadata.document_id] = faiss_id
                document_ids.append(metadata.document_id)
            
            self.next_id += len(documents)
            
            # Persist if configured
            if self.config.enable_indexing:
                await self._persist_index()
            
            logger.debug(f"Added {len(documents)} documents to FAISS index")
            return document_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents to FAISS: {e}")
            raise
    
    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search FAISS index for similar vectors"""
        
        if not self.index:
            raise ValueError("Index not initialized")
        
        try:
            # Prepare query embedding
            query_array = query_embedding.astype('float32').reshape(1, -1)
            
            # Normalize for cosine similarity
            if self.config.similarity_metric == "cosine":
                faiss.normalize_L2(query_array)
            
            # Search index
            search_k = min(top_k * 2, len(self.document_store))  # Get more candidates for filtering
            scores, faiss_ids = self.index.search(query_array, search_k)
            
            # Convert to search results with filtering
            search_results = []
            for i, (score, faiss_id) in enumerate(zip(scores[0], faiss_ids[0])):
                if faiss_id == -1:  # No more results
                    break
                
                if faiss_id not in self.document_store:
                    continue
                
                content, metadata = self.document_store[faiss_id]
                
                # Apply filters
                if filters:
                    if not self._matches_filters(metadata, filters):
                        continue
                
                # Convert score to similarity
                if self.config.similarity_metric == "euclidean":
                    similarity_score = 1.0 / (1.0 + score)
                else:  # cosine or dot product
                    similarity_score = float(score)
                
                search_result = VectorSearchResult(
                    document_id=metadata.document_id,
                    content=content,
                    similarity_score=similarity_score,
                    metadata=metadata
                )
                
                search_results.append(search_result)
                
                if len(search_results) >= top_k:
                    break
            
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search FAISS index: {e}")
            raise
    
    def _matches_filters(self, metadata: DocumentMetadata, filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the given filters"""
        
        for key, value in filters.items():
            if key == "source" and metadata.source != value:
                return False
            elif key == "content_type" and metadata.content_type != value:
                return False
            elif key == "consciousness_score_min" and metadata.consciousness_score < value:
                return False
            elif key == "relevance_score_min" and metadata.relevance_score < value:
                return False
        
        return True
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from FAISS index"""
        
        # FAISS doesn't support deletion directly, so we mark for removal
        # In a production system, you'd rebuild the index periodically
        
        deleted_count = 0
        for doc_id in document_ids:
            if doc_id in self.id_mapping:
                faiss_id = self.id_mapping[doc_id]
                if faiss_id in self.document_store:
                    del self.document_store[faiss_id]
                del self.id_mapping[doc_id]
                deleted_count += 1
        
        logger.debug(f"Marked {deleted_count} documents for deletion in FAISS")
        return deleted_count > 0
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get FAISS index statistics"""
        
        return {
            "total_documents": len(self.document_store),
            "index_size": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "database_type": "faiss"
        }
    
    async def _persist_index(self):
        """Persist FAISS index and metadata to disk"""
        
        try:
            persist_path = Path(self.config.persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)
            
            # Save index
            index_file = persist_path / f"{self.collection_name}.faiss"
            faiss.write_index(self.index, str(index_file))
            
            # Prepare metadata for serialization
            serializable_store = {}
            for faiss_id, (content, metadata) in self.document_store.items():
                serializable_store[str(faiss_id)] = {
                    "content": content,
                    "metadata": {
                        "document_id": metadata.document_id,
                        "source": metadata.source,
                        "title": metadata.title,
                        "content_type": metadata.content_type,
                        "timestamp": metadata.timestamp.isoformat(),
                        "keywords": metadata.keywords,
                        "consciousness_score": metadata.consciousness_score,
                        "relevance_score": metadata.relevance_score,
                        "chunk_index": metadata.chunk_index,
                        "total_chunks": metadata.total_chunks,
                        "custom_metadata": metadata.custom_metadata
                    }
                }
            
            metadata_to_save = {
                "document_store": serializable_store,
                "next_id": self.next_id,
                "collection_name": self.collection_name,
                "dimension": self.dimension
            }
            
            # Save metadata
            metadata_file = persist_path / f"{self.collection_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata_to_save, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to persist FAISS index: {e}")


class VectorDatabaseManager:
    """Main vector database manager supporting multiple backends"""
    
    def __init__(self, config: VectorConfig):
        self.config = config
        self.database: Optional[BaseVectorDatabase] = None
        self.embedding_function: Optional[Callable] = None
        
        # Performance metrics
        self.metrics = {
            'documents_added': 0,
            'searches_performed': 0,
            'total_search_time': 0.0,
            'average_search_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Simple cache for recent searches
        self.search_cache: Dict[str, Tuple[List[VectorSearchResult], datetime]] = {}
    
    async def initialize(self, embedding_function: Callable[[List[str]], List[np.ndarray]]) -> bool:
        """Initialize vector database with embedding function"""
        
        self.embedding_function = embedding_function
        
        # Create appropriate database implementation
        if self.config.database_type == "chroma" and CHROMADB_AVAILABLE:
            self.database = ChromaVectorDatabase(self.config)
        elif self.config.database_type == "faiss" and FAISS_AVAILABLE:
            self.database = FAISSVectorDatabase(self.config)
        elif self.config.database_type == "hybrid":
            # For hybrid, prefer ChromaDB if available, fallback to FAISS
            if CHROMADB_AVAILABLE:
                self.database = ChromaVectorDatabase(self.config)
                logger.info("Using ChromaDB for hybrid configuration")
            elif FAISS_AVAILABLE:
                self.database = FAISSVectorDatabase(self.config)
                logger.info("Using FAISS for hybrid configuration")
            else:
                raise ValueError("Neither ChromaDB nor FAISS is available")
        else:
            available_dbs = []
            if CHROMADB_AVAILABLE:
                available_dbs.append("chroma")
            if FAISS_AVAILABLE:
                available_dbs.append("faiss")
            
            raise ValueError(f"Requested database type '{self.config.database_type}' not available. "
                           f"Available: {available_dbs}")
        
        # Initialize the database
        success = await self.database.initialize()
        if success:
            logger.info(f"Vector database initialized: {self.config.database_type}")
        
        return success
    
    async def add_documents(
        self,
        documents: List[str],
        metadatas: List[DocumentMetadata],
        batch_size: Optional[int] = None
    ) -> List[str]:
        """Add documents to the vector database"""
        
        if not self.database or not self.embedding_function:
            raise ValueError("Database not initialized")
        
        batch_size = batch_size or self.config.batch_size
        all_document_ids = []
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_metas = metadatas[i:i + batch_size]
            
            # Generate embeddings
            embeddings = self.embedding_function(batch_docs)
            
            # Apply consciousness boosting
            consciousness_boosted_embeddings = []
            for j, (embedding, metadata) in enumerate(zip(embeddings, batch_metas)):
                boosted_embedding = self._apply_consciousness_boost(
                    embedding, batch_docs[j], metadata.consciousness_score
                )
                consciousness_boosted_embeddings.append(boosted_embedding)
            
            # Add to database
            batch_ids = await self.database.add_documents(
                batch_docs, consciousness_boosted_embeddings, batch_metas
            )
            all_document_ids.extend(batch_ids)
            
            # Update metrics
            self.metrics['documents_added'] += len(batch_docs)
        
        logger.info(f"Added {len(documents)} documents to vector database")
        return all_document_ids
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> List[VectorSearchResult]:
        """Search for similar documents"""
        
        if not self.database or not self.embedding_function:
            raise ValueError("Database not initialized")
        
        start_time = datetime.now()
        
        # Check cache
        cache_key = self._get_cache_key(query, top_k, filters)
        if use_cache and cache_key in self.search_cache:
            cached_results, cache_time = self.search_cache[cache_key]
            if datetime.now() - cache_time < timedelta(minutes=10):  # 10-minute cache
                self.metrics['cache_hits'] += 1
                return cached_results
        
        self.metrics['cache_misses'] += 1
        
        # Generate query embedding
        query_embeddings = self.embedding_function([query])
        query_embedding = query_embeddings[0]
        
        # Apply consciousness boost to query if it contains consciousness keywords
        consciousness_score = self._calculate_consciousness_score(query)
        if consciousness_score > 0.1:
            query_embedding = self._apply_consciousness_boost(
                query_embedding, query, consciousness_score
            )
        
        # Search database
        results = await self.database.search(query_embedding, top_k, filters)
        
        # Apply similarity threshold filtering
        filtered_results = [
            result for result in results
            if result.similarity_score >= self.config.similarity_threshold
        ]
        
        # Cache results
        if use_cache:
            self.search_cache[cache_key] = (filtered_results, datetime.now())
            
            # Cleanup cache if too large
            if len(self.search_cache) > self.config.max_cache_size:
                await self._cleanup_cache()
        
        # Update metrics
        search_time = (datetime.now() - start_time).total_seconds()
        self.metrics['searches_performed'] += 1
        self.metrics['total_search_time'] += search_time
        self.metrics['average_search_time'] = (
            self.metrics['total_search_time'] / self.metrics['searches_performed']
        )
        
        logger.debug(f"Vector search completed in {search_time:.3f}s, found {len(filtered_results)} results")
        return filtered_results
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the database"""
        
        if not self.database:
            raise ValueError("Database not initialized")
        
        success = await self.database.delete_documents(document_ids)
        if success:
            # Clear cache since documents have been deleted
            self.search_cache.clear()
        
        return success
    
    def _apply_consciousness_boost(
        self,
        embedding: np.ndarray,
        text: str,
        consciousness_score: float
    ) -> np.ndarray:
        """Apply consciousness boosting to embeddings"""
        
        if not self.config.consciousness_weight or consciousness_score <= 0:
            return embedding
        
        # Calculate boost factor based on consciousness score
        boost_factor = 1.0 + (consciousness_score * (self.config.consciousness_weight - 1.0))
        
        # Apply boost by scaling the embedding
        boosted_embedding = embedding * boost_factor
        
        # Normalize to maintain vector properties
        norm = np.linalg.norm(boosted_embedding)
        if norm > 0:
            boosted_embedding = boosted_embedding / norm
        
        return boosted_embedding
    
    def _calculate_consciousness_score(self, text: str) -> float:
        """Calculate consciousness relevance score for text"""
        
        text_lower = text.lower()
        consciousness_matches = 0
        
        for keyword in self.config.consciousness_keywords:
            if keyword.lower() in text_lower:
                consciousness_matches += 1
        
        # Normalize score
        if len(self.config.consciousness_keywords) > 0:
            score = consciousness_matches / len(self.config.consciousness_keywords)
        else:
            score = 0.0
        
        return min(1.0, score)
    
    def _get_cache_key(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]]
    ) -> str:
        """Generate cache key for search"""
        
        cache_data = {
            "query": query,
            "top_k": top_k,
            "filters": filters or {}
        }
        
        return hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()
    
    async def _cleanup_cache(self):
        """Clean up old cache entries"""
        
        cutoff_time = datetime.now() - timedelta(hours=1)
        expired_keys = [
            key for key, (_, cache_time) in self.search_cache.items()
            if cache_time < cutoff_time
        ]
        
        for key in expired_keys:
            del self.search_cache[key]
        
        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        
        base_stats = {
            'performance_metrics': self.metrics.copy(),
            'cache_size': len(self.search_cache),
            'database_type': self.config.database_type,
            'vector_dimension': self.config.vector_dimension
        }
        
        if self.database:
            db_stats = await self.database.get_collection_stats()
            base_stats.update(db_stats)
        
        return base_stats
    
    async def optimize_performance(self):
        """Optimize database performance"""
        
        # Clean up cache
        await self._cleanup_cache()
        
        # If using FAISS and have many deletions, consider rebuilding index
        if isinstance(self.database, FAISSVectorDatabase):
            # In a production system, implement index rebuilding here
            logger.info("FAISS optimization: Consider rebuilding index if many deletions occurred")
        
        logger.info("Vector database performance optimization completed")


# Testing and example usage
async def test_vector_database():
    """Test vector database functionality"""
    
    # Mock embedding function
    def mock_embedding_function(texts: List[str]) -> List[np.ndarray]:
        """Simple mock embedding function"""
        embeddings = []
        for text in texts:
            # Generate random embedding based on text hash for consistency
            hash_value = hashlib.md5(text.encode()).hexdigest()
            np.random.seed(int(hash_value[:8], 16))
            embedding = np.random.random(768).astype('float32')
            embeddings.append(embedding)
        return embeddings
    
    # Create configuration
    config = VectorConfig(
        database_type="faiss",  # Use FAISS for testing (no external dependencies)
        vector_dimension=768,
        collection_name="test_collection"
    )
    
    # Create manager
    manager = VectorDatabaseManager(config)
    
    # Initialize
    success = await manager.initialize(mock_embedding_function)
    if not success:
        print("Failed to initialize vector database")
        return
    
    # Test documents
    documents = [
        "Consciousness in artificial intelligence is a fascinating research area.",
        "Machine learning models are becoming increasingly sophisticated.",
        "The study of self-aware AI systems requires interdisciplinary approaches.",
        "Neural networks can exhibit emergent behaviors.",
        "Artificial consciousness may be the next breakthrough in AI."
    ]
    
    # Create metadata
    metadatas = []
    for i, doc in enumerate(documents):
        consciousness_score = 0.8 if 'consciousness' in doc.lower() or 'self-aware' in doc.lower() else 0.2
        metadata = DocumentMetadata(
            document_id=f"doc_{i}",
            source="test",
            title=f"Test Document {i}",
            content_type="article",
            timestamp=datetime.now(),
            keywords=doc.split()[:3],
            consciousness_score=consciousness_score,
            relevance_score=0.9,
            chunk_index=0,
            total_chunks=1
        )
        metadatas.append(metadata)
    
    # Add documents
    print("Adding documents...")
    document_ids = await manager.add_documents(documents, metadatas)
    print(f"Added documents: {document_ids}")
    
    # Test search
    print("\nTesting search...")
    search_queries = [
        "consciousness research",
        "machine learning",
        "artificial intelligence"
    ]
    
    for query in search_queries:
        results = await manager.search(query, top_k=3)
        print(f"\nQuery: '{query}'")
        for result in results:
            print(f"  - Score: {result.similarity_score:.3f} | {result.content[:60]}...")
    
    # Get statistics
    stats = await manager.get_statistics()
    print("\nDatabase Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(test_vector_database())