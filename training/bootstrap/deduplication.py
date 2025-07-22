"""Data deduplication system for bootstrap training.

This module implements efficient deduplication strategies to ensure
training data quality and prevent overfitting on duplicates.
"""

import hashlib
import json
import pickle
from typing import List, Set, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch
from collections import defaultdict
import mmh3  # MurmurHash3 for fast hashing
import lmdb
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from ...core.blt.patching import DynamicPatcher
from ...core.blt.entropy import calculate_byte_entropy


@dataclass
class DeduplicationStats:
    """Statistics for deduplication process."""
    total_documents: int = 0
    unique_documents: int = 0
    duplicate_documents: int = 0
    total_bytes: int = 0
    unique_bytes: int = 0
    duplicate_bytes: int = 0
    fuzzy_duplicates: int = 0
    near_duplicates: int = 0
    processing_time: float = 0.0
    
    @property
    def deduplication_rate(self) -> float:
        return self.duplicate_documents / max(self.total_documents, 1)
    
    @property
    def byte_reduction_rate(self) -> float:
        return self.duplicate_bytes / max(self.total_bytes, 1)


class MinHashDeduplicator:
    """MinHash-based approximate deduplication."""
    
    def __init__(
        self,
        num_permutations: int = 128,
        threshold: float = 0.8,
        shingle_size: int = 5
    ):
        self.num_permutations = num_permutations
        self.threshold = threshold
        self.shingle_size = shingle_size
        
        # Generate hash functions
        self.hash_funcs = self._generate_hash_functions()
        
        # Storage for signatures
        self.signatures: Dict[str, np.ndarray] = {}
        self.document_map: Dict[str, bytes] = {}
    
    def _generate_hash_functions(self) -> List[Tuple[int, int]]:
        """Generate hash function parameters."""
        # Use different seeds for MurmurHash3
        return [(i, i + 1000) for i in range(self.num_permutations)]
    
    def _create_shingles(self, data: bytes) -> Set[bytes]:
        """Create shingles from byte sequence."""
        shingles = set()
        for i in range(len(data) - self.shingle_size + 1):
            shingle = data[i:i + self.shingle_size]
            shingles.add(shingle)
        return shingles
    
    def _compute_minhash(self, shingles: Set[bytes]) -> np.ndarray:
        """Compute MinHash signature."""
        signature = np.full(self.num_permutations, np.inf)
        
        for shingle in shingles:
            for i, (seed1, seed2) in enumerate(self.hash_funcs):
                # Use MurmurHash3 with different seeds
                hash_val = mmh3.hash(shingle, seed1, signed=False)
                signature[i] = min(signature[i], hash_val)
        
        return signature.astype(np.uint32)
    
    def add_document(self, doc_id: str, data: bytes) -> bool:
        """Add document and check for duplicates."""
        # Create shingles
        shingles = self._create_shingles(data)
        if not shingles:
            return True  # Too short to deduplicate
        
        # Compute signature
        signature = self._compute_minhash(shingles)
        
        # Check for near duplicates
        for existing_id, existing_sig in self.signatures.items():
            similarity = self._estimate_jaccard(signature, existing_sig)
            if similarity >= self.threshold:
                return False  # Near duplicate found
        
        # Add to collection
        self.signatures[doc_id] = signature
        self.document_map[doc_id] = data
        return True
    
    def _estimate_jaccard(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Estimate Jaccard similarity from signatures."""
        return np.mean(sig1 == sig2)
    
    def find_near_duplicates(self, data: bytes) -> List[Tuple[str, float]]:
        """Find near duplicates of given data."""
        shingles = self._create_shingles(data)
        if not shingles:
            return []
        
        signature = self._compute_minhash(shingles)
        duplicates = []
        
        for doc_id, existing_sig in self.signatures.items():
            similarity = self._estimate_jaccard(signature, existing_sig)
            if similarity >= self.threshold:
                duplicates.append((doc_id, similarity))
        
        return sorted(duplicates, key=lambda x: x[1], reverse=True)


class ExactDeduplicator:
    """Exact deduplication using content hashing."""
    
    def __init__(self, hash_func: str = "sha256"):
        self.hash_func = hash_func
        self.hashes: Set[str] = set()
        self.hash_to_id: Dict[str, str] = {}
    
    def compute_hash(self, data: bytes) -> str:
        """Compute hash of data."""
        if self.hash_func == "sha256":
            return hashlib.sha256(data).hexdigest()
        elif self.hash_func == "md5":
            return hashlib.md5(data).hexdigest()
        elif self.hash_func == "xxhash":
            return str(mmh3.hash128(data))
        else:
            raise ValueError(f"Unknown hash function: {self.hash_func}")
    
    def is_duplicate(self, data: bytes) -> Tuple[bool, Optional[str]]:
        """Check if data is duplicate."""
        hash_val = self.compute_hash(data)
        
        if hash_val in self.hashes:
            return True, self.hash_to_id.get(hash_val)
        
        return False, None
    
    def add(self, doc_id: str, data: bytes) -> bool:
        """Add data and return True if unique."""
        hash_val = self.compute_hash(data)
        
        if hash_val in self.hashes:
            return False
        
        self.hashes.add(hash_val)
        self.hash_to_id[hash_val] = doc_id
        return True


class SemanticDeduplicator:
    """Semantic deduplication using embeddings."""
    
    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        threshold: float = 0.95,
        index_type: str = "faiss"
    ):
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.index_type = index_type
        
        # Storage
        self.embeddings: List[np.ndarray] = []
        self.doc_ids: List[str] = []
        
        # Index for fast similarity search
        self.index = None
        self._init_index()
    
    def _init_index(self):
        """Initialize similarity index."""
        if self.index_type == "faiss":
            try:
                import faiss
                # Will be initialized when we know embedding dimension
                self.faiss = faiss
            except ImportError:
                print("FAISS not available, using brute force search")
                self.index_type = "brute"
    
    def _compute_embedding(self, data: bytes) -> np.ndarray:
        """Compute embedding for data."""
        if self.embedding_model is None:
            # Simple bag-of-bytes embedding
            embedding = np.zeros(256)
            for byte in data[:1000]:  # Use first 1000 bytes
                embedding[byte] += 1
            return embedding / (len(data[:1000]) + 1)
        else:
            # Use provided model
            return self.embedding_model.encode(data)
    
    def add_document(self, doc_id: str, data: bytes) -> bool:
        """Add document and check for semantic duplicates."""
        embedding = self._compute_embedding(data)
        
        # Check for duplicates
        if self.embeddings:
            similarities = self._find_similar(embedding)
            if similarities and similarities[0][1] >= self.threshold:
                return False  # Semantic duplicate found
        
        # Add to collection
        self.embeddings.append(embedding)
        self.doc_ids.append(doc_id)
        
        # Update index
        if self.index_type == "faiss" and hasattr(self, 'faiss'):
            if self.index is None:
                # Initialize index with first embedding
                dim = len(embedding)
                self.index = self.faiss.IndexFlatIP(dim)  # Inner product
            self.index.add(embedding.reshape(1, -1).astype(np.float32))
        
        return True
    
    def _find_similar(self, embedding: np.ndarray, k: int = 1) -> List[Tuple[int, float]]:
        """Find similar embeddings."""
        if not self.embeddings:
            return []
        
        if self.index_type == "faiss" and self.index is not None:
            # Use FAISS for fast search
            D, I = self.index.search(
                embedding.reshape(1, -1).astype(np.float32), k
            )
            return [(I[0][i], D[0][i]) for i in range(len(I[0]))]
        else:
            # Brute force search
            similarities = []
            for i, stored_emb in enumerate(self.embeddings):
                sim = np.dot(embedding, stored_emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(stored_emb)
                )
                similarities.append((i, sim))
            
            return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]


class PatchBasedDeduplicator:
    """Deduplication based on BLT patches."""
    
    def __init__(
        self,
        patcher: Optional[DynamicPatcher] = None,
        threshold: float = 0.7,
        min_patch_overlap: int = 3
    ):
        self.patcher = patcher or DynamicPatcher()
        self.threshold = threshold
        self.min_patch_overlap = min_patch_overlap
        
        # Patch storage
        self.patch_index: Dict[bytes, Set[str]] = defaultdict(set)
        self.doc_patches: Dict[str, List[bytes]] = {}
    
    def add_document(self, doc_id: str, data: bytes) -> bool:
        """Add document and check for patch-based duplicates."""
        # Create patches
        patches = self.patcher.create_patches(data)
        
        # Check for duplicates based on patch overlap
        overlap_counts = defaultdict(int)
        for patch in patches:
            if patch in self.patch_index:
                for existing_id in self.patch_index[patch]:
                    overlap_counts[existing_id] += 1
        
        # Check if any document has high overlap
        for existing_id, overlap in overlap_counts.items():
            if existing_id in self.doc_patches:
                total_patches = len(self.doc_patches[existing_id])
                overlap_ratio = overlap / total_patches
                if overlap_ratio >= self.threshold and overlap >= self.min_patch_overlap:
                    return False  # Duplicate found
        
        # Add to index
        self.doc_patches[doc_id] = patches
        for patch in patches:
            self.patch_index[patch].add(doc_id)
        
        return True


class HierarchicalDeduplicator:
    """Hierarchical deduplication with multiple strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Initialize deduplicators
        self.exact_dedup = ExactDeduplicator(hash_func="xxhash")
        self.minhash_dedup = MinHashDeduplicator(
            num_permutations=self.config["minhash_perms"],
            threshold=self.config["minhash_threshold"]
        )
        self.patch_dedup = PatchBasedDeduplicator(
            threshold=self.config["patch_threshold"]
        )
        
        # Statistics
        self.stats = DeduplicationStats()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            "minhash_perms": 128,
            "minhash_threshold": 0.8,
            "patch_threshold": 0.7,
            "min_doc_length": 100,
            "enable_semantic": False
        }
    
    def deduplicate_document(
        self,
        doc_id: str,
        data: bytes
    ) -> Tuple[bool, str]:
        """Deduplicate document through hierarchy."""
        self.stats.total_documents += 1
        self.stats.total_bytes += len(data)
        
        # Skip very short documents
        if len(data) < self.config["min_doc_length"]:
            self.stats.unique_documents += 1
            self.stats.unique_bytes += len(data)
            return True, "too_short"
        
        # Level 1: Exact deduplication
        is_unique = self.exact_dedup.add(doc_id, data)
        if not is_unique:
            self.stats.duplicate_documents += 1
            self.stats.duplicate_bytes += len(data)
            return False, "exact_duplicate"
        
        # Level 2: MinHash near-duplicate detection
        is_unique = self.minhash_dedup.add_document(doc_id, data)
        if not is_unique:
            self.stats.duplicate_documents += 1
            self.stats.duplicate_bytes += len(data)
            self.stats.near_duplicates += 1
            return False, "near_duplicate"
        
        # Level 3: Patch-based deduplication
        is_unique = self.patch_dedup.add_document(doc_id, data)
        if not is_unique:
            self.stats.duplicate_documents += 1
            self.stats.duplicate_bytes += len(data)
            self.stats.fuzzy_duplicates += 1
            return False, "fuzzy_duplicate"
        
        # Document is unique
        self.stats.unique_documents += 1
        self.stats.unique_bytes += len(data)
        return True, "unique"
    
    def get_stats(self) -> DeduplicationStats:
        """Get deduplication statistics."""
        return self.stats


class PersistentDeduplicator:
    """Persistent deduplication with LMDB backend."""
    
    def __init__(self, db_path: str, map_size: int = 10 * 1024**3):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Open LMDB databases
        self.env = lmdb.open(
            str(self.db_path),
            map_size=map_size,
            max_dbs=3
        )
        
        # Create sub-databases
        self.hash_db = self.env.open_db(b'hashes')
        self.minhash_db = self.env.open_db(b'minhashes')
        self.stats_db = self.env.open_db(b'stats')
        
        # In-memory components
        self.minhash_dedup = MinHashDeduplicator()
        self.stats = self._load_stats()
    
    def _load_stats(self) -> DeduplicationStats:
        """Load statistics from database."""
        with self.env.begin() as txn:
            data = txn.get(b'stats', db=self.stats_db)
            if data:
                return pickle.loads(data)
        return DeduplicationStats()
    
    def _save_stats(self):
        """Save statistics to database."""
        with self.env.begin(write=True) as txn:
            txn.put(b'stats', pickle.dumps(self.stats), db=self.stats_db)
    
    def is_duplicate(self, data: bytes) -> Tuple[bool, Optional[str]]:
        """Check if data is duplicate."""
        # Check exact hash
        hash_val = hashlib.sha256(data).digest()
        
        with self.env.begin() as txn:
            existing = txn.get(hash_val, db=self.hash_db)
            if existing:
                return True, existing.decode('utf-8')
        
        # Check MinHash
        duplicates = self.minhash_dedup.find_near_duplicates(data)
        if duplicates:
            return True, duplicates[0][0]
        
        return False, None
    
    def add_document(self, doc_id: str, data: bytes) -> bool:
        """Add document to deduplication database."""
        # Check if duplicate
        is_dup, _ = self.is_duplicate(data)
        if is_dup:
            self.stats.duplicate_documents += 1
            self.stats.duplicate_bytes += len(data)
            return False
        
        # Add to databases
        hash_val = hashlib.sha256(data).digest()
        
        with self.env.begin(write=True) as txn:
            # Add exact hash
            txn.put(hash_val, doc_id.encode('utf-8'), db=self.hash_db)
            
            # Add MinHash signature
            if self.minhash_dedup.add_document(doc_id, data):
                sig = self.minhash_dedup.signatures[doc_id]
                txn.put(
                    doc_id.encode('utf-8'),
                    sig.tobytes(),
                    db=self.minhash_db
                )
        
        self.stats.unique_documents += 1
        self.stats.unique_bytes += len(data)
        self.stats.total_documents += 1
        self.stats.total_bytes += len(data)
        
        # Periodically save stats
        if self.stats.total_documents % 1000 == 0:
            self._save_stats()
        
        return True
    
    def close(self):
        """Close database."""
        self._save_stats()
        self.env.close()


def deduplicate_dataset(
    input_files: List[str],
    output_file: str,
    dedup_strategy: str = "hierarchical",
    config: Optional[Dict[str, Any]] = None,
    num_workers: int = None
) -> DeduplicationStats:
    """Deduplicate a dataset."""
    
    # Create deduplicator
    if dedup_strategy == "hierarchical":
        deduplicator = HierarchicalDeduplicator(config)
    elif dedup_strategy == "persistent":
        db_path = config.get("db_path", "dedup_db/")
        deduplicator = PersistentDeduplicator(db_path)
    else:
        raise ValueError(f"Unknown strategy: {dedup_strategy}")
    
    # Process files
    unique_docs = []
    
    for input_file in tqdm(input_files, desc="Processing files"):
        with open(input_file, 'rb') as f:
            # Assume one document per line
            for line_num, line in enumerate(f):
                doc_id = f"{input_file}:{line_num}"
                
                is_unique, _ = deduplicator.deduplicate_document(doc_id, line.strip())
                if is_unique:
                    unique_docs.append(line)
    
    # Write unique documents
    with open(output_file, 'wb') as f:
        for doc in unique_docs:
            f.write(doc)
    
    # Get final statistics
    stats = deduplicator.get_stats()
    
    # Cleanup
    if hasattr(deduplicator, 'close'):
        deduplicator.close()
    
    return stats