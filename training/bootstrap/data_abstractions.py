"""Data loading abstractions for unified data access.

This module provides abstract interfaces and implementations for
loading data from various sources in a consistent manner.
"""

import json
import gzip
import pickle
from typing import (
    Iterator, List, Dict, Any, Optional, Union, Callable,
    Protocol, TypeVar, Generic
)
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod
import pyarrow.parquet as pq
import h5py
import lmdb
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import mmap
import struct

T = TypeVar('T')


@dataclass
class DataSample:
    """Generic data sample container."""
    id: str
    content: bytes
    metadata: Dict[str, Any]
    source: str
    
    @property
    def text(self) -> str:
        """Get content as text."""
        return self.content.decode('utf-8', errors='replace')
    
    @property
    def size(self) -> int:
        """Get content size in bytes."""
        return len(self.content)


class DataLoader(ABC, Generic[T]):
    """Abstract base class for data loaders."""
    
    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        """Iterate over data samples."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Get total number of samples."""
        pass
    
    def take(self, n: int) -> List[T]:
        """Take first n samples."""
        samples = []
        for i, sample in enumerate(self):
            if i >= n:
                break
            samples.append(sample)
        return samples
    
    def filter(self, predicate: Callable[[T], bool]) -> 'FilteredLoader[T]':
        """Filter samples by predicate."""
        return FilteredLoader(self, predicate)
    
    def map(self, transform: Callable[[T], T]) -> 'MappedLoader[T]':
        """Transform samples."""
        return MappedLoader(self, transform)
    
    def batch(self, batch_size: int) -> 'BatchedLoader[T]':
        """Create batches of samples."""
        return BatchedLoader(self, batch_size)
    
    def shuffle(self, buffer_size: int = 10000) -> 'ShuffledLoader[T]':
        """Shuffle samples with buffer."""
        return ShuffledLoader(self, buffer_size)
    
    def cache(self, cache_dir: Optional[str] = None) -> 'CachedLoader[T]':
        """Cache samples for faster access."""
        return CachedLoader(self, cache_dir)


class TextFileLoader(DataLoader[DataSample]):
    """Load data from text files."""
    
    def __init__(
        self,
        file_paths: Union[str, List[str]],
        encoding: str = 'utf-8',
        delimiter: Optional[str] = '\n',
        max_samples: Optional[int] = None
    ):
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        self.file_paths = [Path(p) for p in file_paths]
        self.encoding = encoding
        self.delimiter = delimiter
        self.max_samples = max_samples
        self._length: Optional[int] = None
    
    def __iter__(self) -> Iterator[DataSample]:
        """Iterate over text samples."""
        sample_count = 0
        
        for file_path in self.file_paths:
            if file_path.suffix == '.gz':
                open_func = gzip.open
                mode = 'rt'
            else:
                open_func = open
                mode = 'r'
            
            with open_func(file_path, mode, encoding=self.encoding) as f:
                if self.delimiter:
                    for line_num, line in enumerate(f):
                        if self.max_samples and sample_count >= self.max_samples:
                            return
                        
                        content = line.rstrip(self.delimiter)
                        if content:  # Skip empty lines
                            yield DataSample(
                                id=f"{file_path.name}:{line_num}",
                                content=content.encode(self.encoding),
                                metadata={"line_number": line_num},
                                source=str(file_path)
                            )
                            sample_count += 1
                else:
                    # Read entire file
                    content = f.read()
                    yield DataSample(
                        id=file_path.name,
                        content=content.encode(self.encoding),
                        metadata={"file_size": len(content)},
                        source=str(file_path)
                    )
                    sample_count += 1
    
    def __len__(self) -> int:
        """Get total number of samples."""
        if self._length is None:
            self._length = sum(1 for _ in self)
        return self._length


class JSONLLoader(DataLoader[DataSample]):
    """Load data from JSONL files."""
    
    def __init__(
        self,
        file_paths: Union[str, List[str]],
        content_field: str = "text",
        id_field: Optional[str] = "id",
        metadata_fields: Optional[List[str]] = None
    ):
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        self.file_paths = [Path(p) for p in file_paths]
        self.content_field = content_field
        self.id_field = id_field
        self.metadata_fields = metadata_fields or []
    
    def __iter__(self) -> Iterator[DataSample]:
        """Iterate over JSONL samples."""
        for file_path in self.file_paths:
            if file_path.suffix == '.gz':
                open_func = gzip.open
                mode = 'rt'
            else:
                open_func = open
                mode = 'r'
            
            with open_func(file_path, mode, encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        data = json.loads(line)
                        
                        # Extract content
                        content = data.get(self.content_field, "")
                        if isinstance(content, list):
                            content = " ".join(str(c) for c in content)
                        
                        # Extract ID
                        if self.id_field and self.id_field in data:
                            sample_id = str(data[self.id_field])
                        else:
                            sample_id = f"{file_path.name}:{line_num}"
                        
                        # Extract metadata
                        metadata = {}
                        for field in self.metadata_fields:
                            if field in data:
                                metadata[field] = data[field]
                        
                        yield DataSample(
                            id=sample_id,
                            content=str(content).encode('utf-8'),
                            metadata=metadata,
                            source=str(file_path)
                        )
                    
                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON lines
    
    def __len__(self) -> int:
        """Get total number of samples."""
        return sum(1 for _ in self)


class ParquetLoader(DataLoader[DataSample]):
    """Load data from Parquet files."""
    
    def __init__(
        self,
        file_paths: Union[str, List[str]],
        content_column: str = "text",
        id_column: Optional[str] = "id",
        batch_size: int = 1000
    ):
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        self.file_paths = [Path(p) for p in file_paths]
        self.content_column = content_column
        self.id_column = id_column
        self.batch_size = batch_size
    
    def __iter__(self) -> Iterator[DataSample]:
        """Iterate over Parquet samples."""
        for file_path in self.file_paths:
            parquet_file = pq.ParquetFile(file_path)
            
            for batch in parquet_file.iter_batches(batch_size=self.batch_size):
                df = batch.to_pandas()
                
                for idx, row in df.iterrows():
                    # Extract content
                    content = str(row[self.content_column])
                    
                    # Extract ID
                    if self.id_column and self.id_column in df.columns:
                        sample_id = str(row[self.id_column])
                    else:
                        sample_id = f"{file_path.name}:{idx}"
                    
                    # Extract metadata (all other columns)
                    metadata = {}
                    for col in df.columns:
                        if col not in [self.content_column, self.id_column]:
                            metadata[col] = row[col]
                    
                    yield DataSample(
                        id=sample_id,
                        content=content.encode('utf-8'),
                        metadata=metadata,
                        source=str(file_path)
                    )
    
    def __len__(self) -> int:
        """Get total number of samples."""
        total = 0
        for file_path in self.file_paths:
            parquet_file = pq.ParquetFile(file_path)
            total += parquet_file.metadata.num_rows
        return total


class HDF5Loader(DataLoader[DataSample]):
    """Load data from HDF5 files."""
    
    def __init__(
        self,
        file_path: str,
        dataset_name: str = "data",
        content_key: str = "text",
        id_key: Optional[str] = "id"
    ):
        self.file_path = Path(file_path)
        self.dataset_name = dataset_name
        self.content_key = content_key
        self.id_key = id_key
    
    def __iter__(self) -> Iterator[DataSample]:
        """Iterate over HDF5 samples."""
        with h5py.File(self.file_path, 'r') as f:
            dataset = f[self.dataset_name]
            
            for idx in range(len(dataset)):
                sample = dataset[idx]
                
                # Extract content
                if self.content_key:
                    content = sample[self.content_key]
                else:
                    content = sample
                
                if isinstance(content, np.ndarray):
                    content = content.tobytes()
                elif isinstance(content, str):
                    content = content.encode('utf-8')
                elif isinstance(content, bytes):
                    pass
                else:
                    content = str(content).encode('utf-8')
                
                # Extract ID
                if self.id_key and self.id_key in sample.dtype.names:
                    sample_id = str(sample[self.id_key])
                else:
                    sample_id = f"{self.dataset_name}:{idx}"
                
                yield DataSample(
                    id=sample_id,
                    content=content,
                    metadata={"index": idx},
                    source=str(self.file_path)
                )
    
    def __len__(self) -> int:
        """Get total number of samples."""
        with h5py.File(self.file_path, 'r') as f:
            return len(f[self.dataset_name])


class LMDBLoader(DataLoader[DataSample]):
    """Load data from LMDB database."""
    
    def __init__(
        self,
        db_path: str,
        map_size: Optional[int] = None,
        readonly: bool = True
    ):
        self.db_path = Path(db_path)
        self.map_size = map_size
        self.readonly = readonly
    
    def __iter__(self) -> Iterator[DataSample]:
        """Iterate over LMDB samples."""
        env = lmdb.open(
            str(self.db_path),
            map_size=self.map_size,
            readonly=self.readonly
        )
        
        try:
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    # Try to deserialize as pickle
                    try:
                        data = pickle.loads(value)
                        if isinstance(data, dict):
                            yield DataSample(
                                id=key.decode('utf-8'),
                                content=data.get('content', b''),
                                metadata=data.get('metadata', {}),
                                source=str(self.db_path)
                            )
                        else:
                            yield DataSample(
                                id=key.decode('utf-8'),
                                content=value,
                                metadata={},
                                source=str(self.db_path)
                            )
                    except:
                        # Treat as raw bytes
                        yield DataSample(
                            id=key.decode('utf-8'),
                            content=value,
                            metadata={},
                            source=str(self.db_path)
                        )
        finally:
            env.close()
    
    def __len__(self) -> int:
        """Get total number of samples."""
        env = lmdb.open(str(self.db_path), readonly=True)
        try:
            return env.stat()['entries']
        finally:
            env.close()


class MemoryMappedLoader(DataLoader[DataSample]):
    """Load data from memory-mapped files for fast access."""
    
    def __init__(
        self,
        index_file: str,
        data_file: str,
        encoding: str = 'utf-8'
    ):
        self.index_file = Path(index_file)
        self.data_file = Path(data_file)
        self.encoding = encoding
        
        # Load index
        self.index = self._load_index()
        
        # Memory map data file
        self.data_mmap = None
    
    def _load_index(self) -> List[Tuple[int, int]]:
        """Load offset index."""
        index = []
        with open(self.index_file, 'rb') as f:
            while True:
                data = f.read(16)  # 2 int64 values
                if not data:
                    break
                offset, length = struct.unpack('qq', data)
                index.append((offset, length))
        return index
    
    def __iter__(self) -> Iterator[DataSample]:
        """Iterate over memory-mapped samples."""
        with open(self.data_file, 'rb') as f:
            self.data_mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            try:
                for idx, (offset, length) in enumerate(self.index):
                    content = self.data_mmap[offset:offset + length]
                    
                    yield DataSample(
                        id=f"sample_{idx}",
                        content=content,
                        metadata={"offset": offset, "length": length},
                        source=str(self.data_file)
                    )
            finally:
                self.data_mmap.close()
    
    def __len__(self) -> int:
        """Get total number of samples."""
        return len(self.index)
    
    def __getitem__(self, idx: int) -> DataSample:
        """Random access to samples."""
        if idx >= len(self.index):
            raise IndexError(f"Index {idx} out of range")
        
        offset, length = self.index[idx]
        
        with open(self.data_file, 'rb') as f:
            f.seek(offset)
            content = f.read(length)
        
        return DataSample(
            id=f"sample_{idx}",
            content=content,
            metadata={"offset": offset, "length": length},
            source=str(self.data_file)
        )


class FilteredLoader(DataLoader[T]):
    """Filtered data loader."""
    
    def __init__(self, parent: DataLoader[T], predicate: Callable[[T], bool]):
        self.parent = parent
        self.predicate = predicate
    
    def __iter__(self) -> Iterator[T]:
        """Iterate over filtered samples."""
        for sample in self.parent:
            if self.predicate(sample):
                yield sample
    
    def __len__(self) -> int:
        """Get number of filtered samples."""
        return sum(1 for _ in self)


class MappedLoader(DataLoader[T]):
    """Mapped data loader."""
    
    def __init__(self, parent: DataLoader[T], transform: Callable[[T], T]):
        self.parent = parent
        self.transform = transform
    
    def __iter__(self) -> Iterator[T]:
        """Iterate over transformed samples."""
        for sample in self.parent:
            yield self.transform(sample)
    
    def __len__(self) -> int:
        """Get number of samples."""
        return len(self.parent)


class BatchedLoader(DataLoader[List[T]]):
    """Batched data loader."""
    
    def __init__(self, parent: DataLoader[T], batch_size: int):
        self.parent = parent
        self.batch_size = batch_size
    
    def __iter__(self) -> Iterator[List[T]]:
        """Iterate over batches."""
        batch = []
        for sample in self.parent:
            batch.append(sample)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        
        if batch:  # Yield remaining samples
            yield batch
    
    def __len__(self) -> int:
        """Get number of batches."""
        parent_len = len(self.parent)
        return (parent_len + self.batch_size - 1) // self.batch_size


class ShuffledLoader(DataLoader[T]):
    """Shuffled data loader with buffer."""
    
    def __init__(self, parent: DataLoader[T], buffer_size: int):
        self.parent = parent
        self.buffer_size = buffer_size
    
    def __iter__(self) -> Iterator[T]:
        """Iterate over shuffled samples."""
        buffer = []
        
        for sample in self.parent:
            buffer.append(sample)
            
            if len(buffer) >= self.buffer_size:
                # Shuffle and yield half the buffer
                np.random.shuffle(buffer)
                half = len(buffer) // 2
                
                for i in range(half):
                    yield buffer.pop()
        
        # Yield remaining samples
        np.random.shuffle(buffer)
        for sample in buffer:
            yield sample
    
    def __len__(self) -> int:
        """Get number of samples."""
        return len(self.parent)


class CachedLoader(DataLoader[T]):
    """Cached data loader for faster access."""
    
    def __init__(
        self,
        parent: DataLoader[T],
        cache_dir: Optional[str] = None
    ):
        self.parent = parent
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file
        parent_id = id(parent)
        self.cache_file = self.cache_dir / f"loader_cache_{parent_id}.pkl"
        self._cache: Optional[List[T]] = None
    
    def _build_cache(self):
        """Build cache from parent loader."""
        if self.cache_file.exists():
            # Load from file
            with open(self.cache_file, 'rb') as f:
                self._cache = pickle.load(f)
        else:
            # Build from parent
            self._cache = list(self.parent)
            
            # Save to file
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self._cache, f)
    
    def __iter__(self) -> Iterator[T]:
        """Iterate over cached samples."""
        if self._cache is None:
            self._build_cache()
        
        for sample in self._cache:
            yield sample
    
    def __len__(self) -> int:
        """Get number of samples."""
        if self._cache is None:
            self._build_cache()
        return len(self._cache)


class CompositeLoader(DataLoader[DataSample]):
    """Combine multiple data loaders."""
    
    def __init__(
        self,
        loaders: List[DataLoader[DataSample]],
        weights: Optional[List[float]] = None,
        strategy: str = "sequential"  # sequential, interleave, weighted
    ):
        self.loaders = loaders
        self.weights = weights or [1.0] * len(loaders)
        self.strategy = strategy
        
        if strategy == "weighted" and len(self.weights) != len(self.loaders):
            raise ValueError("Number of weights must match number of loaders")
    
    def __iter__(self) -> Iterator[DataSample]:
        """Iterate over composite samples."""
        if self.strategy == "sequential":
            # Concatenate loaders
            for loader in self.loaders:
                for sample in loader:
                    yield sample
        
        elif self.strategy == "interleave":
            # Round-robin through loaders
            iterators = [iter(loader) for loader in self.loaders]
            active = list(range(len(iterators)))
            
            while active:
                for i in list(active):
                    try:
                        yield next(iterators[i])
                    except StopIteration:
                        active.remove(i)
        
        elif self.strategy == "weighted":
            # Sample based on weights
            total_samples = sum(len(loader) for loader in self.loaders)
            probabilities = np.array(self.weights) / sum(self.weights)
            
            iterators = [iter(loader) for loader in self.loaders]
            exhausted = set()
            
            for _ in range(total_samples):
                if len(exhausted) >= len(self.loaders):
                    break
                
                # Sample loader based on weights
                valid_indices = [i for i in range(len(self.loaders)) if i not in exhausted]
                valid_probs = probabilities[valid_indices] / probabilities[valid_indices].sum()
                
                loader_idx = np.random.choice(valid_indices, p=valid_probs)
                
                try:
                    yield next(iterators[loader_idx])
                except StopIteration:
                    exhausted.add(loader_idx)
    
    def __len__(self) -> int:
        """Get total number of samples."""
        return sum(len(loader) for loader in self.loaders)


def create_data_loader(
    source_type: str,
    source_path: Union[str, List[str]],
    **kwargs
) -> DataLoader[DataSample]:
    """Factory function to create appropriate data loader."""
    
    if source_type == "text":
        return TextFileLoader(source_path, **kwargs)
    elif source_type == "jsonl":
        return JSONLLoader(source_path, **kwargs)
    elif source_type == "parquet":
        return ParquetLoader(source_path, **kwargs)
    elif source_type == "hdf5":
        return HDF5Loader(source_path, **kwargs)
    elif source_type == "lmdb":
        return LMDBLoader(source_path, **kwargs)
    elif source_type == "mmap":
        return MemoryMappedLoader(source_path, **kwargs)
    else:
        raise ValueError(f"Unknown source type: {source_type}")