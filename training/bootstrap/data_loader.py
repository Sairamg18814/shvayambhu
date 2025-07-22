"""Bootstrap Data Loader for Self-Training Pipeline.

This module implements efficient data loading and preparation for the
bootstrap training phase, including dynamic batching and preprocessing.
"""

import os
import json
import random
import numpy as np
from typing import Dict, List, Optional, Iterator, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging
from collections import defaultdict
import threading
import queue
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch import Tensor

from .data_filters import DataQualityFilter, QualityMetrics
from .objectives import create_default_objectives, MultiObjectiveTrainer
from ...core.blt.patching import BLTInputProcessor
from ...core.blt.entropy import calculate_byte_entropy

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch creation."""
    batch_size: int = 16
    max_sequence_length: int = 1024
    min_sequence_length: int = 64
    dynamic_batching: bool = True
    pad_to_multiple: int = 8
    drop_last: bool = False
    
    # MLM configuration
    mlm_probability: float = 0.15
    mlm_random_probability: float = 0.1
    mlm_keep_probability: float = 0.1
    
    # Patch configuration
    min_patch_size: int = 4
    max_patch_size: int = 32
    adaptive_patching: bool = True
    
    # Quality filtering
    min_quality_score: float = 0.5
    enable_quality_filtering: bool = True


@dataclass
class DataSample:
    """Individual data sample with metadata."""
    text: str
    byte_sequence: bytes
    quality_metrics: Optional[QualityMetrics] = None
    language: str = "unknown"
    content_type: str = "text"
    source: str = "unknown"
    length: int = 0
    entropy: float = 0.0
    
    def __post_init__(self):
        if self.length == 0:
            self.length = len(self.byte_sequence)
        if self.entropy == 0.0:
            self.entropy = calculate_byte_entropy(self.byte_sequence)


class BootstrapDataset(IterableDataset):
    """Dataset for bootstrap training with streaming and filtering."""
    
    def __init__(
        self,
        data_paths: List[str],
        quality_filter: Optional[DataQualityFilter] = None,
        batch_config: Optional[BatchConfig] = None,
        max_samples: Optional[int] = None,
        shuffle: bool = True,
        cache_size: int = 10000,
        num_workers: int = 4
    ):
        self.data_paths = [Path(p) for p in data_paths]
        self.quality_filter = quality_filter
        self.batch_config = batch_config or BatchConfig()
        self.max_samples = max_samples
        self.shuffle = shuffle
        self.cache_size = cache_size
        self.num_workers = num_workers
        
        # Validate data paths
        for path in self.data_paths:
            if not path.exists():
                raise FileNotFoundError(f"Data path not found: {path}")
        
        # Sample cache for shuffling
        self.sample_cache = []
        self.cache_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'samples_loaded': 0,
            'samples_filtered': 0,
            'bytes_processed': 0,
            'language_distribution': defaultdict(int),
            'content_type_distribution': defaultdict(int)
        }
        
        logger.info(f"Initialized BootstrapDataset with {len(self.data_paths)} data files")
    
    def _load_sample_from_line(self, line: str) -> Optional[DataSample]:
        """Load a single sample from a JSON line."""
        try:
            data = json.loads(line.strip())
            
            # Extract text
            text = data.get('text', '')
            if not text:
                return None
            
            # Convert to bytes
            try:
                byte_sequence = text.encode('utf-8')
            except UnicodeEncodeError:
                return None
            
            # Create sample
            sample = DataSample(
                text=text,
                byte_sequence=byte_sequence,
                language=data.get('language', 'unknown'),
                content_type=data.get('content_type', 'text'),
                source=data.get('source', 'unknown'),
                length=len(byte_sequence)
            )
            
            # Apply quality filter if available
            if self.quality_filter:
                is_accepted, quality_metrics = self.quality_filter.filter_document(text)
                if not is_accepted:
                    self.stats['samples_filtered'] += 1
                    return None
                sample.quality_metrics = quality_metrics
            
            # Update statistics
            self.stats['samples_loaded'] += 1
            self.stats['bytes_processed'] += len(byte_sequence)
            self.stats['language_distribution'][sample.language] += 1
            self.stats['content_type_distribution'][sample.content_type] += 1
            
            return sample
            
        except (json.JSONDecodeError, KeyError, ValueError):
            return None
    
    def _load_samples_from_file(self, file_path: Path) -> Iterator[DataSample]:
        """Load samples from a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if self.max_samples and self.stats['samples_loaded'] >= self.max_samples:
                        break
                    
                    sample = self._load_sample_from_line(line)
                    if sample:
                        yield sample
                        
        except Exception as e:
            logger.warning(f"Error loading from {file_path}: {e}")
    
    def _fill_cache(self):
        """Fill the sample cache with data from files."""
        samples_needed = self.cache_size - len(self.sample_cache)
        if samples_needed <= 0:
            return
        
        samples_collected = 0
        
        for file_path in self.data_paths:
            if samples_collected >= samples_needed:
                break
                
            for sample in self._load_samples_from_file(file_path):
                with self.cache_lock:
                    if len(self.sample_cache) < self.cache_size:
                        self.sample_cache.append(sample)
                        samples_collected += 1
                    else:
                        break
                
                if samples_collected >= samples_needed:
                    break
        
        # Shuffle cache if requested
        if self.shuffle and self.sample_cache:
            with self.cache_lock:
                random.shuffle(self.sample_cache)
    
    def __iter__(self) -> Iterator[DataSample]:
        """Iterate over dataset samples."""
        while True:
            # Fill cache if needed
            if len(self.sample_cache) < self.cache_size // 2:
                self._fill_cache()
            
            # Yield samples from cache
            with self.cache_lock:
                if not self.sample_cache:
                    break
                
                sample = self.sample_cache.pop(0)
            
            yield sample
            
            # Check max samples limit
            if self.max_samples and self.stats['samples_loaded'] >= self.max_samples:
                break
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            'samples_loaded': self.stats['samples_loaded'],
            'samples_filtered': self.stats['samples_filtered'],
            'bytes_processed': self.stats['bytes_processed'],
            'acceptance_rate': self.stats['samples_loaded'] / max(
                self.stats['samples_loaded'] + self.stats['samples_filtered'], 1
            ),
            'language_distribution': dict(self.stats['language_distribution']),
            'content_type_distribution': dict(self.stats['content_type_distribution']),
            'cache_size': len(self.sample_cache)
        }


class DynamicBatchCollator:
    """Dynamic batch collator for variable-length sequences."""
    
    def __init__(
        self,
        batch_config: BatchConfig,
        input_processor: BLTInputProcessor,
        multi_objective_trainer: MultiObjectiveTrainer
    ):
        self.config = batch_config
        self.input_processor = input_processor
        self.multi_objective_trainer = multi_objective_trainer
        
        # Special tokens
        self.mask_token_id = 256  # Special mask token for MLM
    
    def __call__(self, samples: List[DataSample]) -> Dict[str, Any]:
        """Collate samples into a batch."""
        if not samples:
            return {}
        
        # Filter by sequence length
        valid_samples = [
            s for s in samples 
            if self.config.min_sequence_length <= len(s.byte_sequence) <= self.config.max_sequence_length
        ]
        
        if not valid_samples:
            return {}
        
        # Sort by length for efficient batching
        if self.config.dynamic_batching:
            valid_samples.sort(key=lambda x: len(x.byte_sequence))
        
        # Prepare batch data
        batch_data = {
            'samples': valid_samples,
            'batch_size': len(valid_samples)
        }
        
        # Process input sequences
        input_ids, patch_boundaries = self._process_input_sequences(valid_samples)
        batch_data['input_ids'] = input_ids
        batch_data['patch_boundaries'] = patch_boundaries
        
        # Create MLM data if needed
        if 'mlm' in self.multi_objective_trainer.objectives:
            mlm_data = self._create_mlm_data(input_ids)
            batch_data.update(mlm_data)
        
        # Add byte sequences for entropy and contrastive objectives
        batch_data['byte_sequences'] = [s.byte_sequence for s in valid_samples]
        
        # Add metadata
        batch_data['languages'] = [s.language for s in valid_samples]
        batch_data['content_types'] = [s.content_type for s in valid_samples]
        batch_data['lengths'] = [s.length for s in valid_samples]
        batch_data['entropies'] = [s.entropy for s in valid_samples]
        
        return batch_data
    
    def _process_input_sequences(
        self, 
        samples: List[DataSample]
    ) -> Tuple[Tensor, List[List[Tuple[int, int]]]]:
        """Process input sequences into tensors."""
        all_input_ids = []
        all_patch_boundaries = []
        
        max_length = max(len(s.byte_sequence) for s in samples)
        
        # Pad to multiple if configured
        if self.config.pad_to_multiple > 1:
            max_length = ((max_length + self.config.pad_to_multiple - 1) 
                         // self.config.pad_to_multiple) * self.config.pad_to_multiple
        
        for sample in samples:
            # Convert bytes to input IDs
            input_ids = list(sample.byte_sequence)
            
            # Pad sequence
            padded_length = max_length
            if len(input_ids) < padded_length:
                input_ids.extend([0] * (padded_length - len(input_ids)))  # Pad with zeros
            
            all_input_ids.append(input_ids)
            
            # Create patch boundaries (simplified for now)
            # In practice, this would use the input processor
            boundaries = []
            patch_size = self.config.min_patch_size
            for i in range(0, len(sample.byte_sequence), patch_size):
                end = min(i + patch_size, len(sample.byte_sequence))
                boundaries.append((i, end))
            
            all_patch_boundaries.append(boundaries)
        
        input_tensor = torch.tensor(all_input_ids, dtype=torch.long)
        
        return input_tensor, all_patch_boundaries
    
    def _create_mlm_data(self, input_ids: Tensor) -> Dict[str, Tensor]:
        """Create masked language modeling data."""
        batch_size, seq_len = input_ids.shape
        
        # Create random mask
        mask_prob = self.config.mlm_probability
        random_tensor = torch.rand(batch_size, seq_len)
        mask = random_tensor < mask_prob
        
        # Create targets (only for masked positions)
        mlm_targets = input_ids.clone()
        mlm_targets[~mask] = -100  # Ignore non-masked positions
        
        # Create masked input
        mlm_input = input_ids.clone()
        
        # Apply masking strategy
        for i in range(batch_size):
            for j in range(seq_len):
                if mask[i, j]:
                    rand_val = random.random()
                    
                    if rand_val < 0.8:  # 80% mask token
                        mlm_input[i, j] = self.mask_token_id
                    elif rand_val < 0.9:  # 10% random token
                        mlm_input[i, j] = random.randint(0, 255)
                    # 10% keep original
        
        return {
            'mlm_input': mlm_input,
            'mlm_targets': mlm_targets,
            'mlm_mask': mask
        }


class BootstrapDataLoader:
    """High-level data loader for bootstrap training."""
    
    def __init__(
        self,
        data_paths: List[str],
        batch_config: Optional[BatchConfig] = None,
        quality_filter_config: Optional[Dict[str, Any]] = None,
        max_samples: Optional[int] = None,
        num_workers: int = 4,
        shuffle: bool = True,
        cache_size: int = 10000
    ):
        self.batch_config = batch_config or BatchConfig()
        
        # Create quality filter
        if quality_filter_config:
            from .data_filters import create_quality_filter
            self.quality_filter = create_quality_filter(quality_filter_config)
        else:
            self.quality_filter = None
        
        # Create dataset
        self.dataset = BootstrapDataset(
            data_paths=data_paths,
            quality_filter=self.quality_filter,
            batch_config=self.batch_config,
            max_samples=max_samples,
            shuffle=shuffle,
            cache_size=cache_size,
            num_workers=num_workers
        )
        
        # Create input processor and multi-objective trainer
        self.input_processor = BLTInputProcessor(
            min_patch_size=self.batch_config.min_patch_size,
            max_patch_size=self.batch_config.max_patch_size
        )
        
        objectives = create_default_objectives()
        self.multi_objective_trainer = MultiObjectiveTrainer(objectives)
        
        # Create batch collator
        self.collator = DynamicBatchCollator(
            self.batch_config,
            self.input_processor,
            self.multi_objective_trainer
        )
        
        logger.info("Initialized BootstrapDataLoader")
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over batches."""
        batch_samples = []
        
        for sample in self.dataset:
            batch_samples.append(sample)
            
            if len(batch_samples) >= self.batch_config.batch_size:
                # Create batch
                batch = self.collator(batch_samples)
                
                if batch:  # Only yield non-empty batches
                    yield batch
                
                batch_samples = []
        
        # Handle remaining samples
        if batch_samples and not self.batch_config.drop_last:
            batch = self.collator(batch_samples)
            if batch:
                yield batch
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = self.dataset.get_statistics()
        
        if self.quality_filter:
            filter_stats = self.quality_filter.get_statistics()
            stats['quality_filter'] = filter_stats
        
        return stats
    
    def save_statistics(self, filepath: str):
        """Save statistics to file."""
        stats = self.get_statistics()
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)


def create_bootstrap_dataloader(
    data_paths: List[str],
    batch_size: int = 16,
    max_sequence_length: int = 1024,
    quality_filter_config: Optional[Dict[str, Any]] = None,
    num_workers: int = 4,
    **kwargs
) -> BootstrapDataLoader:
    """Create a bootstrap data loader with default configuration."""
    
    batch_config = BatchConfig(
        batch_size=batch_size,
        max_sequence_length=max_sequence_length,
        **kwargs
    )
    
    return BootstrapDataLoader(
        data_paths=data_paths,
        batch_config=batch_config,
        quality_filter_config=quality_filter_config,
        num_workers=num_workers
    )


def create_synthetic_data_loader(
    generated_data_path: str,
    original_data_paths: List[str],
    mix_ratio: float = 0.3,
    **kwargs
) -> BootstrapDataLoader:
    """Create a data loader that mixes synthetic and original data.
    
    Args:
        generated_data_path: Path to synthetic/generated data
        original_data_paths: Paths to original training data
        mix_ratio: Ratio of synthetic data in the mix (0.0 to 1.0)
        **kwargs: Additional arguments for BootstrapDataLoader
    
    Returns:
        Configured data loader with mixed data
    """
    # Combine data paths with appropriate sampling
    all_paths = original_data_paths.copy()
    
    # Add synthetic data path multiple times based on mix ratio
    synthetic_weight = int(mix_ratio * len(original_data_paths) / (1 - mix_ratio))
    for _ in range(max(1, synthetic_weight)):
        all_paths.append(generated_data_path)
    
    return BootstrapDataLoader(
        data_paths=all_paths,
        shuffle=True,  # Important for mixing
        **kwargs
    )


# Utility functions for data preparation

def prepare_bootstrap_data(
    input_dir: str,
    output_dir: str,
    quality_filter_config: Optional[Dict[str, Any]] = None,
    chunk_size: int = 10000,
    num_workers: int = 4
) -> Dict[str, Any]:
    """Prepare data for bootstrap training with filtering and chunking.
    
    Args:
        input_dir: Directory containing raw data files
        output_dir: Directory to save prepared data
        quality_filter_config: Configuration for quality filtering
        chunk_size: Number of samples per output chunk
        num_workers: Number of worker processes
        
    Returns:
        Statistics about data preparation
    """
    from .data_filters import filter_dataset
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each input file
    total_stats = {
        'files_processed': 0,
        'total_samples': 0,
        'accepted_samples': 0,
        'output_chunks': 0
    }
    
    for input_file in input_path.glob('*.jsonl'):
        output_file = output_path / f"filtered_{input_file.name}"
        
        # Filter the dataset
        stats = filter_dataset(
            str(input_file),
            str(output_file),
            filter_config=quality_filter_config,
            batch_size=chunk_size
        )
        
        total_stats['files_processed'] += 1
        total_stats['total_samples'] += stats.get('total_processed', 0)
        total_stats['accepted_samples'] += stats.get('accepted', 0)
        total_stats['output_chunks'] += 1
        
        logger.info(f"Processed {input_file.name}: "
                   f"{stats.get('accepted', 0)}/{stats.get('total_processed', 0)} samples accepted")
    
    # Save preparation statistics
    stats_file = output_path / 'preparation_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(total_stats, f, indent=2)
    
    logger.info(f"Data preparation complete: {total_stats['accepted_samples']}/{total_stats['total_samples']} samples accepted")
    
    return total_stats