"""Metal-accelerated operations for BLT components.

This module provides Metal Performance Shaders acceleration for
BLT encoder/decoder operations on Apple Silicon.
"""

import os
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import metal

# Try to import Metal libraries
try:
    import metalcompute as mc
    HAS_METAL = True
except ImportError:
    HAS_METAL = False
    mc = None

try:
    import coremltools as ct
    HAS_COREML = True
except ImportError:
    HAS_COREML = False
    ct = None


@dataclass
class MetalConfig:
    """Configuration for Metal operations."""
    device: str = "default"  # GPU device
    precision: str = "float32"  # float32, float16, int8
    use_shared_memory: bool = True
    max_buffer_size: int = 1024 * 1024 * 1024  # 1GB
    enable_profiling: bool = False


class MetalBLTOps:
    """Metal-accelerated operations for BLT."""
    
    def __init__(self, config: Optional[MetalConfig] = None):
        self.config = config or MetalConfig()
        self.device = None
        self.command_queue = None
        self.library = None
        self.kernels = {}
        
        if HAS_METAL:
            self._initialize_metal()
    
    def _initialize_metal(self):
        """Initialize Metal device and compile kernels."""
        # Get Metal device
        self.device = mc.Device()
        self.command_queue = self.device.create_command_queue()
        
        # Load shader library
        shader_path = os.path.join(
            os.path.dirname(__file__), 
            "../../shaders/blt_kernels.metal"
        )
        
        if os.path.exists(shader_path):
            with open(shader_path, 'r') as f:
                shader_source = f.read()
            
            self.library = self.device.create_library(shader_source)
            
            # Create compute pipelines
            self._create_kernels()
        else:
            # Define kernels inline if shader file not found
            self._create_inline_kernels()
    
    def _create_inline_kernels(self):
        """Create Metal kernels inline."""
        # Byte embedding kernel
        byte_embedding_kernel = """
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void byte_embedding(
            device const uint8_t* bytes [[buffer(0)]],
            device float* embeddings [[buffer(1)]],
            constant uint& embedding_dim [[buffer(2)]],
            constant uint& num_bytes [[buffer(3)]],
            device const float* embedding_table [[buffer(4)]],
            uint gid [[thread_position_in_grid]])
        {
            if (gid >= num_bytes) return;
            
            uint byte_val = bytes[gid];
            uint offset = byte_val * embedding_dim;
            uint out_offset = gid * embedding_dim;
            
            for (uint i = 0; i < embedding_dim; i++) {
                embeddings[out_offset + i] = embedding_table[offset + i];
            }
        }
        """
        
        # Patch aggregation kernel
        patch_aggregation_kernel = """
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void patch_aggregation(
            device const float* byte_embeddings [[buffer(0)]],
            device float* patch_embeddings [[buffer(1)]],
            device const uint* patch_boundaries [[buffer(2)]],
            constant uint& embedding_dim [[buffer(3)]],
            constant uint& num_patches [[buffer(4)]],
            uint gid [[thread_position_in_grid]])
        {
            if (gid >= num_patches) return;
            
            uint start = patch_boundaries[gid * 2];
            uint end = patch_boundaries[gid * 2 + 1];
            uint patch_size = end - start;
            
            uint out_offset = gid * embedding_dim;
            
            // Average pooling
            for (uint i = 0; i < embedding_dim; i++) {
                float sum = 0.0f;
                for (uint j = start; j < end; j++) {
                    sum += byte_embeddings[j * embedding_dim + i];
                }
                patch_embeddings[out_offset + i] = sum / float(patch_size);
            }
        }
        """
        
        # Entropy calculation kernel
        entropy_kernel = """
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void calculate_entropy(
            device const uint8_t* bytes [[buffer(0)]],
            device float* entropy [[buffer(1)]],
            constant uint& window_size [[buffer(2)]],
            constant uint& num_windows [[buffer(3)]],
            uint gid [[thread_position_in_grid]])
        {
            if (gid >= num_windows) return;
            
            // Count byte frequencies in window
            uint counts[256];
            for (int i = 0; i < 256; i++) counts[i] = 0;
            
            uint start = gid * window_size;
            for (uint i = 0; i < window_size; i++) {
                counts[bytes[start + i]]++;
            }
            
            // Calculate entropy
            float ent = 0.0f;
            float log2 = 0.693147180559945309417f;
            
            for (int i = 0; i < 256; i++) {
                if (counts[i] > 0) {
                    float p = float(counts[i]) / float(window_size);
                    ent -= p * log(p) / log2;
                }
            }
            
            entropy[gid] = ent;
        }
        """
        
        # Create kernels from source
        if HAS_METAL:
            self.library = self.device.create_library(
                byte_embedding_kernel + patch_aggregation_kernel + entropy_kernel
            )
            
            self.kernels['byte_embedding'] = self.library.create_function('byte_embedding')
            self.kernels['patch_aggregation'] = self.library.create_function('patch_aggregation')
            self.kernels['calculate_entropy'] = self.library.create_function('calculate_entropy')
    
    def byte_embedding_metal(
        self,
        bytes_array: np.ndarray,
        embedding_table: torch.Tensor,
        embedding_dim: int
    ) -> torch.Tensor:
        """Perform byte embedding using Metal."""
        if not HAS_METAL:
            # Fallback to CPU
            return self._byte_embedding_cpu(bytes_array, embedding_table, embedding_dim)
        
        num_bytes = len(bytes_array)
        
        # Create Metal buffers
        bytes_buffer = self.device.create_buffer(bytes_array)
        embeddings_buffer = self.device.create_buffer(
            num_bytes * embedding_dim * 4  # float32
        )
        table_buffer = self.device.create_buffer(
            embedding_table.cpu().numpy().astype(np.float32)
        )
        
        # Create compute encoder
        compute_encoder = self.command_queue.create_compute_encoder()
        compute_encoder.set_compute_pipeline(self.kernels['byte_embedding'])
        
        # Set buffers
        compute_encoder.set_buffer(bytes_buffer, 0)
        compute_encoder.set_buffer(embeddings_buffer, 1)
        compute_encoder.set_buffer(embedding_dim, 2)
        compute_encoder.set_buffer(num_bytes, 3)
        compute_encoder.set_buffer(table_buffer, 4)
        
        # Dispatch threads
        threads_per_group = 256
        thread_groups = (num_bytes + threads_per_group - 1) // threads_per_group
        compute_encoder.dispatch_thread_groups(thread_groups, threads_per_group)
        
        # Commit and wait
        compute_encoder.end_encoding()
        self.command_queue.commit()
        self.command_queue.wait_until_completed()
        
        # Get results
        embeddings = np.frombuffer(
            embeddings_buffer.contents(), 
            dtype=np.float32
        ).reshape(num_bytes, embedding_dim)
        
        return torch.from_numpy(embeddings)
    
    def patch_aggregation_metal(
        self,
        byte_embeddings: torch.Tensor,
        patch_boundaries: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """Aggregate byte embeddings into patches using Metal."""
        if not HAS_METAL:
            return self._patch_aggregation_cpu(byte_embeddings, patch_boundaries)
        
        num_patches = len(patch_boundaries)
        embedding_dim = byte_embeddings.shape[1]
        
        # Flatten boundaries
        boundaries_flat = np.array(
            [[start, end] for start, end in patch_boundaries],
            dtype=np.uint32
        ).flatten()
        
        # Create Metal buffers
        embeddings_buffer = self.device.create_buffer(
            byte_embeddings.cpu().numpy().astype(np.float32)
        )
        patch_buffer = self.device.create_buffer(
            num_patches * embedding_dim * 4  # float32
        )
        boundaries_buffer = self.device.create_buffer(boundaries_flat)
        
        # Create compute encoder
        compute_encoder = self.command_queue.create_compute_encoder()
        compute_encoder.set_compute_pipeline(self.kernels['patch_aggregation'])
        
        # Set buffers
        compute_encoder.set_buffer(embeddings_buffer, 0)
        compute_encoder.set_buffer(patch_buffer, 1)
        compute_encoder.set_buffer(boundaries_buffer, 2)
        compute_encoder.set_buffer(embedding_dim, 3)
        compute_encoder.set_buffer(num_patches, 4)
        
        # Dispatch
        threads_per_group = 256
        thread_groups = (num_patches + threads_per_group - 1) // threads_per_group
        compute_encoder.dispatch_thread_groups(thread_groups, threads_per_group)
        
        # Commit and wait
        compute_encoder.end_encoding()
        self.command_queue.commit()
        self.command_queue.wait_until_completed()
        
        # Get results
        patch_embeddings = np.frombuffer(
            patch_buffer.contents(),
            dtype=np.float32
        ).reshape(num_patches, embedding_dim)
        
        return torch.from_numpy(patch_embeddings)
    
    def calculate_entropy_metal(
        self,
        bytes_array: np.ndarray,
        window_size: int = 256
    ) -> np.ndarray:
        """Calculate sliding window entropy using Metal."""
        if not HAS_METAL:
            return self._calculate_entropy_cpu(bytes_array, window_size)
        
        num_windows = len(bytes_array) - window_size + 1
        
        # Create Metal buffers
        bytes_buffer = self.device.create_buffer(bytes_array)
        entropy_buffer = self.device.create_buffer(num_windows * 4)  # float32
        
        # Create compute encoder
        compute_encoder = self.command_queue.create_compute_encoder()
        compute_encoder.set_compute_pipeline(self.kernels['calculate_entropy'])
        
        # Set buffers
        compute_encoder.set_buffer(bytes_buffer, 0)
        compute_encoder.set_buffer(entropy_buffer, 1)
        compute_encoder.set_buffer(window_size, 2)
        compute_encoder.set_buffer(num_windows, 3)
        
        # Dispatch
        threads_per_group = 256
        thread_groups = (num_windows + threads_per_group - 1) // threads_per_group
        compute_encoder.dispatch_thread_groups(thread_groups, threads_per_group)
        
        # Commit and wait
        compute_encoder.end_encoding()
        self.command_queue.commit()
        self.command_queue.wait_until_completed()
        
        # Get results
        entropy = np.frombuffer(
            entropy_buffer.contents(),
            dtype=np.float32
        )
        
        return entropy
    
    def optimized_attention_metal(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Optimized attention using Metal."""
        # Use Flash Attention kernel if available
        if hasattr(self, 'flash_attention_kernel'):
            return self._flash_attention_metal(query, key, value, mask)
        else:
            # Fallback to standard attention
            return self._standard_attention(query, key, value, mask)
    
    def _byte_embedding_cpu(
        self,
        bytes_array: np.ndarray,
        embedding_table: torch.Tensor,
        embedding_dim: int
    ) -> torch.Tensor:
        """CPU fallback for byte embedding."""
        indices = torch.from_numpy(bytes_array).long()
        return embedding_table[indices]
    
    def _patch_aggregation_cpu(
        self,
        byte_embeddings: torch.Tensor,
        patch_boundaries: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """CPU fallback for patch aggregation."""
        patches = []
        for start, end in patch_boundaries:
            patch_emb = byte_embeddings[start:end].mean(dim=0)
            patches.append(patch_emb)
        return torch.stack(patches)
    
    def _calculate_entropy_cpu(
        self,
        bytes_array: np.ndarray,
        window_size: int
    ) -> np.ndarray:
        """CPU fallback for entropy calculation."""
        from shvayambhu.core.blt.entropy import calculate_byte_entropy
        
        entropies = []
        for i in range(len(bytes_array) - window_size + 1):
            window = bytes_array[i:i+window_size]
            entropy = calculate_byte_entropy(window)
            entropies.append(entropy)
        
        return np.array(entropies)
    
    def _standard_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Standard attention computation."""
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / np.sqrt(query.shape[-1])
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        
        return output
    
    def profile_operation(self, operation_name: str, *args, **kwargs):
        """Profile a Metal operation."""
        if not self.config.enable_profiling:
            return
        
        # TODO: Implement Metal profiling
        pass


class MetalBLTEncoder:
    """Metal-accelerated BLT encoder."""
    
    def __init__(
        self,
        embedding_dim: int = 768,
        max_patch_size: int = 32,
        metal_config: Optional[MetalConfig] = None
    ):
        self.embedding_dim = embedding_dim
        self.max_patch_size = max_patch_size
        self.metal_ops = MetalBLTOps(metal_config)
        
        # Create embedding table
        self.embedding_table = torch.randn(256, embedding_dim)
    
    def encode(self, byte_sequence: bytes) -> Dict[str, Any]:
        """Encode byte sequence using Metal acceleration."""
        # Convert to numpy array
        bytes_array = np.frombuffer(byte_sequence, dtype=np.uint8)
        
        # Calculate entropy for patching
        entropy = self.metal_ops.calculate_entropy_metal(bytes_array)
        
        # Determine patch boundaries based on entropy
        boundaries = self._determine_boundaries(entropy, len(bytes_array))
        
        # Embed bytes
        byte_embeddings = self.metal_ops.byte_embedding_metal(
            bytes_array,
            self.embedding_table,
            self.embedding_dim
        )
        
        # Aggregate into patches
        patch_embeddings = self.metal_ops.patch_aggregation_metal(
            byte_embeddings,
            boundaries
        )
        
        return {
            'patch_embeddings': patch_embeddings,
            'patch_boundaries': boundaries,
            'byte_embeddings': byte_embeddings,
            'entropy': entropy
        }
    
    def _determine_boundaries(
        self,
        entropy: np.ndarray,
        total_length: int
    ) -> List[Tuple[int, int]]:
        """Determine patch boundaries based on entropy."""
        # Simple implementation - can be optimized
        boundaries = []
        start = 0
        
        while start < total_length:
            # Determine patch size based on local entropy
            if start < len(entropy):
                local_entropy = entropy[start]
                # High entropy -> smaller patches
                patch_size = int(self.max_patch_size * (1 - local_entropy / 8))
                patch_size = max(4, min(patch_size, self.max_patch_size))
            else:
                patch_size = 16  # Default
            
            end = min(start + patch_size, total_length)
            boundaries.append((start, end))
            start = end
        
        return boundaries


class MetalBLTDecoder:
    """Metal-accelerated BLT decoder."""
    
    def __init__(
        self,
        hidden_dim: int = 768,
        vocab_size: int = 256,
        metal_config: Optional[MetalConfig] = None
    ):
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.metal_ops = MetalBLTOps(metal_config)
        
        # Output projection
        self.output_projection = torch.randn(hidden_dim, vocab_size)
    
    def decode(
        self,
        hidden_states: torch.Tensor,
        patch_boundaries: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """Decode hidden states to byte logits using Metal."""
        # TODO: Implement Metal-accelerated decoding
        # For now, use CPU implementation
        
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        
        # Project to vocabulary
        logits = torch.matmul(hidden_states, self.output_projection)
        
        return logits


def create_metal_accelerated_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create Metal-accelerated BLT pipeline components."""
    metal_config = MetalConfig(
        precision=config.get('precision', 'float32'),
        enable_profiling=config.get('profiling', False)
    )
    
    components = {
        'encoder': MetalBLTEncoder(
            embedding_dim=config['patch_embedding_dim'],
            max_patch_size=config['max_patch_size'],
            metal_config=metal_config
        ),
        'decoder': MetalBLTDecoder(
            hidden_dim=config['hidden_dim'],
            vocab_size=config['vocab_size'],
            metal_config=metal_config
        ),
        'metal_ops': MetalBLTOps(metal_config)
    }
    
    return components