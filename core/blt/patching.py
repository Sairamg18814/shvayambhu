"""Dynamic patching system for BLT.

This module implements the dynamic patching algorithm that converts
byte sequences into variable-sized patches based on content entropy.
"""

from typing import List, Tuple, Optional, Union
import numpy as np
import torch
from torch import Tensor

from .entropy import (
    calculate_byte_entropy,
    determine_patch_boundaries,
    adaptive_entropy_patching,
    EntropyCalculator
)


class ByteProcessor:
    """Process raw bytes for BLT input.
    
    Handles UTF-8 validation, normalization, and conversion
    to numerical representations suitable for neural networks.
    """
    
    def __init__(self):
        """Initialize byte processor."""
        self.entropy_calculator = EntropyCalculator()
    
    def process_text(self, text: Union[str, bytes]) -> bytes:
        """Convert text to normalized bytes.
        
        Args:
            text: Input text as string or bytes
            
        Returns:
            UTF-8 encoded bytes
            
        Raises:
            ValueError: If input cannot be encoded as UTF-8
        """
        if isinstance(text, str):
            try:
                return text.encode('utf-8')
            except UnicodeEncodeError as e:
                raise ValueError(f"Failed to encode text as UTF-8: {e}")
        elif isinstance(text, bytes):
            # Validate UTF-8
            try:
                text.decode('utf-8')
                return text
            except UnicodeDecodeError as e:
                raise ValueError(f"Invalid UTF-8 bytes: {e}")
        else:
            raise TypeError(f"Expected str or bytes, got {type(text)}")
    
    def bytes_to_ids(self, byte_sequence: bytes) -> List[int]:
        """Convert bytes to integer IDs (0-255).
        
        Args:
            byte_sequence: Input bytes
            
        Returns:
            List of byte values as integers
        """
        return list(byte_sequence)
    
    def ids_to_bytes(self, ids: List[int]) -> bytes:
        """Convert integer IDs back to bytes.
        
        Args:
            ids: List of integers (0-255)
            
        Returns:
            Reconstructed byte sequence
            
        Raises:
            ValueError: If any ID is outside 0-255 range
        """
        if any(id < 0 or id > 255 for id in ids):
            raise ValueError("All IDs must be in range 0-255")
        return bytes(ids)


class DynamicPatcher:
    """Create variable-sized patches from byte sequences.
    
    Uses entropy-based sizing to create optimal patches:
    - High entropy (random/diverse) -> smaller patches
    - Low entropy (repetitive) -> larger patches
    """
    
    def __init__(
        self,
        min_patch_size: int = 4,
        max_patch_size: int = 32,
        target_patches: int = 64,
        entropy_threshold_low: float = 0.3,
        entropy_threshold_high: float = 0.7
    ):
        """Initialize patcher with configuration.
        
        Args:
            min_patch_size: Minimum bytes per patch
            max_patch_size: Maximum bytes per patch
            target_patches: Target number of patches (approximate)
            entropy_threshold_low: Below this, use larger patches
            entropy_threshold_high: Above this, use smaller patches
        """
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.target_patches = target_patches
        self.entropy_threshold_low = entropy_threshold_low
        self.entropy_threshold_high = entropy_threshold_high
        self.entropy_calculator = EntropyCalculator()
    
    def create_patches(
        self,
        byte_sequence: bytes,
        method: str = "adaptive"
    ) -> List[Tuple[int, int]]:
        """Create patches from byte sequence.
        
        Args:
            byte_sequence: Input bytes to patch
            method: Patching method ("adaptive", "fixed", "entropy")
            
        Returns:
            List of (start, end) tuples defining patches
        """
        if len(byte_sequence) == 0:
            return []
        
        if method == "fixed":
            return self._create_fixed_patches(byte_sequence)
        elif method == "entropy":
            return self._create_entropy_patches(byte_sequence)
        else:  # adaptive
            return self._create_adaptive_patches(byte_sequence)
    
    def _create_fixed_patches(self, byte_sequence: bytes) -> List[Tuple[int, int]]:
        """Create fixed-size patches (baseline method)."""
        patch_size = (self.min_patch_size + self.max_patch_size) // 2
        patches = []
        
        for i in range(0, len(byte_sequence), patch_size):
            end = min(i + patch_size, len(byte_sequence))
            patches.append((i, end))
        
        return patches
    
    def _create_entropy_patches(self, byte_sequence: bytes) -> List[Tuple[int, int]]:
        """Create patches based on entropy boundaries."""
        boundaries = determine_patch_boundaries(
            byte_sequence,
            self.min_patch_size,
            self.max_patch_size,
            self.entropy_threshold_low,
            self.entropy_threshold_high
        )
        
        patches = []
        start = 0
        for end in boundaries:
            patches.append((start, end))
            start = end
        
        return patches
    
    def _create_adaptive_patches(self, byte_sequence: bytes) -> List[Tuple[int, int]]:
        """Create patches using adaptive algorithm."""
        return adaptive_entropy_patching(
            byte_sequence,
            self.target_patches,
            self.min_patch_size,
            self.max_patch_size
        )
    
    def extract_patch_bytes(
        self,
        byte_sequence: bytes,
        patches: List[Tuple[int, int]]
    ) -> List[bytes]:
        """Extract actual byte sequences for each patch.
        
        Args:
            byte_sequence: Original byte sequence
            patches: List of (start, end) boundaries
            
        Returns:
            List of byte sequences for each patch
        """
        return [byte_sequence[start:end] for start, end in patches]
    
    def patches_to_tensor(
        self,
        patches: List[bytes],
        embedding_dim: int = 256,
        device: Optional[torch.device] = None
    ) -> Tensor:
        """Convert patches to tensor representation.
        
        Args:
            patches: List of byte patches
            embedding_dim: Dimension for byte embeddings
            device: Target device for tensor
            
        Returns:
            Tensor of shape (num_patches, max_patch_length, embedding_dim)
        """
        if not patches:
            return torch.zeros(0, 0, embedding_dim, device=device)
        
        # Find maximum patch length for padding
        max_length = max(len(patch) for patch in patches)
        
        # Create padded tensor
        num_patches = len(patches)
        patch_tensor = torch.zeros(
            num_patches, max_length, dtype=torch.long, device=device
        )
        
        # Fill tensor with byte values
        for i, patch in enumerate(patches):
            patch_ids = list(patch)
            patch_tensor[i, :len(patch_ids)] = torch.tensor(patch_ids)
        
        return patch_tensor
    
    def create_attention_mask(
        self,
        patches: List[bytes],
        device: Optional[torch.device] = None
    ) -> Tensor:
        """Create attention mask for patches.
        
        Args:
            patches: List of byte patches
            device: Target device for tensor
            
        Returns:
            Boolean tensor indicating valid positions
        """
        if not patches:
            return torch.zeros(0, 0, dtype=torch.bool, device=device)
        
        max_length = max(len(patch) for patch in patches)
        num_patches = len(patches)
        
        mask = torch.zeros(
            num_patches, max_length, dtype=torch.bool, device=device
        )
        
        for i, patch in enumerate(patches):
            mask[i, :len(patch)] = True
        
        return mask


class PatchEmbedder:
    """Convert byte patches to continuous embeddings."""
    
    def __init__(
        self,
        vocab_size: int = 256,  # Byte vocabulary
        embedding_dim: int = 512,
        max_patch_length: int = 32,
        device: Optional[torch.device] = None
    ):
        """Initialize patch embedder.
        
        Args:
            vocab_size: Size of byte vocabulary (256)
            embedding_dim: Dimension of embeddings
            max_patch_length: Maximum bytes per patch
            device: Target device
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_patch_length = max_patch_length
        self.device = device or torch.device("cpu")
        
        # Byte embeddings
        self.byte_embeddings = torch.nn.Embedding(
            vocab_size, embedding_dim
        ).to(self.device)
        
        # Positional embeddings for within-patch positions
        self.position_embeddings = torch.nn.Embedding(
            max_patch_length, embedding_dim
        ).to(self.device)
        
        # Patch-level positional encoding will be added by the encoder
    
    def embed_patches(
        self,
        patch_tensor: Tensor,
        attention_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Embed patches with byte and position embeddings.
        
        Args:
            patch_tensor: Tensor of byte IDs (batch, num_patches, patch_length)
            attention_mask: Boolean mask for valid positions
            
        Returns:
            Tuple of (embedded_patches, patch_mask)
        """
        batch_size, num_patches, patch_length = patch_tensor.shape
        
        # Get byte embeddings
        byte_embeds = self.byte_embeddings(patch_tensor)
        
        # Add positional embeddings
        positions = torch.arange(
            patch_length, device=self.device
        ).unsqueeze(0).unsqueeze(0).expand(batch_size, num_patches, -1)
        
        pos_embeds = self.position_embeddings(positions)
        
        # Combine embeddings
        embeddings = byte_embeds + pos_embeds
        
        # Apply mask if provided
        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1).float()
        
        # Aggregate patch representations (mean pooling)
        if attention_mask is not None:
            patch_lengths = attention_mask.sum(dim=-1, keepdim=True).float()
            patch_embeds = embeddings.sum(dim=2) / patch_lengths.clamp(min=1.0)
        else:
            patch_embeds = embeddings.mean(dim=2)
        
        # Create patch-level mask (which patches are valid)
        if attention_mask is not None:
            patch_mask = attention_mask.any(dim=-1)
        else:
            patch_mask = torch.ones(
                batch_size, num_patches, dtype=torch.bool, device=self.device
            )
        
        return patch_embeds, patch_mask


# High-level API
class BLTInputProcessor:
    """High-level API for BLT input processing."""
    
    def __init__(
        self,
        min_patch_size: int = 4,
        max_patch_size: int = 32,
        target_patches: int = 64,
        embedding_dim: int = 512,
        device: Optional[torch.device] = None
    ):
        """Initialize BLT input processor.
        
        Args:
            min_patch_size: Minimum patch size
            max_patch_size: Maximum patch size
            target_patches: Target number of patches
            embedding_dim: Dimension of patch embeddings
            device: Target device
        """
        self.byte_processor = ByteProcessor()
        self.patcher = DynamicPatcher(
            min_patch_size=min_patch_size,
            max_patch_size=max_patch_size,
            target_patches=target_patches
        )
        self.embedder = PatchEmbedder(
            embedding_dim=embedding_dim,
            max_patch_length=max_patch_size,
            device=device
        )
        self.device = device
    
    def process(
        self,
        text: Union[str, bytes],
        return_patches: bool = False
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, List[bytes]]]:
        """Process text input to patch embeddings.
        
        Args:
            text: Input text or bytes
            return_patches: Whether to return raw patches
            
        Returns:
            Tuple of (patch_embeddings, patch_mask) or
            (patch_embeddings, patch_mask, patches) if return_patches=True
        """
        # Convert to bytes
        byte_sequence = self.byte_processor.process_text(text)
        
        # Create patches
        patch_boundaries = self.patcher.create_patches(byte_sequence)
        patches = self.patcher.extract_patch_bytes(byte_sequence, patch_boundaries)
        
        # Convert to tensor
        patch_tensor = self.patcher.patches_to_tensor(patches, device=self.device)
        attention_mask = self.patcher.create_attention_mask(patches, device=self.device)
        
        # Embed patches
        patch_embeds, patch_mask = self.embedder.embed_patches(
            patch_tensor.unsqueeze(0),  # Add batch dimension
            attention_mask.unsqueeze(0)
        )
        
        # Remove batch dimension
        patch_embeds = patch_embeds.squeeze(0)
        patch_mask = patch_mask.squeeze(0)
        
        if return_patches:
            return patch_embeds, patch_mask, patches
        else:
            return patch_embeds, patch_mask