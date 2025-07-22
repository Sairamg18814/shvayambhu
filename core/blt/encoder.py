"""Local Encoder for BLT (Byte Latent Transformer).

This module implements the local encoder that converts byte sequences
to patch embeddings using entropy-based dynamic patching.
"""

from typing import Optional, Tuple, List, Dict, Any, Union
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from .patching import BLTInputProcessor, DynamicPatcher, ByteProcessor


class LocalEncoder(nn.Module):
    """Converts byte sequences to patches using entropy-based grouping.
    
    The encoder performs:
    1. Dynamic patching based on content entropy
    2. Byte-to-embedding conversion
    3. Patch aggregation and encoding
    4. Positional encoding at patch level
    """
    
    def __init__(
        self,
        vocab_size: int = 256,
        embedding_dim: int = 512,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        patch_size: int = 16,
        min_patch: int = 4,
        max_patch: int = 32,
        dropout: float = 0.1,
        max_seq_length: int = 1024,
        device: Optional[torch.device] = None
    ):
        """Initialize the local encoder.
        
        Args:
            vocab_size: Size of byte vocabulary (256)
            embedding_dim: Dimension of byte embeddings
            hidden_dim: Hidden dimension for transformer
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            patch_size: Default patch size
            min_patch: Minimum patch size
            max_patch: Maximum patch size
            dropout: Dropout rate
            max_seq_length: Maximum sequence length in patches
            device: Target device
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.min_patch = min_patch
        self.max_patch = max_patch
        self.device = device or torch.device("cpu")
        
        # Byte embeddings
        self.byte_embeddings = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )
        
        # Patch-level positional embeddings
        self.patch_position_embeddings = nn.Embedding(
            max_seq_length, hidden_dim
        )
        
        # Within-patch positional embeddings
        self.local_position_embeddings = nn.Embedding(
            max_patch, embedding_dim
        )
        
        # Projection from embedding to hidden dimension
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        
        # Patch aggregation layers
        self.patch_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for stability
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize processor
        self.input_processor = BLTInputProcessor(
            min_patch_size=min_patch,
            max_patch_size=max_patch,
            embedding_dim=embedding_dim,
            device=self.device
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def compute_entropy(self, byte_sequence: bytes) -> float:
        """Calculate local entropy for dynamic patching.
        
        Args:
            byte_sequence: Input byte sequence
            
        Returns:
            Entropy value between 0 and 1
        """
        from .entropy import calculate_byte_entropy
        return calculate_byte_entropy(byte_sequence)
    
    def create_patches(self, byte_sequence: bytes) -> List[Tuple[int, int]]:
        """Create variable-size patches based on content entropy.
        
        Args:
            byte_sequence: Input byte sequence
            
        Returns:
            List of (start, end) tuples defining patch boundaries
        """
        patcher = DynamicPatcher(
            min_patch_size=self.min_patch,
            max_patch_size=self.max_patch,
            target_patches=64
        )
        return patcher.create_patches(byte_sequence, method="adaptive")
    
    def encode_patches(
        self,
        patches: List[bytes],
        patch_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Encode patches into continuous representations.
        
        Args:
            patches: List of byte patches
            patch_mask: Optional mask for valid patches
            
        Returns:
            Encoded patch representations
        """
        if not patches:
            return torch.zeros(0, self.hidden_dim, device=self.device)
        
        # Convert patches to tensors
        max_length = max(len(patch) for patch in patches)
        num_patches = len(patches)
        
        # Create batch tensor
        batch_tensor = torch.zeros(
            1, num_patches, max_length, dtype=torch.long, device=self.device
        )
        
        # Create attention mask
        attention_mask = torch.zeros(
            1, num_patches, max_length, dtype=torch.bool, device=self.device
        )
        
        # Fill tensors
        for i, patch in enumerate(patches):
            patch_ids = list(patch)
            batch_tensor[0, i, :len(patch_ids)] = torch.tensor(patch_ids)
            attention_mask[0, i, :len(patch_ids)] = True
        
        # Get byte embeddings
        byte_embeds = self.byte_embeddings(batch_tensor)  # (1, patches, length, embed)
        
        # Add local positional embeddings
        positions = torch.arange(max_length, device=self.device)
        local_pos_embeds = self.local_position_embeddings(positions)
        byte_embeds = byte_embeds + local_pos_embeds.unsqueeze(0).unsqueeze(0)
        
        # Apply attention mask
        byte_embeds = byte_embeds * attention_mask.unsqueeze(-1).float()
        
        # Aggregate within patches using attention
        # Reshape for attention: (patches, length, embed)
        byte_embeds_reshaped = byte_embeds.squeeze(0)
        attention_mask_reshaped = attention_mask.squeeze(0)
        
        # Self-attention within each patch
        aggregated = []
        for i in range(num_patches):
            patch_embeds = byte_embeds_reshaped[i:i+1]  # (1, length, embed)
            patch_mask = attention_mask_reshaped[i]      # (length,)
            
            if patch_mask.any():
                # Apply self-attention
                attn_output, _ = self.patch_attention(
                    patch_embeds, patch_embeds, patch_embeds,
                    key_padding_mask=~patch_mask.unsqueeze(0)
                )
                # Mean pooling over valid positions
                valid_length = patch_mask.float().sum()
                patch_repr = attn_output[0, patch_mask].mean(dim=0)
            else:
                patch_repr = torch.zeros(self.embedding_dim, device=self.device)
            
            aggregated.append(patch_repr)
        
        # Stack patch representations
        patch_representations = torch.stack(aggregated, dim=0).unsqueeze(0)  # (1, patches, embed)
        
        # Project to hidden dimension
        patch_hidden = self.input_projection(patch_representations)  # (1, patches, hidden)
        
        return patch_hidden.squeeze(0)
    
    def forward(
        self,
        input_bytes: bytes,
        return_details: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Any]]]:
        """Forward pass of the local encoder.
        
        Args:
            input_bytes: Input byte sequence
            return_details: Whether to return additional details
            
        Returns:
            Encoded representation or tuple of (encoding, details)
        """
        # Create patches
        patch_boundaries = self.create_patches(input_bytes)
        patches = [
            input_bytes[start:end] 
            for start, end in patch_boundaries
        ]
        
        # Encode patches
        patch_encodings = self.encode_patches(patches)  # (patches, hidden)
        
        # Add patch-level positional embeddings
        num_patches = patch_encodings.shape[0]
        positions = torch.arange(num_patches, device=self.device)
        patch_pos_embeds = self.patch_position_embeddings(positions)
        
        # Combine with patch encodings
        patch_encodings = patch_encodings + patch_pos_embeds
        
        # Add batch dimension for transformer
        patch_encodings = patch_encodings.unsqueeze(0)  # (1, patches, hidden)
        
        # Apply transformer encoder
        encoded = self.transformer(patch_encodings)  # (1, patches, hidden)
        
        # Final layer norm
        encoded = self.layer_norm(encoded)
        
        # Remove batch dimension
        encoded = encoded.squeeze(0)  # (patches, hidden)
        
        if return_details:
            details = {
                "num_patches": num_patches,
                "patch_boundaries": patch_boundaries,
                "patch_sizes": [end - start for start, end in patch_boundaries],
                "average_patch_size": sum(end - start for start, end in patch_boundaries) / len(patch_boundaries),
                "entropy": self.compute_entropy(input_bytes)
            }
            return encoded, details
        else:
            return encoded
    
    def batch_forward(
        self,
        input_texts: List[Union[str, bytes]],
        padding: bool = True,
        max_length: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        """Process a batch of inputs.
        
        Args:
            input_texts: List of input texts or bytes
            padding: Whether to pad sequences to same length
            max_length: Maximum sequence length (in patches)
            
        Returns:
            Tuple of (encoded_batch, attention_mask)
        """
        batch_encodings = []
        batch_masks = []
        
        # Process each input
        for text in input_texts:
            # Convert to bytes if needed
            if isinstance(text, str):
                input_bytes = text.encode('utf-8')
            else:
                input_bytes = text
            
            # Encode
            encoded = self.forward(input_bytes)  # (patches, hidden)
            batch_encodings.append(encoded)
            
            # Create mask
            mask = torch.ones(encoded.shape[0], dtype=torch.bool, device=self.device)
            batch_masks.append(mask)
        
        # Pad if requested
        if padding:
            # Find max length
            if max_length is None:
                max_length = max(enc.shape[0] for enc in batch_encodings)
            
            # Pad sequences
            padded_encodings = []
            padded_masks = []
            
            for enc, mask in zip(batch_encodings, batch_masks):
                pad_length = max_length - enc.shape[0]
                if pad_length > 0:
                    # Pad encoding with zeros
                    padding_tensor = torch.zeros(
                        pad_length, self.hidden_dim, 
                        device=self.device
                    )
                    enc = torch.cat([enc, padding_tensor], dim=0)
                    
                    # Pad mask with False
                    mask_padding = torch.zeros(
                        pad_length, dtype=torch.bool, 
                        device=self.device
                    )
                    mask = torch.cat([mask, mask_padding], dim=0)
                elif pad_length < 0:
                    # Truncate
                    enc = enc[:max_length]
                    mask = mask[:max_length]
                
                padded_encodings.append(enc)
                padded_masks.append(mask)
            
            # Stack into batch
            batch_tensor = torch.stack(padded_encodings, dim=0)
            mask_tensor = torch.stack(padded_masks, dim=0)
        else:
            # Return as list
            batch_tensor = batch_encodings
            mask_tensor = batch_masks
        
        return batch_tensor, mask_tensor