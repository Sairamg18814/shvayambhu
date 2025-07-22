"""Local Decoder for BLT (Byte Latent Transformer).

This module implements the local decoder that converts patch embeddings
back to byte sequences using entropy-aware reconstruction.
"""

from typing import Optional, Tuple, List, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from .patching import DynamicPatcher, ByteProcessor
from .entropy import calculate_byte_entropy


class PatchDecoder(nn.Module):
    """Decodes individual patches back to byte sequences."""
    
    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int = 256,
        max_patch_size: int = 32,
        dropout: float = 0.1,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_patch_size = max_patch_size
        self.device = device or torch.device("cpu")
        
        # Patch size prediction head
        self.size_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, max_patch_size + 1),  # +1 for EOS token
            nn.Softmax(dim=-1)
        )
        
        # Byte sequence decoder with attention
        self.byte_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=3
        )
        
        # Byte output projection
        self.byte_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Byte embeddings for decoder input
        self.byte_embeddings = nn.Embedding(vocab_size + 1, hidden_dim)  # +1 for start token
        
        # Positional encoding for bytes within patches
        self.pos_encoding = nn.Embedding(max_patch_size, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Special tokens
        self.start_token = vocab_size  # Start of patch token
        self.eos_token_logit_index = max_patch_size  # End of sequence in size prediction
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate scales."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(
        self, 
        patch_embedding: Tensor,
        target_bytes: Optional[Tensor] = None,
        max_length: Optional[int] = None
    ) -> Dict[str, Tensor]:
        """Forward pass of patch decoder.
        
        Args:
            patch_embedding: Single patch embedding (hidden_dim,)
            target_bytes: Target byte sequence for training (seq_len,)
            max_length: Maximum sequence length for generation
            
        Returns:
            Dictionary containing:
            - predicted_size: Predicted patch size logits
            - byte_logits: Byte prediction logits
            - predicted_bytes: Predicted byte sequence (if generating)
        """
        batch_size = 1  # Single patch
        
        # Predict patch size
        size_logits = self.size_predictor(patch_embedding.unsqueeze(0))  # (1, max_patch_size + 1)
        predicted_size_dist = size_logits.squeeze(0)  # (max_patch_size + 1,)
        
        if target_bytes is not None:
            # Training mode: teacher forcing
            return self._forward_training(patch_embedding, target_bytes, predicted_size_dist)
        else:
            # Generation mode: autoregressive decoding
            max_len = max_length or self.max_patch_size
            return self._forward_generation(patch_embedding, predicted_size_dist, max_len)
    
    def _forward_training(
        self, 
        patch_embedding: Tensor, 
        target_bytes: Tensor,
        size_logits: Tensor
    ) -> Dict[str, Tensor]:
        """Training forward pass with teacher forcing."""
        seq_len = target_bytes.shape[0]
        
        # Create decoder input by shifting target right and adding start token
        decoder_input = torch.cat([
            torch.tensor([self.start_token], device=self.device),
            target_bytes[:-1]
        ])
        
        # Embed decoder input
        byte_embeds = self.byte_embeddings(decoder_input)  # (seq_len, hidden_dim)
        
        # Add positional encoding
        positions = torch.arange(seq_len, device=self.device)
        pos_embeds = self.pos_encoding(positions)
        decoder_input_embeds = byte_embeds + pos_embeds
        decoder_input_embeds = self.dropout(decoder_input_embeds)
        
        # Prepare memory (patch embedding) for decoder
        memory = patch_embedding.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim)
        
        # Decode byte sequence
        decoder_output = self.byte_decoder(
            decoder_input_embeds.unsqueeze(0),  # (1, seq_len, hidden_dim)
            memory  # (1, 1, hidden_dim)
        )
        
        # Project to byte logits
        byte_logits = self.byte_projection(decoder_output.squeeze(0))  # (seq_len, vocab_size)
        
        return {
            "predicted_size": size_logits,
            "byte_logits": byte_logits,
            "target_bytes": target_bytes
        }
    
    def _forward_generation(
        self, 
        patch_embedding: Tensor, 
        size_logits: Tensor,
        max_length: int
    ) -> Dict[str, Tensor]:
        """Generation forward pass with autoregressive decoding."""
        # Sample predicted size
        size_dist = torch.distributions.Categorical(size_logits[:-1])  # Exclude EOS
        predicted_size = size_dist.sample().item() + 1
        target_length = min(predicted_size, max_length)
        
        # Initialize generation
        generated_bytes = []
        current_input = torch.tensor([self.start_token], device=self.device)
        
        # Prepare memory
        memory = patch_embedding.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim)
        
        # Generate bytes autoregressively
        for step in range(target_length):
            # Embed current input
            byte_embed = self.byte_embeddings(current_input)  # (1, hidden_dim)
            pos_embed = self.pos_encoding(torch.tensor([step], device=self.device))
            decoder_input_embed = (byte_embed + pos_embed).unsqueeze(0)  # (1, 1, hidden_dim)
            
            # Decode next byte
            decoder_output = self.byte_decoder(decoder_input_embed, memory)
            byte_logits = self.byte_projection(decoder_output.squeeze(0))  # (1, vocab_size)
            
            # Sample next byte
            byte_probs = F.softmax(byte_logits, dim=-1)
            next_byte = torch.multinomial(byte_probs, 1).item()
            
            generated_bytes.append(next_byte)
            current_input = torch.tensor([next_byte], device=self.device)
        
        predicted_bytes = torch.tensor(generated_bytes, device=self.device)
        
        return {
            "predicted_size": size_logits,
            "predicted_bytes": predicted_bytes,
            "byte_logits": None  # Not available in generation mode
        }


class LocalDecoder(nn.Module):
    """Local decoder that converts patch embeddings back to byte sequences.
    
    This is the final component of the BLT architecture, responsible for
    reconstructing the original byte sequence from processed patch embeddings.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int = 256,
        max_patch_size: int = 32,
        min_patch_size: int = 4,
        dropout: float = 0.1,
        use_entropy_loss: bool = True,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_patch_size = max_patch_size
        self.min_patch_size = min_patch_size
        self.use_entropy_loss = use_entropy_loss
        self.device = device or torch.device("cpu")
        
        # Patch decoder for individual patches
        self.patch_decoder = PatchDecoder(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            max_patch_size=max_patch_size,
            dropout=dropout,
            device=self.device
        )
        
        # Global sequence reconstruction head
        self.sequence_reconstruction = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Final sequence projection
        self.final_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Entropy regularization layer
        if use_entropy_loss:
            self.entropy_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid()
            )
        
        # Byte processor for validation
        self.byte_processor = ByteProcessor()
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        patch_embeddings: Tensor,
        patch_boundaries: Optional[List[Tuple[int, int]]] = None,
        target_sequence: Optional[bytes] = None,
        mode: str = "generate"
    ) -> Dict[str, Any]:
        """Forward pass of local decoder.
        
        Args:
            patch_embeddings: Patch embeddings (num_patches, hidden_dim)
            patch_boundaries: Original patch boundaries for reconstruction
            target_sequence: Target byte sequence for training
            mode: "train" or "generate"
            
        Returns:
            Dictionary containing decoded results
        """
        num_patches = patch_embeddings.shape[0]
        
        if mode == "train" and target_sequence is not None:
            return self._forward_training(
                patch_embeddings, patch_boundaries, target_sequence
            )
        else:
            return self._forward_generation(patch_embeddings)
    
    def _forward_training(
        self,
        patch_embeddings: Tensor,
        patch_boundaries: List[Tuple[int, int]],
        target_sequence: bytes
    ) -> Dict[str, Any]:
        """Training forward pass with target sequence."""
        num_patches = patch_embeddings.shape[0]
        target_bytes = torch.tensor(list(target_sequence), device=self.device)
        
        # Decode individual patches
        patch_losses = []
        patch_predictions = []
        entropy_losses = []
        
        for i, (start, end) in enumerate(patch_boundaries):
            patch_embedding = patch_embeddings[i]
            target_patch = target_bytes[start:end]
            
            # Decode patch
            patch_result = self.patch_decoder(
                patch_embedding, target_bytes=target_patch
            )
            
            # Calculate patch loss
            if patch_result["byte_logits"] is not None:
                patch_loss = F.cross_entropy(
                    patch_result["byte_logits"], 
                    target_patch
                )
                patch_losses.append(patch_loss)
            
            patch_predictions.append(patch_result)
            
            # Entropy regularization
            if self.use_entropy_loss:
                predicted_entropy = self.entropy_predictor(patch_embedding)
                actual_entropy = calculate_byte_entropy(target_patch.cpu().numpy().tobytes())
                entropy_loss = F.mse_loss(
                    predicted_entropy.squeeze(), 
                    torch.tensor(actual_entropy, device=self.device)
                )
                entropy_losses.append(entropy_loss)
        
        # Global sequence reconstruction
        sequence_output, _ = self.sequence_reconstruction(
            patch_embeddings.unsqueeze(0)
        )
        sequence_logits = self.final_projection(sequence_output.squeeze(0))
        
        # Calculate sequence loss
        sequence_loss = F.cross_entropy(sequence_logits, target_bytes)
        
        # Combine losses
        total_patch_loss = torch.stack(patch_losses).mean() if patch_losses else 0
        total_entropy_loss = torch.stack(entropy_losses).mean() if entropy_losses else 0
        
        total_loss = sequence_loss + 0.5 * total_patch_loss + 0.1 * total_entropy_loss
        
        return {
            "loss": total_loss,
            "sequence_loss": sequence_loss,
            "patch_loss": total_patch_loss,
            "entropy_loss": total_entropy_loss,
            "patch_predictions": patch_predictions,
            "sequence_logits": sequence_logits
        }
    
    def _forward_generation(self, patch_embeddings: Tensor) -> Dict[str, Any]:
        """Generation forward pass."""
        num_patches = patch_embeddings.shape[0]
        
        # Decode individual patches
        decoded_patches = []
        predicted_entropies = []
        
        for i in range(num_patches):
            patch_embedding = patch_embeddings[i]
            
            # Decode patch
            patch_result = self.patch_decoder(patch_embedding)
            
            if "predicted_bytes" in patch_result:
                decoded_bytes = patch_result["predicted_bytes"].cpu().numpy()
                decoded_patches.append(decoded_bytes)
            
            # Predict entropy
            if self.use_entropy_loss:
                predicted_entropy = self.entropy_predictor(patch_embedding)
                predicted_entropies.append(predicted_entropy.item())
        
        # Reconstruct full sequence
        if decoded_patches:
            reconstructed_sequence = np.concatenate(decoded_patches)
            reconstructed_bytes = bytes(reconstructed_sequence.astype(np.uint8))
        else:
            reconstructed_bytes = b""
        
        # Global sequence reconstruction for refinement
        sequence_output, _ = self.sequence_reconstruction(
            patch_embeddings.unsqueeze(0)
        )
        sequence_logits = self.final_projection(sequence_output.squeeze(0))
        
        # Sample from sequence logits for alternative reconstruction
        sequence_probs = F.softmax(sequence_logits, dim=-1)
        alternative_bytes = torch.multinomial(sequence_probs, 1).squeeze(-1)
        alternative_sequence = bytes(alternative_bytes.cpu().numpy())
        
        return {
            "reconstructed_bytes": reconstructed_bytes,
            "alternative_bytes": alternative_sequence,
            "decoded_patches": decoded_patches,
            "predicted_entropies": predicted_entropies,
            "sequence_logits": sequence_logits
        }


class AdaptiveDecoder(nn.Module):
    """Adaptive decoder that adjusts strategy based on content type."""
    
    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int = 256,
        max_patch_size: int = 32,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.device = device or torch.device("cpu")
        
        # Main decoder
        self.local_decoder = LocalDecoder(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            max_patch_size=max_patch_size,
            device=self.device
        )
        
        # Content type classifier
        self.content_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 4),  # text, code, binary, mixed
            nn.Softmax(dim=-1)
        )
        
        # Strategy selector based on content type
        self.strategy_weights = nn.Parameter(torch.ones(4, 3))  # 4 content types, 3 strategies
        
    def forward(
        self,
        patch_embeddings: Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        """Adaptive forward pass."""
        # Classify content type
        content_logits = self.content_classifier(patch_embeddings.mean(dim=0))
        content_type = torch.argmax(content_logits).item()
        
        # Select strategy weights
        strategy_weights = F.softmax(self.strategy_weights[content_type], dim=0)
        
        # Decode with multiple strategies
        standard_result = self.local_decoder(patch_embeddings, **kwargs)
        
        # Could implement different strategies here based on content type
        # For now, return standard result with content type info
        standard_result["content_type"] = content_type
        standard_result["content_confidence"] = content_logits
        standard_result["strategy_weights"] = strategy_weights
        
        return standard_result