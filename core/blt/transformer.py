"""Latent Transformer Core for BLT (Byte Latent Transformer).

This module implements the core transformer architecture that processes
patch embeddings in the latent space, optimized for Apple Silicon.
"""

from typing import Optional, Tuple, Dict, Any, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from core.research.memory_aug import MemoryAugmentedNetwork
from utils.hardware.memory_manager import get_memory_manager


class RotaryPositionalEncoding(nn.Module):
    """Rotary Positional Encoding (RoPE) for improved positional awareness."""
    
    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Pre-compute frequency inverse
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Pre-compute rotation matrices for efficiency
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cached cos/sin values for given sequence length."""
        if (self._seq_len_cached is None or 
            seq_len > self._seq_len_cached or 
            self._cos_cached.device != device or 
            self._cos_cached.dtype != dtype):
            
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)
    
    def forward(self, x: Tensor, seq_len: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """Apply rotary positional encoding.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            seq_len: Sequence length (defaults to x.shape[1])
            
        Returns:
            Tuple of (cos, sin) tensors for applying rotation
        """
        if seq_len is None:
            seq_len = x.shape[1]
        
        self._update_cos_sin_cache(seq_len, x.device, x.dtype)
        
        return (
            self._cos_cached[:seq_len, :],
            self._sin_cached[:seq_len, :]
        )


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    """Apply rotary positional embedding to query and key tensors."""
    def rotate_half(x):
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE and optional memory augmentation."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_rope: bool = True,
        use_memory: bool = False,
        memory_size: int = 512,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope
        self.use_memory = use_memory
        self.device = device or torch.device("cpu")
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, device=self.device)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, device=self.device)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False, device=self.device)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, device=self.device)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Rotary positional encoding
        if self.use_rope:
            self.rope = RotaryPositionalEncoding(self.head_dim)
        
        # Memory augmentation
        if self.use_memory:
            self.memory_network = MemoryAugmentedNetwork(
                hidden_dim=hidden_dim,
                memory_size=memory_size,
                num_heads=num_heads
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        x: Tensor, 
        attention_mask: Optional[Tensor] = None,
        key_value_states: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """Forward pass of multi-head attention.
        
        Args:
            x: Input tensor (batch_size, seq_len, hidden_dim)
            attention_mask: Mask to avoid attention on padding tokens
            key_value_states: External key/value states for cross-attention
            use_cache: Whether to return key/value states for caching
            past_key_value: Cached key/value states from previous steps
            
        Returns:
            Tuple of (output, cached_key_value)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        q = self.q_proj(x)
        
        if key_value_states is not None:
            # Cross-attention
            k = self.k_proj(key_value_states[0])
            v = self.v_proj(key_value_states[1])
        else:
            # Self-attention
            k = self.k_proj(x)
            v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, k.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, v.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        
        # Handle cached key/value states
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        
        # Apply rotary positional encoding
        if self.use_rope:
            cos, sin = self.rope(x, k.shape[-2])
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back to original dimensions
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )
        
        # Final linear projection
        output = self.out_proj(attn_output)
        
        # Memory augmentation
        if self.use_memory:
            output = self.memory_network(output, attn_weights)
        
        # Return cached states if requested
        cached_kv = (k, v) if use_cache else None
        
        return output, cached_kv


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation."""
    
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "swiglu",
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim or int(hidden_dim * 8 / 3)  # SwiGLU ratio
        self.activation = activation
        self.device = device or torch.device("cpu")
        
        if activation == "swiglu":
            # SwiGLU requires two linear transformations
            self.gate_proj = nn.Linear(hidden_dim, self.intermediate_dim, bias=False, device=self.device)
            self.up_proj = nn.Linear(hidden_dim, self.intermediate_dim, bias=False, device=self.device)
            self.down_proj = nn.Linear(self.intermediate_dim, hidden_dim, bias=False, device=self.device)
        else:
            # Standard FFN
            self.linear1 = nn.Linear(hidden_dim, self.intermediate_dim, device=self.device)
            self.linear2 = nn.Linear(self.intermediate_dim, hidden_dim, device=self.device)
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        if self.activation == "swiglu":
            for module in [self.gate_proj, self.up_proj, self.down_proj]:
                nn.init.xavier_uniform_(module.weight)
        else:
            for module in [self.linear1, self.linear2]:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of feed-forward network."""
        if self.activation == "swiglu":
            # SwiGLU: Swish(x @ W1) * (x @ W2) @ W3
            gate = F.silu(self.gate_proj(x))  # Swish/SiLU activation
            up = self.up_proj(x)
            intermediate = gate * up
            output = self.down_proj(intermediate)
        else:
            # Standard FFN with GELU
            intermediate = F.gelu(self.linear1(x))
            intermediate = self.dropout(intermediate)
            output = self.linear2(intermediate)
        
        return self.dropout(output)


class TransformerBlock(nn.Module):
    """Single transformer block with pre-layer normalization."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        intermediate_dim: Optional[int] = None,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        use_rope: bool = True,
        use_memory: bool = False,
        memory_size: int = 512,
        activation: str = "swiglu",
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.device = device or torch.device("cpu")
        
        # Layer normalization (pre-norm architecture)
        self.attention_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps, device=self.device)
        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps, device=self.device)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope,
            use_memory=use_memory,
            memory_size=memory_size,
            device=self.device
        )
        
        # Feed-forward network
        self.feed_forward = FeedForward(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            dropout=dropout,
            activation=activation,
            device=self.device
        )
    
    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """Forward pass of transformer block.
        
        Args:
            x: Input tensor (batch_size, seq_len, hidden_dim)
            attention_mask: Attention mask
            use_cache: Whether to cache key/value states
            past_key_value: Cached key/value from previous step
            
        Returns:
            Tuple of (output, cached_key_value)
        """
        # Self-attention with residual connection (pre-norm)
        normed_x = self.attention_norm(x)
        attn_output, cached_kv = self.attention(
            normed_x,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_value=past_key_value
        )
        x = x + attn_output
        
        # Feed-forward with residual connection (pre-norm)
        normed_x = self.ffn_norm(x)
        ffn_output = self.feed_forward(normed_x)
        x = x + ffn_output
        
        return x, cached_kv


class LatentTransformer(nn.Module):
    """Core latent transformer for processing patch embeddings.
    
    This is the heart of the BLT architecture, operating on dynamic patches
    in the latent space with Apple Silicon optimizations.
    """
    
    def __init__(
        self,
        hidden_dim: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        intermediate_dim: Optional[int] = None,
        max_position_embeddings: int = 4096,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        use_rope: bool = True,
        use_memory: bool = True,
        memory_size: int = 1024,
        activation: str = "swiglu",
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        self.device = device or self._get_optimal_device()
        
        # Embedding dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
                use_rope=use_rope,
                use_memory=use_memory and (i % 4 == 0),  # Memory every 4th layer
                memory_size=memory_size,
                activation=activation,
                device=self.device
            )
            for i in range(num_layers)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps, device=self.device)
        
        # Memory manager for optimization
        self.memory_manager = get_memory_manager()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _get_optimal_device(self) -> torch.device:
        """Get optimal device for Apple Silicon."""
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _init_weights(self, module):
        """Initialize weights for all modules."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        patch_embeddings: Tensor,
        attention_mask: Optional[Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
        return_dict: bool = True
    ) -> Dict[str, Any]:
        """Forward pass of the latent transformer.
        
        Args:
            patch_embeddings: Input patch embeddings (batch_size, num_patches, hidden_dim)
            attention_mask: Mask for attention computation
            use_cache: Whether to use KV caching for generation
            past_key_values: Cached key/value states from previous steps
            return_dict: Whether to return a dictionary or tuple
            
        Returns:
            Dictionary containing:
            - last_hidden_state: Final hidden states
            - hidden_states: All layer hidden states (if requested)
            - attentions: Attention weights (if requested)
            - past_key_values: Cached states (if use_cache=True)
        """
        batch_size, seq_len, _ = patch_embeddings.shape
        
        # Apply dropout to embeddings
        hidden_states = self.dropout(patch_embeddings)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, seq_len, 
                dtype=torch.bool, 
                device=hidden_states.device
            )
        
        # Initialize cache storage
        if use_cache:
            if past_key_values is None:
                past_key_values = [None] * self.num_layers
            new_key_values = []
        
        # Pass through transformer layers
        all_hidden_states = []
        for i, layer in enumerate(self.layers):
            all_hidden_states.append(hidden_states)
            
            past_key_value = past_key_values[i] if past_key_values else None
            
            # Apply transformer block
            hidden_states, cached_kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_value=past_key_value
            )
            
            if use_cache:
                new_key_values.append(cached_kv)
        
        # Final layer normalization
        hidden_states = self.final_norm(hidden_states)
        all_hidden_states.append(hidden_states)
        
        if return_dict:
            result = {
                "last_hidden_state": hidden_states,
                "hidden_states": all_hidden_states,
            }
            
            if use_cache:
                result["past_key_values"] = new_key_values
            
            return result
        else:
            outputs = (hidden_states,)
            if use_cache:
                outputs += (new_key_values,)
            return outputs
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        for layer in self.layers:
            layer.forward = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer)
            )
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        stats = self.memory_manager.get_memory_stats()
        return {
            "total_memory_gb": stats.total_gb,
            "used_memory_gb": stats.used_gb,
            "available_memory_gb": stats.available_gb,
            "utilization_percent": stats.utilization_percent
        }
    
    def optimize_for_inference(self):
        """Optimize model for inference on Apple Silicon."""
        # Set to evaluation mode
        self.eval()
        
        # Compile model for faster execution (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                self = torch.compile(self, mode="reduce-overhead")
            except Exception:
                pass  # Fallback if compilation fails
        
        # Move to optimal device
        self.to(self.device)
        
        # Enable memory efficient attention if available
        if hasattr(F, 'scaled_dot_product_attention'):
            for layer in self.layers:
                if hasattr(layer.attention, 'use_flash_attention'):
                    layer.attention.use_flash_attention = True


# Utility functions for model inspection and debugging
def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Get the memory size of a model in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


def analyze_model_architecture(model: LatentTransformer) -> Dict[str, Any]:
    """Analyze and return model architecture statistics."""
    total_params = count_parameters(model)
    model_size_mb = get_model_size_mb(model)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": total_params,
        "model_size_mb": model_size_mb,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "num_heads": model.num_heads,
        "parameters_per_layer": total_params // model.num_layers,
        "memory_usage": model.get_memory_usage()
    }