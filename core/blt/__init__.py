"""BLT (Byte Latent Transformer) module.

Complete implementation of the Byte Latent Transformer architecture
for direct byte-level language modeling without tokenization.
"""

from .entropy import (
    calculate_byte_entropy,
    adaptive_entropy_patching,
    EntropyCalculator
)

from .patching import (
    ByteProcessor,
    PatchEmbedder,
    DynamicPatcher,
    BLTInputProcessor
)

from .encoder import LocalEncoder

from .transformer import (
    LatentTransformer,
    MultiHeadAttention,
    FeedForward,
    TransformerBlock,
    RotaryPositionalEncoding,
    count_parameters,
    get_model_size_mb,
    analyze_model_architecture
)

from .decoder import (
    LocalDecoder,
    PatchDecoder,
    AdaptiveDecoder
)

from .pipeline import (
    BLTPipeline,
    create_blt_model
)

from .m4_pro_optimization import (
    M4ProOptimizedBLT,
    M4ProOptimizationConfig,
    create_m4_pro_optimized_blt
)

__all__ = [
    # Entropy and patching
    'calculate_byte_entropy',
    'adaptive_entropy_patching',
    'EntropyCalculator',
    'ByteProcessor',
    'PatchEmbedder',
    'DynamicPatcher',
    'BLTInputProcessor',
    
    # Core components
    'LocalEncoder',
    'LatentTransformer',
    'LocalDecoder',
    'PatchDecoder',
    'AdaptiveDecoder',
    
    # Transformer components
    'MultiHeadAttention',
    'FeedForward',
    'TransformerBlock',
    'RotaryPositionalEncoding',
    
    # Pipeline and utilities
    'BLTPipeline',
    'create_blt_model',
    'count_parameters',
    'get_model_size_mb',
    'analyze_model_architecture',
    
    # M4 Pro optimizations
    'M4ProOptimizedBLT',
    'M4ProOptimizationConfig',
    'create_m4_pro_optimized_blt'
]