"""Complete BLT (Byte Latent Transformer) Pipeline.

This module integrates the encoder, transformer, and decoder components
into a complete end-to-end pipeline for byte-level language modeling.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
import logging

from .encoder import LocalEncoder
from .transformer import LatentTransformer
from .decoder import LocalDecoder, AdaptiveDecoder
from .patching import BLTInputProcessor
from .entropy import calculate_byte_entropy
from utils.hardware.memory_manager import get_memory_manager
from utils.evaluation.metrics import compute_language_modeling_metrics

logger = logging.getLogger(__name__)


class BLTPipeline(nn.Module):
    """Complete BLT pipeline for byte-level language modeling.
    
    This class integrates all BLT components into a unified model that can
    process raw byte sequences without tokenization.
    """
    
    def __init__(
        self,
        # Encoder parameters
        embedding_dim: int = 512,
        encoder_layers: int = 6,
        encoder_heads: int = 8,
        
        # Transformer parameters
        hidden_dim: int = 1024,
        transformer_layers: int = 24,
        transformer_heads: int = 16,
        
        # Decoder parameters
        decoder_layers: int = 6,
        
        # General parameters
        vocab_size: int = 256,
        max_seq_length: int = 4096,
        patch_size: int = 16,
        min_patch: int = 4,
        max_patch: int = 32,
        dropout: float = 0.1,
        
        # Advanced features
        use_rope: bool = True,
        use_memory: bool = True,
        use_adaptive_decoder: bool = True,
        use_entropy_loss: bool = True,
        
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.device = device or self._get_optimal_device()
        
        # Input processor
        self.input_processor = BLTInputProcessor(
            min_patch_size=min_patch,
            max_patch_size=max_patch,
            embedding_dim=embedding_dim,
            device=self.device
        )
        
        # Local encoder (bytes → patches)
        self.encoder = LocalEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
            patch_size=patch_size,
            min_patch=min_patch,
            max_patch=max_patch,
            dropout=dropout,
            max_seq_length=max_seq_length,
            device=self.device
        )
        
        # Latent transformer (core processing)
        self.transformer = LatentTransformer(
            hidden_dim=hidden_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            max_position_embeddings=max_seq_length,
            dropout=dropout,
            use_rope=use_rope,
            use_memory=use_memory,
            device=self.device
        )
        
        # Local decoder (patches → bytes)
        if use_adaptive_decoder:
            self.decoder = AdaptiveDecoder(
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                max_patch_size=max_patch,
                device=self.device
            )
        else:
            self.decoder = LocalDecoder(
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                max_patch_size=max_patch,
                min_patch_size=min_patch,
                dropout=dropout,
                use_entropy_loss=use_entropy_loss,
                device=self.device
            )
        
        # Memory manager
        self.memory_manager = get_memory_manager()
        
        # Training state
        self.training_step = 0
        self.validation_metrics = {}
        
        # Move to device
        self.to(self.device)
        
    def _get_optimal_device(self) -> torch.device:
        """Get optimal device for Apple Silicon."""
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def forward(
        self,
        input_data: Union[str, bytes, Tensor],
        target_data: Optional[Union[str, bytes, Tensor]] = None,
        mode: str = "train",
        return_details: bool = False,
        use_cache: bool = False,
        past_key_values: Optional[List] = None
    ) -> Dict[str, Any]:
        """Forward pass of the BLT pipeline.
        
        Args:
            input_data: Input data (string, bytes, or tensor)
            target_data: Target data for training
            mode: "train", "eval", or "generate"
            return_details: Whether to return detailed intermediate results
            use_cache: Whether to use KV caching for generation
            past_key_values: Cached key/value states
            
        Returns:
            Dictionary containing model outputs and metrics
        """
        # Prepare input
        if isinstance(input_data, str):
            input_bytes = input_data.encode('utf-8')
        elif isinstance(input_data, bytes):
            input_bytes = input_data
        else:
            raise ValueError("Input must be string or bytes")
        
        results = {"mode": mode, "input_length": len(input_bytes)}
        
        try:
            # 1. Encoding phase: bytes → patch embeddings
            with self.memory_manager.memory_context():
                if return_details:
                    patch_embeddings, encoding_details = self.encoder(
                        input_bytes, return_details=True
                    )
                    results["encoding_details"] = encoding_details
                else:
                    patch_embeddings = self.encoder(input_bytes)
                
                results["num_patches"] = patch_embeddings.shape[0]
            
            # 2. Transformer phase: process patch embeddings
            transformer_output = self.transformer(
                patch_embeddings.unsqueeze(0),  # Add batch dimension
                use_cache=use_cache,
                past_key_values=past_key_values,
                return_dict=True
            )
            
            hidden_states = transformer_output["last_hidden_state"].squeeze(0)
            
            if use_cache:
                results["past_key_values"] = transformer_output["past_key_values"]
            
            # 3. Decoding phase: patch embeddings → bytes
            if mode == "train" and target_data is not None:
                # Training mode with targets
                if isinstance(target_data, str):
                    target_bytes = target_data.encode('utf-8')
                elif isinstance(target_data, bytes):
                    target_bytes = target_data
                else:
                    raise ValueError("Target must be string or bytes")
                
                # Get patch boundaries for training
                patch_boundaries = getattr(self.encoder, '_last_patch_boundaries', None)
                
                decoder_output = self.decoder(
                    hidden_states,
                    patch_boundaries=patch_boundaries,
                    target_sequence=target_bytes,
                    mode="train"
                )
                
                results.update(decoder_output)
                
                # Compute additional metrics
                if "reconstructed_bytes" in decoder_output:
                    metrics = self._compute_metrics(
                        target_bytes, decoder_output["reconstructed_bytes"]
                    )
                    results["metrics"] = metrics
                
            else:
                # Generation/evaluation mode
                decoder_output = self.decoder(hidden_states, mode="generate")
                results.update(decoder_output)
                
                # Compute metrics if target is available
                if target_data is not None:
                    if isinstance(target_data, str):
                        target_bytes = target_data.encode('utf-8')
                    else:
                        target_bytes = target_data
                    
                    if "reconstructed_bytes" in decoder_output:
                        metrics = self._compute_metrics(
                            target_bytes, decoder_output["reconstructed_bytes"]
                        )
                        results["metrics"] = metrics
            
            # Memory usage tracking
            results["memory_usage"] = self.memory_manager.get_memory_stats()._asdict()
            
            return results
            
        except Exception as e:
            logger.error(f"Error in BLT pipeline forward pass: {e}")
            # Return error information for debugging
            return {
                "error": str(e),
                "mode": mode,
                "input_length": len(input_bytes),
                "memory_usage": self.memory_manager.get_memory_stats()._asdict()
            }
    
    def generate(
        self,
        prompt: Union[str, bytes],
        max_length: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        use_cache: bool = True,
        return_details: bool = False
    ) -> Dict[str, Any]:
        """Generate text using the BLT model.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling
            repetition_penalty: Repetition penalty
            use_cache: Whether to use KV caching
            return_details: Whether to return generation details
            
        Returns:
            Dictionary containing generated text and metadata
        """
        self.eval()
        
        if isinstance(prompt, str):
            prompt_bytes = prompt.encode('utf-8')
        else:
            prompt_bytes = prompt
        
        generated_bytes = bytearray(prompt_bytes)
        past_key_values = None
        generation_details = []
        
        with torch.no_grad():
            for step in range(max_length):
                # Get current context (last portion for efficiency)
                context_bytes = bytes(generated_bytes[-1024:])  # Last 1024 bytes
                
                # Forward pass
                output = self.forward(
                    context_bytes,
                    mode="generate",
                    use_cache=use_cache,
                    past_key_values=past_key_values,
                    return_details=return_details
                )
                
                if "error" in output:
                    break
                
                # Update cache
                if use_cache and "past_key_values" in output:
                    past_key_values = output["past_key_values"]
                
                # Get next bytes from reconstruction
                if "reconstructed_bytes" in output:
                    reconstructed = output["reconstructed_bytes"]
                    
                    # Find new bytes (beyond prompt)
                    if len(reconstructed) > len(context_bytes):
                        new_bytes = reconstructed[len(context_bytes):]
                        
                        # Apply sampling if available
                        if len(new_bytes) > 0:
                            # Simple strategy: take first new byte
                            next_byte = new_bytes[0]
                            generated_bytes.append(next_byte)
                            
                            if return_details:
                                generation_details.append({
                                    "step": step,
                                    "byte": next_byte,
                                    "char": chr(next_byte) if 32 <= next_byte <= 126 else f"\\x{next_byte:02x}",
                                    "entropy": output.get("predicted_entropies", [0])[0] if "predicted_entropies" in output else 0
                                })
                    else:
                        # No new content, stop generation
                        break
                else:
                    break
                
                # Check for natural stopping points
                try:
                    current_text = generated_bytes.decode('utf-8', errors='ignore')
                    if current_text.endswith(('.', '!', '?', '\n\n')):
                        break
                except:
                    pass
        
        # Prepare results
        try:
            generated_text = generated_bytes.decode('utf-8')
            prompt_text = prompt_bytes.decode('utf-8')
            new_text = generated_text[len(prompt_text):]
        except UnicodeDecodeError:
            generated_text = generated_bytes.decode('utf-8', errors='replace')
            prompt_text = prompt_bytes.decode('utf-8', errors='replace')
            new_text = generated_text[len(prompt_text):]
        
        result = {
            "prompt": prompt_text,
            "generated_text": new_text,
            "full_text": generated_text,
            "generated_bytes": bytes(generated_bytes),
            "generation_length": len(new_text),
            "total_length": len(generated_text)
        }
        
        if return_details:
            result["generation_details"] = generation_details
        
        return result
    
    def compute_perplexity(
        self,
        text_data: Union[str, bytes, List[str], List[bytes]],
        batch_size: int = 1
    ) -> Dict[str, float]:
        """Compute perplexity on text data.
        
        Args:
            text_data: Text data for evaluation
            batch_size: Batch size for processing
            
        Returns:
            Dictionary containing perplexity metrics
        """
        self.eval()
        
        if isinstance(text_data, (str, bytes)):
            text_data = [text_data]
        
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(text_data), batch_size):
                batch = text_data[i:i+batch_size]
                
                batch_loss = 0.0
                batch_tokens = 0
                
                for text in batch:
                    if isinstance(text, str):
                        text_bytes = text.encode('utf-8')
                    else:
                        text_bytes = text
                    
                    # Split into input and target
                    split_point = len(text_bytes) // 2
                    input_bytes = text_bytes[:split_point]
                    target_bytes = text_bytes[split_point:]
                    
                    # Forward pass
                    output = self.forward(
                        input_bytes,
                        target_data=target_bytes,
                        mode="eval"
                    )
                    
                    if "loss" in output:
                        batch_loss += output["loss"].item()
                        batch_tokens += len(target_bytes)
                
                if batch_tokens > 0:
                    total_loss += batch_loss
                    total_tokens += batch_tokens
                    num_batches += 1
        
        if total_tokens > 0:
            avg_loss = total_loss / num_batches
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            
            return {
                "perplexity": perplexity,
                "average_loss": avg_loss,
                "total_tokens": total_tokens,
                "num_batches": num_batches
            }
        else:
            return {"perplexity": float('inf'), "average_loss": float('inf')}
    
    def _compute_metrics(
        self,
        original_bytes: bytes,
        reconstructed_bytes: bytes
    ) -> Dict[str, float]:
        """Compute reconstruction and quality metrics."""
        # Use decoder's validation method if available
        if hasattr(self.decoder, 'validate_reconstruction'):
            return self.decoder.validate_reconstruction(original_bytes, reconstructed_bytes)
        
        # Fallback basic metrics
        exact_match = float(original_bytes == reconstructed_bytes)
        min_len = min(len(original_bytes), len(reconstructed_bytes))
        
        if min_len > 0:
            char_accuracy = sum(
                a == b for a, b in zip(original_bytes[:min_len], reconstructed_bytes[:min_len])
            ) / max(len(original_bytes), len(reconstructed_bytes))
        else:
            char_accuracy = 0.0
        
        return {
            "exact_match": exact_match,
            "character_accuracy": char_accuracy,
            "length_ratio": min_len / max(len(original_bytes), len(reconstructed_bytes)) if max(len(original_bytes), len(reconstructed_bytes)) > 0 else 0.0
        }
    
    def optimize_for_inference(self):
        """Optimize model for inference performance."""
        # Set to evaluation mode
        self.eval()
        
        # Optimize individual components
        if hasattr(self.encoder, 'optimize_for_inference'):
            self.encoder.optimize_for_inference()
        
        if hasattr(self.transformer, 'optimize_for_inference'):
            self.transformer.optimize_for_inference()
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.transformer, 'gradient_checkpointing_enable'):
            self.transformer.gradient_checkpointing_enable()
        
        # Compile model if available (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                self = torch.compile(self, mode="reduce-overhead")
                logger.info("Model compiled for inference optimization")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        logger.info("BLT pipeline optimized for inference")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        from .transformer import count_parameters, get_model_size_mb
        
        total_params = count_parameters(self)
        encoder_params = count_parameters(self.encoder)
        transformer_params = count_parameters(self.transformer)
        decoder_params = count_parameters(self.decoder)
        
        return {
            "model_type": "BLT (Byte Latent Transformer)",
            "total_parameters": total_params,
            "model_size_mb": get_model_size_mb(self),
            "component_parameters": {
                "encoder": encoder_params,
                "transformer": transformer_params,
                "decoder": decoder_params
            },
            "architecture": {
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "vocab_size": self.vocab_size,
                "max_seq_length": self.max_seq_length
            },
            "device": str(self.device),
            "memory_usage": self.memory_manager.get_memory_stats()._asdict()
        }
    
    def save_checkpoint(self, path: str, include_optimizer: bool = False, **metadata):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_config": {
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "vocab_size": self.vocab_size,
                "max_seq_length": self.max_seq_length
            },
            "training_step": self.training_step,
            "validation_metrics": self.validation_metrics,
            **metadata
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str, strict: bool = True):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        self.training_step = checkpoint.get("training_step", 0)
        self.validation_metrics = checkpoint.get("validation_metrics", {})
        
        logger.info(f"Checkpoint loaded from {path}")
        return checkpoint.get("model_config", {})


def create_blt_model(
    model_size: str = "7b",
    device: Optional[torch.device] = None,
    **kwargs
) -> BLTPipeline:
    """Create a BLT model with predefined configurations.
    
    Args:
        model_size: Model size ("1b", "7b", "13b", "30b")
        device: Target device
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured BLT model
    """
    configs = {
        "1b": {
            "embedding_dim": 512,
            "hidden_dim": 1024,
            "transformer_layers": 12,
            "transformer_heads": 8
        },
        "7b": {
            "embedding_dim": 768,
            "hidden_dim": 2048,
            "transformer_layers": 24,
            "transformer_heads": 16
        },
        "13b": {
            "embedding_dim": 1024,
            "hidden_dim": 3072,
            "transformer_layers": 32,
            "transformer_heads": 24
        },
        "30b": {
            "embedding_dim": 1536,
            "hidden_dim": 4096,
            "transformer_layers": 48,
            "transformer_heads": 32
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Available: {list(configs.keys())}")
    
    config = configs[model_size]
    config.update(kwargs)
    
    return BLTPipeline(device=device, **config)