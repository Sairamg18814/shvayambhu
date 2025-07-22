"""Validation pipeline for model evaluation during training.

This module provides comprehensive validation utilities including
metrics computation, performance evaluation, and quality assessment.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import time
from pathlib import Path
import json
from tqdm import tqdm

from ...core.blt.pipeline import BLTPipeline
from ...core.blt.entropy import calculate_byte_entropy
from .data_abstractions import DataLoader, DataSample
from .statistics_tracker import StatisticsTracker


@dataclass
class ValidationConfig:
    """Configuration for validation."""
    eval_batch_size: int = 32
    max_eval_samples: int = 1000
    
    # Metrics to compute
    compute_perplexity: bool = True
    compute_accuracy: bool = True
    compute_generation_quality: bool = True
    compute_reconstruction: bool = True
    
    # Generation settings
    generation_max_length: int = 512
    generation_temperature: float = 0.8
    generation_top_k: int = 50
    generation_top_p: float = 0.9
    
    # Task-specific evaluation
    eval_tasks: List[str] = field(default_factory=lambda: [
        "next_byte_prediction",
        "masked_byte_modeling",
        "reconstruction",
        "generation_quality"
    ])
    
    # Output settings
    save_predictions: bool = False
    save_generations: bool = True
    output_dir: str = "validation_outputs/"


@dataclass
class ValidationMetrics:
    """Container for validation metrics."""
    perplexity: float = 0.0
    accuracy: Dict[str, float] = field(default_factory=dict)
    loss: float = 0.0
    loss_components: Dict[str, float] = field(default_factory=dict)
    
    # Generation metrics
    generation_quality: float = 0.0
    generation_diversity: float = 0.0
    generation_coherence: float = 0.0
    
    # Reconstruction metrics
    reconstruction_accuracy: float = 0.0
    reconstruction_edit_distance: float = 0.0
    
    # Performance metrics
    inference_latency_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Task-specific metrics
    task_metrics: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(dict))


class ValidationPipeline:
    """Comprehensive validation pipeline."""
    
    def __init__(
        self,
        model: BLTPipeline,
        config: ValidationConfig,
        data_loader: Optional[DataLoader] = None,
        statistics_tracker: Optional[StatisticsTracker] = None
    ):
        self.model = model
        self.config = config
        self.data_loader = data_loader
        self.statistics_tracker = statistics_tracker
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Task evaluators
        self.task_evaluators = self._setup_task_evaluators()
    
    def _setup_task_evaluators(self) -> Dict[str, Callable]:
        """Setup task-specific evaluators."""
        return {
            "next_byte_prediction": self.evaluate_next_byte_prediction,
            "masked_byte_modeling": self.evaluate_masked_byte_modeling,
            "reconstruction": self.evaluate_reconstruction,
            "generation_quality": self.evaluate_generation_quality
        }
    
    @torch.no_grad()
    def validate(
        self,
        data_loader: Optional[DataLoader] = None,
        num_samples: Optional[int] = None
    ) -> ValidationMetrics:
        """Run full validation pipeline."""
        if data_loader is None:
            data_loader = self.data_loader
        
        if data_loader is None:
            raise ValueError("No data loader provided for validation")
        
        # Set model to eval mode
        self.model.eval()
        
        # Initialize metrics
        metrics = ValidationMetrics()
        all_losses = []
        all_predictions = []
        all_targets = []
        
        # Determine number of samples
        num_samples = num_samples or self.config.max_eval_samples
        num_batches = min(
            num_samples // self.config.eval_batch_size,
            len(data_loader)
        )
        
        # Main validation loop
        start_time = time.time()
        total_tokens = 0
        
        for batch_idx, batch in enumerate(tqdm(data_loader, total=num_batches, desc="Validating")):
            if batch_idx >= num_batches:
                break
            
            # Process batch
            batch_metrics, predictions, targets = self._process_batch(batch)
            
            # Accumulate metrics
            all_losses.append(batch_metrics['loss'])
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            total_tokens += sum(len(t) for t in targets)
            
            # Update component losses
            for key, value in batch_metrics.get('loss_components', {}).items():
                if key not in metrics.loss_components:
                    metrics.loss_components[key] = []
                metrics.loss_components[key].append(value)
        
        # Calculate aggregate metrics
        metrics.loss = np.mean(all_losses)
        metrics.perplexity = np.exp(metrics.loss)
        
        # Average component losses
        for key, values in metrics.loss_components.items():
            metrics.loss_components[key] = np.mean(values)
        
        # Calculate performance metrics
        total_time = time.time() - start_time
        metrics.throughput_tokens_per_sec = total_tokens / total_time
        metrics.inference_latency_ms = (total_time / len(all_predictions)) * 1000
        
        # Run task-specific evaluations
        if self.config.eval_tasks:
            for task in self.config.eval_tasks:
                if task in self.task_evaluators:
                    task_metrics = self.task_evaluators[task](
                        all_predictions, all_targets
                    )
                    metrics.task_metrics[task] = task_metrics
        
        # Memory usage
        if torch.cuda.is_available():
            metrics.memory_usage_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            torch.cuda.reset_peak_memory_stats()
        
        # Log to statistics tracker
        if self.statistics_tracker:
            self._log_metrics(metrics)
        
        # Save outputs if requested
        if self.config.save_predictions:
            self._save_predictions(all_predictions, all_targets)
        
        # Set model back to train mode
        self.model.train()
        
        return metrics
    
    def _process_batch(
        self,
        batch: Dict[str, Any]
    ) -> Tuple[Dict[str, float], List[torch.Tensor], List[torch.Tensor]]:
        """Process a single batch."""
        # Extract input bytes
        if isinstance(batch, dict) and 'input_bytes' in batch:
            input_bytes = batch['input_bytes']
        elif isinstance(batch, list):
            # Convert list of samples to tensor
            input_bytes = torch.stack([
                torch.tensor(list(sample.content), dtype=torch.uint8)
                for sample in batch[:self.config.eval_batch_size]
            ])
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")
        
        # Move to device
        device = next(self.model.parameters()).device
        input_bytes = input_bytes.to(device)
        
        # Forward pass
        outputs = self.model(input_bytes, mode='training')
        
        # Calculate loss
        loss = outputs.get('loss', 0.0)
        loss_components = outputs.get('loss_components', {})
        
        # Get predictions
        logits = outputs.get('logits', outputs.get('patch_logits'))
        if logits is not None:
            predictions = torch.argmax(logits, dim=-1)
        else:
            predictions = torch.zeros_like(input_bytes)
        
        # Prepare metrics
        metrics = {
            'loss': loss.item() if isinstance(loss, torch.Tensor) else loss,
            'loss_components': {
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in loss_components.items()
            }
        }
        
        return metrics, [predictions], [input_bytes]
    
    def evaluate_next_byte_prediction(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Evaluate next byte prediction accuracy."""
        correct = 0
        total = 0
        
        for pred_batch, target_batch in zip(predictions, targets):
            # Shift targets for next byte prediction
            pred_next = pred_batch[:, :-1]
            target_next = target_batch[:, 1:]
            
            # Calculate accuracy
            correct += (pred_next == target_next).sum().item()
            total += target_next.numel()
        
        accuracy = correct / max(total, 1)
        
        # Calculate per-position accuracy
        position_accuracies = []
        max_len = max(t.shape[1] for t in targets)
        
        for pos in range(min(max_len - 1, 100)):  # First 100 positions
            pos_correct = 0
            pos_total = 0
            
            for pred_batch, target_batch in zip(predictions, targets):
                if pos < pred_batch.shape[1] - 1:
                    pos_correct += (pred_batch[:, pos] == target_batch[:, pos + 1]).sum().item()
                    pos_total += pred_batch.shape[0]
            
            if pos_total > 0:
                position_accuracies.append(pos_correct / pos_total)
        
        return {
            "accuracy": accuracy,
            "avg_position_accuracy": np.mean(position_accuracies) if position_accuracies else 0.0,
            "accuracy_first_10": np.mean(position_accuracies[:10]) if len(position_accuracies) > 10 else 0.0,
            "accuracy_last_10": np.mean(position_accuracies[-10:]) if len(position_accuracies) > 10 else 0.0
        }
    
    def evaluate_masked_byte_modeling(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Evaluate masked byte modeling accuracy."""
        # Simulate masking (15% of positions)
        mask_prob = 0.15
        correct = 0
        total = 0
        
        for pred_batch, target_batch in zip(predictions, targets):
            # Create random mask
            mask = torch.rand_like(target_batch, dtype=torch.float) < mask_prob
            
            # Only evaluate masked positions
            masked_preds = pred_batch[mask]
            masked_targets = target_batch[mask]
            
            correct += (masked_preds == masked_targets).sum().item()
            total += mask.sum().item()
        
        accuracy = correct / max(total, 1)
        
        return {
            "masked_accuracy": accuracy,
            "masked_positions": total
        }
    
    def evaluate_reconstruction(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Evaluate reconstruction quality."""
        total_accuracy = 0
        total_edit_distance = 0
        num_samples = 0
        
        for pred_batch, target_batch in zip(predictions, targets):
            batch_size = pred_batch.shape[0]
            
            for i in range(batch_size):
                pred_seq = pred_batch[i]
                target_seq = target_batch[i]
                
                # Exact match accuracy
                exact_match = (pred_seq == target_seq).all().item()
                total_accuracy += exact_match
                
                # Edit distance (normalized)
                edit_dist = self._compute_edit_distance(
                    pred_seq.cpu().numpy(),
                    target_seq.cpu().numpy()
                )
                normalized_dist = edit_dist / len(target_seq)
                total_edit_distance += normalized_dist
                
                num_samples += 1
        
        return {
            "reconstruction_accuracy": total_accuracy / max(num_samples, 1),
            "avg_edit_distance": total_edit_distance / max(num_samples, 1),
            "perfect_reconstructions": total_accuracy
        }
    
    def _compute_edit_distance(self, seq1: np.ndarray, seq2: np.ndarray) -> int:
        """Compute Levenshtein edit distance."""
        m, n = len(seq1), len(seq2)
        dp = np.zeros((m + 1, n + 1), dtype=int)
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # Deletion
                        dp[i][j-1],    # Insertion
                        dp[i-1][j-1]   # Substitution
                    )
        
        return dp[m][n]
    
    def evaluate_generation_quality(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Evaluate text generation quality."""
        # Generate samples
        num_generations = min(10, len(targets))
        generations = []
        
        for i in range(num_generations):
            # Use first few bytes as prompt
            prompt = targets[i][:, :50] if i < len(targets) else None
            
            if prompt is not None:
                generated = self._generate_sample(prompt)
                generations.append(generated)
        
        if not generations:
            return {
                "generation_quality": 0.0,
                "generation_diversity": 0.0,
                "generation_coherence": 0.0
            }
        
        # Calculate metrics
        quality_scores = []
        diversity_scores = []
        coherence_scores = []
        
        for gen in generations:
            # Quality: Based on entropy (should be similar to natural text)
            entropy = calculate_byte_entropy(gen.cpu().numpy())
            quality = 1.0 - abs(entropy - 4.5) / 4.5  # Natural text ~4.5 entropy
            quality_scores.append(max(0, quality))
            
            # Diversity: Unique n-grams
            ngrams = set()
            for n in [2, 3, 4]:
                for j in range(len(gen) - n + 1):
                    ngrams.add(tuple(gen[j:j+n].tolist()))
            diversity = len(ngrams) / max(len(gen), 1)
            diversity_scores.append(diversity)
            
            # Coherence: Repetition penalty
            repetitions = 0
            for j in range(len(gen) - 10):
                if (gen[j:j+5] == gen[j+5:j+10]).all():
                    repetitions += 1
            coherence = 1.0 - (repetitions / max(len(gen) - 10, 1))
            coherence_scores.append(coherence)
        
        # Save generations if requested
        if self.config.save_generations:
            self._save_generations(generations)
        
        return {
            "generation_quality": np.mean(quality_scores),
            "generation_diversity": np.mean(diversity_scores),
            "generation_coherence": np.mean(coherence_scores),
            "num_generations": len(generations)
        }
    
    def _generate_sample(
        self,
        prompt: torch.Tensor,
        max_length: Optional[int] = None
    ) -> torch.Tensor:
        """Generate a sample from prompt."""
        max_length = max_length or self.config.generation_max_length
        device = next(self.model.parameters()).device
        
        # Ensure prompt is on correct device
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)
        prompt = prompt.to(device)
        
        generated = prompt.clone()
        
        with torch.no_grad():
            for _ in range(max_length - prompt.shape[1]):
                # Get model predictions
                outputs = self.model(generated, mode='inference')
                logits = outputs.get('logits', outputs.get('patch_logits'))
                
                if logits is None:
                    break
                
                # Get next token logits
                next_logits = logits[:, -1, :] / self.config.generation_temperature
                
                # Apply top-k filtering
                if self.config.generation_top_k > 0:
                    top_k_values, top_k_indices = torch.topk(
                        next_logits, self.config.generation_top_k
                    )
                    next_logits = torch.full_like(next_logits, -float('inf'))
                    next_logits.scatter_(1, top_k_indices, top_k_values)
                
                # Apply top-p (nucleus) filtering
                if self.config.generation_top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > self.config.generation_top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for end of generation (could be based on special tokens)
                if next_token.item() == 0:  # Assuming 0 is end token
                    break
        
        return generated[0]  # Return single sequence
    
    def _log_metrics(self, metrics: ValidationMetrics):
        """Log metrics to statistics tracker."""
        if self.statistics_tracker:
            # Add validation metrics
            for metric_name, value in metrics.accuracy.items():
                self.statistics_tracker.add_validation_metric(
                    f"accuracy/{metric_name}", value
                )
            
            self.statistics_tracker.add_validation_metric("loss", metrics.loss)
            self.statistics_tracker.add_validation_metric("perplexity", metrics.perplexity)
            
            # Add task-specific metrics
            for task, task_metrics in metrics.task_metrics.items():
                for metric_name, value in task_metrics.items():
                    self.statistics_tracker.add_validation_metric(
                        f"{task}/{metric_name}", value
                    )
    
    def _save_predictions(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor]
    ):
        """Save predictions and targets."""
        output_file = self.output_dir / f"predictions_{time.time():.0f}.pt"
        torch.save({
            "predictions": predictions,
            "targets": targets,
            "timestamp": time.time()
        }, output_file)
    
    def _save_generations(self, generations: List[torch.Tensor]):
        """Save generated samples."""
        output_file = self.output_dir / f"generations_{time.time():.0f}.json"
        
        # Convert to text
        generation_texts = []
        for gen in generations:
            try:
                text = bytes(gen.cpu().numpy()).decode('utf-8', errors='replace')
                generation_texts.append(text)
            except:
                generation_texts.append(f"<binary data, length={len(gen)}>")
        
        with open(output_file, 'w') as f:
            json.dump({
                "generations": generation_texts,
                "config": {
                    "temperature": self.config.generation_temperature,
                    "top_k": self.config.generation_top_k,
                    "top_p": self.config.generation_top_p,
                    "max_length": self.config.generation_max_length
                },
                "timestamp": time.time()
            }, f, indent=2)
    
    def run_benchmark(
        self,
        benchmark_suite: str = "standard"
    ) -> Dict[str, float]:
        """Run standardized benchmarks."""
        benchmarks = {
            "standard": [
                "next_byte_accuracy",
                "reconstruction_quality",
                "generation_coherence",
                "inference_speed"
            ],
            "comprehensive": [
                "next_byte_accuracy",
                "masked_modeling",
                "reconstruction_quality",
                "generation_coherence",
                "generation_diversity",
                "inference_speed",
                "memory_efficiency"
            ]
        }
        
        if benchmark_suite not in benchmarks:
            raise ValueError(f"Unknown benchmark suite: {benchmark_suite}")
        
        results = {}
        
        # Run validation to get base metrics
        metrics = self.validate()
        
        # Extract relevant metrics for benchmark
        if "next_byte_accuracy" in benchmarks[benchmark_suite]:
            results["next_byte_accuracy"] = metrics.task_metrics.get(
                "next_byte_prediction", {}
            ).get("accuracy", 0.0)
        
        if "reconstruction_quality" in benchmarks[benchmark_suite]:
            results["reconstruction_quality"] = 1.0 - metrics.task_metrics.get(
                "reconstruction", {}
            ).get("avg_edit_distance", 1.0)
        
        if "generation_coherence" in benchmarks[benchmark_suite]:
            results["generation_coherence"] = metrics.task_metrics.get(
                "generation_quality", {}
            ).get("generation_coherence", 0.0)
        
        if "inference_speed" in benchmarks[benchmark_suite]:
            results["inference_speed"] = metrics.throughput_tokens_per_sec
        
        if "memory_efficiency" in benchmarks[benchmark_suite]:
            results["memory_efficiency"] = 1000.0 / max(metrics.memory_usage_mb, 1.0)
        
        return results