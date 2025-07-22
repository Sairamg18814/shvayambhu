"""Evaluation metrics for language modeling."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import editdistance


def compute_language_modeling_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """Compute standard language modeling metrics.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        targets: Ground truth targets
        mask: Optional mask for valid positions
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Ensure tensors are on same device
    device = predictions.device
    targets = targets.to(device)
    if mask is not None:
        mask = mask.to(device)
    
    # Get predictions
    if predictions.dim() == 3:  # [batch, seq, vocab]
        pred_tokens = predictions.argmax(dim=-1)
    else:
        pred_tokens = predictions
    
    # Apply mask if provided
    if mask is not None:
        valid_predictions = pred_tokens[mask]
        valid_targets = targets[mask]
    else:
        valid_predictions = pred_tokens.flatten()
        valid_targets = targets.flatten()
    
    # Accuracy
    correct = (valid_predictions == valid_targets).float()
    metrics['accuracy'] = correct.mean().item()
    
    # Top-k accuracy
    if predictions.dim() == 3:
        for k in [5, 10]:
            if predictions.size(-1) >= k:
                top_k = predictions.topk(k, dim=-1).indices
                top_k_correct = (top_k == targets.unsqueeze(-1)).any(dim=-1)
                if mask is not None:
                    top_k_correct = top_k_correct[mask]
                metrics[f'top_{k}_accuracy'] = top_k_correct.float().mean().item()
    
    return metrics


def compute_perplexity(
    loss_values: List[float],
    base: float = np.e
) -> float:
    """Compute perplexity from loss values.
    
    Args:
        loss_values: List of loss values
        base: Base for perplexity calculation (e for natural log)
        
    Returns:
        Perplexity value
    """
    avg_loss = np.mean(loss_values)
    return float(base ** avg_loss)


def compute_bleu_score(
    predictions: List[str],
    references: List[List[str]],
    max_n: int = 4,
    weights: Optional[List[float]] = None
) -> Dict[str, float]:
    """Compute BLEU score for generated text.
    
    Simplified BLEU implementation for byte-level evaluation.
    
    Args:
        predictions: List of predicted strings
        references: List of reference strings (multiple per prediction)
        max_n: Maximum n-gram size
        weights: Weights for different n-gram sizes
        
    Returns:
        Dictionary with BLEU scores
    """
    if weights is None:
        weights = [1.0 / max_n] * max_n
    
    scores = {'bleu': 0.0}
    
    for n in range(1, max_n + 1):
        precision_scores = []
        
        for pred, refs in zip(predictions, references):
            # Convert to bytes for byte-level evaluation
            pred_bytes = pred.encode('utf-8')
            ref_bytes_list = [ref.encode('utf-8') for ref in refs]
            
            # Extract n-grams
            pred_ngrams = [pred_bytes[i:i+n] for i in range(len(pred_bytes)-n+1)]
            
            if not pred_ngrams:
                precision_scores.append(0.0)
                continue
            
            # Count matches
            matches = 0
            pred_ngram_counts = Counter(pred_ngrams)
            
            for ref_bytes in ref_bytes_list:
                ref_ngrams = [ref_bytes[i:i+n] for i in range(len(ref_bytes)-n+1)]
                ref_ngram_counts = Counter(ref_ngrams)
                
                for ngram, count in pred_ngram_counts.items():
                    matches += min(count, ref_ngram_counts.get(ngram, 0))
            
            precision = matches / len(pred_ngrams) if pred_ngrams else 0.0
            precision_scores.append(precision)
        
        avg_precision = np.mean(precision_scores)
        scores[f'bleu_{n}'] = avg_precision
    
    # Compute weighted average
    weighted_scores = [scores[f'bleu_{n}'] * w for n, w in enumerate(weights, 1)]
    scores['bleu'] = np.exp(np.mean(np.log(np.maximum(weighted_scores, 1e-10))))
    
    return scores


def compute_edit_distance(
    predictions: List[str],
    references: List[str],
    normalize: bool = True
) -> Dict[str, float]:
    """Compute edit distance metrics.
    
    Args:
        predictions: Predicted strings
        references: Reference strings
        normalize: Whether to normalize by reference length
        
    Returns:
        Dictionary of edit distance metrics
    """
    distances = []
    
    for pred, ref in zip(predictions, references):
        # Work at byte level
        pred_bytes = pred.encode('utf-8')
        ref_bytes = ref.encode('utf-8')
        
        distance = editdistance.eval(pred_bytes, ref_bytes)
        
        if normalize and len(ref_bytes) > 0:
            distance = distance / len(ref_bytes)
            
        distances.append(distance)
    
    return {
        'edit_distance': np.mean(distances),
        'edit_distance_std': np.std(distances),
        'edit_distance_min': np.min(distances),
        'edit_distance_max': np.max(distances)
    }


def compute_byte_accuracy(
    predictions: bytes,
    targets: bytes,
    window_size: Optional[int] = None
) -> Dict[str, float]:
    """Compute byte-level accuracy metrics.
    
    Args:
        predictions: Predicted byte sequence
        targets: Target byte sequence
        window_size: Optional window size for sliding accuracy
        
    Returns:
        Dictionary of accuracy metrics
    """
    if len(predictions) != len(targets):
        # Pad or truncate to same length
        min_len = min(len(predictions), len(targets))
        predictions = predictions[:min_len]
        targets = targets[:min_len]
    
    # Overall accuracy
    correct = sum(p == t for p, t in zip(predictions, targets))
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0.0
    
    metrics = {
        'byte_accuracy': accuracy,
        'total_bytes': total,
        'correct_bytes': correct
    }
    
    # Sliding window accuracy
    if window_size and total >= window_size:
        window_accuracies = []
        for i in range(total - window_size + 1):
            window_pred = predictions[i:i+window_size]
            window_target = targets[i:i+window_size]
            window_correct = sum(p == t for p, t in zip(window_pred, window_target))
            window_accuracies.append(window_correct / window_size)
        
        metrics[f'window_{window_size}_accuracy'] = np.mean(window_accuracies)
        metrics[f'window_{window_size}_std'] = np.std(window_accuracies)
    
    return metrics


def compute_entropy_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """Compute entropy-based metrics.
    
    Args:
        predictions: Model predictions (logits)
        targets: Ground truth targets
        
    Returns:
        Dictionary of entropy metrics
    """
    # Convert logits to probabilities
    probs = torch.softmax(predictions, dim=-1)
    
    # Compute entropy
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    
    # Get predicted tokens
    pred_tokens = predictions.argmax(dim=-1)
    correct = (pred_tokens == targets).float()
    
    # Compute confidence
    confidence = probs.max(dim=-1).values
    
    return {
        'avg_entropy': entropy.mean().item(),
        'entropy_std': entropy.std().item(),
        'avg_confidence': confidence.mean().item(),
        'confidence_on_correct': (confidence * correct).sum().item() / (correct.sum().item() + 1e-10),
        'confidence_on_incorrect': (confidence * (1 - correct)).sum().item() / ((1 - correct).sum().item() + 1e-10)
    }


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate metrics from multiple batches.
    
    Args:
        metrics_list: List of metric dictionaries
        
    Returns:
        Aggregated metrics
    """
    if not metrics_list:
        return {}
    
    aggregated = {}
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())
    
    for key in all_keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            aggregated[key] = np.mean(values)
            if len(values) > 1:
                aggregated[f'{key}_std'] = np.std(values)
    
    return aggregated
