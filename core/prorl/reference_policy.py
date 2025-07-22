"""
Reference Policy Management for ProRL

Manages reference policies to prevent catastrophic forgetting during
prolonged reinforcement learning training.
"""

import torch
import torch.nn as nn
import copy
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReferencePolicyConfig:
    """Configuration for reference policy management"""
    update_frequency: int = 500  # Update reference every N steps
    max_references: int = 10     # Maximum number of reference policies to keep
    similarity_threshold: float = 0.1  # Minimum difference for new reference
    save_interval: int = 1000    # Save references every N steps
    load_on_init: bool = True    # Load existing references on initialization
    reference_dir: str = "checkpoints/references"


class ReferencePolicyManager:
    """Manages multiple reference policies during ProRL training"""
    
    def __init__(self, config: ReferencePolicyConfig):
        self.config = config
        self.reference_policies = {}  # Dict[int, Dict[str, torch.Tensor]]
        self.reference_metadata = {}  # Dict[int, Dict[str, Any]]
        self.current_step = 0
        self.last_update_step = 0
        
        # Create reference directory
        self.reference_dir = Path(config.reference_dir)
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing references if requested
        if config.load_on_init:
            self.load_references()
    
    def add_reference_policy(
        self, 
        policy_state: Dict[str, torch.Tensor],
        step: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a new reference policy"""
        
        # Check if we should add this policy
        if not self._should_add_reference(policy_state, step):
            return False
        
        # Create deep copy of policy state
        reference_state = {
            k: v.clone().detach().cpu() 
            for k, v in policy_state.items()
        }
        
        # Default metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'num_parameters': sum(p.numel() for p in reference_state.values()),
            'avg_parameter_norm': np.mean([
                torch.norm(p).item() for p in reference_state.values()
            ])
        })
        
        # Add to storage
        self.reference_policies[step] = reference_state
        self.reference_metadata[step] = metadata
        
        # Maintain maximum number of references
        self._cleanup_old_references()
        
        logger.info(f"Added reference policy at step {step}")
        return True
    
    def get_closest_reference(
        self, 
        current_policy: Dict[str, torch.Tensor],
        max_distance: Optional[float] = None
    ) -> Optional[Tuple[int, Dict[str, torch.Tensor]]]:
        """Get the reference policy closest to current policy"""
        
        if not self.reference_policies:
            return None
        
        min_distance = float('inf')
        closest_step = None
        
        for step, ref_policy in self.reference_policies.items():
            distance = self._calculate_policy_distance(current_policy, ref_policy)
            
            if distance < min_distance:
                min_distance = distance
                closest_step = step
        
        # Check distance threshold
        if max_distance is not None and min_distance > max_distance:
            return None
        
        return closest_step, self.reference_policies[closest_step]
    
    def get_reference_at_step(self, step: int) -> Optional[Dict[str, torch.Tensor]]:
        """Get reference policy at specific step"""
        return self.reference_policies.get(step)
    
    def get_latest_reference(self) -> Optional[Tuple[int, Dict[str, torch.Tensor]]]:
        """Get the most recent reference policy"""
        if not self.reference_policies:
            return None
        
        latest_step = max(self.reference_policies.keys())
        return latest_step, self.reference_policies[latest_step]
    
    def should_update_reference(self, current_step: int) -> bool:
        """Check if reference policy should be updated"""
        return (
            current_step - self.last_update_step >= self.config.update_frequency or
            len(self.reference_policies) == 0
        )
    
    def update_reference(
        self,
        current_policy: Dict[str, torch.Tensor],
        current_step: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update reference policy if needed"""
        
        if not self.should_update_reference(current_step):
            return False
        
        success = self.add_reference_policy(current_policy, current_step, metadata)
        
        if success:
            self.last_update_step = current_step
            
            # Save references periodically
            if current_step % self.config.save_interval == 0:
                self.save_references()
        
        return success
    
    def _should_add_reference(
        self, 
        policy_state: Dict[str, torch.Tensor],
        step: int
    ) -> bool:
        """Determine if policy should be added as reference"""
        
        # Always add first reference
        if not self.reference_policies:
            return True
        
        # Check if policy is sufficiently different
        closest = self.get_closest_reference(policy_state)
        if closest is None:
            return True
        
        _, closest_policy = closest
        distance = self._calculate_policy_distance(policy_state, closest_policy)
        
        return distance >= self.config.similarity_threshold
    
    def _calculate_policy_distance(
        self,
        policy1: Dict[str, torch.Tensor],
        policy2: Dict[str, torch.Tensor]
    ) -> float:
        """Calculate L2 distance between two policies"""
        
        total_distance = 0.0
        total_params = 0
        
        for key in policy1.keys():
            if key in policy2:
                # Move to same device for comparison
                p1 = policy1[key].flatten()
                p2 = policy2[key].flatten().to(p1.device)
                
                distance = torch.norm(p1 - p2, p=2).item()
                total_distance += distance
                total_params += p1.numel()
        
        return total_distance / max(1, total_params)
    
    def _cleanup_old_references(self):
        """Remove old references if exceeding maximum"""
        
        if len(self.reference_policies) <= self.config.max_references:
            return
        
        # Keep the most recent references
        steps = sorted(self.reference_policies.keys())
        steps_to_remove = steps[:-self.config.max_references]
        
        for step in steps_to_remove:
            del self.reference_policies[step]
            del self.reference_metadata[step]
        
        logger.info(f"Removed {len(steps_to_remove)} old reference policies")
    
    def save_references(self, filepath: Optional[str] = None):
        """Save reference policies to disk"""
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.reference_dir / f"references_{timestamp}.pt"
        
        save_data = {
            'reference_policies': self.reference_policies,
            'reference_metadata': self.reference_metadata,
            'config': self.config.__dict__,
            'current_step': self.current_step,
            'last_update_step': self.last_update_step
        }
        
        torch.save(save_data, filepath)
        logger.info(f"Saved {len(self.reference_policies)} reference policies to {filepath}")
        
        # Also save metadata as JSON for easy inspection
        metadata_file = filepath.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(self.reference_metadata, f, indent=2, default=str)
    
    def load_references(self, filepath: Optional[str] = None):
        """Load reference policies from disk"""
        
        if filepath is None:
            # Find the most recent reference file
            reference_files = list(self.reference_dir.glob("references_*.pt"))
            if not reference_files:
                logger.info("No reference files found to load")
                return
            
            filepath = max(reference_files, key=lambda x: x.stat().st_mtime)
        
        try:
            save_data = torch.load(filepath, map_location='cpu')
            
            self.reference_policies = save_data['reference_policies']
            self.reference_metadata = save_data['reference_metadata']
            self.current_step = save_data.get('current_step', 0)
            self.last_update_step = save_data.get('last_update_step', 0)
            
            logger.info(f"Loaded {len(self.reference_policies)} reference policies from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading reference policies: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about reference policies"""
        
        if not self.reference_policies:
            return {
                'num_references': 0,
                'steps_covered': [],
                'avg_distance_between_refs': 0.0,
                'latest_step': 0
            }
        
        steps = sorted(self.reference_policies.keys())
        
        # Calculate average distance between consecutive references
        distances = []
        if len(steps) > 1:
            for i in range(len(steps) - 1):
                policy1 = self.reference_policies[steps[i]]
                policy2 = self.reference_policies[steps[i + 1]]
                distance = self._calculate_policy_distance(policy1, policy2)
                distances.append(distance)
        
        return {
            'num_references': len(self.reference_policies),
            'steps_covered': steps,
            'step_range': (min(steps), max(steps)) if steps else (0, 0),
            'avg_distance_between_refs': np.mean(distances) if distances else 0.0,
            'latest_step': max(steps) if steps else 0,
            'coverage_ratio': len(steps) / max(1, max(steps) - min(steps) + 1) if len(steps) > 1 else 1.0,
            'metadata_summary': {
                step: {
                    'num_parameters': meta.get('num_parameters', 0),
                    'avg_parameter_norm': meta.get('avg_parameter_norm', 0.0)
                }
                for step, meta in self.reference_metadata.items()
            }
        }
    
    def interpolate_references(
        self,
        step1: int,
        step2: int,
        alpha: float = 0.5
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Interpolate between two reference policies"""
        
        if step1 not in self.reference_policies or step2 not in self.reference_policies:
            return None
        
        policy1 = self.reference_policies[step1]
        policy2 = self.reference_policies[step2]
        
        interpolated = {}
        for key in policy1.keys():
            if key in policy2:
                interpolated[key] = (1 - alpha) * policy1[key] + alpha * policy2[key]
        
        return interpolated
    
    def create_ensemble_reference(
        self, 
        weights: Optional[List[float]] = None
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Create ensemble reference from all stored policies"""
        
        if not self.reference_policies:
            return None
        
        policies = list(self.reference_policies.values())
        
        if weights is None:
            weights = [1.0 / len(policies)] * len(policies)
        
        if len(weights) != len(policies):
            raise ValueError("Number of weights must match number of policies")
        
        ensemble = {}
        
        # Get parameter names from first policy
        first_policy = policies[0]
        
        for key in first_policy.keys():
            # Weighted average of parameters
            weighted_sum = torch.zeros_like(first_policy[key])
            
            for i, policy in enumerate(policies):
                if key in policy:
                    weighted_sum += weights[i] * policy[key]
            
            ensemble[key] = weighted_sum
        
        return ensemble
    
    def clear_references(self):
        """Clear all reference policies"""
        self.reference_policies.clear()
        self.reference_metadata.clear()
        self.current_step = 0
        self.last_update_step = 0
        logger.info("Cleared all reference policies")


class ReferencePolicyEvaluator:
    """Evaluates quality and diversity of reference policies"""
    
    def __init__(self, reference_manager: ReferencePolicyManager):
        self.ref_manager = reference_manager
    
    def evaluate_coverage(self, total_training_steps: int) -> Dict[str, float]:
        """Evaluate how well references cover the training trajectory"""
        
        stats = self.ref_manager.get_statistics()
        steps = stats['steps_covered']
        
        if not steps:
            return {
                'temporal_coverage': 0.0,
                'density': 0.0,
                'uniformity': 0.0
            }
        
        # Temporal coverage: what fraction of training is covered
        min_step, max_step = min(steps), max(steps)
        temporal_coverage = (max_step - min_step) / max(1, total_training_steps)
        
        # Density: average steps between references
        if len(steps) > 1:
            gaps = [steps[i+1] - steps[i] for i in range(len(steps)-1)]
            density = 1.0 / (np.mean(gaps) / total_training_steps + 1e-6)
        else:
            density = 0.0
        
        # Uniformity: how evenly distributed are the references
        if len(steps) > 2:
            expected_gap = (max_step - min_step) / (len(steps) - 1)
            actual_gaps = [steps[i+1] - steps[i] for i in range(len(steps)-1)]
            uniformity = 1.0 / (np.std(actual_gaps) / expected_gap + 1e-6)
        else:
            uniformity = 1.0
        
        return {
            'temporal_coverage': temporal_coverage,
            'density': min(1.0, density),
            'uniformity': min(1.0, uniformity / 10)  # Normalize
        }
    
    def evaluate_diversity(self) -> Dict[str, float]:
        """Evaluate diversity among reference policies"""
        
        policies = list(self.ref_manager.reference_policies.values())
        
        if len(policies) < 2:
            return {
                'avg_pairwise_distance': 0.0,
                'min_distance': 0.0,
                'max_distance': 0.0,
                'diversity_score': 0.0
            }
        
        # Calculate all pairwise distances
        distances = []
        for i in range(len(policies)):
            for j in range(i+1, len(policies)):
                distance = self.ref_manager._calculate_policy_distance(policies[i], policies[j])
                distances.append(distance)
        
        avg_distance = np.mean(distances)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        
        # Diversity score based on average distance and spread
        diversity_score = avg_distance * (1 + np.std(distances))
        
        return {
            'avg_pairwise_distance': avg_distance,
            'min_distance': min_distance,
            'max_distance': max_distance,
            'diversity_score': diversity_score
        }


# Testing and example usage
def test_reference_manager():
    """Test the reference policy manager"""
    
    # Create config
    config = ReferencePolicyConfig(
        update_frequency=10,
        max_references=5,
        similarity_threshold=0.1
    )
    
    # Create manager
    manager = ReferencePolicyManager(config)
    
    # Simulate training with evolving policies
    for step in range(100):
        # Create mock policy (random parameters)
        policy = {
            'layer1.weight': torch.randn(10, 5),
            'layer1.bias': torch.randn(10),
            'layer2.weight': torch.randn(1, 10)
        }
        
        # Add some evolution
        if step > 0:
            prev_step = max(s for s in manager.reference_policies.keys() if s < step) if manager.reference_policies else 0
            if prev_step in manager.reference_policies:
                prev_policy = manager.reference_policies[prev_step]
                for key in policy.keys():
                    if key in prev_policy:
                        # Evolve parameters gradually
                        policy[key] = prev_policy[key] + 0.1 * torch.randn_like(prev_policy[key])
        
        # Update reference
        updated = manager.update_reference(policy, step)
        
        if updated:
            print(f"Step {step}: Added reference policy")
    
    # Print statistics
    stats = manager.get_statistics()
    print(f"\nFinal statistics:")
    for key, value in stats.items():
        if key != 'metadata_summary':
            print(f"{key}: {value}")
    
    # Test evaluator
    evaluator = ReferencePolicyEvaluator(manager)
    coverage = evaluator.evaluate_coverage(100)
    diversity = evaluator.evaluate_diversity()
    
    print(f"\nCoverage evaluation: {coverage}")
    print(f"Diversity evaluation: {diversity}")


if __name__ == "__main__":
    test_reference_manager()