"""
KL Divergence Monitoring for ProRL

Implements KL divergence control to prevent policy collapse during
prolonged reinforcement learning training.
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class KLDivergenceConfig:
    """Configuration for KL divergence monitoring"""
    target_kl: float = 0.01  # Target KL divergence
    max_kl: float = 0.05     # Maximum allowed KL divergence
    adaptive_kl: bool = True  # Whether to adaptively adjust target KL
    kl_window_size: int = 100  # Window for KL history
    early_stop_threshold: float = 0.1  # Stop if KL exceeds this
    reference_update_freq: int = 500  # Update reference policy every N steps
    

class KLDivergenceMonitor:
    """Monitors and controls KL divergence during training"""
    
    def __init__(self, config: KLDivergenceConfig):
        self.config = config
        self.kl_history = deque(maxlen=config.kl_window_size)
        self.step_count = 0
        self.reference_policy_state = None
        self.current_policy_state = None
        
        # Statistics tracking
        self.stats = {
            'total_violations': 0,
            'early_stops': 0,
            'reference_updates': 0,
            'avg_kl': 0.0,
            'max_kl_seen': 0.0,
            'kl_trend': 'stable'
        }
    
    def update_reference_policy(self, policy_state_dict: Dict[str, torch.Tensor]):
        """Update the reference policy state"""
        self.reference_policy_state = {
            k: v.clone().detach() 
            for k, v in policy_state_dict.items()
        }
        self.stats['reference_updates'] += 1
        logger.info(f"Updated reference policy at step {self.step_count}")
    
    def compute_kl_divergence(
        self, 
        current_logits: torch.Tensor, 
        reference_logits: torch.Tensor,
        temperature: float = 1.0
    ) -> float:
        """Compute KL divergence between current and reference policies"""
        # Apply temperature scaling
        current_probs = F.softmax(current_logits / temperature, dim=-1)
        reference_probs = F.softmax(reference_logits / temperature, dim=-1)
        
        # Compute KL divergence: KL(current || reference)
        kl_div = F.kl_div(
            F.log_softmax(current_logits / temperature, dim=-1),
            reference_probs,
            reduction='batchmean'
        )
        
        return kl_div.item()
    
    def compute_policy_kl(
        self, 
        current_model: torch.nn.Module,
        reference_model: torch.nn.Module,
        batch: Dict[str, torch.Tensor]
    ) -> float:
        """Compute KL divergence between policy outputs"""
        with torch.no_grad():
            # Forward pass through both models
            current_outputs = current_model(**batch)
            reference_outputs = reference_model(**batch)
            
            # Extract logits
            current_logits = current_outputs.logits if hasattr(current_outputs, 'logits') else current_outputs
            reference_logits = reference_outputs.logits if hasattr(reference_outputs, 'logits') else reference_outputs
            
            # Compute KL divergence
            kl_div = self.compute_kl_divergence(current_logits, reference_logits)
            
        return kl_div
    
    def check_kl_constraint(self, kl_value: float) -> Dict[str, Any]:
        """Check if KL divergence violates constraints"""
        self.kl_history.append(kl_value)
        self.step_count += 1
        
        # Update statistics
        self.stats['avg_kl'] = np.mean(self.kl_history)
        self.stats['max_kl_seen'] = max(self.stats['max_kl_seen'], kl_value)
        
        # Check for violations
        violation = kl_value > self.config.max_kl
        early_stop = kl_value > self.config.early_stop_threshold
        
        if violation:
            self.stats['total_violations'] += 1
        
        if early_stop:
            self.stats['early_stops'] += 1
        
        # Determine trend
        if len(self.kl_history) >= 10:
            recent_avg = np.mean(list(self.kl_history)[-10:])
            older_avg = np.mean(list(self.kl_history)[-20:-10]) if len(self.kl_history) >= 20 else recent_avg
            
            if recent_avg > older_avg * 1.1:
                self.stats['kl_trend'] = 'increasing'
            elif recent_avg < older_avg * 0.9:
                self.stats['kl_trend'] = 'decreasing'
            else:
                self.stats['kl_trend'] = 'stable'
        
        # Check if reference policy should be updated
        should_update_reference = (
            self.step_count % self.config.reference_update_freq == 0 or
            violation
        )
        
        return {
            'kl_value': kl_value,
            'violation': violation,
            'early_stop': early_stop,
            'should_update_reference': should_update_reference,
            'current_step': self.step_count,
            'avg_kl': self.stats['avg_kl'],
            'trend': self.stats['kl_trend']
        }
    
    def get_adaptive_target_kl(self) -> float:
        """Get adaptive target KL based on training progress"""
        if not self.config.adaptive_kl or len(self.kl_history) < 10:
            return self.config.target_kl
        
        # Adaptive KL based on stability
        recent_std = np.std(list(self.kl_history)[-10:])
        
        if recent_std < 0.001:  # Very stable
            return self.config.target_kl * 1.2  # Allow more exploration
        elif recent_std > 0.01:  # Very unstable
            return self.config.target_kl * 0.8  # Be more conservative
        else:
            return self.config.target_kl
    
    def get_kl_penalty(self, kl_value: float) -> float:
        """Calculate penalty based on KL divergence"""
        target_kl = self.get_adaptive_target_kl()
        
        if kl_value <= target_kl:
            return 0.0  # No penalty within target
        
        # Quadratic penalty for exceeding target
        excess_kl = kl_value - target_kl
        penalty = (excess_kl / self.config.max_kl) ** 2
        
        return min(penalty, 1.0)  # Cap penalty at 1.0
    
    def plot_kl_history(self, save_path: Optional[str] = None):
        """Plot KL divergence history"""
        if len(self.kl_history) < 2:
            return
        
        plt.figure(figsize=(12, 6))
        
        # Main plot
        plt.subplot(1, 2, 1)
        steps = range(len(self.kl_history))
        plt.plot(steps, self.kl_history, 'b-', alpha=0.7, label='KL Divergence')
        plt.axhline(y=self.config.target_kl, color='g', linestyle='--', label='Target KL')
        plt.axhline(y=self.config.max_kl, color='r', linestyle='--', label='Max KL')
        plt.xlabel('Training Steps')
        plt.ylabel('KL Divergence')
        plt.title('KL Divergence Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Moving average
        if len(self.kl_history) >= 10:
            window_size = min(20, len(self.kl_history) // 5)
            moving_avg = np.convolve(self.kl_history, np.ones(window_size)/window_size, mode='valid')
            avg_steps = range(window_size-1, len(self.kl_history))
            plt.plot(avg_steps, moving_avg, 'orange', linewidth=2, label=f'Moving Avg ({window_size})')
            plt.legend()
        
        # Distribution plot
        plt.subplot(1, 2, 2)
        plt.hist(self.kl_history, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(x=self.config.target_kl, color='g', linestyle='--', label='Target KL')
        plt.axvline(x=self.config.max_kl, color='r', linestyle='--', label='Max KL')
        plt.xlabel('KL Divergence')
        plt.ylabel('Frequency')
        plt.title('KL Divergence Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved KL divergence plot to {save_path}")
        else:
            plt.show()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive KL divergence statistics"""
        kl_array = np.array(self.kl_history)
        
        return {
            'current_step': self.step_count,
            'total_violations': self.stats['total_violations'],
            'violation_rate': self.stats['total_violations'] / max(1, self.step_count),
            'early_stops': self.stats['early_stops'],
            'reference_updates': self.stats['reference_updates'],
            'current_kl': kl_array[-1] if len(kl_array) > 0 else 0.0,
            'avg_kl': np.mean(kl_array) if len(kl_array) > 0 else 0.0,
            'std_kl': np.std(kl_array) if len(kl_array) > 0 else 0.0,
            'min_kl': np.min(kl_array) if len(kl_array) > 0 else 0.0,
            'max_kl': np.max(kl_array) if len(kl_array) > 0 else 0.0,
            'trend': self.stats['kl_trend'],
            'adaptive_target_kl': self.get_adaptive_target_kl(),
            'config': {
                'target_kl': self.config.target_kl,
                'max_kl': self.config.max_kl,
                'early_stop_threshold': self.config.early_stop_threshold
            }
        }
    
    def reset(self):
        """Reset the monitor for new training run"""
        self.kl_history.clear()
        self.step_count = 0
        self.reference_policy_state = None
        self.current_policy_state = None
        self.stats = {
            'total_violations': 0,
            'early_stops': 0,
            'reference_updates': 0,
            'avg_kl': 0.0,
            'max_kl_seen': 0.0,
            'kl_trend': 'stable'
        }
        logger.info("KL divergence monitor reset")


class PolicyDistanceCalculator:
    """Calculates various distance metrics between policies"""
    
    @staticmethod
    def parameter_distance(
        policy1_state: Dict[str, torch.Tensor],
        policy2_state: Dict[str, torch.Tensor],
        distance_type: str = 'l2'
    ) -> float:
        """Calculate distance between policy parameters"""
        total_distance = 0.0
        total_params = 0
        
        for key in policy1_state.keys():
            if key in policy2_state:
                p1 = policy1_state[key].flatten()
                p2 = policy2_state[key].flatten()
                
                if distance_type == 'l2':
                    distance = torch.norm(p1 - p2, p=2).item()
                elif distance_type == 'l1':
                    distance = torch.norm(p1 - p2, p=1).item()
                elif distance_type == 'cosine':
                    distance = 1 - F.cosine_similarity(p1, p2, dim=0).item()
                else:
                    raise ValueError(f"Unknown distance type: {distance_type}")
                
                total_distance += distance
                total_params += p1.numel()
        
        return total_distance / max(1, total_params)
    
    @staticmethod
    def output_divergence(
        policy1_outputs: torch.Tensor,
        policy2_outputs: torch.Tensor,
        divergence_type: str = 'kl'
    ) -> float:
        """Calculate divergence between policy outputs"""
        if divergence_type == 'kl':
            # KL divergence
            log_p1 = F.log_softmax(policy1_outputs, dim=-1)
            p2 = F.softmax(policy2_outputs, dim=-1)
            return F.kl_div(log_p1, p2, reduction='batchmean').item()
        
        elif divergence_type == 'js':
            # Jensen-Shannon divergence
            p1 = F.softmax(policy1_outputs, dim=-1)
            p2 = F.softmax(policy2_outputs, dim=-1)
            m = 0.5 * (p1 + p2)
            
            kl1 = F.kl_div(torch.log(p1 + 1e-10), m, reduction='batchmean')
            kl2 = F.kl_div(torch.log(p2 + 1e-10), m, reduction='batchmean')
            
            return 0.5 * (kl1.item() + kl2.item())
        
        elif divergence_type == 'tvd':
            # Total Variation Distance
            p1 = F.softmax(policy1_outputs, dim=-1)
            p2 = F.softmax(policy2_outputs, dim=-1)
            return 0.5 * torch.sum(torch.abs(p1 - p2), dim=-1).mean().item()
        
        else:
            raise ValueError(f"Unknown divergence type: {divergence_type}")


# Testing and utility functions
def test_kl_monitor():
    """Test the KL divergence monitor"""
    config = KLDivergenceConfig(
        target_kl=0.01,
        max_kl=0.05,
        adaptive_kl=True
    )
    
    monitor = KLDivergenceMonitor(config)
    
    # Simulate training with increasing KL divergence
    for step in range(100):
        # Simulate KL values
        if step < 50:
            kl_value = np.random.normal(0.005, 0.001)  # Stable phase
        else:
            kl_value = np.random.normal(0.02, 0.005)   # Increasing phase
        
        kl_value = max(0, kl_value)  # Ensure non-negative
        
        result = monitor.check_kl_constraint(kl_value)
        
        if step % 20 == 0:
            print(f"Step {step}: KL={kl_value:.4f}, "
                  f"Violation={result['violation']}, "
                  f"Trend={result['trend']}")
    
    # Print final statistics
    stats = monitor.get_statistics()
    print(f"\nFinal Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Plot results
    monitor.plot_kl_history("test_kl_plot.png")


if __name__ == "__main__":
    test_kl_monitor()