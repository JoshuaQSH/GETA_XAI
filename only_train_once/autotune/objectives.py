"""
Objective functions for XAI hyperparameter optimization.

Supports multiple optimization objectives:
- Accuracy: Maximize final validation accuracy
- Loss smoothness: Minimize loss variance during training
- Multi-objective: Pareto optimization of multiple goals
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_loss_smoothness(loss_history: List[float], window_size: int = 10) -> float:
    """
    Compute loss smoothness metric from training loss history.
    
    Lower values indicate smoother, more stable training.
    
    Args:
        loss_history: List of loss values during training
        window_size: Window size for computing local variance
    
    Returns:
        Smoothness score (lower is better)
    """
    if len(loss_history) < window_size:
        # For short histories, use full history with smaller effective window
        if len(loss_history) < 3:
            return 1.0  # Return neutral score for very short histories
        window_size = min(window_size, len(loss_history) - 1)
    
    losses = np.array(loss_history)
    
    # Compute rolling variance
    rolling_var = []
    for i in range(len(losses) - window_size + 1):
        window = losses[i:i + window_size]
        rolling_var.append(np.var(window))
    
    # Also penalize large jumps
    diffs = np.abs(np.diff(losses))
    jump_penalty = np.mean(diffs) + np.std(diffs)
    
    # Final smoothness metric
    smoothness = np.mean(rolling_var) + 0.5 * jump_penalty
    
    return smoothness


def compute_convergence_quality(
    loss_history: List[float],
    final_window: int = 20
) -> float:
    """
    Compute convergence quality metric.
    
    Measures how well the loss has converged at the end of training.
    
    Args:
        loss_history: List of loss values
        final_window: Window at end to analyze
    
    Returns:
        Convergence score (lower is better, 0 = perfect convergence)
    """
    if len(loss_history) < 3:
        return 1.0  # Return neutral score for very short histories
    
    # Adjust window to available data
    final_window = min(final_window, len(loss_history))
    
    final_losses = np.array(loss_history[-final_window:])
    
    # Check if still decreasing (bad if slope is very negative)
    slope = np.polyfit(range(len(final_losses)), final_losses, 1)[0]
    
    # Variance in final window (should be low for converged loss)
    variance = np.var(final_losses)
    
    # Penalize if slope is still negative (not converged)
    # Penalize high variance (unstable convergence)
    convergence = variance + max(0, -slope * 10)
    
    return convergence


@dataclass
class AccuracyObjective:
    """
    Objective: Maximize final validation accuracy.
    
    This is the simplest objective - just return negative accuracy
    (negative because optimizers minimize by default).
    """
    
    def __call__(self, metrics: Dict[str, Any]) -> float:
        """
        Compute objective value.
        
        Args:
            metrics: Dictionary containing at least 'final_accuracy'
        
        Returns:
            Negative accuracy (for minimization)
        """
        accuracy = metrics.get('final_accuracy', 0.0)
        return -accuracy  # Negative for minimization


@dataclass
class LossSmoothnessObjective:
    """
    Objective: Minimize loss variance and maximize convergence.
    
    This objective prioritizes stable training with smooth loss curves.
    """
    
    window_size: int = 10
    convergence_weight: float = 0.3
    
    def __call__(self, metrics: Dict[str, Any]) -> float:
        """
        Compute objective value.
        
        Args:
            metrics: Dictionary containing 'loss_history'
        
        Returns:
            Smoothness score (for minimization)
        """
        loss_history = metrics.get('loss_history', [])
        if len(loss_history) < self.window_size:
            return float('inf')
        
        smoothness = compute_loss_smoothness(loss_history, self.window_size)
        convergence = compute_convergence_quality(loss_history)
        
        return smoothness + self.convergence_weight * convergence


@dataclass
class MultiObjective:
    """
    Multi-objective optimization combining accuracy and loss smoothness.
    
    Uses weighted combination: 
        score = -accuracy_weight * accuracy + smoothness_weight * smoothness
    
    Attributes:
        accuracy_weight: Weight for accuracy term (default: 1.0)
        smoothness_weight: Weight for smoothness term (default: 0.1)
        sparsity_weight: Weight for achieved sparsity (default: 0.0)
    """
    
    accuracy_weight: float = 1.0
    smoothness_weight: float = 0.1
    sparsity_weight: float = 0.0
    convergence_weight: float = 0.1
    
    def __call__(self, metrics: Dict[str, Any]) -> float:
        """
        Compute combined objective value.
        
        Args:
            metrics: Dictionary containing:
                - 'final_accuracy': float
                - 'loss_history': List[float]
                - 'final_sparsity': float (optional)
        
        Returns:
            Combined score (for minimization)
        """
        # Accuracy term (negative for minimization)
        accuracy = metrics.get('final_accuracy', 0.0)
        accuracy_term = -self.accuracy_weight * accuracy
        
        # Smoothness term
        loss_history = metrics.get('loss_history', [])
        if len(loss_history) >= 10:
            smoothness = compute_loss_smoothness(loss_history)
            convergence = compute_convergence_quality(loss_history)
        else:
            smoothness = 0.0
            convergence = 0.0
        smoothness_term = self.smoothness_weight * smoothness
        convergence_term = self.convergence_weight * convergence
        
        # Sparsity term (optional)
        sparsity = metrics.get('final_sparsity', 0.0)
        # Penalize if sparsity is far from target (assuming target ~0.5)
        sparsity_term = self.sparsity_weight * abs(sparsity - 0.5)
        
        total = accuracy_term + smoothness_term + convergence_term + sparsity_term
        
        logger.debug(
            f"MultiObjective: acc={accuracy:.4f}, smooth={smoothness:.4f}, "
            f"conv={convergence:.4f}, sparsity={sparsity:.2f} -> {total:.4f}"
        )
        
        return total


def create_objective(
    objective_type: str = 'multi',
    **kwargs
) -> Callable[[Dict[str, Any]], float]:
    """
    Factory function to create objective instances.
    
    Args:
        objective_type: Type of objective
            - 'accuracy': Maximize accuracy only
            - 'smoothness': Minimize loss variance
            - 'multi': Balanced multi-objective
        **kwargs: Additional arguments for objective constructor
    
    Returns:
        Callable objective function
    """
    if objective_type == 'accuracy':
        return AccuracyObjective()
    elif objective_type == 'smoothness':
        return LossSmoothnessObjective(**kwargs)
    elif objective_type == 'multi':
        return MultiObjective(**kwargs)
    else:
        raise ValueError(f"Unknown objective type: {objective_type}")
