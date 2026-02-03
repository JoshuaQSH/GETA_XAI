"""
Autotuning module for XAI-GETA hyperparameter optimization.

This module provides tools for automatically tuning XAI hyperparameters
using Nevergrad evolutionary optimization or Optuna Bayesian optimization.
"""

from .autotune_xai import AutotuneXAI
from .search_space import XAISearchSpace, get_default_search_space
from .objectives import (
    AccuracyObjective,
    LossSmoothnessObjective, 
    MultiObjective,
    compute_loss_smoothness,
)

__all__ = [
    'AutotuneXAI',
    'XAISearchSpace',
    'get_default_search_space',
    'AccuracyObjective',
    'LossSmoothnessObjective',
    'MultiObjective',
    'compute_loss_smoothness',
]
