"""
Search space definitions for XAI hyperparameter tuning.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Any, Optional


@dataclass
class XAISearchSpace:
    """
    Defines the search space for XAI hyperparameters.
    
    Each parameter can be:
    - A tuple (min, max) for continuous/integer parameters
    - A list of choices for categorical parameters
    
    Attributes:
        attribution_method: Captum method choices
        attribution_weight: Weight range [0, 1]
        attribution_freq: Frequency range (integer)
        attribution_n_steps: Integration steps range (integer)
        ema_decay: EMA decay range [0, 1]
        baseline_type: Baseline type choices
        
        # GETA hyperparameters (optional)
        learning_rate: Learning rate range
        target_group_sparsity: Sparsity target range
    """
    
    # XAI-specific parameters
    attribution_method: List[str] = field(default_factory=lambda: [
        'saliency', 'layer_conductance', 'integrated_gradients'
    ])
    attribution_weight: Tuple[float, float] = (0.1, 0.9)
    attribution_freq: Tuple[int, int] = (25, 500)
    attribution_n_steps: Tuple[int, int] = (5, 50)
    ema_decay: Tuple[float, float] = (0.7, 0.99)
    baseline_type: List[str] = field(default_factory=lambda: ['zero', 'mean', 'random'])
    
    # Optional GETA parameters
    learning_rate: Optional[Tuple[float, float]] = None
    target_group_sparsity: Optional[Tuple[float, float]] = None
    
    def to_nevergrad(self):
        """Convert search space to Nevergrad parametrization."""
        try:
            import nevergrad as ng
        except ImportError:
            raise ImportError("Nevergrad not installed. Install with: uv add nevergrad")
        
        params = {}
        
        # Categorical parameters
        params['attribution_method'] = ng.p.Choice(self.attribution_method)
        params['baseline_type'] = ng.p.Choice(self.baseline_type)
        
        # Continuous parameters
        params['attribution_weight'] = ng.p.Scalar(
            lower=self.attribution_weight[0], 
            upper=self.attribution_weight[1]
        )
        params['ema_decay'] = ng.p.Scalar(
            lower=self.ema_decay[0], 
            upper=self.ema_decay[1]
        )
        
        # Integer parameters (use Scalar with int conversion)
        params['attribution_freq'] = ng.p.Scalar(
            lower=self.attribution_freq[0], 
            upper=self.attribution_freq[1]
        ).set_integer_casting()
        params['attribution_n_steps'] = ng.p.Scalar(
            lower=self.attribution_n_steps[0], 
            upper=self.attribution_n_steps[1]
        ).set_integer_casting()
        
        # Optional GETA parameters
        if self.learning_rate is not None:
            params['learning_rate'] = ng.p.Log(
                lower=self.learning_rate[0],
                upper=self.learning_rate[1]
            )
        if self.target_group_sparsity is not None:
            params['target_group_sparsity'] = ng.p.Scalar(
                lower=self.target_group_sparsity[0],
                upper=self.target_group_sparsity[1]
            )
        
        return ng.p.Instrumentation(**params)
    
    def to_optuna(self, trial) -> Dict[str, Any]:
        """Sample from search space using Optuna trial."""
        params = {}
        
        # Categorical parameters
        params['attribution_method'] = trial.suggest_categorical(
            'attribution_method', self.attribution_method
        )
        params['baseline_type'] = trial.suggest_categorical(
            'baseline_type', self.baseline_type
        )
        
        # Continuous parameters
        params['attribution_weight'] = trial.suggest_float(
            'attribution_weight', *self.attribution_weight
        )
        params['ema_decay'] = trial.suggest_float(
            'ema_decay', *self.ema_decay
        )
        
        # Integer parameters
        params['attribution_freq'] = trial.suggest_int(
            'attribution_freq', *self.attribution_freq
        )
        params['attribution_n_steps'] = trial.suggest_int(
            'attribution_n_steps', *self.attribution_n_steps
        )
        
        # Optional GETA parameters
        if self.learning_rate is not None:
            params['learning_rate'] = trial.suggest_float(
                'learning_rate', *self.learning_rate, log=True
            )
        if self.target_group_sparsity is not None:
            params['target_group_sparsity'] = trial.suggest_float(
                'target_group_sparsity', *self.target_group_sparsity
            )
        
        return params
    
    def get_param_names(self) -> List[str]:
        """Get list of parameter names in search space."""
        names = [
            'attribution_method', 'attribution_weight', 'attribution_freq',
            'attribution_n_steps', 'ema_decay', 'baseline_type'
        ]
        if self.learning_rate is not None:
            names.append('learning_rate')
        if self.target_group_sparsity is not None:
            names.append('target_group_sparsity')
        return names


def get_default_search_space(mode: str = 'xai_only') -> XAISearchSpace:
    """
    Get default search space configurations.
    
    Args:
        mode: Search space mode
            - 'xai_only': Only XAI hyperparameters
            - 'xai_fast': XAI with fast methods only
            - 'full': XAI + GETA hyperparameters
    
    Returns:
        XAISearchSpace instance
    """
    if mode == 'xai_only':
        return XAISearchSpace()
    
    elif mode == 'xai_fast':
        return XAISearchSpace(
            attribution_method=['saliency', 'input_x_gradient'],
            attribution_weight=(0.2, 0.8),
            attribution_freq=(50, 300),
            attribution_n_steps=(5, 20),
            ema_decay=(0.8, 0.95),
            baseline_type=['zero', 'mean'],
        )
    
    elif mode == 'full':
        return XAISearchSpace(
            attribution_method=['saliency', 'layer_conductance', 'integrated_gradients'],
            attribution_weight=(0.1, 0.9),
            attribution_freq=(25, 500),
            attribution_n_steps=(5, 50),
            ema_decay=(0.7, 0.99),
            baseline_type=['zero', 'mean', 'random'],
            learning_rate=(1e-5, 1e-2),
            target_group_sparsity=(0.3, 0.7),
        )
    
    else:
        raise ValueError(f"Unknown search space mode: {mode}")


def get_narrow_search_space(
    center_config: Dict[str, Any],
    relative_range: float = 0.2
) -> XAISearchSpace:
    """
    Create a narrow search space around a center configuration.
    
    Useful for fine-tuning after initial coarse search.
    
    Args:
        center_config: Dictionary with center values
        relative_range: Relative range around center (e.g., 0.2 = ±20%)
    
    Returns:
        XAISearchSpace instance
    """
    def narrow_range(value, rel_range, min_val=0.0, max_val=1.0):
        delta = value * rel_range
        return (max(min_val, value - delta), min(max_val, value + delta))
    
    def narrow_int_range(value, rel_range, min_val=1):
        delta = max(1, int(value * rel_range))
        return (max(min_val, value - delta), value + delta)
    
    return XAISearchSpace(
        # Keep categorical as-is or narrow to current value
        attribution_method=[center_config.get('attribution_method', 'saliency')],
        baseline_type=[center_config.get('baseline_type', 'mean')],
        
        # Narrow continuous ranges
        attribution_weight=narrow_range(
            center_config.get('attribution_weight', 0.5), relative_range
        ),
        ema_decay=narrow_range(
            center_config.get('ema_decay', 0.9), relative_range, min_val=0.5, max_val=0.999
        ),
        
        # Narrow integer ranges
        attribution_freq=narrow_int_range(
            center_config.get('attribution_freq', 100), relative_range, min_val=10
        ),
        attribution_n_steps=narrow_int_range(
            center_config.get('attribution_n_steps', 10), relative_range, min_val=2
        ),
    )
