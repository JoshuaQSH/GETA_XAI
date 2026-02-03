"""
Main autotuning class for XAI-GETA hyperparameter optimization.

Supports:
- Nevergrad: Evolutionary strategies (CMA-ES, DE, etc.)
- Optuna: Bayesian optimization with TPE sampler
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .search_space import XAISearchSpace, get_default_search_space
from .objectives import create_objective, MultiObjective

logger = logging.getLogger(__name__)


@dataclass
class AutotuneConfig:
    """Configuration for autotuning."""
    
    # Optimization settings
    budget: int = 50  # Number of trials
    optimizer_backend: str = 'nevergrad'  # 'nevergrad' or 'optuna'
    nevergrad_algorithm: str = 'NGOpt'  # Nevergrad optimizer: NGOpt, CMA, DE, PSO
    
    # Training settings (per trial)
    epochs_per_trial: int = 10
    early_stopping_patience: int = 3
    
    # Hardware
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Logging
    log_dir: Optional[str] = None
    verbose: bool = True
    save_checkpoints: bool = False


class AutotuneXAI:
    """
    Automated hyperparameter tuning for XAI-GETA.
    
    This class provides a high-level interface for optimizing XAI hyperparameters
    using either Nevergrad (evolutionary) or Optuna (Bayesian) optimization.
    
    Example:
        ```python
        autotuner = AutotuneXAI(
            model_factory=lambda: VGG7_BN(),
            train_loader=train_loader,
            val_loader=val_loader,
            search_space=XAISearchSpace(
                attribution_weight=(0.1, 0.9),
                attribution_freq=(50, 300),
            ),
            objective='multi',
            budget=30,
        )
        
        best_config, best_score = autotuner.run()
        ```
    
    Attributes:
        model_factory: Callable that creates a fresh model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        search_space: XAISearchSpace defining hyperparameter ranges
        objective: Objective function or string name
        config: AutotuneConfig with optimization settings
    """
    
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        train_loader: DataLoader,
        val_loader: DataLoader,
        search_space: Optional[XAISearchSpace] = None,
        objective: Union[str, Callable] = 'multi',
        config: Optional[AutotuneConfig] = None,
        # Shorthand config options
        budget: Optional[int] = None,
        epochs_per_trial: Optional[int] = None,
        optimizer_backend: Optional[str] = None,
        device: Optional[str] = None,
        verbose: bool = True,
        # Base GETA config (non-tuned parameters)
        base_geta_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the autotuner.
        
        Args:
            model_factory: Callable that returns a fresh model instance
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            search_space: XAISearchSpace (default: get_default_search_space())
            objective: Objective function or name ('accuracy', 'smoothness', 'multi')
            config: Full AutotuneConfig (overrides shorthand options)
            budget: Number of optimization trials
            epochs_per_trial: Training epochs per trial
            optimizer_backend: 'nevergrad' or 'optuna'
            device: Device to use ('cuda' or 'cpu')
            verbose: Print progress
            base_geta_config: Base GETA config (non-tuned parameters)
        """
        self.model_factory = model_factory
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.search_space = search_space or get_default_search_space()
        
        # Setup objective
        if isinstance(objective, str):
            self.objective = create_objective(objective)
        else:
            self.objective = objective
        
        # Setup config
        self.config = config or AutotuneConfig()
        if budget is not None:
            self.config.budget = budget
        if epochs_per_trial is not None:
            self.config.epochs_per_trial = epochs_per_trial
        if optimizer_backend is not None:
            self.config.optimizer_backend = optimizer_backend
        if device is not None:
            self.config.device = device
        self.config.verbose = verbose
        
        # Base GETA configuration (parameters not being tuned)
        self.base_geta_config = base_geta_config or {
            'variant': 'sgd',
            'lr': 0.01,
            'weight_decay': 1e-4,
            'target_group_sparsity': 0.5,
            'start_pruning_step': 1000,
        }
        
        # Results storage
        self.trial_results: List[Dict[str, Any]] = []
        self.best_config: Optional[Dict[str, Any]] = None
        self.best_score: float = float('inf')
        
        # Setup logging
        if self.config.log_dir:
            Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
    
    def _train_and_evaluate(
        self,
        xai_config: Dict[str, Any],
        trial_id: int = 0,
    ) -> Dict[str, Any]:
        """
        Train model with given XAI config and return metrics.
        
        Args:
            xai_config: XAI hyperparameters to use
            trial_id: Trial identifier for logging
        
        Returns:
            Dictionary with metrics:
                - final_accuracy: Validation accuracy
                - final_loss: Final validation loss
                - loss_history: Training loss history
                - final_sparsity: Achieved group sparsity
        """
        # Import here to avoid circular imports
        from only_train_once import OTO
        from only_train_once.xai_optimizer import XAI_GETA
        
        # Create fresh model
        model = self.model_factory()
        model = model.to(self.config.device)
        
        # Get dummy input shape from train loader
        for batch in self.train_loader:
            if isinstance(batch, dict):
                dummy_input = batch['pixel_values'][:1].to(self.config.device)
            else:
                dummy_input = batch[0][:1].to(self.config.device)
            break
        
        # Initialize OTO
        oto = OTO(model=model, dummy_input=dummy_input)
        
        # Merge base config with XAI config
        geta_config = {**self.base_geta_config}
        
        # Extract XAI-specific params
        attribution_method = xai_config.get('attribution_method', 'saliency')
        attribution_weight = xai_config.get('attribution_weight', 0.5)
        ema_decay = xai_config.get('ema_decay', 0.9)
        attribution_n_steps = xai_config.get('attribution_n_steps', 10)
        compute_attribution_freq = xai_config.get('attribution_freq', 100)
        baseline_type = xai_config.get('baseline_type', 'mean')
        
        # Override learning rate if in search space
        if 'learning_rate' in xai_config:
            geta_config['lr'] = xai_config['learning_rate']
        if 'target_group_sparsity' in xai_config:
            geta_config['target_group_sparsity'] = xai_config['target_group_sparsity']
        
        # Calculate steps based on data
        steps_per_epoch = len(self.train_loader)
        
        # Create XAI-GETA optimizer (instantiate directly, not through oto.xai_geta)
        try:
            param_groups = oto._graph.get_param_groups()
            
            optimizer = XAI_GETA(
                params=param_groups,
                model=model,  # Pass model for Captum attribution
                variant=geta_config.get('variant', 'sgd'),
                lr=geta_config.get('lr', 0.01),
                weight_decay=geta_config.get('weight_decay', 1e-4),
                target_group_sparsity=geta_config.get('target_group_sparsity', 0.5),
                start_projection_step=0,
                projection_periods=2,
                projection_steps=2 * steps_per_epoch,
                start_pruning_step=geta_config.get('start_pruning_step', 2 * steps_per_epoch),
                pruning_periods=2,
                pruning_steps=2 * steps_per_epoch,
                device=self.config.device,
                # XAI params
                attribution_method=attribution_method,
                attribution_weight=attribution_weight,
                ema_decay=ema_decay,
                attribution_n_steps=attribution_n_steps,
                compute_attribution_freq=compute_attribution_freq,
            )
            
            # Store optimizer in OTO for later use
            oto._optimizer = optimizer
            
        except Exception as e:
            logger.warning(f"Trial {trial_id} failed to create optimizer: {e}")
            return {
                'final_accuracy': 0.0,
                'final_loss': float('inf'),
                'loss_history': [],
                'final_sparsity': 0.0,
                'error': str(e),
            }
        
        # Training
        criterion = nn.CrossEntropyLoss()
        loss_history = []
        best_val_acc = 0.0
        patience_counter = 0
        
        steps_per_epoch = len(self.train_loader)
        total_steps = self.config.epochs_per_trial * steps_per_epoch
        
        model.train()
        global_step = 0
        
        for epoch in range(self.config.epochs_per_trial):
            epoch_losses = []
            
            for batch_idx, batch in enumerate(self.train_loader):
                if isinstance(batch, dict):
                    inputs = batch['pixel_values'].to(self.config.device)
                    targets = batch['labels'].to(self.config.device)
                else:
                    inputs, targets = batch
                    inputs = inputs.to(self.config.device)
                    targets = targets.to(self.config.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # XAI-GETA step
                optimizer.step(inputs=inputs, targets=targets)
                
                epoch_losses.append(loss.item())
                loss_history.append(loss.item())
                global_step += 1
            
            # Validation
            val_acc = self._evaluate(model)
            
            if self.config.verbose:
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                logger.info(
                    f"Trial {trial_id} Epoch {epoch+1}/{self.config.epochs_per_trial}: "
                    f"Loss={avg_loss:.4f}, Val Acc={val_acc:.2f}%"
                )
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Trial {trial_id}: Early stopping at epoch {epoch+1}")
                    break
        
        # Get final sparsity
        try:
            sparsity = oto.compute_sparsity()
        except:
            sparsity = 0.0
        
        return {
            'final_accuracy': best_val_acc,
            'final_loss': loss_history[-1] if loss_history else float('inf'),
            'loss_history': loss_history,
            'final_sparsity': sparsity,
        }
    
    def _evaluate(self, model: nn.Module) -> float:
        """Evaluate model on validation set."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, dict):
                    inputs = batch['pixel_values'].to(self.config.device)
                    targets = batch['labels'].to(self.config.device)
                else:
                    inputs, targets = batch
                    inputs = inputs.to(self.config.device)
                    targets = targets.to(self.config.device)
                
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        model.train()
        return 100.0 * correct / total
    
    def run_nevergrad(self) -> Tuple[Dict[str, Any], float]:
        """Run optimization using Nevergrad."""
        try:
            import nevergrad as ng
        except ImportError:
            raise ImportError("Nevergrad not installed. Install with: uv add nevergrad")
        
        # Create parametrization
        parametrization = self.search_space.to_nevergrad()
        
        # Create optimizer
        optimizer_cls = getattr(ng.optimizers, self.config.nevergrad_algorithm)
        ng_optimizer = optimizer_cls(
            parametrization=parametrization,
            budget=self.config.budget,
        )
        
        logger.info(f"Starting Nevergrad optimization with {self.config.budget} trials")
        
        for trial_id in range(self.config.budget):
            # Get next candidate
            candidate = ng_optimizer.ask()
            xai_config = dict(candidate.kwargs)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Trial {trial_id + 1}/{self.config.budget}")
            logger.info(f"Config: {xai_config}")
            
            # Train and evaluate
            start_time = time.time()
            metrics = self._train_and_evaluate(xai_config, trial_id)
            elapsed = time.time() - start_time
            
            # Compute objective
            score = self.objective(metrics)
            
            # Tell optimizer
            ng_optimizer.tell(candidate, score)
            
            # Store results
            result = {
                'trial_id': trial_id,
                'config': xai_config,
                'metrics': metrics,
                'score': score,
                'elapsed_seconds': elapsed,
            }
            self.trial_results.append(result)
            
            # Update best
            if score < self.best_score:
                self.best_score = score
                self.best_config = xai_config.copy()
                logger.info(f"★ New best! Score: {score:.4f}")
            
            logger.info(
                f"Trial {trial_id + 1}: Score={score:.4f}, "
                f"Acc={metrics['final_accuracy']:.2f}%, "
                f"Time={elapsed:.1f}s"
            )
        
        # Get final recommendation
        recommendation = ng_optimizer.provide_recommendation()
        final_config = dict(recommendation.kwargs)
        
        logger.info(f"\n{'='*60}")
        logger.info("Optimization complete!")
        logger.info(f"Best config: {self.best_config}")
        logger.info(f"Best score: {self.best_score:.4f}")
        
        return self.best_config, self.best_score
    
    def run_optuna(self) -> Tuple[Dict[str, Any], float]:
        """Run optimization using Optuna."""
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna not installed. Install with: uv add optuna")
        
        def objective_fn(trial):
            xai_config = self.search_space.to_optuna(trial)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Trial {trial.number + 1}/{self.config.budget}")
            logger.info(f"Config: {xai_config}")
            
            # Train and evaluate
            start_time = time.time()
            metrics = self._train_and_evaluate(xai_config, trial.number)
            elapsed = time.time() - start_time
            
            # Compute objective
            score = self.objective(metrics)
            
            # Store results
            result = {
                'trial_id': trial.number,
                'config': xai_config,
                'metrics': metrics,
                'score': score,
                'elapsed_seconds': elapsed,
            }
            self.trial_results.append(result)
            
            # Report intermediate value for pruning
            trial.report(score, step=0)
            
            logger.info(
                f"Trial {trial.number + 1}: Score={score:.4f}, "
                f"Acc={metrics['final_accuracy']:.2f}%, "
                f"Time={elapsed:.1f}s"
            )
            
            return score
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        
        logger.info(f"Starting Optuna optimization with {self.config.budget} trials")
        
        # Optimize
        study.optimize(
            objective_fn,
            n_trials=self.config.budget,
            show_progress_bar=self.config.verbose,
        )
        
        # Get best
        self.best_config = study.best_params
        self.best_score = study.best_value
        
        logger.info(f"\n{'='*60}")
        logger.info("Optimization complete!")
        logger.info(f"Best config: {self.best_config}")
        logger.info(f"Best score: {self.best_score:.4f}")
        
        return self.best_config, self.best_score
    
    def run(self) -> Tuple[Dict[str, Any], float]:
        """
        Run hyperparameter optimization.
        
        Returns:
            Tuple of (best_config, best_score)
        """
        if self.config.optimizer_backend == 'nevergrad':
            return self.run_nevergrad()
        elif self.config.optimizer_backend == 'optuna':
            return self.run_optuna()
        else:
            raise ValueError(f"Unknown optimizer backend: {self.config.optimizer_backend}")
    
    def save_results(self, filepath: str):
        """Save all trial results to JSON file."""
        # Convert non-serializable types
        results = []
        for r in self.trial_results:
            result = {
                'trial_id': r['trial_id'],
                'config': r['config'],
                'metrics': {
                    'final_accuracy': r['metrics']['final_accuracy'],
                    'final_loss': r['metrics']['final_loss'],
                    'final_sparsity': r['metrics'].get('final_sparsity', 0.0),
                },
                'score': r['score'],
                'elapsed_seconds': r['elapsed_seconds'],
            }
            results.append(result)
        
        output = {
            'best_config': self.best_config,
            'best_score': self.best_score,
            'trials': results,
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def get_best_config_yaml(self) -> str:
        """Get best config as YAML string."""
        if self.best_config is None:
            raise ValueError("No best config found. Run optimization first.")
        
        yaml_str = "xai:\n"
        for key, value in self.best_config.items():
            if isinstance(value, str):
                yaml_str += f"  {key}: '{value}'\n"
            else:
                yaml_str += f"  {key}: {value}\n"
        
        return yaml_str
