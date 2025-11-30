"""
Experiment Tracking Abstraction

Provides a clean interface for tracking experiments (metrics, hyperparameters,
artifacts) that can be swapped between different backends (TensorBoard, 
Weights & Biases, MLflow, etc.)

NOTE: This is different from application logging (Logger class in logging_helpers.py).
      - ExperimentTracker: Records metrics, hyperparameters, experiment results
      - Logger: Records program execution, debugging, error messages
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
from accelerate import Accelerator


class ExperimentTracker(ABC):
    """
    Abstract base class for experiment tracking.
    
    Defines the interface that all tracking backends must implement.
    This allows easy swapping between TensorBoard, WandB, MLflow, etc.
    
    Usage:
        tracker = create_tracker('tensorboard', ...)
        tracker.initialize('my_experiment', config={...})
        tracker.log_metrics({'loss': 0.5}, step=100)
        tracker.finalize()
    """
    
    @abstractmethod
    def initialize(self, project_name: str, config: Dict[str, Any], **kwargs) -> None:
        """
        Initialize the tracking backend.
        
        Args:
            project_name: Name of the experiment/project
            config: Configuration dictionary to track
            **kwargs: Additional backend-specific arguments (e.g., run_name, tags)
        """
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """
        Track scalar metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Training step number
        """
        pass
    
    @abstractmethod
    def log_text(self, tag: str, text: str, step: int) -> None:
        """
        Track text/markdown content.
        
        Args:
            tag: Tag/name for the text
            text: Text content (can be markdown)
            step: Training step number
        """
        pass
    
    @abstractmethod
    def finalize(self) -> None:
        """Clean up and finalize tracking."""
        pass


class TensorBoardTracker(ExperimentTracker):
    """
    TensorBoard implementation of ExperimentTracker.
    
    Wraps Accelerator's TensorBoard tracking functionality.
    """
    
    def __init__(self, accelerator: Accelerator, log_dir: str):
        """
        Initialize TensorBoard tracker.
        
        Args:
            accelerator: Accelerator instance for distributed tracking
            log_dir: Directory for TensorBoard logs
        """
        self.accelerator = accelerator
        self.log_dir = log_dir
        self._initialized = False
    
    def initialize(self, project_name: str, config: Dict[str, Any], **kwargs) -> None:
        """Initialize TensorBoard tracking."""
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        self.accelerator.init_trackers(
            project_name=project_name,
            config=config,
        )
        self._initialized = True
    
    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Track metrics in TensorBoard."""
        if not self._initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() first.")
        
        self.accelerator.log(metrics, step=step)
    
    def log_text(self, tag: str, text: str, step: int) -> None:
        """Track text in TensorBoard."""
        if not self._initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() first.")
        
        # TensorBoard text tracking via accelerator
        self.accelerator.log({tag: text}, step=step)
    
    def finalize(self) -> None:
        """End TensorBoard tracking."""
        if self._initialized:
            self.accelerator.end_training()
            self._initialized = False


class NoOpTracker(ExperimentTracker):
    """
    No-operation tracker that does nothing.
    
    Useful for debugging or when tracking is disabled.
    """
    
    def initialize(self, project_name: str, config: Dict[str, Any]) -> None:
        """No-op initialization."""
        pass
    
    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """No-op metrics tracking."""
        pass
    
    def log_text(self, tag: str, text: str, step: int) -> None:
        """No-op text tracking."""
        pass
    
    def finalize(self) -> None:
        """No-op finalization."""
        pass

import wandb
from pathlib import Path
from typing import Dict, Any


class WandBTracker(ExperimentTracker):
    """
    Weights & Biases implementation of ExperimentTracker.

    IMPORTANT:
    - Runs in OFFLINE mode for HPC clusters like Gadi
    - No internet access required
    - Stores everything locally under wandb_dir
    """

    def __init__(self, wandb_dir: str):
        """
        Args:
            wandb_dir: Directory where WandB offline logs will be stored
        """
        self.wandb_dir = Path(wandb_dir)
        self.wandb_dir.mkdir(parents=True, exist_ok=True)
        self.run = None

    def initialize(self, project_name: str, config: Dict[str, Any], **kwargs) -> None:
        """
        Initialize WandB in offline mode.
        
        Args:
            project_name: Name of the WandB project
            config: Configuration dictionary to log
            **kwargs: Additional arguments passed to wandb.init (e.g., run_name, tags, etc.)
        """
        # Extract run_name if provided, otherwise use default
        run_name = kwargs.pop('run_name', None)
        
        self.run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            dir=str(self.wandb_dir),
            mode="offline",
            resume="allow",
            **kwargs
        )
        wandb.config.update(config, allow_val_change=True)

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log scalar metrics."""
        if self.run is None:
            raise RuntimeError("Tracker not initialized. Call initialize() first.")

        wandb.log(metrics, step=step, commit=True)
    
    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log text or markdown."""
        if self.run is None:
            raise RuntimeError("Tracker not initialized. Call initialize() first.")

        wandb.log({tag: wandb.Html(text)}, step=step)


    def finalize(self) -> None:
        """Finish the WandB run."""
        if self.run is not None:
            self.run.finish()
            self.run = None


# ============================================
# Tracker Factory
# ============================================

def create_tracker(
    backend: str,
    accelerator: Optional[Accelerator] = None,
    log_dir: Optional[str] = None,
) -> ExperimentTracker:

    if backend == 'tensorboard':
        if accelerator is None:
            raise ValueError("accelerator required for TensorBoard tracker")
        if log_dir is None:
            raise ValueError("log_dir required for TensorBoard tracker")
        return TensorBoardTracker(accelerator, log_dir)

    elif backend == 'wandb':
        if log_dir is None:
            raise ValueError("log_dir required for WandB tracker")
        return WandBTracker(log_dir)

    elif backend == 'noop':
        return NoOpTracker()

    else:
        raise ValueError(f"Unknown tracker backend: {backend}")