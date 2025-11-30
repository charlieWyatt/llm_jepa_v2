"""
Configuration for I-JEPA training.

Type hints are imported from builder files for type safety (DRY principle).
"""

from typing import TypedDict, Optional

from src.builders.dataloader_builder import DatasetType
from src.builders.patcher_builder import PatchStrategyType
from src.builders.masker_builder import MaskStrategyType
from src.builders.encoder_builder import EncoderType
from src.builders.loss_calculator_builder import LossCalculatorType


class StrategyConfig(TypedDict, total=False):  # total=False allows optional fields
    """Type-safe configuration dictionary."""
    TRAINING_DATASET: DatasetType
    PATCH_STRATEGY: PatchStrategyType
    PATCH_SIZE: int
    MASK_STRATEGY: MaskStrategyType
    CONTEXT_STRATEGY: MaskStrategyType
    CONTEXT_ENCODER: EncoderType
    TARGET_PREDICTOR: EncoderType
    LOSS_CALCULATOR: LossCalculatorType
    CONTEXT_MODEL_ID: str
    TARGET_MODEL_ID: str
    DEFAULT_EMA_DECAY: float
    BATCH_SIZE: int
    MAX_STEPS: Optional[int]
    CHECKPOINT_INTERVAL: int
    CHECKPOINT_DIR: str
    LOG_INTERVAL: int
    WARMUP_STEPS: int
    LEARNING_RATE: float
    MAX_SEQ_LENGTH: int
    
    # NEW: Masking hyperparameters
    CONTEXT_MASK_RATIO: float  # For random masking (context) OR block size ratio
    TARGET_MASK_RATIO: float   # For random masking (targets) OR block size ratio
    NUM_TARGETS: int           # Number of target blocks to predict

    EXPERIMENT_TRACKER: str



STRATEGY_CONSTS: StrategyConfig = {
    "TRAINING_DATASET": "dolma_sample",
    "PATCH_STRATEGY": "mean",
    "PATCH_SIZE": 2,
    
    # Masking strategies
    "MASK_STRATEGY": "block",      # "random" or "block"
    "CONTEXT_STRATEGY": "block",   # "random" or "block"
    
    # Masking hyperparameters (unified as ratios)
    "CONTEXT_MASK_RATIO": 0.6,     # If block: 60% of sequence length
                                    # If random: 60% of tokens masked
    "TARGET_MASK_RATIO": 0.15,     # If block: 15% of sequence length
                                    # If random: 15% of tokens masked
    "NUM_TARGETS": 1,              # Number of target blocks
    
    "CONTEXT_ENCODER": "longformer",
    "TARGET_PREDICTOR": "longformer",
    "LOSS_CALCULATOR": "l2",
    "CONTEXT_MODEL_ID": "allenai/longformer-base-4096",
    "TARGET_MODEL_ID": "allenai/longformer-base-4096",
    "DEFAULT_EMA_DECAY": 0.99,
    "BATCH_SIZE": 2,
    "MAX_STEPS": 5000,
    "CHECKPOINT_INTERVAL": 10000,
    "CHECKPOINT_DIR": "/g/data/oy87/cw9909/llm_jepa/checkpoints/llm_jepa",
    "LOG_INTERVAL": 10,
    "WARMUP_STEPS": 500,
    "LEARNING_RATE": 5e-4,
    "MAX_SEQ_LENGTH": 4096,
    "EXPERIMENT_TRACKER": 'wandb'
}
