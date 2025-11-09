"""
Configuration for I-JEPA training.

Type hints are imported from builder files for type safety (DRY principle).
"""

from typing import TypedDict

from src.builders.dataloader_builder import DatasetType
from src.builders.patcher_builder import PatchStrategyType
from src.builders.masker_builder import MaskStrategyType
from src.builders.encoder_builder import EncoderType
from src.builders.loss_calculator_builder import LossCalculatorType


class StrategyConfig(TypedDict):
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
    MAX_STEPS: int
    CHECKPOINT_INTERVAL: int
    CHECKPOINT_DIR: str
    LOG_INTERVAL: int
    WARMUP_STEPS: int
    LEARNING_RATE: float
    MAX_SEQ_LENGTH: int


STRATEGY_CONSTS: StrategyConfig = {
    "TRAINING_DATASET": "dolma_sample",
    "PATCH_STRATEGY": "mean",  # "mean" or "max" pooling
    "PATCH_SIZE": 2,  # Number of tokens to aggregate into one patch
    "MASK_STRATEGY": "random",
    "CONTEXT_STRATEGY": "random",
    "CONTEXT_ENCODER": "longformer",
    "TARGET_PREDICTOR": "longformer",
    "LOSS_CALCULATOR": "l2",
    "CONTEXT_MODEL_ID": "allenai/longformer-base-4096",
    "TARGET_MODEL_ID": "allenai/longformer-base-4096",
    "DEFAULT_EMA_DECAY": 0.99,
    "BATCH_SIZE": 2,
    "MAX_STEPS": None,
    "CHECKPOINT_INTERVAL": 10000,
    "CHECKPOINT_DIR": "/g/data/oy87/cw9909/llm_jepa/checkpoints/llm_jepa",
    "LOG_INTERVAL": 10,
    "WARMUP_STEPS": 500,
    "LEARNING_RATE": 5e-4,
    "MAX_SEQ_LENGTH": 4096
}
