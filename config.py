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
    MASK_STRATEGY: MaskStrategyType
    CONTEXT_STRATEGY: MaskStrategyType
    CONTEXT_ENCODER: EncoderType
    TARGET_PREDICTOR: EncoderType
    LOSS_CALCULATOR: LossCalculatorType
    CONTEXT_MODEL_ID: str
    TARGET_MODEL_ID: str


STRATEGY_CONSTS: StrategyConfig = {
    "TRAINING_DATASET": "dolma_sample",
    "PATCH_STRATEGY": "token",
    "MASK_STRATEGY": "random",
    "CONTEXT_STRATEGY": "random",
    "CONTEXT_ENCODER": "longformer",
    "TARGET_PREDICTOR": "longformer",
    "LOSS_CALCULATOR": "l2",
    "CONTEXT_MODEL_ID": "allenai/longformer-base-4096",
    "TARGET_MODEL_ID": "allenai/longformer-base-4096",
}
