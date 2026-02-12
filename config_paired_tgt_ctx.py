"""
Configuration for Paired Target-Context cross-view training.

Uses NL-RX-SYNTH dataset: natural language descriptions paired with regex patterns.
Context = Text (NL), Target = Code (Regex). No masking generators needed.
"""

from typing import TypedDict, Optional

from src.builders.dataloader_builder import DatasetType
from src.builders.patcher_builder import PatchStrategyType
from src.builders.encoder_builder import EncoderType
from src.builders.loss_calculator_builder import LossCalculatorType


class PairedTgtCtxConfig(TypedDict, total=False):
    """Type-safe configuration for paired target-context training."""
    TRAINING_DATASET: DatasetType
    PATCH_STRATEGY: PatchStrategyType
    PATCH_SIZE: int
    CONTEXT_ENCODER: EncoderType
    LOSS_CALCULATOR: LossCalculatorType
    CONTEXT_MODEL_ID: str
    DEFAULT_EMA_DECAY: float
    BATCH_SIZE: int
    MAX_STEPS: Optional[int]
    CHECKPOINT_INTERVAL: int
    CHECKPOINT_DIR: str
    LOG_INTERVAL: int
    WARMUP_STEPS: int
    LEARNING_RATE: float
    LAMBDA_NTP: float
    LAMBDA_JEPA: float
    MAX_TEXT_TOKENS: int
    MAX_CODE_TOKENS: int
    EXPERIMENT_TRACKER: str
    WANDB_RUN_NAME: Optional[str]
    DATA_PATH: Optional[str]


PAIRED_TGT_CTX_CONSTS: PairedTgtCtxConfig = {
    "TRAINING_DATASET": "nl_rx_synth",
    "PATCH_STRATEGY": "mean",
    "PATCH_SIZE": 2,
    "CONTEXT_ENCODER": "olmo",
    "LOSS_CALCULATOR": "l2",
    "CONTEXT_MODEL_ID": "/g/data/oy87/cw9909/hf_models/OLMo-2-0425-1B-Instruct",
    "DEFAULT_EMA_DECAY": 0.99,
    "BATCH_SIZE": 4,
    "MAX_STEPS": 10000,
    "CHECKPOINT_INTERVAL": 2000,
    "CHECKPOINT_DIR": "/g/data/oy87/cw9909/llm_jepa/checkpoints/paired_tgt_ctx_nlrx",
    "LOG_INTERVAL": 10,
    "WARMUP_STEPS": 500,
    "LEARNING_RATE": 5e-4,
    "LAMBDA_NTP": 1.0,
    "LAMBDA_JEPA": 1.0,
    "MAX_TEXT_TOKENS": 256,
    "MAX_CODE_TOKENS": 128,
    "EXPERIMENT_TRACKER": "wandb",
    "WANDB_RUN_NAME": None,
    "DATA_PATH": None,
}
