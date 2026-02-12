"""
Configuration for self-supervised I-JEPA baseline on NL-RX-Synth.

Uses the flat (concatenated NL + regex) dataloader with standard I-JEPA
block masking. Hyperparameters match the paired target-context model
for a fair comparison.
"""

from config import StrategyConfig

STRATEGY_CONSTS: StrategyConfig = {
    "TRAINING_DATASET": "nl_rx_synth_flat",
    "PATCH_STRATEGY": "mean",
    "PATCH_SIZE": 2,

    # Masking strategies
    "MASK_STRATEGY": "block",
    "CONTEXT_STRATEGY": "block",
    "CONTEXT_MASK_RATIO": 0.6,
    "TARGET_MASK_RATIO": 0.15,
    "NUM_TARGETS": 1,

    "CONTEXT_ENCODER": "olmo",
    "TARGET_PREDICTOR": "olmo",
    "LOSS_CALCULATOR": "l2",
    "CONTEXT_MODEL_ID": "/g/data/oy87/cw9909/hf_models/OLMo-2-0425-1B-Instruct",
    "TARGET_MODEL_ID": "/g/data/oy87/cw9909/hf_models/OLMo-2-0425-1B-Instruct",
    "DEFAULT_EMA_DECAY": 0.99,
    "BATCH_SIZE": 4,
    "MAX_STEPS": 10000,
    "CHECKPOINT_INTERVAL": 2000,
    "CHECKPOINT_DIR": "/g/data/oy87/cw9909/llm_jepa/checkpoints/self_sup_baseline_nlrx",
    "LOG_INTERVAL": 10,
    "WARMUP_STEPS": 500,
    "LEARNING_RATE": 5e-4,
    "LAMBDA_NTP": 1.0,
    "LAMBDA_JEPA": 1.0,
    "EXPERIMENT_TRACKER": "wandb",
    "WANDB_RUN_NAME": None,
    "DATA_PATH": None,
}
