import torch
import os
from config import STRATEGY_CONSTS

has_bf16 = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability(0)[0] >= 8  # Ampere+
)

# Calculate batch size dynamically based on world size
# Formula: train_batch_size = micro_batch_size × grad_accum_steps × world_size
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
MICRO_BATCH_SIZE = 1  # Per GPU
GRAD_ACCUM_STEPS = max(
    1, STRATEGY_CONSTS["BATCH_SIZE"] // (MICRO_BATCH_SIZE * WORLD_SIZE))
TRAIN_BATCH_SIZE = MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS * WORLD_SIZE

DS_CONFIG = {
    "train_batch_size": TRAIN_BATCH_SIZE,
    "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
    "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
    "bf16": {"enabled": bool(has_bf16)},
    "fp16": {"enabled": not bool(has_bf16)},
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": False,
        "contiguous_gradients": True
    },
    "activation_checkpointing": {
        "partition_activations": True,
        "contiguous_memory_optimization": True,
        "synchronize_checkpoint_boundary": True,
        "cpu_checkpointing": False
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": STRATEGY_CONSTS["LEARNING_RATE"],
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "torch_adam": True
        }
    },
    "gradient_clipping": 1.0,
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "total_num_steps": STRATEGY_CONSTS["MAX_STEPS"],
            "warmup_min_lr": 0,
            "warmup_max_lr": STRATEGY_CONSTS["LEARNING_RATE"],
            "warmup_num_steps": STRATEGY_CONSTS["WARMUP_STEPS"],
            "warmup_type": "linear"
        }
    }
}
