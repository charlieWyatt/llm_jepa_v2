import torch
from config import STRATEGY_CONSTS

has_bf16 = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability(0)[0] >= 8  # Ampere+
)

DS_CONFIG = {
    "train_batch_size": STRATEGY_CONSTS["BATCH_SIZE"],
    "train_micro_batch_size_per_gpu": 1,   # ↓ from 2
    "gradient_accumulation_steps": 1,      # keep global batch size similar
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
            "lr": 5e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "torch_adam": True
        }
    },
    "gradient_clipping": 1.0
}
