STRATEGY_CONSTS = {
    "TRAINING_DATASET": "dolma_sample",
    "PATCH_STRATEGY": "token",
    "MASK_STRATEGY": "random",  # Options: "random", "block"
    "CONTEXT_STRATEGY": "random",  # Options: "random", "block"
    "CONTEXT_ENCODER": "longformer",
    "TARGET_PREDICTOR": "longformer",
    "LOSS_CALCULATOR": "l2",
    "CONTEXT_MODEL_ID": "allenai/longformer-base-4096",
    "TARGET_MODEL_ID": "allenai/longformer-base-4096",
}

# Masking Strategy Options:
# - "random": RandomMasker - Randomly masks a percentage of patches
# - "block": BlockMasker - Masks a contiguous block of patches
