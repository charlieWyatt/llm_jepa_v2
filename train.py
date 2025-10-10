# train.py
from src.maskers.block_mask_generator import BlockMaskGenerator
from src.maskers.random_mask_generator import RandomMaskGenerator
import sys
import os
import logging
import torch
import torch.nn as nn
import deepspeed

from src.builders.dataloader_builder import dataloader_builder
from src.builders.patcher_builder import patcher_builder
from src.builders.encoder_builder import encoder_builder
from src.encoders.target_encoders.ema_target_encoder import ema_target_encoder
from src.builders.loss_calculator_builder import loss_calculator_builder
from src.maskers.context_target_creator import ContextTargetCreator
from config import STRATEGY_CONSTS

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ----------------------------
# DeepSpeed distributed init
# ----------------------------


def init_dist():
    if torch.cuda.is_available():
        try:
            deepspeed.init_distributed(dist_backend="nccl")
        except Exception:
            pass
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size > 1, rank, local_rank, world_size


IS_DIST, RANK, LOCAL_RANK, WORLD_SIZE = init_dist()


def logi(msg):
    if RANK == 0:
        logger.info(msg)


logi("Starting training pipeline setup...")
logi(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(
        f"GPU[{LOCAL_RANK}]: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# ----------------------------
# Strategy config (unchanged)
# ----------------------------
training_dataset = STRATEGY_CONSTS["TRAINING_DATASET"]
patch_strategy = STRATEGY_CONSTS["PATCH_STRATEGY"]
target_mask_strategy = STRATEGY_CONSTS["MASK_STRATEGY"]
context_mask_strategy = STRATEGY_CONSTS["CONTEXT_STRATEGY"]
context_encoder_type = STRATEGY_CONSTS["CONTEXT_ENCODER"]
target_predictor_type = STRATEGY_CONSTS["TARGET_PREDICTOR"]
loss_calculator_type = STRATEGY_CONSTS["LOSS_CALCULATOR"]

logi(f"Training dataset: {training_dataset}")
logi(f"Patch strategy: {patch_strategy}")
logi(f"Context encoder type: {context_encoder_type}")

DEFAULT_EMA_DECAY = 0.99
BATCH_SIZE = 2

CONTEXT_ENCODER_CONFIG = {
    "hidden_size": 384,
    "num_layers": 6,
    "attention_window": 128,
}
TARGET_ENCODER_CONFIG = {
    "hidden_size": 384,
    "num_layers": 2,
    "attention_window": 256,
}

# ----------------------------
# DeepSpeed config (ZeRO-3)
# bf16 if available, else fp16.
# ----------------------------
has_bf16 = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability(0)[0] >= 8  # Ampere+
)

DS_CONFIG = {
    "train_batch_size": BATCH_SIZE,
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

# ----------------------------
# Build components
# ----------------------------
logi("Building components...")
loss_calculator = loss_calculator_builder(loss_calculator_type).build()

# Create mask generators for context and targets
# Context generator (what the model can see)
if context_mask_strategy == "random":
    context_generator = RandomMaskGenerator(mask_ratio=0.6)  # 60% visible
elif context_mask_strategy == "block":
    context_generator = BlockMaskGenerator(
        span_length=50)  # Large visible block
else:
    raise ValueError(f"Unknown context strategy: {context_mask_strategy}")

# Target generator (what to predict)
if target_mask_strategy == "random":
    target_generator = RandomMaskGenerator(mask_ratio=0.15)  # 15% to predict
elif target_mask_strategy == "block":
    target_generator = BlockMaskGenerator(span_length=10)  # 10-token block
else:
    raise ValueError(f"Unknown target strategy: {target_mask_strategy}")

# Create context-target creator for JEPA-style training
context_target_creator = ContextTargetCreator(
    context_generator=context_generator,
    target_generator=target_generator,
    num_targets=1  # Single target for now (can increase for multi-target JEPA)
)

# Context encoder (trainable)
context_encoder = encoder_builder(context_encoder_type).build(
    model_id=STRATEGY_CONSTS["CONTEXT_MODEL_ID"],
    config=CONTEXT_ENCODER_CONFIG,
)
tokenizer = context_encoder.tokenizer

# EMA target encoder (non-trainable, updated via .update())
target_encoder = ema_target_encoder(context_encoder, DEFAULT_EMA_DECAY)
target_encoder.eval()                 # disable dropout etc.
for p in target_encoder.parameters():  # belt & suspenders (already in your class)
    p.requires_grad = False

# Patcher & loader
patcher = patcher_builder(patch_strategy).build(
    context_encoder, target_encoder)
dataloader = dataloader_builder(training_dataset).build(
    patcher, batch_size=BATCH_SIZE)

# Target predictor (trainable)
target_predictor = encoder_builder(target_predictor_type).build(
    model_id=STRATEGY_CONSTS["TARGET_MODEL_ID"],
    config=TARGET_ENCODER_CONFIG,
)

# ----------------------------
# Create a simple predictor head if target_predictor expects embeddings
# ----------------------------


class PredictorHead(nn.Module):
    """Simple MLP head for predicting target representations from combined embeddings"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------
# Wrap trainables in a single module for DeepSpeed
# ----------------------------


class SimplePredictor(nn.Module):
    """
    Simple predictor that uses full context to predict target representations.

    Uses mean-pooling of context to predict each target position.
    In a more sophisticated version, you'd use cross-attention from targets to context.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, context_repr, target_mask):
        """
        Args:
            context_repr: [B, L, D] - context representations (targets zeroed out)
            target_mask: [B, L] - 1 where to predict

        Returns:
            predictions: [B, num_targets, D]
        """
        B, L, D = context_repr.shape
        device = context_repr.device

        # Find max number of targets
        num_targets = int(target_mask.sum(dim=1).max().item())

        if num_targets == 0:
            return torch.zeros(B, 1, D, device=device)

        predictions = torch.zeros(B, num_targets, D, device=device)

        # For each sample, use full context to predict at target positions
        for i in range(B):
            target_indices = torch.where(target_mask[i] == 1)[0]
            num_tgt = len(target_indices)

            if num_tgt > 0:
                # Mean-pool over context (non-zero positions)
                # context_repr[i] has zeros at target positions
                # Sum over context positions and divide by number of context positions
                context_sum = context_repr[i].sum(dim=0)  # [D]
                num_context = (context_repr[i].abs().sum(
                    dim=-1) > 0).sum()  # Count non-zero positions

                if num_context > 0:
                    context_mean = context_sum / num_context  # [D]
                else:
                    context_mean = torch.zeros(D, device=device)

                # Predict same representation for all targets (simple baseline)
                # In practice, you'd want position-specific predictions
                pred = self.proj(context_mean.unsqueeze(0))  # [1, D]
                predictions[i, :num_tgt] = pred.expand(
                    num_tgt, -1)  # [num_tgt, D]

        return predictions


class TrainableJEPA(nn.Module):
    """
    Correct JEPA implementation for text.

    Key: Model encodes FULL sequence with normal attention, but we zero out
    target regions from the representations, then predict what should be there.
    """

    def __init__(self, context_encoder, predictor):
        super().__init__()
        self.context_encoder = context_encoder
        self.predictor = predictor

    def forward(self, input_ids, attention_mask, context_mask, target_mask):
        """
        JEPA forward pass.

        Args:
            input_ids: [B, L] - Full tokenized sequence
            attention_mask: [B, L] - 1 for real tokens, 0 for padding
            context_mask: [B, L] - 1 for visible positions, 0 for masked
            target_mask: [B, L] - 1 for positions to predict, 0 otherwise

        Returns:
            predicted_targets: [B, num_targets, D] - Predicted representations
        """
        # Encode full sequence with normal attention (respects padding)
        encoder_output = self.context_encoder.model(
            input_ids=input_ids,
            attention_mask=attention_mask  # Only masks padding, not targets!
        )
        full_repr = encoder_output.last_hidden_state  # [B, L, D]

        # Zero out target regions (model shouldn't use info from targets)
        context_repr = full_repr * context_mask.unsqueeze(-1)  # [B, L, D]

        # Predict at target positions
        predicted_targets = self.predictor(context_repr, target_mask)

        return predicted_targets


# Create the predictor and trainable model
hidden_size = CONTEXT_ENCODER_CONFIG["hidden_size"]
predictor = SimplePredictor(hidden_dim=hidden_size)

trainable = TrainableJEPA(
    context_encoder=context_encoder,
    predictor=predictor
)

# ----------------------------
# DeepSpeed initialize (ZeRO-3)
# ----------------------------
model_parameters = [p for p in trainable.parameters() if p.requires_grad]


def _sum_params_module(module, trainable_only=False):
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


pre_total_all = _sum_params_module(trainable, trainable_only=False)
pre_total_trn = _sum_params_module(trainable, trainable_only=True)
pre_global_numels = {name: p.numel()
                     for name, p in trainable.named_parameters()}

if RANK == 0:
    logi(f"[pre-shard] total params (all): {pre_total_all:,}")
    logi(f"[pre-shard] total params (trainable): {pre_total_trn:,}")

engine, optimizer, _, _ = deepspeed.initialize(
    model=trainable,
    model_parameters=model_parameters,  # type: ignore[arg-type]
    config=DS_CONFIG,
)

# Verify ZeRO-3 really active
try:
    zero_on = engine.zero_optimization()
    zero_stage = engine.zero_optimization_stage()
except Exception:
    zero_on = False
    zero_stage = None

logi(
    f"DeepSpeed ZeRO enabled: {zero_on}, stage: {zero_stage}, dp_world_size: {getattr(engine, 'dp_world_size', WORLD_SIZE)}")
assert zero_on and zero_stage == 3, f"Expected ZeRO stage 3, got: {zero_stage}"

# ----------------------------
# Device plumbing
# ----------------------------
device = engine.device if torch.cuda.is_available() else torch.device("cpu")


def to_device(x):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_device(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v) for v in x)
    return x


if hasattr(loss_calculator, "to"):
    loss_calculator = loss_calculator.to(device)
if hasattr(target_encoder, "to"):
    target_encoder = target_encoder.to(device)

torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ----------------------------
# Parameter shard stats (after sharding)
# ----------------------------


def _local_shard_numel(engine, module):
    """Count *this rank's* parameter shard numel. Uses DeepSpeed ds_tensor if present."""
    local = 0
    for p in module.parameters():
        ds_tensor = getattr(p, "ds_tensor", None)
        if ds_tensor is not None:
            local += int(ds_tensor.numel())
        else:
            local += int(p.numel())
    return local


def _local_shard_bytes(engine, module):
    """Exact bytes for this rank's param shards (accounts for mixed dtypes)."""
    total_bytes = 0
    for p in module.parameters():
        ds_tensor = getattr(p, "ds_tensor", None)
        n = int(ds_tensor.numel()) if ds_tensor is not None else int(p.numel())
        total_bytes += n * p.element_size()
    return total_bytes


def _all_gather_i64(value, device):
    """All-gather a single int64 from each rank; returns Python ints."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        t = torch.tensor([int(value)], dtype=torch.long, device=device)
        outs = [torch.zeros_like(t) for _ in range(
            torch.distributed.get_world_size())]
        torch.distributed.all_gather(outs, t)
        return [int(x.item()) for x in outs]
    return [int(value)]


def _fmt_gb(bytes_val):
    return f"{bytes_val / (1024**3):.2f} GB"


def log_param_distribution_and_size(engine, device, rank, world_size):
    """Log global param count + each GPU's share (%) and bytes of its shard."""
    local_numel = _local_shard_numel(engine, engine.module)
    local_bytes = _local_shard_bytes(engine, engine.module)

    shard_numels = _all_gather_i64(local_numel, device)
    shard_bytes = _all_gather_i64(local_bytes, device)

    global_numel = sum(shard_numels)
    global_bytes = sum(shard_bytes)

    if rank == 0:
        logger.info(
            f"[post-shard] global params: {global_numel:,}  (~{_fmt_gb(global_bytes)} across all shards)")
        for r in range(len(shard_numels)):
            pct = 100.0 * shard_numels[r] / max(global_numel, 1)
            logger.info(f"GPU {r+1} (rank {r}): ~{shard_numels[r]:,} params "
                        f"({pct:.2f}%), shard size ≈ {_fmt_gb(shard_bytes[r])}")


def log_gpu_memory_snapshot(tag, rank):
    """Log per-GPU memory (global device view) + this process's PyTorch alloc/reserved."""
    if rank != 0 or not torch.cuda.is_available():
        return
    logger.info(f"[{tag}] GPU memory snapshot (pre-data)")
    for i in range(torch.cuda.device_count()):
        # Global view from CUDA driver:
        free_b, total_b = torch.cuda.mem_get_info(i)
        used_b = total_b - free_b
        # PyTorch view (this process only):
        alloc_b = torch.cuda.memory_allocated(i)
        reserv_b = torch.cuda.memory_reserved(i)
        logger.info(
            f"  GPU {i+1}: used { _fmt_gb(used_b) } / total { _fmt_gb(total_b) } | "
            f"PyTorch alloc { _fmt_gb(alloc_b) }, reserved { _fmt_gb(reserv_b) }"
        )


def log_first_local_params(engine, rank, max_items=10):
    """
    Print the first `max_items` parameters *sharded to this rank*.
    Uses DeepSpeed's per-param ds_tensor to report the local shard size.
    """
    shown = 0
    for name, p in engine.module.named_parameters():
        ds_tensor = getattr(p, "ds_tensor", None)
        local_numel = int(ds_tensor.numel()) if ds_tensor is not None else 0
        if local_numel > 0:
            g = pre_global_numels.get(name, 0)
            logger.info(
                f"[rank {rank}] param {shown+1}: {name} | global={g:,} | "
                f"local_shard={local_numel:,} | dtype={p.dtype} | requires_grad={p.requires_grad}"
            )
            shown += 1
            if shown >= max_items:
                break

    if shown == 0:
        logger.info(
            f"[rank {rank}] no local param shards found (unexpected for ZeRO-3).")


log_param_distribution_and_size(engine, device, RANK, WORLD_SIZE)
log_gpu_memory_snapshot("post-init-models", RANK)
log_first_local_params(engine, RANK, max_items=10)
logger.info("Components built and sharded successfully.")

logi("Components built and sharded successfully.")
logi("Starting training loop...")

# ----------------------------
# Helper function to extract target representations
# ----------------------------


def extract_target_representations(representations, target_mask):
    """
    Extract representations at target positions.

    Args:
        representations: [B, L, D]
        target_mask: [B, L] - 1 where targets are

    Returns:
        targets: [B, num_targets, D]
    """
    B, L, D = representations.shape
    device = representations.device

    num_targets = int(target_mask.sum(dim=1).max().item())

    if num_targets == 0:
        return torch.zeros(B, 1, D, device=device)

    targets = torch.zeros(B, num_targets, D, device=device)

    for i in range(B):
        target_indices = torch.where(target_mask[i] == 1)[0]
        num_tgt = len(target_indices)
        if num_tgt > 0:
            targets[i, :num_tgt] = representations[i, target_indices]

    return targets


# ----------------------------
# Training loop (Batched processing with tensor-based masks)
# ----------------------------
engine.train()
for patch_batch in dataloader:
    engine.zero_grad()
    patch_batch = to_device(patch_batch)

    # Process FULL BATCH at once (not one sequence at a time!)
    batch_token_ids = []
    max_len = 0

    for patches in patch_batch:
        if len(patches) == 0:
            continue
        token_ids = tokenizer.convert_tokens_to_ids(patches)
        batch_token_ids.append(token_ids)
        max_len = max(max_len, len(token_ids))

    if len(batch_token_ids) == 0:
        continue

    # Pad sequences to same length
    padded_ids = []
    attention_masks = []

    for token_ids in batch_token_ids:
        pad_len = max_len - len(token_ids)
        padded = token_ids + [tokenizer.pad_token_id] * pad_len
        attn_mask = [1] * len(token_ids) + [0] * pad_len

        padded_ids.append(padded)
        attention_masks.append(attn_mask)

    # Convert to tensors
    input_ids = torch.tensor(padded_ids, device=device)  # [B, L]
    attention_mask = torch.tensor(
        attention_masks, device=device).float()  # [B, L]

    # STEP 1: Create context and target masks
    mask_pair = context_target_creator.create_context_and_targets(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    context_mask = mask_pair.context_mask  # [B, L]
    target_mask = mask_pair.target_masks[0]  # [B, L]

    # STEP 2: Forward pass - predict target representations
    predicted_targets = engine(
        input_ids=input_ids,
        attention_mask=attention_mask,
        context_mask=context_mask,
        target_mask=target_mask
    )  # [B, num_targets, D]

    # STEP 3: Get ground truth target representations (no gradients)
    with torch.no_grad():
        target_output = target_encoder.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        full_target_repr = target_output.last_hidden_state  # [B, L, D]

        # Extract at target positions
        actual_targets = extract_target_representations(
            full_target_repr,
            target_mask
        )  # [B, num_targets, D]

    # STEP 4: Compute loss
    loss = loss_calculator(predicted_targets, actual_targets)

    # STEP 5: Backward and optimize
    engine.backward(loss)
    engine.step()

    # STEP 6: Update EMA (once per batch!)
    target_encoder.update()

    if RANK == 0:
        num_targets = int(target_mask.sum().item())
        logger.info(
            f"Loss: {loss.item():.6f} | Batch size: {input_ids.shape[0]} | Total targets: {num_targets}")

logi("Training loop finished.")
