# train.py
from src.maskers.block_mask_generator import BlockMaskGenerator
from src.maskers.random_mask_generator import RandomMaskGenerator
from src.maskers.utils import extract_target_representations
import sys
import os
import logging
import torch
import torch.nn as nn
import deepspeed
from typing import Any, Tuple

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


def init_dist() -> tuple[bool, int, int, int]:
    """
    Initialize distributed training environment.

    Returns:
        tuple: (is_distributed, rank, local_rank, world_size)
    """
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


def logi(msg: str) -> None:
    """Log message only on rank 0."""
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
patch_size = STRATEGY_CONSTS["PATCH_SIZE"]
target_mask_strategy = STRATEGY_CONSTS["MASK_STRATEGY"]
context_mask_strategy = STRATEGY_CONSTS["CONTEXT_STRATEGY"]
context_encoder_type = STRATEGY_CONSTS["CONTEXT_ENCODER"]
target_predictor_type = STRATEGY_CONSTS["TARGET_PREDICTOR"]
loss_calculator_type = STRATEGY_CONSTS["LOSS_CALCULATOR"]

logi(f"Training dataset: {training_dataset}")
logi(f"Patch strategy: {patch_strategy}")
logi(f"Patch size: {patch_size}")
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
target_encoder: ema_target_encoder = ema_target_encoder(
    context_encoder, DEFAULT_EMA_DECAY)
target_encoder.eval()                 # disable dropout etc.
for p in target_encoder.parameters():  # belt & suspenders (already in your class)
    p.requires_grad = False

# Build embedding patcher (I-JEPA style)
embedding_patcher = patcher_builder(
    patch_strategy).build(patch_size=patch_size)
logi(f"Embedding patcher: {embedding_patcher}")

# Simple text tokenizer for dataloader


class TextTokenizer:
    """Tokenizes text into tokens for the dataloader."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def create_patches(self, text: str):
        """Returns tokenized text (called 'patches' for dataloader compatibility)."""
        return self.tokenizer.tokenize(text)


text_tokenizer = TextTokenizer(tokenizer=context_encoder.tokenizer)
dataloader = dataloader_builder(training_dataset).build(
    text_tokenizer, batch_size=BATCH_SIZE)

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


def create_patched_embeddings(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    embedding_layer: nn.Module,
    patcher: Any,
    patch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Helper function to create a stream of patches from tokens.

    This applies patching right after tokenization, creating patch embeddings
    that can be used for context/target creation and encoding.

    Args:
        input_ids: [B, L] - Token IDs
        attention_mask: [B, L] - Attention mask (1 for real tokens, 0 for padding)
        embedding_layer: Token embedding layer from the model
        patcher: Patcher instance (MeanPatcher or MaxPatcher)
        patch_size: Size of each patch

    Returns:
        patched_embeddings: [B, L//patch_size, D] - Patched token embeddings
        patched_attention_mask: [B, L//patch_size] - Patched attention mask
    """
    B, L = input_ids.shape

    # Get token embeddings
    token_embeddings = embedding_layer(input_ids)  # [B, L, D]

    # Apply patching to create patch embeddings
    if patch_size > 1:
        patched_embeddings = patcher.patch(
            token_embeddings)  # [B, L//patch_size, D]

        # Also patch attention mask to match
        truncated_length = (L // patch_size) * patch_size
        attention_mask_truncated = attention_mask[:, :truncated_length]
        attention_mask_reshaped = attention_mask_truncated.view(
            B, -1, patch_size)
        patched_attention_mask = attention_mask_reshaped.max(dim=2)[
            0]  # [B, L//patch_size]
    else:
        # patch_size=1: no patching
        patched_embeddings = token_embeddings
        patched_attention_mask = attention_mask

    return patched_embeddings, patched_attention_mask


class SimplePredictor(nn.Module):
    """
    Simple predictor that uses full context to predict target representations.

    Uses mean-pooling of context to predict each target position.
    In a more sophisticated version, you'd use cross-attention from targets to context.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, context_repr: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
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


# Create the predictor
hidden_size = CONTEXT_ENCODER_CONFIG["hidden_size"]
predictor = SimplePredictor(hidden_dim=hidden_size)

# ----------------------------
# DeepSpeed initialize (ZeRO-3)
# ----------------------------
# Wrap trainable components (context_encoder + predictor) in a single module


class TrainableModule(nn.Module):
    """Wrapper for trainable components for DeepSpeed."""

    def __init__(self, context_encoder, predictor):
        super().__init__()
        self.context_encoder = context_encoder
        self.predictor = predictor


trainable = TrainableModule(context_encoder, predictor)

model_parameters = [p for p in trainable.parameters() if p.requires_grad]


def _sum_params_module(module: nn.Module, trainable_only: bool = False) -> int:
    """Count total parameters in a module."""
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


def to_device(x: Any) -> Any:
    """
    Recursively move tensors, dicts, lists, or tuples to device.

    Args:
        x: Input (tensor, dict, list, tuple, or other)

    Returns:
        Same structure with tensors moved to device
    """
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)  # type: ignore[union-attr]
    if isinstance(x, dict):
        return {k: to_device(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v) for v in x)  # type: ignore[arg-type]
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


def _local_shard_numel(engine: Any, module: nn.Module) -> int:
    """
    Count *this rank's* parameter shard numel.

    Uses DeepSpeed ds_tensor if present.

    Args:
        engine: DeepSpeed engine
        module: PyTorch module

    Returns:
        Number of parameters in this rank's shard
    """
    local = 0
    for p in module.parameters():
        ds_tensor = getattr(p, "ds_tensor", None)
        if ds_tensor is not None:
            local += int(ds_tensor.numel())
        else:
            local += int(p.numel())
    return local


def _local_shard_bytes(engine: Any, module: nn.Module) -> int:
    """
    Exact bytes for this rank's param shards.

    Accounts for mixed dtypes.

    Args:
        engine: DeepSpeed engine
        module: PyTorch module

    Returns:
        Total bytes of parameters in this rank's shard
    """
    total_bytes = 0
    for p in module.parameters():
        ds_tensor = getattr(p, "ds_tensor", None)
        n = int(ds_tensor.numel()) if ds_tensor is not None else int(p.numel())
        total_bytes += n * p.element_size()
    return total_bytes


def _all_gather_i64(value: int, device: torch.device) -> list[int]:
    """
    All-gather a single int64 from each rank.

    Args:
        value: Integer value to gather
        device: Device to create tensors on

    Returns:
        List of integers from all ranks (or single value if not distributed)
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        t = torch.tensor([int(value)], dtype=torch.long, device=device)
        outs = [torch.zeros_like(t) for _ in range(
            torch.distributed.get_world_size())]
        torch.distributed.all_gather(outs, t)
        return [int(x.item()) for x in outs]
    return [int(value)]


def _fmt_gb(bytes_val: int) -> str:
    """Format bytes as GB string."""
    return f"{bytes_val / (1024**3):.2f} GB"


def log_param_distribution_and_size(
    engine: Any,
    device: torch.device,
    rank: int,
    world_size: int
) -> None:
    """
    Log global param count + each GPU's share (%) and bytes of its shard.

    Args:
        engine: DeepSpeed engine
        device: Device for tensor operations
        rank: Current process rank
        world_size: Total number of processes
    """
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


def log_gpu_memory_snapshot(tag: str, rank: int) -> None:
    """
    Log per-GPU memory (global device view) + this process's PyTorch alloc/reserved.

    Args:
        tag: Label for this memory snapshot
        rank: Current process rank
    """
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


def log_first_local_params(engine: Any, rank: int, max_items: int = 10) -> None:
    """
    Print the first `max_items` parameters *sharded to this rank*.

    Uses DeepSpeed's per-param ds_tensor to report the local shard size.

    Args:
        engine: DeepSpeed engine
        rank: Current process rank
        max_items: Maximum number of parameters to log
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
# Training loop
# ----------------------------
engine.train()
for token_batch in dataloader:
    engine.zero_grad()
    token_batch = to_device(token_batch)

    # Prepare batch: tokenize and pad sequences
    batch_token_ids = []
    max_len = 0

    for tokens in token_batch:
        if len(tokens) == 0:
            continue
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
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

    # STEP 1: Create stream of patches from tokens (I-JEPA style)
    # This happens BEFORE encoding - we create patch embeddings first
    context_patch_embeds, patched_attention_mask = create_patched_embeddings(
        input_ids=input_ids,
        attention_mask=attention_mask,
        embedding_layer=context_encoder.get_input_embeddings(),
        patcher=embedding_patcher,
        patch_size=patch_size
    )  # [B, L//patch_size, D], [B, L//patch_size]

    # For target encoder, create patches the same way
    with torch.no_grad():
        target_patch_embeds, _ = create_patched_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            embedding_layer=target_encoder.get_input_embeddings(),
            patcher=embedding_patcher,
            patch_size=patch_size
        )  # [B, L//patch_size, D]

    # STEP 2: Create context and target masks on the patch stream
    B, L_patches, D = context_patch_embeds.shape
    dummy_input_ids = torch.zeros(
        B, L_patches, dtype=torch.long, device=device)

    mask_pair = context_target_creator.create_context_and_targets(
        input_ids=dummy_input_ids,
        attention_mask=patched_attention_mask
    )

    context_mask = mask_pair.context_mask  # [B, L_patches]
    target_mask = mask_pair.target_masks[0]  # [B, L_patches]

    # STEP 3: Encode the patch stream (pass patch embeddings through transformer)
    # IMPORTANT: Scale position_ids by patch_size so positional encodings reflect token-space positions
    # Patch 0 → position 0, Patch 1 → position patch_size, Patch 2 → position 2*patch_size, etc.
    position_ids = torch.arange(0, L_patches, device=device) * patch_size
    position_ids = position_ids.unsqueeze(0).expand(B, -1)  # [B, L_patches]

    # Context encoder (trainable)
    context_output = context_encoder.model(
        inputs_embeds=context_patch_embeds,
        attention_mask=patched_attention_mask,
        position_ids=position_ids  # ← Scale positions for patches!
    )
    context_repr = context_output.last_hidden_state  # [B, L_patches, D]

    # Target encoder (EMA, no gradients)
    with torch.no_grad():
        target_output = target_encoder.model(
            inputs_embeds=target_patch_embeds,
            attention_mask=patched_attention_mask,
            position_ids=position_ids  # ← Scale positions for patches!
        )
        target_repr = target_output.last_hidden_state  # [B, L_patches, D]

    # STEP 4: Apply masks to representations
    # Zero out target regions from context (JEPA approach)
    masked_context_repr = context_repr * \
        context_mask.unsqueeze(-1)  # [B, L_patches, D]

    # STEP 5: Predict target representations from masked context
    predicted_targets = predictor(
        masked_context_repr, target_mask)  # [B, num_targets, D]

    # STEP 6: Extract ground truth target representations
    actual_targets = extract_target_representations(
        target_repr,
        target_mask
    )  # [B, num_targets, D]

    # STEP 7: Compute loss
    loss = loss_calculator(predicted_targets, actual_targets)

    # STEP 8: Backward and optimize
    engine.backward(loss)
    engine.step()

    # STEP 9: Update EMA (once per batch!)
    target_encoder.update()

    if RANK == 0:
        num_targets = int(target_mask.sum().item())
        logger.info(
            f"Loss: {loss.item():.6f} | Batch size: {B} | Patches: {L_patches} | Targets: {num_targets}")

logi("Training loop finished.")
