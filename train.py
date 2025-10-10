# train.py
import sys
import os
import logging
import torch
import torch.nn as nn
import deepspeed

from src.builders.dataloader_builder import dataloader_builder
from src.builders.patcher_builder import patcher_builder
from src.builders.masker_builder import masker_builder
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

# Create context-target creator for JEPA-style training
context_strategy = masker_builder(context_mask_strategy).build()
target_strategy = masker_builder(target_mask_strategy).build()

context_target_creator = ContextTargetCreator(
    context_strategy=context_strategy,
    target_strategy=target_strategy,
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


class TrainableJEPA(nn.Module):
    """
    I-JEPA style model for text:
    - Takes full sequence + attention masks (what's visible)
    - Predicts embeddings at specific target positions
    - Uses positional information throughout
    """

    def __init__(self, context_encoder, target_predictor, use_predictor_head=True):
        super().__init__()
        self.context_encoder = context_encoder
        self.use_predictor_head = use_predictor_head

        if use_predictor_head:
            hidden_size = CONTEXT_ENCODER_CONFIG["hidden_size"]
            self.predictor_head = PredictorHead(
                input_dim=hidden_size,
                hidden_dim=hidden_size * 2,
                output_dim=hidden_size
            )
        else:
            self.target_predictor = target_predictor

    def forward(self, token_ids, context_mask, target_mask):
        """
        I-JEPA forward pass with attention masking.

        Args:
            token_ids: Tensor of shape (batch, seq_len) - full sequence
            context_mask: Boolean tensor (batch, seq_len) - True = can attend
            target_mask: Boolean tensor (batch, seq_len) - True = predict here

        Returns:
            predictions: Tensor (batch, num_targets, hidden_dim)
            target_positions: Tensor (batch, num_targets) - indices of target positions
        """
        batch_size, seq_len = token_ids.shape

        # Encode with context mask (model only attends to context positions)
        # Note: attention_mask convention: 1 = attend, 0 = don't attend
        attention_mask = context_mask.long()

        outputs = self.context_encoder.model(
            input_ids=token_ids,
            attention_mask=attention_mask
        )
        encoded = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

        # Extract embeddings at target positions
        predictions = []
        target_positions = []

        for i in range(batch_size):
            # Get indices where target_mask is True
            tgt_indices = torch.where(target_mask[i])[0]  # (num_targets,)

            if len(tgt_indices) > 0:
                # Get embeddings at those positions
                # (num_targets, hidden_dim)
                tgt_embeds = encoded[i, tgt_indices, :]

                if self.use_predictor_head:
                    # Predict target representations
                    # (num_targets, hidden_dim)
                    pred = self.predictor_head(tgt_embeds)
                else:
                    pred = tgt_embeds

                predictions.append(pred)
                target_positions.append(tgt_indices)

        return predictions, target_positions


# Create the trainable model with predictor head
trainable = TrainableJEPA(
    context_encoder, target_predictor, use_predictor_head=True)

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
# Training loop (I-JEPA style with attention masking)
# ----------------------------
engine.train()
for patch_batch in dataloader:
    engine.zero_grad()
    patch_batch = to_device(patch_batch)

    for patches in patch_batch:
        # patches is a list of tokens (strings)
        if len(patches) == 0:
            continue

        seq_len = len(patches)

        # Get context and target masks (numpy arrays of shape (seq_len,))
        context_mask_np, target_masks_np = context_target_creator.create_context_and_targets(
            seq_len)

        # Convert tokens to IDs
        token_ids = tokenizer.convert_tokens_to_ids(patches)
        token_ids_tensor = torch.tensor(
            [token_ids], device=device)  # (1, seq_len)

        # Convert masks to tensors
        context_mask_tensor = torch.from_numpy(
            context_mask_np).bool().unsqueeze(0).to(device)  # (1, seq_len)
        target_mask_tensor = torch.from_numpy(
            target_masks_np[0]).bool().unsqueeze(0).to(device)  # (1, seq_len)

        # Forward pass: predict embeddings at target positions using context
        predictions, target_positions = engine(
            token_ids_tensor, context_mask_tensor, target_mask_tensor)

        # Get ground truth embeddings at target positions from EMA encoder
        with torch.no_grad():
            # Encode full sequence with target encoder (no masking)
            full_attention_mask = torch.ones_like(token_ids_tensor)
            target_outputs = target_encoder.model(
                input_ids=token_ids_tensor,
                attention_mask=full_attention_mask
            )
            # (1, seq_len, hidden_dim)
            target_embeds_full = target_outputs.last_hidden_state

            # Extract embeddings at target positions
            target_embeds = []
            for i in range(len(predictions)):
                tgt_pos = target_positions[i]
                tgt_embeds_at_pos = target_embeds_full[i, tgt_pos, :]
                target_embeds.append(tgt_embeds_at_pos)

        # Compute loss: compare predicted vs ground truth embeddings at target positions
        # For now, just use the first batch item (batch_size=1 in the inner loop)
        if len(predictions) > 0 and len(target_embeds) > 0:
            pred = predictions[0]  # (num_targets, hidden_dim)
            target = target_embeds[0]  # (num_targets, hidden_dim)

            # Simple MSE loss
            loss = torch.nn.functional.mse_loss(pred, target)

            engine.backward(loss)
            engine.step()
            target_encoder.update()

            if RANK == 0:
                logger.info(
                    f"Loss: {float(loss.detach().cpu()):.6f} | Num targets: {len(target_positions[0])}")

logi("Training loop finished.")
