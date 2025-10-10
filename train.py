# train.py
import sys, os, logging, math
import torch
import torch.nn as nn
import torch.distributed as dist
import deepspeed
import math

from contextlib import nullcontext
from torch.utils.data import DataLoader

from src.builders.dataloader_builder import dataloader_builder
from src.builders.patcher_builder import patcher_builder
from src.builders.masker_builder import masker_builder
from src.builders.encoder_builder import encoder_builder
from src.encoders.target_encoders.ema_target_encoder import ema_target_encoder
from src.builders.loss_calculator_builder import loss_calculator_builder
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
    logger.info(f"GPU[{LOCAL_RANK}]: {torch.cuda.get_device_name(torch.cuda.current_device())}")

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

START_OF_CONTEXT_TOKEN = "<SOC>"
END_OF_CONTEXT_TOKEN = "<EOT>"
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

target_creator = masker_builder(target_mask_strategy).build()
context_creator = masker_builder(context_mask_strategy).build()

# Context encoder (trainable)
context_encoder = encoder_builder(context_encoder_type).build(
    model_id=STRATEGY_CONSTS["CONTEXT_MODEL_ID"],
    config=CONTEXT_ENCODER_CONFIG,
)
tokenizer = context_encoder.tokenizer

# EMA target encoder (non-trainable, updated via .update())
target_encoder = ema_target_encoder(context_encoder, DEFAULT_EMA_DECAY)
target_encoder.eval()                 # disable dropout etc.
for p in target_encoder.parameters(): # belt & suspenders (already in your class)
    p.requires_grad = False

# Patcher & loader
patcher = patcher_builder(patch_strategy).build(context_encoder, target_encoder)
dataloader = dataloader_builder(training_dataset).build(patcher, batch_size=BATCH_SIZE)

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
    def __init__(self, context_encoder, target_predictor, use_predictor_head=True):
        super().__init__()
        self.context_encoder = context_encoder
        self.use_predictor_head = use_predictor_head
        
        if use_predictor_head:
            # Use a simple MLP head instead of the full target_predictor model
            # Adjust dimensions based on your model's hidden size
            hidden_size = CONTEXT_ENCODER_CONFIG["hidden_size"]
            self.predictor_head = PredictorHead(
                input_dim=hidden_size,  # Size of combined embeddings
                hidden_dim=hidden_size * 2,
                output_dim=hidden_size  # Should match target encoder output dim
            )
        else:
            # If target_predictor can handle embeddings directly
            self.target_predictor = target_predictor

    def forward(self, context, targets, start_tok, end_tok):
        device = next(self.parameters()).device
        
        # Handle the case where context and targets are lists of token strings
        if isinstance(context, list):
            # If it's a list of lists, take the first one for now
            if isinstance(context[0], list):
                context_tokens = context[0]
            else:
                context_tokens = context
        else:
            context_tokens = context

        if isinstance(targets, list) and len(targets) > 0:
            # If it's a list of lists, process each target
            if isinstance(targets[0], list):
                target_token_lists = targets
            else:
                target_token_lists = [targets]
        else:
            target_token_lists = [targets]

        # Convert context tokens to tensor if needed and encode directly through model
        if isinstance(context_tokens, list) and all(isinstance(x, str) for x in context_tokens):
            context_ids = self.context_encoder.tokenizer.convert_tokens_to_ids(context_tokens)
            context_tensor = torch.tensor([context_ids], device=device)
            
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = torch.ones_like(context_tensor)
            
            # Pass directly to the underlying model, bypassing the tokenizer
            model_inputs = {
                "input_ids": context_tensor,
                "attention_mask": attention_mask
            }
            encoded_context = self.context_encoder.model(**model_inputs).last_hidden_state
        else:
            # If it's already a tensor or different format, handle accordingly
            encoded_context = self.context_encoder(context_tokens)
        
        # Get embeddings for special tokens
        start_emb = self.context_encoder.get_embeddings(start_tok)
        end_emb = self.context_encoder.get_embeddings(end_tok)
        
        predicted_targets = []
        for target_tokens in target_token_lists:
            # Convert target tokens to embeddings
            tgt_emb = self.context_encoder.get_embeddings(target_tokens)
            
            # Concatenate embeddings along sequence dimension
            combined = torch.cat([start_emb, encoded_context, end_emb, tgt_emb], dim=1)
            
            if self.use_predictor_head:
                # Use MLP head on the combined embeddings
                # Take mean pooling across sequence dimension for a fixed-size representation
                combined_pooled = combined.mean(dim=1, keepdim=True)
                pred = self.predictor_head(combined_pooled)
            else:
                # If your target_predictor can handle embeddings directly, use it
                # Otherwise, you'll need to adapt this part based on your specific predictor
                pred = self.target_predictor(combined)
            
            predicted_targets.append(pred)
            
        return encoded_context, predicted_targets

# Create the trainable model with predictor head
trainable = TrainableJEPA(context_encoder, target_predictor, use_predictor_head=True)

# ----------------------------
# DeepSpeed initialize (ZeRO-3)
# ----------------------------
model_parameters = [p for p in trainable.parameters() if p.requires_grad]
def _sum_params_module(module, trainable_only=False):
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))

pre_total_all = _sum_params_module(trainable, trainable_only=False)
pre_total_trn = _sum_params_module(trainable, trainable_only=True)
pre_global_numels = {name: p.numel() for name, p in trainable.named_parameters()}

if RANK == 0:
    logi(f"[pre-shard] total params (all): {pre_total_all:,}")
    logi(f"[pre-shard] total params (trainable): {pre_total_trn:,}")

engine, optimizer, _, _ = deepspeed.initialize(
    model=trainable,
    model_parameters=model_parameters,
    config=DS_CONFIG,
)

# Verify ZeRO-3 really active
try:
    zero_on = engine.zero_optimization()
    zero_stage = engine.zero_optimization_stage()
except Exception:
    zero_on = False
    zero_stage = None

logi(f"DeepSpeed ZeRO enabled: {zero_on}, stage: {zero_stage}, dp_world_size: {getattr(engine, 'dp_world_size', WORLD_SIZE)}")
assert zero_on and zero_stage == 3, f"Expected ZeRO stage 3, got: {zero_stage}"

# ----------------------------
# Device plumbing
# ----------------------------
device = engine.device if torch.cuda.is_available() else torch.device("cpu")

start_tok_tensor = tokenizer.encode(START_OF_CONTEXT_TOKEN, return_tensors="pt").to(device)
end_tok_tensor = tokenizer.encode(END_OF_CONTEXT_TOKEN, return_tensors="pt").to(device)

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
        outs = [torch.zeros_like(t) for _ in range(torch.distributed.get_world_size())]
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
    shard_bytes  = _all_gather_i64(local_bytes, device)

    global_numel = sum(shard_numels)
    global_bytes = sum(shard_bytes)

    if rank == 0:
        logger.info(f"[post-shard] global params: {global_numel:,}  (~{_fmt_gb(global_bytes)} across all shards)")
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
        logger.info(f"[rank {rank}] no local param shards found (unexpected for ZeRO-3).")

log_param_distribution_and_size(engine, device, RANK, WORLD_SIZE)
log_gpu_memory_snapshot("post-init-models", RANK)
log_first_local_params(engine, RANK, max_items=10)
logger.info("Components built and sharded successfully.")

logi("Components built and sharded successfully.")
logi("Starting training loop...")

# ----------------------------
# Training loop (DeepSpeed)
# ----------------------------
engine.train()
for patch_batch in dataloader:
    engine.zero_grad()
    patch_batch = to_device(patch_batch)

    for patches in patch_batch:
        targets = target_creator.create_spans(patches)
        context = context_creator.create_spans(patches)

        targets = to_device(targets)
        context = to_device(context)

        encoded_context, predicted_targets = engine(context, targets, start_tok_tensor, end_tok_tensor)
        with torch.no_grad():
            encoded_target = target_encoder(patches)
        loss = loss_calculator(encoded_target, predicted_targets)

        engine.backward(loss)
        engine.step()
        target_encoder.update()

        if RANK == 0:
            logger.info(f"Loss: {float(loss.detach().cpu())}")

logi("Training loop finished.")