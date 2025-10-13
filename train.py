import json
from src.maskers.block_mask_generator import BlockMaskGenerator
from src.maskers.random_mask_generator import RandomMaskGenerator
from src.maskers.utils import extract_target_representations
import os
import torch
import torch.nn as nn
import deepspeed
from typing import Any, cast, Dict
from src.patchers.helpers import create_patched_embeddings

from src.builders.dataloader_builder import dataloader_builder
from src.builders.patcher_builder import patcher_builder
from src.builders.encoder_builder import encoder_builder
from src.encoders.target_encoders.ema_target_encoder import ema_target_encoder
from src.builders.loss_calculator_builder import loss_calculator_builder
from src.maskers.context_target_creator import ContextTargetCreator
from config import STRATEGY_CONSTS
from src.zero3.my_deepspeed import MyDeepspeed
from src.logging.logging_helpers import logi, log_once, get_logger, training_loop_log, save_checkpoint, calculate_gradient_norm
from src.predictors.simple_predictor import SimplePredictor

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def validate_token_length(token_ids):
    max_allowed_tokens = STRATEGY_CONSTS["PATCH_SIZE"] * STRATEGY_CONSTS["MAX_SEQ_LENGTH"]
    if len(token_ids) > max_allowed_tokens:
        log_once(f"WARNING: Truncating sequence from {len(token_ids)} to {max_allowed_tokens} tokens", RANK)
        return token_ids[:max_allowed_tokens]
    return token_ids

# ----------------------------
# DeepSpeed distributed init
# ----------------------------


def init_dist() -> tuple[bool, int, int, int]:
    """
    Initialize distributed training environment.

    Returns:
        tuple: (is_distributed, rank, local_rank, world_size)
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Get logger for this rank
    logger = get_logger(rank)

    if torch.cuda.is_available() and world_size > 1:
        try:
            deepspeed.init_distributed(dist_backend="nccl")
            logger.log(
                "Successfully initialized distributed training with NCCL backend")
        except RuntimeError as e:
            if "already initialized" in str(e):
                logger.log("Distributed backend already initialized")
            else:
                logger.log(
                    f"WARNING: Failed to initialize distributed training: {e}")
                logger.log("Continuing with single-process training")
                world_size = 1
                rank = 0
        except Exception as e:
            logger.log(f"ERROR: Unexpected error during distributed init: {e}")
            raise

    return world_size > 1, rank, local_rank, world_size


IS_DIST, RANK, LOCAL_RANK, WORLD_SIZE = init_dist()


log_once("Starting training pipeline setup...", RANK)
log_once(f"CUDA available: {torch.cuda.is_available()}", RANK)
if torch.cuda.is_available():
    logi(
        f"GPU[{LOCAL_RANK}]: {torch.cuda.get_device_name(torch.cuda.current_device())}", RANK)


log_once(json.dumps(STRATEGY_CONSTS), RANK)

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
# Build components
# ----------------------------
log_once("Building components...", RANK)
loss_calculator = loss_calculator_builder(
    STRATEGY_CONSTS["LOSS_CALCULATOR"]).build()


if STRATEGY_CONSTS["CONTEXT_STRATEGY"] == "random":
    context_generator = RandomMaskGenerator(mask_ratio=0.6)  # 60% visible
elif STRATEGY_CONSTS["CONTEXT_STRATEGY"] == "block":
    context_generator = BlockMaskGenerator(
        span_length=50)  # Large visible block
else:
    raise ValueError(
        f"Unknown context strategy: {STRATEGY_CONSTS['CONTEXT_STRATEGY']}")

if STRATEGY_CONSTS["MASK_STRATEGY"] == "random":
    target_generator = RandomMaskGenerator(mask_ratio=0.15)  # 15% to predict
elif STRATEGY_CONSTS["MASK_STRATEGY"] == "block":
    target_generator = BlockMaskGenerator(span_length=10)  # 10-token block
else:
    raise ValueError(
        f"Unknown target strategy: {STRATEGY_CONSTS['MASK_STRATEGY']}")

context_target_creator = ContextTargetCreator(
    context_generator=context_generator,
    target_generator=target_generator,
    num_targets=1  # Single target for now (can increase for multi-target JEPA)
)

context_encoder = encoder_builder(STRATEGY_CONSTS["CONTEXT_ENCODER"]).build(
    model_id=STRATEGY_CONSTS["CONTEXT_MODEL_ID"],
    config=CONTEXT_ENCODER_CONFIG,
)
tokenizer = context_encoder.tokenizer

target_encoder: ema_target_encoder = ema_target_encoder(
    context_encoder, STRATEGY_CONSTS["DEFAULT_EMA_DECAY"])
target_encoder.eval()                 # disable dropout etc.
for p in target_encoder.parameters():  # belt & suspenders (already in your class)
    p.requires_grad = False

# Build embedding patcher (I-JEPA style)
embedding_patcher = patcher_builder(
    STRATEGY_CONSTS["PATCH_STRATEGY"]).build(patch_size=STRATEGY_CONSTS["PATCH_SIZE"])
log_once(f"Embedding patcher: {embedding_patcher}", RANK)


class TextTokenizer:
    """Tokenizes text into tokens for the dataloader."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def create_patches(self, text: str):
        """Returns tokenized text (called 'patches' for dataloader compatibility)."""
        return self.tokenizer.tokenize(text)


text_tokenizer = TextTokenizer(tokenizer=context_encoder.tokenizer)
dataloader = dataloader_builder(STRATEGY_CONSTS["TRAINING_DATASET"]).build(
    text_tokenizer, batch_size=STRATEGY_CONSTS["BATCH_SIZE"])

# Create the predictor
predictor = SimplePredictor(hidden_dim=CONTEXT_ENCODER_CONFIG["hidden_size"])
predictor = predictor.to(dtype=torch.bfloat16)


# ----------------------------
# DeepSpeed initialize (ZeRO-3)
# ----------------------------
MyDeepspeed = MyDeepspeed(context_encoder, predictor, RANK, WORLD_SIZE)
engine, optimizer = MyDeepspeed.get_engine_and_optim()


# CRITICAL FIX: Update EMA to track the DeepSpeed-wrapped encoder
# The target_encoder must reference the actual trained model, not the original
target_encoder.context_encoder = engine.module.context_encoder


# ----------------------------
# Device plumbing
# ----------------------------
device = engine.device


def to_device(x: Any) -> Any:
    """
    Recursively move tensors, dicts, lists, or tuples to device.

    Args:
        x: Input (tensor, dict, list, tuple, or other)

    Returns:
        Same structure with tensors moved to device
    """
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

MyDeepspeed.log_gpu_memory_usage()

log_once("Components built and sharded successfully.", RANK)
log_once("Starting training loop...", RANK)


# ----------------------------
# Training loop
# ----------------------------
engine.train()

# Create checkpoint directory
checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
log_once(f"Checkpoint directory: {checkpoint_dir}", RANK)

step = 0
for token_batch in dataloader:
    if step >= STRATEGY_CONSTS["MAX_STEPS"]:
        log_once(
            f"Reached max steps ({STRATEGY_CONSTS['MAX_STEPS']}). Stopping training.", RANK)
        break

    step += 1
    engine.zero_grad()
    token_batch = to_device(token_batch)

    # Prepare batch: tokenize and pad sequences
    batch_token_ids = []
    max_len = 0

    for tokens in token_batch:
        if len(tokens) == 0:
            continue
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # Truncate if needed
        token_ids = validate_token_length(token_ids)
        
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
        patch_size=STRATEGY_CONSTS["PATCH_SIZE"]
    )  # [B, L//patch_size, D], [B, L//patch_size]

    context_patch_embeds = context_patch_embeds.to(dtype=torch.bfloat16)

    # For target encoder, create patches the same way
    with torch.no_grad():
        target_patch_embeds, _ = create_patched_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            embedding_layer=target_encoder.get_input_embeddings(),
            patcher=embedding_patcher,
            patch_size=STRATEGY_CONSTS["PATCH_SIZE"]
        )  # [B, L//patch_size, D]
        target_patch_embeds = target_patch_embeds.to(dtype=torch.bfloat16)

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

    context_mask = context_mask * (1 - target_mask)

    # Context encoder (trainable)
    context_output = context_encoder.model(
        inputs_embeds=context_patch_embeds,
        attention_mask=patched_attention_mask,
    )
    context_repr = context_output.last_hidden_state  # [B, L_patches, D]

    context_repr = context_repr.to(dtype=torch.bfloat16)


    # Target encoder (EMA, no gradients)
    with torch.no_grad():
        target_output = target_encoder.model(
            inputs_embeds=target_patch_embeds,
            attention_mask=patched_attention_mask,
        )
        target_repr = target_output.last_hidden_state  # [B, L_patches, D]
        target_repr = target_repr.to(dtype=torch.bfloat16)

    # STEP 4: Apply masks to representations
    # Zero out target regions from context (JEPA approach)
    masked_context_repr = context_repr * \
        context_mask.unsqueeze(-1)  # [B, L_patches, D]

    masked_context_repr = masked_context_repr.to(dtype=torch.bfloat16)

    # STEP 5: Predict target representations from masked context
    predicted_targets = engine.module.predictor(masked_context_repr, target_mask, context_mask)

    # STEP 6: Extract ground truth target representations
    actual_targets = extract_target_representations(
        target_repr,
        target_mask
    )  # [B, num_targets, D]


    # STEP 7: Compute loss
    loss = loss_calculator(predicted_targets, actual_targets)

    # STEP 8: Backward and optimize
    engine.backward(loss)

    # Calculate gradient norm for monitoring training stability
    total_norm = calculate_gradient_norm(engine.module)

    engine.step()

    # STEP 9: Update EMA (once per batch!)
    target_encoder.update()

    # Logging
    num_targets = int(target_mask.sum().item())
    if step % STRATEGY_CONSTS["LOG_INTERVAL"] == 0:
        # Get current learning rate from optimizer
        current_lr = optimizer.param_groups[0]['lr']

        training_loop_log(
            step=step,
            max_steps=STRATEGY_CONSTS['MAX_STEPS'],
            loss=loss.item(),
            learning_rate=current_lr,
            grad_norm=total_norm,
            batch_size=B,
            num_patches=L_patches,
            num_targets=num_targets,
            rank=RANK
        )

    # Checkpointing
    if step % STRATEGY_CONSTS["CHECKPOINT_INTERVAL"] == 0:
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            step=step,
            engine=engine,
            optimizer=optimizer,
            loss=loss.item(),
            config=cast(Dict[str, Any], STRATEGY_CONSTS),
            rank=RANK,
            is_final=False
        )

log_once("Training loop finished.", RANK)

# Save final checkpoint
save_checkpoint(
    checkpoint_dir=checkpoint_dir,
    step=step,
    engine=engine,
    optimizer=optimizer,
    loss=loss.item(),
    config=cast(Dict[str, Any], STRATEGY_CONSTS),
    rank=RANK,
    is_final=True
)
