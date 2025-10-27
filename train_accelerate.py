import json
import os
import torch
import torch.nn as nn
from typing import Any, cast, Dict

from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin

from src.maskers.block_mask_generator import BlockMaskGenerator
from src.maskers.random_mask_generator import RandomMaskGenerator
from src.maskers.utils import extract_target_representations
from src.patchers.helpers import create_patched_embeddings
from src.builders.dataloader_builder import dataloader_builder
from src.builders.patcher_builder import patcher_builder
from src.builders.encoder_builder import encoder_builder
from src.encoders.target_encoders.ema_target_encoder import ema_target_encoder
from src.builders.loss_calculator_builder import loss_calculator_builder
from src.maskers.context_target_creator import ContextTargetCreator
from config import STRATEGY_CONSTS
from src.logging.logging_helpers import logi, log_once, get_logger, training_loop_log, save_checkpoint, calculate_gradient_norm
from src.predictors.simple_predictor import SimplePredictor

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ============================================
# HELPER FUNCTIONS
# ============================================
def validate_token_length(token_ids):
    """Truncate token sequences that exceed maximum allowed length."""
    max_allowed_tokens = STRATEGY_CONSTS["PATCH_SIZE"] * STRATEGY_CONSTS["MAX_SEQ_LENGTH"]
    if len(token_ids) > max_allowed_tokens:
        return token_ids[:max_allowed_tokens]
    return token_ids



# ============================================
# ACCELERATE INITIALIZATION (replaces all manual DeepSpeed setup)
# ============================================


LOG_DIR = "/g/data/oy87/cw9909/llm_jepa/logs"
os.makedirs(LOG_DIR, exist_ok=True)

accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=1,
    log_with="tensorboard",
    project_dir=LOG_DIR
)

accelerator.init_trackers(
    project_name="llm_jepa_training",
    config=STRATEGY_CONSTS,
)

# Get distributed info from Accelerator
RANK = accelerator.process_index
LOCAL_RANK = accelerator.local_process_index
WORLD_SIZE = accelerator.num_processes
IS_DIST = WORLD_SIZE > 1
device = accelerator.device

log_once("Starting training pipeline setup...", RANK)
log_once(f"CUDA available: {torch.cuda.is_available()}", RANK)
if torch.cuda.is_available():
    logi(f"GPU[{LOCAL_RANK}]: {torch.cuda.get_device_name(LOCAL_RANK)}", RANK)

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
    context_generator = RandomMaskGenerator(mask_ratio=0.6)
elif STRATEGY_CONSTS["CONTEXT_STRATEGY"] == "block":
    context_generator = BlockMaskGenerator(span_length=50)
else:
    raise ValueError(f"Unknown context strategy: {STRATEGY_CONSTS['CONTEXT_STRATEGY']}")

if STRATEGY_CONSTS["MASK_STRATEGY"] == "random":
    target_generator = RandomMaskGenerator(mask_ratio=0.15)
elif STRATEGY_CONSTS["MASK_STRATEGY"] == "block":
    target_generator = BlockMaskGenerator(span_length=10)
else:
    raise ValueError(f"Unknown target strategy: {STRATEGY_CONSTS['MASK_STRATEGY']}")

context_target_creator = ContextTargetCreator(
    context_generator=context_generator,
    target_generator=target_generator,
    num_targets=1
)

context_encoder = encoder_builder(STRATEGY_CONSTS["CONTEXT_ENCODER"]).build(
    model_id=STRATEGY_CONSTS["CONTEXT_MODEL_ID"],
    config=CONTEXT_ENCODER_CONFIG,
)
tokenizer = context_encoder.tokenizer

embedding_patcher = patcher_builder(
    STRATEGY_CONSTS["PATCH_STRATEGY"]).build(patch_size=STRATEGY_CONSTS["PATCH_SIZE"])
log_once(f"Embedding patcher: {embedding_patcher}", RANK)

class TextTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def create_patches(self, text: str):
        return self.tokenizer.tokenize(text)

text_tokenizer = TextTokenizer(tokenizer=context_encoder.tokenizer)
dataloader = dataloader_builder(STRATEGY_CONSTS["TRAINING_DATASET"]).build(
    text_tokenizer, batch_size=STRATEGY_CONSTS["BATCH_SIZE"])

predictor = SimplePredictor(hidden_dim=CONTEXT_ENCODER_CONFIG["hidden_size"])

# ----------------------------
# Create optimizers BEFORE prepare()
# ----------------------------
optimizer = torch.optim.AdamW(
    context_encoder.parameters(),
    lr=STRATEGY_CONSTS.get("LEARNING_RATE", 1e-4),
    weight_decay=0.01
)

predictor_optimizer = torch.optim.AdamW(
    predictor.parameters(),
    lr=STRATEGY_CONSTS.get("LEARNING_RATE", 1e-4),
    weight_decay=0.01
)

# ----------------------------
# Target Encoder (EMA)
# ----------------------------
target_encoder = ema_target_encoder(
    context_encoder, STRATEGY_CONSTS["DEFAULT_EMA_DECAY"])
target_encoder.eval()
for p in target_encoder.parameters():
    p.requires_grad = False

# ----------------------------
# ACCELERATE PREPARE (replaces DeepSpeed initialization)
# ----------------------------
context_encoder, predictor, optimizer, predictor_optimizer, dataloader = accelerator.prepare(
    context_encoder, predictor, optimizer, predictor_optimizer, dataloader
)

target_encoder = target_encoder.to(device)



# ----------------------------
# Training Loop
# ----------------------------
checkpoint_dir = STRATEGY_CONSTS.get("CHECKPOINT_DIR", "./checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

log_once("Starting training loop...", RANK)

for step, token_batch in enumerate(dataloader, start=1):
    max_steps = STRATEGY_CONSTS.get('MAX_STEPS')
    if max_steps is not None and step > max_steps:
        break

    logi(f"Step {step} - START", RANK)
    
    # Prepare batch
    batch_token_ids = []
    max_len = 0

    for tokens in token_batch:
        if len(tokens) == 0:
            continue
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_ids = validate_token_length(token_ids)
        batch_token_ids.append(token_ids)
        max_len = max(max_len, len(token_ids))

    if len(batch_token_ids) == 0:
        continue

    # Pad sequences
    padded_ids = []
    attention_masks = []

    for token_ids in batch_token_ids:
        pad_len = max_len - len(token_ids)
        padded = token_ids + [tokenizer.pad_token_id] * pad_len
        attn_mask = [1] * len(token_ids) + [0] * pad_len
        padded_ids.append(padded)
        attention_masks.append(attn_mask)

    input_ids = torch.tensor(padded_ids, device=device)
    attention_mask = torch.tensor(attention_masks, device=device).float()

    # Create patches
    context_patch_embeds, patched_attention_mask = create_patched_embeddings(
        input_ids=input_ids,
        attention_mask=attention_mask,
        embedding_layer=accelerator.unwrap_model(context_encoder).get_input_embeddings(),
        patcher=embedding_patcher,
        patch_size=STRATEGY_CONSTS["PATCH_SIZE"]
    )

    with torch.no_grad():
        target_patch_embeds = context_patch_embeds.detach().clone()

    # Create masks
    B, L_patches, D = context_patch_embeds.shape
    dummy_input_ids = torch.zeros(B, L_patches, dtype=torch.long, device=device)

    mask_pair = context_target_creator.create_context_and_targets(
        input_ids=dummy_input_ids,
        attention_mask=patched_attention_mask
    )

    context_mask = mask_pair.context_mask
    target_mask = mask_pair.target_masks[0]
    context_mask = context_mask * (1 - target_mask)

    # Global max targets
    local_max_targets = int(target_mask.sum(dim=1).max().item())
    if IS_DIST:
        max_targets_tensor = torch.tensor(local_max_targets, device=device)
        torch.distributed.all_reduce(max_targets_tensor, op=torch.distributed.ReduceOp.MAX)
        global_max_targets = int(max_targets_tensor.item())
    else:
        global_max_targets = local_max_targets

    # Forward pass - use accelerator.unwrap_model() to access model attributes
    context_output = accelerator.unwrap_model(context_encoder).model(
        inputs_embeds=context_patch_embeds,
        attention_mask=patched_attention_mask,
    )
    context_repr = context_output.last_hidden_state

    with torch.no_grad():
        target_output = target_encoder.model(
            inputs_embeds=target_patch_embeds,
            attention_mask=patched_attention_mask,
        )
        target_repr = target_output.last_hidden_state

    # Apply masks
    masked_context_repr = context_repr * context_mask.unsqueeze(-1)

    # Predict targets
    predicted_targets = predictor(
        masked_context_repr,
        target_mask,
        context_mask,
        global_max_targets
    )

    # Extract ground truth
    actual_targets = extract_target_representations(
        target_repr,
        target_mask,
        global_max_targets
    ).detach()

    # Compute loss
    loss = loss_calculator(predicted_targets, actual_targets)

    if torch.isnan(loss) or torch.isinf(loss):
        logi(f"Step {step} - WARNING: Invalid loss detected!", RANK)
        continue

    # Backward - Accelerate handles gradient scaling and accumulation
    accelerator.backward(loss)

    # Calculate gradient norm
    total_norm = calculate_gradient_norm(accelerator.unwrap_model(context_encoder))

    # Step optimizers
    optimizer.step()
    predictor_optimizer.step()
    optimizer.zero_grad()
    predictor_optimizer.zero_grad()

    target_encoder.update(accelerator.unwrap_model(context_encoder))

    # Logging
    if step % STRATEGY_CONSTS["LOG_INTERVAL"] == 0:
        current_lr = optimizer.param_groups[0]['lr']
        training_loop_log(
            step=step,
            max_steps=STRATEGY_CONSTS['MAX_STEPS'],
            loss=loss.item(),
            learning_rate=current_lr,
            grad_norm=total_norm,
            batch_size=B,
            num_patches=L_patches,
            num_targets=int(target_mask.sum().item()),
            rank=RANK
        )
        accelerator.log({
            "train/loss": loss.item(),
            "train/learning_rate": current_lr,
            "train/grad_norm": total_norm,
            "train/batch_size": B,
            "train/num_patches": L_patches,
            "train/num_targets": int(target_mask.sum().item()),
        }, step=step)

    # Checkpointing
    if STRATEGY_CONSTS["CHECKPOINT_INTERVAL"] and step % STRATEGY_CONSTS["CHECKPOINT_INTERVAL"] == 0:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=step,
                engine=accelerator.unwrap_model(context_encoder),
                optimizer=optimizer,
                loss=loss.item(),
                config=cast(Dict[str, Any], STRATEGY_CONSTS),
                rank=RANK,
                is_final=False
            )

log_once("Training loop finished.", RANK)
accelerator.end_training()

# Final checkpoint
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=step,
        engine=accelerator.unwrap_model(context_encoder),
        optimizer=optimizer,
        loss=loss.item(),
        config=cast(Dict[str, Any], STRATEGY_CONSTS),
        rank=RANK,
        is_final=True
    )