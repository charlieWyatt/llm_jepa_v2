import json
import os
import torch
import torch.nn as nn
from typing import Any, cast, Dict
import time
from tqdm import tqdm

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

def format_time(seconds):
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

# ============================================
# TRAINING CONFIGURATION
# ============================================

# Training hyperparameters - ADJUST THESE
NUM_EPOCHS = 1  # Set to 1 for single epoch
GRADIENT_ACCUMULATION_STEPS = 4  # Increase effective batch size by 4x
BASE_BATCH_SIZE = STRATEGY_CONSTS.get("BATCH_SIZE", 2)  # Per-GPU batch size
EFFECTIVE_BATCH_SIZE = BASE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

log_once(f"Training Configuration:", 0)
log_once(f"  Epochs: {NUM_EPOCHS}", 0)
log_once(f"  Base batch size per GPU: {BASE_BATCH_SIZE}", 0)
log_once(f"  Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}", 0)
log_once(f"  Effective batch size per GPU: {EFFECTIVE_BATCH_SIZE}", 0)

# ============================================
# ACCELERATE INITIALIZATION
# ============================================

LOG_DIR = "/g/data/oy87/cw9909/llm_jepa/logs"
os.makedirs(LOG_DIR, exist_ok=True)

accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # Use gradient accumulation
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
    "hidden_size": 768,      # CHANGED: 384 → 768
    "num_layers": 12,        # CHANGED: 6 → 12
    "attention_window": 512, # CHANGED: 128 → 512 (match Longformer)
}
TARGET_ENCODER_CONFIG = {
    "hidden_size": 768,      # CHANGED: 384 → 768
    "num_layers": 12,        # CHANGED: 2 → 12
    "attention_window": 512, # CHANGED: 256 → 512
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


# Calculate steps per epoch
TOTAL_DATASET_SIZE = 9999999999999
STEPS_PER_EPOCH = TOTAL_DATASET_SIZE // (BASE_BATCH_SIZE * WORLD_SIZE)
MAX_STEPS = STEPS_PER_EPOCH * NUM_EPOCHS

log_once(f"Dataset Configuration:", RANK)
log_once(f"  Total dataset size: {TOTAL_DATASET_SIZE:,}", RANK)
log_once(f"  Steps per epoch: {STEPS_PER_EPOCH:,}", RANK)
log_once(f"  Total training steps: {MAX_STEPS:,}", RANK)
log_once(f"  Total samples to process: {MAX_STEPS * BASE_BATCH_SIZE * WORLD_SIZE:,}", RANK)

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
    text_tokenizer, batch_size=BASE_BATCH_SIZE)

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
# ACCELERATE PREPARE
# ----------------------------
context_encoder, predictor, optimizer, predictor_optimizer, dataloader = accelerator.prepare(
    context_encoder, predictor, optimizer, predictor_optimizer, dataloader
)

target_encoder = target_encoder.to(device)

# ----------------------------
# Training Loop
# ----------------------------
base_checkpoint_dir = STRATEGY_CONSTS.get("CHECKPOINT_DIR", "./checkpoints")
checkpoint_dir = os.path.join(base_checkpoint_dir, f"patch_size_{STRATEGY_CONSTS['PATCH_SIZE']}")
os.makedirs(checkpoint_dir, exist_ok=True)

log_once("Starting training loop...", RANK)

# Initialize progress tracking
training_start = time.time()
total_samples = 0
global_step = 0
epoch_samples = 0

# Outer loop for epochs
for epoch in range(NUM_EPOCHS):
    log_once(f"\n{'='*60}", RANK)
    log_once(f"Starting Epoch {epoch + 1}/{NUM_EPOCHS}", RANK)
    log_once(f"{'='*60}\n", RANK)
    
    epoch_start = time.time()
    epoch_samples = 0
    
    pbar = tqdm(
        desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", 
        unit="steps", 
        disable=(RANK != 0)
    )

    for batch_idx, token_batch in enumerate(dataloader):
        step_start = time.time()
        global_step += 1
        
        if global_step % 1000 == 0:
            logi(f"[Rank {RANK}] Step {global_step} - START", RANK)
        
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

        batch_size = len(batch_token_ids)
        total_samples += batch_size
        epoch_samples += batch_size

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

        # Use accelerator's context manager for gradient accumulation
        with accelerator.accumulate(context_encoder, predictor):
            # Forward pass
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
                logi(f"Step {global_step} - WARNING: Invalid loss detected!", RANK)
                continue

            # Backward (handled by accelerator's accumulate context)
            accelerator.backward(loss)

            # Calculate gradient norm (only when we're about to step)
            if accelerator.sync_gradients:
                total_norm = calculate_gradient_norm(accelerator.unwrap_model(context_encoder))
            else:
                total_norm = 0.0

            # Step optimizers (only happens when accumulation is complete)
            optimizer.step()
            predictor_optimizer.step()
            optimizer.zero_grad()
            predictor_optimizer.zero_grad()

            # Update EMA target encoder (only on actual optimizer steps)
            if accelerator.sync_gradients:
                target_encoder.update(accelerator.unwrap_model(context_encoder))

        # Track timing
        step_time = time.time() - step_start
        epoch_elapsed = time.time() - epoch_start
        throughput = epoch_samples / epoch_elapsed if epoch_elapsed > 0 else 0
        
        # Update progress bar
        pbar.update(1)
        
        # Calculate ETA for current epoch
        steps_remaining = STEPS_PER_EPOCH - batch_idx - 1
        if throughput > 0 and steps_remaining > 0:
            samples_remaining = steps_remaining * BASE_BATCH_SIZE
            eta_seconds = samples_remaining / throughput
            eta_str = format_time(eta_seconds)
        else:
            eta_str = "?"

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}', 
            'samples/s': f'{throughput:.1f}',
            'ETA': eta_str
        })

        # Logging
        if global_step % STRATEGY_CONSTS["LOG_INTERVAL"] == 0:
            current_lr = optimizer.param_groups[0]['lr']
            
            training_loop_log(
                step=global_step,
                max_steps=MAX_STEPS,
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
                "train/step_time": step_time,
                "train/throughput": throughput,
                "train/total_samples": total_samples,
                "train/epoch": epoch + 1,
                "train/epoch_progress": batch_idx / STEPS_PER_EPOCH,
            }, step=global_step)

        # Checkpointing
        if STRATEGY_CONSTS.get("CHECKPOINT_INTERVAL") and global_step % STRATEGY_CONSTS["CHECKPOINT_INTERVAL"] == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=global_step,
                    engine=accelerator.unwrap_model(context_encoder),
                    optimizer=optimizer,
                    loss=loss.item(),
                    config=cast(Dict[str, Any], {
                        **STRATEGY_CONSTS,
                        "epoch": epoch + 1,
                        "epoch_step": batch_idx,
                        "total_samples": total_samples,
                    }),
                    rank=RANK,
                    is_final=False
                )

    pbar.close()
    
    epoch_time = time.time() - epoch_start
    log_once(f"\nEpoch {epoch + 1} completed in {format_time(epoch_time)}", RANK)
    log_once(f"Samples processed: {epoch_samples:,}", RANK)
    log_once(f"Average throughput: {epoch_samples / epoch_time:.1f} samples/s\n", RANK)
    
    # Save checkpoint at end of epoch
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            step=global_step,
            engine=accelerator.unwrap_model(context_encoder),
            optimizer=optimizer,
            loss=loss.item(),
            config=cast(Dict[str, Any], {
                **STRATEGY_CONSTS,
                "epoch": epoch + 1,
                "epoch_complete": True,
                "total_samples": total_samples,
            }),
            rank=RANK,
            is_final=(epoch == NUM_EPOCHS - 1)
        )

total_time = time.time() - training_start
log_once(f"\n{'='*60}", RANK)
log_once(f"Training Completed!", RANK)
log_once(f"{'='*60}", RANK)
log_once(f"Total epochs: {NUM_EPOCHS}", RANK)
log_once(f"Total steps: {global_step:,}", RANK)
log_once(f"Total samples: {total_samples:,}", RANK)
log_once(f"Total time: {format_time(total_time)}", RANK)
log_once(f"Average throughput: {total_samples / total_time:.1f} samples/s", RANK)

accelerator.end_training()