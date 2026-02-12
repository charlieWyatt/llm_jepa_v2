"""
Paired Target-Context Cross-View JEPA Training.

Context = Text (NL descriptions), Target = Code (Regex patterns).
No masking generators needed -- text is always context, code is always target.
"""

import json
import os
import torch
import torch.nn as nn
from typing import Any, cast, Dict, List, Tuple
import time
from tqdm import tqdm
from dataclasses import dataclass
import heapq

from accelerate import Accelerator

from src.maskers.utils import extract_target_representations
from src.patchers.helpers import create_patched_embeddings
from src.builders.dataloader_builder import dataloader_builder
from src.builders.patcher_builder import patcher_builder
from src.builders.encoder_builder import encoder_builder
from src.encoders.target_encoders.ema_target_encoder import ema_target_encoder
from src.builders.loss_calculator_builder import loss_calculator_builder
from config_paired_tgt_ctx import PAIRED_TGT_CTX_CONSTS
from src.logging.logging_helpers import (
    logi, log_once, get_logger, training_loop_log,
    save_checkpoint, calculate_gradient_norm,
)
from src.predictors.simple_predictor import SimplePredictor
from src.experiment_trackers.experiment_tracker import create_tracker, ExperimentTracker
from src.losses.ntp_loss import compute_ntp_loss_code_only

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ============================================
# HELPER FUNCTIONS
# ============================================

def create_causal_attention_mask_4d(
    attention_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Standard causal + padding mask in 4D additive format.

    Args:
        attention_mask: [B, L] with 1 for real tokens, 0 for padding.
        device: torch device.

    Returns:
        [B, 1, L, L] additive attention mask (0 = attend, -10000 = ignore).
    """
    B, L = attention_mask.shape
    causal = torch.tril(torch.ones(L, L, device=device))
    causal = causal.unsqueeze(0).unsqueeze(0).expand(B, 1, L, L).clone()
    padding = attention_mask.unsqueeze(1).unsqueeze(2).float()  # [B, 1, 1, L]
    causal = causal * padding
    return (1.0 - causal) * -10000.0


def truncate_to_patch_aligned(token_ids: list, max_tokens: int, patch_size: int) -> list:
    """Truncate token IDs to max_tokens, then trim to be divisible by patch_size."""
    token_ids = token_ids[:max_tokens]
    aligned_len = (len(token_ids) // patch_size) * patch_size
    if aligned_len == 0 and len(token_ids) > 0:
        # Pad up to patch_size so we have at least one patch
        token_ids = token_ids + [0] * (patch_size - len(token_ids))
        aligned_len = patch_size
    return token_ids[:aligned_len]


def pad_and_tensorize(
    batch_token_ids: List[list],
    pad_token_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a list of variable-length token ID lists and convert to tensors.

    Returns:
        input_ids: [B, max_len]
        attention_mask: [B, max_len]
    """
    max_len = max(len(ids) for ids in batch_token_ids)
    padded_ids = []
    attention_masks = []
    for ids in batch_token_ids:
        pad_len = max_len - len(ids)
        padded_ids.append(ids + [pad_token_id] * pad_len)
        attention_masks.append([1] * len(ids) + [0] * pad_len)
    return (
        torch.tensor(padded_ids, device=device),
        torch.tensor(attention_masks, device=device).float(),
    )


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


# ============================================
# SAMPLE TRACKING
# ============================================

@dataclass
class PairedTrainingSample:
    step: int
    loss: float
    raw_text: str
    raw_code: str
    num_text_patches: int
    num_code_patches: int

    def __lt__(self, other):
        return self.loss < other.loss


class SampleTracker:
    def __init__(self, k: int = 5):
        self.k = k
        self.best_samples: List[PairedTrainingSample] = []
        self.worst_samples: List[Tuple[float, PairedTrainingSample]] = []

    def update(self, sample: PairedTrainingSample):
        if len(self.best_samples) < self.k:
            heapq.heappush(self.best_samples, sample)
        elif sample.loss < self.best_samples[0].loss:
            heapq.heapreplace(self.best_samples, sample)

        neg = (-sample.loss, sample)
        if len(self.worst_samples) < self.k:
            heapq.heappush(self.worst_samples, neg)
        elif -sample.loss > self.worst_samples[0][0]:
            heapq.heapreplace(self.worst_samples, neg)

    def get_best(self) -> List[PairedTrainingSample]:
        return sorted(self.best_samples, key=lambda x: x.loss)

    def get_worst(self) -> List[PairedTrainingSample]:
        return sorted([s for _, s in self.worst_samples], key=lambda x: x.loss, reverse=True)


def log_sample_analysis(
    tracker: ExperimentTracker,
    sample_tracker: SampleTracker,
    global_step: int,
    rank: int,
):
    if rank != 0:
        return

    best = sample_tracker.get_best()
    worst = sample_tracker.get_worst()

    def _fmt(samples, header):
        text = f"# {header}\n\n"
        for i, s in enumerate(samples, 1):
            text += (
                f"## Sample {i} (Step {s.step}, Loss: {s.loss:.4f})\n"
                f"**Text patches:** {s.num_text_patches} | **Code patches:** {s.num_code_patches}\n\n"
                f"**NL:** `{s.raw_text[:200]}`\n\n"
                f"**Regex:** `{s.raw_code[:200]}`\n\n---\n\n"
            )
        return text

    tracker.log_text("samples/best", _fmt(best, "BEST (lowest loss)"), step=global_step)
    tracker.log_text("samples/worst", _fmt(worst, "WORST (highest loss)"), step=global_step)

    if best:
        tracker.log_metrics({"samples/avg_best_loss": sum(s.loss for s in best) / len(best)}, step=global_step)
    if worst:
        tracker.log_metrics({"samples/avg_worst_loss": sum(s.loss for s in worst) / len(worst)}, step=global_step)


# ============================================
# TRAINING
# ============================================

def run_train_paired_tgt_ctx():
    CFG = PAIRED_TGT_CTX_CONSTS

    NUM_EPOCHS = 1
    GRADIENT_ACCUMULATION_STEPS = 4
    BASE_BATCH_SIZE = CFG.get("BATCH_SIZE", 4)
    LAMBDA_NTP = CFG.get("LAMBDA_NTP", 1.0)
    LAMBDA_JEPA = CFG.get("LAMBDA_JEPA", 1.0)
    PATCH_SIZE = CFG.get("PATCH_SIZE", 2)
    MAX_TEXT_TOKENS = CFG.get("MAX_TEXT_TOKENS", 256)
    MAX_CODE_TOKENS = CFG.get("MAX_CODE_TOKENS", 128)

    log_once("Paired Target-Context Training Configuration:", 0)
    log_once(f"  Base batch size per GPU: {BASE_BATCH_SIZE}", 0)
    log_once(f"  Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}", 0)
    log_once(f"  Loss weights - LAMBDA_NTP: {LAMBDA_NTP}, LAMBDA_JEPA: {LAMBDA_JEPA}", 0)
    log_once(f"  Patch size: {PATCH_SIZE}", 0)
    log_once(f"  Max text tokens: {MAX_TEXT_TOKENS}, Max code tokens: {MAX_CODE_TOKENS}", 0)

    # ============================================
    # ACCELERATE INITIALIZATION
    # ============================================
    LOG_DIR = "/g/data/oy87/cw9909/llm_jepa/logs"
    os.makedirs(LOG_DIR, exist_ok=True)

    sample_tracker = SampleTracker(k=5)
    SAMPLE_LOG_INTERVAL = 5000

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    )

    if CFG["EXPERIMENT_TRACKER"] == "wandb":
        tracker = create_tracker("wandb", log_dir=LOG_DIR)
    else:
        tracker = create_tracker(CFG["EXPERIMENT_TRACKER"], accelerator=accelerator, log_dir=LOG_DIR)

    tracker.initialize(
        project_name="paired_tgt_ctx_nlrx",
        config=CFG,
        run_name=CFG.get("WANDB_RUN_NAME", None),
    )

    RANK = accelerator.process_index
    LOCAL_RANK = accelerator.local_process_index
    WORLD_SIZE = accelerator.num_processes
    IS_DIST = WORLD_SIZE > 1
    device = accelerator.device

    log_once("Starting paired target-context pipeline setup...", RANK)
    log_once(f"CUDA available: {torch.cuda.is_available()}", RANK)
    if torch.cuda.is_available():
        logi(f"GPU[{LOCAL_RANK}]: {torch.cuda.get_device_name(LOCAL_RANK)}", RANK)

    log_once(json.dumps(CFG), RANK)

    # ----------------------------
    # Build components
    # ----------------------------
    log_once("Building components...", RANK)

    loss_calculator = loss_calculator_builder(CFG["LOSS_CALCULATOR"]).build()

    context_encoder = encoder_builder(CFG["CONTEXT_ENCODER"]).build(
        model_id=CFG["CONTEXT_MODEL_ID"],
    )
    tokenizer = context_encoder.tokenizer
    hidden_size = context_encoder.hidden_size

    # LM head for NTP
    lm_head = None
    vocab_size = len(tokenizer)
    if LAMBDA_NTP > 0:
        if hasattr(context_encoder.model, 'lm_head'):
            lm_head = context_encoder.model.lm_head
            log_once("Using existing LM head from model", RANK)
        else:
            lm_head = nn.Linear(context_encoder.hidden_size, vocab_size, bias=False)
            log_once(f"Created new LM head: {context_encoder.hidden_size} -> {vocab_size}", RANK)

    # Patcher for embeddings
    embedding_patcher = patcher_builder(CFG["PATCH_STRATEGY"]).build(
        patch_size=PATCH_SIZE)
    log_once(f"Embedding patcher: {embedding_patcher}", RANK)

    # TextTokenizer wrapper for the dataloader
    class TextTokenizer:
        def __init__(self, tok):
            self.tokenizer = tok
        def create_patches(self, text: str):
            return self.tokenizer.tokenize(text)

    text_tokenizer = TextTokenizer(tok=tokenizer)
    dataloader = dataloader_builder(CFG["TRAINING_DATASET"]).build(
        text_tokenizer,
        batch_size=BASE_BATCH_SIZE,
        data_path=CFG.get("DATA_PATH"),
    )

    predictor = SimplePredictor(hidden_dim=hidden_size)

    # Max steps
    MAX_STEPS = CFG.get("MAX_STEPS", 10000)
    log_once(f"Total training steps: {MAX_STEPS:,}", RANK)

    # Optimizers
    encoder_params = list(context_encoder.parameters())
    if lm_head is not None and not hasattr(context_encoder.model, 'lm_head'):
        encoder_params += list(lm_head.parameters())

    optimizer = torch.optim.AdamW(
        encoder_params,
        lr=CFG.get("LEARNING_RATE", 5e-4),
        weight_decay=0.01,
    )
    predictor_optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=CFG.get("LEARNING_RATE", 5e-4),
        weight_decay=0.01,
    )

    # Target encoder (EMA)
    target_encoder = ema_target_encoder(context_encoder, CFG["DEFAULT_EMA_DECAY"])
    target_encoder.eval()
    for p in target_encoder.parameters():
        p.requires_grad = False

    # Accelerate prepare
    if lm_head is not None and not hasattr(context_encoder.model, 'lm_head'):
        context_encoder, predictor, lm_head, optimizer, predictor_optimizer = accelerator.prepare(
            context_encoder, predictor, lm_head, optimizer, predictor_optimizer
        )
    else:
        context_encoder, predictor, optimizer, predictor_optimizer = accelerator.prepare(
            context_encoder, predictor, optimizer, predictor_optimizer
        )
    target_encoder = target_encoder.to(device)

    # Checkpoint dir
    base_checkpoint_dir = CFG.get("CHECKPOINT_DIR", "./checkpoints")
    checkpoint_dir = os.path.join(base_checkpoint_dir, f"patch_size_{PATCH_SIZE}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_once("Starting training loop...", RANK)

    # ============================================
    # TRAINING LOOP
    # ============================================
    training_start = time.time()
    total_samples = 0
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        log_once(f"\n{'='*60}", RANK)
        log_once(f"Starting Epoch {epoch + 1}/{NUM_EPOCHS}", RANK)
        log_once(f"{'='*60}\n", RANK)

        epoch_start = time.time()
        epoch_samples = 0

        pbar = tqdm(
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}",
            unit="steps",
            disable=(RANK != 0),
        )

        for batch_idx, batch in enumerate(dataloader):
            step_start = time.time()
            global_step += 1

            if global_step > MAX_STEPS:
                log_once(f"Reached MAX_STEPS={MAX_STEPS}, stopping training", RANK)
                pbar.close()
                break

            text_token_batch = batch["text_tokens"]
            code_token_batch = batch["code_tokens"]
            raw_texts = batch["raw_text"]
            raw_codes = batch["raw_code"]

            # -----------------------------------------------
            # Tokenize, truncate, align to patch_size
            # -----------------------------------------------
            batch_text_ids = []
            batch_code_ids = []

            for text_tokens, code_tokens in zip(text_token_batch, code_token_batch):
                if len(text_tokens) == 0 or len(code_tokens) == 0:
                    continue
                t_ids = tokenizer.convert_tokens_to_ids(text_tokens)
                c_ids = tokenizer.convert_tokens_to_ids(code_tokens)
                t_ids = truncate_to_patch_aligned(t_ids, MAX_TEXT_TOKENS, PATCH_SIZE)
                c_ids = truncate_to_patch_aligned(c_ids, MAX_CODE_TOKENS, PATCH_SIZE)
                if len(t_ids) == 0 or len(c_ids) == 0:
                    continue
                batch_text_ids.append(t_ids)
                batch_code_ids.append(c_ids)

            if len(batch_text_ids) == 0:
                continue

            B = len(batch_text_ids)
            total_samples += B
            epoch_samples += B

            # Pad text and code separately
            text_input_ids, text_attention_mask = pad_and_tensorize(
                batch_text_ids, tokenizer.pad_token_id, device)
            code_input_ids, code_attention_mask = pad_and_tensorize(
                batch_code_ids, tokenizer.pad_token_id, device)

            with accelerator.accumulate(context_encoder, predictor):
                unwrapped_encoder = accelerator.unwrap_model(context_encoder)

                # ============================================
                # JEPA LOSS
                # ============================================
                loss_jepa = torch.tensor(0.0, device=device)
                num_text_patches = 0
                num_code_patches = 0

                if LAMBDA_JEPA > 0:
                    embedding_layer = unwrapped_encoder.get_input_embeddings()

                    # Patch text embeddings (context)
                    text_patch_embeds, text_patched_attn = create_patched_embeddings(
                        input_ids=text_input_ids,
                        attention_mask=text_attention_mask,
                        embedding_layer=embedding_layer,
                        patcher=embedding_patcher,
                        patch_size=PATCH_SIZE,
                    )

                    # Patch code embeddings (target)
                    with torch.no_grad():
                        target_embedding_layer = target_encoder.get_input_embeddings()
                        code_patch_embeds, code_patched_attn = create_patched_embeddings(
                            input_ids=code_input_ids,
                            attention_mask=code_attention_mask,
                            embedding_layer=target_embedding_layer,
                            patcher=embedding_patcher,
                            patch_size=PATCH_SIZE,
                        )

                    num_text_patches = text_patch_embeds.shape[1]
                    num_code_patches = code_patch_embeds.shape[1]

                    # Context encoder forward on text patches
                    text_4d_mask = create_causal_attention_mask_4d(text_patched_attn, device)
                    context_output = unwrapped_encoder.model(
                        inputs_embeds=text_patch_embeds,
                        attention_mask=text_4d_mask,
                    )
                    text_repr = context_output.last_hidden_state  # [B, N_text, D]

                    # Target encoder (EMA, no grad) forward on code patches
                    code_4d_mask = create_causal_attention_mask_4d(code_patched_attn, device)
                    with torch.no_grad():
                        target_output = target_encoder.model(
                            inputs_embeds=code_patch_embeds,
                            attention_mask=code_4d_mask,
                        )
                        code_repr = target_output.last_hidden_state  # [B, N_code, D]

                    # Global max code patches (for distributed padding)
                    local_max_code = int(code_patched_attn.sum(dim=1).max().item())
                    if IS_DIST:
                        max_code_tensor = torch.tensor(local_max_code, device=device)
                        torch.distributed.all_reduce(max_code_tensor, op=torch.distributed.ReduceOp.MAX)
                        global_max_code = int(max_code_tensor.item())
                    else:
                        global_max_code = local_max_code

                    # Predictor: mean-pool text repr -> predict code repr
                    # target_mask arg is unused internally by SimplePredictor,
                    # pass a dummy; context_mask = text_patched_attn
                    dummy_target_mask = torch.zeros(B, num_text_patches, device=device)
                    predicted_code = predictor(
                        text_repr,
                        dummy_target_mask,
                        text_patched_attn,
                        global_max_code,
                    )

                    # Extract target: all real code patches
                    actual_code = extract_target_representations(
                        code_repr,
                        code_patched_attn,
                        global_max_code,
                    ).detach()

                    loss_jepa = loss_calculator(predicted_code, actual_code)

                # ============================================
                # NTP LOSS (code-only supervision)
                # ============================================
                loss_ntp = torch.tensor(0.0, device=device)
                if LAMBDA_NTP > 0:
                    unwrapped_lm_head = (
                        accelerator.unwrap_model(lm_head)
                        if lm_head is not None
                        else unwrapped_encoder.model.lm_head
                    )

                    loss_ntp = compute_ntp_loss_code_only(
                        encoder=unwrapped_encoder,
                        text_input_ids=text_input_ids,
                        code_input_ids=code_input_ids,
                        text_attention_mask=text_attention_mask,
                        code_attention_mask=code_attention_mask,
                        lm_head=unwrapped_lm_head,
                        vocab_size=vocab_size,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                # ============================================
                # COMBINED LOSS
                # ============================================
                total_loss = LAMBDA_JEPA * loss_jepa + LAMBDA_NTP * loss_ntp

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    logi(f"Step {global_step} - WARNING: Invalid loss detected!", RANK)
                    continue

                # ============================================
                # TRACK SAMPLES
                # ============================================
                if RANK == 0:
                    for i in range(min(B, len(raw_texts))):
                        sample = PairedTrainingSample(
                            step=global_step,
                            loss=total_loss.item(),
                            raw_text=raw_texts[i] if i < len(raw_texts) else "",
                            raw_code=raw_codes[i] if i < len(raw_codes) else "",
                            num_text_patches=num_text_patches,
                            num_code_patches=num_code_patches,
                        )
                        sample_tracker.update(sample)

                # Backward
                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    total_norm = calculate_gradient_norm(unwrapped_encoder)
                else:
                    total_norm = 0.0

                optimizer.step()
                predictor_optimizer.step()
                optimizer.zero_grad()
                predictor_optimizer.zero_grad()

                # Update EMA target encoder
                if accelerator.sync_gradients:
                    target_encoder.update(unwrapped_encoder)

            # Timing
            step_time = time.time() - step_start
            epoch_elapsed = time.time() - epoch_start
            throughput = epoch_samples / epoch_elapsed if epoch_elapsed > 0 else 0

            pbar.update(1)
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'ntp': f'{loss_ntp.item():.3f}' if LAMBDA_NTP > 0 else '-',
                'jepa': f'{loss_jepa.item():.3f}' if LAMBDA_JEPA > 0 else '-',
                'samples/s': f'{throughput:.1f}',
            })

            if global_step % CFG["LOG_INTERVAL"] == 0 or global_step == 1:
                current_lr = optimizer.param_groups[0]['lr']
                training_loop_log(
                    step=global_step,
                    max_steps=MAX_STEPS,
                    loss=total_loss.item(),
                    learning_rate=current_lr,
                    grad_norm=total_norm,
                    batch_size=B,
                    num_patches=num_text_patches,
                    num_targets=num_code_patches,
                    rank=RANK,
                )
                metrics = {
                    "train/loss_total": total_loss.item(),
                    "train/learning_rate": current_lr,
                    "train/batch_size": B,
                    "train/step_time": step_time,
                    "train/throughput": throughput,
                    "train/total_samples": total_samples,
                    "train/epoch": epoch + 1,
                }
                if LAMBDA_NTP > 0:
                    metrics["train/loss_ntp"] = loss_ntp.item()
                if LAMBDA_JEPA > 0:
                    metrics["train/loss_jepa"] = loss_jepa.item()
                    metrics["train/num_text_patches"] = num_text_patches
                    metrics["train/num_code_patches"] = num_code_patches
                if accelerator.sync_gradients:
                    metrics["train/grad_norm"] = total_norm
                tracker.log_metrics(metrics, step=global_step)

            if global_step % SAMPLE_LOG_INTERVAL == 0:
                log_sample_analysis(
                    tracker=tracker,
                    sample_tracker=sample_tracker,
                    global_step=global_step,
                    rank=RANK,
                )

            if CFG.get("CHECKPOINT_INTERVAL") and global_step % CFG["CHECKPOINT_INTERVAL"] == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        step=global_step,
                        engine=unwrapped_encoder,
                        optimizer=optimizer,
                        loss=total_loss.item(),
                        config=cast(Dict[str, Any], {
                            **CFG,
                            "epoch": epoch + 1,
                            "total_samples": total_samples,
                        }),
                        rank=RANK,
                        is_final=False,
                    )

        pbar.close()

        epoch_time = time.time() - epoch_start
        log_once(f"\nEpoch {epoch + 1} completed in {format_time(epoch_time)}", RANK)
        log_once(f"Samples processed: {epoch_samples:,}", RANK)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=global_step,
                engine=accelerator.unwrap_model(context_encoder),
                optimizer=optimizer,
                loss=total_loss.item(),
                config=cast(Dict[str, Any], {
                    **CFG,
                    "epoch": epoch + 1,
                    "epoch_complete": True,
                    "total_samples": total_samples,
                }),
                rank=RANK,
                is_final=(epoch == NUM_EPOCHS - 1),
            )
        if global_step > MAX_STEPS:
            break

    total_time = time.time() - training_start
    log_once(f"\n{'='*60}", RANK)
    log_once("Paired Target-Context Training Completed!", RANK)
    log_once(f"{'='*60}", RANK)
    log_once(f"Total steps: {global_step:,}", RANK)
    log_once(f"Total samples: {total_samples:,}", RANK)
    log_once(f"Total time: {format_time(total_time)}", RANK)

    tracker.finalize()


if __name__ == "__main__":
    run_train_paired_tgt_ctx()
