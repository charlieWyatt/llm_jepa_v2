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
from dataclasses import dataclass
from typing import List, Tuple
import heapq
from src.experiment_trackers.experiment_tracker import create_tracker, ExperimentTracker
from src.losses.ntp_loss import compute_ntp_loss
from src.maskers.jepa_attention_mask import create_jepa_attention_mask

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ============================================
# HELPER FUNCTIONS
# ============================================
def validate_token_length(token_ids, max_seq_length, patch_size):
    """Truncate token sequences that exceed maximum allowed length."""
    max_allowed_tokens = patch_size * max_seq_length
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
# SAMPLE TRACKING
# ============================================

@dataclass
class TrainingSample:
    """Container for a training sample with its loss."""
    step: int
    loss: float
    text: str
    num_patches: int
    num_targets: int
    context_patches: str
    target_patches: str
    patch_texts: List[str]
    
    def __lt__(self, other):
        return self.loss < other.loss


class SampleTracker:
    """Tracks best and worst training samples during training."""
    
    def __init__(self, k: int = 5):
        self.k = k
        self.best_samples: List[TrainingSample] = []
        self.worst_samples: List[Tuple[float, TrainingSample]] = []
    
    def update(self, sample: TrainingSample):
        if len(self.best_samples) < self.k:
            heapq.heappush(self.best_samples, sample)
        elif sample.loss < self.best_samples[0].loss:
            heapq.heapreplace(self.best_samples, sample)
        
        neg_loss_sample = (-sample.loss, sample)
        if len(self.worst_samples) < self.k:
            heapq.heappush(self.worst_samples, neg_loss_sample)
        elif -sample.loss > self.worst_samples[0][0]:
            heapq.heapreplace(self.worst_samples, neg_loss_sample)
    
    def get_best_samples(self) -> List[TrainingSample]:
        return sorted(self.best_samples, key=lambda x: x.loss)
    
    def get_worst_samples(self) -> List[TrainingSample]:
        return sorted([s for _, s in self.worst_samples], key=lambda x: x.loss, reverse=True)


class SampleTracker:
    """Tracks best and worst training samples during training."""
    
    def __init__(self, k: int = 5):
        """
        Args:
            k: Number of best/worst samples to track
        """
        self.k = k
        # Min heap for best samples (smallest losses)
        self.best_samples: List[TrainingSample] = []
        # Max heap for worst samples (largest losses) - negate for max heap
        self.worst_samples: List[Tuple[float, TrainingSample]] = []
    
    def update(self, sample: TrainingSample):
        """Update best and worst samples with new sample."""
        # Update best samples (min heap)
        if len(self.best_samples) < self.k:
            heapq.heappush(self.best_samples, sample)
        elif sample.loss < self.best_samples[0].loss:
            heapq.heapreplace(self.best_samples, sample)
        
        # Update worst samples (max heap using negated loss)
        neg_loss_sample = (-sample.loss, sample)
        if len(self.worst_samples) < self.k:
            heapq.heappush(self.worst_samples, neg_loss_sample)
        elif -sample.loss > self.worst_samples[0][0]:
            heapq.heapreplace(self.worst_samples, neg_loss_sample)
    
    def get_best_samples(self) -> List[TrainingSample]:
        """Get k best samples sorted by loss (ascending)."""
        return sorted(self.best_samples, key=lambda x: x.loss)
    
    def get_worst_samples(self) -> List[TrainingSample]:
        """Get k worst samples sorted by loss (descending)."""
        return sorted([s for _, s in self.worst_samples], key=lambda x: x.loss, reverse=True)


def log_sample_analysis(
    tracker: ExperimentTracker,
    sample_tracker: SampleTracker,
    tokenizer,
    global_step: int,
    rank: int
):
    """
    Log best and worst training samples to the Experiment Tracker.
    
    Args:
        accelerator: Accelerator instance
        sample_tracker: SampleTracker with recorded samples
        tokenizer: Tokenizer for decoding
        global_step: Current training step
        rank: Process rank
    """
    if rank != 0:
        return
    
    best_samples = sample_tracker.get_best_samples()
    worst_samples = sample_tracker.get_worst_samples()
    
    best_text = "# BEST TRAINING SAMPLES (Lowest Loss)\n\n"
    for i, sample in enumerate(best_samples, 1):
        best_text += f"## Sample {i} (Step {sample.step}, Loss: {sample.loss:.4f})\n"
        best_text += f"**Patches:** {sample.num_patches} | **Targets:** {sample.num_targets}\n\n"
        best_text += f"**Original Text:**\n```\n{sample.text[:300]}...\n```\n\n"
        best_text += f"**Context Patches:** {sample.context_patches}\n\n"
        best_text += f"**Target Patches:** {sample.target_patches}\n\n"
        if sample.patch_texts:
            best_text += "**Patch Breakdown:**\n"
            for j, patch in enumerate(sample.patch_texts[:10]):  # Show first 10 patches
                best_text += f"  - Patch {j}: `{patch[:50]}...`\n"
            best_text += "\n"
        best_text += "---\n\n"
    
    worst_text = "# WORST TRAINING SAMPLES (Highest Loss)\n\n"
    for i, sample in enumerate(worst_samples, 1):
        worst_text += f"## Sample {i} (Step {sample.step}, Loss: {sample.loss:.4f})\n"
        worst_text += f"**Patches:** {sample.num_patches} | **Targets:** {sample.num_targets}\n\n"
        worst_text += f"**Original Text:**\n```\n{sample.text[:300]}...\n```\n\n"
        worst_text += f"**Context Patches:** {sample.context_patches}\n\n"
        worst_text += f"**Target Patches:** {sample.target_patches}\n\n"
        if sample.patch_texts:
            worst_text += "**Patch Breakdown:**\n"
            for j, patch in enumerate(sample.patch_texts[:10]):  # Show first 10 patches
                worst_text += f"  - Patch {j}: `{patch[:50]}...`\n"
            worst_text += "\n"
        worst_text += "---\n\n"
    
    tracker.log_text("samples/best_samples", best_text, step=global_step)
    tracker.log_text("samples/worst_samples", worst_text, step=global_step)
    
    # Also log statistics
    if best_samples:
        avg_best_loss = sum(s.loss for s in best_samples) / len(best_samples)
        tracker.log_metrics({"samples/avg_best_loss": avg_best_loss}, step=global_step)
    
    if worst_samples:
        avg_worst_loss = sum(s.loss for s in worst_samples) / len(worst_samples)
        tracker.log_metrics({"samples/avg_worst_loss": avg_worst_loss}, step=global_step)
    
    log_once(f"Logged {len(best_samples)} best and {len(worst_samples)} worst samples", rank)


def reconstruct_text_from_batch(
    token_batch: List[List[str]],
    batch_idx: int,
    tokenizer
) -> str:
    """
    Reconstruct text from tokenized batch.
    
    Args:
        token_batch: Batch of tokenized sequences
        batch_idx: Index of sample in batch
        tokenizer: Tokenizer for decoding
    
    Returns:
        Reconstructed text string
    """
    if batch_idx >= len(token_batch):
        return "[Sample index out of range]"
    
    tokens = token_batch[batch_idx]
    # Convert tokens back to text
    try:
        text = tokenizer.convert_tokens_to_string(tokens)
        return text
    except:
        # Fallback: just join tokens
        return " ".join(tokens)


def get_patch_texts(
    tokens: List[str],
    patch_size: int,
    tokenizer
) -> List[str]:
    """
    Break tokens into patches and convert to text.
    
    Args:
        tokens: List of tokens
        patch_size: Number of tokens per patch
        tokenizer: Tokenizer for conversion
    
    Returns:
        List of patch text strings
    """
    patches = []
    for i in range(0, len(tokens), patch_size):
        patch_tokens = tokens[i:i + patch_size]
        try:
            patch_text = tokenizer.convert_tokens_to_string(patch_tokens)
        except:
            patch_text = " ".join(patch_tokens)
        patches.append(patch_text)
    return patches


def describe_mask(mask: torch.Tensor, patch_texts: List[str] = None) -> str:
    """
    Create human-readable description of a mask.
    
    Args:
        mask: Binary mask tensor [L]
        patch_texts: Optional list of patch texts
    
    Returns:
        String description of mask
    """
    mask_np = mask.cpu().numpy()
    masked_indices = [i for i, v in enumerate(mask_np) if v == 1]
    
    if len(masked_indices) == 0:
        return "No patches selected"
    
    # Create description
    desc = f"Indices: {masked_indices[:20]}"  # Show first 20
    if len(masked_indices) > 20:
        desc += f"... ({len(masked_indices)} total)"
    
    if patch_texts:
        desc += "\nSelected patches:\n"
        for idx in masked_indices[:5]:  # Show first 5
            if idx < len(patch_texts):
                desc += f"  [{idx}]: {patch_texts[idx][:30]}...\n"
    
    return desc


# ============================================
# TRAINING
# ============================================
def run_train_jepa():
    NUM_EPOCHS = 1
    GRADIENT_ACCUMULATION_STEPS = 4
    BASE_BATCH_SIZE = STRATEGY_CONSTS.get("BATCH_SIZE", 2)
    EFFECTIVE_BATCH_SIZE = BASE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

    LAMBDA_NTP = STRATEGY_CONSTS.get("LAMBDA_NTP", 0.0)
    LAMBDA_JEPA = STRATEGY_CONSTS.get("LAMBDA_JEPA", 1.0)

    log_once(f"Training Configuration:", 0)
    log_once(f"  Epochs: {NUM_EPOCHS}", 0)
    log_once(f"  Base batch size per GPU: {BASE_BATCH_SIZE}", 0)
    log_once(f"  Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}", 0)
    log_once(f"  Effective batch size per GPU: {EFFECTIVE_BATCH_SIZE}", 0)
    log_once(f"  Loss weights - LAMBDA_NTP: {LAMBDA_NTP}, LAMBDA_JEPA: {LAMBDA_JEPA}", 0)

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

    if STRATEGY_CONSTS["EXPERIMENT_TRACKER"] == "wandb":
        tracker = create_tracker("wandb", log_dir=LOG_DIR)
    else:
        tracker = create_tracker(STRATEGY_CONSTS["EXPERIMENT_TRACKER"], accelerator=accelerator, log_dir=LOG_DIR)

    tracker.initialize(
        project_name="llm_jepa_training",
        config=STRATEGY_CONSTS,
        run_name=STRATEGY_CONSTS.get("WANDB_RUN_NAME", None)
    )

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

    # ----------------------------
    # Build components
    # ----------------------------
    log_once("Building components...", RANK)
    loss_calculator = loss_calculator_builder(
        STRATEGY_CONSTS["LOSS_CALCULATOR"]).build()

    if STRATEGY_CONSTS["CONTEXT_STRATEGY"] == "random":
        context_generator = RandomMaskGenerator(
            mask_ratio=STRATEGY_CONSTS["CONTEXT_MASK_RATIO"]
        )
    elif STRATEGY_CONSTS["CONTEXT_STRATEGY"] == "block":
        context_generator = BlockMaskGenerator(
            span_ratio=STRATEGY_CONSTS["CONTEXT_MASK_RATIO"]
        )
    else:
        raise ValueError(f"Unknown context strategy: {STRATEGY_CONSTS['CONTEXT_STRATEGY']}")

    if STRATEGY_CONSTS["MASK_STRATEGY"] == "random":
        target_generator = RandomMaskGenerator(
            mask_ratio=STRATEGY_CONSTS["TARGET_MASK_RATIO"]
        )
    elif STRATEGY_CONSTS["MASK_STRATEGY"] == "block":
        target_generator = BlockMaskGenerator(
            span_ratio=STRATEGY_CONSTS["TARGET_MASK_RATIO"]
        )
    else:
        raise ValueError(f"Unknown target strategy: {STRATEGY_CONSTS['MASK_STRATEGY']}")

    context_target_creator = ContextTargetCreator(
        context_generator=context_generator,
        target_generator=target_generator,
        num_targets=STRATEGY_CONSTS["NUM_TARGETS"]
    )

    context_encoder = encoder_builder(STRATEGY_CONSTS["CONTEXT_ENCODER"]).build(
        model_id=STRATEGY_CONSTS["CONTEXT_MODEL_ID"],
    )
    tokenizer = context_encoder.tokenizer

    hidden_size = context_encoder.hidden_size
    max_seq_length = context_encoder.max_seq_length

    # Setup for NTP if needed
    lm_head = None
    vocab_size = len(tokenizer)
    if LAMBDA_NTP > 0:
        if hasattr(context_encoder.model, 'lm_head'):
            lm_head = context_encoder.model.lm_head
            log_once("Using existing LM head from model", RANK)
        else:
            lm_head = nn.Linear(context_encoder.hidden_size, vocab_size, bias=False)
            log_once(f"Created new LM head: {context_encoder.hidden_size} -> {vocab_size}", RANK)

    # Calculate steps
    TOTAL_DATASET_SIZE = 9999999999999
    STEPS_PER_EPOCH = TOTAL_DATASET_SIZE // (BASE_BATCH_SIZE * WORLD_SIZE)
    if STRATEGY_CONSTS.get("MAX_STEPS") is not None:
        MAX_STEPS = STRATEGY_CONSTS["MAX_STEPS"]
        NUM_EPOCHS = 1
        log_once(f"Using MAX_STEPS override from config: {MAX_STEPS}", RANK)
    else:
        MAX_STEPS = STEPS_PER_EPOCH * NUM_EPOCHS
        log_once(f"Calculated MAX_STEPS from epochs: {MAX_STEPS}", RANK)

    log_once(f"Dataset Configuration:", RANK)
    log_once(f"  Total training steps: {MAX_STEPS:,}", RANK)

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

    predictor = SimplePredictor(hidden_dim=hidden_size)

    # Optimizer params
    encoder_params = list(context_encoder.parameters())
    if lm_head is not None and not hasattr(context_encoder.model, 'lm_head'):
        encoder_params += list(lm_head.parameters())

    optimizer = torch.optim.AdamW(
        encoder_params,
        lr=STRATEGY_CONSTS.get("LEARNING_RATE", 1e-4),
        weight_decay=0.01
    )

    predictor_optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=STRATEGY_CONSTS.get("LEARNING_RATE", 1e-4),
        weight_decay=0.01
    )

    # Target Encoder (EMA)
    target_encoder = ema_target_encoder(
        context_encoder, STRATEGY_CONSTS["DEFAULT_EMA_DECAY"])
    target_encoder.eval()
    for p in target_encoder.parameters():
        p.requires_grad = False

    # ACCELERATE PREPARE
    if lm_head is not None and not hasattr(context_encoder.model, 'lm_head'):
        context_encoder, predictor, lm_head, optimizer, predictor_optimizer, dataloader = accelerator.prepare(
            context_encoder, predictor, lm_head, optimizer, predictor_optimizer, dataloader
        )
    else:
        context_encoder, predictor, optimizer, predictor_optimizer, dataloader = accelerator.prepare(
            context_encoder, predictor, optimizer, predictor_optimizer, dataloader
        )
    target_encoder = target_encoder.to(device)

    # Checkpoint dir
    base_checkpoint_dir = STRATEGY_CONSTS.get("CHECKPOINT_DIR", "./checkpoints")
    checkpoint_dir = os.path.join(base_checkpoint_dir, f"patch_size_{STRATEGY_CONSTS['PATCH_SIZE']}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_once("Starting training loop...", RANK)

    # Progress tracking
    training_start = time.time()
    total_samples = 0
    global_step = 0
    epoch_samples = 0

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

            if global_step > MAX_STEPS:
                log_once(f"Reached MAX_STEPS={MAX_STEPS}, stopping training", RANK)
                pbar.close()
                break
            
            if global_step % 1000 == 0:
                logi(f"[Rank {RANK}] Step {global_step} - START", RANK)

            original_token_batch = token_batch
            
            # Prepare batch
            batch_token_ids = []
            max_len = 0

            for tokens in token_batch:
                if len(tokens) == 0:
                    continue
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                token_ids = validate_token_length(token_ids, max_seq_length, STRATEGY_CONSTS["PATCH_SIZE"])
                batch_token_ids.append(token_ids)
                max_len = max(max_len, len(token_ids))

            if len(batch_token_ids) == 0:
                continue

            B = len(batch_token_ids)
            total_samples += B
            epoch_samples += B

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

            # Use accelerator's context manager for gradient accumulation
            with accelerator.accumulate(context_encoder, predictor):
                # ============================================
                # NTP LOSS (if enabled)
                # ============================================
                loss_ntp = torch.tensor(0.0, device=device)
                if LAMBDA_NTP > 0:
                    unwrapped_encoder = accelerator.unwrap_model(context_encoder)
                    unwrapped_lm_head = accelerator.unwrap_model(lm_head) if lm_head is not None else unwrapped_encoder.model.lm_head
                    
                    loss_ntp = compute_ntp_loss(
                        encoder=unwrapped_encoder,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        lm_head=unwrapped_lm_head,
                        vocab_size=vocab_size,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                # ============================================
                # JEPA LOSS (if enabled)
                # ============================================
                loss_jepa = torch.tensor(0.0, device=device)
                L_patches = 0
                num_targets_total = 0
                context_mask = None
                target_mask = None
                
                if LAMBDA_JEPA > 0:
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
                    _, L_patches, D = context_patch_embeds.shape
                    dummy_input_ids = torch.zeros(B, L_patches, dtype=torch.long, device=device)

                    mask_pair = context_target_creator.create_context_and_targets(
                        input_ids=dummy_input_ids,
                        attention_mask=patched_attention_mask
                    )

                    context_mask = mask_pair.context_mask
                    target_mask = mask_pair.target_masks[0]
                    context_mask = context_mask * (1 - target_mask)

                    # Create attention mask that blocks targets
                    jepa_attention_mask = create_jepa_attention_mask(
                        patched_attention_mask,
                        context_mask,
                        target_mask
                    )

                    # Global max targets
                    local_max_targets = int(target_mask.sum(dim=1).max().item())
                    if IS_DIST:
                        max_targets_tensor = torch.tensor(local_max_targets, device=device)
                        torch.distributed.all_reduce(max_targets_tensor, op=torch.distributed.ReduceOp.MAX)
                        global_max_targets = int(max_targets_tensor.item())
                    else:
                        global_max_targets = local_max_targets

                    num_targets_total = int(target_mask.sum().item())

                    # Forward pass with JEPA attention mask
                    context_output = accelerator.unwrap_model(context_encoder).model(
                        inputs_embeds=context_patch_embeds,
                        attention_mask=jepa_attention_mask,
                    )
                    context_repr = context_output.last_hidden_state

                    with torch.no_grad():
                        target_output = target_encoder.model(
                            inputs_embeds=target_patch_embeds,
                            attention_mask=patched_attention_mask,
                        )
                        target_repr = target_output.last_hidden_state

                    # Apply masks and predict
                    masked_context_repr = context_repr * context_mask.unsqueeze(-1)

                    predicted_targets = predictor(
                        masked_context_repr,
                        target_mask,
                        context_mask,
                        global_max_targets
                    )

                    actual_targets = extract_target_representations(
                        target_repr,
                        target_mask,
                        global_max_targets
                    ).detach()

                    loss_jepa = loss_calculator(predicted_targets, actual_targets)

                # ============================================
                # COMBINED LOSS
                # ============================================
                total_loss = LAMBDA_NTP * loss_ntp + LAMBDA_JEPA * loss_jepa

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    logi(f"Step {global_step} - WARNING: Invalid loss detected!", RANK)
                    continue

                # ============================================
                # TRACK SAMPLES
                # ============================================
                if RANK == 0 and LAMBDA_JEPA > 0 and context_mask is not None:
                    for i in range(B):
                        sample_text = reconstruct_text_from_batch(original_token_batch, i, tokenizer)
                        
                        if i < len(original_token_batch):
                            patch_texts = get_patch_texts(
                                original_token_batch[i],
                                STRATEGY_CONSTS["PATCH_SIZE"],
                                tokenizer
                            )
                        else:
                            patch_texts = []
                        
                        context_desc = describe_mask(context_mask[i], patch_texts)
                        target_desc = describe_mask(target_mask[i], patch_texts)
                        
                        sample = TrainingSample(
                            step=global_step,
                            loss=total_loss.item(),
                            text=sample_text,
                            num_patches=L_patches,
                            num_targets=int(target_mask[i].sum().item()),
                            context_patches=context_desc,
                            target_patches=target_desc,
                            patch_texts=patch_texts
                        )
                        sample_tracker.update(sample)

                # Backward
                accelerator.backward(total_loss)

                # Calculate gradient norm
                if accelerator.sync_gradients:
                    total_norm = calculate_gradient_norm(accelerator.unwrap_model(context_encoder))
                else:
                    total_norm = 0.0

                # Step optimizers
                optimizer.step()
                predictor_optimizer.step()
                optimizer.zero_grad()
                predictor_optimizer.zero_grad()

                # Update EMA target encoder
                if accelerator.sync_gradients:
                    target_encoder.update(accelerator.unwrap_model(context_encoder))

            # Track timing
            step_time = time.time() - step_start
            epoch_elapsed = time.time() - epoch_start
            throughput = epoch_samples / epoch_elapsed if epoch_elapsed > 0 else 0
            
            pbar.update(1)
            
            steps_remaining = STEPS_PER_EPOCH - batch_idx - 1
            if throughput > 0 and steps_remaining > 0:
                samples_remaining = steps_remaining * BASE_BATCH_SIZE
                eta_seconds = samples_remaining / throughput
                eta_str = format_time(eta_seconds)
            else:
                eta_str = "?"

            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'ntp': f'{loss_ntp.item():.3f}' if LAMBDA_NTP > 0 else '-',
                'jepa': f'{loss_jepa.item():.3f}' if LAMBDA_JEPA > 0 else '-',
                'samples/s': f'{throughput:.1f}',
            })

            if global_step % STRATEGY_CONSTS["LOG_INTERVAL"] == 0 or global_step == 1:
                current_lr = optimizer.param_groups[0]['lr']
                
                training_loop_log(
                    step=global_step,
                    max_steps=MAX_STEPS,
                    loss=total_loss.item(),
                    learning_rate=current_lr,
                    grad_norm=total_norm,
                    batch_size=B,
                    num_patches=L_patches,
                    num_targets=num_targets_total,
                    rank=RANK
                )
                
                metrics = {
                    "train/loss_total": total_loss.item(),
                    "train/learning_rate": current_lr,
                    "train/batch_size": B,
                    "train/step_time": step_time,
                    "train/throughput": throughput,
                    "train/total_samples": total_samples,
                    "train/epoch": epoch + 1,
                    "train/epoch_progress": batch_idx / STEPS_PER_EPOCH,
                }
                
                if LAMBDA_NTP > 0:
                    metrics["train/loss_ntp"] = loss_ntp.item()
                if LAMBDA_JEPA > 0:
                    metrics["train/loss_jepa"] = loss_jepa.item()
                    metrics["train/num_patches"] = L_patches
                    metrics["train/num_targets"] = num_targets_total
                if accelerator.sync_gradients:
                    metrics["train/grad_norm"] = total_norm
                
                tracker.log_metrics(metrics, step=global_step)

            if global_step % SAMPLE_LOG_INTERVAL == 0:
                log_sample_analysis(
                    tracker=tracker,
                    sample_tracker=sample_tracker,
                    tokenizer=tokenizer,
                    global_step=global_step,
                    rank=RANK
                )

            if STRATEGY_CONSTS.get("CHECKPOINT_INTERVAL") and global_step % STRATEGY_CONSTS["CHECKPOINT_INTERVAL"] == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        step=global_step,
                        engine=accelerator.unwrap_model(context_encoder),
                        optimizer=optimizer,
                        loss=total_loss.item(),
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
        
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=global_step,
                engine=accelerator.unwrap_model(context_encoder),
                optimizer=optimizer,
                loss=total_loss.item(),
                config=cast(Dict[str, Any], {
                    **STRATEGY_CONSTS,
                    "epoch": epoch + 1,
                    "epoch_complete": True,
                    "total_samples": total_samples,
                }),
                rank=RANK,
                is_final=(epoch == NUM_EPOCHS - 1)
            )
        if global_step > MAX_STEPS:
            break

    total_time = time.time() - training_start
    log_once(f"\n{'='*60}", RANK)
    log_once(f"Training Completed!", RANK)
    log_once(f"{'='*60}", RANK)
    log_once(f"Total epochs: {NUM_EPOCHS}", RANK)
    log_once(f"Total steps: {global_step:,}", RANK)
    log_once(f"Total samples: {total_samples:,}", RANK)
    log_once(f"Total time: {format_time(total_time)}", RANK)
    log_once(f"Average throughput: {total_samples / total_time:.1f} samples/s", RANK)

    tracker.finalize()

if __name__ == "__main__":
    run_train_jepa()