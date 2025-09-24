import sys
import logging
from src.builders.dataloader_builder import dataloader_builder
from src.builders.patcher_builder import patcher_builder
from src.builders.masker_builder import masker_builder
from src.builders.encoder_builder import encoder_builder
from src.encoders.target_encoders.ema_target_encoder import ema_target_encoder
from src.builders.loss_calculator_builder import loss_calculator_builder
from config import STRATEGY_CONSTS
from torch.cuda.amp import GradScaler
from contextlib import nullcontext
import torch
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def ddp_init():
    if torch.cuda.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, local_rank, dist.get_world_size()
    return False, 0, 1

IS_DDP, LOCAL_RANK, WORLD_SIZE = ddp_init()
RANK = int(os.environ.get("RANK", "0"))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info("Starting training pipeline setup...")

use_cuda = torch.cuda.is_available()
device = torch.device(f"cuda:{LOCAL_RANK}") if use_cuda else torch.device("cpu")
amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_cuda else nullcontext()
scaler = GradScaler(enabled=use_cuda)


# Log only on the first rank
def logi(msg): 
    if RANK == 0: logger.info(msg)

logi(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")


def maybe_to_cuda(module):
    # If it looks like an nn.Module, move it
    if hasattr(module, "to") and callable(getattr(module, "to")):
        module.to(device)
    return module


def to_device(x):
    # Recursively move tensors in common containers
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [to_device(v) for v in x]
        return type(x)(t)
    return x


training_dataset = STRATEGY_CONSTS['TRAINING_DATASET']
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
BATCH_SIZE = 4

CONTEXT_ENCODER_CONFIG = {
    "hidden_size": 384,
    "num_layers": 6,
    "attention_window": 256
}

TARGET_ENCODER_CONFIG = {
    "hidden_size": 384,
    "num_layers": 2,
    "attention_window": 256
}

logi("Building components...")
loss_calculator = loss_calculator_builder(loss_calculator_type).build()
target_creator = masker_builder(target_mask_strategy).build()
context_creator = masker_builder(context_mask_strategy).build()
context_encoder = encoder_builder(context_encoder_type).build(
    model_id=STRATEGY_CONSTS["CONTEXT_MODEL_ID"],
    config=CONTEXT_ENCODER_CONFIG,
)
tokenizer = context_encoder.tokenizer
# Always depends on the context encoder
target_encoder = ema_target_encoder(
    context_encoder, DEFAULT_EMA_DECAY)

# Always depends on the context encoder and target_encoder
patcher = patcher_builder(patch_strategy).build(
    context_encoder, target_encoder)
dataloader = dataloader_builder(training_dataset).build(
    patcher, batch_size=BATCH_SIZE)


target_predictor = encoder_builder(target_predictor_type).build(
    model_id=STRATEGY_CONSTS["TARGET_MODEL_ID"],
    config=TARGET_ENCODER_CONFIG,
)


context_encoder = maybe_to_cuda(context_encoder)
target_encoder = maybe_to_cuda(target_encoder)
target_predictor = maybe_to_cuda(target_predictor)
loss_calculator = maybe_to_cuda(loss_calculator)

if IS_DDP:
    # Only wrap the *trainable* modules you step with the optimizer
    context_encoder = DDP(context_encoder, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True)
    target_predictor = DDP(target_predictor, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True)


params = []
for m in (context_encoder, target_encoder, target_predictor):
    if hasattr(m, "parameters"):
        params += list(m.parameters())
optimizer = torch.optim.AdamW([p for p in params if p.requires_grad], lr=1e-4)

torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


logi("Components built successfully.")

logi("Starting training loop...")
for patch_batch in dataloader:
    optimizer.zero_grad(set_to_none=True)

    patch_batch = to_device(patch_batch)
    for patches in patch_batch:
        targets = target_creator.create_spans(patches)
        context = context_creator.create_spans(patches)

        targets = to_device(targets)
        context = to_device(context)

        with amp_ctx:
            encoded_context = context_encoder(context)
            encoded_target = target_encoder(patches)

            # If your encoders return dicts or non-tensors, adapt `to_device`/loss accordingly
            predicted_targets = []
            for target in targets:
                # If target/context need tokenization, make sure tensors end up on `device`
                pred = target_predictor(
                    START_OF_CONTEXT_TOKEN + encoded_context + END_OF_CONTEXT_TOKEN + target
                )
                predicted_targets.append(pred)

            loss = loss_calculator(encoded_target, predicted_targets)

        # Backprop + step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        target_encoder.update()

        logi.debug(f"    Loss: {loss}")
        logi("Updated!")

logi("Training loop finished.")
# Updates


if IS_DDP:
    dist.barrier()
    dist.destroy_process_group()