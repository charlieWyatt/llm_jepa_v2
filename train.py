from src.builders.dataloader_builder import dataloader_builder
from src.builders.patcher_builder import patcher_builder
from src.builders.masker_builder import masker_builder
from src.builders.encoder_builder import encoder_builder
from src.encoders.target_encoders.ema_target_encoder import ema_target_encoder
from src.builders.loss_calculator_builder import loss_calculator_builder
from config import STRATEGY_CONSTS


training_dataset = STRATEGY_CONSTS['TRAINING_DATASET']
patch_strategy = STRATEGY_CONSTS["PATCH_STRATEGY"]
target_mask_strategy = STRATEGY_CONSTS["MASK_STRATEGY"]
context_mask_strategy = STRATEGY_CONSTS["CONTEXT_STRATEGY"]
context_encoder_type = STRATEGY_CONSTS["CONTEXT_ENCODER"]
target_predictor_type = STRATEGY_CONSTS["TARGET_PREDICTOR"]
loss_calculator_type = STRATEGY_CONSTS["LOSS_CALCULATOR"]

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

loss_calculator = loss_calculator_builder(loss_calculator_type).build()
target_creator = masker_builder(target_mask_strategy).build()
context_creator = masker_builder(context_mask_strategy).build()
context_encoder = encoder_builder(
    context_encoder_type).build(CONTEXT_ENCODER_CONFIG)
tokenizer = context_encoder.tokenizer
# Always depends on the context encoder
target_encoder = ema_target_encoder(
    context_encoder, DEFAULT_EMA_DECAY)

# Always depends on the context encoder and target_encoder
patcher = patcher_builder(patch_strategy).build(
    context_encoder, target_encoder)
dataloader = dataloader_builder(training_dataset).build(
    patcher, batch_size=BATCH_SIZE)


target_predictor = encoder_builder(
    target_predictor_type).build(TARGET_ENCODER_CONFIG)

for patch_batch in dataloader:
    for patches in patch_batch:
        targets = target_creator.create_spans(patches)
        context = context_creator.create_spans(patches)

        encoded_context = context_encoder(context)
        encoded_target = target_encoder(patches)

        predicted_targets = []
        for target in targets:
            predicted_targets.append(
                target_predictor(START_OF_CONTEXT_TOKEN +
                                 encoded_context + END_OF_CONTEXT_TOKEN + target)
            )

        loss = loss_calculator(encoded_target, predicted_targets)
        target_encoder.update()

    # Updates
