from src.builders.dataloader_builder import dataloader_builder
from src.builders.patcher_builder import patcher_builder
from src.builders.masker_builder import masker_builder
from src.builders.encoder_builder import encoder_builder
from src.encoders.target_encoders.ema_target_encoder import ema_target_encoder
from src.builders.loss_calculator_builder import loss_calculator_builder
from dotenv import load_dotenv
import os

load_dotenv()


training_dataset = os.getenv("TRAINING_DATASET")
patch_strategy = os.getenv("PATCH_STRATEGY")
target_mask_strategy = os.getenv("MASK_STRATEGY")
context_mask_strategy = os.getenv("CONTEXT_STRATEGY")
context_encoder_type = os.getenv("CONTEXT_ENCODER")
target_predictor_type = os.getenv("TARGET_PREDICTOR")
loss_calculator_type = os.getenv("LOSS_CALCULATOR")

START_OF_CONTEXT_TOKEN = "<SOC>"
END_OF_CONTEXT_TOKEN = "<EOT>"

loss_calculator = loss_calculator_builder(loss_calculator_type).build()

# Targets
target_creator = masker_builder(target_mask_strategy).build()

# Contexts
context_creator = masker_builder(context_mask_strategy).build()


context_encoder = encoder_builder(context_encoder_type).build()
target_encoder = encoder_builder(target_predictor_type).build()
patcher = patcher_builder(patch_strategy).build()
dataloader = dataloader_builder(training_dataset).build()

target_predictor = target_encoder(context_encoder)

for raw_text in dataloader:
    patches = patcher.create_patches(raw_text)
    encoded_target = target_encoder(patches)

    targets = target_creator.create_spans(patches)
    context = context_creator.create_spans(patches)

    encoded_context = context_encoder(context)

    predicted_targets = []
    for target in targets:
        predicted_targets.append(target_predictor(
            START_OF_CONTEXT_TOKEN + encoded_context + END_OF_CONTEXT_TOKEN + target))

    loss = loss_calculator(encoded_target, predicted_targets)

    target_encoder.update()

    # Updates
