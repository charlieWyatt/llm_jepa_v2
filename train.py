from src.builders.dataloader_builder import dataloader_builder
from src.builders.patcher_builder import patcher_builder
from src.builders.masker_builder import masker_builder
from src.builders.encoder_builder import encoder_builder
from src.encoders.target_encoders.ema_target_encoder import ema_target_encoder
from src.builders.loss_calculator import loss_calculator
import os

load_dotenv()


training_dataset = os.getenv("TRAINING_DATASET")
patch_strategy = os.getenv("PATCH_STRATEGY")
target_mask_strategy = os.getenv("MASK_STRATEGY")
context_mask_strategy = os.getenv("CONTEXT_STRATEGY")
context_encoder_type = os.getenv("CONTEXT_ENCODER")
target_predictor_type = os.getenv("TARGET_PREDICTOR")

START_OF_CONTEXT_TOKEN = "<SOC>"
END_OF_CONTEXT_TOKEN = "<EOT>"

# Targets
target_creator = masker_builder(target_mask_strategy)

# Contexts
context_creator = masker_builder(context_mask_strategy)


context_encoder = encoder_builder(context_encoder_type)
target_encoder = encoder_builder(target_predictor_type)
target_predictor = target_encoder(context_encoder)


dataloader = dataloader_builder(training_dataset)

# Patches
patcher = patcher_builder(patch_strategy)
for raw_text in dataloader:
    patches = patcher.create_patches(raw_text)
    encoded_target = target_encoder(patches)

    targets = target_creator.create_spans(patches)
    context = context_creator.create_spans(patches)

    encoded_context = context_encoder(context)

    predicted_targets = []
    for target in targets:
        predicted_targets.append(target_predictor(START_OF_CONTEXT_TOKEN + encoded_context + END_OF_CONTEXT_TOKEN + target))

    loss = loss_calculator(encoded_target, predicted_targets)

    target_encoder.update()

    




    # Updates