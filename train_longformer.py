# Run this one with - `accelerate launch --num_processes=4 your_script.py`
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.dataloaders.dolma_sample_dataloader import DolmaSampleDataloader
from datasets import IterableDataset
from transformers import (
    AutoTokenizer,
    LongformerConfig,
    LongformerForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Configuration
MODEL_OR_PATH = "/g/data/oy87/cw9909/hf_models/allenai_longformer-base-4096"
OUTPUT_DIR = "/g/data/oy87/cw9909/longformer_mlm_ckpts"
FINAL_MODEL_DIR = "/g/data/oy87/cw9909/longformer_mlm_final"
MAX_STEPS = None
NUM_EPOCHS = 1
BLOCK_SIZE = 4096

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_OR_PATH, use_fast=True, local_files_only=True)

# Load config but initialize model with RANDOM weights
config = LongformerConfig.from_pretrained(MODEL_OR_PATH, local_files_only=True)
model = LongformerForMaskedLM(config)  # Random initialization!

print(f"✓ Model initialized with RANDOM weights: {model.num_parameters():,} parameters")

# Data pipeline
def data_generator():
    class IdentityPatcher:
        def create_patches(self, text):
            return text
    
    dataloader = DolmaSampleDataloader(
        patcher=IdentityPatcher(),
        batch_size=1000,
        streaming=True
    )
    
    for batch in dataloader:
        for text in batch:
            yield {"text": text}

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=BLOCK_SIZE,
        padding="max_length",
    )

raw_dataset = IterableDataset.from_generator(data_generator)
tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Training arguments for 4 GPUs
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    max_steps=MAX_STEPS,
    warmup_steps=500,
    logging_steps=50,
    save_steps=2500,
    save_total_limit=1,
    num_train_epochs=NUM_EPOCHS,
    bf16=True,
    dataloader_num_workers=4,
    ddp_find_unused_parameters=True,
    remove_unused_columns=False,
    report_to="tensorboard"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(FINAL_MODEL_DIR)