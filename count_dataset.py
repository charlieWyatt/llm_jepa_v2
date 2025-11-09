"""
Standalone script to count dataset size.
Run this once before training: python count_dataset.py
"""

import json
import os
from src.builders.dataloader_builder import dataloader_builder
from src.builders.encoder_builder import encoder_builder
from config import STRATEGY_CONSTS

def count_dataset_size(tokenizer, batch_size=100, max_count=None):
    """Count the total number of samples in the dataset."""
    print("Counting dataset size...", flush=True)
    
    class CountTokenizer:
        def __init__(self, tok):
            self.tokenizer = tok
        
        def create_patches(self, text: str):
            return self.tokenizer.tokenize(text)
    
    count_tokenizer = CountTokenizer(tokenizer)
    count_dataloader = dataloader_builder(STRATEGY_CONSTS["TRAINING_DATASET"]).build(
        count_tokenizer, batch_size=batch_size
    )
    
    total_count = 0
    batch_num = 0
    
    for batch in count_dataloader:
        batch_count = len([t for t in batch if len(t) > 0])
        total_count += batch_count
        batch_num += 1
        
        # Print progress every 100 batches
        if batch_num % 100 == 0:
            print(f"  Processed {batch_num} batches, {total_count:,} samples so far...")
        
        if max_count and total_count >= max_count:
            print(f"  Reached max_count limit of {max_count}")
            break
    
    print(f"\n✓ Total dataset size: {total_count:,} samples\n")
    return total_count


def main():
    print("="*60)
    print("Dataset Size Counter")
    print("="*60)
    print(f"\nDataset: {STRATEGY_CONSTS['TRAINING_DATASET']}")
    print(f"Model: {STRATEGY_CONSTS['CONTEXT_MODEL_ID']}\n")
    
    # Build encoder to get tokenizer
    CONTEXT_ENCODER_CONFIG = {
        "hidden_size": 384,
        "num_layers": 6,
        "attention_window": 128,
    }
    
    print("Loading tokenizer...")
    context_encoder = encoder_builder(STRATEGY_CONSTS["CONTEXT_ENCODER"]).build(
        model_id=STRATEGY_CONSTS["CONTEXT_MODEL_ID"],
        config=CONTEXT_ENCODER_CONFIG,
    )
    tokenizer = context_encoder.tokenizer
    print("✓ Tokenizer loaded\n")
    
    # Count dataset
    total_size = count_dataset_size(
        tokenizer, 
        batch_size=100,
        max_count=None  # Set to a number like 10000 to test quickly
    )
    
    # Save to file
    output_dir = "./dataset_info"
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_info = {
        "dataset_name": STRATEGY_CONSTS["TRAINING_DATASET"],
        "total_samples": total_size,
        "model_id": STRATEGY_CONSTS["CONTEXT_MODEL_ID"],
        "batch_size_used_for_counting": 100,
    }
    
    output_path = os.path.join(output_dir, f"{STRATEGY_CONSTS['TRAINING_DATASET']}_size.json")
    
    with open(output_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"✓ Dataset info saved to: {output_path}")
    print("\nYou can now use this in your training script!")
    print("="*60)


if __name__ == "__main__":
    main()