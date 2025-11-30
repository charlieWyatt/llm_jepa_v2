from datasets import load_dataset
import os

LONGBENCH_ROOT = "/g/data/oy87/cw9909/longbench_data"

print("Downloading LongBench v2...")
dataset = load_dataset("THUDM/LongBench-v2")

print(f"Dataset structure: {dataset}")
print(f"Saving to {LONGBENCH_ROOT}...")

# LongBench v2 only has one split called 'train' (which is actually test data)
dataset.save_to_disk(LONGBENCH_ROOT)

print("Download complete!")
print(f"\nDataset info:")
print(f"Number of examples: {len(dataset['train'])}")
print(f"Features: {dataset['train'].features}")