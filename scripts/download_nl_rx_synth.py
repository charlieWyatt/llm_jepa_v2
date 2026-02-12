"""
Download NL-RX-SYNTH dataset for offline use on Gadi HPC.

Run this on a login node with internet access:
    python scripts/download_nl_rx_synth.py
"""

from datasets import load_dataset
from pathlib import Path
import json

OUTPUT_DIR = "/g/data/oy87/cw9909/datasets/nl_rx_synth"


def main():
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Downloading NL-RX-Synth from HuggingFace...")
    ds = load_dataset("CatherineYeh/NL-RX-Synth", split="train")

    out_file = output_path / "train.json"
    print(f"Saving {len(ds)} samples to {out_file}...")

    records = [{"nl": ex["nl"], "regex": ex["regex"]} for ex in ds]
    with open(out_file, "w") as f:
        json.dump(records, f)

    print(f"Done. Saved {len(records)} samples.")


if __name__ == "__main__":
    main()
