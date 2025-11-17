import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, LongformerForMaskedLM

from eval.config import EvalConfig


class LongformerWrapper:
    def __init__(self, checkpoint_path: str, device: str):
        self.checkpoint_path = checkpoint_path
        self.device = device

    def load(self):
        print(f"Loading Longformer from {self.checkpoint_path}...")
        self.model = LongformerForMaskedLM.from_pretrained(self.checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def get_model_name(self):
        path = Path(self.checkpoint_path)
        if "checkpoint-" in path.name:
            step = path.name.split("-")[-1]
            return f"longformer_step_{step}"
        return "longformer"

    def extract_features(self, inputs, config: EvalConfig) -> np.ndarray:
        features = []
        with torch.no_grad():
            for i in tqdm(range(0, len(inputs), config.batch_size)):
                batch = inputs[i:i+config.batch_size]
                if isinstance(batch[0], tuple):
                    inp = self.tokenizer(
                        [b[0] for b in batch],
                        [b[1] for b in batch],
                        padding=True,
                        truncation=True,
                        max_length=config.max_length,
                        return_tensors="pt"
                    )
                else:
                    inp = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=config.max_length,
                        return_tensors="pt"
                    )

                inp = inp.to(self.device)
                outputs = self.model.longformer(**inp, output_hidden_states=True)
                hidden = outputs.hidden_states[-1]

                mask = inp["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

                features.append(pooled.cpu().numpy())
        return np.vstack(features)