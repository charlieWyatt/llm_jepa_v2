import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from eval.config import EvalConfig

class JEPAWrapper:
    def __init__(self, checkpoint_path: str, device: str):
        self.checkpoint_path = checkpoint_path
        self.device = device

    def get_model_name(self):
        name = Path(self.checkpoint_path).stem
        step = name.split("_")[-1]
        return f"jepa_step_{step}"

    def load(self):
        from src.builders.encoder_builder import encoder_builder
        from config import STRATEGY_CONSTS
        
        print(f"Loading JEPA checkpoint from {self.checkpoint_path}...")
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

        embedding_shape = checkpoint["model_state_dict"]["model.embeddings.word_embeddings.weight"].shape
        hidden_size = embedding_shape[1]
        num_layers = sum("model.encoder.layer" in k and "query.weight" in k 
                         for k in checkpoint["model_state_dict"].keys())

        model_config = {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "attention_window": 512,
        }

        encoder = encoder_builder(STRATEGY_CONSTS["CONTEXT_ENCODER"]).build(
            model_id=STRATEGY_CONSTS["CONTEXT_MODEL_ID"],
            config=model_config
        )

        encoder.load_state_dict(checkpoint["model_state_dict"])
        self.model = encoder
        self.tokenizer = encoder.tokenizer
        self.model.to(self.device)
        self.model.eval()

    def extract_features(self, inputs, config: EvalConfig) -> np.ndarray:
        feats = []
        with torch.no_grad():
            for i in tqdm(range(0, len(inputs), config.batch_size)):
                batch = inputs[i:i+config.batch_size]

                if isinstance(batch[0], tuple):
                    tok = self.tokenizer(
                        [b[0] for b in batch],
                        [b[1] for b in batch],
                        padding=True,
                        truncation=True,
                        max_length=config.max_length,
                        return_tensors="pt"
                    )
                else:
                    tok = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=config.max_length,
                        return_tensors="pt"
                    )

                tok = tok.to(self.device)
                embeds = self.model.model.embeddings(tok.input_ids)

                out = self.model.model(
                    input_ids=None,
                    inputs_embeds=embeds,
                    attention_mask=tok.attention_mask,
                    output_hidden_states=True
                )
                hidden = out.hidden_states[-1]

                mask = tok.attention_mask.unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

                feats.append(pooled.cpu().numpy())
        return np.vstack(feats)