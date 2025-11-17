import torch
from dataclasses import dataclass

@dataclass
class EvalConfig:
    max_length: int = 512
    batch_size: int = 8
    train_samples: int | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # linear probe settings
    max_iter: int = 1000
    random_seed: int = 42

    # output directory
    output_dir: str = "./eval_results"