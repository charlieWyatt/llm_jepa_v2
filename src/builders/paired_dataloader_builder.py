import json
from torch.utils.data import Dataset, DataLoader

class PairedTextCodeDataset(Dataset):
    """Dataset that returns (text, code) pairs for LLM-JEPA."""
    
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        with open(jsonl_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                # Spider format: {"text": "...", "code": "..."}
                # or could be {"question": "...", "query": "..."}
                text = item.get("text") or item.get("question") or item.get("description")
                code = item.get("code") or item.get("query") or item.get("sql")
                if text and code:
                    self.samples.append({"text": text, "code": code})
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize text and code separately
        text_tokens = self.tokenizer.tokenize(sample["text"])[:self.max_length]
        code_tokens = self.tokenizer.tokenize(sample["code"])[:self.max_length]
        
        return {
            "text_tokens": text_tokens,
            "code_tokens": code_tokens,
            "raw_text": sample["text"],
            "raw_code": sample["code"],
        }


def collate_paired_batch(batch):
    """Collate function that keeps text and code separate."""
    return {
        "text_tokens": [item["text_tokens"] for item in batch],
        "code_tokens": [item["code_tokens"] for item in batch],
        "raw_text": [item["raw_text"] for item in batch],
        "raw_code": [item["raw_code"] for item in batch],
    }


def build_paired_dataloader(jsonl_path: str, tokenizer, batch_size: int = 4):
    dataset = PairedTextCodeDataset(jsonl_path, tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_paired_batch,
        num_workers=2,
        pin_memory=True,
    )