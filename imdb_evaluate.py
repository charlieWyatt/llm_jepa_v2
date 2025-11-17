"""
IMDB Sentiment Classification Evaluation

This script evaluates trained models on the IMDB dataset using
a linear probe (frozen features + logistic regression classifier).

Supports:
- Longformer (Hugging Face)
- LLM-JEPA (Custom architecture)
"""

import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
from tqdm import tqdm

from datasets import load_from_disk
from transformers import AutoTokenizer, LongformerForMaskedLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


# ============================================
# Configuration
# ============================================
@dataclass
class EvalConfig:
    """Evaluation configuration"""
    imdb_path: str = "/g/data/oy87/cw9909/longformer_eval_datasets/imdb"
    output_dir: str = "./eval_results"
    
    # Hyperparameters
    max_length: int = 512
    batch_size: int = 8
    train_samples: Optional[int] = None  # Use all training samples
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Linear probe settings
    max_iter: int = 1000
    random_seed: int = 42


# ============================================
# Abstract Model Wrapper
# ============================================
class ModelWrapper(ABC):
    """Abstract base class for model wrappers"""
    
    def __init__(self, checkpoint_path: str, device: str):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None
        self.tokenizer = None
    
    @abstractmethod
    def load(self):
        """Load model and tokenizer"""
        pass
    
    @abstractmethod
    def extract_features(self, texts: List[str], config: EvalConfig) -> np.ndarray:
        """Extract features from texts"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return a descriptive name for the model"""
        pass


# ============================================
# Longformer Model Wrapper
# ============================================
class LongformerWrapper(ModelWrapper):
    """Wrapper for Hugging Face Longformer models"""
    
    def load(self):
        """Load Longformer model and tokenizer"""
        print(f"Loading Longformer from {self.checkpoint_path}...")
        
        self.model = LongformerForMaskedLM.from_pretrained(self.checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        
        self.model.eval()
        self.model.to(self.device)
        
        print(f"  Model loaded on {self.device}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def extract_features(self, texts: List[str], config: EvalConfig) -> np.ndarray:
        """Extract features using mean pooling"""
        features_list = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), config.batch_size), desc="Extracting features"):
                batch_texts = texts[i:i + config.batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=config.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                # Forward pass
                outputs = self.model.longformer(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                
                # Mean pooling
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                masked_hidden = hidden_states * attention_mask
                sum_hidden = masked_hidden.sum(dim=1)
                sum_mask = attention_mask.sum(dim=1)
                pooled_features = sum_hidden / sum_mask
                
                features_list.append(pooled_features.cpu().numpy())
        
        return np.vstack(features_list)
    
    def get_model_name(self) -> str:
        """Return model name from checkpoint path"""
        # Extract step number if present
        path = Path(self.checkpoint_path)
        if "checkpoint-" in path.name:
            step = path.name.split("-")[-1]
            return f"longformer_step_{step}"
        return "longformer"


# ============================================
# JEPA Model Wrapper
# ============================================
class JEPAWrapper(ModelWrapper):
    """Wrapper for LLM-JEPA models"""

    def get_model_name(self) -> str:
        path = Path(self.checkpoint_path)
        name = path.stem
        step = name.split("_")[-1]
        return f"jepa_step_{step}"
    
    def extract_features(self, texts: List[str], config: EvalConfig) -> np.ndarray:
        features = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), config.batch_size), desc="Extracting features"):
                batch = texts[i:i+config.batch_size]

                tokens = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=config.max_length,
                    return_tensors="pt"
                ).to(self.device)

                # Forward pass for JEPA
                outputs = self.model.model(
                    input_ids=None,
                    inputs_embeds=self.model.model.embeddings(
                        tokens.input_ids
                    ),
                    attention_mask=tokens.attention_mask,
                    output_hidden_states=True
                )

                # Use last hidden layer
                hidden = outputs.hidden_states[-1]

                # Mean pool like Longformer
                mask = tokens.attention_mask.unsqueeze(-1)
                summed = (hidden * mask).sum(dim=1)
                denom = mask.sum(dim=1)
                pooled = summed / denom

                features.append(pooled.cpu().numpy())

        return np.vstack(features)
    
    def load(self):
        """Load JEPA model and tokenizer"""
        print(f"Loading JEPA from {self.checkpoint_path}...")
        
        # Import JEPA-specific modules
        from src.builders.encoder_builder import encoder_builder
        from config import STRATEGY_CONSTS
        
        # Load checkpoint first to inspect the model architecture
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        # Infer model size from checkpoint weights
        # Check the embedding layer size to determine hidden_size
        embedding_weight_shape = checkpoint['model_state_dict']['model.embeddings.word_embeddings.weight'].shape
        hidden_size = embedding_weight_shape[1]  # Second dimension is hidden_size
        
        # Count the number of layers by counting unique layer indices
        num_layers = sum(1 for k in checkpoint['model_state_dict'].keys() 
                        if 'model.encoder.layer.' in k and '.attention.self.query.weight' in k)
        
        print(f"  Detected from checkpoint: hidden_size={hidden_size}, num_layers={num_layers}")
        
        # Build encoder architecture with detected config
        model_config = {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "attention_window": 512,  # This was constant in your training
        }
        
        context_encoder = encoder_builder(STRATEGY_CONSTS["CONTEXT_ENCODER"]).build(
            model_id=STRATEGY_CONSTS["CONTEXT_MODEL_ID"],
            config=model_config
        )
        
        # Load checkpoint weights
        context_encoder.load_state_dict(checkpoint['model_state_dict'])
        
        self.model = context_encoder
        self.tokenizer = context_encoder.tokenizer
        
        self.model.eval()
        self.model.to(self.device)
        
        print(f"  Model loaded on {self.device}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")


# ============================================
# Data Loading
# ============================================
def load_imdb_dataset(data_path: str) -> Dict:
    """
    Load IMDB dataset from disk.
    
    Args:
        data_path: Path to IMDB dataset directory
        
    Returns:
        Dictionary with 'train' and 'test' splits
    """
    print(f"Loading IMDB dataset from {data_path}...")
    dataset = load_from_disk(data_path)
    
    print(f"  Train samples: {len(dataset['train'])}")
    print(f"  Test samples: {len(dataset['test'])}")
    
    return dataset


# ============================================
# Linear Probe Training and Evaluation
# ============================================
def train_linear_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    config: EvalConfig
) -> LogisticRegression:
    """
    Train a logistic regression classifier on features.
    
    Args:
        train_features: Training features
        train_labels: Training labels
        config: Evaluation configuration
        
    Returns:
        Trained classifier
    """
    print("Training linear probe...")
    
    clf = LogisticRegression(
        max_iter=config.max_iter,
        random_state=config.random_seed
    )
    clf.fit(train_features, train_labels)
    
    print(f"  Training complete")
    
    return clf


def evaluate_classifier(
    clf: LogisticRegression,
    test_features: np.ndarray,
    test_labels: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate classifier and compute metrics.
    
    Args:
        clf: Trained classifier
        test_features: Test features
        test_labels: Test labels
        
    Returns:
        Dictionary with accuracy and F1 score
    """
    predictions = clf.predict(test_features)
    
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    
    return {
        "accuracy": float(accuracy),
        "f1": float(f1)
    }


# ============================================
# Main Evaluation Pipeline
# ============================================
def run_evaluation(model_wrapper: ModelWrapper, config: EvalConfig) -> Dict[str, float]:
    """
    Run complete evaluation pipeline.
    
    Args:
        model_wrapper: Loaded model wrapper
        config: Evaluation configuration
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("="*60)
    print("IMDB Sentiment Classification Evaluation")
    print(f"Model: {model_wrapper.get_model_name()}")
    print("="*60)
    print()
    
    # Load dataset
    dataset = load_imdb_dataset(config.imdb_path)
    
    # Prepare data
    print("\nPreparing data...")
    if config.train_samples is None:
        train_texts = dataset['train']['text']
        train_labels = np.array(dataset['train']['label'])
    else:
        train_texts = dataset['train']['text'][:config.train_samples]
        train_labels = np.array(dataset['train']['label'][:config.train_samples])

    test_texts = dataset['test']['text']
    test_labels = np.array(dataset['test']['label'])

    print(f"  Using {len(train_texts)} training samples")
    print(f"  Using {len(test_texts)} test samples")
    
    # Extract features
    print("\nExtracting training features...")
    train_features = model_wrapper.extract_features(train_texts, config)
    
    print("\nExtracting test features...")
    test_features = model_wrapper.extract_features(test_texts, config)
    
    print(f"  Feature dimension: {train_features.shape[1]}")
    
    # Train and evaluate
    clf = train_linear_probe(train_features, train_labels, config)
    results = evaluate_classifier(clf, test_features, test_labels)
    
    # Print results
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  F1 Score: {results['f1']:.4f}")
    print()
    
    return results


def save_results(results: Dict[str, float], model_name: str, output_dir: str):
    """Save results to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f"{model_name}_imdb_results.json"
    
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print("Done!")


# ============================================
# Entry Point
# ============================================
def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate models on IMDB")
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["longformer", "jepa"],
        help="Type of model to evaluate"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_results",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = EvalConfig(output_dir=args.output_dir)
    
    # Create appropriate model wrapper
    if args.model_type == "longformer":
        model_wrapper = LongformerWrapper(args.checkpoint, config.device)
    elif args.model_type == "jepa":
        model_wrapper = JEPAWrapper(args.checkpoint, config.device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Load model
    model_wrapper.load()
    
    # Run evaluation
    results = run_evaluation(model_wrapper, config)
    
    # Save results
    save_results(results, model_wrapper.get_model_name(), config.output_dir)


if __name__ == "__main__":
    main()