import os
import numpy as np
from datasets import load_from_disk
from typing import Dict, List, Tuple

LONGBENCH_ROOT = "/g/data/oy87/cw9909/longbench_data"

# LongBench v2 domains (all are multiple-choice)
LONGBENCH_DOMAINS = [
    "Single-Document QA",
    "Multi-Document QA", 
    "Long In-context Learning",
    "Long-dialogue History Understanding",
    "Code Repository Understanding",
    "Long Structured Data Understanding"
]

class LongBenchTask:
    def __init__(self, split: str = "test"):
        """
        LongBench v2 loader - all tasks are 4-way multiple choice
        
        Args:
            split: 'test' (the only split in LongBench v2, named 'train' in HF)
        """
        self.ds = load_from_disk(LONGBENCH_ROOT)
        # The dataset has only one split called 'train' but it's actually the test set
        if hasattr(self.ds, 'keys'):
            self.data = self.ds['train']
        else:
            self.data = self.ds
            
    def get_all(self) -> Tuple[List[str], np.ndarray]:
        """
        Get all data as classification task
        
        Returns:
            inputs: List of formatted inputs (context + question + choices)
            labels: Array of labels (0=A, 1=B, 2=C, 3=D)
        """
        inputs = []
        labels = []
        
        for example in self.data:
            # Format: context + question + multiple choice options
            formatted_input = self._format_example(example)
            inputs.append(formatted_input)
            
            # Convert A/B/C/D to 0/1/2/3
            label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
            labels.append(label_map[example["answer"]])
        
        return inputs, np.array(labels)
    
    def get_by_domain(self, domain: str) -> Tuple[List[str], np.ndarray]:
        """Get data filtered by domain"""
        inputs = []
        labels = []
        
        for example in self.data:
            if example["domain"] == domain:
                formatted_input = self._format_example(example)
                inputs.append(formatted_input)
                
                label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                labels.append(label_map[example["answer"]])
        
        return inputs, np.array(labels)
    
    def get_by_difficulty(self, difficulty: str) -> Tuple[List[str], np.ndarray]:
        """Get data filtered by difficulty (easy/hard)"""
        inputs = []
        labels = []
        
        for example in self.data:
            if example["difficulty"] == difficulty:
                formatted_input = self._format_example(example)
                inputs.append(formatted_input)
                
                label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                labels.append(label_map[example["answer"]])
        
        return inputs, np.array(labels)
    
    def get_by_length(self, length_category: str) -> Tuple[List[str], np.ndarray]:
        """Get data filtered by length (short/medium/long)"""
        inputs = []
        labels = []
        
        for example in self.data:
            if example["length"] == length_category:
                formatted_input = self._format_example(example)
                inputs.append(formatted_input)
                
                label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                labels.append(label_map[example["answer"]])
        
        return inputs, np.array(labels)
    
    def _format_example(self, example: Dict) -> str:
        """
        Format an example for model input
        
        Options for formatting:
        1. Full context (might be very long - up to 2M words!)
        2. Question + choices only (loses context)
        3. Truncated context + question + choices
        """
        # Option 1: Full format (use if your model can handle it)
        formatted = f"{example['context']}\n\nQuestion: {example['question']}\n"
        formatted += f"A) {example['choice_A']}\n"
        formatted += f"B) {example['choice_B']}\n"
        formatted += f"C) {example['choice_C']}\n"
        formatted += f"D) {example['choice_D']}"
        
        return formatted
    
    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        stats = {
            "total_samples": len(self.data),
            "by_domain": {},
            "by_difficulty": {},
            "by_length": {},
            "context_lengths": []
        }
        
        for example in self.data:
            # Count by domain
            domain = example["domain"]
            stats["by_domain"][domain] = stats["by_domain"].get(domain, 0) + 1
            
            # Count by difficulty
            diff = example["difficulty"]
            stats["by_difficulty"][diff] = stats["by_difficulty"].get(diff, 0) + 1
            
            # Count by length category
            length = example["length"]
            stats["by_length"][length] = stats["by_length"].get(length, 0) + 1
            
            # Track actual word count (if available)
            context_words = len(example["context"].split())
            stats["context_lengths"].append(context_words)
        
        stats["avg_context_length"] = np.mean(stats["context_lengths"])
        stats["median_context_length"] = np.median(stats["context_lengths"])
        stats["max_context_length"] = np.max(stats["context_lengths"])
        
        return stats