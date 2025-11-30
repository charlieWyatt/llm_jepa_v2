import numpy as np
from typing import List, Dict
import re
from collections import Counter

def normalize_answer(s: str) -> str:
    """Normalize answer for comparison"""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ' '.join(s.split())
    return s

def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score between prediction and ground truth"""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match(prediction: str, ground_truth: str) -> float:
    """Check if prediction exactly matches ground truth"""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_longbench_metrics(predictions: List[str], answers: List[List[str]]) -> Dict[str, float]:
    """
    Compute metrics for LongBench tasks
    
    Args:
        predictions: List of model predictions
        answers: List of ground truth answers (each can have multiple valid answers)
    """
    f1_scores = []
    em_scores = []
    
    for pred, truth_list in zip(predictions, answers):
        # Take max over all possible answers
        max_f1 = max(f1_score(pred, truth) for truth in truth_list)
        max_em = max(exact_match(pred, truth) for truth in truth_list)
        
        f1_scores.append(max_f1)
        em_scores.append(max_em)
    
    return {
        "f1": np.mean(f1_scores),
        "exact_match": np.mean(em_scores),
        "num_samples": len(predictions)
    }