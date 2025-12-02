"""
Metrics tracking and calculation for multi-agent RL
"""
import numpy as np
import json
from typing import List, Dict


class GameMetrics:
    """Track metrics across episodes"""
    
    def __init__(self):
        self.episodes = []
        self.fraudster_rewards = []
        self.defender_rewards = []
        self.system_losses = []
        self.fraud_success_rates = []
        self.fraud_attempts = []
        self.detections = []
    
    def add_episode(self, episode_info: dict):
        """Add episode information"""
        self.episodes.append(len(self.episodes))
        self.fraudster_rewards.append(episode_info['fraudster_reward'])
        self.defender_rewards.append(episode_info['defender_reward'])
        self.system_losses.append(episode_info['system_loss'])
        self.fraud_success_rates.append(episode_info['fraud_success_rate'])
        self.fraud_attempts.append(episode_info['fraud_attempts'])
        self.detections.append(episode_info['detections'])
    
    def get_recent_stats(self, window: int = 100) -> dict:
        """Get statistics over recent window"""
        if len(self.episodes) == 0:
            return {}
        
        start_idx = max(0, len(self.episodes) - window)
        
        return {
            'avg_fraudster_reward': np.mean(self.fraudster_rewards[start_idx:]),
            'avg_defender_reward': np.mean(self.defender_rewards[start_idx:]),
            'avg_system_loss': np.mean(self.system_losses[start_idx:]),
            'avg_fraud_success_rate': np.mean(self.fraud_success_rates[start_idx:]),
            'avg_fraud_attempts': np.mean(self.fraud_attempts[start_idx:]),
            'avg_detections': np.mean(self.detections[start_idx:])
        }
    
    def save(self, path: str):
        """Save metrics to JSON"""
        data = {
            'episodes': self.episodes,
            'fraudster_rewards': self.fraudster_rewards,
            'defender_rewards': self.defender_rewards,
            'system_losses': self.system_losses,
            'fraud_success_rates': self.fraud_success_rates,
            'fraud_attempts': self.fraud_attempts,
            'detections': self.detections
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load metrics from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        metrics = cls()
        metrics.episodes = data['episodes']
        metrics.fraudster_rewards = data['fraudster_rewards']
        metrics.defender_rewards = data['defender_rewards']
        metrics.system_losses = data['system_losses']
        metrics.fraud_success_rates = data['fraud_success_rates']
        metrics.fraud_attempts = data['fraud_attempts']
        metrics.detections = data['detections']
        
        return metrics


def compute_metrics(true_fraud: List[int], predictions: List[int]) -> dict:
    """
    Compute classification metrics for fraud detection
    
    Args:
        true_fraud: Ground truth (1 if fraud, 0 otherwise)
        predictions: Predictions (1 if flagged, 0 otherwise)
    
    Returns:
        metrics: dict with precision, recall, F1, etc.
    """
    true_fraud = np.array(true_fraud)
    predictions = np.array(predictions)
    
    tp = np.sum((true_fraud == 1) & (predictions == 1))
    fp = np.sum((true_fraud == 0) & (predictions == 1))
    tn = np.sum((true_fraud == 0) & (predictions == 0))
    fn = np.sum((true_fraud == 1) & (predictions == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(true_fraud) if len(true_fraud) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }
